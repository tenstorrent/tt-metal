"""ttnn port of VGGT on a single Tenstorrent p150a Blackhole chip.

Every op the aggregator depends on, plus the final DPT conv stack, runs
on the p150a. Weights upload once per install; forward passes only move
input images down and outputs back.

Ports on device:
  - Full transformer Block (100 instances: 24 DINOv2 + 24 frame + 24
    global + 4 camera-head trunk + extras): norm1, qkv, nlp_create_qkv_
    heads, q_norm/k_norm, attention scores/softmax/context, merge_heads,
    proj, ls1, residual add, norm2, fc1, gelu, fc2, ls2, residual add.
    One upload and one download per block.
  - 2D RoPE: cos/sin lookup tables precomputed at install for VGGT's
    fixed 518x518 layout; applied on chip via slice + rotate_half +
    multiply + add.
  - DPTHead.scratch.output_conv2 (3x3 128->32 conv + relu + 1x1
    32->output_dim conv at full 518x518 spatial resolution): the largest
    single DPT chunk runs via ttnn.conv2d HiFi4.
  - Standalone Mlp fallback for camera_head.pose_branch.

Still on CPU (incremental device ports attempted and discarded — see
results.tsv):
  - DPT scratch_forward refinenets: per-conv ttnn wrapper added +255 ms
    because 120 individual upload/download round trips outweigh the
    compute. Needs a device-native scratch_forward that keeps
    intermediates on chip (big rewrite; next logical step).
  - DPT prelude (norm + projects + pos_embed + resize_layers): tiny
    compute, overhead-bound per-op.
  - custom_interpolate bilinear at 518x518: ~13 ms, small chunk.
  - activate_head (exp / expm1 / norm): tiny and precision-sensitive.
  - Image normalization and token concats in Aggregator.forward.

Precision profile:
  - Weights + matmul inputs: bfloat16.
  - Residual accumulator in the Block: fp32 via proj/fc2 dtype=float32.
    bf16 residuals over 48 aggregator blocks collapsed PCC to 0.978.
  - Attention scores + softmax + context intermediate: fp32 via HiFi4 +
    dtype=float32. bf16 softmax over 1374-long rows dropped conf PCC
    below 0.99.
  - proj, fc2, DPT output_conv2: HiFi4 + fp32 dest acc.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


_CACHED_MODEL = None
_INSTALL_DONE: dict = {}
_HIFI_KCONFIG = None

# BF0 option 2 state: pad S to a canonical size so ttnn program-cache only
# ever sees one set of shapes. When padding is active, global attention
# needs an additive mask with -inf on the padding frames' key positions.
# Set by the Aggregator patch right before the forward; cleared after.
_ACTIVE_GLOBAL_MASK = None
_PATCH_COUNT = None  # P = patches + special tokens (1374 for VGGT @ 518x518 patch=14)


class _TTPassed:
    """Wraps a device (ttnn) fp32 tensor so consecutive transformer blocks
    can pass the residual accumulator on-device without a PCIe round-trip.

    Only activated for blocks marked with _tt_can_pass=True. For S=1 the
    frame/global reshape between blocks is a NOP so the shape is stable;
    for S>1 a shape change at frame→global boundaries (e.g. (4,1374,1024)
    → (1,5496,1024)) causes _use_pass=False in the next block, which
    materializes and re-uploads — correct but no PCIe saving at that hop.
    """

    __slots__ = ("_tt_tensor", "_orig_dtype", "_shape_3d", "_logical_shape")

    def __init__(self, tt_tensor, orig_dtype, shape_3d, logical_shape=None):
        self._tt_tensor = tt_tensor
        self._orig_dtype = orig_dtype
        self._shape_3d = tuple(shape_3d)
        self._logical_shape = tuple(logical_shape) if logical_shape is not None else self._shape_3d

    @property
    def shape(self):
        return torch.Size(self._logical_shape)

    @property
    def dtype(self):
        return self._orig_dtype

    def view(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple, torch.Size)):
            new_shape = tuple(int(s) for s in args[0])
        else:
            new_shape = tuple(int(s) for s in args)
        return _TTPassed(self._tt_tensor, self._orig_dtype, self._shape_3d, new_shape)

    def reshape(self, *args):
        return self.view(*args)

    def contiguous(self):
        return self

    def _materialize(self):
        import ttnn as _ttnn
        result = _ttnn.to_torch(self._tt_tensor).to(self._orig_dtype)
        if result.shape != torch.Size(self._logical_shape):
            result = result.view(self._logical_shape)
        return result

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        def _mat(x):
            return x._materialize() if isinstance(x, _TTPassed) else x

        new_args = []
        for a in args:
            if isinstance(a, _TTPassed):
                new_args.append(_mat(a))
            elif isinstance(a, (list, tuple)) and any(isinstance(v, _TTPassed) for v in a):
                new_args.append(type(a)(_mat(v) for v in a))
            else:
                new_args.append(a)
        return func(*new_args, **kwargs)


def _get_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        from reference.torch_vggt import load_vggt
        _CACHED_MODEL = load_vggt(eval_mode=True)
    return _CACHED_MODEL


def _hifi_kconfig(device):
    global _HIFI_KCONFIG
    if _HIFI_KCONFIG is None:
        import ttnn
        _HIFI_KCONFIG = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    return _HIFI_KCONFIG


# ---------- helpers ----------

def _upload(t: torch.Tensor, device, dtype=None, layout=None):
    import ttnn
    t = t.to(torch.bfloat16) if t.dtype != torch.bfloat16 else t
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype or ttnn.bfloat16,
        layout=layout or ttnn.TILE_LAYOUT,
        device=device,
    )


# ---------- 2D RoPE on device ----------
#
# VGGT uses RotaryPositionEmbedding2D with base=100.0. For a fixed image
# layout (B=1, S=1, img_size=518, patch=14) the per-token cos/sin values
# are constant, so we precompute them once and upload as ttnn tensors.
# All 48 aggregator blocks reuse the same 4 tables.
#
# Apply on-device (for a token tensor shaped (B, H, N, Dh=64)):
#   v_part, h_part = split on last dim at Dh/2
#   v_part = v_part * cos_y + rotate_half_D(v_part) * sin_y   with D=32
#   h_part = h_part * cos_x + rotate_half_D(h_part) * sin_x
#   out    = concat([v_part, h_part], dim=-1)
# where rotate_half_D(x) for x shape (B, H, N, 32):
#   x1 = x[..., :16]; x2 = x[..., 16:]; return concat([-x2, x1], dim=-1)


def _install_ttnn_rope_tables(model, device):
    """Upload cos/sin base tables on every RoPE instance. Per-token lookup
    (shape (B, 1, N, D)) is produced at block-forward time from the actual
    pos tensor and cached per-shape — lets us handle varying S (frame
    attention sees N=P, global attention sees N=S*P), not just S=1."""
    import ttnn
    from vggt.layers.rope import RotaryPositionEmbedding2D  # type: ignore
    for m in model.modules():
        if isinstance(m, RotaryPositionEmbedding2D) and not getattr(m, "_tt_rope_ready", False):
            m._tt_device = device
            m._tt_base_freq = float(m.base_frequency)
            # cos/sin host tables are (max_pos_seen, D); grown on demand.
            m._tt_cos_host = None
            m._tt_sin_host = None
            m._tt_max_pos = 0
            # Per-pos-tensor lookup cache: key = (id(pos_tensor), Dh) ->
            # {"cos_y", "sin_y", "cos_x", "sin_x"} each (1, 1, N, D/2) ttnn tile.
            m._tt_lookup_cache = {}
            m._tt_ready = True


def _get_rope_tables_for_pos(rope_module, pos: torch.Tensor, head_dim: int):
    """Given the pos tensor the block was invoked with, return (cos_y,
    sin_y, cos_x, sin_x) as ttnn tensors of shape (1, 1, N, D/2). Cached
    per (pos-identity, head_dim) so 48 aggregator blocks pay for one
    host-side lookup + upload per attention type."""
    import ttnn
    D = head_dim // 2  # per 1D RoPE dim (= 32 for head_dim=64)
    key = (id(pos), pos.shape, head_dim)
    cached = rope_module._tt_lookup_cache.get(key)
    if cached is not None:
        return cached

    dev = rope_module._tt_device
    base = rope_module._tt_base_freq
    max_pos_needed = int(pos.max().item()) + 1
    if rope_module._tt_cos_host is None or rope_module._tt_max_pos < max_pos_needed:
        exponents = torch.arange(0, D, 2).float() / D  # (D/2,)
        inv_freq = 1.0 / (base ** exponents)
        positions = torch.arange(max_pos_needed, dtype=inv_freq.dtype)
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        angles = torch.cat((angles, angles), dim=-1)  # (max_pos, D)
        rope_module._tt_cos_host = angles.cos()
        rope_module._tt_sin_host = angles.sin()
        rope_module._tt_max_pos = max_pos_needed

    cos_t = rope_module._tt_cos_host
    sin_t = rope_module._tt_sin_host

    # pos shape: (Beff, N, 2). Lookup gives (Beff, N, D).
    y = pos[..., 0]  # (Beff, N)
    x = pos[..., 1]
    Beff, N = y.shape

    def lookup_and_upload(table, idx):
        looked = table[idx]  # (Beff, N, D)
        # For Beff > 1 (frame attention with S>1) the table has a real
        # batch dim; we broadcast over heads only.
        t = looked.reshape(Beff, 1, N, D).to(torch.bfloat16).contiguous()
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    tables = {
        "cos_y": lookup_and_upload(cos_t, y),
        "sin_y": lookup_and_upload(sin_t, y),
        "cos_x": lookup_and_upload(cos_t, x),
        "sin_x": lookup_and_upload(sin_t, x),
    }
    rope_module._tt_lookup_cache[key] = tables
    return tables


def _apply_rope_device(tt_tokens, tables, B, H, N, Dh):
    """2D RoPE applied to (B, H, N, Dh) ttnn tensor. Returns same shape."""
    import ttnn
    D = Dh // 2  # 32
    Dq = D // 2  # 16
    cos_y = tables["cos_y"]; sin_y = tables["sin_y"]
    cos_x = tables["cos_x"]; sin_x = tables["sin_x"]

    # Split tokens into v (first D) and h (last D).
    v_part = ttnn.slice(tt_tokens, [0, 0, 0, 0], [B, H, N, D])
    h_part = ttnn.slice(tt_tokens, [0, 0, 0, D], [B, H, N, Dh])

    def rope_1d(part, cos, sin):
        x1 = ttnn.slice(part, [0, 0, 0, 0], [B, H, N, Dq])
        x2 = ttnn.slice(part, [0, 0, 0, Dq], [B, H, N, D])
        rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
        return ttnn.add(ttnn.multiply(part, cos), ttnn.multiply(rot, sin))

    v_out = rope_1d(v_part, cos_y, sin_y)
    h_out = rope_1d(h_part, cos_x, sin_x)
    return ttnn.concat([v_out, h_out], dim=-1)


# ---------- Full on-device Block port ----------

def _install_ttnn_block(model, device):
    """Preload every Block's weights onto device and patch Block.forward
    to run the entire residual path (attention + MLP + LayerScale + adds)
    on the p150a. RoPE still rides on the CPU because its 2D position
    LUT isn't ported yet (separate install step next).
    """
    import ttnn
    import torch.nn as nn
    from vggt.layers.block import Block  # type: ignore
    from vggt.layers.layer_scale import LayerScale  # type: ignore

    for blk in model.modules():
        if not isinstance(blk, Block):
            continue
        if getattr(blk, "_tt_block_ready", False):
            continue
        attn = blk.attn
        mlp = blk.mlp
        if not (isinstance(blk.norm1, nn.LayerNorm) and isinstance(blk.norm2, nn.LayerNorm)):
            continue

        blk._tt_device = device
        blk._tt_heads = attn.num_heads
        blk._tt_head_dim = attn.head_dim

        # Preload norm1 / norm2.
        for src, dst in (("_tt_ln1", blk.norm1), ("_tt_ln2", blk.norm2)):
            g = dst.weight.detach().reshape(1, 1, -1).to(torch.bfloat16)
            b = dst.bias.detach().reshape(1, 1, -1).to(torch.bfloat16)
            setattr(blk, f"{src}_g", _upload(g, device))
            setattr(blk, f"{src}_b", _upload(b, device))
            setattr(blk, f"{src}_eps", float(dst.eps))

        # Preload qkv + proj.
        qw = attn.qkv.weight.detach().t().contiguous().to(torch.bfloat16)
        blk._tt_qkv_w = _upload(qw, device)
        if attn.qkv.bias is not None:
            blk._tt_qkv_b = _upload(
                attn.qkv.bias.detach().reshape(1, 1, -1).to(torch.bfloat16), device
            )
        else:
            blk._tt_qkv_b = None
        pw = attn.proj.weight.detach().t().contiguous().to(torch.bfloat16)
        blk._tt_proj_w = _upload(pw, device)
        if attn.proj.bias is not None:
            blk._tt_proj_b = _upload(
                attn.proj.bias.detach().reshape(1, 1, -1).to(torch.bfloat16), device
            )
        else:
            blk._tt_proj_b = None

        # qk_norm is per-head-dim LayerNorm when Attention was built with
        # qk_norm=True; else Identity.
        blk._tt_qk_norm = isinstance(attn.q_norm, nn.LayerNorm)
        if blk._tt_qk_norm:
            for src, dst in (("_tt_qn", attn.q_norm), ("_tt_kn", attn.k_norm)):
                g = dst.weight.detach().reshape(1, 1, 1, -1).to(torch.bfloat16)
                b = dst.bias.detach().reshape(1, 1, 1, -1).to(torch.bfloat16)
                setattr(blk, f"{src}_g", _upload(g, device))
                setattr(blk, f"{src}_b", _upload(b, device))
                setattr(blk, f"{src}_eps", float(dst.eps))

        # ls1 / ls2.
        for src, dst in (("_tt_ls1", blk.ls1), ("_tt_ls2", blk.ls2)):
            if isinstance(dst, LayerScale):
                g = dst.gamma.detach().reshape(1, 1, -1).to(torch.bfloat16)
                setattr(blk, src, _upload(g, device))
            else:
                setattr(blk, src, None)

        # Mlp fc1 / fc2.
        for src, dst in (("_tt_fc1", mlp.fc1), ("_tt_fc2", mlp.fc2)):
            w = dst.weight.detach().t().contiguous().to(torch.bfloat16)
            b = dst.bias.detach().reshape(1, 1, -1).to(torch.bfloat16)
            setattr(blk, f"{src}_w", _upload(w, device))
            setattr(blk, f"{src}_b", _upload(b, device))

        blk._tt_has_rope = attn.rope is not None
        blk._tt_block_ready = True

    if getattr(Block, "_tt_block_patched", False):
        return

    def tt_block_forward(self, x: torch.Tensor, pos=None) -> torch.Tensor:
        if not getattr(self, "_tt_block_ready", False):
            return self._orig_forward(x, pos=pos)

        import os, time as _time
        H = self._tt_heads
        Dh = self._tt_head_dim
        dev = self._tt_device
        kcfg = _hifi_kconfig(dev)

        # Fast path: previous block left the residual on device — skip upload.
        # Only valid when the logical shape matches the device tensor shape
        # (S=1 has no reshape between frame/global; S>1 frame→global reshape
        # causes a mismatch and falls to the normal upload path).
        _can_pass = getattr(self, "_tt_can_pass", False)
        _use_pass = (isinstance(x, _TTPassed) and _can_pass
                     and x._logical_shape == x._shape_3d)
        if _use_pass:
            tt_x = x._tt_tensor
            B, N, C = x._shape_3d
            _orig_dtype = x._orig_dtype
        else:
            if isinstance(x, _TTPassed):
                x = x._materialize()
            B, N, C = x.shape
            _orig_dtype = x.dtype
            # Hold the residual accumulator in fp32 on device. Matmul/LN
            # inputs are cast down to bf16 per op. Without fp32 accumulation
            # the world_points_conf PCC drops from 0.9946 -> 0.9778 over 48
            # aggregator blocks of bf16-rounded residuals.
            x_f32 = x.to(torch.float32) if x.dtype != torch.float32 else x
            tt_x = ttnn.from_torch(
                x_f32.contiguous(), dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT, device=dev,
            )

        _trace = os.environ.get("VGGT_BLOCK_TRACE", "0") not in ("", "0")
        if _trace:
            _t0 = _time.perf_counter()
            _t_last = _t0
            print(f"[block-trace] enter B={B} N={N} C={C}", flush=True)
            def _tick(tag):
                nonlocal _t_last
                ttnn.synchronize_device(dev)
                now = _time.perf_counter()
                print(f"[block-trace]   +{now-_t_last:.3f}s {tag}", flush=True)
                _t_last = now

        # ---- attention branch ----
        # LN input gets cast to bf16 for the kernel, output stays bf16.
        tt_n = ttnn.typecast(tt_x, ttnn.bfloat16)
        tt_n = ttnn.layer_norm(
            tt_n, weight=self._tt_ln1_g, bias=self._tt_ln1_b, epsilon=self._tt_ln1_eps,
        )
        if _trace: _tick("LN1")
        tt_qkv = ttnn.linear(tt_n, self._tt_qkv_w, bias=self._tt_qkv_b)
        if _trace: _tick("qkv linear")

        # Split qkv on device into (q, k^T, v). nlp_create_qkv_heads wants
        # a 4D input (B, 1, N, 3*H*Dh) and returns:
        #   q:  (B, H, N, Dh)
        #   kt: (B, H, Dh, N)  (pre-transposed for Q @ K^T)
        #   v:  (B, H, N, Dh)
        tt_qkv = ttnn.reshape(tt_qkv, (B, 1, N, 3 * H * Dh))
        tt_q, tt_kt, tt_v = ttnn.experimental.nlp_create_qkv_heads(
            tt_qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=True,
        )
        if _trace: _tick("nlp_create_qkv_heads")

        # qk_norm on device (LayerNorm over head_dim).
        if self._tt_qk_norm:
            tt_q = ttnn.layer_norm(
                tt_q, weight=self._tt_qn_g, bias=self._tt_qn_b, epsilon=self._tt_qn_eps,
            )
            # k is transposed in kt; untranspose, LN, transpose again.
            tt_k = ttnn.permute(tt_kt, (0, 1, 3, 2))
            tt_k = ttnn.layer_norm(
                tt_k, weight=self._tt_kn_g, bias=self._tt_kn_b, epsilon=self._tt_kn_eps,
            )
            tt_kt = ttnn.permute(tt_k, (0, 1, 3, 2))

        # RoPE on device: apply 2D rotary embedding to q and k without
        # round-tripping through host. Tables are computed on demand from
        # the pos tensor the caller passed in (handles variable S, frame
        # vs global attention) and cached on the RoPE module.
        if self._tt_has_rope:
            tables = _get_rope_tables_for_pos(self.attn.rope, pos, Dh)
            if _trace: _tick("rope tables")
            tt_q = _apply_rope_device(tt_q, tables, B, H, N, Dh)
            if _trace: _tick("rope q")
            # k lives transposed in tt_kt; untranspose, apply RoPE, transpose.
            tt_k = ttnn.permute(tt_kt, (0, 1, 3, 2))
            tt_k = _apply_rope_device(tt_k, tables, B, H, N, Dh)
            tt_kt = ttnn.permute(tt_k, (0, 1, 3, 2))
            if _trace: _tick("rope k")

        # Attention compute on device. fp32 intermediate for softmax
        # stability over the 1374-long row (bf16 collapsed conf PCC).
        # NOTE: SDPA (FlashAttention-2) was tested but dropped — it uses
        # bf16 softmax internally, collapsing world_points_conf PCC to 0.89
        # (needs >0.99). fp32 score+softmax+context path is required.
        tt_scores = ttnn.matmul(tt_q, tt_kt, compute_kernel_config=kcfg, dtype=ttnn.float32)
        if _trace: _tick("Q@Kt scores")
        tt_scores = ttnn.multiply(tt_scores, 1.0 / math.sqrt(Dh))
        if _trace: _tick("scale")
        # BF0 option 2: pad-to-canonical-S with attention mask. When the
        # Aggregator wrapper has padded input frames, it publishes an
        # additive global-attention mask (-inf on padding-frame key
        # positions). Apply only to the aggregator's global-attention
        # blocks, detected via N > _PATCH_COUNT (P=1374). Frame-attn N==P,
        # camera-trunk attn N=5 — neither gets the mask.
        # BF0 softmax-at-large-N hang on Blackhole:
        # ttnn.softmax(fp32, ...) hangs at N ≥ ~4100 (observed at N=4122,
        # N=5496). Fix: decompose softmax into individual ops (max / sub /
        # exp / sum / reciprocal), all of which work in fp32 at N=5496.
        # fp32 broadcast add (1,1,1,N)→(1,H,N,N) also works (probed); the
        # earlier session hang was chip-state, not the op itself.
        _LARGE_SOFTMAX_N = 4000
        if N >= _LARGE_SOFTMAX_N:
            # Apply padding mask in fp32 — global attention only.
            if (_ACTIVE_GLOBAL_MASK is not None
                    and _PATCH_COUNT is not None and N > _PATCH_COUNT):
                _gm = _ACTIVE_GLOBAL_MASK
                if _gm.dtype != ttnn.float32:
                    _gm = ttnn.typecast(_gm, ttnn.float32)
                tt_scores = ttnn.add(tt_scores, _gm)
                if _trace: _tick("mask add")
            # Manual stable fp32 softmax: subtract row-max before exp.
            _sm = ttnn.max(tt_scores, dim=-1)
            _sm = ttnn.reshape(_sm, (B, H, N, 1))
            _shifted = ttnn.subtract(tt_scores, _sm)
            if _trace: _tick("softmax: max-sub")
            _e = ttnn.exp(_shifted)
            _es = ttnn.sum(_e, dim=-1)
            _es = ttnn.reshape(_es, (B, H, N, 1))
            tt_attn = ttnn.multiply(_e, ttnn.reciprocal(_es))
            if _trace: _tick("softmax: exp-sum-div")
        else:
            tt_attn = ttnn.softmax(tt_scores, dim=-1, compute_kernel_config=kcfg)
        if _trace: _tick("softmax")
        # Context: match the precision profile of the working (non-fused)
        # attention — fp32 intermediate then back to bf16 for the proj
        # matmul. bf16 context alone pushed world_points_conf to 0.9788.
        tt_ctx = ttnn.matmul(tt_attn, tt_v, compute_kernel_config=kcfg, dtype=ttnn.float32)
        if _trace: _tick("attn@V")

        # Merge heads on device: (B, H, N, Dh) -> (B, N, H*Dh). Permute in
        # fp32 then cast to bf16 for the proj matmul.
        tt_ctx = ttnn.permute(tt_ctx, (0, 2, 1, 3))
        tt_ctx = ttnn.reshape(tt_ctx, (B, N, H * Dh))
        tt_ctx = ttnn.typecast(tt_ctx, ttnn.bfloat16)
        if _trace: _tick("merge heads")

        # proj with HiFi4 + fp32 dest so the residual contribution is fp32.
        tt_attn_out = ttnn.linear(
            tt_ctx, self._tt_proj_w, bias=self._tt_proj_b,
            compute_kernel_config=kcfg, dtype=ttnn.float32,
        )
        if _trace: _tick("proj")
        if self._tt_ls1 is not None:
            tt_attn_out = ttnn.multiply(tt_attn_out, self._tt_ls1)
        # fp32 + fp32 residual add.
        tt_x = ttnn.add(tt_x, tt_attn_out)
        if _trace: _tick("residual1")

        # ---- MLP branch ----
        tt_n = ttnn.typecast(tt_x, ttnn.bfloat16)
        tt_n = ttnn.layer_norm(
            tt_n, weight=self._tt_ln2_g, bias=self._tt_ln2_b, epsilon=self._tt_ln2_eps,
        )
        if _trace: _tick("LN2")
        tt_m = ttnn.linear(tt_n, self._tt_fc1_w, bias=self._tt_fc1_b)
        if _trace: _tick("fc1")
        tt_m = ttnn.gelu(tt_m)
        if _trace: _tick("gelu")
        # fc2 emits fp32 for the residual stream.
        tt_mlp = ttnn.linear(
            tt_m, self._tt_fc2_w, bias=self._tt_fc2_b,
            compute_kernel_config=kcfg, dtype=ttnn.float32,
        )
        if _trace: _tick("fc2")
        if self._tt_ls2 is not None:
            tt_mlp = ttnn.multiply(tt_mlp, self._tt_ls2)
        tt_x = ttnn.add(tt_x, tt_mlp)
        if _trace: _tick("residual2")

        if _can_pass:
            if _trace:
                print(f"[block-trace] B={B} N={N} C={C} total={_time.perf_counter()-_t0:.3f}s", flush=True)
            return _TTPassed(tt_x, _orig_dtype, (B, N, C))
        result = ttnn.to_torch(tt_x).to(_orig_dtype)
        if _trace:
            _tick("download")
            print(f"[block-trace] B={B} N={N} C={C} total={_time.perf_counter()-_t0:.3f}s", flush=True)
        return result

    Block._orig_forward = Block.forward
    Block.forward = tt_block_forward
    Block._tt_block_patched = True

    # NestedTensorBlock (used by DINOv2 patch_embed) overrides forward() with
    # an isinstance(x, Tensor) guard that raises AssertionError for _TTPassed.
    # Patch it to forward _TTPassed directly to tt_block_forward.
    try:
        from vggt.layers.block import NestedTensorBlock  # type: ignore
        if not getattr(NestedTensorBlock, "_tt_nested_patched", False):
            _orig_ntb_fwd = NestedTensorBlock.forward
            def _ntb_forward(self, x_or_x_list):
                if isinstance(x_or_x_list, _TTPassed):
                    return tt_block_forward(self, x_or_x_list)
                return _orig_ntb_fwd(self, x_or_x_list)
            NestedTensorBlock.forward = _ntb_forward
            NestedTensorBlock._tt_nested_patched = True
    except (ImportError, AttributeError):
        pass


# ---------- standalone Mlp fallback (camera_head.pose_branch) ----------

def _install_ttnn_mlp(model, device):
    """Route any standalone Mlp (not owned by a Block) through ttnn."""
    import ttnn
    from vggt.layers.mlp import Mlp  # type: ignore
    from vggt.layers.block import Block  # type: ignore

    block_mlp_ids = {id(blk.mlp) for blk in model.modules() if isinstance(blk, Block)}

    for m in model.modules():
        if isinstance(m, Mlp) and id(m) not in block_mlp_ids and not getattr(m, "_tt_ready", False):
            w1 = m.fc1.weight.detach().t().contiguous().to(torch.bfloat16)
            b1 = m.fc1.bias.detach().reshape(1, 1, -1).to(torch.bfloat16)
            w2 = m.fc2.weight.detach().t().contiguous().to(torch.bfloat16)
            b2 = m.fc2.bias.detach().reshape(1, 1, -1).to(torch.bfloat16)
            m._tt_w1 = _upload(w1, device)
            m._tt_b1 = _upload(b1, device)
            m._tt_w2 = _upload(w2, device)
            m._tt_b2 = _upload(b2, device)
            m._tt_device = device
            m._tt_ready = True

    if getattr(Mlp, "_tt_patched", False):
        return

    def ttnn_forward(self, x: torch.Tensor) -> torch.Tensor:
        if not getattr(self, "_tt_ready", False):
            return self._orig_forward(x)
        tt_in = _upload(x, self._tt_device)
        tt_mid = ttnn.linear(tt_in, self._tt_w1, bias=self._tt_b1)
        tt_mid = ttnn.gelu(tt_mid)
        tt_out = ttnn.linear(tt_mid, self._tt_w2, bias=self._tt_b2)
        return ttnn.to_torch(tt_out).to(x.dtype)

    Mlp._orig_forward = Mlp.forward
    Mlp.forward = ttnn_forward
    Mlp._tt_patched = True


# ---------- install orchestration ----------

def _install_ttnn_dpt_scratch(model, device):
    """Port DPTHead.scratch_forward (layer{1..4}_rn → refinenet{4,3,2,1}
    → output_conv1) to ttnn. Biggest remaining CPU chunk (~194 ms × 2
    heads at B=1 S=1). A prior attempt (see TODO.md P0 / `2b2bf2d`
    discard row) broke PCC to -0.08 because layouts between chained
    conv2d / linear / upsample / add calls weren't right. This port
    uses mast3r's verified layout pattern (copied from
    `tt-metal/models/demos/mast3r/tt/ttnn_dust3r.py:813+`):
      - conv2d input: ROW_MAJOR flat (1, 1, B·H·W, C).
      - conv2d output chains directly into conv2d / relu / add (TILE flat).
      - upsample input: ROW_MAJOR NHWC (B, H, W, C) via _flat_to_nhwc.
      - 1×1 out_conv: via ttnn.linear (TILE flat) rather than ttnn.conv2d.

    The only non-integer upsample (refinenet4: 19→37, scale ≈ 1.95)
    falls back to a host round-trip since ttnn.upsample bilinear expects
    integer scale factors. The downloaded tensor is (BS, 256, 19, 19) bf16
    ≈ 184 KB — cheap compared to the compute savings elsewhere.

    Gated on `_tt_scratch_ready`. Env `VGGT_TT_SCRATCH_COMPARE=1` runs
    device + host scratch_forward side-by-side and prints per-refinenet
    PCC — fastest way to isolate a layout bug if one reappears.
    """
    import ttnn
    import torch.nn as nn
    from vggt.heads.dpt_head import DPTHead, ResidualConvUnit, FeatureFusionBlock, custom_interpolate  # type: ignore

    def _wt(t):
        return ttnn.from_torch(t.detach().to(torch.bfloat16), dtype=ttnn.bfloat16)

    def _bt(t):
        return ttnn.from_torch(t.detach().reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16)

    def _w_lin(t):
        out_c, in_c = t.shape[0], t.shape[1]
        w_t = t.detach().reshape(out_c, in_c).t().contiguous()
        return ttnn.from_torch(w_t.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def _b_lin(t):
        return ttnn.from_torch(
            t.detach().reshape(1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

    for h in model.modules():
        if not isinstance(h, DPTHead) or getattr(h, "_tt_scratch_ready", False):
            continue
        if getattr(h, "feature_only", False):
            continue
        if not hasattr(h.scratch, "output_conv1"):
            continue

        sw = {}
        # layer_rn: Conv2d 3×3 bias=False, in_ch varies (256/512/1024/1024), out=256.
        for i in (1, 2, 3, 4):
            conv = getattr(h.scratch, f"layer{i}_rn")
            sw[f"l{i}_rn_w"] = _wt(conv.weight)
            sw[f"l{i}_rn_in_c"] = conv.in_channels
            sw[f"l{i}_rn_out_c"] = conv.out_channels  # always 256 in VGGT

        # refinenets. refinenet4 has_residual=False (no resConfUnit1).
        for r in (1, 2, 3, 4):
            ref = getattr(h.scratch, f"refinenet{r}")
            has_res = ref.has_residual
            sw[f"r{r}_has_res"] = has_res
            if has_res:
                u = ref.resConfUnit1
                sw[f"r{r}_u1_c1_w"] = _wt(u.conv1.weight)
                sw[f"r{r}_u1_c1_b"] = _bt(u.conv1.bias)
                sw[f"r{r}_u1_c2_w"] = _wt(u.conv2.weight)
                sw[f"r{r}_u1_c2_b"] = _bt(u.conv2.bias)
            u = ref.resConfUnit2
            sw[f"r{r}_u2_c1_w"] = _wt(u.conv1.weight)
            sw[f"r{r}_u2_c1_b"] = _bt(u.conv1.bias)
            sw[f"r{r}_u2_c2_w"] = _wt(u.conv2.weight)
            sw[f"r{r}_u2_c2_b"] = _bt(u.conv2.bias)
            # out_conv is 1×1, keep as linear weights.
            sw[f"r{r}_out_w"] = _w_lin(ref.out_conv.weight)
            sw[f"r{r}_out_b"] = _b_lin(ref.out_conv.bias)

        # output_conv1: 3×3 256→128 bias=True.
        sw["oc1_w"] = _wt(h.scratch.output_conv1.weight)
        sw["oc1_b"] = _bt(h.scratch.output_conv1.bias)
        sw["oc1_in_c"] = h.scratch.output_conv1.in_channels
        sw["oc1_out_c"] = h.scratch.output_conv1.out_channels

        h._tt_scratch_weights = sw
        h._tt_device = device
        h._tt_scratch_ready = True

    if getattr(DPTHead, "_tt_scratch_patched", False):
        return

    _SCRATCH_KCFG = None

    def _kcfg(dev):
        nonlocal _SCRATCH_KCFG
        if _SCRATCH_KCFG is None:
            _SCRATCH_KCFG = ttnn.init_device_compute_kernel_config(
                dev.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return _SCRATCH_KCFG

    def _conv2d(tt_x, w_t, b_t, in_c, out_c, B, H, W, dev, k=3, stride=1, padding=1, dtype=None):
        return ttnn.conv2d(
            input_tensor=tt_x, weight_tensor=w_t, bias_tensor=b_t,
            device=dev, in_channels=in_c, out_channels=out_c,
            batch_size=B, input_height=H, input_width=W,
            kernel_size=(k, k), stride=(stride, stride), padding=(padding, padding),
            compute_config=_kcfg(dev),
            **({"dtype": dtype} if dtype is not None else {}),
        )

    def _linear_1x1(tt_x, w_lin, b_lin, dtype=None):
        if tt_x.layout != ttnn.TILE_LAYOUT:
            tt_x = ttnn.to_layout(tt_x, ttnn.TILE_LAYOUT)
        kwargs = {"compute_kernel_config": _kcfg(tt_x.device())}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return ttnn.linear(tt_x, w_lin, bias=b_lin, **kwargs)

    def _flat_to_nhwc(tt_x, B, H, W, C):
        if tt_x.layout != ttnn.ROW_MAJOR_LAYOUT:
            tt_x = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(tt_x, (B, H, W, C))

    def _nhwc_to_flat(tt_x, B, H, W, C):
        if tt_x.is_sharded():
            tt_x = ttnn.sharded_to_interleaved(tt_x, ttnn.DRAM_MEMORY_CONFIG)
        tt_x = ttnn.reshape(tt_x, (1, 1, B * H * W, C))
        if tt_x.layout != ttnn.TILE_LAYOUT:
            tt_x = ttnn.to_layout(tt_x, ttnn.TILE_LAYOUT)
        return tt_x

    def _resconv(tt_x, w, prefix, ch, B, H, W, dev):
        # VGGT's host ResidualConvUnit uses nn.ReLU(inplace=True), which
        # mutates `x` in-place on the first activation. So the residual
        # add is actually `conv2(...) + relu(x)`, not `conv2(...) + x`.
        # Precision: both conv outputs in fp32. DPT's expp1-activated
        # conf channels lose ~4% PCC with bf16 anywhere in the chain.
        tt_relu = ttnn.relu(tt_x)
        tt_c1 = _conv2d(tt_relu, w[f"{prefix}_c1_w"], w[f"{prefix}_c1_b"], ch, ch, B, H, W, dev,
                        dtype=ttnn.float32)
        # ttnn.relu on fp32 — the ttnn implementation should support it.
        tt_c1 = ttnn.relu(tt_c1)
        tt_c2 = _conv2d(tt_c1, w[f"{prefix}_c2_w"], w[f"{prefix}_c2_b"], ch, ch, B, H, W, dev,
                        dtype=ttnn.float32)
        tt_residual = ttnn.typecast(tt_relu, ttnn.float32)
        return ttnn.add(tt_residual, tt_c2)  # fp32; caller casts if needed

    def _upload_nchw_as_flat(nchw: torch.Tensor, dev):
        """(BS, C, H, W) host -> ROW_MAJOR flat (1, 1, BS·H·W, C) device bf16."""
        Bs, C, H, W = nchw.shape
        nhwc = nchw.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)
        flat = nhwc.reshape(1, 1, Bs * H * W, C)
        return ttnn.from_torch(flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev), Bs, H, W

    def _download_flat_to_nchw(tt_x, Bs, H, W, C):
        out = ttnn.to_torch(tt_x).to(torch.float32)
        return out.reshape(Bs, H, W, C).permute(0, 3, 1, 2).contiguous()

    def _refinenet_device(name, tt_prev, tt_skip, tgt_hw, ch, B, Hi, Wi, w, dev, use_host_upsample=False):
        """Run one FeatureFusionBlock on device.
          tt_prev: (1, 1, B·Hi·Wi, ch) TILE — previous refinenet's output
          tt_skip: (1, 1, B·Hi·Wi, ch) TILE — skip tensor (None for refinenet4)
          tgt_hw: (target_H, target_W) after upsample
        """
        r = int(name[-1])
        if tt_skip is not None:
            # Skip path through _resconv returns fp32; add to tt_prev (could
            # be bf16 from upsample or fp32 from prior _resconv). Cast to
            # fp32 to keep the fusion add in fp32 precision.
            tt_skip_proc = _resconv(tt_skip, w, f"r{r}_u1", ch, B, Hi, Wi, dev)
            if tt_prev.dtype != ttnn.float32:
                tt_prev = ttnn.typecast(tt_prev, ttnn.float32)
            tt_prev = ttnn.add(tt_prev, tt_skip_proc)
        tt_out = _resconv(tt_prev, w, f"r{r}_u2", ch, B, Hi, Wi, dev)

        # Upsample — demands BFLOAT16 input on device. Cast just here so
        # the residual chain stayed in fp32.
        if tt_out.dtype != ttnn.bfloat16:
            tt_out = ttnn.typecast(tt_out, ttnn.bfloat16)
        target_H, target_W = tgt_hw
        if use_host_upsample:
            # Host round-trip: ttnn.upsample bilinear wants integer scale factor.
            nchw = _download_flat_to_nchw(tt_out, B, Hi, Wi, ch)
            up = torch.nn.functional.interpolate(
                nchw, size=(target_H, target_W), mode="bilinear", align_corners=True,
            )
            tt_up, _, _, _ = _upload_nchw_as_flat(up, dev)
            # tt_up is ROW_MAJOR flat; out_conv wants TILE flat.
            tt_up = ttnn.to_layout(tt_up, ttnn.TILE_LAYOUT)
        else:
            assert target_H == 2 * Hi and target_W == 2 * Wi, f"non-integer upsample {Hi}->{target_H}"
            tt_nhwc = _flat_to_nhwc(tt_out, B, Hi, Wi, ch)
            tt_nhwc = ttnn.upsample(tt_nhwc, scale_factor=2, mode="bilinear")
            tt_up = _nhwc_to_flat(tt_nhwc, B, target_H, target_W, ch)

        # 1×1 out_conv as linear. fp32 output keeps the feed into the
        # next refinenet (or output_conv1) in fp32.
        return _linear_1x1(tt_up, w[f"r{r}_out_w"], w[f"r{r}_out_b"], dtype=ttnn.float32)

    def tt_scratch_forward(self, features):
        if not getattr(self, "_tt_scratch_ready", False):
            return self._orig_scratch_forward(features)
        import os
        compare = bool(int(os.environ.get("VGGT_TT_SCRATCH_COMPARE", "0") or "0"))

        dev = self._tt_device
        w = self._tt_scratch_weights
        layer_1, layer_2, layer_3, layer_4 = features
        # Expected host shapes at B=1 S=1: (1, 256,148,148), (1,512,74,74),
        # (1,1024,37,37), (1,1024,19,19). For S>1 first dim is B*S.
        Bs = layer_1.shape[0]

        # For Bs > 1, the per-refinenet host upsample round-trip doubles
        # (or more) in data volume, and the Python/ttnn overhead per
        # refinenet compounds: net wall-clock regresses vs the host path
        # at S=2 (+7 % in practice). Fall back to original host forward
        # for Bs > 1 until a device fp32 bilinear upsample is available.
        # Opt out via VGGT_TT_SCRATCH_ALL_BS=1.
        if Bs > 1 and os.environ.get("VGGT_TT_SCRATCH_ALL_BS", "0") in ("", "0"):
            return self._orig_scratch_forward(features)

        # Upload 4 features as ROW_MAJOR flat and run layer_rn (3×3 conv, no bias).
        def _layer_rn(idx, host_t):
            tt_in, Bs_i, Hi, Wi = _upload_nchw_as_flat(host_t, dev)
            return _conv2d(
                tt_in, w[f"l{idx}_rn_w"], None,
                w[f"l{idx}_rn_in_c"], w[f"l{idx}_rn_out_c"],
                Bs_i, Hi, Wi, dev,
            ), Hi, Wi

        tt_l1, H1, W1 = _layer_rn(1, layer_1)   # 148
        tt_l2, H2, W2 = _layer_rn(2, layer_2)   # 74
        tt_l3, H3, W3 = _layer_rn(3, layer_3)   # 37
        tt_l4, H4, W4 = _layer_rn(4, layer_4)   # 19

        if compare:
            # layer_rn probes: verify conv weights / upload / output_dtype.
            # Use host features directly; layer_rn does not mutate its input.
            for idx, (tt_li, Hi, Wi, host_x) in enumerate(
                [(tt_l1, H1, W1, layer_1), (tt_l2, H2, W2, layer_2),
                 (tt_l3, H3, W3, layer_3), (tt_l4, H4, W4, layer_4)], 1,
            ):
                got = _download_flat_to_nchw(tt_li, Bs, Hi, Wi, 256)
                ref = getattr(self.scratch, f"layer{idx}_rn")(host_x).detach()
                a = got.float().flatten(); b = ref.float().flatten()
                a_c = a - a.mean(); b_c = b - b.mean()
                denom = (a_c.norm() * b_c.norm()).item()
                pcc_v = float((a_c @ b_c).item() / denom) if denom > 0 else 0.0
                print(f"[scratch compare] layer{idx}_rn: PCC={pcc_v:.4f}  shape={list(got.shape)}")

        # refinenet4 (no residual, non-integer upsample 19→37 via host).
        tt_p = _refinenet_device(
            "r4", tt_l4, None, (H3, W3), 256, Bs, H4, W4, w, dev,
            use_host_upsample=True,
        )
        if compare:
            _compare_refinenet(self, 4, tt_p, Bs, H3, W3, features)

        # refinenet3/2/1 (with residual). Device bf16 bilinear upsample
        # drops conf PCC below 0.99; host upsample (fp32 bilinear via
        # torch, then re-upload in bf16) is needed.
        tt_p = _refinenet_device("r3", tt_p, tt_l3, (H2, W2), 256, Bs, H3, W3, w, dev,
                                 use_host_upsample=True)
        if compare:
            _compare_refinenet(self, 3, tt_p, Bs, H2, W2, features)
        tt_p = _refinenet_device("r2", tt_p, tt_l2, (H1, W1), 256, Bs, H2, W2, w, dev,
                                 use_host_upsample=True)
        if compare:
            _compare_refinenet(self, 2, tt_p, Bs, H1, W1, features)
        tt_p = _refinenet_device("r1", tt_p, tt_l1, (H1 * 2, W1 * 2), 256, Bs, H1, W1, w, dev,
                                 use_host_upsample=True)
        if compare:
            _compare_refinenet(self, 1, tt_p, Bs, H1 * 2, W1 * 2, features)

        # output_conv1: 3×3 256→128 bias=True at (296, 296). fp32 output
        # feeds into host custom_interpolate → output_conv2 → activate_head,
        # and conf channels are precision-sensitive (expp1 activation).
        Hf, Wf = H1 * 2, W1 * 2
        tt_out = _conv2d(tt_p, w["oc1_w"], w["oc1_b"], w["oc1_in_c"], w["oc1_out_c"],
                         Bs, Hf, Wf, dev, dtype=ttnn.float32)
        return _download_flat_to_nchw(tt_out, Bs, Hf, Wf, w["oc1_out_c"])

    def _compare_refinenet(self, idx, tt_x, Bs, H, W, features):
        """Debug helper: download device tensor, run host scratch up to the
        same point, print PCC. Only active under VGGT_TT_SCRATCH_COMPARE=1."""
        import ttnn as _t
        got = _download_flat_to_nchw(tt_x, Bs, H, W, 256)
        # Run the host version but stop at refinenet{idx}.
        with torch.no_grad():
            layer_1, layer_2, layer_3, layer_4 = features
            ref_ls = [self.scratch.layer1_rn(layer_1), self.scratch.layer2_rn(layer_2),
                      self.scratch.layer3_rn(layer_3), self.scratch.layer4_rn(layer_4)]
            out = self.scratch.refinenet4(ref_ls[3], size=ref_ls[2].shape[2:])
            if idx <= 3:
                out = self.scratch.refinenet3(out, ref_ls[2], size=ref_ls[1].shape[2:])
            if idx <= 2:
                out = self.scratch.refinenet2(out, ref_ls[1], size=ref_ls[0].shape[2:])
            if idx == 1:
                out = self.scratch.refinenet1(out, ref_ls[0])
        a = got.float().flatten()
        b = out.detach().float().flatten()
        a_c = a - a.mean(); b_c = b - b.mean()
        denom = (a_c.norm() * b_c.norm()).item()
        pcc_v = float((a_c @ b_c).item() / denom) if denom > 0 else 0.0
        print(f"[scratch compare] refinenet{idx}: PCC={pcc_v:.4f}  shape={list(got.shape)}")

    DPTHead._orig_scratch_forward = DPTHead.scratch_forward
    DPTHead.scratch_forward = tt_scratch_forward
    DPTHead._tt_scratch_patched = True


def _dpt_prelude_on_device(head, x_cpu, dpt_idx, Bs, patch_h, patch_w, dev, kcfg):
    """Run one DPT prelude iteration on device.

    x_cpu : (Bs, N_patches=37*37, dim_in=2048) CPU tensor
    Returns: (Bs, out_c, H_out, W_out) float32 CPU tensor
    """
    import ttnn
    p = head._tt_pre
    out_c = p[f"out_c_{dpt_idx}"]
    N = patch_h * patch_w  # 1369

    # Upload tokens: bf16 TILE (Bs, N, dim_in).
    tt_x = ttnn.from_torch(
        x_cpu.to(torch.bfloat16).contiguous(),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
    )

    # LayerNorm: (Bs, N, dim_in) → (Bs, N, dim_in) bf16.
    tt_x = ttnn.layer_norm(
        tt_x,
        weight=head._tt_pre_ln_g,
        bias=head._tt_pre_ln_b,
        epsilon=head._tt_pre_ln_eps,
    )

    # 1×1 projection as linear: (Bs, N, dim_in) → (Bs, N, out_c).
    tt_x = ttnn.linear(tt_x, p[f"proj_w_{dpt_idx}"], bias=p[f"proj_b_{dpt_idx}"],
                       compute_kernel_config=kcfg)

    # pos_embed add: (1, N, out_c) broadcast over Bs.
    if head.pos_embed:
        tt_x = ttnn.add(tt_x, p[f"pe_{dpt_idx}"])

    resize_type = p[f"resize_type_{dpt_idx}"]

    if resize_type == "identity":
        # (Bs, N, out_c) TILE → ROW_MAJOR → NCHW (Bs, out_c, H, W) CPU.
        tt_x = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
        raw = ttnn.to_torch(tt_x).to(torch.float32)  # (Bs, N, out_c)
        return raw.reshape(Bs, patch_h, patch_w, out_c).permute(0, 3, 1, 2).contiguous()

    # Flatten to (1, 1, Bs*N, out_c) ROW_MAJOR — required by conv2d/conv_transpose2d.
    tt_x = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.reshape(tt_x, (1, 1, Bs * N, out_c))

    if resize_type == "conv_transpose":
        k = p[f"resize_k_{dpt_idx}"]
        s = p[f"resize_s_{dpt_idx}"]
        pad = p[f"resize_p_{dpt_idx}"]
        H_out = (patch_h - 1) * s[0] - 2 * pad[0] + k[0]
        W_out = (patch_w - 1) * s[1] - 2 * pad[1] + k[1]
        tt_out = ttnn.conv_transpose2d(
            input_tensor=tt_x,
            weight_tensor=p[f"resize_w_{dpt_idx}"],
            bias_tensor=p[f"resize_b_{dpt_idx}"],
            device=dev,
            in_channels=out_c,
            out_channels=out_c,
            batch_size=Bs,
            input_height=patch_h,
            input_width=patch_w,
            kernel_size=k,
            stride=s,
            padding=pad,
        )
    else:  # stride-2 Conv2d
        k = p[f"resize_k_{dpt_idx}"]
        s = p[f"resize_s_{dpt_idx}"]
        pad = p[f"resize_p_{dpt_idx}"]
        H_out = (patch_h + 2 * pad[0] - k[0]) // s[0] + 1
        W_out = (patch_w + 2 * pad[1] - k[1]) // s[1] + 1
        tt_out = ttnn.conv2d(
            input_tensor=tt_x,
            weight_tensor=p[f"resize_w_{dpt_idx}"],
            bias_tensor=p[f"resize_b_{dpt_idx}"],
            device=dev,
            in_channels=out_c,
            out_channels=out_c,
            batch_size=Bs,
            input_height=patch_h,
            input_width=patch_w,
            kernel_size=k,
            stride=s,
            padding=pad,
            compute_config=kcfg,
        )

    # conv output: (1, 1, Bs*H_out*W_out, out_c) ROW_MAJOR.
    raw = ttnn.to_torch(tt_out).to(torch.float32)
    return raw.reshape(Bs, H_out, W_out, out_c).permute(0, 3, 1, 2).contiguous()


def _install_ttnn_dpt_prelude(model, device):
    """Preload DPT prelude weights (norm + projects + pos_embed + resize_layers)
    onto device so _forward_impl can skip the CPU compute (~110 ms × 2 heads).

    Per dpt_idx (0–3), projects is a 1×1 Conv2d stored as a linear weight,
    pos_embed is precomputed for the fixed 37×37 / 518×518 geometry, and
    resize_layers weights are stored for ttnn.conv_transpose2d / ttnn.conv2d.
    Does NOT patch _forward_impl — that is done by _install_ttnn_dpt_output_conv2
    which checks _tt_prelude_ready.
    """
    import ttnn
    import torch.nn as nn
    from vggt.heads.dpt_head import DPTHead  # type: ignore
    from vggt.heads.utils import create_uv_grid, position_grid_to_embed  # type: ignore

    PATCH_HW = 37       # 518 // 14
    N_PATCHES = PATCH_HW * PATCH_HW  # 1369

    for h in model.modules():
        if not isinstance(h, DPTHead) or getattr(h, "_tt_prelude_ready", False):
            continue

        # LN weights: (1, 1, dim_in) for ttnn.layer_norm broadcasting over (Bs, N, dim_in).
        ln_g = h.norm.weight.detach().reshape(1, 1, -1).to(torch.bfloat16)
        ln_b = h.norm.bias.detach().reshape(1, 1, -1).to(torch.bfloat16)
        h._tt_pre_ln_g = ttnn.from_torch(ln_g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        h._tt_pre_ln_b = ttnn.from_torch(ln_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        h._tt_pre_ln_eps = float(h.norm.eps)

        pre = {}

        for dpt_idx, proj in enumerate(h.projects):
            out_c = proj.weight.shape[0]
            in_c  = proj.weight.shape[1]
            pre[f"out_c_{dpt_idx}"] = out_c

            # 1×1 conv as linear matmul weight (in_c, out_c) TILE.
            w = proj.weight.detach().reshape(out_c, in_c).t().contiguous().to(torch.bfloat16)
            pre[f"proj_w_{dpt_idx}"] = ttnn.from_torch(
                w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            pre[f"proj_b_{dpt_idx}"] = (
                ttnn.from_torch(
                    proj.bias.detach().reshape(1, 1, -1).to(torch.bfloat16),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                ) if proj.bias is not None else None
            )

            # pos_embed: precomputed for 37×37 grid at aspect_ratio=1.0 (518×518 input).
            # Stored as (1, N=1369, out_c) TILE for broadcast add to (Bs, N, out_c).
            # _apply_pos_embed does permute(2,0,1)[None].expand before adding to NCHW;
            # here we flatten the spatial dims to match the (Bs, N, C) token layout.
            pe = create_uv_grid(PATCH_HW, PATCH_HW, aspect_ratio=1.0)  # (37, 37, 2)
            pe = position_grid_to_embed(pe, out_c)                       # (37, 37, out_c)
            pe = pe * 0.1                                                 # ratio=0.1
            pre[f"pe_{dpt_idx}"] = ttnn.from_torch(
                pe.reshape(1, N_PATCHES, out_c).to(torch.bfloat16).contiguous(),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
            )

        for dpt_idx, rl in enumerate(h.resize_layers):
            if isinstance(rl, nn.ConvTranspose2d):
                # PyTorch ConvTranspose2d weight: (in_c, out_c, kH, kW) matches ttnn format.
                # Bias: ttnn.conv_transpose2d internally calls prepare_conv_bias which expects
                # (1, 1, 1, out_c) — same as ttnn.conv2d, not the 1D shape the API docs state.
                pre[f"resize_type_{dpt_idx}"] = "conv_transpose"
                pre[f"resize_w_{dpt_idx}"] = ttnn.from_torch(
                    rl.weight.detach().to(torch.bfloat16), dtype=ttnn.bfloat16
                )
                pre[f"resize_b_{dpt_idx}"] = (
                    ttnn.from_torch(
                        rl.bias.detach().reshape(1, 1, 1, -1).to(torch.bfloat16),
                        dtype=ttnn.bfloat16,
                    ) if rl.bias is not None else None
                )
                pre[f"resize_k_{dpt_idx}"] = (rl.kernel_size[0], rl.kernel_size[1])
                pre[f"resize_s_{dpt_idx}"] = (rl.stride[0], rl.stride[1])
                pre[f"resize_p_{dpt_idx}"] = (rl.padding[0], rl.padding[1])
                pre[f"resize_out_c_{dpt_idx}"] = rl.out_channels
            elif isinstance(rl, nn.Conv2d):
                pre[f"resize_type_{dpt_idx}"] = "conv"
                pre[f"resize_w_{dpt_idx}"] = ttnn.from_torch(
                    rl.weight.detach().to(torch.bfloat16), dtype=ttnn.bfloat16
                )
                pre[f"resize_b_{dpt_idx}"] = (
                    ttnn.from_torch(
                        rl.bias.detach().reshape(1, 1, 1, -1).to(torch.bfloat16),
                        dtype=ttnn.bfloat16,
                    ) if rl.bias is not None else None
                )
                pre[f"resize_k_{dpt_idx}"] = (rl.kernel_size[0], rl.kernel_size[1])
                pre[f"resize_s_{dpt_idx}"] = (rl.stride[0], rl.stride[1])
                pre[f"resize_p_{dpt_idx}"] = (rl.padding[0], rl.padding[1])
                pre[f"resize_out_c_{dpt_idx}"] = rl.out_channels
            else:
                pre[f"resize_type_{dpt_idx}"] = "identity"

        h._tt_pre = pre
        h._tt_device = device
        h._tt_prelude_ready = True


def _install_ttnn_dpt_output_conv2(model, device):
    """Port DPTHead.scratch.output_conv2 (3x3 conv -> relu -> 1x1 conv at
    518x518) to ttnn.conv2d. This chain is the biggest single DPT chunk
    (~227 ms per head). output_conv1 + scratch_forward + interpolate still
    run on host for now; this port isolates the final conv stack.
    """
    import ttnn
    import torch.nn as nn
    from vggt.heads.dpt_head import DPTHead, custom_interpolate  # type: ignore
    from vggt.heads.head_act import activate_head  # type: ignore

    for h in model.modules():
        if isinstance(h, DPTHead) and not getattr(h, "_tt_oc2_ready", False):
            # track_head.feature_extractor is a feature_only DPTHead without
            # output_conv2 — skip.
            if getattr(h, "feature_only", False):
                continue
            if not hasattr(h.scratch, "output_conv2"):
                continue
            if not isinstance(h.scratch.output_conv2, nn.Sequential):
                continue
            seq = h.scratch.output_conv2
            c3x3 = seq[0]  # Conv2d 128 -> 32 k=3 s=1 p=1
            c1x1 = seq[2]  # Conv2d 32 -> output_dim k=1
            if not (isinstance(c3x3, nn.Conv2d) and isinstance(c1x1, nn.Conv2d)):
                continue

            # ttnn.conv2d wants weights as (out, in, kH, kW) in bf16.
            h._tt_oc2_w0 = ttnn.from_torch(
                c3x3.weight.detach().to(torch.bfloat16), dtype=ttnn.bfloat16,
            )
            h._tt_oc2_b0 = ttnn.from_torch(
                c3x3.bias.detach().reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
            )
            h._tt_oc2_w1 = ttnn.from_torch(
                c1x1.weight.detach().to(torch.bfloat16), dtype=ttnn.bfloat16,
            )
            h._tt_oc2_b1 = ttnn.from_torch(
                c1x1.bias.detach().reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16,
            )
            h._tt_oc2_in_c = c3x3.in_channels
            h._tt_oc2_mid_c = c3x3.out_channels
            h._tt_oc2_out_c = c1x1.out_channels
            h._tt_device = device
            h._tt_oc2_ready = True

    if getattr(DPTHead, "_tt_oc2_patched", False):
        return

    _DPT_KCFG = None

    def _dpt_kcfg(dev):
        nonlocal _DPT_KCFG
        if _DPT_KCFG is None:
            _DPT_KCFG = ttnn.init_device_compute_kernel_config(
                dev.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return _DPT_KCFG

    def tt_forward_impl(self, aggregated_tokens_list, images,
                        patch_start_idx, frames_start_idx=None,
                        frames_end_idx=None):
        if not getattr(self, "_tt_oc2_ready", False):
            return self._orig_forward_impl(
                aggregated_tokens_list, images, patch_start_idx,
                frames_start_idx, frames_end_idx,
            )
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        use_tt_prelude = getattr(self, "_tt_prelude_ready", False)
        out = []
        for dpt_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]
            Bs = B * S
            x = x.reshape(Bs, -1, x.shape[-1])  # (Bs, N_patches, dim_in)

            if use_tt_prelude:
                x = _dpt_prelude_on_device(
                    self, x, dpt_idx, Bs, patch_h, patch_w,
                    self._tt_device, _dpt_kcfg(self._tt_device),
                )
            else:
                x = self.norm(x)
                x = x.permute(0, 2, 1).reshape((Bs, x.shape[-1], patch_h, patch_w))
                x = self.projects[dpt_idx](x)
                if self.pos_embed:
                    x = self._apply_pos_embed(x, W, H)
                x = self.resize_layers[dpt_idx](x)
            out.append(x)

        fused = self.scratch_forward(out)
        fused = custom_interpolate(
            fused,
            (int(patch_h * self.patch_size / self.down_ratio),
             int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear", align_corners=True,
        )
        if self.pos_embed:
            fused = self._apply_pos_embed(fused, W, H)

        if self.feature_only:
            return fused.view(B, S, *fused.shape[1:])

        # ---- output_conv2 on device ----
        # fused is (BS, 128, 518, 518) fp32 on host. Permute to NHWC flat and upload.
        Bf, Cin, Hf, Wf = fused.shape
        x_nhwc = fused.permute(0, 2, 3, 1).contiguous().to(torch.bfloat16)  # (BS, Hf, Wf, Cin)
        x_flat = x_nhwc.reshape(1, 1, Bf * Hf * Wf, Cin)
        tt_x = ttnn.from_torch(
            x_flat, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self._tt_device,
        )
        tt_x = ttnn.conv2d(
            input_tensor=tt_x, weight_tensor=self._tt_oc2_w0, bias_tensor=self._tt_oc2_b0,
            device=self._tt_device,
            in_channels=self._tt_oc2_in_c, out_channels=self._tt_oc2_mid_c,
            batch_size=Bf, input_height=Hf, input_width=Wf,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
            compute_config=_dpt_kcfg(self._tt_device),
        )
        tt_x = ttnn.relu(tt_x)
        tt_x = ttnn.conv2d(
            input_tensor=tt_x, weight_tensor=self._tt_oc2_w1, bias_tensor=self._tt_oc2_b1,
            device=self._tt_device,
            in_channels=self._tt_oc2_mid_c, out_channels=self._tt_oc2_out_c,
            batch_size=Bf, input_height=Hf, input_width=Wf,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0),
            compute_config=_dpt_kcfg(self._tt_device),
        )
        out_host = ttnn.to_torch(tt_x).to(torch.float32)
        # Conv2d output comes back as (1, 1, BS*Hf*Wf, out_c). Reshape to NHWC then NCHW.
        out_host = out_host.reshape(Bf, Hf, Wf, self._tt_oc2_out_c).permute(0, 3, 1, 2).contiguous()

        preds, conf = activate_head(
            out_host, activation=self.activation, conf_activation=self.conf_activation,
        )
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    DPTHead._orig_forward_impl = DPTHead._forward_impl
    DPTHead._forward_impl = tt_forward_impl
    DPTHead._tt_oc2_patched = True


def _prewarm_seqs(model, device, seqs, img_size: int = 518):
    """Run dummy zero-input forwards at each requested S to populate the
    ttnn program cache. ttnn compiles a new program on first encounter of
    any (shape, layout, memory_config) tuple; without prewarm, the first
    real forward at an unseen S hangs 20+ min in per-op compile across the
    dozens of matmul/softmax/layer_norm/conv2d/nlp_create_qkv_heads
    variants exercised by the aggregator Block. Paying that cost at
    install time makes the first real inference latency predictable.
    """
    import time
    for S in seqs:
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(torch.zeros(1, S, 3, img_size, img_size, dtype=torch.float32))
        print(f"[vggt] prewarm S={S}: {time.perf_counter() - t0:.1f}s")


def _install_ttnn_aggregator_padding(model, device, s_canon: int):
    """Patch Aggregator.forward to always run at S=s_canon. BF0 option 2.

    ttnn program-cache keys on (shape, layout, memory_config). Global
    attention in the VGGT aggregator sees N = S × P (P=1374 @ 518×518).
    A fresh S introduces an untold number of new compile shapes that
    cumulatively hang for 20+ min. By always running at a single s_canon
    the program cache only ever sees one set of shapes, so the hang is
    paid once at install time.

    Padding strategy: replicate the last real frame to fill up to s_canon
    so patch_embed + layer_norm remain numerically sensible. Global
    attention is masked so real frames do not attend to padded key
    positions (see _ACTIVE_GLOBAL_MASK wiring in tt_block_forward).
    Frame attention is per-frame so the padding frames produce their own
    outputs that we then slice away.

    Outputs are sliced back to the caller's real S before return.
    """
    import ttnn
    from vggt.models.aggregator import Aggregator  # type: ignore
    import torch.nn as nn

    global _PATCH_COUNT
    # Infer P (patches + special tokens). 518/14=37 → 37*37=1369 patches,
    # + patch_start_idx (1 camera + 4 register tokens) = 1374.
    # This is invariant for VGGT's fixed 518 input size.
    for m in model.modules():
        if isinstance(m, Aggregator):
            aggregator = m
            break
    else:
        raise RuntimeError("No Aggregator found in model")

    num_patches = (518 // aggregator.patch_size) ** 2
    P = num_patches + aggregator.patch_start_idx
    _PATCH_COUNT = P
    aggregator._tt_device = device
    aggregator._tt_s_canon = s_canon

    # Precompute the additive global-attention mask shape (1, 1, 1, S*P):
    # 0 for valid token positions, -inf for padding. Broadcasts over batch,
    # heads, and query dim. We hold one mask per (s_canon, S_real) pair;
    # built on demand.
    aggregator._tt_mask_cache = {}

    def _mask_for(S_real: int):
        key = (s_canon, S_real)
        cached = aggregator._tt_mask_cache.get(key)
        if cached is not None:
            return cached
        total = s_canon * P
        # fp32 mask: added directly to fp32 scores in tt_block_forward.
        # fp32 broadcast add (1,1,1,N)→(1,H,N,N) works at N=5496 (probed).
        m = torch.zeros(1, 1, 1, total, dtype=torch.float32)
        m[..., S_real * P:] = float("-inf")
        tt_m = ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        aggregator._tt_mask_cache[key] = tt_m
        return tt_m

    if getattr(Aggregator, "_tt_padding_patched", False):
        Aggregator._tt_s_canon = s_canon
        return

    orig_forward = Aggregator.forward

    def padded_forward(self, images):
        global _ACTIVE_GLOBAL_MASK
        B, S_real, Ci, H, W = images.shape
        S_canon = getattr(self, "_tt_s_canon", S_real)
        if S_real == S_canon:
            # No padding needed.
            return orig_forward(self, images)
        if S_real > S_canon:
            raise RuntimeError(
                f"S_real={S_real} exceeds the canonical S={S_canon} that was "
                f"prewarmed at install. Set VGGT_S_CANON to a larger value "
                f"(e.g. {max(4, S_real)}) and pay the extra install-time compile."
            )
        # Pad by replicating the last real frame.
        pad_count = S_canon - S_real
        last = images[:, -1:].contiguous()
        padding = last.expand(B, pad_count, Ci, H, W)
        images_padded = torch.cat([images, padding], dim=1).contiguous()

        _ACTIVE_GLOBAL_MASK = _mask_for(S_real)
        try:
            output_list, patch_start_idx = orig_forward(self, images_padded)
        finally:
            _ACTIVE_GLOBAL_MASK = None
        # Slice outputs [B, S_canon, P, 2C] → [B, S_real, P, 2C].
        output_list = [o[:, :S_real].contiguous() for o in output_list]
        return output_list, patch_start_idx

    Aggregator.forward = padded_forward
    Aggregator._tt_padding_patched = True


def _ensure_installed(device, prewarm_seqs=(1,), s_canon: Optional[int] = None):
    if _INSTALL_DONE.get(id(device)):
        return
    import os
    model = _get_model()

    # BF0 option 2: pad-to-canonical-S. Set VGGT_S_CANON (or pass
    # s_canon=N explicitly) to run the aggregator at S=N always,
    # padding short inputs + masking global-attn. This pins the
    # ttnn program-cache to one set of shapes, avoiding the 20+ min
    # first-forward compile stall at each new S. Default: no padding
    # (s_canon=1) — preserves known-good S=1 / S=2 path.
    if s_canon is None:
        s_canon = int(os.environ.get("VGGT_S_CANON", "1") or "1")

    # RoPE tables must exist before the block patch reads them.
    _install_ttnn_rope_tables(model, device)
    _install_ttnn_block(model, device)

    # Mark blocks eligible for on-device residual passthrough (skip per-block
    # PCIe round-trip). Aggregator frame/global blocks see constant shape for
    # S=1 (NOP reshape) so _use_pass is always True in that regime. DINOv2
    # patch_embed blocks also chain at constant (B*S, P, C) shape.
    from vggt.models.aggregator import Aggregator  # type: ignore
    for _m in model.modules():
        if isinstance(_m, Aggregator):
            for _blk in list(_m.frame_blocks) + list(_m.global_blocks):
                if getattr(_blk, "_tt_block_ready", False):
                    _blk._tt_can_pass = True
            if hasattr(_m.patch_embed, "blocks"):
                for _blk in _m.patch_embed.blocks:
                    if getattr(_blk, "_tt_block_ready", False):
                        _blk._tt_can_pass = True
            break

    _install_ttnn_mlp(model, device)
    # P0 scratch_forward port: layer_rn + refinenet{4,3,2,1} + output_conv1
    # on device. host upsample per refinenet (device bilinear in bf16
    # dropped conf PCC). ~8 % wall-clock win vs host scratch, PCC 0.9957
    # (vs 0.9959 baseline). Opt out via VGGT_TT_SCRATCH=0 if troubleshooting.
    if os.environ.get("VGGT_TT_SCRATCH", "1") not in ("", "0"):
        _install_ttnn_dpt_scratch(model, device)
    # P1 DPT prelude port: norm + projects + pos_embed + resize_layers on device.
    # Tested at S=1: PCC 0.9957 (= baseline), latency +6 ms vs CPU baseline.
    # No gain because hot-path CPU compute (~20 ms) ≈ device PCIe overhead (8
    # upload+download round-trips × 5-11 MB each). Would require keeping prelude
    # outputs on device and feeding directly into scratch_forward (avoids
    # download+re-upload) to break even. Disabled by default; enable via
    # VGGT_TT_PRELUDE=1 to test or to explore at larger S.
    if os.environ.get("VGGT_TT_PRELUDE", "0") not in ("", "0"):
        _install_ttnn_dpt_prelude(model, device)
    _install_ttnn_dpt_output_conv2(model, device)
    if s_canon > 1:
        _install_ttnn_aggregator_padding(model, device, s_canon)
    _INSTALL_DONE[id(device)] = True
    if prewarm_seqs:
        # When padding is active, all real forwards run at S=s_canon, so
        # prewarm only at s_canon (ignore the legacy per-S warmups).
        prewarm_actual = (s_canon,) if s_canon > 1 else prewarm_seqs
        _prewarm_seqs(model, device, prewarm_actual)


def vggt_forward(images: torch.Tensor, device: Any = None,
                 query_points: Optional[torch.Tensor] = None,
                 prewarm_seqs=(1,)) -> Dict[str, torch.Tensor]:
    if device is None:
        raise RuntimeError("ttnn device handle required")
    _ensure_installed(device, prewarm_seqs=prewarm_seqs)
    model = _get_model()
    with torch.no_grad():
        return model(images, query_points=query_points)
