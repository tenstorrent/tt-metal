"""ttnn port of VGGT on a single Tenstorrent p150a Blackhole chip.

Strategy: keep the torch reference structure intact and monkey-patch
specific sub-modules to route their compute through ttnn. Each port lifts
one op class off CPU; weights are uploaded once at install time and then
re-used for every subsequent forward call.

Ports applied (all on the p150a device):
  - Every transformer Block (72 aggregator + 24 DINOv2 + 4 camera-head
    trunk = 100) runs its full residual path on device: norm1 + qkv +
    (optional qk_norm) + attention scores/softmax/context + merge_heads
    + proj + ls1 + residual_add + norm2 + fc1 + gelu + fc2 + ls2 +
    residual_add. Only RoPE still drops to host (2D position handling
    needs a separate port).
  - Every standalone Mlp (notably camera_head.pose_branch) runs on device
    as a fallback when no owning Block is present.

Precision profile:
  - Weights + matmul inputs: bfloat16.
  - Attention scores + softmax + context: fp32 intermediate via HiFi4 +
    dtype=float32. Bf16 softmax over 1374-long rows collapsed the
    world_points_conf PCC; fp32 keeps it above 0.99.
  - proj and mlp matmuls: HiFi4 + fp32 dest accumulation.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch


_CACHED_MODEL = None
_INSTALL_DONE: dict = {}
_HIFI_KCONFIG = None


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

        B, N, C = x.shape
        H = self._tt_heads
        Dh = self._tt_head_dim
        dev = self._tt_device
        kcfg = _hifi_kconfig(dev)

        # Hold the residual accumulator in fp32 on device. Matmul/LN
        # inputs are cast down to bf16 per op. Without fp32 accumulation
        # the world_points_conf PCC drops from 0.9946 -> 0.9778 over 48
        # aggregator blocks of bf16-rounded residuals.
        x_f32 = x.to(torch.float32) if x.dtype != torch.float32 else x
        tt_x = ttnn.from_torch(
            x_f32.contiguous(), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=dev,
        )

        # ---- attention branch ----
        # LN input gets cast to bf16 for the kernel, output stays bf16.
        tt_n = ttnn.typecast(tt_x, ttnn.bfloat16)
        tt_n = ttnn.layer_norm(
            tt_n, weight=self._tt_ln1_g, bias=self._tt_ln1_b, epsilon=self._tt_ln1_eps,
        )
        tt_qkv = ttnn.linear(tt_n, self._tt_qkv_w, bias=self._tt_qkv_b)

        # Split qkv on device into (q, k^T, v). nlp_create_qkv_heads wants
        # a 4D input (B, 1, N, 3*H*Dh) and returns:
        #   q:  (B, H, N, Dh)
        #   kt: (B, H, Dh, N)  (pre-transposed for Q @ K^T)
        #   v:  (B, H, N, Dh)
        tt_qkv = ttnn.reshape(tt_qkv, (B, 1, N, 3 * H * Dh))
        tt_q, tt_kt, tt_v = ttnn.experimental.nlp_create_qkv_heads(
            tt_qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=True,
        )

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

        # RoPE still on host (drop down, apply, come back).
        if self._tt_has_rope:
            q = ttnn.to_torch(tt_q)
            k = ttnn.to_torch(tt_kt).transpose(-2, -1).contiguous()
            q = self.attn.rope(q, pos)
            k = self.attn.rope(k, pos)
            tt_q = _upload(q, dev)
            tt_kt = _upload(k.transpose(-2, -1).contiguous(), dev)

        # Attention compute on device. fp32 intermediate for softmax
        # stability over the 1374-long row (bf16 collapsed conf PCC).
        tt_scores = ttnn.matmul(tt_q, tt_kt, compute_kernel_config=kcfg, dtype=ttnn.float32)
        tt_scores = ttnn.multiply(tt_scores, 1.0 / math.sqrt(Dh))
        tt_attn = ttnn.softmax(tt_scores, dim=-1, compute_kernel_config=kcfg)
        # Context: match the precision profile of the working (non-fused)
        # attention — fp32 intermediate then back to bf16 for the proj
        # matmul. bf16 context alone pushed world_points_conf to 0.9788.
        tt_ctx = ttnn.matmul(tt_attn, tt_v, compute_kernel_config=kcfg, dtype=ttnn.float32)

        # Merge heads on device: (B, H, N, Dh) -> (B, N, H*Dh). Permute in
        # fp32 then cast to bf16 for the proj matmul.
        tt_ctx = ttnn.permute(tt_ctx, (0, 2, 1, 3))
        tt_ctx = ttnn.reshape(tt_ctx, (B, N, H * Dh))
        tt_ctx = ttnn.typecast(tt_ctx, ttnn.bfloat16)

        # proj with HiFi4 + fp32 dest so the residual contribution is fp32.
        tt_attn_out = ttnn.linear(
            tt_ctx, self._tt_proj_w, bias=self._tt_proj_b,
            compute_kernel_config=kcfg, dtype=ttnn.float32,
        )
        if self._tt_ls1 is not None:
            tt_attn_out = ttnn.multiply(tt_attn_out, self._tt_ls1)
        # fp32 + fp32 residual add.
        tt_x = ttnn.add(tt_x, tt_attn_out)

        # ---- MLP branch ----
        tt_n = ttnn.typecast(tt_x, ttnn.bfloat16)
        tt_n = ttnn.layer_norm(
            tt_n, weight=self._tt_ln2_g, bias=self._tt_ln2_b, epsilon=self._tt_ln2_eps,
        )
        tt_m = ttnn.linear(tt_n, self._tt_fc1_w, bias=self._tt_fc1_b)
        tt_m = ttnn.gelu(tt_m)
        # fc2 emits fp32 for the residual stream.
        tt_mlp = ttnn.linear(
            tt_m, self._tt_fc2_w, bias=self._tt_fc2_b,
            compute_kernel_config=kcfg, dtype=ttnn.float32,
        )
        if self._tt_ls2 is not None:
            tt_mlp = ttnn.multiply(tt_mlp, self._tt_ls2)
        tt_x = ttnn.add(tt_x, tt_mlp)

        return ttnn.to_torch(tt_x).to(x.dtype)

    Block._orig_forward = Block.forward
    Block.forward = tt_block_forward
    Block._tt_block_patched = True


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

def _ensure_installed(device):
    if _INSTALL_DONE.get(id(device)):
        return
    model = _get_model()
    _install_ttnn_block(model, device)
    _install_ttnn_mlp(model, device)
    _INSTALL_DONE[id(device)] = True


def vggt_forward(images: torch.Tensor, device: Any = None,
                 query_points: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    if device is None:
        raise RuntimeError("ttnn device handle required")
    _ensure_installed(device)
    model = _get_model()
    with torch.no_grad():
        return model(images, query_points=query_points)
