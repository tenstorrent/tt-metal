# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiMo ViT (``visual.*``) on one 1x8 row, tensor-parallel (TP=8).

Per chip: 4 of the 32 Q heads and 1 of the 8 KV heads (chip c holds Q heads 4c..4c+3 and the KV head they share),
a 576-wide slice of the SwiGLU MLP and a 640-wide slice of the merger MLP. The residual stream [N, 1280] is
replicated; the attention output projection, the MLP down projection and the merger's second linear end in an
all-reduce over the row. In BF16 that is ~180 MB of weights per chip (full replication would be ~1.4 GB).

Attention (see reference/vision.py for the semantics):

* Full-attention blocks (0/9/18/27): flash SDPA with ``cu_window_seqlens``, one window per frame / image.
* Window blocks: |i - j| <= 64 inside each frame plus a per-head sink on the frame's first key. Queries run in blocks
  of 64 against the 192 keys around them (the block and 64 on each side): [blocks, 1, 4 heads x 64, 192] scores
  from one batched matmul, an additive mask (band, frame boundaries, padding; the sink as an indicator times the
  head's sink), softmax, and a batched matmul with V. The scores cost 1.5x the band's useful work, nothing else
  grows with the sequence.
* Column-order blocks (5-8, 14-17, 23-26): the residual stream is permuted (an embedding lookup of its rows) on
  entering and leaving each run, as HF does; RoPE tables come in both orders. The band mask is the same in both
  orders (frames keep their row range).

The 1/sqrt(64) attention scale is folded into the Q weights and bias (a power of two: exact in bf16).

Every attention sequence (a frame pair of a video or an image) is independent, so a long video runs as several
passes of whole frames (``max_rows`` patch rows per pass); a single image always runs in one pass.
"""

from dataclasses import dataclass

import torch

import ttnn

from ...reference import vision as ref
from ..common import MeshPlan, as_weight, cache_fill, cache_name, compute_config
from ..decoder import TtRMSNorm

BLOCK = 64  # query rows per window-attention block; keys per block: BLOCK + 2 * window
MASK_NEG = -1.0e9
DEFAULT_MAX_ROWS = 16384


@dataclass(frozen=True)
class VisionNumerics:
    """``matmul_fidelity``: the linears' math fidelity (attention matmuls, softmax, RoPE and SDPA always run HiFi4
    with fp32 accumulation). ``residual_dtype``: the residual stream between blocks. ``reduce_dtype``: the partial
    sums of the row-parallel projections (attention output, MLP down, merger) and their all-reduce."""

    matmul_fidelity: object = ttnn.MathFidelity.HiFi2
    residual_dtype: object = ttnn.bfloat16
    reduce_dtype: object = ttnn.bfloat16


NUMERICS = {
    "fast": VisionNumerics(),
    "balanced": VisionNumerics(ttnn.MathFidelity.HiFi4, ttnn.float32, ttnn.bfloat16),
    "accurate": VisionNumerics(ttnn.MathFidelity.HiFi4, ttnn.float32, ttnn.float32),
}


def padded_rows(n):
    """Pass row count bucket: a multiple of 128 (SDPA chunks, 64-row attention blocks, 32-row merged tiles), coarser
    as ``n`` grows (at most ~1/16 padding), so programs are reused across similar sizes."""
    step = 128
    while n > 16 * step:
        step *= 2
    return -(-n // step) * step


@dataclass
class PassGeometry:
    """Host-side layout of one pass: its sequences (frames / images) back to back, padded to ``rows``."""

    grids: list  # [(t, h, w)] in row order; every temporal step of an entry is its own sequence
    n: int  # real patch rows
    rows: int  # padded rows

    @classmethod
    def build(cls, grids):
        n = sum(t * h * w for t, h, w in grids)
        return cls(grids=list(grids), n=n, rows=padded_rows(n))

    def seq_bounds(self):
        """[0, ..., n, rows]: the sequences, then the padding as one more window (cu_window_seqlens)."""
        cu = ref.cu_seqlens(self.grids).tolist()
        return cu + ([self.rows] if self.rows > self.n else [])


def plan_passes(grid_thw, max_rows):
    """Split items into passes of whole frames, at most ``max_rows`` patch rows each (a single frame / image larger
    than that gets a pass of its own). Returns [(row_start, row_end, [(t, h, w), ...])]."""
    passes, cur, start, rows = [], [], 0, 0
    for t, h, w in ref.as_grid_list(grid_thw):
        frame = h * w
        left = t
        while left:
            fit = max(1, (max_rows - rows) // frame) if rows else max(1, max_rows // frame)
            take = min(left, fit)
            if rows and rows + take * frame > max_rows:
                passes.append((start, start + rows, cur))
                start, rows, cur = start + rows, 0, []
                continue
            cur.append((take, h, w))
            rows += take * frame
            left -= take
    if cur:
        passes.append((start, start + rows, cur))
    return passes


def band_masks(geo, window, heads):
    """Window-attention mask and sink indicator, each [blocks, 1, heads * BLOCK, BLOCK + 2 * window] bf16.

    Block b's queries are rows b*BLOCK .. +BLOCK (repeated per head, head-major), its keys rows b*BLOCK - window ..
    b*BLOCK + BLOCK + window. ``mask``: 0 where key j is visible from query i (same sequence, |i - j| <= window,
    0 <= j < rows), MASK_NEG elsewhere. ``sink``: 1 where a visible key is its sequence's first row."""
    rows = geo.rows
    nb = rows // BLOCK
    span = BLOCK + 2 * window
    bounds = geo.seq_bounds()
    seq_id = torch.empty(rows, dtype=torch.int64)
    is_first = torch.zeros(rows, dtype=torch.bool)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        seq_id[a:b] = s
        if a < geo.n:
            is_first[a] = True
    i = torch.arange(rows).view(nb, BLOCK, 1)
    j = (torch.arange(nb) * BLOCK - window).view(nb, 1, 1) + torch.arange(span).view(1, 1, span)
    inside = (j >= 0) & (j < rows)
    jc = j.clamp(0, rows - 1)
    visible = inside & ((i - j).abs() <= window) & (seq_id[i] == seq_id[jc])
    mask = torch.where(visible, 0.0, MASK_NEG)
    sink = (visible & is_first[jc]).float()
    expand = lambda t: t.unsqueeze(1).repeat(1, heads, 1, 1).reshape(nb, 1, heads * BLOCK, span)
    return expand(mask).bfloat16(), expand(sink).bfloat16()


class PassInputs:
    """Device tensors describing one pass (replicated on the row): RoPE tables in both orders, the order
    permutations, the SDPA windows and the band masks."""

    def __init__(self, plan, cfg, geo):
        self.geo = geo
        rows = geo.rows
        mesh = plan.mesh_device
        rep = plan.replicate()
        merge = cfg.spatial_merge_size
        unit = cfg.merge_unit

        def dev(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh, mesh_mapper=rep)

        pad = rows - geo.n
        freqs = ref.rotary_freqs(geo.grids, cfg.head_dim, merge)
        emb = torch.cat((freqs, freqs), dim=-1)
        index = ref.col_window_index(geo.grids, merge)
        col_emb = ref.apply_index(emb, index, unit)

        def table(e, fn, fill):
            t = torch.cat([fn(e), torch.full((pad, e.shape[-1]), fill)]) if pad else fn(e)
            return dev(t.reshape(1, 1, rows, -1))

        self.cos = {False: table(emb, torch.cos, 1.0), True: table(col_emb, torch.cos, 1.0)}
        self.sin = {False: table(emb, torch.sin, 0.0), True: table(col_emb, torch.sin, 0.0)}

        # Row permutations (padding rows stay in place): to column order and back.
        to_col = ref.patch_row_index(index, unit)
        to_col = torch.cat([to_col, torch.arange(geo.n, rows)])
        to_row = torch.argsort(to_col)
        self.order = {
            True: dev(to_col.reshape(1, rows).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            False: dev(to_row.reshape(1, rows).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        }
        self.cu = dev(torch.tensor(geo.seq_bounds(), dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        mask, sink = band_masks(geo, cfg.window_size, cfg.num_heads // plan.tp)
        self.mask = dev(mask)
        self.sink = dev(sink)
        # The halo around the first / last window block's keys (masked out; zeros keep them finite).
        self.halo = dev(torch.zeros(1, 1, cfg.window_size, cfg.head_dim))

    def deallocate(self):
        tensors = [*self.cos.values(), *self.sin.values(), *self.order.values(), self.cu, self.mask, self.sink]
        for t in tensors + [self.halo]:
            t.deallocate(True)


def norm_bf16(norm, x):
    """RMSNorm of the residual stream, as bf16 matmul input."""
    h = norm(x, prefill=True)
    if h.dtype == ttnn.bfloat16:
        return h
    out = ttnn.typecast(h, ttnn.bfloat16)
    h.deallocate(True)
    return out


class TtVisionBlock:
    def __init__(self, plan, cfg, i, sd, cache_dir, weight_dtype, numerics, mm_config, attn_config):
        self.plan = plan
        self.cfg = cfg
        self.numerics = numerics
        self.i = i
        self.full = cfg.is_full(i)
        self.col = cfg.is_col(i)
        self.has_sink = cfg.has_sink(i)
        self.mm_config = mm_config
        self.attn_config = attn_config
        tp = plan.tp
        hd = cfg.head_dim
        self.local_heads = cfg.num_heads // tp
        self.local_kv = cfg.num_kv_heads // tp
        assert self.local_kv == 1 and self.local_heads * tp == cfg.num_heads, "one KV head per chip"
        prefix = f"vit_tp{tp}_b{i}"
        name = lambda n: cache_name(cache_dir, prefix, n)
        p = f"blocks.{i}."
        w = {}
        if sd is not None:
            hidden = cfg.hidden_size
            scale = hd**-0.5
            qkv_w, qkv_b = sd[p + "attn.qkv.weight"], sd[p + "attn.qkv.bias"]
            q_dim, kv_dim = cfg.num_heads * hd, cfg.num_kv_heads * hd
            q_w, k_w, v_w = qkv_w[:q_dim] * scale, qkv_w[q_dim : q_dim + kv_dim], qkv_w[q_dim + kv_dim :]
            q_b, k_b, v_b = qkv_b[:q_dim] * scale, qkv_b[q_dim : q_dim + kv_dim], qkv_b[q_dim + kv_dim :]
            qh, kh = self.local_heads * hd, hd
            chips_w, chips_b = [], []
            for c in range(tp):
                chips_w += [q_w[c * qh : (c + 1) * qh], k_w[c * kh : (c + 1) * kh], v_w[c * kh : (c + 1) * kh]]
                chips_b += [q_b[c * qh : (c + 1) * qh], k_b[c * kh : (c + 1) * kh], v_b[c * kh : (c + 1) * kh]]
            w["wqkv"] = torch.cat(chips_w).T.contiguous()  # [hidden, tp * (qh + 2 kh)], chip blocks side by side
            w["bqkv"] = torch.cat(chips_b).reshape(1, -1)
            w["wo"] = sd[p + "attn.proj.weight"].T.contiguous()  # [heads * hd, hidden]: chip c's rows = its heads
            w["bo"] = self._first_chip_bias(sd[p + "attn.proj.bias"], tp)
            w["wg"] = sd[p + "mlp.gate_proj.weight"].T.contiguous()
            w["bg"] = sd[p + "mlp.gate_proj.bias"].reshape(1, -1)
            w["wu"] = sd[p + "mlp.up_proj.weight"].T.contiguous()
            w["bu"] = sd[p + "mlp.up_proj.bias"].reshape(1, -1)
            w["wd"] = sd[p + "mlp.down_proj.weight"].T.contiguous()
            w["bd"] = self._first_chip_bias(sd[p + "mlp.down_proj.bias"], tp)
            if self.has_sink:
                sinks = sd[p + "attn.sinks"].float()
                span = BLOCK + 2 * cfg.window_size
                col = sinks.view(tp, self.local_heads, 1).repeat_interleave(BLOCK, dim=1)  # [tp, heads * 64, 1]
                w["sink"] = col.expand(tp, self.local_heads * BLOCK, span).permute(1, 0, 2).reshape(1, 1, -1, tp * span)
            assert hidden == w["wo"].shape[1]

        def load(key, mapper, dtype=weight_dtype):
            return as_weight(w.get(key), plan, mapper, dtype, name(key))

        self.wqkv = load("wqkv", plan.shard(-1))
        self.bqkv = load("bqkv", plan.shard(-1), ttnn.bfloat16)
        self.wo = load("wo", plan.shard(-2))
        self.bo = load("bo", plan.shard(-1), ttnn.bfloat16)
        self.wg = load("wg", plan.shard(-1))
        self.bg = load("bg", plan.shard(-1), ttnn.bfloat16)
        self.wu = load("wu", plan.shard(-1))
        self.bu = load("bu", plan.shard(-1), ttnn.bfloat16)
        self.wd = load("wd", plan.shard(-2))
        self.bd = load("bd", plan.shard(-1), ttnn.bfloat16)
        self.sink = load("sink", plan.shard(-1), ttnn.bfloat16) if self.has_sink else None
        eps = cfg.rms_norm_eps
        self.norm1 = TtRMSNorm(plan, sd and sd[p + "norm1.weight"], eps, name("norm1"))
        self.norm2 = TtRMSNorm(plan, sd and sd[p + "norm2.weight"], eps, name("norm2"))

    @staticmethod
    def _first_chip_bias(bias, tp):
        """[1, tp * n]: the bias in chip 0's slice, zeros elsewhere, added once by the all-reduce."""
        return torch.cat([bias.reshape(1, -1), torch.zeros(1, bias.numel() * (tp - 1))], dim=1)

    def linear(self, x, w, b=None, dtype=ttnn.bfloat16):
        return ttnn.linear(
            x, w, bias=b, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.mm_config
        )

    def __call__(self, x, inp):
        res = self.numerics.residual_dtype
        h = norm_bf16(self.norm1, x)
        attn = self.attention(h, inp)
        h.deallocate(True)
        x2 = ttnn.add(x, attn, dtype=res, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        h = norm_bf16(self.norm2, x2)
        gate = self.linear(h, self.wg, self.bg)
        up = self.linear(h, self.wu, self.bu)
        h.deallocate(True)
        act = ttnn.mul(
            gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate.deallocate(True)
        up.deallocate(True)
        down = self.linear(act, self.wd, self.bd, dtype=self.numerics.reduce_dtype)
        act.deallocate(True)
        down = self.plan.all_reduce(down)
        out = ttnn.add(x2, down, dtype=res, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x2.deallocate(True)
        down.deallocate(True)
        return out

    def attention(self, h, inp):
        qkv = self.linear(h, self.wqkv, self.bqkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.local_heads,
            num_kv_heads=self.local_kv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv.deallocate(True)
        cos, sin = inp.cos[self.col], inp.sin[self.col]
        q_rot = ttnn.experimental.rotary_embedding_hf(q, cos, sin, compute_kernel_config=self.attn_config)
        k_rot = ttnn.experimental.rotary_embedding_hf(k, cos, sin, compute_kernel_config=self.attn_config)
        q.deallocate(True)
        k.deallocate(True)
        if self.full:
            o = ttnn.transformer.scaled_dot_product_attention(
                q_rot,
                k_rot,
                v,
                is_causal=False,
                scale=1.0,
                cu_window_seqlens=inp.cu,
                program_config=self._sdpa_program(inp.geo.rows),
                compute_kernel_config=self.attn_config,
            )
        else:
            o = self.band_attention(q_rot, k_rot, v, inp)
        for t in (q_rot, k_rot, v):
            t.deallocate(True)
        cat = ttnn.experimental.nlp_concat_heads(o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o.deallocate(True)
        out = self.linear(cat, self.wo, self.bo, dtype=self.numerics.reduce_dtype)
        cat.deallocate(True)
        return self.plan.all_reduce(out)

    def _sdpa_program(self, rows):
        chunk = 256 if rows >= 2048 else 128
        while rows % chunk:
            chunk //= 2
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.plan.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )

    def _windows(self, t, nb, zeros):
        """[1, 1, rows, d] -> [nb, 1, BLOCK + 2 * window, d]: block b's keys, rows b*BLOCK - window .. +span."""
        window = self.cfg.window_size
        assert window % BLOCK == 0, window
        halo = window // BLOCK
        d = t.shape[-1]
        padded = ttnn.concat([zeros, t, zeros], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        blocks = ttnn.reshape(padded, (nb + 2 * halo, 1, BLOCK, d))
        parts = [
            ttnn.slice(blocks, (s, 0, 0, 0), (s + nb, 1, BLOCK, d), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for s in range(2 * halo + 1)
        ]
        out = ttnn.concat(parts, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for part in parts:
            part.deallocate(True)
        padded.deallocate(True)
        return out

    def band_attention(self, q, k, v, inp):
        rows = q.shape[-2]
        nb = rows // BLOCK
        heads, hd = self.local_heads, self.cfg.head_dim
        qb = ttnn.reshape(q, (heads, nb, BLOCK, hd))
        qb_t = ttnn.permute(qb, (1, 0, 2, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        qb = ttnn.reshape(qb_t, (nb, 1, heads * BLOCK, hd))
        kw = self._windows(k, nb, inp.halo)
        vw = self._windows(v, nb, inp.halo)
        scores = ttnn.matmul(
            qb,
            kw,
            transpose_b=True,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attn_config,
        )
        qb_t.deallocate(True)
        kw.deallocate(True)
        if self.sink is not None:
            sink = ttnn.mul(inp.sink, self.sink, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            mask = ttnn.add(inp.mask, sink, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sink.deallocate(True)
        else:
            mask = inp.mask
        masked = ttnn.add(scores, mask, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        scores.deallocate(True)
        probs = ttnn.softmax(
            masked,
            dim=-1,
            numeric_stable=True,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attn_config,
        )
        masked.deallocate(True)
        if mask is not inp.mask:
            mask.deallocate(True)
        probs_bf16 = ttnn.typecast(probs, ttnn.bfloat16)
        probs.deallocate(True)
        ob = ttnn.matmul(probs_bf16, vw, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.attn_config)
        probs_bf16.deallocate(True)
        vw.deallocate(True)
        ob = ttnn.reshape(ob, (nb, heads, BLOCK, hd))
        o = ttnn.permute(ob, (1, 0, 2, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ob.deallocate(True)
        return ttnn.reshape(o, (1, heads, rows, hd))


class TtMiMoVisionTransformer:
    """The ViT on one 1x8 mesh (or submesh). ``forward(pixel_values, grid_thw)`` -> host bf16 [N / 4, 4096].

    ``state_dict``: ``reference.vision.load_vision_state_dict`` output, or None to load everything from a warm
    ``cache_dir``. ``weight_dtype``: matmul weights (norms, biases and sinks stay bf16)."""

    def __init__(
        self,
        mesh_device,
        cfg,
        state_dict=None,
        cache_dir=None,
        weight_dtype=ttnn.bfloat16,
        max_rows=DEFAULT_MAX_ROWS,
        num_links=2,
        numerics=NUMERICS["accurate"],
    ):
        assert tuple(mesh_device.shape)[0] == 1, f"one row (1 x TP), got {tuple(mesh_device.shape)}"
        self.plan = MeshPlan(mesh_device, num_links=num_links)
        self.cfg = cfg
        self.max_rows = max_rows
        tp = self.plan.tp
        sd = state_dict
        self.numerics = numerics
        self.mm_config = compute_config(numerics.matmul_fidelity)
        self.attn_config = compute_config(ttnn.MathFidelity.HiFi4, fp32_acc=True, packer_l1_acc=False)
        name = lambda n: cache_name(cache_dir, f"vit_tp{tp}", n)
        w = {}
        if sd is not None:
            w["patch"] = sd["patch_embed.proj.weight"].reshape(cfg.hidden_size, -1).T.contiguous()
            w["m0"] = sd["merger.mlp.0.weight"].T.contiguous()
            w["m2"] = sd["merger.mlp.2.weight"].T.contiguous()
        load = lambda key, mapper: as_weight(w.get(key), self.plan, mapper, weight_dtype, name(key))
        self.patch_w = load("patch", self.plan.replicate())
        self.merger_w0 = load("m0", self.plan.shard(-1))
        self.merger_w2 = load("m2", self.plan.shard(-2))
        self.merger_norm = TtRMSNorm(self.plan, sd and sd["merger.ln_q.weight"], 1e-6, name("ln_q"))
        self.blocks = [
            TtVisionBlock(self.plan, cfg, i, sd, cache_dir, weight_dtype, numerics, self.mm_config, self.attn_config)
            for i in range(cfg.depth)
        ]

    @classmethod
    def from_checkpoint(cls, mesh_device, model_path, cache_dir=None, **kwargs):
        """Converts from the checkpoint once per ``cache_dir`` (``cache_fill`` lock, as the LLM's weights), then
        loads the cached tensors."""
        cfg = ref.VisionConfig.from_model_path(model_path)
        tp = tuple(mesh_device.shape)[1]
        with cache_fill(cache_dir, f"vit_tp{tp}") as need:
            sd = ref.load_vision_state_dict(model_path) if need else None
            return cls(mesh_device, cfg, sd, cache_dir, **kwargs)

    # -- one pass ------------------------------------------------------------------------------------------------

    @staticmethod
    def _gather_rows(x, index):
        """bf16 [1, 1, rows, d] -> its rows in ``index`` order (an embedding lookup into the tensor itself)."""
        rows, d = x.shape[-2], x.shape[-1]
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        table = ttnn.reshape(rm, (rows, d))
        out = ttnn.embedding(index, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rm.deallocate(True)
        return ttnn.reshape(out, (1, 1, rows, d))

    def _reorder(self, x, index):
        if x.dtype == ttnn.bfloat16:
            return self._gather_rows(x, index)
        # The lookup takes bf16 tables only: move an fp32 stream as bf16 high and low parts (16 mantissa bits).
        hi = ttnn.typecast(x, ttnn.bfloat16)
        diff = ttnn.subtract(x, hi, dtype=x.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        lo = ttnn.typecast(diff, ttnn.bfloat16)
        diff.deallocate(True)
        parts = [self._gather_rows(t, index) for t in (hi, lo)]
        hi.deallocate(True)
        lo.deallocate(True)
        out = ttnn.add(parts[0], parts[1], dtype=x.dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in parts:
            t.deallocate(True)
        return out

    def _merger(self, x):
        cfg = self.cfg
        rows = x.shape[-2]
        h = norm_bf16(self.merger_norm, x)
        rm = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h.deallocate(True)
        merged_rm = ttnn.reshape(rm, (1, 1, rows // cfg.merge_unit, cfg.merged_dim))
        merged = ttnn.to_layout(merged_rm, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rm.deallocate(True)
        kw = dict(dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.mm_config)
        h0 = ttnn.linear(merged, self.merger_w0, **kw)
        merged.deallocate(True)
        g = ttnn.gelu(h0, fast_and_approximate_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h0.deallocate(True)
        out = ttnn.linear(g, self.merger_w2, **{**kw, "dtype": self.numerics.reduce_dtype})
        g.deallocate(True)
        return self.plan.all_reduce(out)

    def _host_rows(self, x, geo, to_row=None):
        """Chip 0's [rows, d] -> torch [n, d] in row (processor) order."""
        t = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).reshape(geo.rows, -1)
        if to_row is not None:
            t = t[to_row]
        return t[: geo.n]

    def forward_pass(self, pixel_values, geo, hidden=None):
        """One pass: pixel rows [geo.n, patch_dim] -> host bf16 [geo.n / 4, out_hidden]. ``hidden``: a list that
        collects each block's output (host fp32 [n, hidden], row order) for tests."""
        cfg = self.cfg
        inp = PassInputs(self.plan, cfg, geo)
        pv = pixel_values.to(torch.bfloat16)
        if geo.rows > geo.n:
            pv = torch.cat([pv, torch.zeros(geo.rows - geo.n, pv.shape[-1], dtype=pv.dtype)])
        x_in = ttnn.from_torch(
            pv.reshape(1, 1, geo.rows, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.plan.mesh_device,
            mesh_mapper=self.plan.replicate(),
        )
        x = ttnn.linear(
            x_in,
            self.patch_w,
            dtype=self.numerics.residual_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_config,
        )
        x_in.deallocate(True)
        to_row_host = None
        if hidden is not None:
            to_col = ref.patch_row_index(ref.col_window_index(geo.grids, cfg.spatial_merge_size), cfg.merge_unit)
            to_row_host = torch.argsort(to_col)
        col = False
        for i, blk in enumerate(self.blocks):
            enter, leave = cfg.order_switches(i)
            if enter or leave:
                x_new = self._reorder(x, inp.order[enter])
                x.deallocate(True)
                x = x_new
                col = enter
            x_new = blk(x, inp)
            x.deallocate(True)
            x = x_new
            if hidden is not None:
                hidden.append(self._host_rows(x, geo, to_row_host if col else None).float())
        if col:
            x_new = self._reorder(x, inp.order[False])
            x.deallocate(True)
            x = x_new
        out = self._merger(x)
        x.deallocate(True)
        inp.deallocate()
        result = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(geo.rows // cfg.merge_unit, -1)
        out.deallocate(True)
        return result[: geo.n // cfg.merge_unit].to(torch.bfloat16)

    def forward(self, pixel_values, grid_thw, hidden=None):
        """pixel_values [N, patch_dim] (processor layout, any float dtype), grid_thw [items, 3] -> host bf16
        [N / 4, out_hidden_size], items in order."""
        outs = []
        for start, end, grids in plan_passes(grid_thw, self.max_rows):
            outs.append(self.forward_pass(pixel_values[start:end], PassGeometry.build(grids), hidden))
        return torch.cat(outs, dim=0)

    __call__ = forward
