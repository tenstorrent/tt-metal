# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Attention of MiMo on the [8, 4] mesh: SP over axis 0, heads over TP axis 1. Full layers (0, 5, ..): causal, no
sink, 1 kv head per chip. Sliding layers (1-4, ..): causal window 128 with a per-head sink, 2 kv heads per chip.

Per chip, one chunk of 5120 tokens:
    x [1, 1, 160, 4096] local rows -> all_gather over TP -> [640, 4096] (the SP row's tokens)
    -> fused qkv linear with the TP rank's own checkpoint slab -> [640, 16 x 192 | 192 | 192]
       (v is zero-padded 128 -> 192 so nlp_create_qkv_heads splits one projection; v is sliced back to 128 after;
       full layers: fp32 out, QKV_FP32; sliding layers: fp32 out, SWA_QKV_FP32, RoPE composed in fp32)
    -> RoPE (HF rotate_half) on dims 0-63 of q and k, cos / sin rows looked up at each local row's absolute position
    -> q x 192^-0.5 and v x attention_value_scale (fp32 tensor operands; FULL_SCALES_AFTER_PROJ off: folded into the
       q / v weight rows), q to bf16 for the SDPA
    -> k, v written into the served caches (update_padded_kv_cache over SP, pad rows zeroed)
    -> ring_joint_sdpa causal over SP, reading the whole prefix from the caches (V head dim 128)
       sliding layers: the 192-wide zero-padded v is also written to an unserved attention-only cache (the op's
       sliding path needs V head dim == QK head dim), ring_joint_sdpa runs with sliding_window_size and
       attention_sink over the compact predecessor halo, and the output is sliced back to 128
    -> concat heads [640, 2048] -> o_proj row-parallel (this rank's 2048 input columns) -> reduce_scatter over TP
    -> [1, 1, 160, 4096]

The chunk's slot, prefix length and end live in device scalars, and the positions in a device tensor; ``stage``
(the runtime's per-chunk metadata step, outside the forward) refreshes them, so the forward has no host work.
"""

from __future__ import annotations

import torch

import ttnn

from . import ccl
from .kv_cache import LayerKV, chunk_positions
from .settings import settings

SWA_K_CHUNK = 128  # the ring SDPA's chunked sliding path accepts k chunk 128 only
SWA_HALO = 128  # predecessor halo rows per slot: the window - 1 keys rounded up to whole k chunks


def _compute_config(mesh, fidelity: str, fp32: bool, packer_l1_acc: bool = False):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, fidelity),
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=packer_l1_acc,
    )


def _scalar(mesh, value: int = 0):
    return ttnn.from_torch(
        torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


class ChunkMeta:
    """Device-resident per-chunk metadata shared by the layers: slot, prefix length (kv_actual), end of the real
    tokens, and every local row's absolute position ([1, W] uint32 per SP row, replicated over TP)."""

    def __init__(self, mesh, chunk: int, window: int | None = None):
        self.mesh, self.chunk, self.window = mesh, chunk, window
        self.sp = tuple(mesh.shape)[0]
        self.slot = _scalar(mesh)
        self.kv_actual = _scalar(mesh)
        self.valid_end = _scalar(mesh, chunk)
        self.positions = ttnn.from_torch(
            self._host_positions(0),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, None)),
        )
        # [1, 1, W, 1] bf16 per chip: 1 on the rows before valid_end, 0 on pad rows (multi-head caches zero their
        # pad rows through it, zero_padded_kv_cache takes one head only)
        self.valid_rows = ttnn.from_torch(
            self._host_valid(0, chunk),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, None)),
        )

        # sliding layers: [1, 1, W, 2 * HALO + W] fp32 additive window mask (0 / -1e30) per chip over the keys
        # [halo slot 0 | halo slot 1 | the chip's own rows] (see SlidingAttention below)
        self.swa_mask = None
        if window:
            self.swa_mask = ttnn.from_torch(
                self._host_swa_mask(0, chunk),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, None)),
            )
            sel, cached = self._host_halo_select(0, chunk)
            self.halo_select, self.halo_cached = (
                ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(0, None)),
                )
                for t in (sel, cached)
            )

    def _halo_positions(self, p: torch.Tensor) -> torch.Tensor:
        """[2 * HALO] key positions of the two halo slots of one chip's rows p (unused slot: -1e9)."""
        breaks = [0] + [i for i in range(1, len(p)) if p[i] != p[i - 1] + 1]
        halo = torch.full((2, SWA_HALO), -(10**9), dtype=torch.int64)
        for slot, b in enumerate(breaks[:2]):
            halo[slot] = p[b] - SWA_HALO + torch.arange(SWA_HALO)
        return halo.flatten()

    def _host_halo_select(self, start: int, end: int):
        """Halo rows of this chunk's own tokens come from its bf16 k / v, not from the cache (the device's bf16 ->
        bfloat8_b typecast on the cache write is biased, +0.3 %, which the sink-dominated softmax turns into a -5 %
        norm error on each slab's first rows). sel [sp, 1, 2 HALO, chunk]: one-hot from a halo row to its row of the
        SP all-gathered k (SP row s, local row i = column s W + i); cached [sp, 1, 2 HALO, 1]: 1 where the halo row
        is before the chunk (from the cache, as the ring SDPA gathered it)."""
        pos = self._host_positions(start)  # [sp, W]
        where = {int(g): j for j, g in enumerate(pos.flatten().tolist())}
        sel = torch.zeros(self.sp, 1, 2 * SWA_HALO, self.chunk)
        cached = torch.zeros(self.sp, 1, 2 * SWA_HALO, 1)
        for r in range(self.sp):
            for j, g in enumerate(self._halo_positions(pos[r]).tolist()):
                if start <= g < end:
                    sel[r, 0, j, where[g]] = 1.0
                elif 0 <= g < start:
                    cached[r, 0, j, 0] = 1.0
        return sel.to(torch.bfloat16), cached.to(torch.bfloat16)

    def _host_swa_mask(self, start: int, end: int) -> torch.Tensor:
        """Query row i (position p_i) sees key j (position k_j) when p_i - window < k_j <= p_i, k_j >= 0 and k_j <
        end. Key positions: the chip's rows, and the 128 positions before each run of consecutive rows (the ring
        SDPA's halo slots: slot 0 precedes the first run, slot 1 the second when the chunk wraps an SP slab)."""
        pos = self._host_positions(start)  # [sp, W]
        out = []
        for r in range(self.sp):
            p = pos[r]
            keys = torch.cat((self._halo_positions(p), p))
            qi, kj = p[:, None], keys[None, :]
            ok = (kj <= qi) & (kj > qi - self.window) & (kj >= 0) & (kj < end)
            out.append(torch.where(ok, 0.0, -1e30))
        return torch.stack(out)[:, None].float()  # [sp, 1, W, 2 * HALO + W]

    def _host_valid(self, start: int, end: int) -> torch.Tensor:
        return (self._host_positions(start) < end).to(torch.bfloat16)[:, None, :, None]  # [sp, 1, W, 1]

    def _host_positions(self, start: int) -> torch.Tensor:
        return chunk_positions(start, self.sp, self.chunk)  # [sp, W]: SP row r -> its chip's [1, W]

    def stage(self, slot: int, start: int, end: int | None = None) -> None:
        """Host -> device copies of the chunk's metadata into the existing buffers (trace-safe addresses)."""
        end = start + self.chunk if end is None else end
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        for t, v in ((self.slot, slot), (self.kv_actual, start), (self.valid_end, end)):
            h = ttnn.from_torch(
                torch.tensor([v], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(h, t)
        h = ttnn.from_torch(
            self._host_positions(start),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, None)),
        )
        ttnn.copy_host_to_device_tensor(h, self.positions)
        h = ttnn.from_torch(
            self._host_valid(start, end),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, None)),
        )
        ttnn.copy_host_to_device_tensor(h, self.valid_rows)
        if self.swa_mask is not None:
            h = ttnn.from_torch(
                self._host_swa_mask(start, end),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, None)),
            )
            ttnn.copy_host_to_device_tensor(h, self.swa_mask)
            for t, dev in zip(self._host_halo_select(start, end), (self.halo_select, self.halo_cached)):
                h = ttnn.from_torch(
                    t,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, None)),
                )
                ttnn.copy_host_to_device_tensor(h, dev)


class RopeTables:
    """cos / sin [max_seq, rope_dim] bf16 ROW_MAJOR, replicated; HF layout (the half-split frequencies repeated).
    Also the bf16 residuals (fp32 table - its bf16 rounding): ``lookup(fp32=True)`` returns hi + lo in fp32
    (``ttnn.embedding`` takes bf16 tables only)."""

    def __init__(self, mesh, inv_freq: torch.Tensor, max_seq: int):
        pos = torch.arange(max_seq, dtype=torch.float32)
        emb = torch.outer(pos, inv_freq.float())
        emb = torch.cat((emb, emb), dim=-1)
        rep = ttnn.ReplicateTensorToMesh(mesh)
        cos, sin = emb.cos(), emb.sin()
        self.cos, self.sin, self.cos_lo, self.sin_lo = (
            ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            for t in (cos, sin, cos - cos.to(torch.bfloat16).float(), sin - sin.to(torch.bfloat16).float())
        )

    def lookup(self, positions, fp32: bool = False):
        """positions [1, W] uint32 -> cos, sin [1, 1, W, rope_dim] TILE: bf16, or fp32 (hi + lo) with ``fp32``."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        out = []
        for table, lo_table in ((self.cos, self.cos_lo), (self.sin, self.sin_lo)):
            v = ttnn.unsqueeze_to_4D(ttnn.embedding(positions, table, layout=ttnn.TILE_LAYOUT, memory_config=mc))
            if fp32:
                lo = ttnn.unsqueeze_to_4D(
                    ttnn.embedding(positions, lo_table, layout=ttnn.TILE_LAYOUT, memory_config=mc)
                )
                hi32 = ttnn.typecast(v, ttnn.float32, memory_config=mc)
                lo32 = ttnn.typecast(lo, ttnn.float32, memory_config=mc)
                for t in (v, lo):
                    t.deallocate(True)
                v = ttnn.add(hi32, lo32, memory_config=mc)
                hi32.deallocate(True)
                lo32.deallocate(True)
            out.append(v)
        return out


def layer_dims(cfg: dict, layer_idx: int) -> dict:
    """Heads, head dims, RoPE theta, window and sink of one layer (the reference Layer's config keys: sliding layers
    read the ``swa_`` variants)."""
    swa = cfg["hybrid_layer_pattern"][layer_idx] == 1
    sfx = "swa_" if swa else ""
    return {
        "heads": cfg.get(f"{sfx}num_attention_heads", cfg["num_attention_heads"]),
        "kv_heads": cfg.get(f"{sfx}num_key_value_heads", cfg["num_key_value_heads"]),
        "head_dim": cfg.get(f"{sfx}head_dim", cfg["head_dim"]),
        "v_head_dim": cfg.get(f"{sfx}v_head_dim", cfg["v_head_dim"]),
        "rope_theta": cfg["swa_rope_theta"] if swa else cfg["rope_theta"],
        "window": cfg["sliding_window"] if swa else None,
        "sink": bool(cfg.get("add_swa_attention_sink_bias" if swa else "add_full_attention_sink_bias", False)),
    }


def rank_weights(ck, layer_idx: int, cfg: dict, tp: int, fold_scales: bool = True, bf16_first: str = "none"):
    """fp32 per-TP-rank weights in the checkpoint's [out, in] orientation (the linears run with transpose_b, so the
    bfloat8_b blocks of 16 run along K as in the precision model; [in, out] blocks along the output and costs single
    output columns up to 9x their error): qkv [tp, lh D + lkv D + lkv D, H] (q rows x D^-0.5, v rows x
    attention_value_scale and zero-padded 128 -> D so nlp_create_qkv_heads splits it) and o_proj [tp, H, lh Dv]
    (rank r's input columns). The checkpoint's own rank slabs: rank r holds q heads r*lh.., kv heads r*lkv..
    bf16_first ("none", "qkv", "o", "all"): those dequantized weights are rounded to bf16 before the scales are folded
    in (as the reference holds them), so the later bfloat8_b rounding starts from the precision model's values."""
    dims = layer_dims(cfg, layer_idx)
    heads, kv_heads, D, Dv = dims["heads"], dims["kv_heads"], dims["head_dim"], dims["v_head_dim"]
    lh, lkv = heads // tp, kv_heads // tp
    p = f"model.layers.{layer_idx}.self_attn"
    wq, wk, wv = ck.qkv(f"{p}.qkv_proj", heads * D, kv_heads * D, kv_heads * Dv)
    wq, wk, wv = wq.float(), wk.float(), wv.float()
    if bf16_first in ("qkv", "all"):
        wq, wk, wv = (w.to(torch.bfloat16).float() for w in (wq, wk, wv))
    if fold_scales:  # else the module applies them after the projection (sliding layers)
        wq = wq * D**-0.5
        wv = wv * (cfg.get("attention_value_scale") or 1.0)
    slabs = []
    for r in range(tp):
        q_r = wq[r * lh * D : (r + 1) * lh * D]
        k_r = wk[r * lkv * D : (r + 1) * lkv * D]
        v_r = wv[r * lkv * Dv : (r + 1) * lkv * Dv].view(lkv, Dv, -1)
        v_r = torch.nn.functional.pad(v_r, (0, 0, 0, D - Dv)).reshape(lkv * D, -1)
        slabs.append(torch.cat((q_r, k_r, v_r)).contiguous())
    wo = ck.get(f"{p}.o_proj.weight").float()  # [H, heads * Dv]
    if bf16_first in ("o", "all"):
        wo = wo.to(torch.bfloat16).float()
    return torch.stack(slabs), wo.view(-1, tp, lh * Dv).permute(1, 0, 2).contiguous()


class Attention:
    """One attention layer, full (MiMo layers 0, 5, ...) or sliding (1-4, ...), from ``layer_dims``; weights come
    from the checkpoint in fp32 and are rounded once to bfloat8_b on the device."""

    def __init__(
        self,
        mesh,
        ck,
        layer_idx: int,
        cfg: dict,
        *,
        max_seq: int,
        chunk: int,
        slots: int,
        ring: ccl.RingResources,
        meta: ChunkMeta,
        rope: RopeTables,
        kv: LayerKV | None = None,
    ):
        self.mesh, self.ring, self.meta, self.rope = mesh, ring, meta, rope
        self.sp, self.tp = tuple(mesh.shape)
        dims = layer_dims(cfg, layer_idx)
        self.heads, self.kv_heads = dims["heads"], dims["kv_heads"]
        self.head_dim, self.v_head_dim = dims["head_dim"], dims["v_head_dim"]
        self.window = dims["window"]
        self.rope_dim = int(self.head_dim * cfg["partial_rotary_factor"])
        assert self.heads % self.tp == 0 and self.kv_heads % self.tp == 0
        self.lh, self.lkv = self.heads // self.tp, self.kv_heads // self.tp
        self.max_seq, self.chunk = max_seq, chunk
        self.local_seq = max_seq // self.sp

        # the softmax scale is folded into the q rows in fp32 and SDPA runs at scale 1 (the kernel may truncate a
        # non-power-of-two scale to bf16: 192^-0.5 -0.54 %, known issue "SDPA attention sink ... truncated scale").
        # With scale 1 the kernel's sink term exp((sink - max) * scale) takes the checkpoint's sink logit as is.
        self.scale = 1.0
        # sliding layers keep the softmax and value scales out of the bfloat8_b weights and apply them after the
        # projection, as the reference does: folded in, the rounding moved single o_proj output columns past their
        # limit (layer 1 column 2331 0.19 vs 0.16)
        self.fold_scales = not self.window
        # full layers with fp32 q / v (QKV_FP32, SCALES_AFTER_PROJ): the scales are applied to the fp32 projection
        # (fp32 tensor operands) instead of being folded into the bfloat8_b weights, as the precision model rounds
        # the weights before scaling; the SDPA still runs at scale 1 on the scaled bf16 q
        self.scale_after = not self.window and settings.get("QKV_FP32") and settings.get("FULL_SCALES_AFTER_PROJ")
        if self.scale_after:
            self.fold_scales = False
        self.v_scale = cfg.get("attention_value_scale") or 1.0
        self.true_scale = self.head_dim**-0.5
        wqkv, wo = rank_weights(
            ck,
            layer_idx,
            cfg,
            self.tp,
            fold_scales=self.fold_scales,
            bf16_first="none" if self.window else settings.get("FULL_WEIGHTS_BF16_FIRST"),
        )
        if self.window and settings.get("SWA_WEIGHTS_BF16_FIRST"):
            # fp32 -> bf16 -> bfloat8_b, as the precision model rounds them (fp32 -> bfloat8_b directly picks other
            # shared exponents and doubled layer 1's K error vs the model: output column 529 0.13 vs 0.04)
            wqkv, wo = wqkv.to(torch.bfloat16).float(), wo.to(torch.bfloat16).float()
        self.wqkv = self._sharded_weight(wqkv.unsqueeze(0))  # [1, tp, N, H] -> per chip [1, 1, N, H]
        self.wo = self._sharded_weight(wo.unsqueeze(0))
        self.sink = self.sink32 = None
        if dims["sink"]:  # [heads] -> per chip [1, lh, 1, 1] bf16 TILE: this rank's q heads
            sink = ck.get(f"model.layers.{layer_idx}.self_attn.attention_sink_bias").float()
            self.sink = ttnn.from_torch(
                sink.to(torch.bfloat16).view(1, self.heads, 1, 1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 1)),
            )
            self.sink32 = ttnn.from_torch(
                sink.view(1, self.heads, 1, 1),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 1)),
            )
        self.swa_cfg = _compute_config(mesh, "HiFi4", True)
        # the softmax scale as an fp32 tensor operand: a scalar operand is rounded to bf16 (192^-0.5 +0.13 %, which the
        # sink-dominated softmax turned into a -1.4 % output bias)
        self.vscale32 = ttnn.from_torch(
            torch.full((1, 1, 1, 1), self.v_scale),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.scale32 = ttnn.from_torch(
            torch.full((1, 1, 1, 1), self.head_dim**-0.5),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        # the served caches: the engine's (allocate_kv_cache, tt/runners/kv_caches.py) when given, else this layer's
        self.kv = kv or LayerKV(
            mesh,
            slots,
            self.lkv,
            self.local_seq,
            self.head_dim,
            self.v_head_dim,
            attn_v_dim=self.head_dim if self.window else None,
        )
        self.mm_cfg = _compute_config(
            mesh, settings.get("ATTN_FIDELITY"), settings.get("ATTN_FP32_ACC"), packer_l1_acc=True
        )
        self.rope_cfg = _compute_config(mesh, "HiFi4", True)
        # fp32 dest off: the kv_actual_isl_tensor path runs only the streaming SDPA compute (TT_FATAL otherwise)
        self.sdpa_cfg = _compute_config(mesh, settings.get("SDPA_FIDELITY"), False)
        if self.window:
            # the op's chunked sliding path takes q chunks 64 / 128 and k chunk 128 only (the halo granularity)
            self.sdpa_program = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ring.sdpa_grid,
                q_chunk_size=settings.get("SWA_SDPA_Q_CHUNK"),
                k_chunk_size=SWA_K_CHUNK,
                exp_approx_mode=False,
            )
        else:
            self.sdpa_program = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=ring.sdpa_grid,
                q_chunk_size=settings.get("SDPA_Q_CHUNK"),
                k_chunk_size=settings.get("SDPA_K_CHUNK"),
                exp_approx_mode=False,
                segmented_accumulation=settings.get("SDPA_SEGMENTED"),
            )

    def _sharded_weight(self, w: torch.Tensor):
        return ttnn.from_torch(
            w,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(None, 1)),
        )

    def load_prefix(self, slot: int, key: torch.Tensor, value: torch.Tensor, length: int) -> None:
        """Harness boundary (never inside a forward): the global prefix key [KV, >= length, D] / value [KV, ..,
        Dv] in the served block-cyclic layout of ``slot``; every other row of the slot is zero."""
        w = self.chunk // self.sp
        g = torch.arange(length)
        row, local = (g // w) % self.sp, (g // self.chunk) * w + g % w
        srcs = [("key", key), ("value", value)]
        if self.kv.attn_value is not None:
            srcs.append(("attn_value", torch.nn.functional.pad(value.float(), (0, self.head_dim - value.shape[-1]))))
        for name, src in srcs:
            cache = getattr(self.kv, name)
            width = src.shape[-1]
            host = torch.zeros(self.sp, self.tp, self.kv.slots, self.lkv, self.local_seq, width)
            heads = src[:, :length].float().view(self.tp, self.lkv, length, width)
            for r in range(self.sp):
                m = row == r
                host[r][:, slot][:, :, local[m]] = heads[:, :, m]
            host = host.permute(0, 2, 1, 3, 4, 5).reshape(
                self.sp * self.kv.slots, self.tp * self.lkv, self.local_seq, width
            )
            new = ttnn.from_torch(
                host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=cache.memory_config(),
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=tuple(self.mesh.shape), dims=(0, 1)),
            )
            new.update_tensor_topology(cache.tensor_topology())
            cache.deallocate(True)
            setattr(self.kv, name, new)

    def __call__(self, x):
        """x [1, 1, 160, H] bf16 (this chip's residual rows) -> [1, 1, 160, H] bf16."""
        D, Dv, R = self.head_dim, self.v_head_dim, self.rope_dim
        xg = ccl.gather_rows(x)
        # full layers: q / k / v stay fp32 through RoPE, so the cache writes round fp32 values to bfloat8_b (the
        # device's bf16 -> bfloat8_b typecast rounds the many exact ties away from zero: +0.28 % on k / v norms)
        # sliding layers (SWA_QKV_FP32): the same, and the composed window softmax reads the fp32 q / k / v
        qkv_fp32 = settings.get("SWA_QKV_FP32" if self.window else "QKV_FP32")
        qkv = ttnn.linear(
            xg,
            self.wqkv,
            transpose_b=True,
            dtype=ttnn.float32 if qkv_fp32 else ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_cfg,
        )
        xg.deallocate(True)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.lh,
            num_kv_heads=self.lkv,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv.deallocate(True)

        rope32 = qkv_fp32 and bool(self.window) and settings.get("SWA_ROPE_FP32")
        cos, sin = self.rope.lookup(self.meta.positions, fp32=rope32)
        rope = self._rope32 if rope32 else self._rope
        q = rope(q, cos, sin, R)
        k = rope(k, cos, sin, R)
        cos.deallocate(True)
        sin.deallocate(True)
        if self.scale_after:
            q = self._scale(q, self.scale32)
            v = self._scale(v, self.vscale32)
        q_soft = None  # sliding layers: the composed softmax's q (fp32 with SWA_QKV_FP32)
        if qkv_fp32:  # the SDPA reads bf16 q
            q32 = q
            q = ttnn.typecast(q32, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if self.window:
                q_soft = q32
            else:
                q32.deallocate(True)
        if not self.fold_scales and not self.scale_after:
            if qkv_fp32:
                v = self._scale(v, self.vscale32)
            else:
                v = ttnn.multiply(v, self.v_scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.lkv > 1:  # pad rows zeroed before the write (zero_padded_kv_cache below takes single-head caches only)
            k = self._zero_pad_rows(k)
            v = self._zero_pad_rows(v)
        v_full = v
        v = ttnn.slice(v_full, (0, 0, 0, 0), (1, self.lkv, v_full.shape[2], Dv), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        writes = [(self.kv.key, k), (self.kv.value, v)]
        if self.kv.attn_value is not None:  # sliding: the zero-padded 192-wide v for the SDPA (its pad columns are 0)
            writes.append((self.kv.attn_value, v_full))
        else:
            v_full.deallocate(True)

        for cache, t in writes:
            t8 = ttnn.typecast(t, ttnn.bfloat8_b)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                t8,
                self.meta.slot,
                self.meta.kv_actual,
                layer_idx=0,
                num_layers=1,
                cluster_axis=ccl.SP_AXIS,
            )
            if self.lkv == 1:
                ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
                    cache,
                    self.meta.slot,
                    self.meta.valid_end,
                    layer_idx=0,
                    num_layers=1,
                    chunk_size_global=self.chunk,
                    cluster_axis=ccl.SP_AXIS,
                    pad_align=ttnn.TILE_SIZE,
                )
            t8.deallocate(True)
            if not self.window:
                t.deallocate(True)

        if self.window:
            out = self._sliding_sdpa(q, k, v, q_soft)
            for t in (k, v, v_full) + ((q_soft,) if q_soft is not None else ()):
                t.deallocate(True)
        else:
            out = self._full_sdpa(q)
        q.deallocate(True)
        heads = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out.deallocate(True)
        fp32_sum = settings.get("ATTN_OUT_FP32")
        partial = ttnn.linear(
            heads,
            self.wo,
            transpose_b=True,
            dtype=ttnn.float32 if fp32_sum else ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.mm_cfg,
        )
        heads.deallocate(True)
        y = ccl.scatter_rows(partial)
        partial.deallocate(True)
        if fp32_sum:  # the TP partials summed in fp32, rounded to bf16 once
            y32 = y
            y = ttnn.typecast(y32, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            y32.deallocate(True)
        return y

    def _scale(self, t, factor):
        out = ttnn.multiply(t, factor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t.deallocate(True)
        return out

    def _zero_pad_rows(self, t):
        """t [1, heads, W, d] x the chunk's row-validity mask: rows at or past valid_end become 0 (exact for the
        rest), so the cache's pad rows are zero once written, as zero_padded_kv_cache leaves them."""
        out = ttnn.multiply(t, self.meta.valid_rows, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t.deallocate(True)
        return out

    def _sdpa_op(self):
        # the fork adds exact_exp (accurate softmax exp; the source op's streaming kernel hard-codes the
        # Schraudolph approximation, ~1.3 % rel error on layer 0's attention output)
        if settings.get("SDPA_EXACT_EXP"):
            return ttnn.bringup.ring_joint_scaled_dot_product_attention, {"exact_exp": True}
        return ttnn.transformer.ring_joint_scaled_dot_product_attention, {}

    def _ring_args(self):
        return dict(
            joint_strategy="rear",
            program_config=self.sdpa_program,
            scale=self.scale,
            compute_kernel_config=self.sdpa_cfg,
            dim=2,
            multi_device_global_semaphore=self.ring.semaphores,
            num_links=self.ring.num_links,
            cluster_axis=ccl.SP_AXIS,
            mesh_device=self.mesh,
            topology=ttnn.Topology.Linear,
            ccl_core_grid_offset=self.ring.ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            is_balanced=False,
            slot_id=self.meta.slot,
            kv_actual_isl_tensor=self.meta.kv_actual,
            kv_cache_num_layers=1,
            kv_cache_layer_idx=0,
        )

    def _sliding_sdpa(self, q, k, v, q_soft=None):
        """Window + sink attention of this chip's rows. The ring SDPA (window, sink) brings the predecessor halo of
        K / V into its compact gather buffers (two slots: a chunk that wraps an SP slab needs both); the softmax
        itself is then composed in fp32 over [halo slot 0 | halo slot 1 | own rows] (``_window_softmax``): the
        sink takes nearly all the softmax mass on these layers (scores ~ -24 vs sink ~ 0.8), so the output scales
        as exp(score - sink) and the op's bf16-dest scores put a 6 % bias on its own output."""
        D, Dv = self.head_dim, self.v_head_dim
        halo = SWA_HALO
        buf_k = self.ring.gather_buffer("swa_k", self.lkv, 2 * halo, D, ttnn.bfloat8_b)
        buf_v = self.ring.gather_buffer("swa_v", self.lkv, 2 * halo, D, ttnn.bfloat8_b)
        sdpa, extra = self._sdpa_op()
        out, _, _ = sdpa(
            q,
            self.kv.key,
            self.kv.attn_value,
            None,
            None,
            None,
            persistent_output_buffer_k=buf_k,
            persistent_output_buffer_v=buf_v,
            logical_n=self.sp * self.local_seq,
            attention_sink=self.sink,
            sliding_window_size=self.window,
            **{**self._ring_args(), "scale": self.scale if self.fold_scales else self.true_scale},
            **extra,
        )
        out.deallocate(True)
        mc = ttnn.DRAM_MEMORY_CONFIG
        hv8 = ttnn.slice(buf_v, (0, 0, 0, 0), (1, self.lkv, 2 * halo, Dv), memory_config=mc)
        hk = self._halo(buf_k, k, k.dtype)
        hv = self._halo(hv8, v, v.dtype)
        hv8.deallocate(True)
        k_ext = ttnn.concat((hk, k), dim=2, memory_config=mc)  # [1, lkv, 2 * halo + W, D]
        v_ext = ttnn.concat((hv, v), dim=2, memory_config=mc)
        hk.deallocate(True)
        hv.deallocate(True)
        o = self._window_softmax(q if q_soft is None else q_soft, k_ext, v_ext)
        k_ext.deallocate(True)
        v_ext.deallocate(True)
        return o

    def _halo(self, ring_buf, own, dtype=ttnn.bfloat16):
        """[1, lkv, 2 HALO, d] in ``dtype`` (``own``'s): rows before the chunk from the ring SDPA's gathered cache
        rows, rows inside it from the SP all-gather of this chunk's ``own`` rows (one-hot select, exact), pad /
        unused rows 0."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        cache = ttnn.typecast(ring_buf, dtype, memory_config=mc)
        from_cache = ttnn.multiply(cache, self.meta.halo_cached, memory_config=mc)
        cache.deallocate(True)
        gathered = ttnn.all_gather(own, dim=2, cluster_axis=ccl.SP_AXIS, memory_config=mc)  # [1, lkv, chunk, d]
        if dtype == ttnn.float32:  # exact one-hot select of fp32 rows (split into bf16 hi + lo)
            inside = self._mm32(self.meta.halo_select, gathered)
        else:
            inside = ttnn.matmul(
                self.meta.halo_select, gathered, dtype=dtype, memory_config=mc, compute_kernel_config=self.swa_cfg
            )
        gathered.deallocate(True)
        out = ttnn.add(from_cache, inside, memory_config=mc)
        from_cache.deallocate(True)
        inside.deallocate(True)
        return out

    def _window_softmax(self, q, k_ext, v_ext):
        """fp32 softmax(q k^T + mask, sink) v per kv group: q [1, lh, W, D] bf16 / fp32, k_ext / v_ext
        [1, lkv, N, D | Dv] -> [1, lh, W, Dv]. The softmax scale multiplies the fp32 scores; the sink is one more
        logit per head."""
        mc, cfg = ttnn.DRAM_MEMORY_CONFIG, self.swa_cfg
        W, D, Dv = q.shape[2], self.head_dim, self.v_head_dim
        N = k_ext.shape[2]
        rep = self.lh // self.lkv
        outs = []
        for g in range(self.lkv):
            qg = ttnn.slice(q, (0, g * rep, 0, 0), (1, (g + 1) * rep, W, D), memory_config=mc)
            qg = ttnn.reshape(qg, (1, 1, rep * W, D))
            kg = ttnn.slice(k_ext, (0, g, 0, 0), (1, g + 1, N, D), memory_config=mc)
            vg = ttnn.slice(v_ext, (0, g, 0, 0), (1, g + 1, N, Dv), memory_config=mc)
            sc = self._mm32(qg, kg, transpose_b=True)
            qg.deallocate(True)
            kg.deallocate(True)
            sc = ttnn.reshape(sc, (1, rep, W, N))
            if not self.fold_scales:
                scaled = ttnn.multiply(sc, self.scale32, memory_config=mc)
                sc.deallocate(True)
                sc = scaled
            masked = ttnn.add(sc, self.meta.swa_mask, memory_config=mc)
            sc.deallocate(True)
            sink = ttnn.slice(self.sink32, (0, g * rep, 0, 0), (1, (g + 1) * rep, 1, 1), memory_config=mc)
            m = ttnn.max(masked, dim=-1, keepdim=True, memory_config=mc)
            m2 = ttnn.maximum(m, sink, memory_config=mc)
            m.deallocate(True)
            p = ttnn.subtract(masked, m2, memory_config=mc)
            masked.deallocate(True)
            p = ttnn.exp(p, fast_and_approximate_mode=False, memory_config=mc)
            z = ttnn.sum(p, dim=-1, keepdim=True, memory_config=mc, compute_kernel_config=cfg)
            ds = ttnn.subtract(sink, m2, memory_config=mc)
            es = ttnn.exp(ds, fast_and_approximate_mode=False, memory_config=mc)
            z2 = ttnn.add(z, es, memory_config=mc)
            for t in (ds, es, z, m2, sink):
                t.deallocate(True)
            p = ttnn.reshape(p, (1, 1, rep * W, N))
            o = self._mm32(p, vg)
            p.deallocate(True)
            vg.deallocate(True)
            o = ttnn.reshape(o, (1, rep, W, Dv))
            on = ttnn.divide(o, z2, memory_config=mc)
            o.deallocate(True)
            z2.deallocate(True)
            if settings.get("SWA_OUT_FP32"):
                outs.append(on)
            else:
                outs.append(ttnn.typecast(on, ttnn.bfloat16, memory_config=mc))
                on.deallocate(True)
        out = ttnn.concat(outs, dim=1, memory_config=mc)
        for t in outs:
            t.deallocate(True)
        return out

    def _mm32(self, a, b, transpose_b=False):
        """fp32-out matmul. SWA_SPLIT_MM with an fp32 operand: each fp32 operand is split into bf16 hi + lo (RNE
        typecasts) and a b = ah bh + ah bl + al bh, every product exact in bf16 x bf16 -> fp32 dest; the matmul
        reads fp32 operands at reduced (tf32-like) precision, which moved the sink-dominated output norm."""
        mc, cfg = ttnn.DRAM_MEMORY_CONFIG, self.swa_cfg
        kw = dict(transpose_b=transpose_b, dtype=ttnn.float32, memory_config=mc, compute_kernel_config=cfg)
        if not settings.get("SWA_SPLIT_MM") or (a.dtype != ttnn.float32 and b.dtype != ttnn.float32):
            return ttnn.matmul(a, b, **kw)
        ah, al = self._split(a)
        bh, bl = self._split(b)
        out = ttnn.matmul(ah, bh, **kw)
        for x, y in ((ah, bl), (al, bh)):
            if x is None or y is None:
                continue
            t = ttnn.matmul(x, y, **kw)
            s = ttnn.add(out, t, memory_config=mc)
            out.deallocate(True)
            t.deallocate(True)
            out = s
        for t in (ah, al, bh, bl):
            if t is not None:
                t.deallocate(True)
        return out

    def _split(self, t):
        """fp32 t -> (hi, lo) bf16 with hi + lo = t to ~16 mantissa bits; a bf16 t -> (a copy, None)."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        if t.dtype != ttnn.float32:
            return ttnn.clone(t, memory_config=mc), None
        hi = ttnn.typecast(t, ttnn.bfloat16, memory_config=mc)
        hi32 = ttnn.typecast(hi, ttnn.float32, memory_config=mc)
        r = ttnn.subtract(t, hi32, memory_config=mc)
        hi32.deallocate(True)
        lo = ttnn.typecast(r, ttnn.bfloat16, memory_config=mc)
        r.deallocate(True)
        return hi, lo

    def _full_sdpa(self, q):
        D, Dv = self.head_dim, self.v_head_dim
        gather = self.sp * self.local_seq
        sdpa, extra = self._sdpa_op()
        out, _, _ = sdpa(
            q,
            self.kv.key,
            self.kv.value,
            None,
            None,
            None,
            persistent_output_buffer_k=self.ring.gather_buffer("full_k", self.lkv, gather, D, ttnn.bfloat8_b),
            persistent_output_buffer_v=self.ring.gather_buffer("full_v", self.lkv, gather, Dv, ttnn.bfloat8_b),
            logical_n=gather,
            **self._ring_args(),
            **extra,
        )
        return out

    def _rope32(self, t, cos, sin, R):
        """HF rotate_half RoPE on the first R dims of every head, composed from fp32 eltwise ops: t * cos +
        rotate_half(t) * sin. The fused rotary_embedding reads fp32 operands at reduced precision (truncated), which
        biased |q k| and so the sink-dominated sliding output (+1.9 % median row norm)."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        shape = tuple(t.shape)
        h = R // 2
        x1 = ttnn.slice(t, (0, 0, 0, 0), shape[:-1] + (h,), memory_config=mc)
        x2 = ttnn.slice(t, (0, 0, 0, h), shape[:-1] + (R,), memory_config=mc)
        rest = ttnn.slice(t, (0, 0, 0, R), shape, memory_config=mc)
        rot = ttnn.slice(t, (0, 0, 0, 0), shape[:-1] + (R,), memory_config=mc)
        t.deallocate(True)
        n2 = ttnn.neg(x2, memory_config=mc)
        x2.deallocate(True)
        half = ttnn.concat((n2, x1), dim=-1, memory_config=mc)
        n2.deallocate(True)
        x1.deallocate(True)
        a = ttnn.multiply(rot, cos, memory_config=mc)
        b = ttnn.multiply(half, sin, memory_config=mc)
        rot.deallocate(True)
        half.deallocate(True)
        rotated = ttnn.add(a, b, memory_config=mc)
        a.deallocate(True)
        b.deallocate(True)
        out = ttnn.concat((rotated, rest), dim=-1, memory_config=mc)
        rotated.deallocate(True)
        rest.deallocate(True)
        return out

    def _rope(self, t, cos, sin, R):
        """HF rotate_half RoPE on the first R dims of every head."""
        shape = tuple(t.shape)
        rot = ttnn.slice(t, (0, 0, 0, 0), shape[:-1] + (R,), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rest = ttnn.slice(t, (0, 0, 0, R), shape, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        t.deallocate(True)
        rotated = ttnn.experimental.rotary_embedding(
            rot, cos, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.rope_cfg
        )
        rot.deallocate(True)
        out = ttnn.concat((rotated, rest), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        rotated.deallocate(True)
        rest.deallocate(True)
        return out
