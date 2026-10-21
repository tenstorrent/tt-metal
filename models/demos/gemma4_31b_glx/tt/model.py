# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 31B on the SP8 x TP4 Galaxy, all on device: embedding -> 60 blocks (run_block over the reference's block
graph) -> final norm -> tied, soft-capped LM head. The residual stays on the device from the embedding to the norm.

Residual layout between blocks: [1, 1, C / (SP * TP), 5376] TILE per chip (C = 5120: 160 rows), fp32 by default
(RESIDUAL_STREAM; the norm outputs feeding attention and the MLP are bf16). SP row r holds the 640 tokens whose
served-cache rows it owns, TP column t the t-th 160 of them.

Any tile-aligned chunk start (serving_contract.md, "Attention and cache writes"). For a chunk at start s, SP row r's
input row i is the token at the position its cache write lands on: local row lr = local_offset(s, r) + i of the
block-cyclic cache (update_padded_kv_cache's staircase, writer_update_padded_kv_cache.cpp), position
(lr // 640) * 5120 + r * 640 + lr % 640. That is tt-d-gen's ring_sdpa_reshuffle, and what the ring SDPA's chunked Q
mapping assumes (chunked_q_mapping.hpp). For a chunk-aligned s it is the natural order.

embed(tokens, start) stages the chunk's inputs: the token ids in that order (PAD_ID -> 0), their positions, and the
on-device scalars (slot, start, actual_end = start + the real tokens before PAD_ID). The per-chunk tensors every
layer shares (RoPE values looked up from load-time tables, the pad-row keep mask, the global causal masks) are built
from them on the device, once per chunk (ChunkMeta). layer() then runs no host transfer.
"""

import time
from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4_31b_glx.reference._bringup_interface import Ctx, Step, run_block

from . import weights
from .attention import SP_AXIS, TP_AXIS, SlidingAttention, sliding_kv_indices, sliding_rope_tables
from .global_attention import ROTARY_DIM, GlobalAttention, global_kv_indices, pack_global_kv
from .mlp import MLP
from .residual import ResidualAdd
from .rms_norm import RMSNorm
from .settings import settings

PAD_ID = 0xFFFFFFFF
MASK = -1e9

# The reference's block graph (reference/gemma4_ref.py block_graph; the same for every layer).
BLOCK = [
    Step("attn_norm", ("in",), "attn_norm", "norm"),
    Step("attention", ("attn_norm",), "attn_out", "attention", stateful=True),
    Step("post_attn_norm", ("attn_out",), "attn_post", "norm"),
    Step("attn_residual", ("in", "attn_post"), "h_mid", "residual"),
    Step("ffn_norm", ("h_mid",), "ffn_norm", "norm"),
    Step("mlp", ("ffn_norm",), "mlp_out", "mlp"),
    Step("post_ffn_norm", ("mlp_out",), "mlp_post", "norm"),
    Step("ffn_residual", ("h_mid", "mlp_post"), "out", "residual"),
]
NORM_WEIGHTS = {
    "attn_norm": "input_layernorm",
    "post_attn_norm": "post_attention_layernorm",
    "ffn_norm": "pre_feedforward_layernorm",
    "post_ffn_norm": "post_feedforward_layernorm",
}


def chunk_positions(start: int, chunk: int, sp: int) -> torch.Tensor:
    """[chunk] absolute positions in device row order (SP row r: rows r * L + [0, L), L = chunk / sp) for a chunk at
    start: the block-cyclic cache rows update_padded_kv_cache writes it to (module docstring)."""
    slab = chunk // sp
    group, boundary, offset = start // chunk, (start // slab) % sp, start % slab
    out = []
    for r in range(sp):
        first = group * slab + (slab if r < boundary else (offset if r == boundary else 0))
        lr = torch.arange(first, first + slab)
        out.append((lr // slab) * chunk + r * slab + lr % slab)
    pos = torch.cat(out)
    assert torch.equal(torch.sort(pos - start).values, torch.arange(chunk)), start
    return pos


@dataclass
class ChunkMeta:
    """One chunk's on-device inputs, shared by every layer (built in Model.embed)."""

    start: int
    end: int
    order: torch.Tensor  # [chunk] chunk row (position - start) of every device row (harness boundary only)
    slot: object  # [1, 1, 1, 1] uint32 scalars, replicated
    kv_actual: object
    valid: object
    sliding_rope: tuple  # cos, sin [1, 1, L, 256] per chip
    global_rope: tuple  # cos, sin [1, 1, L, 512] fp32 per chip (packed-Q lane order)
    keep: object  # [1, 1, L, 1] fp32: 1 for a real token, 0 for a pad row; None when the chunk has no pad rows
    chunk_mask: object  # [1, 1, L, chunk] bf16: 0 where this chunk's key (SP-gathered row) is causal for the query
    prefix_rows: int  # local cache rows of the slabs holding positions below start
    prefix_mask: object  # [1, 1, 1, SP * prefix_rows] fp32: MASK on cache keys at or past start; None: all visible
    owned: list  # device tensors to free with the meta
    slot_index: int = 0  # the KV slot (host int) of the slot scalar


class State:
    """The served KV cache of the model's layers (slot 0): the attention modules' own caches."""

    def __init__(self, model, max_seq: int):
        self.model, self.max_seq = model, max_seq

    def formats(self) -> dict:
        return {"key": ttnn.bfloat8_b, "value": ttnn.bfloat8_b}

    def cache(self, layer: int) -> dict:
        attn = self.model.attn[layer]
        if isinstance(attn, GlobalAttention):
            return {"kv": attn.cache}
        return {"key": attn.cache_k, "value": attn.cache_v}

    def load_prefix(self, layer: int, tensors: dict, n: int):
        """Harness boundary: the reference's state for positions [0, n) (key / value [Hkv, >= n, D], NeoX lanes) into
        slot 0, one chunk at a time through the cache write (rows past n of the last record zero). The model's next
        sequential chunk (embed with start None) then starts at n, so embed stages it and not the first layer."""
        m, attn = self.model, self.model.attn[layer]
        m.next_start = n
        sp, tp = m.sp, m.tp
        mapper = ttnn.ShardTensor2dMesh(m.mesh, (sp, tp), dims=(0, 1))
        for cs in range(0, n, m.chunk):
            ce = min(n, cs + m.chunk)

            def rows(x):
                out = torch.zeros(x.shape[0], m.chunk, x.shape[-1], dtype=torch.float32)
                out[:, : ce - cs] = x[:, cs:ce].float()
                return out

            if isinstance(attn, GlobalAttention):
                x = pack_global_kv(rows(tensors["key"]), rows(tensors["value"]))
                x = x.reshape(x.shape[0], sp, m.slab, attn.packed).permute(1, 0, 2, 3).contiguous()
                # host-rounded bfp8 (hooks._GlobalAttentionStep: the device's typecast of bf16-exact values is biased)
                t = ttnn.from_torch(
                    x,
                    device=m.mesh,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
                attn.load_rows(t, cs, ce)
                ttnn.deallocate(t)
            else:
                kv = []
                for name in ("key", "value"):
                    x = rows(tensors[name])
                    if name == "key":
                        x = x[..., sliding_kv_indices(attn.d)]
                    x = x.reshape(x.shape[0], sp, m.slab, attn.d).permute(1, 0, 2, 3).contiguous()
                    kv.append(
                        ttnn.from_torch(
                            x,
                            device=m.mesh,
                            dtype=ttnn.float32,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=mapper,
                        )
                    )
                attn.load_rows(kv[0], kv[1], cs, ce)
                for t in kv:
                    ttnn.deallocate(t)

    def _read(self, cache, n: int) -> torch.Tensor:
        """Slot 0's rows of positions [0, n): [heads (TP-major), n, width] float."""
        m = self.model
        local = -(-n // m.chunk) * m.slab
        _, h, _, w = cache.shape
        part = ttnn.slice(cache, (0, 0, 0, 0), (1, h, local, w), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        chips = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(part)]  # mesh row-major: r * TP + t
        ttnn.deallocate(part)
        lr = torch.arange(local)
        out = torch.zeros(m.tp * h, local * m.sp, w)
        for r in range(m.sp):
            pos = (lr // m.slab) * m.chunk + r * m.slab + lr % m.slab
            for t in range(m.tp):
                out[t * h : (t + 1) * h, pos] = chips[r * m.tp + t].reshape(h, local, w)
        return out[:, :n]

    def to_torch(self, layer: int, n: int) -> dict:
        """The reference's state layout: key / value [Hkv, n, D], NeoX lanes."""
        attn = self.model.attn[layer]
        if isinstance(attn, GlobalAttention):
            packed = self._read(attn.cache, n)
            rotary, nonrotary, value_order = global_kv_indices(attn.d)
            value = torch.zeros(packed.shape[0], n, attn.d)
            value[..., value_order] = packed[..., ROTARY_DIM:]
            key = torch.zeros_like(value)
            key[..., rotary] = packed[..., :ROTARY_DIM]
            key[..., nonrotary] = value[..., nonrotary] * attn.k_gamma[nonrotary]
            return {"key": key, "value": value}
        k, v = self._read(attn.cache_k, n), self._read(attn.cache_v, n)
        key = torch.zeros_like(k)
        key[..., sliding_kv_indices(attn.d)] = k
        return {"key": key, "value": v}

    def free(self):
        pass  # the caches belong to the layers' attention modules (the model's lifetime)


class Model:
    def __init__(self, mesh, spec, layers, lm_head: bool = True, caches: dict | None = None):
        """caches: {layer: served cache(s)} allocated by the serving adapter (tt/serving.py): a global layer's packed
        tensor, a sliding layer's (K, V); None: every attention module allocates its own (KV_SLOTS slots)."""
        t0 = time.time()
        self.mesh = mesh
        self.sp, self.tp = sp, tp = tuple(mesh.shape)
        ckpt = spec.get("paths.hf")
        cfg = weights.text_config(ckpt)
        self.chunk, self.max_seq = int(spec.get("target.chunk")), int(spec.get("target.seq"))
        self.slab = self.chunk // sp
        self.layers = list(layers)
        self.hidden = cfg["hidden_size"]
        replicate = ttnn.ReplicateTensorToMesh(mesh)

        # Embedding: bf16 ROW_MAJOR (ttnn.embedding's only weight format), hidden split over TP.
        embed = weights.load(ckpt, "embed_tokens.weight")  # [V, H] bf16
        self.vocab = embed.shape[0]
        self.embed_w = ttnn.from_torch(
            embed.reshape(1, 1, *embed.shape),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(None, -1)),
        )
        self.embed_scale = float(self.hidden**0.5)
        self.stream = ttnn.float32 if settings.get("RESIDUAL_STREAM") == "fp32" else ttnn.bfloat16
        eps = cfg["rms_norm_eps"]

        self.norms, self.attn, self.mlp, self.res = {}, {}, {}, {}
        for i in self.layers:
            self.norms[i] = {
                step: RMSNorm(mesh, weights.load(ckpt, f"layers.{i}.{name}.weight"), eps)
                for step, name in NORM_WEIGHTS.items()
            }
            scale = weights.load(ckpt, f"layers.{i}.layer_scalar").float().item()
            self.res[i] = (ResidualAdd(dtype=self.stream), ResidualAdd(scale, dtype=self.stream))
            self.mlp[i] = MLP(
                mesh,
                {n: weights.load(ckpt, f"layers.{i}.mlp.{n}.weight") for n in ("gate_proj", "up_proj", "down_proj")},
            )
            sliding = cfg["layer_types"][i] == "sliding_attention"
            names = ["q_proj", "k_proj", "o_proj", "q_norm", "k_norm"] + (["v_proj"] if sliding else [])
            w = {n: weights.load(ckpt, f"layers.{i}.self_attn.{n}.weight").float() for n in names}
            own = {} if caches is None else {"caches" if sliding else "cache": caches[i]}
            cls = SlidingAttention if sliding else GlobalAttention
            self.attn[i] = cls(mesh, w, cfg, max_seq=self.max_seq, chunk=self.chunk, chunk_tables=False, **own)
        self.final = RMSNorm(mesh, weights.load(ckpt, "norm.weight"), eps)
        self.has_global = any(isinstance(a, GlobalAttention) for a in self.attn.values())
        self.has_sliding = any(isinstance(a, SlidingAttention) for a in self.attn.values())

        # RoPE tables [max_seq, width], looked up per chunk by position (ttnn.embedding: bf16 tables). "hilo": each fp32
        # table as a bf16 head plus a bf16 remainder, summed in fp32 after the lookup (~2^-17 relative).
        self.rope_lookup = settings.get("MODEL_ROPE_LOOKUP")

        def tables(cos, sin):
            out = []
            for x in (cos, sin):
                hi = x.to(torch.bfloat16)
                parts = [hi] + ([(x - hi.float()).to(torch.bfloat16)] if self.rope_lookup == "hilo" else [])
                out.append(
                    [
                        ttnn.from_torch(
                            p.contiguous(),
                            device=mesh,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            mesh_mapper=replicate,
                        )
                        for p in parts
                    ]
                )
            return out

        first = {type(a): a for a in self.attn.values()}
        if self.has_sliding:
            self.sliding_tables = tables(*sliding_rope_tables(cfg, self.max_seq))
            self.sliding_fp32 = first[SlidingAttention].rope == "elementwise"
        if self.has_global:
            g = first[GlobalAttention]
            # The packed-Q order's first 128 lanes rotate (adjacent pairs over inv_freq[0..63]); the rest are cos 1, sin 0.
            freqs = g.inv_freq[: ROTARY_DIM // 2].double().repeat_interleave(2)
            emb = torch.arange(self.max_seq, dtype=torch.float64)[:, None] * freqs[None, :]
            self.global_tables = tables(emb.cos().float(), emb.sin().float())
            rest = g.d - ROTARY_DIM
            self.global_rest = [
                ttnn.from_torch(
                    torch.full((1, 1, self.slab, rest), v),
                    device=mesh,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=replicate,
                )
                for v in (1.0, 0.0)
            ]

        self.slot = self._scalar(0)
        self.slot_scalars = {0: self.slot}  # slot index -> its replicated scalar, uploaded once per slot
        self.meta = None
        self.next_start = 0

        # LM head: the tied embedding^T, bfp8, vocab split over all chips (SP row r, TP column t: vocab block r * TP + t).
        self.lm_head = None
        if lm_head:
            self.softcap = cfg.get("final_logit_softcapping")
            block = self.vocab // (sp * tp)
            wt = embed.transpose(0, 1).reshape(self.hidden, sp, tp * block).permute(1, 0, 2).unsqueeze(1)
            self.lm_head = ttnn.from_torch(
                wt.contiguous(),
                device=mesh,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(0, 3)),
            )
            self.head_cfg = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=getattr(ttnn.MathFidelity, settings.get("LM_HEAD_FIDELITY")),
                math_approx_mode=settings.get("MATH_APPROX_MODE"),
                fp32_dest_acc_en=settings.get("LM_HEAD_FP32_ACC"),
                packer_l1_acc=settings.get("PACKER_L1_ACC"),
            )
        del embed
        self.load_seconds = time.time() - t0

    # ---- staging (harness / server boundary) ---------------------------------------------------------------------
    def _scalar(self, value: int):
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=self.mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _upload(self, x, dtype, layout, dims=None):
        mapper = (
            ttnn.ReplicateTensorToMesh(self.mesh)
            if dims is None
            else ttnn.ShardTensor2dMesh(self.mesh, (self.sp, self.tp), dims=dims)
        )
        return ttnn.from_torch(
            x, device=self.mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=mapper
        )

    def _lookup(self, positions, parts, fp32: bool):
        """RoPE values of this chip's rows: [1, 1, L, width], fp32 (head + remainder) or bf16 (head only)."""
        vals = [ttnn.unsqueeze_to_4D(ttnn.embedding(positions, p, layout=ttnn.TILE_LAYOUT)) for p in parts]
        if not fp32:
            for v in vals[1:]:
                ttnn.deallocate(v)
            return vals[0]
        out = ttnn.typecast(vals[0], ttnn.float32)
        for v in vals[1:]:
            v32 = ttnn.typecast(v, ttnn.float32)
            s = ttnn.add(out, v32, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for t in (out, v32):
                ttnn.deallocate(t)
            out = s
        for v in vals:
            ttnn.deallocate(v)
        return out

    def _free_meta(self):
        if self.meta is not None:
            for t in self.meta.owned:
                ttnn.deallocate(t)
            self.meta = None

    def _stage(self, start: int, end: int, slot: int = 0):
        """The chunk's ChunkMeta (module docstring), writing KV slot `slot`."""
        assert start % ttnn.TILE_SIZE == 0 and 0 <= start < end <= start + self.chunk, (start, end)
        if slot not in self.slot_scalars:
            self.slot_scalars[slot] = self._scalar(slot)
        assert start + self.chunk <= self.max_seq, (start, self.max_seq)  # the server pulls the last chunk back
        self._free_meta()
        pos = chunk_positions(start, self.chunk, self.sp)
        owned = []

        def keep(t):
            owned.append(t)
            return t

        positions = keep(self._upload(pos.reshape(1, self.chunk), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, None)))
        qpos = keep(self._upload(pos.float().reshape(1, 1, self.chunk, 1), ttnn.float32, ttnn.TILE_LAYOUT, (2, None)))
        kv_actual, valid = keep(self._scalar(start)), keep(self._scalar(end))

        sliding_rope = global_rope = ()
        if self.has_sliding:
            sliding_rope = tuple(keep(self._lookup(positions, p, self.sliding_fp32)) for p in self.sliding_tables)
        if self.has_global:
            rot = [self._lookup(positions, p, True) for p in self.global_tables]
            global_rope = tuple(
                keep(ttnn.concat((r, rest), dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                for r, rest in zip(rot, self.global_rest)
            )
            for t in rot:
                ttnn.deallocate(t)

        pad_keep = None
        if end < start + self.chunk:
            pad_keep = keep(ttnn.lt(qpos, float(end), dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG))

        chunk_mask, prefix_rows, prefix_mask = None, -(-start // self.chunk) * self.slab, None
        if self.has_global:
            # Query (this chip's row) vs key (the SP-gathered chunk rows, SP row major = device row order).
            kpos = self._upload(pos.float().reshape(1, 1, 1, self.chunk), ttnn.float32, ttnn.TILE_LAYOUT)
            later = ttnn.gt(kpos, qpos, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            chunk_mask = keep(ttnn.multiply(later, MASK, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            for t in (kpos, later):
                ttnn.deallocate(t)
            if start % self.chunk:
                # Cache keys of the prefix slabs, SP-gathered: SP row r' local row lr holds position
                # (lr // L) * C + r' * L + lr % L; this chunk's own rows (written before the attention) are masked.
                lr = torch.arange(prefix_rows)
                cpos = torch.cat(
                    [(lr // self.slab) * self.chunk + r * self.slab + lr % self.slab for r in range(self.sp)]
                )
                ck = self._upload(cpos.float().reshape(1, 1, 1, -1), ttnn.float32, ttnn.TILE_LAYOUT)
                late = ttnn.ge(ck, float(start), dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                prefix_mask = keep(ttnn.multiply(late, MASK, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                for t in (ck, late):
                    ttnn.deallocate(t)

        self.meta = ChunkMeta(
            start=start,
            end=end,
            order=pos - start,
            slot=self.slot_scalars[slot],
            kv_actual=kv_actual,
            valid=valid,
            sliding_rope=sliding_rope,
            global_rope=global_rope,
            keep=pad_keep,
            chunk_mask=chunk_mask,
            prefix_rows=prefix_rows,
            prefix_mask=prefix_mask,
            owned=owned,
            slot_index=slot,
        )
        return positions

    # ---- the ladder / server interface ---------------------------------------------------------------------------
    def new_state(self, max_seq: int) -> State:
        assert max_seq <= self.max_seq, (max_seq, self.max_seq)
        self.next_start = 0
        return State(self, max_seq)

    def embed(self, tokens, start: int | None = None):
        """One chunk of token ids (natural order, PAD_ID past the real tokens) at absolute start -> the residual.
        start None: the chunk after the one embedded last since new_state, or after the prefix State.load_prefix loaded
        (sequential prefill, as the ladder and the profile call it),
        so its inputs are staged here and not by the first layer (layer() restages a chunk-aligned start that differs).
        """
        if start is None:
            start = self.next_start if self.next_start + self.chunk <= self.max_seq else 0
        tokens = torch.as_tensor(tokens).long().flatten()
        assert tokens.numel() == self.chunk, (tokens.numel(), self.chunk)
        pad = tokens == PAD_ID
        n_real = int(pad.int().argmax()) if bool(pad.any()) else self.chunk
        assert not bool((~pad[n_real:]).any()), "PAD_ID only after the real tokens"
        self._stage(start, start + n_real)
        self.next_start = start + self.chunk
        ids = torch.where(pad, 0, tokens)[self.meta.order]
        tok = self._upload(ids.reshape(1, self.chunk), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, None))
        return self._embed_ids(tok)

    def embed_device(self, payload, start: int, end: int, slot: int = 0):
        """The server's chunk input, already on the device -> the residual. payload: [SP, 1, C / SP] uint32 sharded
        on the SP axis, TP-replicated, in device row order (tt-d-gen's ring_sdpa_reshuffle by start, = chunk_positions),
        PAD_ID past end; not consumed. PAD_ID is clamped to a valid id on the device (deepseek_v3_d_p
        tt_parallel_embedding): its rows never reach a real token (causal) and the keep mask zeroes their K / V."""
        self._stage(start, end, slot)
        self.next_start = start + self.chunk
        ids = ttnn.reshape(payload, (1, self.slab))
        return self._embed_ids(ttnn.minimum(ids, self.vocab - 1))

    def _embed_ids(self, tok):
        """[1, L] uint32 ids per chip (device row order; consumed) -> the residual."""
        e = ttnn.embedding(tok, self.embed_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(tok)
        # * sqrt(H) in fp32 (a bf16 scalar would be 73.5, not 73.32), into the residual stream dtype.
        e32 = ttnn.typecast(ttnn.unsqueeze_to_4D(e), ttnn.float32)
        ttnn.deallocate(e)
        x = ttnn.multiply(e32, self.embed_scale, dtype=self.stream, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(e32)
        full = ttnn.all_gather(x, dim=3, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        h = ttnn.mesh_partition(full, dim=2, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(full)
        return h

    def _component(self, i: int, name: str):
        if name in NORM_WEIGHTS:
            norm = self.norms[i][name]
            return lambda ctx, x: norm(x)
        if name == "attention":
            attn = self.attn[i]
            return lambda ctx, x: attn(x, ctx.start, self.meta)
        if name == "attn_residual":
            return lambda ctx, a, b: self.res[i][0](a, b)
        if name == "ffn_residual":
            return lambda ctx, a, b: self.res[i][1](a, b)
        if name == "mlp":
            return lambda ctx, x: self.mlp[i](x)
        raise KeyError(name)

    def layer(self, i: int, h, start: int, state, end: int | None = None):
        """Block i on the residual h of the chunk staged by embed (start / end must match it)."""
        end = start + self.chunk if end is None else end
        meta = self.meta
        if meta is None or (meta.start, meta.end) != (start, end):
            # A residual from the host (from_host: a ladder rung that starts mid-stack) at a chunk-aligned start, whose
            # device row order is the natural one: stage its inputs here once, the first layer of the run.
            assert start % self.chunk == 0 and end == start + self.chunk, f"chunk [{start}, {end}) not staged by embed"
            self._stage(start, end)
        left = {}
        for s in BLOCK:
            for n in s.inputs:
                left[n] = left.get(n, 0) + 1

        def component(name):
            step = next(s for s in BLOCK if s.name == name)
            fn = self._component(i, name)

            def call(ctx, *xs):
                y = fn(ctx, *xs)
                for n, x in zip(step.inputs, xs):
                    left[n] -= 1
                    if left[n] == 0 and n != "in":  # the caller owns (and frees) the block input
                        ttnn.deallocate(x)
                return y

            return call

        return run_block(BLOCK, component, Ctx(i, start, self.chunk, state), h)

    def final_norm(self, h):
        return self.final(h)

    def logits(self, hidden, rows) -> torch.Tensor:
        """Soft-capped logits [len(rows), V] of chunk rows `rows` (chunk order). On device: gather the rows (an
        embedding lookup of the gathered hidden by row index), the vocab-split LM head matmul, 30 * tanh(x / 30)."""
        assert self.lm_head is not None, "device_model(..., lm_head=False)"
        rows = torch.as_tensor(rows).long()
        inv = torch.empty(self.chunk, dtype=torch.long)
        inv[self.meta.order] = torch.arange(self.chunk)
        n = rows.numel()
        n32 = -(-n // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        idx = torch.zeros(1, n32, dtype=torch.long)
        idx[0, :n] = inv[rows]
        xg = ttnn.all_gather(hidden, dim=2, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xa = ttnn.all_gather(xg, dim=2, cluster_axis=SP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(xg)
        table = ttnn.to_layout(xa, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(xa)
        idx_t = self._upload(idx, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        sel = ttnn.unsqueeze_to_4D(ttnn.embedding(idx_t, table, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        for t in (idx_t, table):
            ttnn.deallocate(t)
        lg = ttnn.linear(
            sel,
            self.lm_head,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.head_cfg,
        )
        ttnn.deallocate(sel)
        if self.softcap:
            x = ttnn.multiply(lg, 1.0 / self.softcap, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(lg)
            th = ttnn.tanh(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(x)
            lg = ttnn.multiply(th, self.softcap, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(th)
        composer = ttnn.create_mesh_composer(
            self.mesh, ttnn.MeshComposerConfig(dims=[0, 3], mesh_shape_override=ttnn.MeshShape(self.sp, self.tp))
        )
        out = ttnn.to_torch(lg, mesh_composer=composer)  # [SP, 1, n32, TP * block]
        ttnn.deallocate(lg)
        return out[:, 0].permute(1, 0, 2).reshape(n32, -1)[:n].float()

    def from_host(self, x: torch.Tensor):
        """Harness boundary: a chunk [C, H] at a chunk-aligned start (natural order = device row order) to the
        residual layout; layer() stages the chunk's inputs."""
        self._free_meta()
        return self._upload(
            x.reshape(self.sp, self.tp, self.chunk // (self.sp * self.tp), x.shape[-1]).float(),
            self.stream,
            ttnn.TILE_LAYOUT,
            (0, 1),
        )

    def to_host(self, t) -> torch.Tensor:
        """Harness boundary: the residual -> [C, H] float in natural chunk order."""
        composer = ttnn.create_mesh_composer(
            self.mesh, ttnn.MeshComposerConfig(dims=[0, 1], mesh_shape_override=ttnn.MeshShape(self.sp, self.tp))
        )
        out = ttnn.to_torch(t, mesh_composer=composer)
        out = out.reshape(-1, out.shape[-1]).float()
        if self.meta is None:
            return out
        nat = torch.empty_like(out)
        nat[self.meta.order] = out
        return nat

    def free(self, t):
        ttnn.deallocate(t)

    def sync(self):
        ttnn.synchronize_device(self.mesh)
