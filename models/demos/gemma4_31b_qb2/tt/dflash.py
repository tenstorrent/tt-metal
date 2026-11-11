# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""DFlash block-diffusion speculative decoding for Gemma4 31B on QB2 (batch 1, greedy).

Drafter: z-lab/gemma-4-31B-it-DFlash (arXiv 2602.06036), five Qwen3-style layers that draft
block_size-1 tokens in one non-causal forward. Its context is the target residual stream after
target_layer_ids, concatenated and projected by fc (+ hidden_norm); its queries come from the noise
block [RAW embed(pending), RAW embed(mask) x (block-1)] (no sqrt(H) scale); its lm_head is the
target's tied, vocab-sharded BF8 head.

Drafter on device (TP4 like the target): per chip 16 Q / 2 KV heads and 2688 MLP columns; O and
down projections reduce through the target's decode all-reduce slots. Every draft runs one
32-row tile: rows 0..15 carry the context rows committed by the previous verify (their K/V only),
rows 16..16+block-1 carry the noise block. Both write their roped K/V into a per-layer 2304-slot
ring before attention, so attention is one masked SDPA over the ring. Ring slot of position p is
(p % 2304 % 72) * 32 + p % 2304 // 72: any 72 consecutive positions occupy distinct 32-row
tiles, so one paged_update_cache call writes all 32 rows without the per-row tile
read-modify-write racing. The mask keeps the last 2048 written positions (2048 - block committed
plus the current block), the context of the verified 2048-slot ring, which also makes the 2048
sliding window a no-op mask; the 256 spare slots take a tile-padded prefill seed window.

Target verify: the QB2 decode path at batch=block, every row on the one user's page table at
positions p..p+block-1, with serialized per-row cache writes (Decoder.decode_forward
row_update_idxs) so every row's K/V is in the cache before decode SDPA, which then attends
0..position per row (causal within the block). Taps are copies of the named layers' outputs.

Iteration (target consumed positions < p; pending token t at p; drafter ring holds context < p):
draft d1..d(B-1); verify [t, d1..] at p..p+B-1 -> greedy g0..; a = longest prefix with
d(i+1) == g(i); emit d1..da, g(a); the next draft commits the tap rows 0..a into the ring; the
pending token becomes g(a) at p+a+1. Rejected rows' target K/V beyond the accepted point are
overwritten by the next verify before any row attends to them.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.gemma4_31b_qb2.tt.decoder import Decoder

DRAFTER_ID = "z-lab/gemma-4-31B-it-DFlash"
ROWS = 32
CTX_ROWS = 16  # rows 0..15 of the draft tile: committed context; rows 16..31: noise block
# Hidden-width RMSNorms of the draft tile on 28 width-sharded cores (interleaved: 1 core, 83 us each).
SHARDED_NORMS = os.environ.get("DFLASH_SHARDED_NORMS", "1") == "1"
# Opt-in: layer projections as DRAM-sharded decode matmuls (the target's _dlinear scheme) for drafters
# with a DRAM_GEOMETRY. Measured slower for the DFlash2 BF8 draft (6.35 vs 6.06 ms: the sharded-input
# conversions and staged BF8 packs cost more than the bandwidth gained), and its circular buffers
# overflow L1 with BF16 weights (the target's geometry assumes BF4).
DRAM_SHARDED = os.environ.get("DFLASH_DRAM_SHARDED", "0") == "1"
# Prefill taps of exactly the last min(2048, L) rows (the first verified path). Its unaligned slice and
# exact-row pads build length-dependent programs; the default tile-aligned window does not.
TAP_EXACT = os.environ.get("DFLASH_TAP_EXACT", "0") == "1"


def drafter_checkpoint(repo=DRAFTER_ID, revision=None, env="DFLASH_MODEL"):
    override = os.environ.get(env)
    if override:
        return Path(override)
    return Path(snapshot_download(repo, revision=revision, local_files_only=True))


def ring_slot(position, ring=2048):
    """Any `ring // 32` consecutive positions land in distinct 32-row tiles of the ring."""
    tiles = ring // 32
    i = position % ring
    return (i % tiles) * 32 + i // tiles


def rope_tables(head_dim, theta, positions):
    """HF rotate_half cos/sin [positions, head_dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def packed(*weights):
    """Torch [out, in] weights -> [in, 4 * local outs], each mesh rank owning its local blocks."""
    chunks = [w.chunk(4, dim=0) for w in weights]
    return torch.cat([torch.cat([c[r] for c in chunks], dim=0) for r in range(4)], dim=0).T


class DFlashDrafter:
    """z-lab DFlash drafter. Subclasses override _configure/_load_layer/_layer/device_step/read_drafts."""

    REPO, REVISION = DRAFTER_ID, None
    # Ring slots per layer: the context, the block and a tile-padded seed window; 72 tiles give 256-key
    # SDPA chunks.
    RING = 2304
    # Prompt positions a seed makes valid: the last min(L, CONTEXT). z-lab: a draft then sees the last
    # 2048 written positions (2048 - block committed + the block), as in the verified 2048-slot ring, so
    # its 2048 sliding window never masks; _configure sets the anchor-relative window to CONTEXT - block.
    CONTEXT = 2048
    window = None
    # Flash-decode attention needs every KV head's query heads x 32 rows in L1 (see _decode_attention).
    decode_attention = False
    # name -> (input cores, readers per DRAM bank, K tiles per block); None keeps interleaved weights.
    DRAM_GEOMETRY = None
    # Prefill tap rows seeded into the ring: a tile-aligned window ending at the tile-padded length, so at
    # least 2048 real positions for any tile-padding remainder (padding rows are selected to zero and never
    # marked valid). Every seed op runs at this fixed row count, whatever the prompt length.
    SEED_ROWS = 2080
    DEFAULT_WEIGHTS = "bf16"  # DFLASH_WEIGHTS=bf8|bf16 overrides

    def __init__(self, model, checkpoint=None, *, block_size=None, max_positions=8192, weight_dtype=None):
        self.model = model
        self.mesh = mesh = model.mesh
        path = Path(checkpoint or drafter_checkpoint(self.REPO, self.REVISION))
        cfg = json.loads((path / "config.json").read_text())
        self.cfg = cfg
        self._configure(cfg, block_size)
        assert 2 <= self.block <= CTX_ROWS
        self.max_positions = max_positions
        self.weight_dtype = weight_dtype or (
            ttnn.bfloat8_b if os.environ.get("DFLASH_WEIGHTS", self.DEFAULT_WEIGHTS) == "bf8" else ttnn.bfloat16
        )
        sd = {}
        with safe_open(path / "model.safetensors", framework="pt") as f:
            for k in f.keys():
                sd[k] = f.get_tensor(k)
        self.state_dict = sd
        grid = mesh.compute_with_storage_grid_size()
        self.grid = (grid.x, grid.y)
        # fc output columns are sharded and gathered: replicated it is 346 MB read per draft.
        self.fc = self._tensor(sd["fc.weight"].T, dim=1)
        self.hidden_mem = model.layers[0]._mem(self.hidden, 28)
        end = self.hidden_mem.shard_spec.grid.bounding_box().end
        self.hidden_norm_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=[end.x + 1, end.y + 1],
            subblock_w=3,
            block_h=1,
            block_w=self.hidden // 28 // 32,
            inplace=False,
        )
        self.hidden_norm = self._norm(sd["hidden_norm.weight"])  # prefill seed: many rows, interleaved
        self.context_norm = self._hidden_norm_weight(sd["hidden_norm.weight"])
        self.final_norm = self._hidden_norm_weight(sd["norm.weight"])
        self.layers = [self._load_layer(sd, i) for i in range(self.num_layers)]
        self.dram_sharded = DRAM_SHARDED and self.DRAM_GEOMETRY is not None
        if self.dram_sharded:
            dg = mesh.dram_grid_size()
            banks = dg.x * dg.y
            bank_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))})
            for layer in self.layers:
                for name, (_, readers, _) in self.DRAM_GEOMETRY.items():
                    w = layer[name]
                    kdim, ndim = list(w.shape)[-2:]
                    width = (ndim + banks * readers * 32 - 1) // (banks * readers * 32) * readers * 32
                    mem = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(bank_grid, (kdim, width), ttnn.ShardOrientation.ROW_MAJOR),
                    )
                    layer[name + "_dram"] = ttnn.to_memory_config(w, mem)
                    if name != "qkv":  # the prefill seed keeps the interleaved qkv for its many rows
                        del layer[name]
        cos, sin = rope_tables(self.head_dim, self.theta, max_positions)
        self.cos_host, self.sin_host = cos, sin
        self.cos = self._tensor(cos.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.sin = self._tensor(sin.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Ring writes use a single RING-row page; 16 identical entries give 64-byte table rows
        # and the bounded (modulo) path accepts a table wider than the one physical page.
        self.ring_table = self._tensor(
            torch.zeros(ROWS, 16, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        block_rows = torch.zeros(1, 1, ROWS, self.hidden)
        block_rows[:, :, CTX_ROWS:] = 1
        self.block_rows = self._tensor(block_rows, dtype=ttnn.bfloat16)
        self.ring_write_mem = ttnn.create_sharded_memory_config(
            (32, self.head_dim),
            ttnn.num_cores_to_corerangeset(ROWS, grid, True),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.grid),
            q_chunk_size=32,
            k_chunk_size=int(os.environ.get("DFLASH_SDPA_K_CHUNK", "256")),
            exp_approx_mode=False,
        )
        self.decode_sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(*self.grid),
            q_chunk_size=0,
            k_chunk_size=64,
            exp_approx_mode=False,
            max_cores_per_head_batch=16,
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.mm_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Persistent per-draft inputs (refreshed from host; a captured trace reads them in place).
        self.tokens = self._upload(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        self.rope_idx = self._upload(torch.zeros(1, ROWS, dtype=torch.int32), ttnn.uint32)
        self.slot_idx = self._upload(torch.full((ROWS,), -1, dtype=torch.int32), ttnn.int32)
        self.mask = self._tensor(torch.zeros(1, 1, ROWS, self.RING), dtype=ttnn.bfloat16)
        # Verify outputs consumed by the next draft: tap rows of the last verify.
        self.taps = [
            self._tensor(torch.zeros(1, 1, self.block, self.hidden), dtype=ttnn.bfloat16) for _ in self.tap_layers
        ]
        self.draft_out = self._upload(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        # Seed row fill (taps shorter than SEED_ROWS) and row-validity masks: rows past the prompt (tile
        # padding) are selected to zero.
        self.seed_zeros = self._tensor(torch.zeros(1, 1, self.SEED_ROWS, self.hidden), dtype=ttnn.bfloat16)
        self.seed_ones = self._tensor(torch.ones(1, 1, self.SEED_ROWS, self.hidden), dtype=ttnn.bfloat16)
        self.seed_tails = {
            r: self._tensor(
                (torch.arange(32) < r).reshape(1, 1, 32, 1).expand(1, 1, 32, self.hidden).float(), dtype=ttnn.bfloat16
            )
            for r in range(1, 32)
        }
        self.reset()

    def _configure(self, cfg, block_size):
        self.hidden = cfg["hidden_size"]
        self.heads = cfg["num_attention_heads"] // 4
        self.kv_heads = cfg["num_key_value_heads"] // 4
        self.head_dim = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.block = block_size or cfg["block_size"]
        self.mask_token = cfg["dflash_config"]["mask_token_id"]
        self.tap_layers = list(cfg["dflash_config"]["target_layer_ids"])
        self.theta = float(cfg.get("rope_theta", 1e6))
        self.num_layers = cfg["num_hidden_layers"]
        assert all(t in ("sliding_attention", "full_attention") for t in cfg["layer_types"])
        assert cfg.get("sliding_window", self.CONTEXT) >= self.CONTEXT
        self.window = self.CONTEXT - self.block
        assert self.RING >= max(self.CONTEXT, self.SEED_ROWS) and self.RING // 32 >= ROWS
        assert (self.hidden, self.heads, self.kv_heads, self.head_dim) == (5376, 16, 2, 128)
        self.local_mlp = cfg["intermediate_size"] // 4

    def _tensor(self, w, *, dim=None, dtype=None, layout=ttnn.TILE_LAYOUT):
        mesh = self.mesh
        return ttnn.from_torch(
            w.contiguous(),
            device=mesh,
            dtype=self.weight_dtype if dtype is None else dtype,
            layout=layout,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if dim is None else ttnn.ShardTensorToMesh(mesh, dim=dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _norm(self, w):
        return self._tensor(w.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16)

    def _hidden_norm_weight(self, w):
        if not SHARDED_NORMS:
            return self._norm(w)
        return self._tensor(w.reshape(1, 1, -1, 32), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _ring(self):
        return ttnn.zeros(
            [1, self.kv_heads, self.RING, self.head_dim],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _load_layer(self, sd, i):
        p = f"layers.{i}."
        a, m = p + "self_attn.", p + "mlp."
        return dict(
            qkv=self._tensor(packed(sd[a + "q_proj.weight"], sd[a + "k_proj.weight"], sd[a + "v_proj.weight"]), dim=1),
            o=self._tensor(sd[a + "o_proj.weight"].T, dim=0),
            gate_up=self._tensor(packed(sd[m + "gate_proj.weight"], sd[m + "up_proj.weight"]), dim=1),
            down=self._tensor(sd[m + "down_proj.weight"].T, dim=0),
            input_norm=self._hidden_norm_weight(sd[p + "input_layernorm.weight"]),
            post_norm=self._hidden_norm_weight(sd[p + "post_attention_layernorm.weight"]),
            q_norm=self._norm(sd[a + "q_norm.weight"]),
            k_norm=self._norm(sd[a + "k_norm.weight"]),
            k_ring=self._ring(),
            v_ring=self._ring(),
        )

    # ------------------------------------------------------------------ host state

    def _upload(self, x, dtype):
        return ttnn.from_torch(
            x.contiguous(),
            device=self.mesh,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    @staticmethod
    def _refresh(x, dest):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(x.contiguous(), dtype=dest.dtype, layout=dest.layout), dest)

    def reset(self):
        self.slot_pos = np.full(self.RING, -1, dtype=np.int64)
        self.mask_host = None

    def _mask(self, position):
        valid = self.slot_pos >= 0
        if self.window is not None:
            # Anchor-relative window: context positions [position - window, position - 1].
            valid &= self.slot_pos >= position - self.window
        valid = torch.from_numpy(valid)
        mask = torch.where(valid, 0.0, float("-inf")).reshape(1, 1, 1, self.RING).expand(1, 1, ROWS, self.RING)
        if self.mask_host is None or not torch.equal(mask, self.mask_host):
            self.mask_host = mask.clone()
            self._refresh(mask.bfloat16(), self.mask)

    def prepare(self, anchor, position, committed_start=None, committed=0):
        """Host inputs for draft of pending `anchor` at `position`; commits `committed` rows of the
        last verify (positions committed_start..) into the context in the same forward."""
        if position + self.block > self.max_positions:
            raise ValueError("Drafter RoPE table is smaller than the requested position")
        tokens = torch.zeros(1, 1, 1, ROWS, dtype=torch.int32)
        tokens[..., CTX_ROWS] = anchor
        tokens[..., CTX_ROWS + 1 : CTX_ROWS + self.block] = self.mask_token
        rope = torch.full((1, ROWS), position, dtype=torch.int32)
        slots = torch.full((ROWS,), -1, dtype=torch.int32)
        for r in range(committed):
            rope[0, r] = committed_start + r
            slots[r] = ring_slot(committed_start + r, self.RING)
            self.slot_pos[slots[r]] = committed_start + r
        for r in range(self.block):
            rope[0, CTX_ROWS + r] = position + r
            slots[CTX_ROWS + r] = ring_slot(position + r, self.RING)
            self.slot_pos[slots[CTX_ROWS + r]] = position + r
        self._refresh(tokens, self.tokens)
        self._refresh(rope, self.rope_idx)
        self._refresh(slots, self.slot_idx)
        self._mask(position)

    # ------------------------------------------------------------------ device graph

    def _linear(self, x, w, dtype=ttnn.bfloat16):
        """M = one tile: multicast the activation, every core streams its own weight columns."""
        cores = self.grid[0] * self.grid[1]
        m, n, kt = x.padded_shape[-2] // 32, w.padded_shape[-1] // 32, w.padded_shape[-2] // 32
        pn = (n + cores - 1) // cores
        sw = max(i for i in range(1, min(pn, 8) + 1) if pn % i == 0)
        sh = max(i for i in range(1, min(m, 8 // sw) + 1) if m % i == 0)
        block = max(i for i in range(1, min(kt, 8) + 1) if kt % i == 0)
        config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=self.grid,
            in0_block_w=block,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=m,
            per_core_N=pn,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        return ttnn.linear(
            x,
            w,
            program_config=config,
            compute_kernel_config=self.mm_compute,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _dlinear(self, x, layer, name, dtype=ttnn.bfloat16):
        """DRAM-sharded decode matmul of the 32-row tile. Row-parallel outputs (o, down) are staged
        BF16 on 56 cores then packed BF8 in place for the all-reduce, as in the target's sliding layers."""
        if not self.dram_sharded:
            return self._linear(x, layer[name], dtype)
        w = layer[name + "_dram"]
        kdim, ndim = w.shape[-2], w.shape[-1]
        cores, readers, k_block = self.DRAM_GEOMETRY[name]
        staged = dtype == ttnn.bfloat8_b
        x = ttnn.to_memory_config(x, self.model.layers[0]._mem(kdim, cores))
        config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=k_block,
            per_core_M=1,
            per_core_N=3 if staged else (ndim + cores * 32 - 1) // (cores * 32),
            num_workers_per_dram_bank=readers,
        )
        y = ttnn.linear(
            x,
            w,
            program_config=config,
            compute_kernel_config=self.mm_compute,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        if staged:
            return ttnn.typecast(y, ttnn.bfloat8_b, memory_config=y.memory_config())
        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

    def _rms(self, x, w):
        return ttnn.rms_norm(x, weight=w, epsilon=self.eps, compute_kernel_config=self.compute)

    def _hrms(self, x, w):
        """RMSNorm over the hidden width of the 32-row draft tile."""
        if not SHARDED_NORMS:
            return self._rms(x, w)
        x = ttnn.to_memory_config(x, self.hidden_mem)
        y = ttnn.rms_norm(
            x,
            weight=w,
            epsilon=self.eps,
            program_config=self.hidden_norm_config,
            compute_kernel_config=self.compute,
            memory_config=self.hidden_mem,
        )
        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

    def _rope_rows(self, idx, rows):
        cos = ttnn.embedding(idx, self.cos, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(idx, self.sin, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(cos, (1, 1, rows, self.head_dim)), ttnn.reshape(sin, (1, 1, rows, self.head_dim))

    def _reduce(self, x, name):
        # The target's decode all-reduce slots: BF8 partials in, BF16 sum out (sliding layer 0).
        assert x.dtype == ttnn.bfloat8_b
        y = self.model.layers[0]._decode_reduce(x, name)
        return ttnn.to_memory_config(y, ttnn.DRAM_MEMORY_CONFIG)

    def _context(self):
        """hidden_norm(fc(taps)) of the last verify, rows 0..block-1 of a zero-padded 32-row tile."""
        taps = ttnn.concat(self.taps, dim=3)
        ctx = self.model.gather_hidden(self._linear(taps, self.fc))
        ctx = self._hrms(ctx, self.context_norm)
        return ttnn.pad(ctx, [(0, 0), (0, 0), (0, ROWS - self.block), (0, 0)], value=0.0)

    def _embed_block(self):
        model = self.model
        x = ttnn.embedding(self.tokens, model.embedding, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(x, [1, 1, ROWS, self.hidden // 4])
        return ttnn.to_memory_config(model.gather_hidden(x), ttnn.DRAM_MEMORY_CONFIG)

    def _ring_write(self, t, ring):
        t = ttnn.to_memory_config(ttnn.transpose(t, 1, 2), self.ring_write_mem)
        ttnn.experimental.paged_update_cache(
            ring, t, update_idxs_tensor=self.slot_idx, page_table=self.ring_table, cache_position_modulo=self.RING
        )

    def _attention(self, xin, layer, cos, sin):
        """Block queries and [context | block] K/V of the 32-row tile -> all-reduced o_proj output."""
        qkv = self._dlinear(xin, layer, "qkv")
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.heads,
            num_kv_heads=self.kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q = ttnn.experimental.rotary_embedding(self._rms(q, layer["q_norm"]), cos, sin)
        k = ttnn.experimental.rotary_embedding(self._rms(k, layer["k_norm"]), cos, sin)
        self._ring_write(k, layer["k_ring"])
        self._ring_write(v, layer["v_ring"])
        attn = (self._decode_attention if self.decode_attention else self._sdpa)(q, layer)
        o = self._dlinear(
            ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG), layer, "o", ttnn.bfloat8_b
        )
        return self._reduce(o, "o")

    def _sdpa(self, q, layer):
        return ttnn.transformer.scaled_dot_product_attention(
            q,
            layer["k_ring"],
            layer["v_ring"],
            attn_mask=self.mask,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
            program_config=self.sdpa_config,
            compute_kernel_config=self.compute,
        )

    def _decode_attention(self, q, layer):
        """Flash-decode over the ring: KV heads as the batch, each KV head's query heads x 32 rows as
        packed MQA heads. Splits the ring's keys across cores; prefill SDPA runs one (head, q chunk)
        per core, 8 busy cores for the whole ring."""
        group = self.heads // self.kv_heads
        ring = (self.kv_heads, 1, self.RING, self.head_dim)
        out = ttnn.transformer.scaled_dot_product_attention_decode(
            ttnn.reshape(q, (1, self.kv_heads, group * ROWS, self.head_dim)),
            ttnn.reshape(layer["k_ring"], ring),
            ttnn.reshape(layer["v_ring"], ring),
            is_causal=False,
            attn_mask=self.packed_mask,
            scale=1.0 / math.sqrt(self.head_dim),
            program_config=self.decode_sdpa_config,
            compute_kernel_config=self.compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(out, (1, self.heads, ROWS, self.head_dim))

    def _mlp(self, h, layer):
        gu = self._dlinear(h, layer, "gate_up")
        act = ttnn.multiply(
            gu[..., : self.local_mlp],
            gu[..., self.local_mlp :],
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        )
        return self._reduce(self._dlinear(act, layer, "down", ttnn.bfloat8_b), "down")

    def _layer(self, x, ctx, layer, cos, sin):
        xin = ttnn.add(ttnn.multiply(self._hrms(x, layer["input_norm"]), self.block_rows), ctx)
        x = ttnn.add(x, self._attention(xin, layer, cos, sin))
        return ttnn.add(x, self._mlp(self._hrms(x, layer["post_norm"]), layer))

    def block_hidden(self):
        """Final-normed hidden [1,1,32,5376] of the draft tile (rows 16.. are the block)."""
        cos, sin = self._rope_rows(self.rope_idx, ROWS)
        if self.decode_attention:
            group = self.heads // self.kv_heads
            self.packed_mask = ttnn.repeat(self.mask, ttnn.Shape([1, self.kv_heads, group, 1]))
        ctx = self._context()
        x = self._embed_block()
        for layer in self.layers:
            x = self._layer(x, ctx, layer, cos, sin)
        return self._hrms(x, self.final_norm)

    def logits(self, hidden=None):
        """Target lm_head (vocab-sharded, no softcap) over the draft tile."""
        model = self.model
        return ttnn.linear(
            self.block_hidden() if hidden is None else hidden,
            model.head,
            core_grid=model.head_grid,
            dtype=ttnn.bfloat16,
            compute_kernel_config=model.head_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def device_step(self, sample):
        """The draft trace body. `sample(logits, out)` is the target's greedy sampler."""
        sample(self.logits(), self.draft_out)

    def read_drafts(self, anchor):
        ids = ttnn.to_torch(ttnn.get_device_tensors(self.draft_out)[0]).reshape(-1)
        return ids[CTX_ROWS + 1 : CTX_ROWS + self.block].long().tolist()

    # ------------------------------------------------------------------ prefill seed

    def tap_window(self, length):
        """Prefill rows [start, end) to capture for the seed."""
        if TAP_EXACT:
            return max(0, length - min(2048, self.SEED_ROWS)), length
        end = (length + 31) // 32 * 32
        return max(0, end - self.SEED_ROWS), end

    def _valid_rows(self, x, valid):
        """x [1,1,SEED_ROWS,H] with rows >= valid selected to zero. A select (not a multiply) so NaN/Inf in
        tile padding cannot leak. The mask is a fixed-shape concat of tile-aligned slices of persistent
        tensors, so no program depends on valid."""
        rows = x.shape[2]
        if valid >= rows:
            return x
        full, r = divmod(valid, 32)
        parts = [self.seed_ones[:, :, : 32 * full, :]] if full else []
        if r:
            parts.append(self.seed_tails[r])
        zeros = rows - 32 * full - (32 if r else 0)
        if zeros:
            parts.append(self.seed_zeros[:, :, :zeros, :])
        mask = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=2)
        return ttnn.where(mask, x, 0.0)

    def seed(self, taps, start, valid=None):
        """Write context rows for positions start..start+valid-1 from prefill taps {layer: [1,1,n,H]}
        (rows of positions start..start+n-1, n <= SEED_ROWS; rows past valid are tile padding and never
        seeded). The taps are filled to SEED_ROWS rows with zeros, so every op from fc to the ring copy
        runs at one shape for any prompt length. slot_pos marks the last min(L, CONTEXT) prompt positions,
        L = start + valid: the rows a tile window holds beyond those stay masked."""
        self.reset()
        rows = self.SEED_ROWS
        n = taps[self.tap_layers[0]].shape[2]
        valid = n if valid is None else valid
        tiled = (n + 31) // 32 * 32
        assert valid <= n <= rows <= self.RING
        parts = []
        for layer in self.tap_layers:
            t = taps[layer]
            if tiled != n:
                # Exact-length taps only: pad to whole tiles; those rows are masked below.
                t = ttnn.pad(t, [(0, 0), (0, 0), (0, tiled - n), (0, 0)], value=0.0)
            if tiled < rows:
                t = ttnn.concat([t, self.seed_zeros[:, :, : rows - tiled, :]], dim=2)
            parts.append(t)
        cat = ttnn.concat(parts, dim=3)
        del parts
        ctx = self.model.gather_hidden(ttnn.linear(cat, self.fc, compute_kernel_config=self.mm_compute))
        del cat
        hn = self._rms(ctx, self.hidden_norm)
        if valid < tiled:
            # Tap rows past valid (tile padding) may hold anything; the filled rows are exact zeros.
            hn = self._valid_rows(hn, valid)
        rope = []
        for table in (self.cos_host, self.sin_host):
            # Rows past valid are zero (their K is zero), so the table needs no rows past the prompt.
            host = torch.zeros(1, 1, rows, self.head_dim, dtype=torch.bfloat16)
            host[0, 0, :valid] = table[start : start + valid].bfloat16()
            rope.append(
                ttnn.from_torch(
                    host,
                    device=self.mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                )
            )
        for layer in self.layers:
            qkv = ttnn.linear(hn, layer["qkv"], compute_kernel_config=self.mm_compute)
            _, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv,
                num_heads=self.heads,
                num_kv_heads=self.kv_heads,
                transpose_k_heads=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            k = ttnn.experimental.rotary_embedding(self._rms(k, layer["k_norm"]), *rope)
            for t, ring in ((k, layer["k_ring"]), (v, layer["v_ring"])):
                ttnn.copy(self._ring_order(t, start), ring)
        length = start + valid
        first = max(0, length - self.CONTEXT)
        assert start <= first, f"seed window starts at {start}, after position {first}: context not covered"
        for pos in range(first, length):
            self.slot_pos[ring_slot(pos, self.RING)] = pos

    def _ring_order(self, t, start):
        """[1,kv,n,hd] rows of positions start.. -> [1,kv,RING,hd] in ring_slot order."""
        ring, n = self.RING, t.shape[2]
        tiles = ring // 32
        r = ttnn.pad(t, [(0, 0), (0, 0), (0, ring - n), (0, 0)], value=0.0) if n < ring else t
        k0 = start % ring
        if k0:
            # Row of position start + m belongs at ring index (start + m) % ring.
            r = ttnn.concat([r[:, :, ring - k0 :, :], r[:, :, : ring - k0, :]], dim=2)
        # Ring index i = tiles * j + t lands at slot 32 t + j: transpose the [32, tiles] grid of rows.
        r = ttnn.reshape(r, (self.kv_heads, 32, tiles, self.head_dim))
        r = ttnn.transpose(r, 1, 2)
        return ttnn.reshape(r, (1, self.kv_heads, ring, self.head_dim))


class DFlashSpeculator:
    """Batch-1 greedy speculative generation: drafter + QB2 target verify of `block` rows."""

    def __init__(self, generator, drafter, *, capacity):
        self.gen = generator
        self.model = generator.model
        self.mesh = self.model.mesh
        self.drafter = drafter
        self.block = drafter.block
        self.state = generator.allocate_cache(batch=1, capacity=capacity)
        self.capacity = self.state.capacity
        if drafter.max_positions < self.capacity:
            # Every position the cache can hold needs a drafter RoPE row (prefill seed and drafts).
            raise ValueError(f"Drafter max_positions {drafter.max_positions} < cache capacity {self.capacity}")
        b = self.block
        upload = generator._upload
        self.tables = {kind: upload(t[0:1]) for kind, t in self.state.page_tables.items()}
        self.verify_tables = {kind: upload(t[0:1].repeat(b, 1)) for kind, t in self.state.page_tables.items()}
        self.verify_tokens = upload(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        self.verify_positions = upload(torch.zeros(b, dtype=torch.int32))
        self.row_idxs = [upload(torch.full((b,), -1, dtype=torch.int32)) for _ in range(b)]
        # Bounded sliding layers: fused K/V writes at position % ring slots on the ring's pages.
        self.ring_slots = Decoder.SLIDING_WINDOW_PAGES * Decoder.PAGE_SIZE
        sliding = self.state.page_tables.get("sliding_attention")
        self.ring = None
        if sliding is not None:
            self.ring_idxs = [upload(torch.full((b,), -1, dtype=torch.int32)) for _ in range(b)]
            pages = sliding[0:1, : Decoder.SLIDING_WINDOW_PAGES].repeat(b, 1)
            self.ring = (self.ring_idxs, upload(pages))
        self.verify_out = upload(torch.zeros(1, 1, 1, ROWS, dtype=torch.int32), ttnn.uint32)
        self.draft_out = getattr(drafter, "draft_out", None)
        self.traces = {}
        self.timing = {}

    # ------------------------------------------------------------------ device steps

    def _sample(self, logits, out):
        shape = ttnn.Shape([1, 1, ROWS, self.model.config.vocab_size // 4])
        self.gen.sampler(ttnn.reshape(logits, shape, shape), tt_out_tok=out)

    def _verify_step(self, keep_logits=False):
        taps = {l: None for l in self.drafter.tap_layers}
        logits = self.model.decode_device(
            self.verify_tokens,
            self.verify_positions,
            page_tables=self.verify_tables,
            kv_cache=self.state.kv,
            batch=self.block,
            rope_positions=self.verify_positions,
            taps=taps,
            row_update_idxs=self.row_idxs,
            row_update_ring=self.ring,
        )
        for buf, l in zip(self.drafter.taps, self.drafter.tap_layers):
            ttnn.copy(taps[l], buf)
        self._sample(logits, self.verify_out)
        return logits if keep_logits else None

    def _draft_step(self):
        self.drafter.device_step(self._sample)

    def _run(self, name, step):
        trace = self.traces.get(name)
        if trace is None:
            step()
        else:
            ttnn.execute_trace(self.mesh, trace, cq_id=0, blocking=False)

    def capture(self, prompt):
        """Warm up eagerly on `prompt`, then capture the draft and verify traces."""
        self.release()
        self.gen.configure_sampling(temperature=0.0)
        t = self.prefill(prompt)
        self.draft(t, len(prompt))
        self.verify([t] * self.block, len(prompt))
        for name, step in (("draft", self._draft_step), ("verify", self._verify_step)):
            trace = ttnn.begin_trace_capture(self.mesh, cq_id=0)
            step()
            ttnn.end_trace_capture(self.mesh, trace, cq_id=0)
            self.traces[name] = trace
        ttnn.synchronize_device(self.mesh)

    def release(self):
        if self.traces:
            ttnn.synchronize_device(self.mesh)
        for trace in self.traces.values():
            ttnn.release_trace(self.mesh, trace)
        self.traces.clear()

    def _read(self, t, rows):
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1)[:rows].long()

    def set_verify_positions(self, position):
        b = self.block
        self.drafter._refresh(torch.arange(position, position + b, dtype=torch.int32), self.verify_positions)
        for r in range(b):
            row = torch.full((b,), -1, dtype=torch.int32)
            row[r] = position + r
            self.drafter._refresh(row, self.row_idxs[r])
            if self.ring is not None:
                row[r] = (position + r) % self.ring_slots
                self.drafter._refresh(row, self.ring_idxs[r])

    def set_verify_tokens(self, tokens):
        ids = torch.zeros(1, 1, 1, ROWS, dtype=torch.int32)
        ids.reshape(-1)[: len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
        self.drafter._refresh(ids, self.verify_tokens)

    def set_verify(self, tokens, position):
        self.set_verify_positions(position)
        self.set_verify_tokens(tokens)

    def verify(self, tokens, position, keep_logits=False):
        """Run verify rows (eager or traced); returns greedy ids (and logits when eager)."""
        self.set_verify(tokens, position)
        if keep_logits:
            logits = self._verify_step(keep_logits=True)
            return self._read(self.verify_out, self.block), logits
        self._run("verify", self._verify_step)
        return self._read(self.verify_out, self.block)

    def draft(self, anchor, position, committed_start=None, committed=0):
        self.drafter.prepare(anchor, position, committed_start, committed)
        self._run("draft", self._draft_step)
        return self.drafter.read_drafts(anchor)

    def prefill(self, prompt):
        """Target prefill of one prompt with tap capture; seeds the drafter. Returns first greedy id."""
        model, n = self.model, len(prompt)
        if n + 1 + self.block > self.capacity:
            raise ValueError("Prompt exceeds the speculator cache capacity")
        ids = torch.zeros(1, (n + 31) // 32 * 32, dtype=torch.int32)
        ids[0, :n] = torch.tensor(prompt, dtype=torch.int32)
        taps = {l: None for l in self.drafter.tap_layers}
        window = self.drafter.tap_window(n)
        logits = model.prefill_device(
            self.gen._upload(ids, ttnn.uint32),
            sequence_length=n,
            page_tables=self.tables,
            kv_cache=self.state.kv,
            taps=taps,
            tap_window=window,
        )
        self._sample(logits, self.verify_out)
        first = int(self._read(self.verify_out, 1)[0])
        self.drafter.seed(taps, window[0], n - window[0])
        return first

    # ------------------------------------------------------------------ generation

    def generate(self, prompt, max_new_tokens):
        """Greedy speculative generation. Returns (tokens, stats).

        Stops cleanly at the cache capacity: a request that does not fit returns the tokens that do
        (stats["truncated"]); the capacity never exceeds the drafter's RoPE table (checked at init).
        """
        # The last block may run block - 1 positions past max_new_tokens.
        fits = self.capacity - self.block - len(prompt)
        if fits < 1:
            raise ValueError(f"Prompt {len(prompt)} + block {self.block} does not fit capacity {self.capacity}")
        truncated = max_new_tokens > fits
        if truncated:
            logger.warning(f"Speculative generation stops at capacity {self.capacity}: {fits} of {max_new_tokens} tokens")
            max_new_tokens = fits
        # Acceptance compares draft and verify ids exactly: both samplers must be greedy.
        self.gen.configure_sampling(temperature=0.0)
        sync = lambda: ttnn.synchronize_device(self.mesh)  # noqa: E731
        sync()
        start = time.perf_counter()
        t = self.prefill(prompt)
        sync()
        ttft = time.perf_counter() - start
        out = [t]
        p = len(prompt)
        committed_start, committed = None, 0
        accepted, draft_s, verify_s = [], 0.0, 0.0
        start = time.perf_counter()
        while len(out) < max_new_tokens:
            assert p + self.block <= self.capacity
            self.drafter.prepare(t, p, committed_start, committed)
            t0 = time.perf_counter()
            self._run("draft", self._draft_step)
            # Verify positions do not depend on the drafts: upload them while the draft runs.
            self.set_verify_positions(p)
            drafts = self.drafter.read_drafts(t)
            t1 = time.perf_counter()
            self.set_verify_tokens([t] + drafts)
            t2 = time.perf_counter()
            self._run("verify", self._verify_step)
            greedy = self._read(self.verify_out, self.block).tolist()
            t3 = time.perf_counter()
            draft_s += t1 - t0
            verify_s += t3 - t2
            a = 0
            while a < len(drafts) and drafts[a] == greedy[a]:
                a += 1
            accepted.append(a)
            out += drafts[:a] + [greedy[a]]
            committed_start, committed = p, a + 1
            p += a + 1
            t = greedy[a]
        decode_s = time.perf_counter() - start
        iters = len(accepted)
        # Conservative: tokens past max_new_tokens from the last block are not counted.
        new_tokens = min(len(out), max_new_tokens) - 1
        stats = dict(
            ttft_s=ttft,
            decode_s=decode_s,
            iterations=iters,
            new_tokens=new_tokens,
            tokens_per_s=new_tokens / decode_s,
            traced=bool(self.traces),
            truncated=truncated,
            mean_accept=float(np.mean(accepted)) + 1.0,
            draft_ms=1000 * draft_s / iters,
            verify_ms=1000 * verify_s / iters,
            host_ms=1000 * (decode_s - draft_s - verify_s) / iters,
            accepted=accepted,
        )
        return out[:max_new_tokens], stats
