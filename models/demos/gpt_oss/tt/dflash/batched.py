# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy DFlash for 32 users at once (batch 32) on the batched gpt-oss-20b decode model (batched_decode.py, dense MoE).

A round, for every user u: the drafter proposes 7 tokens after the user's last token; the target checks the 8-row
block [last token, 7 drafts] of every user in one 256-row verify pass and keeps the drafts up to the first mismatch plus
its own next token. Every pass uses the same row order: row 32 i + u is row i of user u's block, at position
start_u + i, so a 32-row step i holds one row of each user (the batched decode's layout: one K/V cache write per step).

Passes (device tensors throughout; the host writes the per-round inputs and reads two 256-row argmax results):
  draft: the noise block [last token, mask x 7] through the drafter's 8 layers (256-row matmuls; Q / K norms and RoPE
      for all 256 rows at once with per-row cos / sin; attention in 8 steps of 32 rows: K/V of every step written first,
      then each step attends to its user's whole block, not causal), its LM head, a per-device argmax;
  verify: the block [last token, 7 drafts] through the target: norm, QKV, o_proj and the all-reduce over all 256 rows;
      the attention core in 8 steps of 32 rows (step i is the batched decode attention at positions start_u + i, so it
      sees the K/V rows of steps 0 .. i: causal within the block; Q and K rotated by one op, K and V written by one op),
      the MoE on all 256 rows at once (router, every expert's gate / up / down as 256-row matmuls in the dense layout of
      the batched decode, each expert weight read once per pass, the routing weights zeroing unrouted experts), the
      target's LM head, a per-device argmax; the outputs of target layers 1, 6, 11, 16, 21 kept for the drafter;
  context: those features of all 256 rows into the drafter's K/V cache at positions start_u + i (rows past the
      accepted ones are rewritten by the next round's block before they are read).
"""

import math
import time

import torch

import ttnn
from models.tt_transformers.tt.load_checkpoints import reverse_permute, reverse_permute_1d

from .draft import draft_snapshot, load_draft_config

TILE = ttnn.TILE_SIZE
U32 = 32  # rows per attention step: one row per user


def _replicate(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh if device else None, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _copy(mesh, dst, t, dtype):
    """Write host tensor t into the persistent device tensor dst (a trace input)."""
    ttnn.copy_host_to_device_tensor(_replicate(mesh, t, dtype, device=False), dst)


class _TPAllReduce:
    """TP all-reduce of [rows, hidden] partial sums over the mesh row with all_reduce_async: persistent scratch and a
    global semaphore per slot, two slots used alternately (each call is an all-rank synchronization, so no device reuses
    a slot before every device is done with it). ttnn.all_reduce, run back to back with other ops, now and then left a
    wrong tile on one device (verify outputs differed between identical runs); this one is repeatable. Inputs go in as
    BF8 (the batched decode's all-reduce precision) on 90 cores, so the [rows, 4 x hidden] scratch fits L1."""

    CORES, LINKS = 90, 2
    _instances = {}

    def __init__(self, mesh, rows, hidden):
        from ..fused_decode.config import width_sharded_memory_config

        D = mesh.shape[1]
        self.mesh = mesh
        self.memory_config = width_sharded_memory_config(mesh, hidden, self.CORES, rows=rows)
        scratch = width_sharded_memory_config(mesh, hidden * D, self.CORES, rows=rows)
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.slots = [
            (
                ttnn.empty([1, 1, rows, hidden * D], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=scratch),
                ttnn.create_global_semaphore(mesh, cores, 0),
            )
            for _ in range(2)
        ]  # fmt: skip
        self.calls = 0

    @classmethod
    def get(cls, x):
        mesh, rows, hidden = x.device(), x.shape[-2], x.shape[-1]
        key = (id(mesh), rows, hidden)
        if key not in cls._instances:
            cls._instances[key] = cls(mesh, rows, hidden)
        return cls._instances[key]

    @classmethod
    def reset(cls):
        """Start every pass on slot 0 (each pass makes an even number of calls: the slot order is the same in every
        captured trace and replay)."""
        for inst in cls._instances.values():
            inst.calls = 0

    def __call__(self, x, memory_config):
        xs = ttnn.to_memory_config(x, self.memory_config, dtype=ttnn.bfloat8_b)
        scratch, semaphore = self.slots[self.calls % 2]
        self.calls += 1
        out = ttnn.experimental.all_reduce_async(
            xs,
            scratch,
            cluster_axis=1,
            mesh_device=self.mesh,
            multi_device_global_semaphore=semaphore,
            memory_config=self.memory_config,
            dtype=ttnn.bfloat16,
            topology=ttnn.Topology.Ring,
            num_links=self.LINKS,
        )
        xs.deallocate(True)
        if memory_config is None:
            return out
        moved = ttnn.to_memory_config(out, memory_config)
        out.deallocate(True)
        return moved


def _all_reduce(x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """TP all-reduce over the mesh row of a [1, 1, rows, hidden] partial (any layout); the BF16 sum in memory_config."""
    return _TPAllReduce.get(x)(x, memory_config)


def _mm_config_256(k, n):
    """Program config for the drafter's and the verify's [256, k] x [k, n] projections (None: the auto config).
    Measured on one device: the auto config spreads these narrow outputs over few cores with all 8 row tiles each;
    2D over 10 x 8 cores (one row tile, n / 10 columns each) is 1.4-2.5x faster (drafter QKV 74 -> 37 us, gate / up
    86 -> 49, down 73 -> 51, o_proj 45 -> 33; verify QKV BF8 68 -> 27); the drafter's fc (k = 14400) is fastest 1D
    over 90 cores (487 -> 293 us)."""
    kt, nt = k // TILE, n // TILE
    if k == 14400 and nt == 90:
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9), in0_block_w=10, out_subblock_h=1, out_subblock_w=1,
            out_block_h=8, out_block_w=1, per_core_M=8, per_core_N=1, fuse_batch=True, fused_activation=None,
            mcast_in0=True,
        )  # fmt: skip
    if nt % 10:
        return None
    pn = nt // 10
    bw = {90: 10, 60: 12, 32: 8}.get(kt, max(d for d in range(1, 17) if kt % d == 0))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(10, 8), in0_block_w=bw, out_subblock_h=1,
        out_subblock_w=max(d for d in range(1, 5) if pn % d == 0), out_block_h=1, out_block_w=pn, per_core_M=1,
        per_core_N=pn, transpose_mcast=False, fused_activation=None,
    )  # fmt: skip


class _Argmax:
    """Row argmax over a vocabulary split over the mesh columns: each device's best (value, index) on device; the host
    combines the devices (read after the pass)."""

    def __init__(self, per_dev, vocab):
        self.per_dev, self.vocab = per_dev, vocab

    def device(self, logits):
        return ttnn.topk(logits, k=1, dim=-1, sorted=True)

    def combine(self, out, mesh):
        """The devices' (value, index) combined on device: [1, 1, rows, 1] FLOAT32 global token ids, the same on every
        device (the host combine's choice: padded vocabulary columns excluded, ties to the first device)."""
        D = mesh.get_num_devices()
        if not hasattr(self, "offsets"):
            rep = ttnn.ReplicateTensorToMesh(mesh)
            put = lambda t: ttnn.from_torch(
                t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
            )
            self.offsets = put((torch.arange(D) * self.per_dev).float().reshape(1, 1, 1, D))
        vals, idxs = out
        g = []
        for t in (vals, idxs):
            f = ttnn.typecast(t, ttnn.float32)
            g.append(
                ttnn.all_gather(
                    f, dim=3, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )  # fmt: skip
            f.deallocate(True)
        v, gi = g
        gi = ttnn.add(gi, self.offsets, output_tensor=gi)
        pad = ttnn.ge(gi, float(self.vocab))
        v = ttnn.where(pad, -1e30, v, output_tensor=v)
        pad.deallocate(True)
        best = ttnn.max(v, dim=-1, keepdim=True)
        hit = ttnn.eq(v, best)
        v.deallocate(True)
        best.deallocate(True)
        cand = ttnn.where(hit, gi, 1e9)
        hit.deallocate(True)
        gi.deallocate(True)
        tok = ttnn.min(cand, dim=-1, keepdim=True)
        cand.deallocate(True)
        return tok

    @staticmethod
    def read(tok):
        """Host copy of combine()'s token ids (device 0's)."""
        return ttnn.to_torch(ttnn.get_device_tensors(tok)[0]).reshape(-1).long().tolist()

    def host(self, out):
        vals, idxs = out
        v = torch.stack([ttnn.to_torch(t).float().reshape(-1) for t in ttnn.get_device_tensors(vals)])
        i = torch.stack([ttnn.to_torch(t).long().reshape(-1) for t in ttnn.get_device_tensors(idxs)])
        gi = i + torch.arange(v.shape[0])[:, None] * self.per_dev
        v = torch.where(gi < self.vocab, v, torch.full_like(v, -1e30))
        best = v.argmax(0)  # the first device with the maximum
        return gi[best, torch.arange(v.shape[1])].tolist()


class BatchedKVWrite:
    """Every user's block rows (rotated K / V heads in the prefill layout [1, kv_heads, rows, 64], L1 interleaved) ->
    the users' paged BF16 caches, one core per user (fused_decode/kernels/b32_kv_write.cpp: K on BRISC, V on NCRISC),
    one or several layers per program: replaces the 8 per-step head splits + paged_fused_update_cache calls."""

    def __init__(self, mesh, users, block, kv_heads, head_dim, block_size):
        self.mesh, self.users, self.block = mesh, users, block
        self.kv_heads, self.head_tiles, self.block_size = kv_heads, head_dim // TILE, block_size
        grid = mesh.compute_with_storage_grid_size()
        self.core_list = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:users]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.core_list])

    def __call__(self, ks, vs, caches, pos_ui, page_table):
        """ks / vs: per layer [1, kv_heads, rows, hd]; caches: per layer (k_cache, v_cache)."""
        from ..fused_decode.boundary import KERNEL_DIR

        n = len(caches)
        k0, v0 = caches[0]
        ct = [0, self.block, self.kv_heads, self.head_tiles, self.block_size, page_table.shape[-1] * 4, self.users]
        for t in (k0, v0, pos_ui, page_table, ks[0]):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        more = [a for l in range(1, n) for a in (caches[l][0].buffer_address(), caches[l][1].buffer_address(),
                                                  ks[l].buffer_address(), vs[l].buffer_address())]  # fmt: skip
        for u, c in enumerate(self.core_list):
            rt[c.x][c.y] = [u, ks[0].buffer_address(), vs[0].buffer_address(), k0.buffer_address(),
                            v0.buffer_address(), pos_ui.buffer_address(), page_table.buffer_address()] + more  # fmt: skip
        defines = [("KV_LAYERS", str(n))] if n > 1 else []
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "b32_kv_write.cpp"),
                core_ranges=self.cores,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[sel],
                    noc=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[sel],
                ),
                defines=[("KV_SEL", str(sel))] + defines,
            )
            for sel in (0, 1)
        ]
        scratch = 512 + n * 2 * 2 * self.kv_heads * self.head_tiles * 2048
        cb = ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=self.cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=scratch)],
        )
        io = [pos_ui, page_table] + [t for l in range(n) for t in (ks[l], vs[l], caches[l][0], caches[l][1])]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb]))


class BatchedDraft:
    """The z-lab drafter (8 layers, TP like a dense target layer) for `users` sequences, each with its own paged BF16
    K/V cache; draft and context passes over all users' rows at once."""

    def __init__(self, mesh_device, target, max_positions, users, window=None, block_size=64):
        from safetensors.torch import load_file

        self.mesh_device = mesh_device
        self.target = target
        self.path = draft_snapshot()
        c = load_draft_config(self.path)
        D = mesh_device.shape[1]
        self.hidden, self.hd = c["hidden_size"], c["head_dim"]
        self.nq, self.nkv = c["num_attention_heads"] // D, c["num_key_value_heads"] // D
        self.inter = c["intermediate_size"] // D
        self.layers = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        self.block = c["block_size"]
        self.mask_token_id = c["dflash_config"]["mask_token_id"]
        self.target_layer_ids = c["dflash_config"]["target_layer_ids"]
        self.ctx_in = len(self.target_layer_ids) * self.hidden
        self.window = window
        self.users = users
        self.rows = users * self.block
        w = load_file(str(self.path / "model.safetensors"))
        mesh = mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)

        # BF16 weights (BF8: round 49.85 -> 49.69 ms, acceptance 4.59 -> 4.53 tokens per round: slower overall)
        wdt = ttnn.bfloat16

        def put(t, mapper=rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def norm_w(t):
            return put(t.float().reshape(1, 1, -1, TILE), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        cols = ttnn.ShardTensorToMesh(mesh, dim=-1)
        rows = ttnn.ShardTensorToMesh(mesh, dim=-2)
        hd, nq, nkv = self.hd, self.nq, self.nkv

        def per_device(parts):
            """[q | k | v]-style row blocks: each device's slice of every part, side by side."""
            return torch.cat(
                [torch.cat([p[d * (p.shape[0] // D) : (d + 1) * (p.shape[0] // D)] for p in parts]) for d in range(D)]
            )

        self.fc_full = put(w["fc.weight"].T)  # replicated: the features are replicated on every device
        self.hidden_norm = norm_w(w["hidden_norm.weight"])
        self.norm = norm_w(w["norm.weight"])
        self.L = []
        for i in range(self.layers):
            p, a, m = f"layers.{i}.", f"layers.{i}.self_attn.", f"layers.{i}.mlp."
            q = reverse_permute(w[a + "q_proj.weight"], D * nq, D * nq * hd, self.hidden)
            k = reverse_permute(w[a + "k_proj.weight"], D * nkv, D * nkv * hd, self.hidden)
            v = w[a + "v_proj.weight"]
            qb = reverse_permute(w[a + "q_proj.bias"][:, None], D * nq, D * nq * hd, 1)[:, 0]
            kb = reverse_permute(w[a + "k_proj.bias"][:, None], D * nkv, D * nkv * hd, 1)[:, 0]
            vb = w[a + "v_proj.bias"]
            g, u = w[m + "gate_proj.weight"], w[m + "up_proj.weight"]
            self.L.append(
                dict(
                    ln1=norm_w(w[p + "input_layernorm.weight"]),
                    ln2=norm_w(w[p + "post_attention_layernorm.weight"]),
                    qkv=put(per_device([q, k, v]).T, cols, dtype=wdt),
                    qkv_b=put(per_device([qb, kb, vb]).reshape(1, -1), cols),
                    q_norm=norm_w(reverse_permute_1d(w[a + "q_norm.weight"])),
                    k_norm=norm_w(reverse_permute_1d(w[a + "k_norm.weight"])),
                    o=put(w[a + "o_proj.weight"].T, rows, dtype=wdt),
                    o_b=put(w[a + "o_proj.bias"].reshape(1, -1)),
                    g=put(g.T, cols, dtype=wdt),  # gate and up as two matmuls: no slices of a joint output
                    u=put(u.T, cols, dtype=wdt),
                    down=put(w[m + "down_proj.weight"].T, rows, dtype=wdt),
                )
            )
        self.block_size = block_size
        self.num_blocks = -(-(-(-(max_positions + self.block) // block_size)) // 8) * 8  # (32-byte page-table rows)
        user_tables = torch.arange(users * self.num_blocks, dtype=torch.int32).reshape(users, self.num_blocks)
        self.user_tables = user_tables
        self.page_table = _replicate(mesh, user_tables, ttnn.int32)  # row u: user u (each 32-row step)
        from ..batched_decode import sharded_norm_program_config
        from ..fused_decode.config import width_sharded_memory_config

        self.res_mc = width_sharded_memory_config(mesh, self.hidden, _TPAllReduce.CORES, rows=self.rows)
        self.norm_pc = sharded_norm_program_config(self.res_mc)
        self.kv_write = BatchedKVWrite(mesh, users, self.block, nkv, hd, block_size)
        # attention: block attention (b1's kernels, uniform), 8 users per program on 4 x 3 cores each
        self.ba = _UserBlockAttention(
            mesh, users, self.block, nq, nkv, hd, block_size, 8, 3, uniform=True, window=window or 0
        )
        heads_split = ttnn.ShardTensorToMesh(mesh, dim=1)
        self.caches = []
        for _ in range(self.layers):
            z = torch.zeros(users * self.num_blocks, D * nkv, block_size, hd)
            self.caches.append(
                tuple(
                    ttnn.from_torch(
                        z,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=heads_split,
                    )
                    for _ in range(2)
                )
            )
        from ..model import create_rope_setup

        self.rope = create_rope_setup(mesh, target.hf_config, max_local_batch_size=U32, datatype=ttnn.bfloat16)
        # all rows' Q / K norms and RoPE at once (prefill layout, per-row cos / sin); K / V written by BatchedKVWrite,
        # attention by the block-attention kernels
        self.ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # (HiFi2 / LoFi: no faster at 256 rows -- not compute-bound)
        self.mm_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.argmax = _Argmax(target.lm_head_weight.shape[-1], target.vocab_size)
        # the drafter's LM head on a BFP4 copy of the target's with BF8 rows: 0.89 -> 0.52 ms per pass, live-like
        # acceptance on a fixed text 3.07 (BF16 rows x BF8 weight: 3.05 / 3.10 in two runs)
        self.lm_head = ttnn.typecast(target.lm_head_weight, ttnn.bfloat4_b)

    # -- pieces ----------------------------------------------------------------------------------------------------
    def _linear(self, x, w, bias=None):
        pc = _mm_config_256(w.shape[-2], w.shape[-1]) if x.shape[-2] == self.rows else None
        return ttnn.linear(
            x,
            w,
            bias=bias,
            program_config=pc,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.mm_ck,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def rows_cos_sin(self, pos_rows):
        """Per-row cos / sin [1, 1, rows, hd] for positions pos_rows (device [1, rows] UINT32)."""
        out = []
        for m in (self.rope.cos_matrix, self.rope.sin_matrix):
            t = ttnn.embedding(pos_rows, m, layout=ttnn.TILE_LAYOUT)
            out.append(ttnn.reshape(t, [1, 1, self.rows, self.hd]))
        return out

    def _heads_qk(self, qkv, L, cos, sin, with_q):
        """[1, 1, rows, qkv] projection -> (Q or None, K, V) heads in the prefill layout [1, heads, rows, hd] (L1),
        K (and Q) head-normed and rotated, every row at its own position."""
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=self.nq, num_kv_heads=self.nkv, transpose_k_heads=False, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        trans = self.rope.get_both_trans_mats()["prefill"]

        def norm_rope(t, w):
            n = ttnn.rms_norm(t, epsilon=self.eps, weight=w, compute_kernel_config=self.ck)
            t.deallocate(True)
            out = ttnn.experimental.rotary_embedding_llama(n, cos, sin, trans, is_decode_mode=False)
            n.deallocate(True)
            return out

        k = norm_rope(k, L["k_norm"])
        if with_q:
            q = norm_rope(q, L["q_norm"])
        else:
            q.deallocate(True)
            q = None
        return q, k, v

    # -- prompt context ---------------------------------------------------------------------------------------------
    def add_prompt_context(self, features, user, chunk=2048):
        """The prompt's context K / V (prefill-style, paged_fill_cache of block-aligned chunks)."""
        n = features.shape[0]
        mesh = self.mesh_device
        pt_host = self.user_tables[user][None]
        qw, kvw = self.nq * self.hd, self.nkv * self.hd
        for s0 in range(0, n, chunk):
            rows = min(chunk, -(-(n - s0) // self.block_size) * self.block_size)
            x = torch.zeros(1, 1, rows, self.ctx_in)
            x[0, 0, : min(rows, n - s0)] = features[s0 : s0 + rows].float()
            tx = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ctx = self._linear(tx, self.fc_full)
            tx.deallocate(True)
            ctx = ttnn.rms_norm(ctx, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
            cos = self.rope.cos_matrix_prefill[:, :, s0 : s0 + rows, :]
            sin = self.rope.sin_matrix_prefill[:, :, s0 : s0 + rows, :]
            trans = self.rope.get_both_trans_mats()["prefill"]
            b0 = s0 // self.block_size
            chunk_pt = _replicate(mesh, pt_host[:, b0 : b0 + rows // self.block_size].contiguous(), ttnn.int32)
            for i in range(self.layers):
                L = self.L[i]
                qkv = self._linear(ctx, L["qkv"], L["qkv_b"])
                for j, cache in enumerate(self.caches[i]):
                    t = ttnn.slice(qkv, (0, 0, 0, qw + j * kvw), (1, 1, rows, qw + (j + 1) * kvw))
                    t = ttnn.transpose(ttnn.reshape(t, [1, rows, self.nkv, self.hd]), 1, 2)
                    if j == 0:
                        t = ttnn.rms_norm(t, epsilon=self.eps, weight=L["k_norm"], compute_kernel_config=self.ck)
                        t = ttnn.experimental.rotary_embedding_llama(t, cos, sin, trans, is_decode_mode=False)
                    ttnn.experimental.paged_fill_cache(cache, t, chunk_pt, batch_idx=0)
                    t.deallocate(True)
                qkv.deallocate(True)
            chunk_pt.deallocate(True)
            ctx.deallocate(True)

    # -- passes -----------------------------------------------------------------------------------------------------
    def context(self, features, pos, rot):
        """features: device [1, 1, rows, 5 * hidden] (row 32 i + u); pos: per-step positions; rot: per-row (cos, sin)."""
        ctx = self._linear(features, self.fc_full)
        ctx2 = ttnn.rms_norm(ctx, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
        ctx.deallocate(True)
        ks, vs = [], []  # every layer's K / V, then one write program for the 8 layers
        for li in range(self.layers):
            L = self.L[li]
            qkv = self._linear(ctx2, L["qkv"], L["qkv_b"])
            _, k, v = self._heads_qk(qkv, L, *rot, with_q=False)
            qkv.deallocate(True)
            ks.append(k)
            vs.append(v)
        ctx2.deallocate(True)
        self.kv_write(ks, vs, self.caches, pos[self.block], self.page_table)
        for t in ks + vs:
            t.deallocate(True)

    def draft(self, tokens, pos, rot, bound):
        """tokens: device [1, rows] UINT32 noise block (row 32 i + u: user u's last token for i = 0, else the mask).
        Returns the per-device argmax of every row (rows 32 i + u, i >= 1: the draft of position start_u + i)."""
        _TPAllReduce.reset()
        emb = ttnn.embedding(tokens, self.target.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # the FP32 residual stays width-sharded on the all-reduce's 90 cores: the residual adds and the norms run on
        # the shards (no DRAM round trips; the DRAM norm ran on 8 cores)
        h = ttnn.to_memory_config(ttnn.reshape(emb, [1, 1, self.rows, self.hidden]), self.res_mc, dtype=ttnn.float32)

        def norm(h, w):
            n = ttnn.rms_norm(
                h, epsilon=self.eps, weight=w, program_config=self.norm_pc, memory_config=self.res_mc,
                compute_kernel_config=self.ck,
            )  # fmt: skip
            x = ttnn.to_memory_config(n, ttnn.DRAM_MEMORY_CONFIG)
            n.deallocate(True)
            return x

        for li in range(self.layers):
            L = self.L[li]
            x = norm(h, L["ln1"])
            qkv = self._linear(x, L["qkv"], L["qkv_b"])
            x.deallocate(True)
            q_rows, k, v = self._heads_qk(qkv, L, *rot, with_q=True)
            qkv.deallocate(True)
            self.kv_write([k], [v], [self.caches[li]], pos[self.block], self.page_table)  # every row's K/V first
            k.deallocate(True)
            v.deallocate(True)
            kc, vc = self.caches[li]
            # block attention (b1's kernels, one instance per user, uniform): a user's 8 rows see the same positions
            # (start .. start + 7, not causal), every user's K/V read once per layer
            q = ttnn.permute(q_rows, (0, 2, 1, 3))  # [1, rows, heads, hd]
            q_rows.deallocate(True)
            attn = self.ba(q, kc, vc, pos[self.block], self.page_table, None, self.hd**-0.5)
            q.deallocate(True)
            perm = ttnn.permute(attn, (0, 2, 1, 3))
            attn.deallocate(True)
            heads = ttnn.experimental.nlp_concat_heads(perm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            perm.deallocate(True)
            partial = self._linear(heads, L["o"])
            heads.deallocate(True)
            o = _all_reduce(partial, None)  # (the residual's shards)
            partial.deallocate(True)
            o = ttnn.add(o, L["o_b"], output_tensor=o)
            h2 = ttnn.add(h, o, memory_config=self.res_mc, dtype=ttnn.float32)
            h.deallocate(True)
            o.deallocate(True)
            h = h2
            x = norm(h, L["ln2"])
            g, u = self._linear(x, L["g"]), self._linear(x, L["u"])
            x.deallocate(True)
            act = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            g.deallocate(True)
            u.deallocate(True)
            partial = self._linear(act, L["down"])
            act.deallocate(True)
            mlp = _all_reduce(partial, None)
            partial.deallocate(True)
            h2 = ttnn.add(h, mlp, memory_config=self.res_mc, dtype=ttnn.float32)
            h.deallocate(True)
            mlp.deallocate(True)
            h = h2
        x = norm(h, self.norm)
        h.deallocate(True)
        x8 = ttnn.typecast(x, ttnn.bfloat8_b)
        x.deallocate(True)
        logits = ttnn.matmul(x8, self.lm_head, dtype=ttnn.bfloat16)
        x8.deallocate(True)
        out = self.argmax.device(logits)
        logits.deallocate(True)
        return out


class _UserBlockAttention:
    """Block attention for the verify's full-attention layers (b1's dflash/block_attention.py kernels, one instance per
    user): each user's K / V history is read once for its 8 block rows (causal within the block, sinks), not once per
    row as 8 decode-SDPA steps do. `per_program` users run side by side in one program, each on its own
    4 x slices cores (kv head x packed Q tile x history slice); 32 / per_program programs per layer."""

    WAIT_SEM = 8  # the kernels' wait semaphore (count 0: Q and the block's K / V are written by earlier ops)

    def __init__(self, mesh, users, block, n_heads, n_kv, head_dim, block_size, per_program, slices, min_chunks=4, uniform=False, window=0,):  # fmt: skip
        """uniform (the drafter): every row of a block attends [start, start + 7]'s history up to start + 7 (no causal
        mask inside the block, no sinks), within the last `window` positions."""
        from .block_attention import BlockAttention

        self.mesh, self.users, self.block, self.P = mesh, users, block, per_program
        grid = mesh.compute_with_storage_grid_size()
        per = BlockAttention.cores_needed(n_kv, block, slices)
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        assert per * per_program <= len(cores) and users % per_program == 0
        self.inst = [
            BlockAttention(
                mesh,
                block,
                n_heads,
                n_kv,
                head_dim,
                block_size,
                cores[j * per : (j + 1) * per],
                slices,
                q_rows=True,
                min_chunks=min_chunks,
                uniform=uniform,
                window=window,
            )
            for j in range(per_program)
        ]

    def __call__(self, q, k_cache, v_cache, pos_ui, page_table, sinks, scale):
        """q: [1, rows, heads, hd] (row 32 i + u, rotated); pos_ui: INT32 [rows] (8 u + i: start_u + i); page_table:
        INT32 [users, blocks]. Returns the attention output in q's layout."""
        out = ttnn.empty(list(q.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        for p0 in range(0, self.users, self.P):
            kernels, cbs, sems, io = [], [], [], []
            for j, inst in enumerate(self.inst):
                kk, cc, tio = inst.parts(
                    q, k_cache, v_cache, pos_ui, page_table, sinks, scale, wait=(self.WAIT_SEM, 0),
                    user=(p0 + j, self.users), out=out,
                )  # fmt: skip
                kernels += kk
                cbs += cc
                sems += inst.semaphores() + [
                    ttnn.SemaphoreDescriptor(self.WAIT_SEM, ttnn.CoreType.WORKER, inst.cores, 0)
                ]
                io += [t for t in tio[:-1] if all(t is not u for u in io)]
            ttnn.generic_op(io + [out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return out


class BatchedVerify:
    """The target's 256-row verify pass for 32 users on the batched decode model's layers (DecoderLayer.batched_decode):
    attention in 8 batched-decode steps, the MoE on all rows at once, norms on 32-row residual shards as the decode."""

    def __init__(self, model, page_table, kv_cache, users, block, capture_layers):
        self.model = model
        self.mesh = model.mesh_device
        self.users, self.block = users, block
        self.rows = users * block
        self.page_table = page_table
        self.kv_cache = kv_cache
        self.capture_layers = list(capture_layers)
        self.hidden = model.hf_config.hidden_size
        cfg = model.hf_config
        self.top_k, self.num_experts = cfg.num_experts_per_tok, cfg.num_local_experts
        self.swiglu_limit = cfg.swiglu_limit
        self.alpha = 1.702  # ExpertConfig.alpha
        self.moe = []
        for layer in model.layers:
            assert layer.batched_decode, "BatchedVerify runs on the batched decode model (batch 32, dense MoE)"
            dw = layer.mlp.experts.dense_decode_weights
            n_act = dw.gate.shape[-1]
            self.moe.append(
                dict(
                    gate=ttnn.to_memory_config(dw.gate, ttnn.DRAM_MEMORY_CONFIG),
                    up=ttnn.to_memory_config(dw.up, ttnn.DRAM_MEMORY_CONFIG),
                    down=ttnn.to_memory_config(dw.down, ttnn.DRAM_MEMORY_CONFIG),
                    gate_bias=ttnn.slice(dw.gate_bias, [0, 0, 0, 0], [1, 1, 1, n_act]),
                    up_bias=ttnn.slice(dw.up_bias, [0, 0, 0, 0], [1, 1, 1, n_act]),
                    down_bias=dw.down_bias,
                    expand=dw.expand,
                )
            )
        self.ck = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # down [rows, E * I_local] x [E * I_local, hidden]: a long K and a narrow N, which the auto-configured matmul
        # spreads poorly (0.75 ms per layer); 2D over 10 x 8 cores, 16-tile K blocks: 0.27 ms
        n_tiles, m_tiles = self.hidden // TILE, self.rows // TILE
        self.down_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(10, m_tiles),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=1,
            out_block_w=n_tiles // 10,
            per_core_M=1,
            per_core_N=n_tiles // 10,
            transpose_mcast=False,
            fused_activation=None,
        )
        self.down_ck = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,  # FP32 sums over the 23040-long K (4.5x smaller error, same time)
            packer_l1_acc=True,
        )
        # gate / up [rows, hidden] x [hidden, E * I_local] and the routing-weight expand [rows, E] x [E, E * I_local]:
        # 1D over 72 cores (all 8 row tiles x 10 output tiles each; gate / up in 2-tile K blocks, the auto config's
        # sums), outputs width-sharded in L1 ([rows, 320] per core), SwiGLU and the routing multiply there: MoE
        # 1.16 -> 1.00 ms per layer, identical output
        n_act = self.moe[0]["gate"].shape[-1]
        moe_cores = ttnn.CoreCoord(9, 8)
        per_n = n_act // TILE // (moe_cores.x * moe_cores.y)

        def moe_pc(in0_block_w):
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=moe_cores,
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=5,
                out_block_h=m_tiles,
                out_block_w=per_n,
                per_core_M=m_tiles,
                per_core_N=per_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

        self.gate_pc, self.expand_pc = moe_pc(2), moe_pc(1)
        self.moe_mc = ttnn.create_sharded_memory_config(
            (self.rows, per_n * TILE),
            ttnn.CoreGrid(y=moe_cores.y, x=moe_cores.x),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # router: the whole K in one block per output tile (TopKRouter.dense_decode's accumulation order: the same
        # logits), one core per 32-row tile
        k_tiles = self.hidden // TILE
        self.router_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.rows // TILE, 1),
            in0_block_w=k_tiles,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=-(-self.num_experts // TILE),
            per_core_M=1,
            per_core_N=-(-self.num_experts // TILE),
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        self.router_ck = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.argmax = _Argmax(model.lm_head_weight.shape[-1], model.vocab_size)
        from ..fused_decode.config import RESIDUAL_CORES, sdpa_decode_program_config, width_sharded_memory_config

        attn0 = model.layers[0].self_attn
        mc = model.layers[0].self_attn.mesh_config
        self.local_heads = mc.shard_size(attn0.config.num_heads)
        self.local_kv_heads = mc.shard_size(attn0.config.num_kv_heads)
        hd = attn0.config.head_dim
        self.residual_mc = width_sharded_memory_config(self.mesh, self.hidden, RESIDUAL_CORES, rows=self.rows)
        self.head_grid_mc = ttnn.create_sharded_memory_config(
            shape=(TILE, hd),
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # full-attention layers: SDPA on the whole grid (11 x 10; the decode's 8 x 8 holds one user per core), Q read
        # from DRAM (a sharded Q must sit on the SDPA grid's cores): at 4K context attention 22.2 -> 20.9 ms per pass,
        # same output; the 128-token sliding-window layers keep the decode's 8 x 8 config and sharded Q
        self.sdpa_pc = {
            False: ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
                q_chunk_size=0,
                k_chunk_size=sdpa_decode_program_config(False).k_chunk_size,
                exp_approx_mode=False,
            ),
            True: sdpa_decode_program_config(True),
        }
        # the unpadded decode o_proj (batched_decode.BatchedOProj) is DRAM width-sharded: interleaved copies
        self.oproj_w = [
            ttnn.to_memory_config(layer.self_attn.batched_oproj.weight, ttnn.DRAM_MEMORY_CONFIG)
            for layer in model.layers
        ]
        # QKV / o_proj over 256 rows: the batched decode's compute config (HiFi2, BF16 accumulation in L1)
        self.qkv_ck = attn0.batched_qkv.compute_kernel_config
        # Q and K rotated by one op and K / V written by one op (the fused decode's layout: Q heads on cores 0-31, K on
        # 32-63; cos / sin rows for the Q users then the K users)
        from ..model import create_rope_setup

        self.rope_qk = create_rope_setup(
            self.mesh, cfg, max_local_batch_size=users, datatype=ttnn.bfloat16, use_qk_fused=True
        )
        self.trans_qk = self.rope_qk.get_both_trans_mats()["decode"]
        # full-attention layers: block attention (each user's K / V read once for its 8 rows), 8 users per program on 3
        # history slices each (96 cores): 24 layers' attention at 16K context 47.9 -> 18.1 ms (16 x 1: 20.3, 4 x 6:
        # 20.6), at 300 11.3 -> 10.9 ms
        self.block_attn = _UserBlockAttention(
            self.mesh, users, block, self.local_heads, self.local_kv_heads, hd, kv_cache[1][0].shape[2], 8, 3
        )
        # the QKV head input's width-sharded cores (interleaved_to_sharded_partial takes a rectangle)
        self.qkv_grid = ttnn.CoreCoord(10, (self.local_heads + 2 * self.local_kv_heads) // 10)

    def _router(self, layer, x):
        """Dense [rows, experts] routing weights: each row's softmax over its top-k logits, zeros elsewhere (as
        TopKRouter.dense_decode, for any number of rows)."""
        r = layer.mlp.router
        logits = ttnn.linear(
            x,
            r.weight,
            bias=r.bias,
            program_config=self.router_pc,
            compute_kernel_config=self.router_ck,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        values, indices = ttnn.topk(logits, k=self.top_k, dim=-1, sorted=True)
        indices.deallocate(True)
        kth = ttnn.min(values, dim=-1, keepdim=True)
        values.deallocate(True)
        top = ttnn.max(logits, dim=-1, keepdim=True)
        selected = ttnn.ge(logits, kth)
        kth.deallocate(True)
        e = ttnn.exp(ttnn.subtract(logits, top, output_tensor=logits), output_tensor=logits)
        top.deallocate(True)
        e = ttnn.multiply(e, selected, output_tensor=e)
        selected.deallocate(True)
        total = ttnn.sum(e, dim=-1, keepdim=True)
        w = ttnn.divide(e, total)
        total.deallocate(True)
        e.deallocate(True)
        return w

    def _moe(self, li, layer, x):
        """All experts for all rows ([rows, E * I_local] matmuls), unrouted experts zeroed by the routing weights; then
        the TP all-reduce. x: [1, 1, rows, hidden] normed (DRAM)."""
        UWP, U = ttnn.UnaryWithParam, ttnn.UnaryOpType
        # gate / up / expand / SwiGLU in L1 interleaved (the elementwise ops run faster there than on the matmuls'
        # width shards: SwiGLU 228 -> 160 us; BF8 gate / up outputs broke the verify: agreement 0.31)
        mc = ttnn.L1_MEMORY_CONFIG
        gu_dt = ttnn.bfloat16
        W = self.moe[li]
        w = self._router(layer, x)
        x8 = ttnn.typecast(x, ttnn.bfloat8_b)
        # the biases inside the matmuls (with the FP32 down below: the fixed-text agreement of the separate adds)
        gate, up = (
            ttnn.linear(
                x8,
                W[n],
                bias=W[n + "_bias"],
                program_config=self.gate_pc,
                memory_config=mc,
                dtype=gu_dt,
                compute_kernel_config=self.ck,
            )
            for n in ("gate", "up")
        )
        x8.deallocate(True)
        w_cols = ttnn.matmul(w, W["expand"], program_config=self.expand_pc, memory_config=mc, dtype=ttnn.bfloat16)
        limit, alpha = self.swiglu_limit, self.alpha
        act = ttnn.multiply(
            up,
            gate,
            input_tensor_a_activations=[UWP(U.CLAMP_TSS, -limit, limit), UWP(U.ADD_UNARY_SFPU, 1.0)],
            input_tensor_b_activations=[
                UWP(U.CLAMP_TSS, -3.0e38, limit),
                UWP(U.MUL_UNARY_SFPU, alpha),
                UWP(U.SILU),
                UWP(U.MUL_UNARY_SFPU, 1.0 / alpha),
            ],
            memory_config=mc,
        )
        gate.deallocate(True)
        up.deallocate(True)
        # BF8 into the 2D down (DRAM): half the activation reads; the down's sums stay FP32
        act8 = ttnn.multiply(act, w_cols, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        act.deallocate(True)
        w_cols.deallocate(True)
        down = ttnn.matmul(
            act8, W["down"], program_config=self.down_pc, dtype=ttnn.bfloat16, compute_kernel_config=self.down_ck
        )
        act8.deallocate(True)
        bias = ttnn.matmul(w, W["down_bias"], dtype=ttnn.bfloat16)
        w.deallocate(True)
        down = ttnn.add(down, bias, output_tensor=down)
        bias.deallocate(True)
        out = _all_reduce(down)
        down.deallocate(True)
        return out

    def _attention(self, li, layer, res, pos, rot_mats):
        """The layer's attention for all rows: input norm, QKV and o_proj (+ all-reduce) once over the 256 rows (their
        weights read once per pass); per 32-row step the decode attention core: heads, RoPE, paged K/V write at
        positions start_u + i, paged SDPA decode (step i sees steps 0 .. i). res: [rows, hidden] width-sharded residual;
        rot_mats[i]: step i's fused-QK cos / sin. Returns res + attention (same layout)."""
        attn_mod, model = layer.self_attn, self.model
        w, cfg = attn_mod.weights, attn_mod.config
        nh, nkv, hd = self.local_heads, self.local_kv_heads, cfg.head_dim
        normed = layer.input_layernorm.forward_sharded(res)
        x = ttnn.to_memory_config(normed, ttnn.DRAM_MEMORY_CONFIG)
        normed.deallocate(True)
        qkv = ttnn.linear(
            x,
            w.wqkv,
            bias=w.wqkv_bias,
            program_config=_mm_config_256(w.wqkv.shape[-2], w.wqkv.shape[-1]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.qkv_ck,
        )
        x.deallocate(True)
        k_cache, v_cache = self.kv_cache[li]
        block_attn = self.block_attn is not None and not cfg.sliding_window
        outs = []
        for i in range(self.block):
            # step i's 32 rows straight from the interleaved QKV into the width-sharded head input
            part = ttnn.interleaved_to_sharded_partial(
                qkv, self.qkv_grid, [U32, hd], self.block, i, ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
            )  # fmt: skip
            # Q heads on cores 0-31, K on 32-63: Q and K rotated by one op, K and V written by one op
            q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
                part, num_heads=nh, num_kv_heads=nkv, overlap_qk_coregrid=False, memory_config=self.head_grid_mc
            )
            part.deallocate(True)
            cos, sin = rot_mats[i]
            q2, k2 = ttnn.experimental.rotary_embedding_llama_fused_qk(q, k, cos, sin, self.trans_qk)
            q.deallocate(True)
            k.deallocate(True)
            ttnn.experimental.paged_fused_update_cache(
                k_cache, k2, v_cache, v, update_idxs_tensor=pos[i], page_table=self.page_table
            )
            k2.deallocate(True)
            v.deallocate(True)
            if block_attn:  # the 8 steps' Q go to block attention together
                outs.append(q2)
                continue
            if not cfg.sliding_window:
                q3 = ttnn.to_memory_config(q2, ttnn.DRAM_MEMORY_CONFIG)
                q2.deallocate(True)
                q2 = q3
            outs.append(
                ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q2,
                    k_cache,
                    v_cache,
                    cur_pos_tensor=pos[i],
                    sliding_window_size=cfg.sliding_window,
                    attention_sink=w.decode_sinks,
                    page_table_tensor=self.page_table,
                    scale=cfg.scaling,
                    program_config=self.sdpa_pc[bool(cfg.sliding_window)],
                    compute_kernel_config=attn_mod.program_config.get_compute_kernel_config(),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            q2.deallocate(True)
        qkv.deallocate(True)
        # [1, rows, heads, hd] (row 32 i + u) -> [1, heads, rows, hd] -> [1, 1, rows, heads * hd]: one concat, one
        # permute and one head concat per layer (not a reshard, a head concat and a copy per step)
        sdpa = ttnn.concat(outs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in outs:
            t.deallocate(True)
        if block_attn:
            q_all = sdpa
            sdpa = self.block_attn(
                q_all, k_cache, v_cache, pos[self.block], self.page_table, w.decode_sinks, cfg.scaling
            )
            q_all.deallocate(True)
        perm = ttnn.permute(sdpa, (0, 2, 1, 3))
        sdpa.deallocate(True)
        heads = ttnn.experimental.nlp_concat_heads(perm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        perm.deallocate(True)
        op = attn_mod.batched_oproj
        o = ttnn.matmul(
            heads,
            self.oproj_w[li],
            program_config=_mm_config_256(heads.shape[-1], self.oproj_w[li].shape[-1]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.qkv_ck,
        )
        heads.deallocate(True)
        # the decode's BF8 partial sum (the bias inside the matmul with a BF8 output: top-1 vs HF 0.08)
        partial = ttnn.add(o, op.bias, dtype=ttnn.bfloat8_b)
        o.deallocate(True)
        summed_s = _all_reduce(partial, self.residual_mc)
        partial.deallocate(True)
        out = ttnn.add(res, summed_s, memory_config=self.residual_mc, dtype=ttnn.bfloat16)
        summed_s.deallocate(True)
        return out

    def _sharded_rows(self, x, norm, residual_mc):
        """Norm each 32-row tile of x on its width-sharded residual shards (the batched decode's norm)."""
        parts = []
        for i in range(self.block):
            xi = ttnn.slice(x, [0, 0, i * U32, 0], [1, 1, (i + 1) * U32, self.hidden])
            res = ttnn.to_memory_config(xi, residual_mc, dtype=ttnn.bfloat16)
            xi.deallocate(True)
            n = norm.forward_sharded(res)
            res.deallocate(True)
            parts.append(ttnn.to_memory_config(n, ttnn.DRAM_MEMORY_CONFIG))
            n.deallocate(True)
        out = ttnn.concat(parts, dim=2)
        for t in parts:
            t.deallocate(True)
        return out

    def forward(self, tokens, pos, rot_idx):
        """tokens: device [1, rows] UINT32 (row 32 i + u). pos[i] / rot_idx[i]: step i's [32] INT32 positions and
        [1, 32] UINT32 RoPE indices. Returns (per-device argmax of every row, device features [1, 1, rows, 5 hidden]).
        """
        _TPAllReduce.reset()
        model = self.model
        emb = ttnn.embedding(tokens, model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(emb, [1, 1, self.rows, self.hidden])
        feats = []
        res = ttnn.to_memory_config(x, self.residual_mc, dtype=ttnn.bfloat16)
        x.deallocate(True)
        rot_mats = []  # every layer's RoPE rows of step i (once per pass, not per layer)
        for r in rot_idx:
            r2 = ttnn.concat([r, r], dim=1)  # the Q users' rows, then the K users'
            rot_mats.append(self.rope_qk.get_rot_mats(r2))
            r2.deallocate(True)
        for li, layer in enumerate(model.layers):
            res2 = self._attention(li, layer, res, pos, rot_mats)
            res.deallocate(True)
            n2 = layer.post_attention_layernorm.forward_sharded(res2)
            normed = ttnn.to_memory_config(n2, ttnn.DRAM_MEMORY_CONFIG)
            n2.deallocate(True)
            x = ttnn.to_memory_config(res2, ttnn.DRAM_MEMORY_CONFIG)
            res2.deallocate(True)
            moe = self._moe(li, layer, normed)
            normed.deallocate(True)
            x2 = ttnn.add(x, moe, dtype=ttnn.bfloat16)
            x.deallocate(True)
            moe.deallocate(True)
            if li in self.capture_layers:
                feats.append(ttnn.clone(x2))
            res = ttnn.to_memory_config(x2, self.residual_mc)
            x = x2
            if li < len(model.layers) - 1:
                x2.deallocate(True)
        for t in (t for pair in rot_mats for t in pair):
            t.deallocate(True)
        features = ttnn.concat(feats, dim=-1)
        for t in feats:
            t.deallocate(True)
        normed_s = model.norm.forward_sharded(res)
        res.deallocate(True)
        x.deallocate(True)
        normed = ttnn.to_memory_config(normed_s, ttnn.DRAM_MEMORY_CONFIG)
        normed_s.deallocate(True)
        normed8 = ttnn.typecast(normed, ttnn.bfloat8_b)  # (to_memory_config to the same layout returns its input)
        normed.deallocate(True)
        logits8 = ttnn.matmul(normed8, model.lm_head_weight, dtype=ttnn.bfloat8_b)  # as the batched decode's LM head
        normed8.deallocate(True)
        logits = ttnn.typecast(logits8, ttnn.bfloat16)
        logits8.deallocate(True)
        out = self.argmax.device(logits)
        logits.deallocate(True)
        return out, features


class BatchedDFlashRunner:
    """Greedy DFlash for `users` sequences (each prefilled into its own KV slot) on the batched decode model."""

    def __init__(self, generator, model, page_table, kv_cache, users=32, window=1024, eos_ids=(200002, 200012)):
        from .capture import TargetCapture

        self.generator, self.model, self.kv_cache = generator, model, kv_cache
        mesh = model.mesh_device
        self.mesh = mesh
        self.users = users
        self.page_table_host = page_table
        self.page_table = _replicate(mesh, page_table, ttnn.int32)
        self.V = model.vocab_size
        self.eos_ids = set(eos_ids)
        max_pos = page_table.shape[-1] * 64
        self.draft = BatchedDraft(mesh, model, max_pos, users, window=window)
        self.block = B = self.draft.block
        self.rows = users * B
        self.capture = TargetCapture(model, self.draft.target_layer_ids)
        kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
        self.verify = BatchedVerify(model, self.page_table, kv, users, B, self.draft.target_layer_ids)
        # persistent per-round inputs (the traces' inputs)
        z = torch.zeros(users, dtype=torch.int32)
        self.d_tokens = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        self.v_tokens = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        # per-step positions, then (entry B) every row's position user-major (8 u + i: start_u + i, block attention)
        self.pos = [_replicate(mesh, z, ttnn.int32) for _ in range(B)] + [
            _replicate(mesh, torch.zeros(self.rows, dtype=torch.int32), ttnn.int32)
        ]
        self.rot = [_replicate(mesh, z.reshape(1, users), ttnn.uint32) for _ in range(B)]
        self.bound = _replicate(mesh, z, ttnn.int32)
        self.pos_rows = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        # the per-round inputs travel as two packed host writes (int32: per-step positions, user-major positions, the
        # draft bound; uint32: the draft tokens, RoPE indices, per-row positions), unpacked on device into the tensors
        # above at the start of a pass (one host-to-device copy each instead of 19)
        U = users
        self.unpack_i = [(t, [U]) for t in self.pos[:B]] + [(self.pos[B], [self.rows]), (self.bound, [U])]
        self.unpack_u = (
            [(self.d_tokens, [1, self.rows])] + [(t, [1, U]) for t in self.rot] + [(self.pos_rows, [1, self.rows])]
        )
        n_i = sum(math.prod(shape) for _, shape in self.unpack_i)
        n_u = sum(math.prod(shape) for _, shape in self.unpack_u)
        self.in_i = _replicate(mesh, torch.zeros(1, n_i, dtype=torch.int32), ttnn.int32)
        self.in_u = _replicate(mesh, torch.zeros(1, n_u, dtype=torch.int32), ttnn.uint32)
        self.one_trace = True
        # the next round's inputs computed on device at the end of a round (acceptance on device): every round is
        # queued before the previous round's result is read
        self.queue_rounds = True
        put = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self.iota8 = put(torch.arange(B).float().reshape(1, 1, B, 1).expand(1, 1, B, U).contiguous())  # row i: i
        self.mask_rest = put(torch.full((1, 1, 1, self.rows - U), float(self.draft.mask_token_id)))
        self.limit = put(torch.zeros(1, 1, 1, U))  # each user's last start position (its token budget)
        self.first_step = ttnn.from_torch(
            (torch.arange(self.rows) < users).float().reshape(1, 1, 1, self.rows),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.traces = None

    def prefill(self, prompts):
        firsts = []
        for u, p in enumerate(prompts):
            logits = self.generator.prefill_forward_text(
                torch.tensor([p]),
                # the generator takes the page table rows of the prompts it is given (row 0 for a one-prompt call):
                # this user's row (the whole table wrote every user's K/V into user 0's blocks)
                page_table=self.page_table_host[u : u + 1],
                kv_cache=self.kv_cache,
                prompt_lens=[len(p)],
                empty_slots=[u],
                enable_trace=False,
                warmup_prefill=False,
            )
            firsts.append(int(logits.reshape(-1)[: self.V].float().argmax()))
            self.draft.add_prompt_context(self.capture.prefill_features(len(p)), user=u)
        return firsts

    # -- passes (eager, or captured once and replayed) ----------------------------------------------------------------
    def _rot_mats(self):
        return self.draft.rows_cos_sin(self.pos_rows)

    def _unpack(self):
        """The packed per-round inputs into their tensors (slices of the two packed rows)."""
        for pack, parts in ((self.in_i, self.unpack_i), (self.in_u, self.unpack_u)):
            off = 0
            for dst, shape in parts:
                n = math.prod(shape)
                part = ttnn.reshape(ttnn.slice(pack, [0, off], [1, off + n]), shape)
                ttnn.copy(part, dst)
                part.deallocate(True)
                off += n

    def _draft_pass(self):
        self._unpack()
        return self.draft.draft(self.d_tokens, self.pos, self._rot_mats(), self.bound)

    def _verify_pass(self):
        self._unpack()  # (the step-by-step path may run verify without a draft pass)
        out, features = self.verify.forward(self.v_tokens, self.pos, self.rot)
        self.draft.context(features, self.pos, self._rot_mats())
        features.deallocate(True)
        return out

    def _round_pass(self):
        """A whole round on the device: draft, the devices' argmax combined, the verify block [last token, 7 drafts]
        built from it, verify + context. Returns the draft and verify token ids ([1, 1, rows, 1] FLOAT32)."""
        d_out = self._draft_pass()
        d_tok = self.draft.argmax.combine(d_out, self.mesh)
        for t in d_out:
            t.deallocate(True)
        row = ttnn.transpose(d_tok, -2, -1)  # [1, 1, 1, rows]
        lasts = ttnn.typecast(ttnn.to_layout(self.d_tokens, ttnn.TILE_LAYOUT), ttnn.float32)  # rows 0-31: last tokens
        vt = ttnn.where(self.first_step, ttnn.reshape(lasts, [1, 1, 1, self.rows]), row)
        row.deallocate(True)
        lasts.deallocate(True)
        v_tokens = ttnn.reshape(ttnn.to_layout(ttnn.typecast(vt, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, self.rows])
        vt.deallocate(True)
        out, features = self.verify.forward(v_tokens, self.pos, self.rot)
        v_tokens.deallocate(True)
        self.draft.context(features, self.pos, self._rot_mats())
        features.deallocate(True)
        v_tok = self.verify.argmax.combine(out, self.mesh)
        for t in out:
            t.deallocate(True)
        if self.queue_rounds:
            self._next_inputs(d_tok, v_tok)
        return d_tok, v_tok

    def _next_inputs(self, d_tok, v_tok):
        """The acceptance on device: per user the drafts up to the first mismatch with the target (k), the next start
        start + k + 1 (capped at the user's budget: finished users repeat their block), the target's token at row k as
        the next last token; the next round's packed inputs written from them (FLOAT32 math: ids < 2^24 are exact)."""
        U, B = self.users, self.block
        d8, v8 = (ttnn.reshape(ttnn.transpose(t, -2, -1), [1, 1, B, U]) for t in (d_tok, v_tok))  # [row i, user u]
        m = ttnn.eq(ttnn.slice(d8, [0, 0, 1, 0], [1, 1, B, U]), ttnn.slice(v8, [0, 0, 0, 0], [1, 1, B - 1, U]))
        run = ttnn.slice(m, [0, 0, 0, 0], [1, 1, 1, U])
        k = ttnn.clone(run)
        for i in range(1, B - 1):  # leading matches: the running product of the row matches
            run = ttnn.multiply(run, ttnn.slice(m, [0, 0, i, 0], [1, 1, i + 1, U]), output_tensor=run)
            k = ttnn.add(k, run, output_tensor=k)
        run.deallocate(True)
        m.deallocate(True)
        last = ttnn.sum(ttnn.multiply(ttnn.eq(self.iota8, k), v8), dim=2, keepdim=True)  # the target's token at row k
        bound = ttnn.slice(self.in_i, [0, 2 * self.rows], [1, 2 * self.rows + U])  # start + B - 1 (int32)
        start = ttnn.typecast(ttnn.to_layout(ttnn.reshape(bound, [1, 1, 1, U]), ttnn.TILE_LAYOUT), ttnn.float32)
        start = ttnn.minimum(ttnn.add(start, k, output_tensor=start) - float(B - 2), self.limit)  # + k + 1
        steps = ttnn.add(self.iota8, start)  # [i, u]: start_u + i
        rows_i = ttnn.reshape(steps, [1, 1, 1, self.rows])
        rows_u = ttnn.reshape(ttnn.transpose(steps, -2, -1), [1, 1, 1, self.rows])
        vi = ttnn.concat([rows_i, rows_u, start + float(B - 1)], dim=-1)
        vu = ttnn.concat([last, self.mask_rest, rows_i, rows_i], dim=-1)
        for src, dst, dtype in ((vi, self.in_i, ttnn.int32), (vu, self.in_u, ttnn.uint32)):
            t = ttnn.reshape(ttnn.to_layout(ttnn.typecast(src, dtype), ttnn.ROW_MAJOR_LAYOUT), list(dst.shape))
            ttnn.copy(t, dst)
            t.deallocate(True)

    def capture_traces(self):
        """One trace for the draft pass, one for verify + context (the step-by-step passes), and one for the whole
        round (generate; one trace per round: no host round trip between draft and verify, b1 row 41)."""
        mesh = self.mesh
        self.release_traces()
        for t in list(self._draft_pass()) + list(self._verify_pass()) + list(self._round_pass()):
            t.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.draft_out = self._draft_pass()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        tid2 = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.verify_out = self._verify_pass()
        ttnn.end_trace_capture(mesh, tid2, cq_id=0)
        tid3 = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.round_out = self._round_pass()
        ttnn.end_trace_capture(mesh, tid3, cq_id=0)
        self.traces = (tid, tid2, tid3)

    def release_traces(self):
        if self.traces is not None:
            ttnn.synchronize_device(self.mesh)
            for t in self.traces:
                ttnn.release_trace(self.mesh, t)
        self.traces = None

    def _set_inputs(self, starts, lasts):
        mesh, U, B = self.mesh, self.users, self.block
        toks = torch.full((1, self.rows), self.draft.mask_token_id, dtype=torch.int32)
        toks[0, :U] = torch.tensor(lasts, dtype=torch.int32)
        st = torch.tensor(starts, dtype=torch.int32)
        steps = st[None] + torch.arange(B)[:, None]  # [step i, user u]: start_u + i
        vi = torch.cat([steps.reshape(-1), steps.T.reshape(-1), st + B - 1])
        vu = torch.cat([toks.reshape(-1), steps.reshape(-1), steps.reshape(-1)])
        _copy(mesh, self.in_i, vi.reshape(1, -1), ttnn.int32)
        _copy(mesh, self.in_u, vu.reshape(1, -1), ttnn.uint32)

    def _run(self, which):
        if self.traces is None:
            return (self._draft_pass if which == 0 else self._verify_pass)(), True
        ttnn.execute_trace(self.mesh, self.traces[which], cq_id=0, blocking=False)
        return (self.draft_out if which == 0 else self.verify_out), False

    def _accept(self, ids, accept, done, dr, post):
        """Each user's drafts up to the first mismatch with the target's tokens, plus the target's next token."""
        U, B = self.users, self.block
        for u in range(U):
            d = [dr[i * U + u] for i in range(1, B)]
            p = [post[i * U + u] for i in range(B)]
            k = 0
            while k < B - 1 and d[k] == p[k]:
                k += 1
            if not done(u):
                ids[u] += d[:k] + [p[k]]
                accept[u].append(k + 1)

    def _enqueue_round(self):
        """Queue one traced round and non-blocking reads of its token ids. Returns (host tensors, event)."""
        ttnn.execute_trace(self.mesh, self.traces[2], cq_id=0, blocking=False)
        host = [t.cpu(blocking=False) for t in self.round_out]
        return host, ttnn.record_event(self.mesh, 0)

    def _generate_queued(self, ids, lens, accept, done, times, max_new_tokens):
        """Traced rounds with the acceptance on device (b1's queued rounds): round r + 1 is queued before round r's
        result is read (the device writes its inputs), so the host's read and bookkeeping overlap the next round."""
        U = self.users
        limit = torch.tensor([lens[u] + max_new_tokens for u in range(U)], dtype=torch.float32).reshape(1, 1, 1, U)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(limit, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)),
            self.limit,
        )  # fmt: skip
        t = time.perf_counter()
        pending = [self._enqueue_round()]
        while pending:
            # a round adds at most B tokens: queue the next one unless every user may finish with this one
            if len(pending) == 1 and any(len(ids[u]) - lens[u] + self.block < max_new_tokens for u in range(U)):
                pending.append(self._enqueue_round())
            host, event = pending.pop(0)
            ttnn.event_synchronize(event)
            dr, post = (ttnn.to_torch(ttnn.get_device_tensors(h)[0]).reshape(-1).long().tolist() for h in host)
            self._accept(ids, accept, done, dr, post)
            now = time.perf_counter()
            times.append((now - t, now - t, 0.0, 0.0))
            t = now
            if all(done(u) for u in range(U)):
                break
            if not pending:
                pending.append(self._enqueue_round())
        if pending:  # a round queued past the end: let it finish
            ttnn.synchronize_device(self.mesh)

    def generate(self, prompts, max_new_tokens, firsts):
        """Returns (per-user new tokens, per-user accepted counts, per-round seconds)."""
        mesh = self.mesh
        U, B = self.users, self.block
        lens = [len(p) for p in prompts]
        ids = [list(p) + [f] for p, f in zip(prompts, firsts)]
        accept = [[] for _ in range(U)]
        times = []

        def done(u):
            return len(ids[u]) - lens[u] >= max_new_tokens or bool(set(ids[u][lens[u] :]) & self.eos_ids)

        while not all(done(u) for u in range(U)):
            t0 = time.perf_counter()
            starts = [len(ids[u]) - 1 for u in range(U)]
            lasts = [ids[u][-1] for u in range(U)]
            self._set_inputs(starts, lasts)
            t1 = time.perf_counter()
            if self.traces is not None and self.one_trace and self.queue_rounds:
                self._generate_queued(ids, lens, accept, done, times, max_new_tokens)
                break
            if self.traces is not None and self.one_trace:  # one trace per round
                ttnn.execute_trace(mesh, self.traces[2], cq_id=0, blocking=False)
                dr, post = (_Argmax.read(t) for t in self.round_out)
                t2 = t3 = t4 = time.perf_counter()
                self._accept(ids, accept, done, dr, post)
                t5 = time.perf_counter()
                times.append((t5 - t0, t2 - t1, 0.0, (t1 - t0) + (t5 - t4)))  # round, device (draft + verify), -, host
                continue
            d_out, owned = self._run(0)
            dr = self.draft.argmax.host(d_out)
            t2 = time.perf_counter()
            if owned:
                for t in d_out:
                    t.deallocate(True)
            vt = torch.tensor(lasts + dr[U:], dtype=torch.int32).reshape(1, self.rows)
            _copy(mesh, self.v_tokens, vt, ttnn.uint32)
            t3 = time.perf_counter()
            v_out, owned = self._run(1)
            post = self.verify.argmax.host(v_out)
            t4 = time.perf_counter()
            if owned:
                for t in v_out:
                    t.deallocate(True)
            self._accept(ids, accept, done, dr, post)
            t5 = time.perf_counter()
            times.append((t5 - t0, t2 - t1, t4 - t3, (t1 - t0) + (t3 - t2) + (t5 - t4)))  # round, draft, verify, host
        return [ids[u][lens[u] : lens[u] + max_new_tokens] for u in range(U)], accept, times
