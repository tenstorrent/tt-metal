# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash verify of the gpt-oss target in one pass: the M = 8 block rows [last token, d1 .. d7] of one user at
positions p .. p + 7 through all layers at once, with the fused-decode streamed ops in their multi-row form
(fused_decode/: the residual stream in the row-tile layout of boundary.py, every weight read once for the M rows):

  embeddings -> entry boundary (row-tile residual + input norm)
  per layer: one program: QKV (+ RoPE + K / V cache write of the M rows) + attention of the M rows:
               sliding-window layers: row attention (dflash/row_attention.py: one core per (row, kv head));
               full-attention layers: block attention (dflash/block_attention.py: the rows share every K / V read)
             -> o_proj with the boundary all-reduce send
             -> one program: boundary (all-reduce + residual + post-attention norm) + router (per-row top-4 + union)
                + union-expert gate|up + down with the next boundary's send (dflash/moe_fused.py)
  LM head: two stages (BFP4 prescreen + BF8 rescore of each row's candidates, dflash/lm_rescore.py), then the
  on-device argmax merge (dflash/argmax.py): the round's acceptance and the next round's inputs.

The target's argmax for row r is its prediction for position p + r + 1; the K / V rows of rejected drafts sit past the
accepted length and are overwritten by the next block (attention reads positions <= each row's own). Layer outputs
1, 6, 11, 16, 21 (the drafter's context) are copied into the drafter's context buffer per round.
"""

import os

import torch

import ttnn

from ..experts.weights import load_decode_down_bal
from ..fused_decode.boundary import (
    KERNEL_DIR,
    DecodeBoundary,
    PendingBoundary,
    flat_memory_config,
    flat_shape,
    rowtile_pages,
)
from ..fused_decode.config import (
    GATE_UP_STREAM_READERS,
    LM_HEAD_STREAM_PREFETCH,
    OPROJ_DECODE_WEIGHT_DTYPE,
    QKV_DECODE_WEIGHT_DTYPE,
    QKV_STREAM_READERS,
    ROUTER_DECODE_WEIGHT_DTYPE,
)
from ..fused_decode.stream import LinearStream, QKVRopeCache, rope_rotation_tiles
from ..fused_decode.union import act_buffer, union_buffers
from .argmax import ArgmaxGather
from .block_attention import BlockAttention, free_cores
from .lm_head import padded_lm_head
from .lm_rescore import LMRescore
from .moe_fused import MoEFused
from .row_attention import RowAttention

TILE = ttnn.TILE_SIZE
BOUNDARY_HELPERS = 4  # the multi-core boundary's helper cores (grid row 2: no streamed-op worker sits there)
BOUNDARY_LINKS = 2  # one fabric sender per link (4093 -> 4067 us verify pass at ISL 256)
OPROJ_READERS = 3  # o_proj readers per bank (5.020 -> 4.997 ms round at ISL 256 vs 2)
DOWN_READERS = 3
BA_SLICES = 24  # block attention slices per (kv head, tile group), capped by the free cores
# block attention leaves (1..3, 1) free: leaders whose K / V reads starved for 100+ us in some layers at ISL 128K
# (block attention 228 -> 340-385 us there; 233 us flat without them, 23 slices instead of 24)
BA_EXCLUDE = [(1, 1), (2, 1), (3, 1)]
LM_READERS = 2  # the BF8 head's padded layout (the rescore reads it)
LM_PRESCREEN = 8  # each row's best candidates per stage-1 core
LM_PRESCREEN_READERS = 3  # the BFP4 stage-1 head's readers per bank
QKV_WAIT_SEM = 3  # the QKV writers -> the attention cores (Q and the block's K / V rows written)


class FusedVerify:
    def __init__(self, model, page_table, kv_cache, rows=8, capture_layers=(1, 6, 11, 16, 21)):
        """model: the fused-decode gpt-oss Model (one user); page_table: its [1, blocks] INT32 page table (torch);
        kv_cache: per layer [k_cache, v_cache]."""
        self.model = model
        mesh = self.mesh = model.mesh_device
        cfg = model.hf_config
        self.M = M = rows
        self.H = H = cfg.hidden_size
        self.layers = model.layers
        self.kv_cache = kv_cache
        self.capture_layers = list(capture_layers)
        attn0 = self.layers[0].self_attn.config
        tp = mesh.shape[1]
        self.nq, self.nkv, self.hd = attn0.num_heads // tp, attn0.num_kv_heads // tp, attn0.head_dim
        q_cols, k_cols = self.nq * self.hd // TILE, self.nkv * self.hd // TILE
        self.boundary = DecodeBoundary(
            mesh,
            H,
            cfg.rms_norm_eps,
            cluster_axis=1,
            rows=M,
            links=BOUNDARY_LINKS,
            ccl="fabric",
            helpers=[(x, 2) for x in range(BOUNDARY_HELPERS)],
        )
        b = self.boundary
        self.gamma_in = [b.flat_gamma(L.input_layernorm) for L in self.layers]
        self.gamma_post = [b.flat_gamma(L.post_attention_layernorm) for L in self.layers]
        self.gamma_final = b.flat_gamma(model.norm)

        self.qkv = LinearStream(
            mesh,
            H // TILE,
            (self.nq + 2 * self.nkv) * self.hd,
            QKV_DECODE_WEIGHT_DTYPE,
            readers=QKV_STREAM_READERS,
            out_mode=3,
            heads=(q_cols, k_cols),
            rows=M,
        )
        self.o_proj = LinearStream(
            mesh,
            self.nq * self.hd // TILE,
            H,
            OPROJ_DECODE_WEIGHT_DTYPE,
            readers=OPROJ_READERS,
            x_pages=self.hd // TILE,
            out_mode=5,
            rows=M,
            gather_split=True,
        )
        mlp0 = self.layers[0].mlp
        self.top_k, self.n_experts = mlp0.router.top_k, mlp0.router.num_experts
        self.router = LinearStream(
            mesh,
            H // TILE,
            self.n_experts,
            ROUTER_DECODE_WEIGHT_DTYPE,
            readers=1,
            out_mode=4,
            out_page_bytes=128,
            heads=(self.top_k, self.n_experts),
            rows=M,
            # off the DRAM bank rows (the MoE's readers take them, dflash/moe_fused.py)
            exclude=[(x, 9) for x in range(mesh.compute_with_storage_grid_size().x)],
        )
        ex = mlp0.experts
        inter_pad = ex.decode_weights.intermediate_padded
        # the QKV program's attention cores: clear of the QKV writers and the boundary cores
        taken = {(c.x, c.y) for c in ttnn.corerange_to_cores(self.qkv.cores, row_wise=True)}
        taken |= {(c.x, c.y) for c in ttnn.corerange_to_cores(b.cores, row_wise=True)}
        taken |= {(h.x, h.y) for h in b.helpers}
        grid = mesh.compute_with_storage_grid_size()
        # sliding-window layers: row attention, 8 x 2 cores; each core holds one row's window: the window's positions
        # span at most window / 32 + 1 chunks of 32
        window = max(L.self_attn.config.sliding_window or 0 for L in self.layers)
        free = [
            (x0, y0)
            for y0 in range(grid.y - 1)
            for x0 in range(grid.x - 7)
            if not any((x0 + i % 8, y0 + i // 8) in taken for i in range(M * self.nkv))
        ]
        self.row_attn = RowAttention(
            mesh,
            M,
            self.nq,
            self.nkv,
            self.hd,
            kv_cache[0][0].shape[2],
            max_chunks=-(-window // TILE) + 1,
            cores_start=(0, 5) if (0, 5) in free else free[0],
        )
        # full-attention layers: block attention
        full = [i for i, L in enumerate(self.layers) if not L.self_attn.config.sliding_window]
        ba_taken = taken | set(BA_EXCLUDE)
        per_slice = BlockAttention.cores_needed(self.nkv, M, 1)
        S = min(BA_SLICES, (grid.x * grid.y - len(ba_taken)) // per_slice)
        self.block_attn = BlockAttention(
            mesh,
            M,
            self.nq,
            self.nkv,
            self.hd,
            kv_cache[full[0]][0].shape[2],
            free_cores(mesh, ba_taken, BlockAttention.cores_needed(self.nkv, M, S)),
            S,
        )
        # the MoE's down weights in the balanced layout (experts/weights.py load_decode_down_bal: one more copy per
        # layer, ~57 MB per device)
        self.down_bal, bal_plan = {}, None
        for li, L in enumerate(self.layers):
            e = L.mlp.experts
            self.down_bal[li], bal_plan = load_decode_down_bal(
                mesh, e.config, e.mesh_config, e.decode_weights, e.weight_dtype, e.tensor_cache_path
            )
        self.moe_fused = MoEFused(
            mesh,
            b,
            self.router,
            H,
            inter_pad,
            ex.config.swiglu_limit,
            ex.config.alpha,
            M,
            gate_up_readers=GATE_UP_STREAM_READERS,
            down_readers=DOWN_READERS,
            real_inter=cfg.intermediate_size // tp,
            down_bal_plan=bal_plan,
        )
        # LM head, two stages: the BFP4 head keeps each row's LM_PRESCREEN best columns per core; the device's best
        # per row are rescored with the BF8 weights (laid out round-robin over the banks for the rescore's reads)
        term = model.decode_terminal
        self.vd = term.vd
        cache_dir = os.environ.get("TT_CACHE_PATH")
        self.lm_w8, lm_n = padded_lm_head(mesh, term, LM_READERS, cache_dir, round_robin=True)
        self.lm_weights, n4 = padded_lm_head(mesh, term, LM_PRESCREEN_READERS, cache_dir, dtype=ttnn.bfloat4_b)
        # both streams keep the vocab's column order (only the padding past vd differs): same column indices
        assert n4 >= self.vd and lm_n >= self.vd
        self.lm_n4 = n4
        self.lm_head = LinearStream(
            mesh,
            H // TILE,
            n4,
            ttnn.bfloat4_b,
            readers=LM_PRESCREEN_READERS,
            out_mode=6,
            prefetch_cols=LM_HEAD_STREAM_PREFETCH,
            rows=M,
            out_page_bytes=8 * LM_PRESCREEN,  # the top-K lists: one page per (core, row)
        )
        self.rescore = LMRescore(mesh, M, LM_PRESCREEN, len(self.lm_head.assignments), H)

        rep = ttnn.ReplicateTensorToMesh(mesh)
        dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
        # [1, M, heads, head_dim] (tile-padded to 32 head rows): the QKV output row attention reads
        self.q = ttnn.from_torch(
            torch.zeros(1, M, self.nq, self.hd),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=l1,  # (row attention's Q reads skip DRAM latency)
            mesh_mapper=rep,
        )
        self.pages = rowtile_pages(H, M)
        self.partial = ttnn.from_torch(
            torch.zeros(flat_shape(H, self.pages)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=flat_memory_config(mesh, H, self.pages),
            mesh_mapper=rep,
        )
        self.union, self.weights = union_buffers(mesh, M)
        self.act = act_buffer(mesh, inter_pad, M)
        self.lists1 = ttnn.from_torch(  # the BFP4 stage's per-core top-K lists
            torch.zeros(len(self.lm_head.assignments) * M, LM_PRESCREEN * 2, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=l1,
            mesh_mapper=rep,
        )
        self.lists = ttnn.from_torch(
            torch.zeros(1, LM_PRESCREEN * M * 2, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=l1,
            mesh_mapper=rep,
        )
        # inputs: token ids (written by the drafter's argmax merge), row positions, the rows' page table (user 0's row
        # repeated)
        self.tokens = ttnn.from_torch(
            torch.zeros(1, M, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=l1,
            mesh_mapper=rep,
        )
        self.pos = ttnn.from_torch(
            torch.zeros(M, dtype=torch.int32), dtype=ttnn.int32, device=mesh, memory_config=dram, mesh_mapper=rep
        )
        pt = page_table.reshape(1, -1).int()
        self.page_table = ttnn.from_torch(
            pt.repeat(M, 1), dtype=ttnn.int32, device=mesh, memory_config=dram, mesh_mapper=rep
        )
        self.rot = rope_rotation_tiles(mesh)
        self.cos_table, self.sin_table = model.rope_setup.cos_matrix, model.rope_setup.sin_matrix
        # the on-device merge of the rescored lists (dflash/argmax.py): result = [block tokens, 8 target argmaxes]
        # (UINT32 [1, 2 M], read by read_result)
        self.argmax_gather = ArgmaxGather(mesh, M, LM_PRESCREEN, self.vd, kind=1)
        self.result = ttnn.from_torch(
            torch.zeros(1, 2 * M, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=l1,
            mesh_mapper=rep,
        )
        # set by the drafter (dflash/runner.py): its context buffer and its per-round input tensors
        self.ctx_buffer = self.accept = None

    def set_context_buffer(self, buf, split):
        """The drafter's per-device row-tile context buffer (FusedDraft.ctx_buf): device d receives global context tiles
        split[d] = [s, e) of the layer-major 5 x hidden features (whole [M, 32] tiles, one L1 copy per layer)."""
        self.ctx_buffer = (buf, split)

    def set_accept(self, mask_id, inputs):
        """inputs = [verify positions, drafter tokens, positions, RoPE rows, attention bound, context positions, context
        RoPE rows]: the argmax merge computes each round's acceptance and writes the next round's inputs on device
        (dflash/argmax.py ACCEPT)."""
        self.accept = (mask_id, inputs)

    def read_result(self):
        """Host: (block tokens, target argmaxes) from device 0's result buffer."""
        r = ttnn.to_torch(ttnn.get_device_tensors(self.result)[0]).reshape(-1).long().tolist()
        return r[: self.M], r[self.M :]

    def set_positions(self, start):
        """The first round's row positions start .. start + M - 1 (later rounds: written on device)."""
        p = ttnn.from_torch(
            torch.arange(start, start + self.M, dtype=torch.int32),
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(p, self.pos)

    def forward(self):
        """The verify pass on the current inputs: the argmax merge writes [block, target argmaxes] into self.result and
        the next round's inputs; the captured layer outputs go to the drafter's context buffer."""
        assert self.ctx_buffer is not None and self.accept is not None, "set_context_buffer / set_accept first"
        M, b = self.M, self.boundary
        ba = self.block_attn
        emb = ttnn.embedding(
            self.tokens, self.model.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        emb = ttnn.reshape(emb, (1, 1, M, self.H))
        pending = PendingBoundary(b, emb, self.gamma_in[0], entry=True)
        for i, layer in enumerate(self.layers):
            attn = layer.self_attn
            k_cache, v_cache = self.kv_cache[i]
            rk = QKVRopeCache.rows(
                self.rot, self.cos_table, self.sin_table, self.pos, self.page_table, k_cache, v_cache
            )
            sliding = bool(attn.config.sliding_window)
            wait = (QKV_WAIT_SEM, len(self.qkv.assignments))
            if sliding:  # row attention fused into the QKV program
                kern, cbs, io = self.row_attn.parts(
                    self.q,
                    k_cache,
                    v_cache,
                    self.pos,
                    self.page_table,
                    attn.weights.decode_sinks,
                    attn.config.scaling,
                    window=attn.config.sliding_window,
                    wait=wait,
                )
                fuse = (kern, cbs, io, self.row_attn.noc_cores(), QKV_WAIT_SEM, [])
                self.qkv(pending, attn.weights.wqkv_stream, (self.q, self.q, self.q), rope_kv=rk, fuse=fuse)
                sdpa = self.row_attn.out
            else:  # block attention fused into the QKV program (Q in its packed layout)
                kern, cbs, io = ba.parts(
                    ba.q,
                    k_cache,
                    v_cache,
                    self.pos,
                    self.page_table,
                    attn.weights.decode_sinks,
                    attn.config.scaling,
                    wait=wait,
                )
                fuse = (kern, cbs, io, ba.notify_order(), QKV_WAIT_SEM, ba.semaphores())
                self.qkv(pending, attn.weights.wqkv_stream, (ba.q, ba.q, ba.q), rope_kv=rk, spec_tt=ba.Tt, fuse=fuse)
                sdpa = ba.out
            res = pending.residual_out
            pending.x.deallocate(True)
            pending.residual.deallocate(True)
            if i - 1 in self.capture_layers:  # layer i - 1's output = this layer's input residual
                self._capture(i - 1, res)
            # (the attention outputs are persistent: reused by every layer)
            self.o_proj(
                sdpa,
                attn.weights.o_proj_stream,
                self.partial,
                send=(b, "attn"),
                spec_tt=None if sliding else ba.Tt,
            )
            pending = PendingBoundary(b, res, self.gamma_post[i], self.partial, "attn", sent=True)
            dw = layer.mlp.experts.decode_weights
            self.moe_fused(
                pending,
                layer.mlp.router.weight_stream,
                self.union,
                self.weights,
                dw.gate_up_stream,
                self.act,
                self.down_bal[i],
                self.partial,
                layer=i,
            )
            res = pending.residual_out
            pending.residual.deallocate(True)
            pending.x.deallocate(True)
            gamma = self.gamma_final if i + 1 == len(self.layers) else self.gamma_in[i + 1]
            pending = PendingBoundary(b, res, gamma, self.partial, "moe", sent=True)
        self.lm_head(pending, self.lm_weights, self.lists1, extra_defines=[("LM_TOPK", str(LM_PRESCREEN))])
        self.rescore(self.lists1, pending.x, self.lm_w8, self.lists)
        self.argmax_gather(self.lists, self.tokens, self.result, accept=self.accept)
        pending.x.deallocate(True)
        pending.residual.deallocate(True)
        pending.residual_out.deallocate(True)

    def _capture(self, L, res):
        """Layer L's output rows (row-tile residual, sharded over the boundary helpers) -> this device's slice of the
        drafter's context buffer (kernels/l1_copy.cpp on the helpers)."""
        buf, split = self.ctx_buffer
        slot = self.capture_layers.index(L)
        b, kt = self.boundary, self.H // TILE
        tb = self.M * 64
        lo, hi = slot * kt, (slot + 1) * kt
        mesh_program = ttnn.MeshProgramDescriptor()
        for coord in b.coords:
            s_, e = split[coord[1]]
            a, z = max(lo, s_), min(hi, e)
            n = max(0, z - a)
            rt = ttnn.RuntimeArgs()
            # each helper copies its part of the range (the residual is sharded over them, b.PH pages each)
            shard = b.PH * 2048
            src0, src1 = (a - lo) * tb, (a - lo) * tb + n * tb
            for h, c in enumerate(b.helpers):
                u, v = max(src0, h * shard), min(src1, (h + 1) * shard)
                rt[c.x][c.y] = [
                    res.buffer_address() + (u - h * shard) if v > u else 0,
                    buf.buffer_address() + (a - s_) * tb + (u - src0) if v > u else 0,
                    max(0, v - u),
                ]
            kernel = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "l1_copy.cpp"),
                core_ranges=b.helper_cores,
                compile_time_args=[1 if buf.memory_config().buffer_type == ttnn.BufferType.DRAM else 0],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                ),
            )
            mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = ttnn.ProgramDescriptor(
                kernels=[kernel], semaphores=[], cbs=[]
            )
        ttnn.generic_op([res, buf], mesh_program)
