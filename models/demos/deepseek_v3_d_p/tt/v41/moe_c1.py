# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 decode MoE for one live row (``V41_DECODE_C1_MOE=1``).

The batched decode step runs the MoE on a tile whose first ``users_per_rank`` rows of every rank are real, whatever
is live: at c1 the 31 idle rows (token 0) select 6 experts each and go through the dispatch, the experts and the
combine ( ~0.5 to 0.65 ms a layer). With one live row the fabric chain is not needed:
:class:`C1Moe` gathers the row to every chip (a one-hot matmul picks it out of the live rank's tile into row 0 of
an otherwise zero tile, an SP all-gather of the tiles and a one-hot sum over the gathered rows broadcast it, a TP
all-gather completes it), runs the gate and the exact top-k redundantly, selects the row's experts that are this
chip's (:func:`c1_select`, one core) into the routed expert FFN's ``counts`` and the weights of the local sum,
runs ``unified_routed_expert_moe`` on a 12-region buffer that holds the row at the head of each region (one tile
row an expert), sums the local outputs with a ``[32, 384] x [384, 5120]`` matmul, sums the partials over the 32
chips (TP reduce-scatter, then the SP all-gather and one-hot sum again), adds the shared expert's output and
places the result at the live row of the live rank's tile with a second one-hot matmul (the other rows zero). The
gate, ``router_select``, ``fp8_qdq``, the shared expert and the routed expert FFN are the MoE's own; the
collectives are ``V41Collectives``' and the pool's. Measured (``tests/v41/test_v41_moe_c1.py`` under trace): ~500
us a layer against ~1,120 for the 32-row path on random rows; the routed expert FFN is ~90 us an expert (20 MB of
bfp4 weights), the shared expert 72, the two SP all-gathers 44 each, the gate 45, 28 programs.

The one-hot tiles ``sel`` (row 0 = e_u on the live rank) and ``put`` (row u = e_0 on the live rank) come from
two integer rows of the step's host inputs (:meth:`C1Moe.host_rows`) through :meth:`C1Moe.onehots` (two
``ttnn.embedding`` lookups of an identity table), once a step.
"""

import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import _fused_compute_config
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import _cores
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp8_qdq
from models.demos.deepseek_v3_d_p.tt.v41.routing import DecodeGate, router_select

_KERNELS = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_TILE = 32
_SENTINEL = 32  # the identity table's zero row
# the routed expert FFN op: ``fused`` (moe_fused_swiglu, the in-tree op for few tokens an expert) or ``composite``
# (unified_routed_expert_moe, today's prefill op: ~90 us an expert at one token)
_FFN = os.environ.get("V41_DECODE_C1_FFN", "composite")
assert _FFN in ("fused", "composite"), _FFN
_HIFI4 = ttnn.types.BlackholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False, math_approx_mode=False
)


def _accessors(*tensors) -> list:
    args = []
    for t in tensors:
        args += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    return args


def c1_select(indices, weights, table, num_experts: int):
    """``indices``: uint16 TILE ``[T, K]`` (``T <= 32``; row 0 the live row), ``weights``: bf16 row-major
    ``[T, K]``, ``table``: this chip's ``global_expert_idx_table`` (uint32 row-major, ``EPC`` words). Returns
    ``(counts, w)``: uint32 row-major ``[1, E]`` (1 at the row's experts that are this chip's) and bf16 TILE
    ``[32, 32 * EPC]`` (row 0, column ``32 l`` = the row's weight of local expert ``l``, else 0)."""
    E = num_experts
    T, K = int(indices.shape[-2]), int(indices.shape[-1])
    assert T <= _TILE and K <= 16, (T, K)
    assert indices.dtype == ttnn.uint16 and indices.layout == ttnn.TILE_LAYOUT, (indices.dtype, indices.layout)
    assert weights.dtype == ttnn.bfloat16 and weights.layout == ttnn.ROW_MAJOR_LAYOUT, (weights.dtype, weights.layout)
    assert tuple(weights.shape)[-2:] == (T, K), tuple(weights.shape)
    assert table.dtype == ttnn.uint32 and table.layout == ttnn.ROW_MAJOR_LAYOUT, (table.dtype, table.layout)
    epc = int(table.shape[-1])
    device = indices.device()
    counts = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, E]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    w = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, _TILE, _TILE * epc]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    coords, cores = _cores(device, 1)
    args = ttnn.RuntimeArgs()
    cx, cy = coords[0]
    args[cx][cy] = [
        indices.buffer_address(),
        weights.buffer_address(),
        table.buffer_address(),
        counts.buffer_address(),
        w.buffer_address(),
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=f"{_KERNELS}/c1_select.cpp",
        core_ranges=cores,
        compile_time_args=[0, E, K, epc] + _accessors(indices, weights, table, counts, w),
        runtime_args=args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    scratch = 64 + 2048 + 64 + (epc * 4 + 63) // 64 * 64 + (E * 4 + 63) // 64 * 64 + epc * 2048
    fmt = ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=scratch)
    cbs = [ttnn.CBDescriptor(total_size=scratch, core_ranges=cores, format_descriptors=[fmt])]
    ttnn.generic_op(
        [indices, weights, table, counts, w], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=cbs)
    )
    return counts, w


class C1Moe:
    """The c1 MoE of one decode layer's ``TtV41Moe`` (its gate a :class:`DecodeGate`): see the module docstring.
    ``ccl``: the model's ``V41Collectives``; ``pool``: the decoder's ``V41DecodePool`` (its ``tt_ccl``)."""

    def __init__(self, v41_moe, ccl, pool):
        moe = v41_moe.moe
        assert isinstance(moe.gate, DecodeGate), "the c1 MoE needs the decode routing (V41_DECODE_ROUTING=1)"
        self.moe, self.ccl, self.pool = moe, ccl, pool
        self.gate: DecodeGate = moe.gate
        self.routed = moe.routed_expert
        self.k = moe.num_experts_per_tok
        self.E = moe.num_routed_experts
        self.epc = self.routed.experts_per_chip
        self.table = self.routed.global_expert_idx_table  # uint32 [EPC] per chip, the chip's global expert ids
        assert self.table.layout == ttnn.ROW_MAJOR_LAYOUT and int(self.table.shape[-1]) == self.epc, self.table
        device = moe.mesh_device
        # the region of local expert l starts at row 32 l of the FFN buffer: a constant per chip, built from the
        # chip's expert table (the FFN reads region[global id] for each of its experts; the other entries unread)
        hidden = int(self.routed.gate_projs[0].shape[-2])
        self.hidden = hidden
        region = []
        for chip in ttnn.get_device_tensors(self.table):
            ids = ttnn.to_torch(chip).flatten().to(torch.int64)
            row = torch.zeros(1, self.E, dtype=torch.int32)
            row[0, ids] = torch.arange(self.epc, dtype=torch.int32) * _TILE
            region.append(row)
        self.region = ttnn.from_torch(
            torch.stack(region).view(*tuple(device.shape), 1, self.E),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=device.shape, dims=(0, 1)),
        )
        self.region = ttnn.reshape(self.region, (1, self.E))
        # R[32 l, 0] = 1: ``R @ x_full`` puts the row at the head of each expert's region
        R = torch.zeros(self.epc * _TILE, _TILE)
        R[::_TILE, 0] = 1.0
        self.R = self._replicated(R.view(1, 1, self.epc * _TILE, _TILE), ttnn.bfloat16)
        # row 0 all ones: ``sum8 @ gathered`` sums the SP column's tiles (their rows 1..31 zero) into row 0
        sum8 = torch.zeros(_TILE, _TILE * pool.sp)
        sum8[0] = 1.0
        self.sum8 = self._replicated(sum8.view(1, 1, _TILE, _TILE * pool.sp), ttnn.bfloat16)
        self.debug = None  # a dict: the call keeps its ``counts`` and ``indices`` there (tests)
        # the FFN's program-cache entry for this shape is built here, on an empty L1: ``ttnn.reduce_scatter`` (the
        # decode tail's TP reduce-scatter) leaves a 512-byte L1 buffer at 914,176 after its first call, and the
        # routed experts' static CBs (to 1,004,864 in this mode) are rejected at compile against it; today's
        # 32-row FFN entry is built before any reduce-scatter runs and never meets it
        zeros = ttnn.from_torch(
            torch.zeros(1, 1, self.epc * _TILE, hidden),
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        none = self._replicated(torch.zeros(1, self.E, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(self._ffn(ttnn.reshape(zeros, (self.epc * _TILE, hidden)), none))
        ttnn.deallocate(none)

    def _replicated(self, t: torch.Tensor, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            t,
            device=self.moe.mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.moe.mesh_device),
        )

    # --- the one-hot tiles, once a step ------------------------------------------------------------------------
    @staticmethod
    def host_rows(sp: int, rows: int, live: tuple[int, int] | None) -> dict:
        """The two integer rows per rank (``[sp, rows]`` int32): ``c1sel`` (word 0 = u on the live rank) and
        ``c1put`` (word u = 0 on the live rank), every other word the sentinel. ``live``: (rank, u) or None."""
        sel = torch.full((sp, rows), _SENTINEL, dtype=torch.int32)
        put = torch.full((sp, rows), _SENTINEL, dtype=torch.int32)
        if live is not None:
            r, u = live
            sel[r, 0] = u
            put[r, u] = 0
        return {"c1sel": sel, "c1put": put}

    @staticmethod
    def identity_table(mesh_device):
        """``[33, 32]`` bf16 row-major: the identity and a zero row (the sentinel), replicated."""
        eye = torch.cat([torch.eye(_TILE), torch.zeros(1, _TILE)]).to(torch.bfloat16)
        return ttnn.from_torch(
            eye,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @staticmethod
    def onehots(ids_sel, ids_put, table):
        """``ids_*``: uint32 row-major ``[1, 1, 1, 32]`` (the device slices of ``host_rows``) -> the ``sel`` and
        ``put`` tiles, bf16 TILE ``[1, 1, 32, 32]``."""
        out = []
        for ids in (ids_sel, ids_put):
            flat = ttnn.reshape(ids, (1, 1, _TILE))
            tile = ttnn.embedding(flat, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            out.append(ttnn.reshape(tile, (1, 1, _TILE, _TILE)))
        return tuple(out)

    # --- the layer -----------------------------------------------------------------------------------------------
    def _ffn(self, x_in, counts):
        """The routed expert FFN on the 12-region buffer (bfp8 TILE ``[EPC * 32, hidden]``): local expert ``l``'s
        rows at ``32 l`` where ``counts`` says so. ``fused``: ``moe_fused_swiglu`` into a zeroed output (the
        unselected regions stay zero); ``composite``: ``unified_routed_expert_moe`` in place."""
        routed = self.routed
        if _FFN == "fused":
            out = ttnn.zeros_like(x_in, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.experimental.deepseek_prefill.moe_fused_swiglu(
                x_in,
                routed.gate_projs,
                routed.up_projs,
                routed.down_projs,
                counts,
                self.table,
                input_m_tiles=1,
                core_grid=ttnn.UNIFIED_ROUTED_EXPERT_CORE_GRID,
                compute_kernel_config=_fused_compute_config(routed.compute_kernel_config),
                activation=routed.activation,
                output=out,
                expert_region_offsets=self.region,
                read_x_at_offset=True,
                max_active_tokens=_TILE,
                gate_biases=routed.gate_biases,
                up_biases=routed.up_biases,
                down_biases=routed.down_biases,
            )
            ttnn.deallocate(x_in)
            return out
        return ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            x_in,
            self.region,
            counts,
            self.table,
            routed.gate_projs,
            routed.up_projs,
            routed.down_projs,
            max_dispatched_tokens_per_expert=_TILE,
            compute_kernel_config=routed.compute_kernel_config,
            activation=routed.activation,
            gate_biases=routed.gate_biases,
            up_biases=routed.up_biases,
            down_biases=routed.down_biases,
        )

    def _sp_all_reduce(self, t):
        """``t`` ``[1, 1, 32, W]`` with rows 1..31 zero -> the same shape on every chip of the SP column, row 0 the
        sum of the column's row 0s: one all-gather of the tiles (dim 2) and a one-hot matmul over the 32 x sp
        gathered rows. A ring reduce-scatter and all-gather over the 8 chips cost 46 us each (7 serialized hops);
        the all-gather alone is one of them, the matmul 3 us."""
        if self.pool.sp == 1:
            return t
        gathered = self.pool.gather_sp(t, dim=2)
        out = ttnn.matmul(self.sum8, gathered, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gathered)
        return out

    def __call__(self, h, sel, put):
        """``h``: bf16 TILE ``[1, 1, 32, hidden/tp]`` (the normed residual tile), ``sel`` / ``put`` of
        :meth:`onehots` -> the MoE output of the live row at its row of the live rank's tile, the other rows zero;
        the same shape as ``h``."""
        moe, gate, cfg = self.moe, self.gate, self.gate.config
        # 1, 2: the live row's TP shard to every chip
        picked = ttnn.matmul(sel, h, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xs = self._sp_all_reduce(picked)
        ttnn.deallocate(picked)
        # 3, 4: the gate and the exact top-k of row 0 (rows 1..31 the sentinel)
        xs2 = ttnn.reshape(xs, (_TILE, int(xs.shape[-1])))
        logits = gate._gate._device_matmul(xs2)
        zero, bias32 = gate._constants(logits)
        l32 = ttnn.typecast(logits, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(logits)
        activations = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT),
        ]
        score = ttnn.add(l32, zero, input_tensor_a_activations=activations, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(l32)
        rank = ttnn.add(score, bias32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        weights, indices = router_select(rank, score, self.k, 1e-20, cfg.route_scale, 1)
        ttnn.deallocate(rank)
        ttnn.deallocate(score)
        # 5: the row's experts that are this chip's
        counts, W = c1_select(indices, weights, self.table, self.E)
        if self.debug is not None:
            self.debug.update(counts=counts, indices=indices)
        else:
            ttnn.deallocate(indices)
        ttnn.deallocate(weights)
        # 6: the experts' input: the QDQ'd row, completed over TP
        xq = fp8_qdq(xs)
        ttnn.deallocate(xs)
        x_full = self.ccl.tp_all_gather(xq, dim=3)
        ttnn.deallocate(xq)
        # 7: the shared expert (rows 1..31 zero in, zero out)
        shared = moe.shared_expert(ttnn.squeeze(x_full, dim=0))
        shared = ttnn.unsqueeze(shared, dim=0)
        # 8, 9: the row at the head of every local expert's region; the selected experts run in place
        x_in = ttnn.matmul(
            self.R, x_full, compute_kernel_config=_HIFI4, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(x_full)
        x_in = ttnn.reshape(x_in, (self.epc * _TILE, self.hidden))
        y = self._ffn(x_in, counts)
        if self.debug is None:
            ttnn.deallocate(counts)
        # 10: the local weighted sum (row 0)
        y = ttnn.reshape(y, (1, 1, self.epc * _TILE, self.hidden))
        part = ttnn.matmul(W, y, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y)
        ttnn.deallocate(W)
        # 11: the sum over the 32 chips, this chip's TP slice
        tp_part = self.ccl.tp_reduce_scatter(part, dim=3)
        ttnn.deallocate(part)
        summed = self._sp_all_reduce(tp_part)
        ttnn.deallocate(tp_part)
        # 12: + the shared expert, placed at the live row
        full = ttnn.add(summed, shared, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(summed)
        ttnn.deallocate(shared)
        out = ttnn.matmul(put, full, compute_kernel_config=_HIFI4, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(full)
        return out
