# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 decode MoE tail: the weighted sum over the top-k slots of the combine buffer for one tile of
tokens.

``TtReduceModule`` runs ``post_combine_reduce`` on the combine buffer (``[1, 1, T, K, H]`` row-major, one row
per
(token, slot)): one core per 32-token chunk multiplies each slot's row by its scalar weight into an L1
accumulator,
skipping the slots whose expert is not in the chip's dispatch group, then tilizes the chunk: ~158 us a layer
on 2
cores for the decode tile. :func:`moe_tail` does the same sum as one program on 80 cores,
each owning two of the 160 column tiles of H = 5,120: its reader builds the K activation tiles of a column
tile
straight from the combine rows (the rows that are on, the rest zero) and K score tiles (column 0 = the slot's
weight where on), the compute kernel MACs them with a column broadcast in fp32 and packs one tile. After
``ssinghal/dsv4p1``'s ``moe_tail``, with the mask from the expert dispatch table as
``post_combine_reduce`` takes it and the rows read as our combine lays them out.

:class:`DecodeReduce` keeps ``TtReduceModule.forward``'s contract (the weighted sum, then the TP
reduce-scatter) and
is installed on a decode MoE by :func:`install_decode_tail` (``V41_DECODE_TAIL=1``, with the decode routing's
32-row tile).
"""

import os

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import _cores

_KERNELS = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_TILE = 32
_TILE_BYTES = 2048
_CB_ACT, _CB_SCORE, _CB_SCRATCH, _CB_OUT = 0, 1, 2, 16
_CORES = int(os.environ.get("V41_TAIL_CORES", 80))  # a bisection knob


def _accessors(*tensors) -> list:
    args = []
    for t in tensors:
        args += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    return args


def _cb(index: int, size: int, cores, page: int, dtype=ttnn.bfloat16) -> ttnn.CBDescriptor:
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=size, core_ranges=cores, format_descriptors=[fmt])


def moe_tail(combine, scores, indices, table, n_cores: int = 80):
    """``combine``: row-major bf16 ``[..., T, K, H]`` DRAM (``T <= 32``, ``H % 32 == 0``), one page per
    (token,
    slot) row. ``scores``: row-major bf16 ``[..., T, K]`` (one page per token); ``indices``: row-major uint16
    of
    the same shape; ``table``: this chip's row of the expert dispatch table, int32 row-major ``[1, E + 1]``
    (``-1``
    for the experts not in the chip's dispatch group, and the pad rows' sentinel column). Returns
    ``out[t] = sum_k [table[idx[t, k]] >= 0] * scores[t, k] * combine[t, k, :]`` as bf16 TILE ``[1, 1, T, H]``
    DRAM, the K products accumulated in fp32."""
    shape = tuple(combine.shape)
    T, K, H = shape[-3], shape[-2], shape[-1]
    assert T <= _TILE and H % _TILE == 0, (T, H)
    assert combine.dtype == ttnn.bfloat16 and combine.layout == ttnn.ROW_MAJOR_LAYOUT, (combine.dtype, combine.layout)
    assert tuple(scores.shape)[-2:] == (T, K) and scores.dtype == ttnn.bfloat16, (tuple(scores.shape), scores.dtype)
    assert tuple(indices.shape)[-2:] == (T, K) and indices.dtype == ttnn.uint16, (tuple(indices.shape), indices.dtype)
    assert scores.layout == ttnn.ROW_MAJOR_LAYOUT and indices.layout == ttnn.ROW_MAJOR_LAYOUT
    assert table.dtype == ttnn.int32 and table.layout == ttnn.ROW_MAJOR_LAYOUT, (table.dtype, table.layout)
    device = combine.device()
    tiles = H // _TILE
    while tiles % n_cores:
        n_cores -= 1
    tpc = tiles // n_cores
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, T, H]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    coords, cores = _cores(device, n_cores)
    assert len(coords) == n_cores, (len(coords), n_cores)
    reader_args, writer_args = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(coords):
        reader_args[cx][cy] = [
            combine.buffer_address(),
            scores.buffer_address(),
            indices.buffer_address(),
            table.buffer_address(),
            i * tpc,
        ]
        writer_args[cx][cy] = [out.buffer_address(), i * tpc]
    table_words = int(table.shape[-1])
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/moe_tail_reader.cpp",
            core_ranges=cores,
            compile_time_args=[_CB_ACT, _CB_SCORE, _CB_SCRATCH, T, K, tpc, table_words]
            + _accessors(combine, scores, indices, table),
            runtime_args=reader_args,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/moe_tail_writer.cpp",
            core_ranges=cores,
            compile_time_args=[_CB_OUT, tpc] + _accessors(out),
            runtime_args=writer_args,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{_KERNELS}/moe_tail_compute.cpp",
            core_ranges=cores,
            compile_time_args=[tpc, K, _CB_ACT, _CB_SCORE, _CB_OUT],
            runtime_args=ttnn.RuntimeArgs(),
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
            ),
        ),
    ]
    scratch = 64 + 2 * _TILE * 64 + (table_words * 4 + 63) // 64 * 64 + T * K * tpc * 64
    cbs = [
        _cb(_CB_ACT, K * tpc * _TILE_BYTES, cores, _TILE_BYTES),
        _cb(_CB_SCORE, K * _TILE_BYTES, cores, _TILE_BYTES),
        _cb(_CB_SCRATCH, scratch, cores, scratch, ttnn.uint32),
        _cb(_CB_OUT, tpc * _TILE_BYTES, cores, _TILE_BYTES),
    ]
    ttnn.generic_op(
        [combine, scores, indices, table, out],
        ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs),
    )
    return out


# the tail's TP reduce-scatter: ``minimal`` (the default) is ``V41Collectives``'
# ``reduce_scatter_minimal_async`` on
# the model's resident semaphores, the step's other TP reduce-scatters; ``legacy`` is ``ttnn.reduce_scatter``,
# whose
# program-cache entry is first built inside the first decode step and keeps its global semaphores (8 x 64 B)
# where
# the step's transients left room, far under the top-of-L1 stack of the warm-up's entries: the routed expert
# FFN's
# static CBs (to 1,197,376) then clash with them at the next block ("L1 buffer allocated at 911104"; a
# decode-only test met the same block at 914,176 in a decode-only unit test)
_TAIL_RS = os.environ.get("V41_DECODE_TAIL_RS", "minimal")
assert _TAIL_RS in ("legacy", "minimal"), _TAIL_RS


class DecodeReduce:
    """``TtReduceModule`` for the decode tile: :func:`moe_tail`, then the module's TP reduce-scatter. With
    ``V41_SHARED_IN_FFN=1`` the decode shared expert (``moe_shared.DecodeSharedExpert``) leaves its ``[T, H]``
    partial in ``pending_shared`` and the sum takes it before the reduce-scatter."""

    def __init__(self, reduce, ccl=None):
        self._reduce = reduce
        self._ccl = ccl
        self.pending_shared = None
        assert _TAIL_RS == "legacy" or ccl is not None, "V41_DECODE_TAIL_RS=minimal needs the collectives"

    def __getattr__(self, name):
        return getattr(self._reduce, name)

    def __call__(self, combine_output, weights=None, indices=None, expert_dispatch_table=None):
        assert weights is not None and indices is not None and expert_dispatch_table is not None
        summed = moe_tail(combine_output, weights, indices, expert_dispatch_table, n_cores=_CORES)
        if self.pending_shared is not None:
            shared, self.pending_shared = self.pending_shared, None
            total = ttnn.add(summed, ttnn.reshape(shared, summed.shape))
            ttnn.deallocate(summed)
            ttnn.deallocate(shared)
            summed = total
        r = self._reduce
        if r.mesh_device.shape[r.cluster_axis] > 1:
            if _TAIL_RS == "minimal":
                return self._ccl.tp_reduce_scatter(summed, dim=3)
            return ttnn.reduce_scatter(
                summed, dim=-1, cluster_axis=r.cluster_axis, num_links=r.num_links, topology=r.topology
            )
        return summed

    forward = __call__


def install_decode_tail(v41_moe, ccl=None) -> None:
    """Swap ``v41_moe.moe``'s reduce module for :class:`DecodeReduce` (``ccl``: the model's
    ``V41Collectives``, for
    ``V41_DECODE_TAIL_RS=minimal``)."""
    moe = v41_moe.moe
    if not isinstance(moe.reduce_module, DecodeReduce):
        moe.reduce_module = DecodeReduce(moe.reduce_module, ccl)
