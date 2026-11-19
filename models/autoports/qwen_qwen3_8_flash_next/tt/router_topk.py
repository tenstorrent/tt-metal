# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ATen-compatible top10 for the pinned Qwen router's 512 FP32 logits.

Inputs are BF16-rounded logits stored as FP32 TILE [1,1,Q,512], 1<=Q<=32.
Outputs are selected FP32 logits and UINT32 indices, TILE [1,1,Q,10]. The
selection matches the installed HF CPU ATen/libstdc++ tie policy, rather than
inventing an index-based tie break. FP32 router accumulation and its BF16 output
rounding boundary are caller-owned; this operation performs no float arithmetic.

The functional call allocates outputs on device; warm before capture. Prepared
RouterTopK objects accept persistent outputs and perform no runtime allocation.
All selection and data movement execute on the device; no host fallback exists.
"""

from pathlib import Path

_ALGORITHM_SOURCE = Path(__file__).with_name("kernels").joinpath("router_topk_algorithm.hpp").read_text()


def algorithm_source():
    return _ALGORITHM_SOURCE


def kernel_source():
    return (
        '#include "api/dataflow/dataflow_api.h"\n'
        + algorithm_source()
        + r"""
inline uint32_t router_face(uint32_t row, uint32_t column) {
    return ((row / 16) * 2 + column / 16) * 256 + (row % 16) * 16 + column % 16;
}
void kernel_main() {
    constexpr auto ia = TensorAccessorArgs<0>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto oa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(ia, get_arg_val<uint32_t>(0), 4096);
    const auto scores = TensorAccessor(sa, get_arg_val<uint32_t>(1), 4096);
    const auto indices = TensorAccessor(oa, get_arg_val<uint32_t>(2), 4096);
    const uint32_t row = get_arg_val<uint32_t>(3);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    volatile tt_l1_ptr uint32_t* raw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    RouterItem* queue = reinterpret_cast<RouterItem*>(scratch + 2048);
    volatile tt_l1_ptr uint32_t* selected = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 6144);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 6272);
    for (uint32_t tile = 0; tile < 16; ++tile) {
        for (uint32_t half = 0; half < 2; ++half) {
            noc_async_read(input.get_noc_addr(tile) + router_face(row, half * 16) * 4,
                           scratch + tile * 128 + half * 64, 64);
        }
    }
    noc_async_read_barrier();
    for (uint32_t i = 0; i < 512; ++i) queue[i] = {raw[i], i};
    router_top10(queue);
    for (uint32_t i = 0; i < 32; ++i) {
        selected[i] = i < 10 ? queue[i].bits : 0;
        ids[i] = i < 10 ? queue[i].index : 0;
    }
    for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t offset = router_face(row, half * 16) * 4;
        noc_async_write(scratch + 6144 + half * 64, scores.get_noc_addr(0) + offset, 64);
        noc_async_write(scratch + 6272 + half * 64, indices.get_noc_addr(0) + offset, 64);
    }
    noc_async_write_barrier();
}
"""
    )


class RouterTopK:
    def __init__(self, logits, scores, indices):
        import ttnn

        self._ttnn = ttnn
        self.tensors = [logits, scores, indices]
        shape = tuple(logits.shape)
        if len(shape) != 4 or shape[:2] != (1, 1) or shape[3] != 512 or not 1 <= shape[2] <= 32:
            raise ValueError("Expected router logits [1,1,Q,512], 1<=Q<=32")
        for tensor, dtype, expected in (
            (logits, ttnn.float32, shape),
            (scores, ttnn.float32, (1, 1, shape[2], 10)),
            (indices, ttnn.uint32, (1, 1, shape[2], 10)),
        ):
            if tuple(tensor.shape) != expected or tensor.dtype != dtype or tensor.layout != ttnn.TILE_LAYOUT:
                raise ValueError("Router topk requires matching tiled FP32 logits/scores and UINT32 indices")
            if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("Router topk tensors must use interleaved DRAM")
        addresses = [tensor.buffer_address() for tensor in self.tensors]
        if len(set(addresses)) != 3:
            raise ValueError("Router topk buffers must not alias")
        grid = logits.device().compute_with_storage_grid_size()
        if grid.x * grid.y < shape[2]:
            raise ValueError("Router topk requires one worker per logical row")
        cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(shape[2])]
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
        runtime_args = ttnn.RuntimeArgs()
        for row, core in enumerate(cores):
            runtime_args[core.x][core.y] = [*addresses, row]
        compile_args = []
        for tensor in self.tensors:
            compile_args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
        self.program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=kernel_source(),
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=ranges,
                    compile_time_args=compile_args,
                    runtime_args=runtime_args,
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=8192,
                    core_ranges=ranges,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=8192)
                    ],
                )
            ],
            semaphores=[],
        )

    def prepare(self):
        self._ttnn.experimental.prepare_generic_op(self.tensors, self.program)

    def forward(self):
        result = self._ttnn.generic_op(self.tensors, self.program)
        return self.tensors[1], result


def aten_router_topk(logits, *, scores=None, indices=None):
    import ttnn

    shape = (1, 1, logits.shape[-2], 10)
    if scores is None:
        scores = ttnn.empty(shape, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=logits.device())
    if indices is None:
        indices = ttnn.empty(shape, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=logits.device())
    return RouterTopK(logits, scores, indices).forward()
