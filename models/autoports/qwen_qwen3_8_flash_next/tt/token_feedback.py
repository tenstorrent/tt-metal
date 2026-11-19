# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact resident UINT32 token selection; no sampling mathematics.

The native where operation requires TILE inputs. This dataflow-only operation
preserves the common sampler's persistent ROW_MAJOR token and membership ABI.
Standalone tracked and real-generator probes cover exact inactive-row feedback.
"""

_SELECT_SOURCE = r"""
inline void select_feedback(volatile uint32_t* current,
                            const volatile uint32_t* saved,
                            const volatile uint32_t* active) {
    for (uint32_t lane = 0; lane < 32; ++lane)
        current[lane] = active[lane] != 0u ? current[lane] : saved[lane];
}
"""


def kernel_source():
    return (
        '#include <stdint.h>\n#include "api/dataflow/dataflow_api.h"\n'
        + _SELECT_SOURCE
        + r"""
void kernel_main() {
    constexpr auto ca = TensorAccessorArgs<0>();
    constexpr auto sa = TensorAccessorArgs<ca.next_compile_time_args_offset()>();
    constexpr auto aa = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto current = TensorAccessor(ca, get_arg_val<uint32_t>(0), 128);
    const auto saved = TensorAccessor(sa, get_arg_val<uint32_t>(1), 128);
    const auto active = TensorAccessor(aa, get_arg_val<uint32_t>(2), 128);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_0);
    noc_async_read(current.get_noc_addr(0), scratch, 128);
    noc_async_read(saved.get_noc_addr(0), scratch + 128, 128);
    noc_async_read(active.get_noc_addr(0), scratch + 256, 128);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    for (uint32_t lane = 0; lane < 32; ++lane) ASSERT(words[64 + lane] <= 1u);
    select_feedback(words, words + 32, words + 64);
    noc_async_write(scratch, current.get_noc_addr(0), 128);
    noc_async_write_barrier();
}
"""
    )


class DeviceTokenFeedback:
    """Prepare once, then enqueue with fixed addresses and runtime membership.

    active is the existing DeviceSeedState.active buffer, updated only at request
    transitions. prepare() executes the operation: invoke with initially zero
    token buffers, or save/restore caller state around setup. No buffers are
    allocated by this helper. The operation mutates current only.
    """

    def __init__(self, current, saved, active):
        import ttnn

        self.t = ttnn
        self.tensors = [current, saved, active]
        for tensor, shape in zip(self.tensors, ((1, 1, 1, 32), (1, 1, 1, 32), (32,)), strict=True):
            if tuple(tensor.shape) != shape:
                raise ValueError(f"Expected token feedback shape {shape}")
            if tensor.dtype != ttnn.uint32 or tensor.layout != ttnn.ROW_MAJOR_LAYOUT:
                raise ValueError("Token feedback requires exact UINT32 ROW_MAJOR buffers")
            if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("Token feedback requires interleaved DRAM buffers")
        if active.buffer_address() in (current.buffer_address(), saved.buffer_address()):
            raise ValueError("Active membership must not alias mutable token buffers")
        core = ttnn.CoreCoord(0, 0)
        ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        runtime = ttnn.RuntimeArgs()
        runtime[0][0] = [tensor.buffer_address() for tensor in self.tensors]
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
                    runtime_args=runtime,
                    config=ttnn.WriterConfigDescriptor(),
                )
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=384,
                    core_ranges=ranges,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=384)
                    ],
                )
            ],
            semaphores=[],
        )
        self.prepared = False

    def prepare(self):
        if not self.prepared:
            self.t.experimental.prepare_generic_op(self.tensors, self.program)
            self.t.generic_op(self.tensors, self.program)
            self.prepared = True

    def forward(self):
        if not self.prepared:
            raise RuntimeError("Token feedback must be prepared before capture")
        self.t.generic_op(self.tensors, self.program)
