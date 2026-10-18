# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated, unselected unified-routed-expert comparison for the fused stage.

Install ``UnifiedMoE`` as ``tt.fused_decoder.FusedMoE`` before constructing a
decoder, or construct it directly with the selected MoE factory arguments.
No device output is read on the host. The inherited caller pads each logical
tail to a 32-token group and crops the result afterward.

The native operation fixes LoFi/BF16 DEST and BFP8 intermediates/output; its
compute-config argument does not override those choices. This candidate needs
real-weight PCC and traced latency evidence before selection.
"""

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedMoE

_DISPATCH_COPY = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto xa = TensorAccessorArgs<0>();
    constexpr auto ya = TensorAccessorArgs<xa.next_compile_time_args_offset()>();
    constexpr auto ca = TensorAccessorArgs<ya.next_compile_time_args_offset()>();
    const auto x = TensorAccessor(xa, get_arg_val<uint32_t>(0));
    const auto y = TensorAccessor(ya, get_arg_val<uint32_t>(1));
    const auto counts = TensorAccessor(ca, get_arg_val<uint32_t>(2));
    const uint32_t row = get_arg_val<uint32_t>(3);
    CircularBuffer row_cb(0), count_cb(1);
    row_cb.reserve_back(1);
    count_cb.reserve_back(1);
    Noc noc;
    noc.async_read(x, CoreLocalMem<uint32_t>(row_cb.get_write_ptr()), 5120, {.page_id=row}, {});
    noc.async_read(counts, CoreLocalMem<uint32_t>(count_cb.get_write_ptr()), 2048, {.page_id=0}, {});
    noc.async_read_barrier();
    const volatile uint32_t* active = reinterpret_cast<volatile uint32_t*>(count_cb.get_write_ptr());
    for (uint32_t expert = 0; expert < 512; ++expert) {
        if (active[expert] == 0) continue;
        // Every active expert receives the complete 32-row union group. Each
        // core owns one row, and expert regions are disjoint. The row scratch
        // remains unchanged until all writes have completed.
        noc.async_write(CoreLocalMem<uint32_t>(row_cb.get_write_ptr()), y, 5120,
                        {}, {.page_id=expert * 32 + row});
    }
    noc.async_write_barrier();
}
"""


class UnifiedMoE(FusedMoE):
    """Same routing/shared expert as FusedMoE, with native unified routed FFN."""

    @classmethod
    def from_state_dict(cls, state, **kwargs):
        obj = super().from_state_dict(state, **kwargs)
        if (obj.experts, obj.hidden, obj.intermediate) != (512, 2560, 640):
            raise ValueError("The isolated dispatch kernel requires E512/H2560/I640")
        # Reuse the selected path's persistent 80 MiB BF16 row-major capacity
        # buffer. The unified wrapper allocates its own BFP8 output, so the
        # selected path's BF16 output template is unnecessary here.
        del obj.output_template
        obj._dispatch_coords = [ttnn.CoreCoord(i % 8, i // 8) for i in range(32)]
        obj._dispatch_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in obj._dispatch_coords])
        return obj

    def _dispatch_input(self, x, counts):
        source = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        args = (
            ttnn.TensorAccessorArgs(source).get_compile_time_args()
            + ttnn.TensorAccessorArgs(self.capacity_input).get_compile_time_args()
            + ttnn.TensorAccessorArgs(counts).get_compile_time_args()
        )
        kernel = ttnn.KernelDescriptor(
            kernel_source=_DISPATCH_COPY,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=self._dispatch_cores,
            compile_time_args=args,
            runtime_args=[
                (
                    (c.x, c.y),
                    [source.buffer_address(), self.capacity_input.buffer_address(), counts.buffer_address(), i],
                )
                for i, c in enumerate(self._dispatch_coords)
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        cbs = [
            ttnn.CBDescriptor(
                total_size=size,
                core_ranges=self._dispatch_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
            )
            for index, size, dtype in ((0, 5120, ttnn.bfloat16), (1, 2048, ttnn.uint32))
        ]
        ttnn.generic_op([source, self.capacity_input, counts], ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs))
        return self.capacity_input

    def _group(self, x):
        if tuple(x.shape) != (1, 1, 32, self.hidden):
            raise ValueError("The inherited MoE caller must supply one padded 32-token group")
        logits = ttnn.linear(x, self.router, compute_kernel_config=self.compute)
        values, indices = ttnn.topk(logits, k=32, dim=-1, sorted=True)
        selected = ttnn.softmax(
            ttnn.add(values, self.top_mask), dim=-1, numeric_stable=True, compute_kernel_config=self.compute
        )
        routing = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected)
        active = ttnn.max(routing, dim=2, keepdim=True) > 0
        counts = ttnn.typecast(ttnn.where(active, 32, 0), ttnn.uint32)
        counts = ttnn.reshape(ttnn.to_layout(counts, ttnn.ROW_MAJOR_LAYOUT), (self.experts,))
        dispatched = self._dispatch_input(x, counts)
        down = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
            dispatched,
            self.expert_offsets,
            counts,
            self.expert_ids,
            self.expert_gates,
            self.expert_ups,
            self.expert_downs,
            32,
        )
        down = ttnn.reshape(ttnn.typecast(down, ttnn.bfloat16), (1, self.experts, 32, self.hidden))
        # ROW_MAJOR input makes the native wrapper allocate uninitialized
        # BFP8 output. It never writes inactive regions. True conditional
        # selection clears those regions, including NaNs; multiplication by
        # zero is insufficient. Active-region non-finites remain visible to
        # correctness checks instead of being silently sanitized.
        down = ttnn.where(ttnn.permute(active, (0, 3, 2, 1)), down, 0.0)
        if self.retain_intermediates:
            self.intermediates = dict(routing=routing, counts=counts, down=down)
        result = ttnn.sum(ttnn.multiply(down, ttnn.permute(routing, (0, 3, 2, 1))), dim=1, keepdim=True)
        shared_packed = ttnn.linear(x, self.shared_packed, compute_kernel_config=self.compute)
        sg, su = shared_packed[..., :640], shared_packed[..., 640:1280]
        scale = shared_packed[..., 1280:1281]
        shared = ttnn.linear(
            ttnn.multiply(sg, su, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            self.shared_down,
            compute_kernel_config=self.compute,
        )
        return ttnn.add(result, ttnn.multiply(shared, scale, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]))
