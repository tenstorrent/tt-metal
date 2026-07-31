import ttnn
import torch


def _single_core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def sharded_cfg(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=_single_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp"

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_gate_acc = 0, cb_up_acc = 1, cb_reduce_gate_in = 2, cb_reduce_up_in = 3, cb_gate_silu = 4;
    constexpr uint32_t HN_PAD = 6;
    constexpr uint32_t BLOCK_TILES = 1 * HN_PAD;
    compute_kernel_hw_startup(cb_gate_acc, cb_reduce_gate_in, cb_gate_silu);
    ActivationInitHelper<KernelActivation::SILU>::init();

    cb_reserve_back(cb_gate_acc, BLOCK_TILES); cb_push_back(cb_gate_acc, BLOCK_TILES);
    cb_reserve_back(cb_up_acc, BLOCK_TILES); cb_push_back(cb_up_acc, BLOCK_TILES);
    cb_reserve_back(cb_reduce_gate_in, BLOCK_TILES); cb_push_back(cb_reduce_gate_in, BLOCK_TILES);
    cb_reserve_back(cb_reduce_up_in, BLOCK_TILES); cb_push_back(cb_reduce_up_in, BLOCK_TILES);
    cb_wait_front(cb_gate_acc, BLOCK_TILES);
    cb_wait_front(cb_up_acc, BLOCK_TILES);
    cb_wait_front(cb_reduce_gate_in, BLOCK_TILES);
    cb_wait_front(cb_reduce_up_in, BLOCK_TILES);

    {
        CircularBuffer gate_buf(cb_gate_acc), rg_buf(cb_reduce_gate_in), silu_buf(cb_gate_silu);
        add_bias_bcast_rows<BiasBroadcast::Elementwise, OutputCBLayout::SubblockMajor,
                             bias_add_config::NoPostBias, SiluActivation>(
            gate_buf, rg_buf, silu_buf, BiasAddShape::of(1, 1, 1, HN_PAD), {}, 0);
    }
    cb_wait_front(cb_gate_silu, BLOCK_TILES);
    cb_pop_front(cb_gate_silu, BLOCK_TILES);
}
"""

device = ttnn.open_device(device_id=0)
try:
    m, n = 32, 192
    mem = sharded_cfg((m, n))
    gate_acc = ttnn.from_torch(
        torch.zeros(m, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )
    up_acc = ttnn.from_torch(
        torch.zeros(m, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )
    reduce_gate_in = ttnn.from_torch(
        torch.zeros(m, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )
    reduce_up_in = ttnn.from_torch(
        torch.zeros(m, n), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
    )

    def scratch_cb(cb_id, num_tiles):
        tile_size = ttnn.tile_size(ttnn.bfloat8_b)
        fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=ttnn.bfloat8_b, page_size=tile_size)
        return ttnn.CBDescriptor(total_size=tile_size * num_tiles, core_ranges=_single_core(), format_descriptors=[fmt])

    compute = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_single_core(),
        compile_time_args=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
            bfp8_pack_precise=True,
        ),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(0, gate_acc),
        ttnn.cb_descriptor_from_sharded_tensor(1, up_acc),
        ttnn.cb_descriptor_from_sharded_tensor(2, reduce_gate_in),
        ttnn.cb_descriptor_from_sharded_tensor(3, reduce_up_in),
        scratch_cb(4, 6),
    ]
    descriptor = ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
    result = ttnn.generic_op([gate_acc, up_acc, reduce_gate_in, reduce_up_in], descriptor)
    ttnn.synchronize_device(device)
    print("BIAS ADD (1 call): OK, no hang")
finally:
    ttnn.close_device(device)
