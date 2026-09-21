# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""One-tile example of MAX padding bits and their signed interpretations."""

import pytest
import torch
import ttnn


KERNEL = r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"

void kernel_main() {
    constexpr uint32_t cb_in = 0;
    constexpr uint32_t cb_out = 16;
    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    // Both CBs are resident sharded tensors; no reader/writer kernels are needed.
    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    sfpu_reduce_init<PoolType::MAX, DataFormat::Int32>();
    sfpu_reduce<PoolType::MAX, DataFormat::Int32, ReduceDim::REDUCE_ROW>(0);
    tile_regs_commit();
    tile_regs_wait();
    PACK((llk_pack_reduce_mask_config<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_out)));
    pack_tile(0, cb_out);
    PACK((llk_pack_reduce_mask_clear()));
    tile_regs_release();
}
"""


def test_int32_max_padding(device):
    if device.arch() not in (ttnn.device.Arch.BLACKHOLE, ttnn.device.Arch.WORMHOLE_B0):
        pytest.skip("This Int32 SFPU reduction example supports Blackhole and Wormhole")

    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    memory = ttnn.create_sharded_memory_config(
        shape=(32, 32),
        core_grid=cores,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # Every row is [-33, -32, ..., -2], so its MAX is -2 (distinct from the padding).
    source = torch.arange(-33, -1, dtype=torch.int32).repeat(32, 1)
    inp, out = [
        ttnn.from_torch(t, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)
        for t in (source, torch.full_like(source, 123))
    ]
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
            )
        ],
        cbs=[ttnn.cb_descriptor_from_sharded_tensor(0, inp), ttnn.cb_descriptor_from_sharded_tensor(16, out)],
    )
    actual = ttnn.to_torch(ttnn.generic_op([inp, out], program))
    padding = actual[0, 1].item()
    raw = padding & 0xFFFFFFFF
    sign_magnitude = -(raw & 0x7FFFFFFF) if raw & 0x80000000 else raw
    print(f"Input row 0: {source[0].tolist()}")
    print(f"Output row 0: {actual[0].tolist()}")
    print(f"All 32 output rows identical: {torch.equal(actual, actual[0:1].expand_as(actual))}")
    print(f"Padding raw bits: 0x{raw:08x}")
    print(f"Padding as two's complement (ttnn.to_torch): {padding}")
    print(f"Same padding bits as sign-magnitude (LLK default convention): {sign_magnitude}")

    expected = torch.full_like(source, -1)
    expected[:, 0] = source.max(dim=1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert raw == 0xFFFFFFFF
    assert sign_magnitude == -2147483647
