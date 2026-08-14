"""Minimal repro: DestReuseBinary (mul_reuse_dest_tiles) at a chosen DEST slot.

out = x * gamma, full tiles, bf16, no broadcast, no in-place.  One core, one tile-row of S tiles.
MODE 0 = CopyTile -> DestReuseBinary(mul)      (the dest-reuse path)
MODE 1 = BinaryFpu(mul) on the same two CBs    (control: normal two-operand FPU mul, same slot)
"""
import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn

TILE, BF16 = 32, 2048
CB_X, CB_G, CB_OUT = 0, 1, 9

KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_x = 0, cb_g = 1, cb_out = 9;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t BLK = get_compile_time_arg_val(1);
    constexpr ckl::Dst SLOT = static_cast<ckl::Dst>(get_compile_time_arg_val(2));
    constexpr uint32_t MODE = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(cb_x, cb_g, cb_out);

    constexpr auto x_in = ckl::input(cb_x, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Block);
    constexpr auto g_in = ckl::input(
        cb_g, ckl::WaitPolicy::None, ckl::PopPolicy::None, ckl::OperandKind::Row, ckl::TileOffset::Unset);
    constexpr auto o = ckl::output(cb_out, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd);

    cb_wait_front(cb_x, S);
    cb_wait_front(cb_g, S);
    if constexpr (MODE == 0) {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(1, S).block_size(BLK),
            ckl::CopyTile<x_in, SLOT>{},
            ckl::DestReuseBinary<g_in, ckl::BinaryFpuOp::Mul, ckl::DestReuseType::DEST_TO_SRCA, SLOT>{},
            ckl::PackTile<o, SLOT>{});
    } else {
        ckl::eltwise_chain(
            ckl::IterationShape::grid(1, S).block_size(BLK),
            ckl::BinaryFpu<ckl::BinaryFpuOp::Mul, x_in, g_in, SLOT>{},
            ckl::PackTile<o, SLOT>{});
    }
}
"""

PUBLISH = r"""
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    cb_reserve_back(0, S); cb_push_back(0, S);
    cb_reserve_back(1, S); cb_push_back(1, S);
}
"""


def crs():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def shard(shape):
    return ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=crs(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def run(device, s, blk, slot, mode):
    width = s * TILE
    torch.manual_seed(3)
    x = (torch.rand(TILE, width) * 2 - 1).to(torch.bfloat16)
    g = (torch.rand(TILE, width) + 0.5).to(torch.bfloat16)
    q = lambda t: t.to(torch.bfloat16).to(torch.float32)
    xf, gf = q(x), q(g)
    expected = xf * gf

    dev = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard((TILE, width))
    )
    xd, gd = dev(x), dev(g)
    od = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, shard((TILE, width))
    )
    c = crs()
    pd = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PUBLISH,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=c,
                compile_time_args=[s],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=KERNEL,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=c,
                compile_time_args=[s, blk, slot, mode],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
                ),
            ),
        ],
        semaphores=[],
        cbs=[
            ttnn.cb_descriptor_from_sharded_tensor(CB_X, xd),
            ttnn.cb_descriptor_from_sharded_tensor(CB_G, gd),
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, od),
        ],
    )
    ttnn.generic_op([xd, gd, od], pd)
    ttnn.synchronize_device(device)
    a = ttnn.to_torch(od).to(torch.float32)

    tag = f"S={s} BLK={blk} SLOT=D{slot} MODE={'dest_reuse' if mode == 0 else 'plain_fpu_mul'}"
    per_tile = []
    for tc in range(s):
        aa, ee = a[:, tc * TILE : (tc + 1) * TILE], expected[:, tc * TILE : (tc + 1) * TILE]
        per_tile.append((aa - ee).abs().max().item() / (ee.pow(2).mean().sqrt().item() + 1e-9))
    worst = max(range(s), key=lambda i: per_tile[i])
    print(f"\n{tag}\n  per-tile rel err: " + " ".join(f"{v:.3f}" for v in per_tile))
    if per_tile[worst] > 0.1:
        aa = a[:, worst * TILE : (worst + 1) * TILE]
        ee = expected[:, worst * TILE : (worst + 1) * TILE]
        gg = gf[:, worst * TILE : (worst + 1) * TILE]
        faces = [
            [
                (aa[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] - ee[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16])
                .abs()
                .max()
                .item()
                / (ee.pow(2).mean().sqrt().item() + 1e-9)
                for c in (0, 1)
            ]
            for r in (0, 1)
        ]
        print(
            f"  BAD tile {worst}; face err [f0 f1 / f2 f3] = "
            f"{faces[0][0]:.3f} {faces[0][1]:.3f} / {faces[1][0]:.3f} {faces[1][1]:.3f}"
        )
        # smoking gun: is a == x*(1+gamma) on the bad face?  (dest not cleared -> accumulate)
        xx = xf[:, worst * TILE : (worst + 1) * TILE]
        notclr = xx * (1.0 + gg)
        f3 = (slice(16, 32), slice(16, 32))
        print(f"  bad face 3: max|a - x*gamma|      = {(aa[f3] - ee[f3]).abs().max().item():.4g}")
        print(
            f"  bad face 3: max|a - x*(1+gamma)|  = {(aa[f3] - notclr[f3]).abs().max().item():.4g}   <-- 0 => DEST not cleared"
        )
    else:
        print("  OK")
    for t in (xd, gd, od):
        ttnn.deallocate(t)


device = ttnn.open_device(device_id=0)
try:
    run(device, 8, 8, 0, 0)  # reproduce: window 0..7
    run(device, 8, 1, 7, 0)  # DECISIVE: only slot 7, one tile per DEST window
    run(device, 8, 1, 6, 0)  # control: only slot 6
    run(device, 8, 1, 0, 0)  # control: only slot 0
    run(device, 8, 1, 7, 1)  # control: slot 7, plain two-operand FPU mul (no dest reuse)
finally:
    ttnn.close_device(device)
