"""Boundary rule + raw-LLK repro for the dest-reuse missed-DEST-clear bug.

MODE 0 = kernel_lib chain: CopyTile -> DestReuseBinary(mul)
MODE 2 = RAW LLK: copy_tile -> mul_reuse_dest_tiles<DEST_TO_SRCA>   (no kernel_lib at all)
Swept over DEST slot x {fp32_dest_acc_en, dst_full_sync_en}, which move DEST_AUTO_LIMIT.
"""
import os

os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import torch, ttnn

TILE = 32
CB_X, CB_G, CB_OUT = 0, 1, 9

KERNEL = r"""
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
namespace ckl = compute_kernel_lib;

constexpr uint32_t cb_x = 0, cb_g = 1, cb_out = 9;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t BLK = get_compile_time_arg_val(1);
    constexpr uint32_t SLOTV = get_compile_time_arg_val(2);
    constexpr ckl::Dst SLOT = static_cast<ckl::Dst>(SLOTV);
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
        // RAW LLK: exactly what the helper emits, written by hand.
        cb_reserve_back(cb_out, S);
        for (uint32_t i = 0; i < S; ++i) {
            tile_regs_acquire();
            copy_tile_to_dst_init_short(cb_x);
            copy_tile(cb_x, i, SLOTV);
            mul_reuse_dest_init<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g);
            mul_reuse_dest_tiles<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g, i, SLOTV);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(SLOTV, cb_out, i);
            tile_regs_release();
        }
        cb_push_back(cb_out, S);
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


def run(device, s, blk, slot, mode, fp32=False, full_sync=False):
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
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    fp32_dest_acc_en=fp32,
                    math_approx_mode=False,
                    dst_full_sync_en=full_sync,
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

    limit = (8 if full_sync else 4) if fp32 else (16 if full_sync else 8)
    tag = (
        f"MODE={'chain' if mode == 0 else 'RAW'} SLOT=D{slot} BLK={blk} "
        f"fp32={int(fp32)} full_sync={int(full_sync)} (DEST_AUTO_LIMIT={limit})"
    )
    rel = [
        (
            (a[:, t * TILE : (t + 1) * TILE] - expected[:, t * TILE : (t + 1) * TILE]).abs().max().item()
            / (expected[:, t * TILE : (t + 1) * TILE].pow(2).mean().sqrt().item() + 1e-9)
        )
        for t in range(s)
    ]
    worst = max(range(s), key=lambda i: rel[i])
    verdict = "BAD " if rel[worst] > 0.1 else "ok  "
    extra = ""
    if rel[worst] > 0.1:
        aa, ee = a[:, worst * TILE : (worst + 1) * TILE], expected[:, worst * TILE : (worst + 1) * TILE]
        gg, xx = gf[:, worst * TILE : (worst + 1) * TILE], xf[:, worst * TILE : (worst + 1) * TILE]
        rn = ee.pow(2).mean().sqrt().item() + 1e-9
        fe = [
            (aa[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] - ee[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16])
            .abs()
            .max()
            .item()
            / rn
            for r in (0, 1)
            for c in (0, 1)
        ]
        f3 = (slice(16, 32), slice(16, 32))
        noclr = (aa[f3] - (xx * (1.0 + gg))[f3]).abs().max().item()
        zero = (aa == 0).float().mean().item()
        extra = f" | faces={' '.join(f'{v:.2f}' for v in fe)}" f" | |a-x*(1+g)| on f3={noclr:.4g} | zerofrac={zero:.2f}"
    print(f"  {verdict}{tag}  worst tile {worst} rel={rel[worst]:.3f}{extra}")
    for t in (xd, gd, od):
        ttnn.deallocate(t)


device = ttnn.open_device(device_id=0)
try:
    print("\n--- RAW LLK (kernel_lib entirely out of the picture), bf16 DEST, half sync, limit=8")
    for slot in (0, 6, 7):
        run(device, 8, 1, slot, 2)
    print("\n--- fp32_dest_acc_en=True -> limit=4: does the bug move to D3?")
    for slot in (0, 2, 3, 7):
        run(device, 4, 1, slot, 2, fp32=True)
    print("\n--- dst_full_sync_en=True -> limit=16: where is it now?")
    for slot in (0, 6, 7, 8, 14, 15):
        run(device, 4, 1, slot, 2, full_sync=True)
finally:
    ttnn.close_device(device)
