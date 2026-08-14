"""RAW-LLK dest-reuse repro, no kernel_lib. Sweeps DEST slot x sync/accum mode."""
import os, sys
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

constexpr uint32_t cb_x = 0, cb_g = 1, cb_out = 9;

void kernel_main() {
    constexpr uint32_t S = get_compile_time_arg_val(0);
    constexpr uint32_t SLOT = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_x, cb_g, cb_out);
    cb_wait_front(cb_x, S);
    cb_wait_front(cb_g, S);
    cb_reserve_back(cb_out, S);
    for (uint32_t i = 0; i < S; ++i) {
        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_x);
        copy_tile(cb_x, i, SLOT);
        mul_reuse_dest_init<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g);
        mul_reuse_dest_tiles<ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_g, i, SLOT);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(SLOT, cb_out, i);
        tile_regs_release();
    }
    cb_push_back(cb_out, S);
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
        shape=shape, core_grid=crs(), strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)


def run(device, s, slot, fp32=False, full_sync=False):
    width = s * TILE
    torch.manual_seed(3)
    x = (torch.rand(TILE, width) * 2 - 1).to(torch.bfloat16)
    g = (torch.rand(TILE, width) + 0.5).to(torch.bfloat16)
    q = lambda t: t.to(torch.bfloat16).to(torch.float32)
    xf, gf = q(x), q(g)
    expected = xf * gf
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                                    memory_config=shard((TILE, width)))
    xd, gd = dev(x), dev(g)
    od = ttnn.allocate_tensor_on_device(ttnn.Shape([TILE, width]), ttnn.bfloat16, ttnn.TILE_LAYOUT,
                                        device, shard((TILE, width)))
    c = crs()
    pd = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(kernel_source=PUBLISH,
                                  source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=c, compile_time_args=[s],
                                  config=ttnn.ReaderConfigDescriptor()),
            ttnn.KernelDescriptor(kernel_source=KERNEL,
                                  source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                  core_ranges=c, compile_time_args=[s, slot],
                                  config=ttnn.ComputeConfigDescriptor(
                                      math_fidelity=ttnn.MathFidelity.HiFi2,
                                      fp32_dest_acc_en=fp32, math_approx_mode=False,
                                      dst_full_sync_en=full_sync)),
        ],
        semaphores=[],
        cbs=[ttnn.cb_descriptor_from_sharded_tensor(CB_X, xd),
             ttnn.cb_descriptor_from_sharded_tensor(CB_G, gd),
             ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, od)],
    )
    ttnn.generic_op([xd, gd, od], pd)
    ttnn.synchronize_device(device)
    a = ttnn.to_torch(od).to(torch.float32)
    limit = (8 if full_sync else 4) if fp32 else (16 if full_sync else 8)
    rn = expected.pow(2).mean().sqrt().item() + 1e-9
    rel = (a - expected).abs().max().item() / rn
    fe, noclr = "", ""
    if rel > 0.1:
        aa, ee = a[:, :TILE], expected[:, :TILE]
        gg, xx = gf[:, :TILE], xf[:, :TILE]
        fl = [(aa[r*16:(r+1)*16, cc*16:(cc+1)*16] - ee[r*16:(r+1)*16, cc*16:(cc+1)*16]).abs().max().item()/rn
              for r in (0, 1) for cc in (0, 1)]
        fe = " faces=" + " ".join(f"{v:.2f}" for v in fl)
        f3 = (slice(16, 32), slice(16, 32))
        noclr = (f" |a-x*(1+g)|@f3={(aa[f3]-(xx*(1.0+gg))[f3]).abs().max().item():.3g}"
                 f" zerofrac={(aa == 0).float().mean().item():.2f}")
    print(f"  {'BAD ' if rel > 0.1 else 'ok  '}SLOT=D{slot:<2} fp32={int(fp32)} full_sync={int(full_sync)} "
          f"limit={limit:<2} rel={rel:.3f}{fe}{noclr}")
    for t in (xd, gd, od):
        ttnn.deallocate(t)


device = ttnn.open_device(device_id=0)
try:
    print("\n--- half sync, bf16 DEST (limit 8) : baseline repro")
    for slot in (0, 6, 7):
        run(device, 2, slot)
    pass
finally:
    ttnn.close_device(device)
