# Softmax production shapes (scratch, not for merge), under `python -m tracy -r -p --no-web-server -o <dir> prof_sm.py <case>...`:
# per case 2 warm-up calls, a signpost "<case>", SM_NMEAS measured calls. dram = DRAM interleaved, l1 = L1 interleaved.
import gc
import hashlib
import os
import sys

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("SM_NMEAS", "12"))
DT = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "fp32": ttnn.float32}
MEM = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}


def tens(d, shape, dt, mem, seed=70):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * 4.0
    return ttnn.from_torch(x, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=d, memory_config=mem)


def cfg(d, fid=ttnn.MathFidelity.HiFi4, fp32=False):
    return ttnn.init_device_compute_kernel_config(d.arch(), math_fidelity=fid, math_approx_mode=False,
                                                  fp32_dest_acc_en=fp32, packer_l1_acc=False)


def inter(shape, dt, ns, mem="dram", fp32=False, default_cfg=True):
    def setup(d):
        return (tens(d, shape, dt, MEM[mem]),)

    def run(d, x):
        if default_cfg and not fp32:
            return ttnn.softmax(x, dim=-1, numeric_stable=ns)
        return ttnn.softmax(x, dim=-1, numeric_stable=ns, compute_kernel_config=cfg(d, fp32=fp32))

    return setup, run


def sharded(b, h, s, w, dt, ns, fp32=False):
    def setup(d):
        shape = (b, h, s, w)
        mem = ttnn.create_sharded_memory_config(shape, core_grid=ttnn.CoreGrid(y=h, x=b),
                                                strategy=ttnn.ShardStrategy.HEIGHT,
                                                orientation=ttnn.ShardOrientation.ROW_MAJOR)
        return (tens(d, shape, dt, mem),)

    def run(d, x):
        pc = ttnn.SoftmaxShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(b, h),
                                                       subblock_w=6 if not fp32 else 3, block_h=s // 32, block_w=w // 32)
        y = ttnn.clone(x)
        return ttnn.scale_mask_softmax_in_place(y, program_config=pc, compute_kernel_config=cfg(d, fp32=fp32),
                                                numeric_stable=ns)

    return setup, run


CASES = {}
for dt in ("bf8", "bf16"):
    for ns in (0, 1):
        CASES[f"seg1_{dt}_ns{ns}"] = inter((1, 1, 16384, 256), dt, bool(ns))
        CASES[f"seg4_{dt}_ns{ns}"] = inter((1, 8, 256, 256), dt, bool(ns))
    CASES[f"seg2_{dt}_ns0"] = inter((1, 2, 4096, 256), dt, False)
    CASES[f"seg3_{dt}_ns0"] = inter((1, 5, 1024, 256), dt, False)
    CASES[f"seg1l1_{dt}_ns0"] = inter((1, 1, 16384, 256), dt, False, mem="l1")
    for ns in (0, 1):
        CASES[f"sh_{dt}_ns{ns}"] = sharded(8, 8, 384, 384, dt, bool(ns))
CASES["qwen_router_32x256"] = inter((1, 1, 32, 256), "bf16", True)
CASES["qwen_router_2048x256"] = inter((1, 1, 2048, 256), "bf16", True)
CASES["gemma_router_32x128"] = inter((1, 1, 32, 128), "bf16", True)
CASES["clip_l_1x12x77x77_fp32acc"] = inter((1, 12, 77, 77), "bf16", True, fp32=True, default_cfg=False)
CASES["t5_fp32_1x16x256x256"] = inter((1, 16, 256, 256), "fp32", True, fp32=True, default_cfg=False)
CASES["unit_1x32x512x512"] = inter((1, 32, 512, 512), "bf16", True)
CASES["unit_1x8x1024x1024"] = inter((1, 8, 1024, 1024), "bf16", True)


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    y = run(dev, *args)
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                t = ttnn.to_torch(y).contiguous()
                h = hashlib.md5(t.view(torch.int16 if t.element_size() == 2 else torch.int32).numpy().tobytes()).hexdigest()[:12]
                print(f"BITS {name} {h}", flush=True)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
