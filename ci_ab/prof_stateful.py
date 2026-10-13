# Round 3 (#58757): device time of the ttnn ops that reach the Blackhole Welford update (the four-instruction row of
# #58826) and its callers' per-tile re-init (the two pre-allgather layernorm kernels), plus group norm in its Welford
# (two-pass statistics) form, run under `python -m tracy -r -p --no-web-server -o <dir> prof_stateful.py <cases>`; a
# signpost precedes the measured calls of every case. Warm-up 2 calls, measured R3_NMEAS calls.
import gc
import os
import sys

import torch
import ttnn
from tracy import signpost

if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")  # card rule of 2026-10-07 12:35 UTC; the mock cluster opens no card

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))


def ckc(dev, fp32=True):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32,
        packer_l1_acc=True,
    )


def full_grid(dev):
    g = dev.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})


def tile(t, dt, dev):
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def case_lnpre(shape, dt, residual, fp32_acc=True):
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16

    def setup(dev):
        torch.manual_seed(0)
        x = tile(torch.randn(shape, dtype=tdt), dt, dev)
        r = tile(torch.randn(shape, dtype=tdt), dt, dev) if residual else None
        recip = ttnn.create_layer_norm_reciprocals(dev, full_grid(dev), shape[-1])
        return (x, r, recip, ckc(dev, fp32_acc))

    def run(dev, x, r, recip, cfg):
        kw = {"residual_input_tensor": r} if r is not None else {}
        return ttnn.layer_norm_pre_all_gather(
            x,
            dtype=dt,
            compute_kernel_config=cfg,
            program_config=ttnn.LayerNormDefaultProgramConfig(use_welford=True),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            recip_tensor=recip,
            **kw,
        )

    return setup, run


def case_ditpre(shape, dt):
    tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16

    def setup(dev):
        torch.manual_seed(0)
        x = tile(torch.randn(shape, dtype=tdt), dt, dev)
        recip = ttnn.create_layer_norm_reciprocals(dev, full_grid(dev), shape[-1])
        return (x, recip, ckc(dev))

    def run(dev, x, recip, cfg):
        return ttnn.experimental.dit_layernorm_pre_allgather(x, recip, compute_kernel_config=cfg)

    return setup, run


def case_gn(N, C, H, W, G, nob, cy, cx):
    def setup(dev):
        torch.manual_seed(0)
        grid = ttnn.CoreGrid(y=cy, x=cx)
        t = torch.rand((N, C, H, W), dtype=torch.bfloat16).permute(0, 2, 3, 1).reshape(N, 1, W * H, C)
        x = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        x = ttnn.tilize_with_zero_padding(x, use_multicore=True)
        [g, b], m = ttnn.dram_group_norm_params_from_torch(
            [torch.rand((C,), dtype=torch.bfloat16), torch.rand((C,), dtype=torch.bfloat16)],
            C,
            G,
            dev,
            core_grid=grid,
            return_mask=True,
        )
        return (x, g, b, m, grid)

    def run(dev, x, g, b, m, grid):
        return ttnn.group_norm(
            x,
            num_groups=G,
            input_mask=m,
            weight=g,
            bias=b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=grid,
            inplace=False,
            num_out_blocks=nob,
            use_welford=True,
        )

    return setup, run


CASES = {}
for shp, sn in [((1, 1, 32, 1024), "32x1024"), ((1, 1, 256, 2048), "256x2048"), ((1, 1, 1024, 4096), "1024x4096")]:
    for dt, dn in [(ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")]:
        CASES[f"lnpre_{dn}_{sn}"] = case_lnpre(shp, dt, False)
        CASES[f"lnpre_res_{dn}_{sn}"] = case_lnpre(shp, dt, True)
    CASES[f"lnpre_res_bf16nf_{sn}"] = case_lnpre(shp, ttnn.bfloat16, True, False)


def case_ema(B, C, T, grid):
    def setup(dev):
        torch.manual_seed(0)
        x = tile(torch.rand((1, B, C, T), dtype=torch.bfloat16), ttnn.bfloat16, dev)
        return (x,)

    def run(dev, x):
        return ttnn.ema(x, alpha=0.3, core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]) if grid else None)

    return setup, run


CASES["ema_2x4096x2048"] = case_ema(2, 4096, 2048, None)
CASES["ema_2x4096x2048_g4x4"] = case_ema(2, 4096, 2048, (4, 4))
CASES["ema_1x1024x512"] = case_ema(1, 1024, 512, None)
# DiT pre-allgather: FLUX (3072 / 4 devices) and WAN (5120 / 4 devices) widths per device, bf16 (the models), fp32
for seq in (512, 4096):
    for w in (768, 1280):
        CASES[f"ditpre_bf16_{seq}x{w}"] = case_ditpre((1, 1, seq, w), ttnn.bfloat16)
CASES["ditpre_fp32_512x768"] = case_ditpre((1, 1, 512, 768), ttnn.float32)
CASES["ditpre_fp32_4096x1280"] = case_ditpre((1, 1, 4096, 1280), ttnn.float32)
# group norm, Welford statistics, SDXL / SD 1.4 VAE shapes of test_group_norm_DRAM (N, C, H, W, G, nob, cy, cx)
CASES["gn_1x512x64x64"] = case_gn(1, 512, 64, 64, 32, 1, 8, 8)
CASES["gn_1x256x256x256"] = case_gn(1, 256, 256, 256, 32, 8, 8, 8)
CASES["gn_1x1920x16x16"] = case_gn(1, 1920, 16, 16, 32, 1, 4, 4)


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0, l1_small_size=0)
    try:
        for name in names:
            setup, run = CASES[name]
            args = None
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
