"""Refinement 1, part 2:
  (a) is the launch/sync FLOOR core-count dependent?  (the A0 clause's premise:
      "dispatch/sync cost scales with the core count")  -> sync_only vs core cap.
  (b) where does C16 depth-2 actually pay?  -> depth1 vs depth2 across the bench
      regimes, with blocks-per-core reported so the gate can key on it.
"""
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics
import torch
import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as tpd

_KEY = "DEVICE KERNEL DURATION [ns]"
N_WARMUP, N_TRIALS, N_ROUNDS = 3, 10, 5
_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR


def _crs(ex, ey):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ex, ey))})


def read_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for p in programs:
            res = getattr(p, "program_analyses_results", None) or {}
            e = res.get(_KEY)
            if e is not None:
                total += float(e.duration)
                found = True
    return total if found else None


def measure(device, fn):
    for _ in range(N_WARMUP):
        fn()
    ttnn.synchronize_device(device)
    read_ns(device)
    s = []
    for _ in range(N_ROUNDS):
        for _ in range(N_TRIALS):
            fn()
        v = read_ns(device)
        if v is not None:
            s.append(v / N_TRIALS)
    if not s:
        return None, None
    return statistics.median(s), (statistics.stdev(s) if len(s) > 1 else 0.0)


device = ttnn.open_device(device_id=0)
try:
    # ---------- (a) sync_only floor vs core count ----------
    shape = (1, 1, 2048, 32)
    torch.manual_seed(0)
    ti = ttnn.from_torch(
        torch.randn(shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("\n=== (a) d_tall_narrow: sync_only + full vs core cap ===")
    print(f"{'cap':>5}{'cores':>6}{'blk/core':>9}{'sync_only':>11}{'no_dm':>9}{'full':>9}")
    for cap in [64, 32, 16, 8, 4]:
        tpd.CORE_CAP_OVERRIDE = cap
        po = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        plan = tpd.build_plan(ti, po, device, use_multicore=True)
        bpc = max(u["row_count"] * u["chunk_count"] for u in plan["work"])
        out = {}
        for label, sdm, sc in (("sync_only", "1", "1"), ("no_dm", "1", "0"), ("full", "0", "0")):
            os.environ["TILIZE_SKIP_DM"] = sdm
            os.environ["TILIZE_SKIP_COMPUTE"] = sc
            ns, _ = measure(device, lambda t=ti: tilize(t, ttnn.DRAM_MEMORY_CONFIG))
            out[label] = ns
        os.environ["TILIZE_SKIP_DM"] = "0"
        os.environ["TILIZE_SKIP_COMPUTE"] = "0"
        print(
            f"{cap:>5}{plan['ncores']:>6}{bpc:>9}{out['sync_only']:>11.0f}" f"{out['no_dm']:>9.0f}{out['full']:>9.0f}"
        )
    tpd.CORE_CAP_OVERRIDE = None

    # ---------- (b) depth-1 vs depth-2 across regimes ----------
    REG = {
        "a_square": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16),
        "b_wide_short": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16),
        "c_single_core": dict(shape=(1, 1, 512, 512), dtype=ttnn.bfloat16, mc=False),
        "d_tall_narrow": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16),
        "e_square_fp32": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.float32),
        "g_dram_to_shard": dict(
            shape=(1, 1, 2048, 512),
            dtype=ttnn.bfloat16,
            out_cfg=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, _L1, ttnn.ShardSpec(_crs(7, 7), (256, 64), _ROW)
            ),
        ),
        "g_shard_to_dram": dict(
            shape=(1, 1, 2048, 512),
            dtype=ttnn.bfloat16,
            in_cfg=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, _L1, ttnn.ShardSpec(_crs(7, 7), (256, 64), _ROW)
            ),
        ),
        "x_wide_1core": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, mc=False),
        "n_tiny": dict(shape=(1, 1, 64, 128), dtype=ttnn.bfloat16),
    }
    print("\n=== (b) C16: depth-1 vs depth-2 ===")
    print(
        f"{'regime':<17}{'cores':>6}{'chk':>4}{'blk/core':>9}"
        f"{'cbB d2':>9}{'ns d2':>9}{'cbB d1':>9}{'ns d1':>9}{'d1/d2':>7}"
    )
    for name, s in REG.items():
        in_cfg = s.get("in_cfg", ttnn.DRAM_MEMORY_CONFIG)
        out_cfg = s.get("out_cfg", ttnn.DRAM_MEMORY_CONFIG)
        dt = s["dtype"]
        torch.manual_seed(0)
        t = torch.randn(s["shape"], dtype=torch.float32) if dt == ttnn.float32 else torch.randn(s["shape"]).bfloat16()
        tin = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg)
        res = {}
        for depth_flag in (True, False):
            po = ttnn.allocate_tensor_on_device(ttnn.Shape(list(s["shape"])), dt, ttnn.TILE_LAYOUT, device, out_cfg)
            plan = tpd.build_plan(tin, po, device, use_multicore=s.get("mc", True), use_double_buffer=depth_flag)
            bpc = (
                max(u["row_count"] * u["chunk_count"] for u in plan["work"])
                if plan["path"] == "generic"
                else plan["num_blocks"]
            )
            ns, _ = measure(
                device,
                lambda x=tin, c=out_cfg, d=depth_flag, m=s.get("mc", True): tilize(
                    x, c, use_multicore=m, use_double_buffer=d
                ),
            )
            res[depth_flag] = (plan["ncores"], plan["chunk_wt"], bpc, plan["cb_bytes_per_core"], ns)
        c2 = res[True]
        c1 = res[False]
        print(
            f"{name:<17}{c2[0]:>6}{c2[1]:>4}{c2[2]:>9}{c2[3]:>9}{c2[4]:>9.0f}"
            f"{c1[3]:>9}{c1[4]:>9.0f}{c1[4] / c2[4]:>7.3f}"
        )
finally:
    ttnn.close_device(device)
