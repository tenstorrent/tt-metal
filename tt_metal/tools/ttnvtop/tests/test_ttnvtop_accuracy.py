# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# ttnvtop accuracy regression: run a matmul of known FLOPs and assert that
# ttnvtop's LoFi-equivalent TFLOPs reading is within tolerance of ttnn's
# analytical achieved TFLOPs (2*M*K*N / elapsed).
#
# Requires ttnvtop-collector to be running (writes /dev/shm/tt_device_<id>_util).
# Tests are SKIPPED if the SHM file is absent — keeps CI green until the
# collector is part of the standard test runner.
#
# Run: pytest -xvs tt_metal/tools/ttnvtop/tests/test_ttnvtop_accuracy.py

import os
import struct
import time

import pytest
import torch
import ttnn

SHM_PATH_CHIP0 = "/dev/shm/tt_device_0_util"
SHM_PATH_CHIP1 = "/dev/shm/tt_device_1_util"
HEADER_FMT = "<4sHHQIIQQIIII4I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
PER_CORE_FMT = "<6B10H2x3I"
PER_CORE_SIZE = struct.calcsize(PER_CORE_FMT)


def _read_shm(path):
    """Return (aiclk_mhz, n_cores, est_tflops_lofi_equivalent, avg_fpu_frac, avg_sfpu_frac).

    PerCoreView tuple layout from struct.unpack("<6B10H2x3I"):
      r[0..5]  = noc_x, noc_y, logical_x, logical_y, is_remote, dispatched
      r[6]     = sfpu_busy_p1000   (Phase 2.1.d)
      r[7]     = dispatch_busy_p1000
      r[8]     = compute_busy_p1000  (FPU)
      r[9..15] = unpack, pack, stall, noc0_in, noc0_out, noc1_in, noc1_out
      r[16..18]= samples_seen, last_kernel_id, reserved_1
    """
    with open(path, "rb") as f:
        data = f.read()
    hdr = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    aiclk_mhz = hdr[11]
    n_cores = hdr[8]
    sum_fpu_p1000 = 0
    sum_sfpu_p1000 = 0
    for i in range(n_cores):
        r = struct.unpack(PER_CORE_FMT, data[HEADER_SIZE + i * PER_CORE_SIZE : HEADER_SIZE + (i + 1) * PER_CORE_SIZE])
        sum_sfpu_p1000 += r[6]
        sum_fpu_p1000 += r[8]
    avg_fpu_frac = sum_fpu_p1000 / n_cores / 1000.0
    avg_sfpu_frac = sum_sfpu_p1000 / n_cores / 1000.0
    # Viewer's formula: peak_greq_per_s = n_cores * aiclk_mhz / 1000
    #                   peak_tflops = peak_greq_per_s * 4096 / 1000  (LoFi: 4096 muladds per FPU req = 8192 FLOPs)
    peak_tflops = n_cores * aiclk_mhz / 1000.0 * 4096 / 1000.0
    est_tflops = peak_tflops * avg_fpu_frac
    return aiclk_mhz, n_cores, est_tflops, avg_fpu_frac, avg_sfpu_frac


def _require_collector():
    if not os.path.exists(SHM_PATH_CHIP0):
        pytest.skip(f"ttnvtop collector not running (no {SHM_PATH_CHIP0}). Start with `ttnvtop-collector &` first.")


def _run_matmul_and_measure(device, program_config, compute_kernel_config, dtype, M, K, N, seconds=30):
    """Run matmul in a hot loop for `seconds`, sampling ttnvtop SHM every 3s. Return ttnn_tflops, avg_ttnvtop_tflops."""
    flops_per_iter = 2 * M * K * N
    in0 = torch.randn((1, 1, M, K)).bfloat16()
    in1 = torch.randn((1, 1, K, N)).bfloat16()
    in0_t = ttnn.from_torch(in0, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    in1_t = ttnn.from_torch(in1, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # warmup
    for _ in range(10):
        ttnn.matmul(in0_t, in1_t, program_config=program_config, compute_kernel_config=compute_kernel_config)
    ttnn.synchronize_device(device)

    samples = []
    sfpu_samples = []
    start = time.time()
    last_sample = start
    iters = 0
    while time.time() - start < seconds:
        ttnn.matmul(in0_t, in1_t, program_config=program_config, compute_kernel_config=compute_kernel_config)
        iters += 1
        now = time.time()
        # Sample SHM every 3s, skip first 3s to let EWMA settle after warmup.
        if now - last_sample >= 3.0 and now - start >= 3.0:
            _, _, est_tflops, _, avg_sfpu = _read_shm(SHM_PATH_CHIP0)
            samples.append(est_tflops)
            sfpu_samples.append(avg_sfpu)
            last_sample = now

    ttnn.synchronize_device(device)
    elapsed = time.time() - start
    ttnn_tflops = iters * flops_per_iter / elapsed / 1e12
    avg_ttnvtop_tflops = sum(samples) / len(samples) if samples else 0.0
    avg_sfpu_frac = sum(sfpu_samples) / len(sfpu_samples) if sfpu_samples else 0.0
    return ttnn_tflops, avg_ttnvtop_tflops, len(samples), avg_sfpu_frac


def test_ttnvtop_idle_chip_reads_zero():
    """Chip 1 should be idle while chip 0 is being used by this test's device open."""
    _require_collector()
    if not os.path.exists(SHM_PATH_CHIP1):
        pytest.skip("N300 not present (no chip 1)")
    _, _, est_tflops, avg_frac, avg_sfpu = _read_shm(SHM_PATH_CHIP1)
    # No workload pinned to chip 1 from this test. Allow 1% noise floor on each pipe.
    assert avg_frac < 0.01, f"idle chip 1 reports FPU compute% = {avg_frac:.3f}, expected ~0"
    assert avg_sfpu < 0.01, f"idle chip 1 reports SFPU% = {avg_sfpu:.3f}, expected ~0"
    assert est_tflops < 1.0, f"idle chip 1 reports {est_tflops:.2f} TF, expected ~0"


def test_ttnvtop_lofi_matmul_within_15pct():
    """LoFi + bfloat8_b matmul: ttnvtop's LoFi-equivalent TFLOPs should match ttnn's
    analytical TFLOPs within ±15% (aiclk can throttle and per-core saturation
    varies run-to-run).
    """
    _require_collector()

    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=3855488)
    try:
        cg = device.compute_with_storage_grid_size()
        grid = (cg.x, cg.y)

        M, K, N = 32 * 16 * grid[1], 32 * 16 * grid[0], 32 * 16 * grid[0]
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=K // grid[0] // 32,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=M // grid[1] // 32,
            per_core_N=N // grid[0] // 32,
            transpose_mcast=False,
            fused_activation=None,
        )
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        ttnn_tf, tt_tf, n_samples, avg_sfpu = _run_matmul_and_measure(
            device, program_config, compute_kernel_config, ttnn.bfloat8_b, M, K, N, seconds=30
        )
    finally:
        ttnn.close_device(device)

    print(f"\n[ttnvtop-accuracy] LoFi bf8b {M}x{K}x{N} grid={grid}")
    print(f"[ttnvtop-accuracy]   ttnn analytical: {ttnn_tf:.2f} TF")
    print(f"[ttnvtop-accuracy]   ttnvtop measured: {tt_tf:.2f} TF  (mean of {n_samples} samples)")
    print(f"[ttnvtop-accuracy]   ratio: {tt_tf / ttnn_tf:.3f}×  sfpu%={avg_sfpu*100:.2f}")

    assert n_samples >= 3, f"too few ttnvtop samples ({n_samples}), collector may not be running"
    ratio = tt_tf / ttnn_tf
    assert 0.85 <= ratio <= 1.15, (
        f"ttnvtop ({tt_tf:.2f} TF) vs ttnn ({ttnn_tf:.2f} TF) ratio {ratio:.3f}× "
        f"outside [0.85, 1.15] tolerance. ttnvtop FPU-counter calibration has drifted."
    )


def test_ttnvtop_hifi2_matmul_overreports_by_2x():
    """HiFi2 uses 2× the FPU cycles per tile as LoFi for the same FLOPs.
    ttnvtop reports LoFi-equivalent TFLOPs, so a HiFi2 workload should read
    ~2× ttnn's analytical TFLOPs. This test pins that expected definitional
    gap so any change to the calibration constant doesn't silently break it.

    Uses ttnn.matmul with no program_config (which defaults to HiFi2 for bf16).

    After Phase 2.1.c (set TTNVTOP_PHASE21C_FLIPS=1):
      - HiFi2 ratio expected to drop to ~1.0 (collector now divides by fidelity cycles)
      - Without env var, asserts the current 2.0× behavior
    """
    _require_collector()

    device = ttnn.open_device(device_id=0)
    try:
        M, K, N = 4096, 4096, 4096
        flops_per_iter = 2 * M * K * N
        in0 = torch.randn((1, 1, M, K)).bfloat16()
        in1 = torch.randn((1, 1, K, N)).bfloat16()
        in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        for _ in range(5):
            ttnn.matmul(in0_t, in1_t)
        ttnn.synchronize_device(device)

        samples = []
        start = time.time()
        last_sample = start
        iters = 0
        while time.time() - start < 30:
            ttnn.matmul(in0_t, in1_t)
            iters += 1
            now = time.time()
            if now - last_sample >= 3.0 and now - start >= 3.0:
                _, _, est_tflops, _, _ = _read_shm(SHM_PATH_CHIP0)
                samples.append(est_tflops)
                last_sample = now

        ttnn.synchronize_device(device)
        elapsed = time.time() - start
        ttnn_tf = iters * flops_per_iter / elapsed / 1e12
        tt_tf = sum(samples) / len(samples) if samples else 0.0
        n_samples = len(samples)
    finally:
        ttnn.close_device(device)

    phase21c = os.environ.get("TTNVTOP_PHASE21C_FLIPS") == "1"
    expected_str = "expected ~1.0 (Phase 2.1.c)" if phase21c else "expected ~2.0"
    print(f"\n[ttnvtop-accuracy] HiFi2 (default) bf16 {M}x{K}x{N}")
    print(f"[ttnvtop-accuracy]   ttnn analytical: {ttnn_tf:.2f} TF")
    print(f"[ttnvtop-accuracy]   ttnvtop measured: {tt_tf:.2f} TF  (mean of {n_samples} samples)")
    print(f"[ttnvtop-accuracy]   ratio: {tt_tf / ttnn_tf:.3f}× ({expected_str})")

    assert n_samples >= 3, f"too few ttnvtop samples ({n_samples})"
    ratio = tt_tf / ttnn_tf
    if phase21c:
        # Phase 2.1.c: collector divides by fidelity cycles, so HiFi2 should
        # match analytical TFLOPs within ±15% just like LoFi.
        assert 0.85 <= ratio <= 1.15, (
            f"HiFi2 (Phase 2.1.c) ratio {ratio:.3f}× outside [0.85, 1.15]. Expected ~1.0 — "
            f"the fidelity-aware collector should make HiFi2 read like LoFi."
        )
    else:
        assert 1.7 <= ratio <= 2.3, (
            f"HiFi2 ratio {ratio:.3f}× outside [1.7, 2.3]. Either the kernel switched math fidelity, "
            f"or the FPU counter now counts something different — investigate before touching the calibration constant. "
            f"(If Phase 2.1.c just landed, set TTNVTOP_PHASE21C_FLIPS=1 to flip the expected ratio to ~1.0.)"
        )


def _run_matmul_with_fidelity(fidelity, seconds=25):
    """Run 4096^3 bfloat16 matmul with specified math_fidelity, measure both
    ttnn analytical TFLOPs and ttnvtop's reported TFLOPs. Returns (ttnn_tf, ttnvtop_tf, avg_compute_frac, n_samples).

    Uses compute_kernel_config without a program_config — ttnn picks a default
    matmul kernel, which accepts the fidelity override cleanly.
    """
    device = ttnn.open_device(device_id=0)
    try:
        M, K, N = 4096, 4096, 4096
        flops_per_iter = 2 * M * K * N
        in0 = torch.randn((1, 1, M, K)).bfloat16()
        in1 = torch.randn((1, 1, K, N)).bfloat16()
        in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=fidelity, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
        )

        for _ in range(5):
            ttnn.matmul(in0_t, in1_t, compute_kernel_config=ckc)
        ttnn.synchronize_device(device)

        tflops_samples = []
        compute_samples = []
        start = time.time()
        last_sample = start
        iters = 0
        while time.time() - start < seconds:
            ttnn.matmul(in0_t, in1_t, compute_kernel_config=ckc)
            iters += 1
            now = time.time()
            if now - last_sample >= 3.0 and now - start >= 3.0:
                _, _, est_tflops, avg_frac, _ = _read_shm(SHM_PATH_CHIP0)
                tflops_samples.append(est_tflops)
                compute_samples.append(avg_frac)
                last_sample = now

        ttnn.synchronize_device(device)
        elapsed = time.time() - start
        ttnn_tf = iters * flops_per_iter / elapsed / 1e12
        ttnvtop_tf = sum(tflops_samples) / len(tflops_samples) if tflops_samples else 0.0
        avg_compute = sum(compute_samples) / len(compute_samples) if compute_samples else 0.0
        return ttnn_tf, ttnvtop_tf, avg_compute, len(tflops_samples)
    finally:
        ttnn.close_device(device)


def test_ttnvtop_overcount_scales_with_fidelity():
    """Directly verify the thesis that explains the 2× HiFi2 over-report:

    ttnvtop's hardware FPU counter ticks per FPU-busy cycle. The viewer converts
    cycles→TFLOPs assuming LoFi rates. So the overcount factor is exactly the
    ratio of fidelity cycles/tile to LoFi cycles/tile:

        overcount = ttnvtop_tf / ttnn_tf
        LoFi  → ~1.0   (16 cycles/tile ÷ 16 = 1)
        HiFi2 → ~2.0   (32 cycles/tile ÷ 16 = 2)
        HiFi4 → ~4.0   (64 cycles/tile ÷ 16 = 4)

    This is a *workload-independent* invariant: it holds whether the kernel is
    compute-bound (ttnn throughput differs, FPU% same) or memory-bound (ttnn
    throughput same, FPU% differs). Either way, the FPU spends cycle_ratio more
    cycles per unit FLOPs at higher fidelity than LoFi, and ttnvtop over-reports
    by exactly that ratio.

    Testing the ratio-of-ratios is the cleanest way to pin the semantic without
    assuming a compute-bound workload. Any code change that makes ttnvtop
    fidelity-aware must update this test — the expected hifi/lofi overcount
    ratios drop from ~2.0/~4.0 toward ~1.0.

    After Phase 2.1.c (set TTNVTOP_PHASE21C_FLIPS=1):
      - HiFi2/LoFi and HiFi4/LoFi overcount ratios expected to drop to ~1.0
        (collector now divides by fidelity cycles)
      - Without env var, asserts the current 2.0× / 4.0× behavior
    """
    _require_collector()

    lofi_ttnn, lofi_ttnvtop, lofi_c, lofi_n = _run_matmul_with_fidelity(ttnn.MathFidelity.LoFi)
    hifi2_ttnn, hifi2_ttnvtop, hifi2_c, hifi2_n = _run_matmul_with_fidelity(ttnn.MathFidelity.HiFi2)
    hifi4_ttnn, hifi4_ttnvtop, hifi4_c, hifi4_n = _run_matmul_with_fidelity(ttnn.MathFidelity.HiFi4)
    lofi_overcount = lofi_ttnvtop / lofi_ttnn
    hifi2_overcount = hifi2_ttnvtop / hifi2_ttnn
    hifi4_overcount = hifi4_ttnvtop / hifi4_ttnn
    hifi2_vs_lofi = hifi2_overcount / lofi_overcount
    hifi4_vs_lofi = hifi4_overcount / lofi_overcount

    print(
        f"\n[overcount-scales] LoFi : ttnn={lofi_ttnn:.2f}TF  ttnvtop={lofi_ttnvtop:.2f}TF  compute%={lofi_c*100:.1f}  overcount={lofi_overcount:.2f}×"
    )
    print(
        f"[overcount-scales] HiFi2: ttnn={hifi2_ttnn:.2f}TF  ttnvtop={hifi2_ttnvtop:.2f}TF  compute%={hifi2_c*100:.1f}  overcount={hifi2_overcount:.2f}×"
    )
    print(
        f"[overcount-scales] HiFi4: ttnn={hifi4_ttnn:.2f}TF  ttnvtop={hifi4_ttnvtop:.2f}TF  compute%={hifi4_c*100:.1f}  overcount={hifi4_overcount:.2f}×"
    )
    print(f"[overcount-scales] HiFi2/LoFi overcount ratio = {hifi2_vs_lofi:.2f}  (expected ~2.0 — 32/16 cycles/tile)")
    print(f"[overcount-scales] HiFi4/LoFi overcount ratio = {hifi4_vs_lofi:.2f}  (expected ~4.0 — 64/16 cycles/tile)")

    assert (
        lofi_n >= 3 and hifi2_n >= 3 and hifi4_n >= 3
    ), f"too few samples (lofi={lofi_n}, hifi2={hifi2_n}, hifi4={hifi4_n})"

    phase21c = os.environ.get("TTNVTOP_PHASE21C_FLIPS") == "1"

    if phase21c:
        # Phase 2.1.c: collector divides by fidelity cycles, so HiFi2 and HiFi4
        # should look just like LoFi — overcount ratios collapse to ~1.0.
        assert 0.80 <= hifi2_vs_lofi <= 1.25, (
            f"HiFi2/LoFi (Phase 2.1.c) ratio {hifi2_vs_lofi:.2f} outside [0.80, 1.25]. Expected ~1.0. "
            f"Fidelity-aware collector should make HiFi2 read like LoFi."
        )
        assert 0.80 <= hifi4_vs_lofi <= 1.25, (
            f"HiFi4/LoFi (Phase 2.1.c) ratio {hifi4_vs_lofi:.2f} outside [0.80, 1.25]. Expected ~1.0. "
            f"Fidelity-aware collector should make HiFi4 read like LoFi."
        )
    else:
        # Primary invariant 1: HiFi2 overcount is ~2× LoFi's. Workload-independent.
        assert 1.7 <= hifi2_vs_lofi <= 2.3, (
            f"HiFi2/LoFi overcount ratio {hifi2_vs_lofi:.2f} outside [1.7, 2.3]. Expected ~2.0. "
            f"If this changes, either ttnvtop became fidelity-aware, the FPU counter semantics "
            f"changed, or HiFi2 no longer uses 2× the cycles of LoFi. "
            f"(If Phase 2.1.c just landed, set TTNVTOP_PHASE21C_FLIPS=1 to flip the expected ratio to ~1.0.)"
        )

        # Primary invariant 2: HiFi4 overcount is ~4× LoFi's. Same cycle-ratio logic.
        assert 3.4 <= hifi4_vs_lofi <= 4.6, (
            f"HiFi4/LoFi overcount ratio {hifi4_vs_lofi:.2f} outside [3.4, 4.6]. Expected ~4.0. "
            f"If this changes, either ttnvtop became fidelity-aware, the FPU counter semantics "
            f"changed, or HiFi4 no longer uses 4× the cycles of LoFi. "
            f"(If Phase 2.1.c just landed, set TTNVTOP_PHASE21C_FLIPS=1 to flip the expected ratio to ~1.0.)"
        )

    # Secondary: LoFi overcount should be near 1.0 — ttnvtop matches reality when workload matches its assumption.
    # This invariant holds in BOTH worlds: pre-2.1.c LoFi was already accurate; post-2.1.c, all fidelities are.
    assert 0.85 <= lofi_overcount <= 1.25, (
        f"LoFi overcount {lofi_overcount:.2f} outside [0.85, 1.25]. "
        f"ttnvtop's LoFi calibration constant has drifted from real LoFi FPU throughput."
    )


def test_ttnvtop_full_grid_peak_via_eth_dispatch():
    """Push all 64 Tensix cores (8×8 full grid) into a LoFi matmul and verify
    ttnvtop reads the resulting high throughput correctly.

    The default device config reserves 8 Tensix cores for dispatch, leaving
    only 7×8=56 for compute. Opening the device with `dispatch_core_type=ETH`
    moves dispatch to active ethernet cores, freeing the full 8×8 Tensix grid.

    Peak WH LoFi @ 1 GHz, all 64 cores = 64 × 4096 muladds/cycle × 1e9 / 1e12
    = 262 TF. With thermal throttle (~700 MHz sustained) the expected
    achievable is ~180 TF. This test asserts we clear a conservative floor
    and that ttnvtop tracks ttnn within 15%.
    """
    _require_collector()

    device = ttnn.open_device(
        device_id=0,
        l1_small_size=24576,
        trace_region_size=3855488,
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    )
    try:
        cg = device.compute_with_storage_grid_size()
        grid = (cg.x, cg.y)
        if grid[0] < 8 or grid[1] < 8:
            pytest.skip(
                f"ETH dispatch did not expose full 8×8 grid (got {grid}). "
                f"Likely the current arch/board config reserves some Tensix cores even with ETH dispatch."
            )

        # 4096^3 on 8×8: per_core_M = 4096/8/32 = 16 tiles.
        M, K, N = 4096, 4096, 4096
        TILE = 32
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=K // 8 // TILE,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=M // 8 // TILE,
            per_core_N=N // 8 // TILE,
            transpose_mcast=False,
            fused_activation=None,
        )
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        ttnn_tf, tt_tf, n_samples, _ = _run_matmul_and_measure(
            device, program_config, compute_kernel_config, ttnn.bfloat8_b, M, K, N, seconds=45
        )
    finally:
        ttnn.close_device(device)

    print(f"\n[full-grid-peak] 8×8 grid LoFi bf8b {M}×{K}×{N}")
    print(f"[full-grid-peak]   ttnn analytical:  {ttnn_tf:.2f} TF")
    print(f"[full-grid-peak]   ttnvtop measured: {tt_tf:.2f} TF  (mean of {n_samples} samples)")
    print(f"[full-grid-peak]   ratio:            {tt_tf / ttnn_tf:.3f}×  (expected ~1.0)")

    assert n_samples >= 5, f"too few ttnvtop samples ({n_samples})"

    # 1. Floor-check that the 8×8 grid produced real work. Thermal throttle
    # caps the 56→64 uplift to a few %; observed 96 TF vs 93 TF on 56 cores.
    # The grid assertion above already verified we got 64 cores; this floor
    # just guards against a silent kernel fallback to a slow path.
    assert ttnn_tf > 80.0, (
        f"Full 8×8 grid LoFi only achieved {ttnn_tf:.1f} TF — suspiciously low. "
        f"Likely the kernel fell back to a slower path, or the chip is severely thermally throttled."
    )

    # 2. Primary: ttnvtop tracks ttnn within ±15% at peak throughput.
    ratio = tt_tf / ttnn_tf
    assert 0.85 <= ratio <= 1.15, (
        f"ttnvtop ({tt_tf:.2f} TF) vs ttnn ({ttnn_tf:.2f} TF) ratio {ratio:.3f}× "
        f"outside [0.85, 1.15] tolerance at peak. Calibration issue surfaces at high utilization."
    )


# ---------------------------------------------------------------------------
# Phase 2.1.d: bfp4 matmul + SFPU counter coverage.
#
# The matmul tests above pin the FPU side. These four tests pin:
#   - bfp4 matmul tracks analytical ttnn TFLOPs (new dtype path)
#   - matmul lights F, not S  (proves the bars are independent)
#   - ttnn.exp lights S, not F  (proves SFPU counter actually reflects SFPU activity)
#   - ttnn.softmax lights both  (mixed realistic workload)
#
# If any of these flip, the alternating counter_sel state machine in the
# collector probably regressed, or someone broke the shm_schema offset.
# ---------------------------------------------------------------------------


def _run_sfpu_and_measure(device, fn, shape, seconds=20):
    """Hot-loop a pure-SFPU unary on a tensor of the given shape. Sample SHM
    every 3s after a 3s EWMA-settle delay. Returns (avg_fpu_frac, avg_sfpu_frac, n_samples).
    """
    H, W = shape
    x = torch.randn((1, 1, H, W)).bfloat16()
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    for _ in range(5):
        fn(x_t)
    ttnn.synchronize_device(device)

    fpu_samples = []
    sfpu_samples = []
    start = time.time()
    last_sample = start
    while time.time() - start < seconds:
        fn(x_t)
        now = time.time()
        if now - last_sample >= 3.0 and now - start >= 3.0:
            _, _, _, avg_fpu, avg_sfpu = _read_shm(SHM_PATH_CHIP0)
            fpu_samples.append(avg_fpu)
            sfpu_samples.append(avg_sfpu)
            last_sample = now

    ttnn.synchronize_device(device)
    avg_fpu = sum(fpu_samples) / len(fpu_samples) if fpu_samples else 0.0
    avg_sfpu = sum(sfpu_samples) / len(sfpu_samples) if sfpu_samples else 0.0
    return avg_fpu, avg_sfpu, len(fpu_samples)


def test_ttnvtop_bfp4_matmul_tracks_analytical():
    """bfp4 matmul under LoFi should read like bfp8 matmul: ttnvtop within
    ±15% of ttnn analytical. The FPU counter ticks once per FPU request
    regardless of input dtype — the LoFi muladd count (4096 per request)
    is a property of the fidelity, not the data width. This test pins that
    invariant.
    """
    _require_collector()

    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=3855488)
    try:
        cg = device.compute_with_storage_grid_size()
        grid = (cg.x, cg.y)

        M, K, N = 32 * 16 * grid[1], 32 * 16 * grid[0], 32 * 16 * grid[0]
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=K // grid[0] // 32,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=M // grid[1] // 32,
            per_core_N=N // grid[0] // 32,
            transpose_mcast=False,
            fused_activation=None,
        )
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        ttnn_tf, tt_tf, n_samples, avg_sfpu = _run_matmul_and_measure(
            device, program_config, ckc, ttnn.bfloat4_b, M, K, N, seconds=30
        )
    finally:
        ttnn.close_device(device)

    print(f"\n[bfp4] LoFi bf4b {M}x{K}x{N} grid={grid}")
    print(f"[bfp4]   ttnn analytical: {ttnn_tf:.2f} TF")
    print(f"[bfp4]   ttnvtop measured: {tt_tf:.2f} TF  (mean of {n_samples} samples)")
    print(f"[bfp4]   ratio: {tt_tf / ttnn_tf:.3f}×  sfpu%={avg_sfpu*100:.2f}")

    assert n_samples >= 3, f"too few ttnvtop samples ({n_samples})"
    ratio = tt_tf / ttnn_tf
    assert 0.85 <= ratio <= 1.15, (
        f"bfp4 ttnvtop ({tt_tf:.2f} TF) vs ttnn ({ttnn_tf:.2f} TF) ratio {ratio:.3f}× "
        f"outside [0.85, 1.15]. Either the bf4b dispatch path changed fidelity, "
        f"or ttnvtop's LoFi calibration constant is wrong."
    )
    # SFPU should stay low on a pure matmul. Treat as a cross-check that the
    # alternating counter_sel isn't accidentally double-counting FPU into SFPU.
    assert avg_sfpu < 0.05, (
        f"bfp4 matmul reports SFPU%={avg_sfpu*100:.2f}, expected <5%. "
        f"Alternating counter_sel state machine may be misrouting reads."
    )


def test_ttnvtop_sfpu_exp_lights_sfpu_not_fpu():
    """Pure vector unary (ttnn.exp) goes entirely through the SFPU — no
    matmul tiles, no FPU requests. The S bar should light up; the F bar
    should stay near zero. This is the tightest test we have that the SFPU
    counter read actually reflects SFPU activity (and isn't just aliased
    FPU).
    """
    _require_collector()

    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=3855488)
    try:
        avg_fpu, avg_sfpu, n_samples = _run_sfpu_and_measure(
            device, lambda t: ttnn.exp(t), shape=(8192, 8192), seconds=25
        )
    finally:
        ttnn.close_device(device)

    print(f"\n[sfpu-exp] shape=(8192, 8192) bf16")
    print(f"[sfpu-exp]   F = {avg_fpu*100:.1f}%  (expected near 0)")
    print(f"[sfpu-exp]   S = {avg_sfpu*100:.1f}%  (expected substantial)")
    print(f"[sfpu-exp]   samples: {n_samples}")

    assert n_samples >= 3, f"too few ttnvtop samples ({n_samples})"
    # SFPU fraction should be meaningfully above noise. Threshold is
    # conservative — ttnn.exp on a full grid typically hits 20-60%+ SFPU
    # busy. Raise if it hits noise floor consistently.
    assert avg_sfpu > 0.10, (
        f"ttnn.exp reports SFPU%={avg_sfpu*100:.2f} (expected > 10%). "
        f"Either the SFPU counter is not being read at all, or the workload "
        f"isn't actually exercising SFPU. Check collector log for "
        f"`[live-probe] ... sel=1 ...` entries with non-zero last_sfpu_out_h."
    )
    # FPU should stay quiet — ttnn.exp does not issue FPU requests.
    # Allow some noise floor for data movement / helper ops.
    assert avg_fpu < 0.10, (
        f"ttnn.exp reports FPU%={avg_fpu*100:.2f} (expected < 10%). "
        f"Either the kernel is using matmul ops unexpectedly, or the alternating "
        f"counter_sel state machine is cross-wiring SFPU reads into the FPU field."
    )


def test_ttnvtop_sfpu_softmax_lights_both_pipes():
    """Softmax is an SFPU-heavy workload that also includes MATH reductions.
    Both F and S bars should be visibly active. Pins that mixed SFPU/FPU
    workloads surface both signals correctly — unlike the pre-2.1.d world
    where softmax looked like a low-utilization workload because only FPU
    was reported.
    """
    _require_collector()

    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=3855488)
    try:
        # Smaller last dim makes per-row softmax tractable; more rows keep
        # all cores busy.
        avg_fpu, avg_sfpu, n_samples = _run_sfpu_and_measure(
            device, lambda t: ttnn.softmax(t, dim=-1), shape=(4096, 2048), seconds=25
        )
    finally:
        ttnn.close_device(device)

    print(f"\n[sfpu-softmax] shape=(4096, 2048) bf16 softmax(dim=-1)")
    print(f"[sfpu-softmax]   F = {avg_fpu*100:.1f}%")
    print(f"[sfpu-softmax]   S = {avg_sfpu*100:.1f}%")
    print(f"[sfpu-softmax]   samples: {n_samples}")

    assert n_samples >= 3, f"too few ttnvtop samples ({n_samples})"
    # Weak lower bound on SFPU — softmax always exercises SFPU for exp + reciprocal.
    assert avg_sfpu > 0.05, (
        f"softmax reports SFPU%={avg_sfpu*100:.2f}, expected > 5%. " f"SFPU counter may not be sampling."
    )
    # The FPU bound is weak because softmax uses reduce ops which are MATH-pipe;
    # just asserting > 0 would flake. We assert it's at least visibly nonzero.
    # If the FPU side ever reports zero here, suspect the alternating state machine
    # is starving the FPU branch.
    assert avg_fpu > 0.01, (
        f"softmax reports FPU%={avg_fpu*100:.2f}, expected > 1%. "
        f"Either the reduce ops aren't going through MATH, or the FPU branch "
        f"of the alternating sampler is never firing."
    )


# ---------------------------------------------------------------------------
# Trace-capture variants.
#
# All tests above drive the device from a Python hot loop — each iter pays
# ~10-20μs of host-side ttnn dispatch overhead. On the device, that shows up
# as a small gap between kernels which ttnvtop's D bar dips into. Trace
# capture records one set of ops and replays it as a single device command,
# eliminating the per-iter host overhead.
#
# Why these are worth their own tests:
#   1. Steady-state throughput is typically higher under trace (no dispatch
#      stall between iters), so this is where ttnvtop should see its highest
#      F and S readings.
#   2. Confirms the collector's signal pipeline still works correctly when
#      the workload uses ttnn.execute_trace — dispatch attribution, perf
#      counter sampling, and the EWMA should all behave identically regardless
#      of whether the work came from a python dispatch or a device-side trace.
#   3. Guards against a regression where some future ttnvtop change queries
#      python-visible state (e.g. kernel id from launch_msg) and breaks under
#      trace mode where that state is set differently.
# ---------------------------------------------------------------------------


def _run_traced_and_measure(device, build_fn, iters_per_replay=32, seconds=25):
    """Capture `iters_per_replay` invocations of build_fn into one trace, then
    replay in a hot loop. Samples ttnvtop SHM every 3s after a 3s EWMA settle.
    Returns (avg_fpu_frac, avg_sfpu_frac, avg_dispatch_frac, n_samples, replays).

    build_fn(device) must issue one ttnn op (it'll be called iters_per_replay
    times during trace capture). Warmup (JIT pre-compile) happens before trace.
    """
    # Warmup: ensures kernels are JIT-compiled before the trace region is
    # allocated (trace capture doesn't handle new kernels gracefully).
    for _ in range(5):
        build_fn(device)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(iters_per_replay):
        build_fn(device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.synchronize_device(device)

    fpu_samples = []
    sfpu_samples = []
    dispatch_samples = []
    start = time.time()
    last_sample = start
    replays = 0

    try:
        while time.time() - start < seconds:
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            replays += 1
            now = time.time()
            if now - last_sample >= 3.0 and now - start >= 3.0:
                _, _, _, avg_fpu, avg_sfpu = _read_shm(SHM_PATH_CHIP0)
                # Also pull dispatch% — raw bytes re-parsed from SHM.
                with open(SHM_PATH_CHIP0, "rb") as f:
                    data = f.read()
                hdr = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
                n_cores = hdr[8]
                sum_disp = 0
                for i in range(n_cores):
                    r = struct.unpack(
                        PER_CORE_FMT, data[HEADER_SIZE + i * PER_CORE_SIZE : HEADER_SIZE + (i + 1) * PER_CORE_SIZE]
                    )
                    sum_disp += r[7]  # dispatch_busy_p1000
                avg_disp = sum_disp / n_cores / 1000.0
                fpu_samples.append(avg_fpu)
                sfpu_samples.append(avg_sfpu)
                dispatch_samples.append(avg_disp)
                last_sample = now
        ttnn.synchronize_device(device)
    finally:
        try:
            ttnn.release_trace(device, tid)
        except Exception:
            pass  # device close will release anyway

    avg_fpu = sum(fpu_samples) / len(fpu_samples) if fpu_samples else 0.0
    avg_sfpu = sum(sfpu_samples) / len(sfpu_samples) if sfpu_samples else 0.0
    avg_disp = sum(dispatch_samples) / len(dispatch_samples) if dispatch_samples else 0.0
    return avg_fpu, avg_sfpu, avg_disp, len(fpu_samples), replays


def test_ttnvtop_trace_lofi_matmul_saturates_dispatch():
    """Under trace capture, a LoFi bf8b matmul should:
      - pin dispatch% near 100% (no Python dispatch gap between kernels), and
      - hit at least as much FPU% as the python-dispatched variant does.

    If dispatch% is visibly lower than under Python dispatch, something
    about trace execution is fooling the go_msg.signal sampler. If FPU% is
    lower, the perf-counter path is somehow diverging from how it behaves
    under python dispatch.
    """
    _require_collector()

    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=3855488)
    try:
        cg = device.compute_with_storage_grid_size()
        grid = (cg.x, cg.y)
        M, K, N = 32 * 16 * grid[1], 32 * 16 * grid[0], 32 * 16 * grid[0]
        in0 = torch.randn((1, 1, M, K)).bfloat16()
        in1 = torch.randn((1, 1, K, N)).bfloat16()
        in0_t = ttnn.from_torch(in0, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        in1_t = ttnn.from_torch(in1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
        program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=K // grid[0] // 32,
            out_subblock_h=4,
            out_subblock_w=2,
            per_core_M=M // grid[1] // 32,
            per_core_N=N // grid[0] // 32,
            transpose_mcast=False,
            fused_activation=None,
        )
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def do_matmul(_dev):
            ttnn.matmul(in0_t, in1_t, program_config=program_config, compute_kernel_config=ckc)

        avg_fpu, avg_sfpu, avg_disp, n_samples, replays = _run_traced_and_measure(
            device, do_matmul, iters_per_replay=32, seconds=30
        )
    finally:
        ttnn.close_device(device)

    print(f"\n[trace-matmul] LoFi bf8b {M}×{K}×{N} grid={grid}, {replays} replays")
    print(f"[trace-matmul]   F = {avg_fpu*100:.1f}%  S = {avg_sfpu*100:.1f}%  D = {avg_disp*100:.1f}%")
    print(f"[trace-matmul]   samples: {n_samples}")

    assert n_samples >= 3, f"too few ttnvtop samples ({n_samples})"
    # Dispatch is high, but not 100% — each ttnn.matmul trace entry is a full
    # program launch, and between programs the device dispatcher has a small
    # gap before the next go_msg propagates to the cores. Under python
    # dispatch this gap exists too but is hidden by device-queue buffering.
    # Observed baseline on WH N300: ~80-85%. Pin > 70% to catch a real
    # regression without flaking on run-to-run noise.
    assert avg_disp > 0.70, (
        f"Traced matmul reports D%={avg_disp*100:.2f}, expected > 70%. "
        f"Either the go_msg sampler is missing traced kernels (possible if "
        f"trace uses a different go_message_index convention) or the trace "
        f"replay is taking a very different path than expected."
    )
    # FPU should track high — trace doesn't change the math. Note that on
    # WH this typically reads ~50% because the matmul spends significant
    # cycles on unpack/pack stalls even though "dispatched" reads high.
    # This is a property of the kernel, not of ttnvtop.
    assert avg_fpu > 0.30, (
        f"Traced matmul reports FPU%={avg_fpu*100:.2f}, expected > 30%. "
        f"If python-dispatched LoFi matmul hits ~50-60% FPU and traced drops "
        f"much below, suspect the collector is failing to read perf counters "
        f"during trace replay, or the alternating counter_sel state machine "
        f"is mis-routing reads into the SFPU branch."
    )
    # SFPU should still be quiet — pure matmul.
    assert avg_sfpu < 0.05, (
        f"Traced matmul reports SFPU%={avg_sfpu*100:.2f}, expected < 5%. "
        f"Something is routing FPU reads into the SFPU branch under trace."
    )
    # Extra sanity cross-check: F% should be at least roughly half of D%.
    # If D pins at 80% but F collapses to <10%, that's a real regression in
    # the perf counter sampling path under trace.
    assert avg_fpu > 0.5 * avg_disp - 0.1, (
        f"F%={avg_fpu*100:.2f} is much lower than expected given D%={avg_disp*100:.2f}. "
        f"This usually means the FPU counter is being under-sampled under trace."
    )


def test_ttnvtop_trace_sfpu_exp_lights_sfpu_not_fpu():
    """Same ttnn.exp workload as the non-trace SFPU test, but driven via trace
    replay. Proves the alternating counter_sel sampler continues to attribute
    SFPU reads correctly when the workload dispatches at full device speed.
    """
    _require_collector()

    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=3855488)
    try:
        H, W = 8192, 8192
        x = torch.randn((1, 1, H, W)).bfloat16()
        x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        def do_exp(_dev):
            ttnn.exp(x_t)

        avg_fpu, avg_sfpu, avg_disp, n_samples, replays = _run_traced_and_measure(
            device, do_exp, iters_per_replay=16, seconds=25
        )
    finally:
        ttnn.close_device(device)

    print(f"\n[trace-sfpu-exp] shape=({H},{W}) bf16, {replays} replays")
    print(f"[trace-sfpu-exp]   F = {avg_fpu*100:.1f}%  S = {avg_sfpu*100:.1f}%  D = {avg_disp*100:.1f}%")
    print(f"[trace-sfpu-exp]   samples: {n_samples}")

    assert n_samples >= 3, f"too few ttnvtop samples ({n_samples})"
    assert avg_disp > 0.80, (
        f"Traced exp reports D%={avg_disp*100:.2f}, expected > 80%. " f"Trace dispatch gap is wider than expected."
    )
    # Steady-state SFPU should be meaningfully above the non-trace floor.
    assert avg_sfpu > 0.10, (
        f"Traced exp reports SFPU%={avg_sfpu*100:.2f}, expected > 10%. "
        f"Mirror test (non-trace) also reports SFPU%>10%; if this one fails "
        f"but the non-trace variant passes, the alternating counter_sel sampler "
        f"has a timing-dependent bug that surfaces under back-to-back trace replay."
    )
    assert avg_fpu < 0.10, (
        f"Traced exp reports FPU%={avg_fpu*100:.2f}, expected < 10%. "
        f"Exp uses SFPU only; FPU should stay near zero even under trace."
    )


# ---------------------------------------------------------------------------
# Phase 2.1.c: TIME% column population.
#
# After the LLK between-tile sampler hooks land (queued in a tt_llk submodule
# PR), the collector drains per-program core-cycle counts each second and the
# program registrar publishes them as the new `cycles_in_window` field. The
# viewer then derives a TIME% column from those values.
#
# This test asserts that AT LEAST ONE entry in the registry has a non-zero
# cycles_in_window after running a busy LoFi matmul for 25s. It's skipped by
# default until 2.1.c is deployed; flipping the @pytest.mark.skip decorator
# (or removing it) is the single line to enable when the LLK hooks land.
# ---------------------------------------------------------------------------

REGISTRY_PATH = "/dev/shm/tt_program_registry"
# Mirrors common/program_registry.hpp once Agent A's schema bump lands:
#   header: magic[4] + version + entry_size + capacity + writer_pid + epoch_us
#         + atomic<u32> write_cursor + reserved[4]   = 48 bytes (unchanged)
#   entry (v2): runtime_id (u32) + pid (u32) + epoch_us (u64) + name[96]
#             + cycles_in_window (u64)               = 120 bytes
REG_HEADER_SIZE = 48
REG_ENTRY_FMT_V2 = "<II Q 96s Q"
REG_ENTRY_SIZE_V2 = struct.calcsize(REG_ENTRY_FMT_V2)


def _read_registry_cycles_in_window():
    """Walk /dev/shm/tt_program_registry and return list of (runtime_id, name, cycles_in_window).

    Returns [] if the file is absent or schema doesn't match v2 (raises a
    pytest.skip in that case so the test fails informatively).
    """
    if not os.path.exists(REGISTRY_PATH):
        pytest.skip(f"program registry not present at {REGISTRY_PATH} — workload not registered")
    with open(REGISTRY_PATH, "rb") as f:
        data = f.read()
    if len(data) < REG_HEADER_SIZE:
        pytest.skip("registry file truncated; cannot read header")
    # version is the second u16 in the header struct (after the 4-byte magic).
    version = struct.unpack_from("<H", data, 4)[0]
    entry_size = struct.unpack_from("<H", data, 6)[0]
    capacity = struct.unpack_from("<I", data, 8)[0]
    write_cursor = struct.unpack_from("<I", data, 32)[0]
    if version < 2 or entry_size != REG_ENTRY_SIZE_V2:
        pytest.skip(
            f"registry schema mismatch (version={version}, entry_size={entry_size}); "
            f"Phase 2.1.c (v2 with cycles_in_window) not yet deployed."
        )
    n = min(write_cursor, capacity)
    out = []
    for i in range(n):
        off = REG_HEADER_SIZE + i * REG_ENTRY_SIZE_V2
        rid, _pid, _epoch, name_bytes, cyc = struct.unpack_from(REG_ENTRY_FMT_V2, data, off)
        name = name_bytes.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        out.append((rid, name, cyc))
    return out


@pytest.mark.skip(reason="enable after Phase 2.1.c LLK hooks land")
def test_ttnvtop_time_pct_present():
    """Sanity-check: after running a busy LoFi matmul, at least one program in
    the registry has cycles_in_window > 0. This is the data the viewer's TIME%
    column consumes.

    Skipped by default (Phase 2.1.c not yet deployed). Remove or flip the
    @pytest.mark.skip decorator once the LLK between-tile sampler hooks are
    landed and the collector is publishing cycles_in_window.
    """
    _require_collector()

    # Run for 25s with LoFi to keep the FPU busy and let multiple drain windows
    # populate cycles_in_window for the matmul program.
    _run_matmul_with_fidelity(ttnn.MathFidelity.LoFi, seconds=25)

    entries = _read_registry_cycles_in_window()
    assert len(entries) > 0, (
        "registry has no entries after running matmul — workload didn't set "
        "TTNVTOP_REGISTER_PROGRAMS=1, or registrar library failed to publish."
    )
    nonzero = [(rid, name, cyc) for (rid, name, cyc) in entries if cyc > 0]
    print(f"\n[time-pct-present] {len(entries)} registered programs, {len(nonzero)} with cycles_in_window > 0")
    for rid, name, cyc in nonzero[:5]:
        print(f"[time-pct-present]   rid={rid}  name={name!r}  cycles_in_window={cyc}")
    assert len(nonzero) > 0, (
        f"no program in the registry has cycles_in_window > 0 after 25s LoFi matmul. "
        f"Either the LLK between-tile sampler isn't writing the per-program counter, "
        f"or the collector isn't draining it into the registry's cycles_in_window field. "
        f"Total entries: {len(entries)}."
    )
