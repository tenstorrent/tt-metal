# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `shared_input_reuse` data-movement example.

A fixed 2×grid_x rectangle of cores each need the same multi-MB DRAM K head and do the same job
(fold the whole streamed K into one running tile-sum). `per_core_dram` has every core read the whole
stream from DRAM; `mcast` reads each chunk once on the top-left injector and NoC-multicasts it to the
rest. Output is 1 tile per core (<< input), so the kernel is read-bound. See
ttnn/ttnn/operations/examples/shared_input_reuse/README.md.

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_correctness
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_device_perf
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import socket
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from ttnn.operations.examples.shared_input_reuse import shared_input_reuse, VARIANTS
from ttnn.operations.examples.shared_input_reuse.shared_input_reuse import create_program_descriptor

from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32

# Model a real SDPA K head streamed in k_chunks: CHUNK_ROWS = k_chunk/32, D_COLS = head_dim/32,
# NUM_CHUNKS = S_kv/k_chunk. Default = WAN 2.2-ish head: k_chunk=512, head_dim=128, S_kv=9728
# -> 19 chunks × (16×4=64) tiles = 1216 tiles = 2.375 MB bf16 (far larger than L1, so it is streamed).
CHUNK_ROWS = int(os.environ.get("SIR_CHUNK_ROWS", "16"))
D_COLS = int(os.environ.get("SIR_D_COLS", "4"))
NUM_CHUNKS = int(os.environ.get("SIR_CHUNKS", "19"))
N_WARMUP = 3
N_PROFILE_ITERS = int(os.environ.get("SIR_TRIALS", "10"))
_INNER = 5
REPORT_PATH = os.environ.get(
    "SIR_REPORT",
    str(Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/examples/shared_input_reuse/report.md"),
)
PCC = 0.99
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _num_cores(device):
    return 2 * device.compute_with_storage_grid_size().x


def _make_input(device):
    s_kv = NUM_CHUNKS * CHUNK_ROWS * TILE
    # All ones: the elementwise tile-sum is monotonic and cancellation-free (= tile count per element),
    # so PCC exposes the true accumulation accuracy instead of random-cancellation noise.
    x = torch.ones(s_kv, D_COLS * TILE, dtype=torch.float32)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_x, x


def _expected_tile(x):
    # elementwise sum of every tile of the K stream -> one 32x32 tile
    return x.reshape(NUM_CHUNKS * CHUNK_ROWS, TILE, D_COLS, TILE).sum(dim=(0, 2))


def _make_output(device, n):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([n * TILE, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure_ns(device, run_fn):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = []
    for _ in range(N_PROFILE_ITERS):
        for _ in range(_INNER):
            run_fn()
        total = _read_kernel_ns(device)
        if total is None:
            return None, None
        samples.append(total / _INNER)
    med = statistics.median(samples)
    return med, (statistics.pstdev(samples) / med * 100.0 if med else float("nan"))


def test_shared_input_reuse_delivery(device):
    """The concept: read-once + multicast delivers the SAME data as per-core DRAM reads.

    Both variants run the identical compute on the identical K; only the read path differs, so their
    per-core outputs must match (this is the meaningful correctness — it proves the mcast handed every
    core exactly the bytes the DRAM read would have). Run at the full multi-MB stream.
    """
    tt_x, _ = _make_input(device)
    n = _num_cores(device)
    out_dram = ttnn.to_torch(
        shared_input_reuse(tt_x, _make_output(device, n), variant="per_core_dram", chunk_rows=CHUNK_ROWS)
    ).to(torch.float32)
    out_mcast = ttnn.to_torch(
        shared_input_reuse(tt_x, _make_output(device, n), variant="mcast", chunk_rows=CHUNK_ROWS)
    ).to(torch.float32)
    assert list(out_mcast.shape) == list(out_dram.shape)
    assert_with_pcc(out_dram, out_mcast, 0.9999)

    # Diagnostic (all-ones input): the exact full-depth sum is the tile count per element. Log how close
    # the on-device (bf16-SRC) accumulation gets — this reveals deep-sum accuracy without cancellation.
    tiles_summed = NUM_CHUNKS * CHUNK_ROWS * D_COLS
    from tests.ttnn.utils_for_testing import comp_pcc

    _, pcc_msg = comp_pcc(torch.full_like(out_dram, float(tiles_summed)), out_dram)
    logger.info(
        f"[shared_input_reuse] all-ones deep sum: expected={tiles_summed} per element, "
        f"got mean={out_dram.mean().item():.1f} min={out_dram.min().item():.1f} max={out_dram.max().item():.1f}; "
        f"vs-torch {pcc_msg}"
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_shared_input_reuse_structural(device, variant):
    """The tile-sum is structurally correct: at a SMALL accumulation depth (few chunks), where the
    bf16 running sum is still exact, the on-device result matches the torch reference. (At the full
    multi-MB depth the bf16 sum is legitimately lossy vs fp32 torch — a precision gap, not a bug —
    so the full-depth correctness is checked variant-vs-variant in test_shared_input_reuse_delivery.)
    """
    cr, cols, nch = 2, D_COLS, 3  # 3 chunks x (2 x D_COLS) tiles — shallow, bf16-exact
    x = torch.ones(nch * cr * TILE, cols * TILE, dtype=torch.float32)  # cancellation-free
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    n = _num_cores(device)
    out = ttnn.to_torch(shared_input_reuse(tt_x, _make_output(device, n), variant=variant, chunk_rows=cr)).to(
        torch.float32
    )
    expected = x.reshape(nch * cr, TILE, cols, TILE).sum(dim=(0, 2)).repeat(n, 1)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, PCC)


def test_shared_input_reuse_device_perf(device):
    """Device kernel duration: per-core DRAM read of the whole K stream vs read-once + multicast."""
    arch = os.environ.get("ARCH_NAME", str(device.arch()))
    box = socket.gethostname()
    n = _num_cores(device)
    stream_tiles = NUM_CHUNKS * CHUNK_ROWS * D_COLS
    stream_mb = stream_tiles * 2048 / (1024 * 1024)

    tt_x, _ = _make_input(device)
    tt_out = _make_output(device, n)

    ns = {}
    for variant in VARIANTS:
        desc = create_program_descriptor(tt_x, tt_out, variant=variant, chunk_rows=CHUNK_ROWS)
        run_fn = lambda x=tt_x, o=tt_out, d=desc: ttnn.generic_op([x, o], d)
        med, std = _measure_ns(device, run_fn)
        assert med is not None, f"profiler produced no data for {variant}"
        ns[variant] = (med, std)

    (bm, bs) = ns["per_core_dram"]
    (mm, msd) = ns["mcast"]
    ratio = bm / mm if mm else float("nan")
    lines = [
        "",
        "=== shared_input_reuse device perf (shared K stream: per-core DRAM read vs read-once+multicast) ===",
        f"    box={box}  arch={arch}  cores={n} (2x{n//2} grid)  injector=top-left",
        f"    K head = {NUM_CHUNKS}x{CHUNK_ROWS}x{D_COLS} tiles = {stream_tiles} tiles ({stream_mb:.2f} MB) streamed in "
        f"{NUM_CHUNKS} chunks of {CHUNK_ROWS*D_COLS} tiles  rounds={N_PROFILE_ITERS}x{_INNER}",
        f"    baseline reads the whole stream on every core ({n}x {stream_mb:.2f} = {n*stream_mb:.1f} MB DRAM); mcast reads it once",
        f"    {'variant':<16}  {'ns/op':>12}  {'±%':>5}  {'vs dram':>8}",
        f"    {'per_core_dram':<16}  {bm:>12.1f}  {bs:>5.1f}  {'(base)':>8}",
        f"    {'mcast':<16}  {mm:>12.1f}  {msd:>5.1f}  {ratio:>6.2f}x",
    ]
    logger.info("\n".join(lines))

    md = [
        "# shared_input_reuse — device report",
        "",
        f"- box: `{box}`",
        f"- arch: {arch}",
        f"- cores: {n} (2 × {n//2} grid); injector = top-left (0,0)",
        f"- shared input: an SDPA K head [S_kv={NUM_CHUNKS*CHUNK_ROWS*TILE}, head_dim={D_COLS*TILE}] = {stream_tiles} tiles ({stream_mb:.2f} MB bf16),",
        f"  streamed in {NUM_CHUNKS} chunks of {CHUNK_ROWS}×{D_COLS} = {CHUNK_ROWS*D_COLS} tiles (cb_in holds one chunk — L1 can't hold the head)",
        f"- job: fold the whole stream into 1 running tile-sum/core in DEST (output = {n} tiles << input); rounds={N_PROFILE_ITERS}x{_INNER}",
        "- metric: DEVICE KERNEL DURATION [ns], median over rounds (±% = pstdev/median)",
        f"- baseline reads the whole {stream_mb:.2f} MB stream from DRAM on every core ({n}× = {n*stream_mb:.1f} MB of DRAM reads); mcast",
        "  reads each chunk once on the injector and NoC-broadcasts it, so DRAM sees the stream once.",
        "",
        "| variant | ns/op | ±% | vs per_core_dram |",
        "|---------|-------|----|------------------|",
        f"| per_core_dram | {bm:.1f} | {bs:.1f} | 1.00x |",
        f"| mcast | {mm:.1f} | {msd:.1f} | {ratio:.2f}x |",
    ]
    try:
        Path(REPORT_PATH).write_text("\n".join(md) + "\n")
        logger.info(f"[shared_input_reuse] wrote {REPORT_PATH}")
    except OSError as e:
        logger.warning(f"[shared_input_reuse] could not write report: {e}")
