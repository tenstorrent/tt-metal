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
from ttnn.operations.examples.shared_input_reuse import WINDOW_VARIANTS
from ttnn.operations.examples.shared_input_reuse.resident_window import (
    create_window_program_descriptor,
    window_rectangle,
)

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

    bm, bs = ns["per_core_dram"]
    mm, msd = ns["mcast"]
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


def _measure_interleaved_ns(device, run_fns, trials=N_PROFILE_ITERS):
    """Median/min/max ns per op for each callable, with the windows interleaved A,B,A,B,... so
    slow drift on the shared device hits every variant equally."""
    for fn in run_fns.values():
        for _ in range(N_WARMUP):
            fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = {k: [] for k in run_fns}
    for _ in range(trials):
        for k, fn in run_fns.items():
            for _ in range(_INNER):
                fn()
            total = _read_kernel_ns(device)
            assert total is not None, f"profiler produced no data for {k}"
            samples[k].append(total / _INNER)
    return {k: (statistics.median(v), min(v), max(v)) for k, v in samples.items()}


# Streamed shared input: ("stream", chunk_rows, d_cols, num_chunks). All shapes are bf16 tiles; the worker
# rectangle is 2 x grid_x Tensix cores (16 on Wormhole B0 n150, 24 on Blackhole p150b), injector at (0,0).
#   helps:          64-tile chunks (131072 B per multicast), 1216 tiles total
#   measure:        16-tile chunks, same 1216 tiles: loses on WH B0, wins on BH, so only logged
#   measure:        8-tile chunks (16384 B per multicast): WH B0 0.82x, BH p150b 24 Tensix cores 0.96x, but
#                   1.51x on a BH p150b whose 2 x 13 = 26-Tensix-core baseline is slower, so only logged
#   does_not_help:  1-tile chunks (2048 B per multicast, d_cols = 1), SAME 1216 tiles total
# Resident window: ("window", num_cores, window_tiles, tile_rows_per_core, chunk_tiles, lever). num_cores None =
# the whole compute grid (130 Tensix cores on BH p150b). The baseline is always `per_core`; r = window bytes /
# x bytes per Tensix core = 1 / tile_rows_per_core.
#   helps:          full grid, 128-tile (256 KB) window, 2 tile-rows of x per Tensix core (r = 0.5)
#   does_not_help:  full grid, 8-tile (16 KB) window, 8 tile-rows per Tensix core (r = 0.125)
#   does_not_help:  32 Tensix cores, 128-tile window, 8 tile-rows per Tensix core (r = 0.125)
#   does_not_help:  lockstep_mcast with 1-tile chunks (a handshake per tile), where oneshot_mcast helps
# Thresholds hold on every measured board (stream: WH B0 n150, BH p150b; window: BH p150b only); `measure`
# regimes flip between boards and only log -- numbers per board in README.md.
_REGIME_CASES = [
    pytest.param(("stream", 16, 4, 19), "helps", 1.17, id="helps_chunk64tiles_1216tiles"),  # WH 1.34x, BH 1.93x
    pytest.param(("stream", 4, 4, 76), "measure", None, id="measure_chunk16tiles_1216tiles"),  # WH 0.93x, BH 1.32x
    pytest.param(
        ("stream", 2, 4, 152), "measure", None, id="measure_chunk8tiles_1216tiles"
    ),  # WH 0.82x, BH 0.96x / 1.51x
    pytest.param(
        ("stream", 1, 1, 1216), "does_not_help", 1.0, id="does_not_help_chunk1tile_1216tiles"
    ),  # WH 0.70x, BH 0.64x
    pytest.param(("window", None, 128, 2, 32, "oneshot_mcast"), "helps", 1.10, id="window_helps_fullgrid_w128_r050"),
    pytest.param(
        ("window", None, 8, 8, 32, "oneshot_mcast"), "does_not_help", 1.05, id="window_does_not_help_fullgrid_w8_r0125"
    ),
    pytest.param(
        ("window", 32, 128, 8, 32, "oneshot_mcast"), "does_not_help", 1.05, id="window_does_not_help_32cores_w128_r0125"
    ),
    pytest.param(
        ("window", None, 32, 2, 1, "lockstep_mcast"), "does_not_help", 1.05, id="window_does_not_help_lockstep_chunk1"
    ),
]


def _stream_regime(device, chunk_rows, d_cols, num_chunks):
    n = _num_cores(device)
    s_kv = num_chunks * chunk_rows * TILE
    x = torch.ones(s_kv, d_cols * TILE, dtype=torch.float32)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = _make_output(device, n)

    # Correctness: both variants must deliver identical per-core sums.
    outs = {}
    fns = {}
    for variant in VARIANTS:
        desc = create_program_descriptor(tt_x, tt_out, variant=variant, chunk_rows=chunk_rows)
        fns[variant] = lambda x=tt_x, o=tt_out, d=desc: ttnn.generic_op([x, o], d)
        outs[variant] = ttnn.to_torch(fns[variant]() or tt_out).to(torch.float32)
    assert_with_pcc(outs["per_core_dram"], outs["mcast"], 0.9999)

    res = _measure_interleaved_ns(device, fns)
    chunk_tiles = chunk_rows * d_cols
    desc = (
        f"{n} Tensix cores, {num_chunks * chunk_tiles} tiles total in {num_chunks} chunks of {chunk_tiles} tiles "
        f"({chunk_tiles * 2048} B/chunk)"
    )
    return res, "per_core_dram", "mcast", desc


def _window_regime(device, num_cores, window_tiles, tile_rows_per_core, chunk_tiles, lever):
    cols, rows_grid = window_rectangle(device, num_cores)
    n = cols * rows_grid
    rows = n * tile_rows_per_core * TILE
    tt_x, tt_w, tt_out, expected = _window_tensors(device, rows, window_tiles)
    fns = _window_fns(device, tt_x, tt_w, tt_out, ["per_core", lever], num_cores, chunk_tiles)
    for fn in fns.values():
        _check_window(fn, tt_out, expected)
    res = _measure_interleaved_ns(device, fns)
    xbytes = rows * window_tiles * TILE * 2
    desc = (
        f"{n} Tensix cores ({cols}x{rows_grid}), window {window_tiles} tiles ({window_tiles * 2048} B) in "
        f"{chunk_tiles}-tile chunks, x [{rows}, {window_tiles * TILE}] = {xbytes / 2**20:.1f} MiB "
        f"({tile_rows_per_core} tile-rows per Tensix core, r = {1 / tile_rows_per_core:.3f})"
    )
    return res, "per_core", lever, desc


@pytest.mark.parametrize("case,regime,threshold", _REGIME_CASES)
def test_shared_input_reuse_regimes(device, case, regime, threshold):
    """When reading a shared input once and multicasting it pays, and when it does not.

    Streamed input (same total stream, different chunk size): the per-chunk mcast handshake (wait for all
    receivers' acks, then broadcast) has a fixed cost per chunk. With few large chunks it is amortized and the
    DRAM-read saving wins; with many small chunks the handshake dominates and `mcast` loses to per-core reads.

    Resident window (x * w, every Tensix core streams its own rows and needs all of w): a one-shot multicast of
    w removes (N - 1) copies of the window from DRAM. It pays when the window is a large share of each core's
    own x bytes (r >= 0.5) and not when it is small (r <= 0.125), whatever the Tensix-core count; a per-chunk
    handshake (lockstep, 1-tile chunks) costs more than the reads it saves.
    Asserts: helps -> speedup >= threshold; does_not_help -> speedup <= threshold; measure -> logged only.
    """
    if case[0] == "stream":
        res, base_name, lever, desc = _stream_regime(device, *case[1:])
    else:
        res, base_name, lever, desc = _window_regime(device, *case[1:])
    base, mc = res[base_name][0], res[lever][0]
    speedup = base / mc
    arch = {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH"}.get(str(device.arch()).rsplit(".", 1)[-1], str(device.arch()))
    logger.info(
        f"[{arch}] [shared_input_reuse regimes] {regime}: {desc} | "
        f"{base_name} {base:.0f} ns [{res[base_name][1]:.0f}-{res[base_name][2]:.0f}]  "
        f"{lever} {mc:.0f} ns [{res[lever][1]:.0f}-{res[lever][2]:.0f}]  speedup {speedup:.2f}x  "
        f"trials={N_PROFILE_ITERS}x{_INNER}"
    )
    if regime == "helps":
        assert speedup >= threshold, f"expected {lever} speedup >= {threshold:.2f}x, got {speedup:.2f}x"
    elif regime == "does_not_help":
        assert speedup <= threshold, f"expected {lever} speedup <= {threshold:.2f}x (no gain), got {speedup:.2f}x"


# ---------------------------------------------------------------------------------------------------------
# Resident window: every Tensix core streams its OWN rows of x (DRAM-bound read + write) and also needs the
# same small vector w resident in L1: out = x * w. per_core reads w from DRAM on every Tensix core;
# oneshot_mcast reads it once and multicasts it (no handshake); lockstep_mcast adds a per-chunk handshake and
# fetches the whole window before streaming. See README.md, "Second situation".
# ---------------------------------------------------------------------------------------------------------


def _window_tensors(device, rows, window_tiles, seed=0):
    torch.manual_seed(seed)
    wdim = window_tiles * TILE
    x = torch.randn(rows, wdim, dtype=torch.float32).to(torch.bfloat16)
    w = torch.randn(1, wdim, dtype=torch.float32).to(torch.bfloat16)
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_x, tt_w = ttnn.from_torch(x, **kw), ttnn.from_torch(w, **kw)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, wdim]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_x, tt_w, tt_out, (x.float() * w.float())


def _window_rows(stream_bytes, window_tiles, num_cores):
    """Rows of x (multiple of 32, at least one tile-row per Tensix core) for a stream of ~stream_bytes of x."""
    tile_rows = max(num_cores, round(stream_bytes / (window_tiles * 2048)))
    return tile_rows * TILE


def _window_fns(device, tt_x, tt_w, tt_out, variants, num_cores, chunk_tiles):
    fns = {}
    for variant in variants:
        desc = create_window_program_descriptor(
            tt_x, tt_w, tt_out, variant=variant, num_cores=num_cores, chunk_tiles=chunk_tiles
        )
        fns[variant] = lambda x=tt_x, w=tt_w, o=tt_out, d=desc: ttnn.generic_op([x, w, o], d)
    return fns


def _check_window(fn, tt_out, expected):
    fn()
    got = ttnn.to_torch(tt_out).float()
    assert_with_pcc(expected, got, 0.999)
    assert torch.allclose(got, expected, rtol=2e-2, atol=2e-2), f"max abs err {(got - expected).abs().max()}"


@pytest.mark.parametrize("variant", WINDOW_VARIANTS)
@pytest.mark.parametrize(
    "num_cores,window_tiles,tile_rows_per_core,chunk_tiles",
    [(None, 40, 2, 32), (16, 1, 3, 32), (32, 256, 1, 32), (64, 33, 5, 8)],
    ids=["fullgrid_w40_c32", "c16_w1", "c32_w256", "c64_w33_c8"],
)
def test_shared_input_reuse_window_correctness(
    device, variant, num_cores, window_tiles, tile_rows_per_core, chunk_tiles
):
    """Every variant computes x * w exactly as torch does, on every Tensix core, including partial chunks."""
    cols, rows_grid = window_rectangle(device, num_cores)
    n = cols * rows_grid
    rows = (tile_rows_per_core * n + 1) * TILE  # +1 tile-row: an uneven split
    tt_x, tt_w, tt_out, expected = _window_tensors(device, rows, window_tiles)
    fns = _window_fns(device, tt_x, tt_w, tt_out, [variant], num_cores, chunk_tiles)
    _check_window(fns[variant], tt_out, expected)
    _check_window(fns[variant], tt_out, expected)  # second launch: semaphores and the window start over


def _env_list(name, default):
    return [None if v in ("full", "all") else int(v) for v in os.environ.get(name, default).split(",")]


def _measure_window_case(device, num_cores, window_tiles, stream_bytes, variants, chunk, trials, repeats):
    """Median ns per variant over `repeats` DRAM placements (a pad tensor shifts x/w/out by a different number
    of pages each time; the A/B windows are interleaved within each placement). Returns
    ({variant: median ns}, {variant: (min, max) speedup vs variants[0] over placements}, rows)."""
    cols, rows_grid = window_rectangle(device, num_cores)
    rows = _window_rows(stream_bytes, window_tiles, cols * rows_grid)
    per_rep = []
    for rep in range(repeats):
        pad = ttnn.allocate_tensor_on_device(
            ttnn.Shape([TILE * (1 + 37 * rep), TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        tt_x, tt_w, tt_out, expected = _window_tensors(device, rows, window_tiles, seed=rep)
        fns = _window_fns(device, tt_x, tt_w, tt_out, variants, num_cores, chunk)
        for fn in fns.values():
            _check_window(fn, tt_out, expected)  # correctness gate for every variant at every point
        per_rep.append({k: v[0] for k, v in _measure_interleaved_ns(device, fns, trials=trials).items()})
        for t in (tt_x, tt_w, tt_out, pad):
            ttnn.deallocate(t)
    med = {v: statistics.median(r[v] for r in per_rep) for v in variants}
    spread = {
        v: (min(r[variants[0]] / r[v] for r in per_rep), max(r[variants[0]] / r[v] for r in per_rep))
        for v in variants[1:]
    }
    return med, spread, rows


def test_shared_input_reuse_window_sweep(device):
    """Log-only sweep: Tensix cores x window tiles x stream bytes (env SIR_WIN_*; the CLI's --window mode)."""
    cores_list = _env_list("SIR_WIN_CORES", "16,32,64,full")
    windows = _env_list("SIR_WIN_TILES", "1,8,32,128,256")
    streams = [int(float(v) * 2**20) for v in os.environ.get("SIR_WIN_STREAM_MB", "16,64").split(",")]
    variants = os.environ.get("SIR_WIN_VARIANTS", ",".join(WINDOW_VARIANTS)).split(",")
    chunk = int(os.environ.get("SIR_WIN_CHUNK", "32"))
    trials = int(os.environ.get("SIR_TRIALS", "3"))
    repeats = int(os.environ.get("SIR_WIN_REPEATS", "3"))
    arch = str(device.arch()).rsplit(".", 1)[-1]
    lines = [
        "",
        f"=== shared_input_reuse resident-window sweep  box={socket.gethostname()}  arch={arch}  chunk={chunk} tiles  "
        f"median of {repeats} DRAM placements x {trials}x{_INNER} launches ===",
        f"    {'cores':>5} {'window':>16} {'x MB':>6} {'rows':>8}  "
        + "  ".join(f"{v:>15}" for v in variants)
        + "   speedup vs "
        + variants[0]
        + " [min-max over placements]   "
        + variants[0]
        + " GB/s (x rd+wr)",
    ]
    for num_cores in cores_list:
        cols, rows_grid = window_rectangle(device, num_cores)
        n = cols * rows_grid
        for wtiles in windows:
            for stream in streams:
                med, spread, rows = _measure_window_case(
                    device, num_cores, wtiles, stream, variants, chunk, trials, repeats
                )
                base = med[variants[0]]
                xbytes = rows * wtiles * TILE * 2
                line = (
                    f"    {n:>5} {wtiles:>5} t {wtiles * 2048:>8} B {xbytes / 2**20:>6.1f} {rows:>8}  "
                    + "  ".join(f"{med[v]:>12.0f} ns" for v in variants)
                    + "   "
                    + " ".join(f"{base / med[v]:.2f}x [{spread[v][0]:.2f}-{spread[v][1]:.2f}]" for v in variants[1:])
                    + f"   {2 * xbytes / base:.0f}"
                )
                lines.append(line)
                logger.info(line)
    logger.info("\n".join(lines))
