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

test_shared_input_reuse_device_perf only logs its table unless SIR_REPORT names a Markdown file to write.
The resident-window sweep is an opt-in measurement sweep:

    SIR_SWEEPS=1 scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_shared_input_reuse.py::test_shared_input_reuse_window_sweep
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
from tests.ttnn.unit_tests.operations.examples.perf_test_utils import (
    check_regime,
    interleaved_ns,
    read_kernel_ns,
    require_cores,
    sweep_gate,
)
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
# test_shared_input_reuse_device_perf writes its Markdown report only when SIR_REPORT names a path (the CLI's
# --report); by default it only logs, so a plain run never rewrites the committed report.md.
REPORT_PATH = os.environ.get("SIR_REPORT", "")
PCC = 0.99


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


def _measure_ns(device, run_fn):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    read_kernel_ns(device)
    samples = []
    for _ in range(N_PROFILE_ITERS):
        for _ in range(_INNER):
            run_fn()
        total = read_kernel_ns(device)
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
    if not REPORT_PATH:
        return
    try:
        Path(REPORT_PATH).write_text("\n".join(md) + "\n")
        logger.info(f"[shared_input_reuse] wrote {REPORT_PATH}")
    except OSError as e:
        logger.warning(f"[shared_input_reuse] could not write report: {e}")


# Regimes: interleaved A,B,A,B... trials per variant (each trial = _INNER launches), verdict from
# `perf_test_utils.check_regime`.
_REGIME_TRIALS = max(7, N_PROFILE_ITERS)

# Streamed shared input: ("stream", chunk_rows, d_cols, num_chunks). All shapes are bf16 tiles; the worker
# rectangle is 2 x grid_x Tensix cores (16 on Wormhole B0 n150, 24 or 26 on Blackhole p150b), injector at (0,0),
# so these regimes build on any grid.
#   helps:          64-tile chunks (131072 B per multicast), 1216 tiles total
#   measure:        16-tile chunks, same 1216 tiles: loses on WH B0, wins on BH, so only logged
#   measure:        8-tile chunks (16384 B per multicast): WH B0 0.82x, BH p150b 24 Tensix cores 0.96x, but
#                   1.51x on a BH p150b whose 2 x 13 = 26-Tensix-core baseline is slower, so only logged
#   does_not_help:  1-tile chunks (2048 B per multicast, d_cols = 1), SAME 1216 tiles total
# Resident window: ("window", num_cores, window_tiles, tile_rows_per_core, chunk_tiles, lever). num_cores None =
# the whole compute grid (130 Tensix cores on BH p150b); a fixed count skips on a grid with fewer Tensix cores.
# The baseline is always `per_core`; r = window bytes / x bytes per Tensix core = 1 / tile_rows_per_core.
#   helps:          full grid, 128-tile (256 KB) window, 2 tile-rows of x per Tensix core (r = 0.5)
#   does_not_help:  full grid, 8-tile (16 KB) window, 8 tile-rows per Tensix core (r = 0.125)
#   does_not_help:  32 Tensix cores, 128-tile window, 8 tile-rows per Tensix core (r = 0.125)
#   does_not_help:  lockstep_mcast with 1-tile chunks (a handshake per tile), where oneshot_mcast helps
# The last field is the speedup (baseline ns / lever ns) measured per architecture, from the comment beside
# the regime, README.md and report.md; a (low, high) pair spans the boards / runs measured (BH: P150, P100a and
# two p150b grids for the streamed regimes; the regime run and the sweep row for the window regimes). The bound
# each regime is asserted against is derived from the number for the architecture it runs on
# (`perf_test_utils.regime_bound`): "helps" asserts half the measured gain but never closer than 10% to the
# measurement; "does_not_help" asserts <= max(1.0, 1.10 x the highest measured value). An architecture with no
# number (Wormhole for every window regime) is logged, not asserted; `measure` regimes always only log. A missed
# bound is re-measured once and re-checked on the pooled samples.
_REGIME_CASES = [
    pytest.param(
        ("stream", 16, 4, 19), "helps", {"wh": (1.32, 1.34), "bh": (1.71, 2.27)}, id="helps_chunk64tiles_1216tiles"
    ),  # WH 1.34x, BH 1.93x
    pytest.param(
        ("stream", 4, 4, 76), "measure", {"wh": 0.93, "bh": (1.32, 2.03)}, id="measure_chunk16tiles_1216tiles"
    ),  # WH 0.93x, BH 1.32x
    pytest.param(
        ("stream", 2, 4, 152), "measure", {"wh": 0.82, "bh": (0.96, 1.51)}, id="measure_chunk8tiles_1216tiles"
    ),  # WH 0.82x, BH 0.96x / 1.51x
    pytest.param(
        ("stream", 1, 1, 1216), "does_not_help", {"wh": 0.70, "bh": 0.64}, id="does_not_help_chunk1tile_1216tiles"
    ),  # WH 0.70x, BH 0.64x
    pytest.param(
        ("window", None, 128, 2, 32, "oneshot_mcast"),
        "helps",
        {"bh": (1.19, 1.22)},
        id="window_helps_fullgrid_w128_r050",
    ),  # BH 1.20x (regime), 1.21x [1.19-1.22] (sweep)
    pytest.param(
        ("window", None, 8, 8, 32, "oneshot_mcast"),
        "does_not_help",
        {"bh": (0.99, 1.00)},
        id="window_does_not_help_fullgrid_w8_r0125",
    ),  # BH 1.00x (regime), 0.99x (sweep)
    pytest.param(
        ("window", 32, 128, 8, 32, "oneshot_mcast"),
        "does_not_help",
        {"bh": (0.96, 1.00)},
        id="window_does_not_help_32cores_w128_r0125",
    ),  # BH 1.00x (regime), 0.96x (sweep)
    pytest.param(
        ("window", None, 32, 2, 1, "lockstep_mcast"),
        "does_not_help",
        {"bh": 0.88},
        id="window_does_not_help_lockstep_chunk1",
    ),  # BH 0.88x
]


def _stream_regime(device, chunk_rows, d_cols, num_chunks):
    n = _num_cores(device)
    s_kv = num_chunks * chunk_rows * TILE
    x = torch.ones(s_kv, d_cols * TILE, dtype=torch.float32)
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = _make_output(device, n)

    # Correctness: both variants must deliver identical per-Tensix-core sums.
    outs = {}
    fns = {}
    for variant in VARIANTS:
        desc = create_program_descriptor(tt_x, tt_out, variant=variant, chunk_rows=chunk_rows)
        fns[variant] = lambda x=tt_x, o=tt_out, d=desc: ttnn.generic_op([x, o], d)
        outs[variant] = ttnn.to_torch(fns[variant]() or tt_out).to(torch.float32)
    assert_with_pcc(outs["per_core_dram"], outs["mcast"], 0.9999)

    chunk_tiles = chunk_rows * d_cols
    desc = (
        f"{n} Tensix cores, {num_chunks * chunk_tiles} tiles total in {num_chunks} chunks of {chunk_tiles} tiles "
        f"({chunk_tiles * 2048} B/chunk)"
    )
    return fns, "per_core_dram", "mcast", desc


def _window_regime(device, num_cores, window_tiles, tile_rows_per_core, chunk_tiles, lever):
    if num_cores is not None:
        require_cores(device, num_cores, what=f"a {num_cores}-Tensix-core window regime")
    cols, rows_grid = window_rectangle(device, num_cores)
    n = cols * rows_grid
    rows = n * tile_rows_per_core * TILE
    tt_x, tt_w, tt_out, expected = _window_tensors(device, rows, window_tiles)
    fns = _window_fns(device, tt_x, tt_w, tt_out, ["per_core", lever], num_cores, chunk_tiles)
    for fn in fns.values():
        _check_window(fn, tt_out, expected)
    xbytes = rows * window_tiles * TILE * 2
    desc = (
        f"{n} Tensix cores ({cols}x{rows_grid}), window {window_tiles} tiles ({window_tiles * 2048} B) in "
        f"{chunk_tiles}-tile chunks, x [{rows}, {window_tiles * TILE}] = {xbytes / 2**20:.1f} MiB "
        f"({tile_rows_per_core} tile-rows per Tensix core, r = {1 / tile_rows_per_core:.3f})"
    )
    return fns, "per_core", lever, desc


@pytest.mark.parametrize("case,kind,measured", _REGIME_CASES)
def test_shared_input_reuse_regimes(request, device, case, kind, measured):
    """When reading a shared input once and multicasting it pays, and when it does not.

    Streamed input (same total stream, different chunk size): the per-chunk mcast handshake (wait for all
    receivers' acks, then broadcast) has a fixed cost per chunk. With few large chunks it is amortized and the
    DRAM-read saving wins; with many small chunks the handshake dominates and `mcast` loses to per-Tensix-core
    reads.

    Resident window (x * w, every Tensix core streams its own rows and needs all of w): a one-shot multicast of
    w removes (N - 1) copies of the window from DRAM. It pays when the window is a large share of each core's
    own x bytes (r >= 0.5) and not when it is small (r <= 0.125), whatever the Tensix-core count; a per-chunk
    handshake (lockstep, 1-tile chunks) costs more than the reads it saves.
    Asserts against a bound derived from the speedup measured on this architecture (`check_regime`);
    `measure` regimes and architectures with no measured number are logged only.
    """
    if case[0] == "stream":
        fns, base_name, lever, desc = _stream_regime(device, *case[1:])
    else:
        fns, base_name, lever, desc = _window_regime(device, *case[1:])

    def measure():
        samples = interleaved_ns(device, fns, trials=_REGIME_TRIALS, launches=_INNER, warmup=N_WARMUP)
        return samples[base_name], samples[lever]

    check_regime(
        device,
        request.node.callspec.id,
        kind,
        measured,
        measure,
        detail=f"{desc} | {base_name} vs {lever}, {_INNER} launches per trial",
    )


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
    if num_cores is not None:
        require_cores(device, num_cores, what=f"window correctness on {num_cores} Tensix cores")
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
    of pages, one 2048-byte tile each, every time; the A/B windows are interleaved within each placement). Returns
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
        samples = interleaved_ns(device, fns, trials=trials, launches=_INNER, warmup=N_WARMUP)
        per_rep.append({k: statistics.median(v) for k, v in samples.items()})
        for t in (tt_x, tt_w, tt_out, pad):
            ttnn.deallocate(t)
    med = {v: statistics.median(r[v] for r in per_rep) for v in variants}
    spread = {
        v: (min(r[variants[0]] / r[v] for r in per_rep), max(r[variants[0]] / r[v] for r in per_rep))
        for v in variants[1:]
    }
    return med, spread, rows


def _rectangle_fits(device, num_cores):
    try:
        window_rectangle(device, num_cores)
    except ValueError:
        return False
    return True


@sweep_gate("SIR_SWEEPS")
def test_shared_input_reuse_window_sweep(device):
    """Log-only sweep: Tensix cores x window tiles x stream bytes (env SIR_WIN_*; the CLI's --window mode).

    A measurement sweep: skipped unless SIR_SWEEPS=1 (the CLI's --window sets it) or EXAMPLES_SWEEPS=1.
    Tensix-core counts that do not form a rectangle on this compute grid are left out (and logged)."""
    cores_list = _env_list("SIR_WIN_CORES", "16,32,64,full")
    too_big = [c for c in cores_list if not _rectangle_fits(device, c)]
    if too_big:
        logger.warning(
            f"[shared_input_reuse] window sweep: skipping Tensix-core counts {too_big} (do not fit the grid)"
        )
    cores_list = [c for c in cores_list if c not in too_big]
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
        f"median of {repeats} DRAM placements x {trials}x{_INNER} launches  (cores = Tensix cores) ===",
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
