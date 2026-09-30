# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quasar-native binary_ng: proof that the native factory ran, and that it computed the right answer.

Routing is read from the inspector's kernels.yaml, which names the kernel SOURCES actually bound. That
is the only observable that separates the two factories at this stage -- they are a mechanical copy of
each other, so cycles, values and the active-core set are identical by construction. Numbers come from
the shared _run helper (golden + PCC floor + constant-output guard).

Configuration-gated on TTNN_QSR_NATIVE rather than parametrized: the program cache resolves the factory
from a cached index, so flipping the knob in-process would silently re-run the previous program. One
process per state -- see the run commands in debug/run_targeted_matrix.sh.
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.binary_ng_quasar_test_utils import _run

# Resolve the inspector output the way the runtime does: <logs_dir>/generated/inspector, where
# logs_dir is TT_METAL_LOGS_PATH if set, else the root dir (TT_METAL_HOME, else cwd). See
# tt_metal/llrt/rtoptions.cpp. A hardcoded absolute path would be the only one in this directory and
# would silently miss the file under any relocated logs dir.
_LOGS_ROOT = pathlib.Path(os.environ.get("TT_METAL_LOGS_PATH") or os.environ.get("TT_METAL_HOME") or os.getcwd())
INSPECTOR_YAML = _LOGS_ROOT / "generated" / "inspector" / "kernels.yaml"

# Wall-clock at module import. pytest imports every test module during collection, strictly before any
# fixture runs, so kernels.yaml -- created when the `device` fixture opens the device -- is necessarily
# newer than this. See bound_kernel_sources() for why that matters.
_IMPORT_TIME = time.time()

# The benchmark shape: 32x40 tiles = 1280 tiles, ~40/core on the Quasar simulator's 8x4 worker grid.
# Matches _INTERLEAVED_SHAPE in test_binary_ng_no_bcast.py so cycle counts are comparable.
_INTERLEAVED_SHAPE = (32 * 32, 40 * 32)


def _native_enabled():
    # "0" and unset both mean OFF -- matching native_tuning()'s env_bool in binary_ng_utils.cpp. A
    # `"TTNN_QSR_NATIVE" in os.environ` test here would silently disagree with the C++ side for =0.
    return os.environ.get("TTNN_QSR_NATIVE", "") not in ("", "0")


def bound_kernel_sources():
    """Kernel `source:` lines from THIS process's inspector output.

    Do NOT unlink kernels.yaml first: the inspector opens it once with std::ios::trunc at device
    creation and holds the handle, and the `device` fixture runs before the test body -- so deleting the
    path unlinks a live inode and it never reappears. Truncate-at-open is what makes the contents this
    process's; the mtime assertion is what rejects a stale leftover, which for the fallback arm would
    otherwise be a false green (a previous run's file holds exactly the kernels_dfb sources it asserts).
    """
    mtime = INSPECTOR_YAML.stat().st_mtime
    assert mtime >= _IMPORT_TIME, (
        f"{INSPECTOR_YAML} predates this process (mtime {mtime} < import {_IMPORT_TIME}); it is a stale "
        "leftover, so it proves nothing about this run. Is the inspector disabled?"
    )
    text = INSPECTOR_YAML.read_text()
    return [ln.split("source:", 1)[1].strip() for ln in text.splitlines() if "source:" in ln]


def _run_benchmark_add(device):
    # The shared helper, so this test inherits the same golden, PCC floor and constant-output guard as
    # the rest of the suite rather than rolling its own weaker check. bf16 add on the native slice is in
    # fact bit-exact against torch; test_native_output_is_bit_exact asserts that stronger property.
    return _run(device, "add", ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16, _INTERLEAVED_SHAPE)


def _add_returning_operands(device, h_tiles=32, w_tiles=40, seed=0):
    # _run returns only the output, and the bit-exact oracle needs the operands AS THE DEVICE SAW THEM
    # (bf16-rounded), so build them here rather than re-deriving from the fp32 originals.
    torch.manual_seed(seed)
    shape = (h_tiles * 32, w_tiles * 32)
    cfg = {
        "dtype": ttnn.bfloat16,
        "device": device,
        "layout": ttnn.TILE_LAYOUT,
        "memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }
    ta = ttnn.from_torch(torch.randn(shape, dtype=torch.float32), **cfg)
    tb = ttnn.from_torch(torch.randn(shape, dtype=torch.float32), **cfg)
    out = ttnn.experimental.quasar.add(ta, tb, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    return out, ta, tb


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
def test_native_output_is_bit_exact(device):
    """bf16 add on this slice is EXACT, so compare bits rather than PCC.

    fp32_dest_acc_en is false for bf16 and HiFi4 is fidelity-inert for FPU add, so the device result
    equals torch's bf16 add bit-for-bit. The int16 view is required, not cosmetic: torch.equal on
    floats reports False for NaN vs itself. randn produces no NaN/Inf, so that path stays unexercised.
    """
    out, ta, tb = _add_returning_operands(device)
    golden = (ttnn.to_torch(ta).float() + ttnn.to_torch(tb).float()).to(torch.bfloat16)
    got = ttnn.to_torch(out)
    mismatches = (got != golden).sum().item()
    assert torch.equal(
        got.contiguous().view(torch.int16), golden.contiguous().view(torch.int16)
    ), f"{mismatches} of {golden.numel()} elements differ from the bf16 oracle"


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
def test_native_factory_is_engaged(device):
    _run_benchmark_add(device)
    sources = bound_kernel_sources()
    qsr = [s for s in sources if "kernels_qsr/" in s]
    dfb = [s for s in sources if "kernels_dfb/" in s]
    assert len(qsr) == 3, f"expected 3 kernels_qsr sources, got {qsr}"
    assert dfb == [], f"native run still bound kernels_dfb sources: {dfb}"


@pytest.mark.skipif(_native_enabled(), reason="negative control needs TTNN_QSR_NATIVE off")
def test_fallback_factory_is_engaged(device):
    _run_benchmark_add(device)
    sources = bound_kernel_sources()
    assert [s for s in sources if "kernels_qsr/" in s] == [], "fallback run bound native kernels"
    assert len([s for s in sources if "kernels_dfb/" in s]) == 3, f"expected 3 kernels_dfb sources, got {sources}"


# --- Uneven tile counts ---------------------------------------------------------------------------
#
# The native path once required total_tiles % (num_cores * lcm(R,C,W)) == 0. That gate is gone: each
# kernel derives its own share from thread_id and num_threads, so a remainder gives the low thread ids
# one extra tile and a thread may legitimately draw ZERO tiles.
#
# Shapes are (h_tiles, w_tiles) products chosen to hit the distinct behaviours, with Tc = the
# per-CLUSTER tile count on the simulator's 8x4 (32-cluster) grid -- a metal CoreCoord addresses a whole
# Quasar cluster, so split_work_to_cores distributes over clusters. Ring depth is
# entries_per_thread * max(R,C), i.e. 2 at the default thread counts and 8 at R=C=4, so which Tc first
# makes the tile stream WRAP depends on the arm:
#
#   Tc=1  empty threads on every axis      Tc=5  tail on every axis, no empties
#   Tc=2  empties on R/C while W is even   Tc=6  tail on R/C while W is even
#   Tc=3  a single empty thread            Tc=9  wraps the ring on every arm
#   Tc=4  even everywhere (the control)    Tc=41 tail at depth, wraps repeatedly
#
# The "empty thread" readings hold only on a multi-thread arm -- at the default 1,1,1 every count is
# even by definition. _NATIVE_ARMS is what makes them real; see test_native_uneven_tile_counts_multithread.
#
# 31/33/129 instead exercise the CORE-level split: fewer cores than the grid, and two core groups whose
# counts differ by one.
_RAGGED_TILE_COUNTS = [
    32,  # Tc=1  -- at 4,4,2: 3 of 4 readers, 3 of 4 Neos and 1 of 2 writers get NO work
    64,  # Tc=2
    96,  # Tc=3
    128,  # Tc=4  -- control, even everywhere
    160,  # Tc=5  -- at 2,4,2 the C and W axes disagree about who owns the tail
    192,  # Tc=6
    288,  # Tc=9
    1312,  # Tc=41 -- tail at depth
    31,  # fewer cores than the grid; no core gets zero, the grid is under-filled
    33,  # two core groups AND empty threads
    129,  # two core groups, tails, no empties
]

# (R, C, W) arms run as subprocesses. Constrained by two independent rules, so most tuples are NOT
# usable: the gate requires R <= C and W <= C, and each DFB requires max(p,c) % min(p,c) == 0. A tuple
# violating either falls back to the single-threaded factory and then passes while testing nothing --
# which is why every arm asserts routing, not just values.
#
# 4,4,2 alone is NOT sufficient: its input DFBs are entirely num_tcs_to_rr = 1, so it never reaches the
# handle_final_credits tail branch. 1,4,4 and 4,4,1 are what drive num_tcs_to_rr = 4, with one thread
# owning every counter, on the producer and consumer side respectively.
#   1,1,1  degenerate control -- every N = 1, and no thread can draw zero tiles
#   4,4,2  the measured operating point; maximal empty-thread pressure at low Tc
#   1,4,4  num_tcs_to_rr = 4 on the PRODUCER side
#   4,4,1  num_tcs_to_rr = 4 on the CONSUMER side
#   2,4,2  N = 2 on both sides at once
_NATIVE_ARMS = [(1, 1, 1), (4, 4, 2), (1, 4, 4), (4, 4, 1), (2, 4, 2)]

# Run in a fresh process per arm: native_tuning() reads the thread counts ONCE per process into a
# static, and the program cache resolves the factory from a cached index, so setting the env vars
# mid-process changes nothing. Same constraint that makes this module gate on TTNN_QSR_NATIVE rather
# than parametrize it. Routing is read after close_device() so the inspector has certainly flushed.
#
# The child is a fixed script. Its parameters arrive as data in the environment (ARM_RCW and
# ARM_TILE_COUNTS, JSON), never spliced into the source.
_ARM_SRC = """
import json, os, pathlib, sys
import torch, ttnn

R, C, W = json.loads(os.environ["ARM_RCW"])
TILE_COUNTS = json.loads(os.environ["ARM_TILE_COUNTS"])

mismatches, failures = [], []
device = ttnn.open_device(device_id=0)
try:
    for tiles in TILE_COUNTS:
        torch.manual_seed(tiles)
        cfg = dict(dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT,
                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ta = ttnn.from_torch(torch.randn((32, tiles * 32), dtype=torch.float32), **cfg)
        tb = ttnn.from_torch(torch.randn((32, tiles * 32), dtype=torch.float32), **cfg)
        out = ttnn.experimental.quasar.add(ta, tb, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                           dtype=ttnn.bfloat16)
        golden = (ttnn.to_torch(ta).float() + ttnn.to_torch(tb).float()).to(torch.bfloat16)
        got = ttnn.to_torch(out)
        if not torch.equal(got.contiguous().view(torch.int16), golden.contiguous().view(torch.int16)):
            mismatches.append("{} tiles: {} of {} elements differ".format(
                tiles, int((got != golden).sum().item()), golden.numel()))
finally:
    ttnn.close_device(device)

root = pathlib.Path(os.environ.get("TT_METAL_LOGS_PATH") or os.environ.get("TT_METAL_HOME") or os.getcwd())
sources = [ln.split("source:", 1)[1].strip()
           for ln in (root / "generated" / "inspector" / "kernels.yaml").read_text().splitlines()
           if "source:" in ln]
if [s for s in sources if "kernels_dfb/" in s]:
    failures.append("fell back to kernels_dfb, so this arm proved nothing")
if not [s for s in sources if "kernels_qsr/" in s]:
    failures.append("bound no kernels_qsr sources")

if failures or mismatches:
    print("FAIL R={} C={} W={}".format(R, C, W))
    for f in failures + mismatches:
        print("  " + f)
    # 2 means only the output is wrong: every op ran and routed as expected.
    sys.exit(1 if failures else 2)
print("OK R={} C={} W={} over {} shapes".format(R, C, W, len(TILE_COUNTS)))
"""


def _run_arm(rcw, tile_counts, knobs, timeout):
    """Run one (R, C, W) arm of _ARM_SRC in its own process.

    Every inherited TTNN_QSR_* variable is stripped first, so an arm's configuration is exactly the
    thread counts plus `knobs`, whatever the calling shell has set.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("TTNN_QSR_")}
    env.update(
        {
            "TTNN_QSR_NATIVE": "1",
            "TTNN_QSR_READER_THREADS": str(rcw[0]),
            "TTNN_QSR_COMPUTE_THREADS": str(rcw[1]),
            "TTNN_QSR_WRITER_THREADS": str(rcw[2]),
            "ARM_RCW": json.dumps(list(rcw)),
            "ARM_TILE_COUNTS": json.dumps(list(tile_counts)),
            **knobs,
        }
    )
    return subprocess.run([sys.executable, "-c", _ARM_SRC], env=env, capture_output=True, text=True, timeout=timeout)


class _WrongOutput(Exception):
    """An arm ran to completion on the expected factory, and its output differs from the bf16 oracle."""


def _check_arm(p, what, expect_wrong=None):
    """Pass on exit 0, raise _WrongOutput on exit 2, and fail the test on anything else.

    The arm scripts exit 2 only when every op completed and routed as expected and the output is wrong.
    An expected-failure marker restricted to _WrongOutput therefore cannot hide a crash, a hang, a
    refusal or a fallback. expect_wrong maps each shape's label to the fraction of elements that a known
    defect leaves wrong. With it, only that signature raises _WrongOutput, so a new defect fails the test.
    """
    if p.returncode == 2:
        if expect_wrong is not None:
            measured = {
                m.group(1): int(m.group(2)) / int(m.group(3))
                for m in re.finditer(r"^\s+(.+): (\d+) of (\d+) elements differ$", p.stdout, re.MULTILINE)
            }
            off = {
                label: (measured.get(label), expected)
                for label, expected in expect_wrong.items()
                if label not in measured or abs(measured[label] - expected) > 0.01
            }
            assert not off and measured.keys() == expect_wrong.keys(), (
                f"{what}: the output is wrong, but not with the known defect's signature "
                f"(label: (measured, expected)) {off}\n{p.stdout}"
            )
        raise _WrongOutput(f"{what}: output differs from the bf16 oracle\n{p.stdout}")
    assert p.returncode == 0, f"{what} failed:\n{p.stdout}\n{p.stderr[-2000:]}"


def _add_bit_exact(device, tiles, seed):
    """One (1 x tiles) add against the bf16 oracle. Returns (exact, mismatches, total).

    A 1-tile-tall shape is a legitimate one-dimensional sweep here because the native reader walks
    page = start_tile_id + k -- a pure linear index -- so behaviour depends on the tile COUNT alone and
    not on how it factors into height x width.

    The golden is rounded to bf16, not left in fp32: the device packs bf16, and the sum of two bf16
    values needs 9 mantissa bits where bf16 has 8, so an fp32 comparison reports ~54% mismatch on
    random data purely from rounding.

    The verdict comes from the int16 view for the reason test_native_output_is_bit_exact gives -- float
    comparison reports NaN as unequal to itself. The `!=` count is a diagnostic for the message only.
    """
    out, ta, tb = _add_returning_operands(device, h_tiles=1, w_tiles=tiles, seed=seed)
    golden = (ttnn.to_torch(ta).float() + ttnn.to_torch(tb).float()).to(torch.bfloat16)
    got = ttnn.to_torch(out)
    exact = torch.equal(got.contiguous().view(torch.int16), golden.contiguous().view(torch.int16))
    return exact, int((got != golden).sum().item()), golden.numel()


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("tiles", _RAGGED_TILE_COUNTS)
def test_native_uneven_tile_counts_are_bit_exact(device, tiles):
    before = set(bound_kernel_sources())
    exact, mismatches, total = _add_bit_exact(device, tiles, seed=tiles)
    assert exact, f"{mismatches} of {total} elements differ at {tiles} tiles"

    # Without this the test is a false green: the shape could have been rejected by some OTHER gate
    # condition, silently run the single-threaded fallback, and pass. Assert on THIS op's new bindings,
    # not the whole file -- sources accumulate for the life of the process, so a sibling module that
    # legitimately binds kernels_dfb (bcast, scalar, resnet_add) would otherwise fail this test for an
    # unrelated reason when the directory runs together.
    new_sources = [s for s in bound_kernel_sources() if s not in before]
    assert [
        s for s in new_sources if "kernels_qsr/" in s
    ], f"{tiles} tiles bound no new native kernels (new bindings: {new_sources})"
    assert [s for s in new_sources if "kernels_dfb/" in s] == [], f"{tiles} tiles fell back to kernels_dfb"


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
def test_native_even_and_uneven_share_a_process(device):
    """Even and ragged shapes must not poison each other through the program cache.

    The op has no compute_program_hash override, so the framework hashes the tensor specs -- different
    shapes therefore take different cache entries. This asserts that empirically rather than trusting
    it, and in the order most likely to break: warm the cache on an even shape, run a ragged one, then
    return to the even shape and re-check it.
    """
    for tiles in (128, 129, 128, 32, 128):
        exact, mismatches, total = _add_bit_exact(device, tiles, seed=1)
        assert exact, f"{mismatches} of {total} differ at {tiles} tiles (cache interference?)"


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("rcw", _NATIVE_ARMS, ids=lambda t: "R{}C{}W{}".format(*t))
def test_native_uneven_tile_counts_multithread(rcw):
    """Uneven tile counts only matter above one thread, so run the ragged sweep on real arms.

    Every test above this one runs at whatever thread counts the invocation happens to carry, and the
    default is 1,1,1 -- where my_tiles == num_tiles, every strided loop has stride 1, and no thread can
    draw zero tiles. Those tests would pass on a build that had never implemented uneven splitting.
    This one sets the counts explicitly, so an empty thread and a ragged tail are actually reached.

    Requests no `device` fixture on purpose: each arm opens its own device inside its own process.
    """
    p = _run_arm(rcw, _RAGGED_TILE_COUNTS, knobs={}, timeout=3600)
    _check_arm(p, f"arm R={rcw[0]} C={rcw[1]} W={rcw[2]}")


# --- Dataflow batching above one tile counter per role ---------------------------------------------
#
# A DM role that owns several tile counters batches WITHIN each counter and rotates BETWEEN them. Counter
# c holds the tiles from thread_id + c*num_threads spaced num_tcs*num_threads apart, so a batch drawn
# from one counter strides by that product. That mapping is the DFB's producer/consumer pairing as
# implemented today, not a documented contract. If the pairing changes upstream, nothing else in the
# tree notices: the output silently permutes, or a counter is asked for tiles it is never credited and
# the writer blocks forever. This test is what would notice.
#
# Per-cluster tile counts on the 32-cluster grid are tiles/32, then split over a role's counters. At 1
# and 3 some counters draw nothing (zero-work threads); at 9 every counter gets one short batch and
# none a full one; at 41 counter 0 gets one full batch of 8 plus a tail; at 65 counter 0 gets two full
# batches, the ring wraps at slot 16, and a tail of one follows.
_DM_BATCH_TILE_COUNTS = [32, 96, 288, 1312, 2080]

# Reader owns C/R counters and the writer C/W. 4,4,2 is the shipping thread split with the writer on
# two counters; 1,4,1 puts both roles on four. Together they cover every counter count the walk has
# to rotate through.
_DM_BATCH_ARMS = [(4, 4, 2), (1, 4, 1)]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("rcw", _DM_BATCH_ARMS, ids=lambda t: "R{}C{}W{}".format(*t))
def test_native_dm_batch_above_one_counter_is_bit_exact(rcw):
    """dm_batch=8 on a role that owns more than one tile counter must stay bit-exact.

    Depth 16 is the smallest legal depth: a batch occupies n consecutive slots and the ring wraps only
    afterwards, so the factory demands entries_per_thread >= 2n and entries_per_thread % n == 0.

    One process per arm, as above. The timeout is the failure detector for a broken pairing: a counter
    asked for more tiles than it receives hangs rather than corrupts. The arm inherits no TTNN_QSR_*
    knob, so a shell that set TILES_PER_CYCLE cannot turn this into a compute-batching run.
    """
    p = _run_arm(
        rcw,
        _DM_BATCH_TILE_COUNTS,
        knobs={"TTNN_QSR_DM_BATCH": "8", "TTNN_QSR_ENTRIES_PER_THREAD": "16"},
        timeout=600,
    )
    _check_arm(p, f"dm_batch=8 arm R={rcw[0]} C={rcw[1]} W={rcw[2]}")


# --- Compute batching --------------------------------------------------------------------------------
#
# TTNN_QSR_TILES_PER_CYCLE=8 batches eight tiles per tile_regs_acquire. The pack path spaces a batch by
# one entry rather than by the ring stride, so the output is right only at ring stride 1 -- R = C = W =
# 1 -- until the pack path applies the stride per tile. The factory runs any stride; the tests above
# stride 1 expect the wrong output. Per-cluster tile counts on the 32-cluster grid: 1 runs a single
# short batch and no full one; 9 runs one full batch of 8 then a tail of 1; 64 runs eight full batches,
# wrapping the 16-deep ring three times, and no tail; 65 adds a tail of 1 that starts exactly on a wrap
# boundary.
_TILES_PER_CYCLE_TILE_COUNTS = [32, 288, 2048, 2080]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("dm_batch", [1, 8], ids=["compute-only", "both-knobs"])
def test_native_tiles_per_cycle_is_bit_exact(dm_batch):
    """Compute batching at 1,1,1 must stay bit-exact, alone and together with dataflow batching.

    The compute kernel splits a thread's share into full chunks of num_tiles_per_cycle and one remainder
    chunk; a truncating divide there once dropped the tail and hung. The tile counts above reach every
    branch of that split. both-knobs is the configuration that measured 1.81x throughput at 1,1,1.
    """
    p = _run_arm(
        (1, 1, 1),
        _TILES_PER_CYCLE_TILE_COUNTS,
        knobs={
            "TTNN_QSR_TILES_PER_CYCLE": "8",
            "TTNN_QSR_DM_BATCH": str(dm_batch),
            "TTNN_QSR_ENTRIES_PER_THREAD": "16",
        },
        timeout=600,
    )
    _check_arm(p, f"tiles_per_cycle=8 dm_batch={dm_batch}")


# Above ring stride 1 compute batching runs but loses most of each batch, because the pack path spaces
# the tiles of a batch by one entry rather than by the ring stride. strict=True: once the pack path is
# fixed these tests pass, and the run fails until the marker is removed.
_PACK_STRIDE_XFAIL = pytest.mark.xfail(
    raises=_WrongOutput,
    strict=True,
    reason="compute batching above ring stride 1: the pack path does not apply the ring stride per tile",
)


def _pack_stride_wrong_fraction(tiles_per_cluster, compute_threads, batch, stride):
    """The fraction of a cluster's output tiles that the pack-stride defect leaves wrong.

    A batch of n tiles at ring stride s lands only ceil(n/s) of them. Each compute thread's share runs in
    full batches and one tail, and thread t takes one extra tile while t < tiles % threads. On craq-sim
    this matched every measured shape to within 0.1 percentage points.
    """
    landed = 0
    for thread in range(compute_threads):
        share = tiles_per_cluster // compute_threads + (thread < tiles_per_cluster % compute_threads)
        full, tail = divmod(share, batch)
        landed += full * -(-batch // stride) - (-tail // stride)
    return 1 - landed / tiles_per_cluster


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@_PACK_STRIDE_XFAIL
def test_native_tiles_per_cycle_above_stride_one():
    """4,4,2 with both batch knobs at 8 runs natively, and its output is wrong until the pack is fixed.

    The ring stride is 4 on every DFB. 288 tiles give each compute thread two or three tiles, one short
    batch; 2080 give it sixteen or seventeen, two full batches of 8 and a tail. A refusal, a hang, a
    fallback or wrong output without the pack defect's signature fails this test instead of passing as
    an expected failure.
    """
    tile_counts = [288, 2080]
    p = _run_arm(
        (4, 4, 2),
        tile_counts,
        knobs={
            "TTNN_QSR_TILES_PER_CYCLE": "8",
            "TTNN_QSR_DM_BATCH": "8",
            "TTNN_QSR_ENTRIES_PER_THREAD": "16",
        },
        timeout=900,
    )
    # The linear split gives each of the 32 clusters an equal share of these counts.
    expect_wrong = {f"{t} tiles": _pack_stride_wrong_fraction(t // 32, 4, 8, 4) for t in tile_counts}
    _check_arm(p, "4,4,2 with both batch knobs at 8", expect_wrong=expect_wrong)


# Each batching guard, the knob values that trip it, and the text its TT_FATAL carries. A batch needs a
# ring at least twice as deep (double buffering) and a depth that is a multiple of the batch (a batch
# occupies consecutive slots and the ring wraps only after it). Compute batching above ring stride 1 is
# not refused; test_native_tiles_per_cycle_above_stride_one checks what it produces.
_GUARD_ARMS = [
    (
        "tiles_per_cycle-depth",
        (1, 1, 1),
        {"TTNN_QSR_TILES_PER_CYCLE": "8", "TTNN_QSR_ENTRIES_PER_THREAD": "8"},
        "TTNN_QSR_TILES_PER_CYCLE=8 needs TTNN_QSR_ENTRIES_PER_THREAD >= 16",
    ),
    (
        "tiles_per_cycle-divides",
        (1, 1, 1),
        {"TTNN_QSR_TILES_PER_CYCLE": "8", "TTNN_QSR_ENTRIES_PER_THREAD": "20"},
        "TTNN_QSR_TILES_PER_CYCLE=8 must divide TTNN_QSR_ENTRIES_PER_THREAD=20",
    ),
    (
        "dm_batch-depth",
        (1, 1, 1),
        {"TTNN_QSR_DM_BATCH": "8", "TTNN_QSR_ENTRIES_PER_THREAD": "8"},
        "TTNN_QSR_DM_BATCH=8 needs TTNN_QSR_ENTRIES_PER_THREAD >= 16",
    ),
    (
        "dm_batch-divides",
        (1, 1, 1),
        {"TTNN_QSR_DM_BATCH": "8", "TTNN_QSR_ENTRIES_PER_THREAD": "20"},
        "TTNN_QSR_DM_BATCH=8 must divide TTNN_QSR_ENTRIES_PER_THREAD=20",
    ),
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("arm", _GUARD_ARMS, ids=[a[0] for a in _GUARD_ARMS])
def test_native_batching_guards_refuse(arm):
    """A knob setting the factory cannot run must fail with its own message, not run and hang.

    The message is asserted, not only a non-zero exit: an arm that died for another reason -- no
    simulator, a build break -- would otherwise pass as a refusal.
    """
    _, rcw, knobs, message = arm
    p = _run_arm(rcw, [32], knobs=knobs, timeout=300)
    assert p.returncode != 0, f"{arm[0]}: the factory ran instead of refusing:\n{p.stdout}"
    assert message in p.stdout + p.stderr, f"{arm[0]}: refused for another reason:\n{p.stdout}\n{p.stderr[-2000:]}"


# --- Borrowed L1 shards ------------------------------------------------------------------------------
#
# With all three operands L1-sharded with one memory config the factory borrows the resident shards: each
# DFB is the shard itself, the reader publishes credits, the writer has nothing to do, and no byte moves
# over the NoC. A shard below is (name, strategy, shard shape in elements, inclusive core range (x0, y0, x1,
# y1), tensor shape[, options]). Every grid fits the simulator's 8x4 worker grid.
#
# Per-core tile counts: 16 divides by the ring stride of every arm below (1,4,1 and 4,4,2 both have
# stride 4 on each DFB); 12 leaves a remainder of 4 after one full compute chunk of 8 at 1,1,1.
# height16_again repeats height16's spec while height16's tensors are still alive, so it hits the
# program cache and passes only if the cached program rebinds the borrowed shards to the new buffers.
# height16_in_place adds b into a with add_, so the output DFB and the a DFB borrow one shard.
# height256 puts 256 tiles on each core, past the 255-entry limit of the derived NoC rings, which a
# borrowed ring does not have; its cores hold none of the other shards.
_BORROWED_SHARDS = [
    ("height16", "height", [4 * 32, 4 * 32], (0, 0, 0, 3), (4 * 4 * 32, 4 * 32)),
    ("block16", "block", [4 * 32, 4 * 32], (0, 0, 1, 1), (2 * 4 * 32, 2 * 4 * 32)),
    ("width16", "width", [4 * 32, 4 * 32], (0, 0, 3, 0), (4 * 32, 4 * 4 * 32)),
    ("block12", "block", [3 * 32, 4 * 32], (0, 0, 1, 1), (2 * 3 * 32, 2 * 4 * 32)),
    ("height16_again", "height", [4 * 32, 4 * 32], (0, 0, 0, 3), (4 * 4 * 32, 4 * 32)),
    ("height16_in_place", "height", [4 * 32, 4 * 32], (0, 0, 0, 3), (4 * 4 * 32, 4 * 32), {"in_place": True}),
    ("height256", "height", [256 * 32, 32], (4, 0, 4, 1), (2 * 256 * 32, 32)),
]

# Tensors that do not divide into whole shards, so the last clusters hold fewer real tiles than the
# others. Every cluster still processes its full shard: the buffer allocates it on every cluster, and
# the pad tiles of the output are never read back. The shard tile count, 4 or 16, divides by the ring
# stride of every borrowed arm.
_UNEVEN_SHARDS = [
    ("height_last_1_of_4", "height", [4 * 32, 32], (0, 0, 0, 1), (5 * 32, 32)),
    ("width_last_1_of_4", "width", [32, 4 * 32], (0, 0, 1, 0), (32, 5 * 32)),
    ("block_corner_1_of_4", "block", [2 * 32, 2 * 32], (0, 0, 1, 1), (3 * 32, 3 * 32)),
    ("block_corner_1_of_16", "block", [4 * 32, 4 * 32], (0, 0, 2, 2), (9 * 32, 9 * 32)),
]

# Shards whose tile count does not divide by 4. Every shard runs on all four Neos: the borrowed part keeps
# the largest multiple of 4, and the 1-3 tiles past it go through small owned rings, one entry per compute
# thread. A shard of 1-3 tiles has no borrowed part, so all of it goes through the rings.
# height1 is one tile per core. uneven2 is a column three tiles tall over shards of two, so the boundary
# core holds a partial shard and processes its full rounded-up count anyway. idle_clusters spreads the
# same column over four clusters, so two of them hold no real tile at all and compute only padding.
# width_10_uneven leaves the second cluster 5 real tiles of 10. 5, 6, 7, 9, 10 and 67 tiles leave 1, 2, 3,
# 1, 2 and 3 tiles for the rings. block_6_in_place adds into a, so a's shard receives the copied-out
# tail; block_6_again repeats block_6's spec while its tensors are alive, a cache hit that must rebind.
_SMALL_SHARDS = [
    ("height1", "height", [32, 32], (0, 0, 0, 3), (4 * 32, 32)),
    ("uneven2", "height", [2 * 32, 32], (0, 0, 0, 1), (3 * 32, 32)),
    ("idle_clusters", "height", [2 * 32, 32], (0, 0, 0, 3), (3 * 32, 32)),
    ("block_6", "block", [2 * 32, 3 * 32], (0, 0, 1, 1), (2 * 2 * 32, 2 * 3 * 32)),
    ("height_9", "height", [9 * 32, 32], (0, 0, 0, 1), (2 * 9 * 32, 32)),
    ("width_10_uneven", "width", [32, 10 * 32], (0, 0, 1, 0), (32, 15 * 32)),
    ("height_3", "height", [3 * 32, 32], (0, 0, 0, 1), (2 * 3 * 32, 32)),
    ("height_5", "height", [5 * 32, 32], (0, 0, 0, 1), (2 * 5 * 32, 32)),
    ("height_7", "height", [7 * 32, 32], (0, 0, 0, 1), (2 * 7 * 32, 32)),
    ("height_67", "height", [67 * 32, 32], (5, 0, 5, 1), (2 * 67 * 32, 32)),
    ("block_6_in_place", "block", [2 * 32, 3 * 32], (0, 0, 1, 1), (2 * 2 * 32, 2 * 3 * 32), {"in_place": True}),
    ("block_6_again", "block", [2 * 32, 3 * 32], (0, 0, 1, 1), (2 * 2 * 32, 2 * 3 * 32)),
]

# One shard per child: the profiler CSV has no dispatch key, so two ops in one process would blend per core.
_FOUR_NEO_SHARDS = [
    s for s in _SMALL_SHARDS if s[0] in ("height1", "uneven2", "height_3", "block_6", "height_7", "height_67")
]

# Inputs and output sharded on the same grid but with different shard specs: the height shards of core
# i and the width shard of core i cover different tiles, so borrowing in place would add the wrong
# rows into every column. The "out" option is the output's own (strategy, shard shape, core range).
_MISMATCHED_SHARDS = [
    (
        "height_in_width_out",
        "height",
        [32, 4 * 32],
        (0, 0, 0, 3),
        (4 * 32, 4 * 32),
        {"out": ("width", [4 * 32, 32], (0, 0, 0, 3))},
    ),
]

# Tuned thread counts. On the borrowed path the factory runs one reader and one writer thread whatever
# is tuned, since they copy only the tail tiles there. 1,1,1 takes the default compute batch of
# min(8, shard tiles) and stride 1. 1,4,1 puts the reader and the writer on four counters each, and 4,4,2
# checks that the factory maps its tuned counts to that same program. So the DM side never outnumbers
# the Tensix side here, a case that some craq-sim builds corrupt.
_BORROWED_ARMS = [(1, 1, 1), (1, 4, 1), (4, 4, 2)]

# Same shape as _ARM_SRC: a fixed child script, parameters as JSON in the environment. ARM_EXPECT names
# the factory every op in the child must have bound: "qsr" for native, "dfb" for the fallback. The
# strategy "l1" or "dram" places a tensor interleaved, and then the shard shape and range are None. A
# shard entry may carry a sixth element, a dict of options: "out" gives the output its own placement
# (otherwise it takes the inputs'), "b" does the same for b, "in_place" adds b into a with add_, and
# "grid" (x0, y0, x1, y1) runs the op on that sub-core grid. Every tensor stays alive to the end, so an
# entry that repeats an earlier spec lands at new addresses and must hit the program cache. With
# ARM_EXPECT_NEOS or ARM_EXPECT_DMS (and the device profiler on), every core must show that many Neos
# running the compute, or DM cores running a reader or writer thread; ARM_EXPECT_CORES names how many
# cores must report.
_SHARD_ARM_SRC = """
import collections, csv, json, os, pathlib, sys
import torch, ttnn

R, C, W = json.loads(os.environ["ARM_RCW"])
SHARDS = json.loads(os.environ["ARM_SHARDS"])
EXPECT = os.environ["ARM_EXPECT"]
EXPECT_NEOS = int(os.environ.get("ARM_EXPECT_NEOS", "0"))
EXPECT_DMS = int(os.environ.get("ARM_EXPECT_DMS", "0"))
EXPECT_CORES = int(os.environ.get("ARM_EXPECT_CORES", "0"))
STRATEGY = {"height": ttnn.ShardStrategy.HEIGHT, "block": ttnn.ShardStrategy.BLOCK,
            "width": ttnn.ShardStrategy.WIDTH}
INTERLEAVED = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}
root = pathlib.Path(os.environ.get("TT_METAL_LOGS_PATH") or os.environ.get("TT_METAL_HOME") or os.getcwd())
profile_csv = root / "generated" / "profiler" / ".logs" / "profile_log_device.csv"
if (EXPECT_NEOS or EXPECT_DMS) and profile_csv.exists():
    profile_csv.unlink()


def memory_config(strategy, shard_shape, rng):
    if strategy in INTERLEAVED:
        return INTERLEAVED[strategy]
    x0, y0, x1, y1 = rng
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((x0, y0), (x1, y1))}),
        strategy=STRATEGY[strategy],
        use_height_and_width_as_shard_shape=True,
    )


mismatches, failures = [], []
keep, seen = [], set()
device = ttnn.open_device(device_id=0)
try:
    for i, entry in enumerate(SHARDS):
        name, strategy, shard_shape, rng, shape = entry[:5]
        opts = entry[5] if len(entry) > 5 else {}
        torch.manual_seed(i)
        mem = memory_config(strategy, shard_shape, rng)
        out_mem = memory_config(*opts["out"]) if "out" in opts else mem
        b_mem = memory_config(*opts["b"]) if "b" in opts else mem
        cfg = dict(dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        ta = ttnn.from_torch(torch.randn(shape, dtype=torch.float32), memory_config=mem, **cfg)
        tb = ttnn.from_torch(torch.randn(shape, dtype=torch.float32), memory_config=b_mem, **cfg)
        golden = (ttnn.to_torch(ta).float() + ttnn.to_torch(tb).float()).to(torch.bfloat16)
        cached = device.num_program_cache_entries()
        grid = {}
        if "grid" in opts:
            x0, y0, x1, y1 = opts["grid"]
            grid = {"sub_core_grids": ttnn.CoreRangeSet({ttnn.CoreRange((x0, y0), (x1, y1))})}
        if opts.get("in_place"):
            out = ttnn.experimental.quasar.add_(ta, tb)
        else:
            out = ttnn.experimental.quasar.add(ta, tb, memory_config=out_mem, dtype=ttnn.bfloat16, **grid)
        spec = json.dumps(entry[1:])
        if spec in seen and device.num_program_cache_entries() != cached:
            failures.append("{}: repeats an earlier spec but missed the program cache".format(name))
        seen.add(spec)
        keep.extend([ta, tb, out])
        got = ttnn.to_torch(out)
        if not torch.equal(got.contiguous().view(torch.int16), golden.contiguous().view(torch.int16)):
            mismatches.append("{}: {} of {} elements differ".format(
                name, int((got != golden).sum().item()), golden.numel()))
finally:
    ttnn.close_device(device)

sources = [ln.split("source:", 1)[1].strip()
           for ln in (root / "generated" / "inspector" / "kernels.yaml").read_text().splitlines()
           if "source:" in ln]
qsr = [s for s in sources if "kernels_qsr/" in s]
dfb = [s for s in sources if "kernels_dfb/" in s]
if EXPECT == "qsr" and (dfb or not qsr):
    failures.append("expected every op on kernels_qsr, bound qsr={} dfb={}".format(len(qsr), len(dfb)))
if EXPECT == "dfb" and (qsr or not dfb):
    failures.append("expected every op on kernels_dfb, bound qsr={} dfb={}".format(len(qsr), len(dfb)))
if EXPECT_NEOS or EXPECT_DMS:
    # A Neo ran the compute on a core when one of its TRISCs recorded a kernel zone there, and a DM core
    # ran a reader or writer thread when it recorded one.
    lines = profile_csv.read_text().splitlines() if profile_csv.exists() else []
    neos, dms = collections.defaultdict(set), collections.defaultdict(set)
    for row in csv.DictReader(lines[1:], skipinitialspace=True):
        risc = row["RISC processor type"].strip()
        if "KERNEL" not in row["zone name"]:
            continue
        core = (row["core_x"].strip(), row["core_y"].strip())
        if risc.startswith("QUASAR_NEO"):
            neos[core].add(risc.split("_")[1])
        elif risc.startswith("QUASAR_DM"):
            dms[core].add(risc)
    for expected, found, what in ((EXPECT_NEOS, neos, "Neos"), (EXPECT_DMS, dms, "DM cores")):
        counts = sorted({len(v) for v in found.values()})
        if expected and counts != [expected]:
            failures.append("expected {} {} on every core, the profiler shows {}".format(expected, what, counts))
        if expected and EXPECT_CORES and len(found) != EXPECT_CORES:
            failures.append("expected {} cores to run {}, the profiler shows {}".format(EXPECT_CORES, what, len(found)))

if failures or mismatches:
    print("FAIL R={} C={} W={}".format(R, C, W))
    for f in failures + mismatches:
        print("  " + f)
    # 2 means only the output is wrong: every op ran and routed as expected.
    sys.exit(1 if failures else 2)
print("OK R={} C={} W={} over {} shards".format(R, C, W, len(SHARDS)))
"""


def _run_shard_arm(rcw, shards, expect, timeout, knobs=None):
    """Run one (R, C, W) arm of _SHARD_ARM_SRC in its own process.

    Inherited TTNN_QSR_* variables are stripped, as in _run_arm; `knobs` adds settings on top. With the
    device profiler on, inherited DPRINT and streaming-profiler variables are stripped too, because the
    runtime refuses to start with either of them next to it. A build without Tracy skips the test.
    """
    profiling = (knobs or {}).get("TT_METAL_DEVICE_PROFILER") == "1"
    stripped = ("TTNN_QSR_", "TT_METAL_DPRINT_", "TT_METAL_STREAMING_PROFILER") if profiling else ("TTNN_QSR_",)
    env = {k: v for k, v in os.environ.items() if not k.startswith(stripped)}
    env.update(
        {
            "TTNN_QSR_NATIVE": "1",
            "TTNN_QSR_READER_THREADS": str(rcw[0]),
            "TTNN_QSR_COMPUTE_THREADS": str(rcw[1]),
            "TTNN_QSR_WRITER_THREADS": str(rcw[2]),
            "ARM_RCW": json.dumps(list(rcw)),
            "ARM_SHARDS": json.dumps(list(shards)),
            "ARM_EXPECT": expect,
            **(knobs or {}),
        }
    )
    p = subprocess.run([sys.executable, "-c", _SHARD_ARM_SRC], env=env, capture_output=True, text=True, timeout=timeout)
    if profiling and "requires a Tracy-enabled build" in p.stdout + p.stderr:
        pytest.skip("the device profiler needs a Tracy-enabled build of tt-metal")
    return p


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("rcw", _BORROWED_ARMS, ids=lambda t: "R{}C{}W{}".format(*t))
def test_native_borrowed_shards_are_bit_exact(rcw):
    """Height, block and width shards must run native and bit-exact at every borrowed arm.

    A wrong credit count hangs into the timeout, credits on the wrong counter fail the bit comparison,
    and a cached program that keeps the first call's shard addresses fails the repeat.
    """
    p = _run_shard_arm(rcw, _BORROWED_SHARDS, expect="qsr", timeout=900)
    _check_arm(p, f"borrowed arm R={rcw[0]} C={rcw[1]} W={rcw[2]}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("rcw", _BORROWED_ARMS, ids=lambda t: "R{}C{}W{}".format(*t))
def test_native_borrowed_uneven_shards_are_bit_exact(rcw):
    """Shards whose last clusters hold fewer real tiles must run native and bit-exact at every arm.

    Every role processes the full shard's tile count. Roles that disagree on a boundary cluster's count
    hang into the timeout.
    """
    p = _run_shard_arm(rcw, _UNEVEN_SHARDS, expect="qsr", timeout=900)
    _check_arm(p, f"uneven shards R={rcw[0]} C={rcw[1]} W={rcw[2]}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("rcw", _BORROWED_ARMS + [(1, 2, 1)], ids=lambda t: "R{}C{}W{}".format(*t))
def test_native_borrowed_indivisible_shards_stay_native(rcw):
    """A shard whose tile count does not divide by the tuned compute count must stay native and bit-exact.

    A borrowed ring sized to the whole shard dies on the DFB host assertion, and a tail copied from the
    wrong address or packed into the wrong ring fails the bit comparison. 1,2,1 runs two-entry tail rings.
    At 1,1,1 every shard divides, and 9 and 10 tiles leave a remainder after one compute batch of 8.
    """
    p = _run_shard_arm(rcw, _SMALL_SHARDS, expect="qsr", timeout=900)
    _check_arm(p, f"indivisible shards R={rcw[0]} C={rcw[1]} W={rcw[2]}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("shard", _FOUR_NEO_SHARDS, ids=[s[0] for s in _FOUR_NEO_SHARDS])
def test_native_borrowed_indivisible_shards_use_every_compute_thread(shard):
    """A borrowed shard must run on all four Neos when 4 does not divide it, however few tiles it has.

    The borrowed part keeps the largest multiple of 4, which is none below 4 tiles, and the leftover tiles
    go through owned rings of one entry per compute thread. A factory that drops to fewer compute threads
    instead fails the Neo count, which the child reads from the device profiler; wrong output or a fallback
    fails as usual.
    """
    p = _run_shard_arm(
        (1, 4, 1),
        [shard],
        expect="qsr",
        timeout=900,
        knobs={"TT_METAL_DEVICE_PROFILER": "1", "ARM_EXPECT_NEOS": "4"},
    )
    _check_arm(p, f"{shard[0]} on four Neos")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
def test_native_borrowed_mismatched_shard_specs_fall_back():
    """Inputs and output on one grid with different shard specs must not be borrowed, and must be right.

    A predicate that compares grids alone borrows the shards in place and fails the bit comparison. The
    fallback reshards over the NoC, so both checks hold only when the predicate compares the full spec.
    """
    p = _run_shard_arm((1, 1, 1), _MISMATCHED_SHARDS, expect="dfb", timeout=900)
    _check_arm(p, "mismatched shard specs")


# 64 tiles per core: at 1,4,1 each compute thread takes 16, two full batches of 8 at ring stride 4.
_BATCHED_BORROWED_SHARDS = [
    ("block64", "block", [8 * 32, 8 * 32], (0, 0, 1, 1), (2 * 8 * 32, 2 * 8 * 32)),
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@_PACK_STRIDE_XFAIL
def test_native_borrowed_tiles_per_cycle_above_stride_one():
    """Borrowed 1,4,1 at N=8 runs natively, and its output is wrong until the pack path is fixed.

    1,4,1 is the borrowed configuration and N=8 its batch once the pack path applies the ring stride,
    so this is the test that becomes a plain pass when that fix lands.
    """
    p = _run_shard_arm(
        (1, 4, 1),
        _BATCHED_BORROWED_SHARDS,
        expect="qsr",
        timeout=900,
        knobs={"TTNN_QSR_TILES_PER_CYCLE": "8"},
    )
    _check_arm(p, "borrowed 1,4,1 at N=8", expect_wrong={"block64": _pack_stride_wrong_fraction(64, 4, 8, 4)})


# L1-interleaved operands. Bank k of an L1-interleaved tensor holds its pages k, k+32, k+64, ... back to
# back, one bank per cluster, so a cluster's slices of a, b and c hold the same pages and it can compute
# them in place. Every bank reserves ceil(P/32) pages: 100 and 250 tiles leave the high banks a pad slot
# that no page reads. l1_128_again repeats l1_128 at new addresses, a cache hit that must rebind.
_L1_INTERLEAVED = [
    ("l1_128", "l1", None, None, (4 * 32, 32 * 32)),
    ("l1_256", "l1", None, None, (8 * 32, 32 * 32)),
    ("l1_100_pad", "l1", None, None, (10 * 32, 10 * 32)),
    ("l1_250_pad", "l1", None, None, (10 * 32, 25 * 32)),
    ("l1_128_again", "l1", None, None, (4 * 32, 32 * 32)),
    ("l1_128_in_place", "l1", None, None, (4 * 32, 32 * 32), {"in_place": True}),
]

# Cases that keep the NoC path at C = 4: 160, 64 and 8 tiles leave 5, 2 and 1 per bank, which 4 does not
# divide; l1_128_sub_grid runs on 16 of the 32 bank cores; the others place an operand in DRAM. At C = 1
# the first three borrow, and l1_8_idle leaves 24 clusters with a pad slot only.
_L1_INTERLEAVED_NOC = [
    ("l1_160", "l1", None, None, (5 * 32, 32 * 32)),
    ("l1_64", "l1", None, None, (2 * 32, 32 * 32)),
    ("l1_8_idle", "l1", None, None, (2 * 32, 4 * 32)),
    ("l1_128_sub_grid", "l1", None, None, (4 * 32, 32 * 32), {"grid": (0, 0, 3, 3)}),
    ("l1_b_in_dram", "l1", None, None, (4 * 32, 32 * 32), {"b": ("dram", None, None)}),
    ("l1_out_in_dram", "l1", None, None, (4 * 32, 32 * 32), {"out": ("dram", None, None)}),
    ("dram_a", "dram", None, None, (4 * 32, 32 * 32), {"b": ("l1", None, None), "out": ("l1", None, None)}),
]

# One case per child: the profiler CSV has no dispatch key, so two ops in one process would blend per core.
# Each case carries the number of cores that must run it. height16 is a borrowed F3 shard on 4 cores: a
# known-borrowed control for the DM count that does not pass through the L1-interleaved gate.
_L1_INTERLEAVED_DM_BORROWED = [(_BORROWED_SHARDS[0], 4)] + [
    (s, 32) for s in _L1_INTERLEAVED if s[0] in ("l1_128", "l1_250_pad", "l1_128_in_place")
]
_L1_INTERLEAVED_DM_NOC = [(s, {"l1_8_idle": 8, "l1_128_sub_grid": 16}.get(s[0], 32)) for s in _L1_INTERLEAVED_NOC]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("rcw", _BORROWED_ARMS + [(1, 2, 1)], ids=lambda t: "R{}C{}W{}".format(*t))
def test_native_l1_interleaved_is_bit_exact(rcw):
    """L1-interleaved operands must run native and bit-exact at every arm, borrowed or not.

    Credits on the wrong counter or a slice read at the wrong offset fail the bit comparison, and a cached
    program that keeps the first call's slice addresses fails the repeat.
    """
    p = _run_shard_arm(rcw, _L1_INTERLEAVED + _L1_INTERLEAVED_NOC, expect="qsr", timeout=900)
    _check_arm(p, f"L1-interleaved R={rcw[0]} C={rcw[1]} W={rcw[2]}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("case, cores", _L1_INTERLEAVED_DM_BORROWED, ids=[c[0][0] for c in _L1_INTERLEAVED_DM_BORROWED])
def test_native_l1_interleaved_borrows_in_place(case, cores):
    """L1-interleaved operands whose tile count per cluster divides by 4 must be borrowed at tuned 4,4,2.

    The borrowed and the NoC path bind the same kernel sources, so the routing check cannot tell them
    apart. The thread counts can: a borrowed program runs one reader and one writer thread, and the NoC
    path runs the tuned four and two. The child reads the DM cores and the Neos from the device profiler.
    """
    p = _run_shard_arm(
        (4, 4, 2),
        [case],
        expect="qsr",
        timeout=900,
        knobs={
            "TT_METAL_DEVICE_PROFILER": "1",
            "ARM_EXPECT_DMS": "2",
            "ARM_EXPECT_NEOS": "4",
            "ARM_EXPECT_CORES": str(cores),
        },
    )
    _check_arm(p, f"{case[0]} borrowed at 4,4,2")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("case, cores", _L1_INTERLEAVED_DM_NOC, ids=[c[0][0] for c in _L1_INTERLEAVED_DM_NOC])
def test_native_l1_interleaved_keeps_the_noc_path(case, cores):
    """An indivisible tile count per cluster, a partial grid, or an operand outside L1 must keep the NoC path.

    A borrowed ring must divide by the compute count, and the L1-interleaved path has no tail rings, so a
    borrowed ring of 5, 2 or 1 tiles at C = 4 dies on the DFB host assertion. On a grid that is not the
    bank cores, the pages of the banks outside it would never be computed. A DRAM operand has no slice in
    the cluster's L1 to borrow.
    """
    p = _run_shard_arm(
        (4, 4, 2),
        [case],
        expect="qsr",
        timeout=900,
        knobs={
            "TT_METAL_DEVICE_PROFILER": "1",
            "ARM_EXPECT_DMS": "6",
            "ARM_EXPECT_NEOS": "4",
            "ARM_EXPECT_CORES": str(cores),
        },
    )
    _check_arm(p, f"{case[0]} on the NoC path at 4,4,2")
