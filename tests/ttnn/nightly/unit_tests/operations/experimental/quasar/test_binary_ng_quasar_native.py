# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quasar-native binary_ng: proof that the native factory ran, and that it computed the predicted answer.

Routing is read from the inspector's kernels.yaml, which names the kernel SOURCES actually bound. That
is the only observable that separates the two factories at this stage -- they are a mechanical copy of
each other, so cycles, values and the active-core set are identical by construction. The native tests
compare bits with the bf16 oracle; the fallback test uses the shared _run helper (golden + PCC floor +
constant-output guard).

Configuration-gated on TTNN_QSR_NATIVE rather than parametrized: the program cache resolves the factory
from a cached index, so flipping the knob in-process would silently re-run the previous program. One
process per state -- see the run commands in debug/run_targeted_matrix.sh.

Every TTNN_QSR_* tuning knob left unset takes the rule's value (resolve_native_config in
binary_ng_utils.cpp): 4 compute threads; where the operands move over the NoC, a reader thread per compute
thread and the writer threads that the 6 DM cores leave, and one of each where they are borrowed; batches
of 8 tiles in the DM kernels, and in the compute where each compute thread owns one tile counter of each
ring; rings of 16 entries per thread. Above ring stride 1 a compute batch packs wrong until the pack path
applies the ring stride per tile, so each test predicts its output from its configuration: the tests in
this process from the invocation's knobs, the arms from their own. A test that predicts wrong output is a
strict expected failure, and it passes only on exactly the predicted wrong tiles.
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


# The rule's value of every knob left unset. Mirrors resolve_native_config in binary_ng_utils.cpp: a test whose
# prediction disagrees with the device is how a change on either side shows up.
_RULE_COMPUTE_THREADS = 4
_RULE_TILES_PER_CYCLE = 8
_USER_DM_CORES = 6

# The worker clusters of the simulator's 8x4 grid, which every test here assumes.
_CLUSTERS = 32


def _resolve(knobs, moves):
    """(R, C, W, compute batch) of a program run with `knobs`, a mapping of TTNN_QSR_* names to values.

    `moves` is True on the NoC path and False when every operand is borrowed. A borrowed program runs one
    reader and one writer thread whatever is set. An empty value counts as unset, as in native_tuning().
    """

    def knob(name, rule):
        value = knobs.get(name, "")
        return int(value) if value else rule

    c = knob("TTNN_QSR_COMPUTE_THREADS", _RULE_COMPUTE_THREADS)
    r = knob("TTNN_QSR_READER_THREADS", c) if moves else 1
    w = knob("TTNN_QSR_WRITER_THREADS", min(c, max(1, _USER_DM_CORES - r))) if moves else 1
    batch = knob("TTNN_QSR_TILES_PER_CYCLE", _RULE_TILES_PER_CYCLE if r <= c and w <= c else 1)
    return r, c, w, batch


def _pack_stride_wrong_tiles(tiles_per_cluster, knobs, moves):
    """The output tiles that the pack-stride defect leaves wrong.

    tiles_per_cluster lists each cluster's tile count. A compute batch of n tiles at output ring stride s =
    max(C, W) lands only ceil(n/s) of them. Each compute thread's share runs in full batches and one short
    batch, and thread t takes one extra tile while t < tiles % C. The tests compare the count tile for tile.
    """
    _, c, w, batch = _resolve(knobs, moves)
    stride = max(c, w)
    landed = 0
    for tiles in tiles_per_cluster:
        for thread in range(c):
            share = tiles // c + (thread < tiles % c)
            full, tail = divmod(share, batch)
            landed += full * -(-batch // stride) - (-tail // stride)
    return sum(tiles_per_cluster) - landed


def _noc_split(tiles, clusters=_CLUSTERS):
    """Each cluster's tile count on the NoC path, which split_work_to_cores spreads over the worker clusters."""
    per_cluster, extra = divmod(tiles, clusters)
    return [per_cluster + 1] * extra + [per_cluster] * (clusters - extra)


def _wrong_tiles(got, golden):
    """The 32x32 tiles of a 2D output in which any element differs from the oracle, compared as int16."""
    differ = got.contiguous().view(torch.int16) != golden.contiguous().view(torch.int16)
    height, width = differ.shape[-2:]
    return int(differ.reshape(-1, height // 32, 32, width // 32, 32).any(dim=4).any(dim=2).sum())


def _threads(r, c, w):
    return {"TTNN_QSR_READER_THREADS": str(r), "TTNN_QSR_COMPUTE_THREADS": str(c), "TTNN_QSR_WRITER_THREADS": str(w)}


# Compute batching off. Every other knob keeps the rule's value, so this is the rule's program short of the one
# defect, and its output is exact.
_COMPUTE_BATCH_1 = {"TTNN_QSR_TILES_PER_CYCLE": "1"}

# One tile per DM transfer and per compute batch, in rings of two entries per thread, so that a thread's ring
# wraps every two tiles.
_PER_TILE = {"TTNN_QSR_DM_BATCH": "1", "TTNN_QSR_ENTRIES_PER_THREAD": "2", "TTNN_QSR_TILES_PER_CYCLE": "1"}

# The invocation's knobs, which every test in this process runs with.
_PARENT_KNOBS = {k: v for k, v in os.environ.items() if k.startswith("TTNN_QSR_")}


class _WrongOutput(Exception):
    """An op ran to completion on the expected factory, and its output differs from the bf16 oracle."""


# Above output ring stride 1 compute batching runs but loses most of each batch, because the pack path spaces
# the tiles of a batch by one entry rather than by the ring stride. strict=True: once the pack path is fixed
# these tests pass, and the run fails until the predictions change.
_PACK_STRIDE_REASON = "compute batching above ring stride 1: the pack path does not apply the ring stride per tile"
_PACK_STRIDE_XFAIL = pytest.mark.xfail(raises=_WrongOutput, strict=True, reason=_PACK_STRIDE_REASON)


def _expectation_marks(expected_wrong):
    """A test that predicts wrong output is a strict expected failure."""
    return [_PACK_STRIDE_XFAIL] if expected_wrong > 0 else []


def _matches_prediction(got, golden, expected_wrong, what):
    """Assert that exactly the predicted number of output tiles differ from the oracle; 0 means exact.

    Returns True when the output is wrong as predicted.
    """
    wrong = _wrong_tiles(got, golden)
    total = golden.numel() // (32 * 32)
    assert wrong == expected_wrong, f"{what}: {wrong} of {total} tiles differ, the prediction is {expected_wrong}"
    return wrong > 0


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
    # The shared helper, so the fallback test inherits the same golden, PCC floor and constant-output guard
    # as the rest of the suite rather than rolling its own weaker check.
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


# The benchmark shape under the invocation's knobs: 40 tiles per cluster on the NoC path.
_BENCHMARK_WRONG = _pack_stride_wrong_tiles(_noc_split(32 * 40), _PARENT_KNOBS, moves=True)


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.xfail(_BENCHMARK_WRONG > 0, raises=_WrongOutput, strict=True, reason=_PACK_STRIDE_REASON)
def test_native_output_matches_the_prediction(device):
    """bf16 add on this slice is EXACT, so compare bits rather than PCC.

    fp32_dest_acc_en is false for bf16 and HiFi4 is fidelity-inert for FPU add, so the device result
    equals torch's bf16 add bit-for-bit. The int16 view is required, not cosmetic: torch.equal on
    floats reports False for NaN vs itself. randn produces no NaN/Inf, so that path stays unexercised.
    Under the rule's compute batch of 8 at ring stride 4, exactly the predicted tiles are wrong.
    """
    out, ta, tb = _add_returning_operands(device)
    golden = (ttnn.to_torch(ta).float() + ttnn.to_torch(tb).float()).to(torch.bfloat16)
    if _matches_prediction(ttnn.to_torch(out), golden, _BENCHMARK_WRONG, "32x40 tiles"):
        raise _WrongOutput("32x40 tiles: wrong as predicted")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
def test_native_factory_is_engaged(device):
    """The benchmark add binds the three kernels_qsr sources and no kernels_dfb source.

    Routing only, with no value check: under the rule's compute batching the predicted tiles are wrong,
    which test_native_output_matches_the_prediction checks on the same shape.
    """
    _add_returning_operands(device)
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
# Quasar cluster, so split_work_to_cores distributes over clusters. Under the per-tile settings ring depth
# is 2 * max(R,C), i.e. 2 at 1,1,1 and 8 at R=C=4, so which Tc first makes the tile stream WRAP depends on
# the arm:
#
#   Tc=1  empty threads on every axis      Tc=5  tail on every axis, no empties
#   Tc=2  empties on R/C while W is even   Tc=6  tail on R/C while W is even
#   Tc=3  a single empty thread            Tc=9  wraps the ring on every arm
#   Tc=4  even everywhere (the control)    Tc=41 tail at depth, wraps repeatedly
#
# The "empty thread" readings hold only on a multi-thread arm -- at 1,1,1 every count is even by
# definition. _NATIVE_ARMS is what makes them real; see test_native_uneven_tile_counts_multithread.
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
# usable: each DFB requires max(p,c) % min(p,c) == 0, which the gate checks, and R and W stay at or below
# C, because some craq-sim builds corrupt the output when the DM side outnumbers the Tensix side. A tuple
# the gate refuses falls back to the single-threaded factory and then passes while testing nothing --
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

# Run in a fresh process per arm: native_tuning() reads the knobs ONCE per process into a static, and
# the program cache resolves the factory from a cached index, so setting the env vars mid-process
# changes nothing. Same constraint that makes this module gate on TTNN_QSR_NATIVE rather than
# parametrize it. Routing is read after close_device() so the inspector has certainly flushed.
#
# The child is a fixed script. Its parameters arrive as data in the environment (ARM_TILE_COUNTS, JSON),
# never spliced into the source.
_ARM_SRC = """
import json, os, pathlib, sys
import torch, ttnn

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
        differ = got.contiguous().view(torch.int16) != golden.contiguous().view(torch.int16)
        if differ.any():
            h, w = differ.shape[-2:]
            wrong = int(differ.reshape(-1, h // 32, 32, w // 32, 32).any(dim=4).any(dim=2).sum())
            mismatches.append("{} tiles: {} of {} tiles differ".format(tiles, wrong, differ.numel() // 1024))
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
    print("FAIL")
    for f in failures + mismatches:
        print("  " + f)
    # 2 means only the output is wrong: every op ran and routed as expected.
    sys.exit(1 if failures else 2)
print("OK over {} shapes".format(len(TILE_COUNTS)))
"""


def _run_arm(tile_counts, knobs, timeout):
    """Run one configuration of _ARM_SRC in its own process.

    Every inherited TTNN_QSR_* variable is stripped first, so an arm runs exactly `knobs`, and the rule's
    value of every knob that `knobs` leaves unset, whatever the calling shell has set.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("TTNN_QSR_")}
    env.update({"TTNN_QSR_NATIVE": "1", "ARM_TILE_COUNTS": json.dumps(list(tile_counts)), **knobs})
    return subprocess.run([sys.executable, "-c", _ARM_SRC], env=env, capture_output=True, text=True, timeout=timeout)


def _check_arm(p, what, expect_wrong=None):
    """Pass on exit 0, raise _WrongOutput on the predicted wrong output, and fail the test on anything else.

    The arm scripts exit 2 only when every op completed and routed as expected and the output is wrong.
    An expected-failure marker restricted to _WrongOutput therefore cannot hide a crash, a hang, a
    refusal or a fallback. expect_wrong maps the label of each shape the prediction gets wrong to the
    number of output tiles the pack-stride defect leaves wrong. Only exactly those counts raise
    _WrongOutput, so a new defect, or a shape that is wrong where the prediction is 0, fails the test.
    """
    expect_wrong = expect_wrong or {}
    if p.returncode == 2:
        measured = {
            m.group(1): int(m.group(2))
            for m in re.finditer(r"^\s+(.+): (\d+) of \d+ tiles differ$", p.stdout, re.MULTILINE)
        }
        assert measured == expect_wrong, (
            f"{what}: the wrong tiles are not the predicted ones (label: count), "
            f"measured {measured}, predicted {expect_wrong}\n{p.stdout}"
        )
        raise _WrongOutput(f"{what}: output differs from the bf16 oracle as predicted\n{p.stdout}")
    assert p.returncode == 0, f"{what} failed:\n{p.stdout}\n{p.stderr[-2000:]}"


def _add_one_row(device, tiles, seed):
    """One (1 x tiles) add. Returns (output, bf16 oracle).

    A 1-tile-tall shape is a legitimate one-dimensional sweep here because the native reader walks
    page = start_tile_id + k -- a pure linear index -- so behaviour depends on the tile COUNT alone and
    not on how it factors into height x width.

    The golden is rounded to bf16, not left in fp32: the device packs bf16, and the sum of two bf16
    values needs 9 mantissa bits where bf16 has 8, so an fp32 comparison reports ~54% mismatch on
    random data purely from rounding.
    """
    out, ta, tb = _add_returning_operands(device, h_tiles=1, w_tiles=tiles, seed=seed)
    golden = (ttnn.to_torch(ta).float() + ttnn.to_torch(tb).float()).to(torch.bfloat16)
    return ttnn.to_torch(out), golden


def _parent_prediction(tiles):
    return _pack_stride_wrong_tiles(_noc_split(tiles), _PARENT_KNOBS, moves=True)


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize(
    "tiles", [pytest.param(t, marks=_expectation_marks(_parent_prediction(t))) for t in _RAGGED_TILE_COUNTS], ids=str
)
def test_native_uneven_tile_counts_match_the_prediction(device, tiles):
    before = set(bound_kernel_sources())
    got, golden = _add_one_row(device, tiles, seed=tiles)
    wrong_as_predicted = _matches_prediction(got, golden, _parent_prediction(tiles), f"{tiles} tiles")

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
    if wrong_as_predicted:
        raise _WrongOutput(f"{tiles} tiles: wrong as predicted")


# Even, ragged across clusters, and ragged across threads. Under the rule each thread gets at most one tile,
# a batch of one, so the output is exact and a wrong tile can only come from the program cache.
_CACHE_SEQUENCE = (128, 33, 128, 32, 128)
_CACHE_WRONG = max(_parent_prediction(t) for t in _CACHE_SEQUENCE)


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.xfail(_CACHE_WRONG > 0, raises=_WrongOutput, strict=True, reason=_PACK_STRIDE_REASON)
def test_native_even_and_uneven_share_a_process(device):
    """Even and ragged shapes must not poison each other through the program cache.

    The op has no compute_program_hash override, so the framework hashes the tensor specs -- different
    shapes therefore take different cache entries. This asserts that empirically rather than trusting
    it, and in the order most likely to break: warm the cache on an even shape, run a ragged one, then
    return to the even shape and re-check it. Each visit must give that shape's predicted output.
    """
    wrong = [
        _matches_prediction(
            *_add_one_row(device, tiles, seed=1), _parent_prediction(tiles), f"{tiles} tiles, visit {i}"
        )
        for i, tiles in enumerate(_CACHE_SEQUENCE)
    ]
    if any(wrong):
        raise _WrongOutput("wrong as predicted")


# How the DM kernels and the compute move tiles in an arm. per-tile runs one tile per transfer and per batch
# in two-entry rings, so the rings wrap at the low tile counts. batched keeps the rule's DM batch of 8 in
# 16-entry rings, with the compute batch at 1 so the output stays exact at every thread count.
_BATCHING = {"per-tile": _PER_TILE, "batched": _COMPUTE_BATCH_1}


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("batching", list(_BATCHING))
@pytest.mark.parametrize("rcw", _NATIVE_ARMS, ids=lambda t: "R{}C{}W{}".format(*t))
def test_native_uneven_tile_counts_multithread(rcw, batching):
    """Uneven tile counts only matter above one thread, so run the ragged sweep on real arms.

    The tests above run at whatever knobs the invocation carries. This one sets the thread counts
    explicitly, so an empty thread and a ragged tail are reached at every arm, with and without DM
    batching.

    Requests no `device` fixture on purpose: each arm opens its own device inside its own process.
    """
    p = _run_arm(_RAGGED_TILE_COUNTS, knobs={**_threads(*rcw), **_BATCHING[batching]}, timeout=3600)
    _check_arm(p, f"arm R={rcw[0]} C={rcw[1]} W={rcw[2]} {batching}")


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
    """The rule's DM batch of 8 on a role that owns more than one tile counter must stay bit-exact.

    The rule's depth of 16 is the smallest legal depth for it: a batch occupies n consecutive slots and the
    ring wraps only afterwards, so the factory demands entries_per_thread >= 2n and entries_per_thread % n
    == 0. The compute batch is 1, so the output stays exact at ring stride 4.

    One process per arm, as above. The timeout is the failure detector for a broken pairing: a counter
    asked for more tiles than it receives hangs rather than corrupts.
    """
    p = _run_arm(_DM_BATCH_TILE_COUNTS, knobs={**_threads(*rcw), **_COMPUTE_BATCH_1}, timeout=600)
    _check_arm(p, f"dm_batch=8 arm R={rcw[0]} C={rcw[1]} W={rcw[2]}")


# --- Compute batching --------------------------------------------------------------------------------
#
# The rule's compute batch of 8 runs eight tiles per tile_regs_acquire. The pack path spaces a batch by
# one entry rather than by the ring stride, so the output is right only at output ring stride 1 -- C = W
# = 1 -- until the pack path applies the stride per tile. The factory runs any stride; the tests above
# stride 1 expect the wrong output. Per-cluster tile counts on the 32-cluster grid: 1 runs a single
# short batch and no full one; 9 runs one full batch of 8 then a tail of 1; 64 runs eight full batches,
# wrapping the 16-deep ring three times, and no tail; 65 adds a tail of 1 that starts exactly on a wrap
# boundary.
_TILES_PER_CYCLE_TILE_COUNTS = [32, 288, 2048, 2080]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("dm_batch", [1, 8], ids=["compute-batch-only", "both-batches"])
def test_native_tiles_per_cycle_is_bit_exact(dm_batch):
    """The rule's compute batch at 1,1,1 must stay bit-exact, alone and together with its DM batch.

    The compute kernel splits a thread's share into full chunks of num_tiles_per_cycle and one remainder
    chunk; a truncating divide there once dropped the tail and hung. The tile counts above reach every
    branch of that split. both-batches measured 1.81x the throughput of no batching at 1,1,1 on craq-sim
    (marginal cycles per tile).
    """
    knobs = {**_threads(1, 1, 1), **({"TTNN_QSR_DM_BATCH": "1"} if dm_batch == 1 else {})}
    p = _run_arm(_TILES_PER_CYCLE_TILE_COUNTS, knobs=knobs, timeout=600)
    _check_arm(p, f"tiles_per_cycle=8 dm_batch={dm_batch}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@_PACK_STRIDE_XFAIL
def test_native_tiles_per_cycle_above_stride_one():
    """With every knob unset the NoC path runs the rule's 4,4,2 and both batches of 8, and its output is
    wrong until the pack path is fixed.

    The ring stride is 4 on every DFB. 288 tiles give each compute thread two or three tiles, one short
    batch; 2080 give it sixteen or seventeen, two full batches of 8 and a tail. A refusal, a hang, a
    fallback or wrong output without the pack defect's signature fails this test instead of passing as
    an expected failure, and so does an exact output: the rule then does not batch.
    """
    tile_counts = [288, 2080]
    p = _run_arm(tile_counts, knobs={}, timeout=900)
    expect_wrong = {f"{t} tiles": _pack_stride_wrong_tiles(_noc_split(t), {}, moves=True) for t in tile_counts}
    _check_arm(p, "the rule on the NoC path", expect_wrong=expect_wrong)


# Each guard, the knob values that trip it, and the text its TT_FATAL carries. A batch needs a ring at least
# twice as deep (double buffering) and a depth that is a multiple of the batch (a batch occupies consecutive
# slots and the ring wraps only after it). A compute batch also needs each compute thread to own one tile
# counter of each ring, so R <= C and W <= C. The reader and writer threads must fit the 6 user DM cores.
# Each value is named with its source, the knob or the rule. The compute batch is checked first, so the
# DM-batch arms set it to 1. Compute batching above ring stride 1 is not refused;
# test_native_tiles_per_cycle_above_stride_one checks what it produces.
_GUARD_ARMS = [
    (
        "tiles_per_cycle-depth",
        {**_threads(1, 1, 1), "TTNN_QSR_TILES_PER_CYCLE": "8", "TTNN_QSR_ENTRIES_PER_THREAD": "8"},
        "tiles_per_cycle=8 (TTNN_QSR_TILES_PER_CYCLE) needs entries_per_thread >= 16 for double buffering, "
        "got entries_per_thread=8 (TTNN_QSR_ENTRIES_PER_THREAD)",
    ),
    (
        "tiles_per_cycle-divides",
        {**_threads(1, 1, 1), "TTNN_QSR_TILES_PER_CYCLE": "8", "TTNN_QSR_ENTRIES_PER_THREAD": "20"},
        "tiles_per_cycle=8 (TTNN_QSR_TILES_PER_CYCLE) must divide entries_per_thread=20 (TTNN_QSR_ENTRIES_PER_THREAD)",
    ),
    (
        "dm_batch-depth",
        {**_threads(1, 1, 1), "TTNN_QSR_DM_BATCH": "8", "TTNN_QSR_ENTRIES_PER_THREAD": "8", **_COMPUTE_BATCH_1},
        "dm_batch=8 (TTNN_QSR_DM_BATCH) needs entries_per_thread >= 16 for double buffering, "
        "got entries_per_thread=8 (TTNN_QSR_ENTRIES_PER_THREAD)",
    ),
    (
        "dm_batch-divides",
        {**_threads(1, 1, 1), "TTNN_QSR_DM_BATCH": "8", "TTNN_QSR_ENTRIES_PER_THREAD": "20", **_COMPUTE_BATCH_1},
        "dm_batch=8 (TTNN_QSR_DM_BATCH) must divide entries_per_thread=20 (TTNN_QSR_ENTRIES_PER_THREAD)",
    ),
    (
        "rule-batch-depth",
        {**_threads(1, 1, 1), "TTNN_QSR_ENTRIES_PER_THREAD": "8"},
        "tiles_per_cycle=8 (rule) needs entries_per_thread >= 16 for double buffering, "
        "got entries_per_thread=8 (TTNN_QSR_ENTRIES_PER_THREAD)",
    ),
    (
        "batch-beside-more-readers",
        {"TTNN_QSR_READER_THREADS": "2", "TTNN_QSR_COMPUTE_THREADS": "1", "TTNN_QSR_TILES_PER_CYCLE": "8"},
        "tiles_per_cycle=8 (TTNN_QSR_TILES_PER_CYCLE) needs R <= C and W <= C, so that each compute thread owns "
        "one tile counter of each ring, got R=2 (TTNN_QSR_READER_THREADS) C=1 (TTNN_QSR_COMPUTE_THREADS) W=1 (rule)",
    ),
    (
        "dm-cores",
        {"TTNN_QSR_WRITER_THREADS": "4"},
        "R + W must be <= 6 (Quasar has 6 user DM cores), got R=4 (rule) + W=4 (TTNN_QSR_WRITER_THREADS)",
    ),
    (
        # A reader count that leaves no DM core still gets one writer thread, so this check refuses it. A
        # writer count of 0 would pass the check and make the gate divide by zero.
        "readers-fill-the-dm-cores",
        {"TTNN_QSR_READER_THREADS": "6"},
        "R + W must be <= 6 (Quasar has 6 user DM cores), got R=6 (TTNN_QSR_READER_THREADS) + W=1 (rule)",
    ),
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("arm", _GUARD_ARMS, ids=[a[0] for a in _GUARD_ARMS])
def test_native_knob_guards_refuse(arm):
    """A knob setting the factory cannot run must fail with its own message, not run and hang.

    The message is asserted, not only a non-zero exit: an arm that died for another reason -- no
    simulator, a build break -- would otherwise pass as a refusal.
    """
    _, knobs, message = arm
    p = _run_arm([32], knobs=knobs, timeout=300)
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

# The bit-exact configurations of the borrowed path. There the factory runs one reader and one writer
# thread whatever is set, since they copy only the tail tiles. 1,1,1 keeps the rule's compute batch of 8
# at ring stride 1. rule-n1 is the rule's 1,4,1, which puts the reader and the writer on four counters
# each, with the compute batch at 1; 4,4,2-n1 checks that set reader and writer counts map to that same
# program. So the DM side never outnumbers the Tensix side here, a case that some craq-sim builds corrupt.
_BORROWED_CONFIGS = {
    "R1C1W1": _threads(1, 1, 1),
    "rule-n1": _COMPUTE_BATCH_1,
    "R4C4W2-n1": {**_threads(4, 4, 2), **_COMPUTE_BATCH_1},
}
# Two compute threads, on two-entry tail rings.
_BORROWED_CONFIGS_C2 = {**_BORROWED_CONFIGS, "R1C2W1-n1": {**_threads(1, 2, 1), **_COMPUTE_BATCH_1}}


def _shard_tiles(entry):
    """Tiles in one shard of a _SHARD_ARM_SRC entry."""
    height, width = entry[2]
    return (height // 32) * (width // 32)


def _tensor_tiles(entry):
    """Tiles in the tensor of a _SHARD_ARM_SRC entry."""
    height, width = entry[4]
    return (height // 32) * (width // 32)


def _shard_cores(entry):
    """Cores in the inclusive core range of a _SHARD_ARM_SRC entry."""
    x0, y0, x1, y1 = entry[3]
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def _expect_wrong(cases, knobs):
    """The prediction of each case that `knobs` leaves wrong: {name: wrong output tiles}.

    A case is (entry, moves, tiles per cluster): moves is True on the NoC path and False when every operand
    is borrowed. The tile counts must cover the whole output, with no pad slot;
    test_native_prediction_tables_cover_their_cases checks that.
    """
    predicted = {entry[0]: _pack_stride_wrong_tiles(split, knobs, moves) for entry, moves, split in cases}
    return {name: wrong for name, wrong in predicted.items() if wrong > 0}


def _config_params(configs, cases):
    """One pytest param per configuration, a strict expected failure where the prediction is wrong output."""
    return [
        pytest.param(name, knobs, marks=_expectation_marks(sum(_expect_wrong(cases, knobs).values())), id=name)
        for name, knobs in configs.items()
    ]


# Same shape as _ARM_SRC: a fixed child script, parameters as JSON in the environment. ARM_EXPECT names
# the factory every op in the child must have bound: "qsr" for native, "dfb" for the fallback. The
# strategy "l1" or "dram" places a tensor interleaved, and then the shard shape and range are None. The
# strategies "dram_height", "dram_width" and "dram_block" shard it over all DRAM banks, and then the range
# is None. A shard entry may carry a sixth element, a dict of options: "out" gives the output its own placement
# (otherwise it takes the inputs'), "b" does the same for b, "in_place" adds b into a with add_, and
# "grid" (x0, y0, x1, y1) runs the op on that sub-core grid. Every tensor stays alive to the end, so an
# entry that repeats an earlier spec lands at new addresses and must hit the program cache. With
# ARM_EXPECT_NEOS or ARM_EXPECT_DMS (and the device profiler on), every core must show that many Neos
# running the compute, or DM cores running a reader or writer thread; ARM_EXPECT_CORES names how many
# cores must report.
_SHARD_ARM_SRC = """
import collections, csv, json, os, pathlib, sys
import torch, ttnn

SHARDS = json.loads(os.environ["ARM_SHARDS"])
EXPECT = os.environ["ARM_EXPECT"]
EXPECT_NEOS = int(os.environ.get("ARM_EXPECT_NEOS", "0"))
EXPECT_DMS = int(os.environ.get("ARM_EXPECT_DMS", "0"))
EXPECT_CORES = int(os.environ.get("ARM_EXPECT_CORES", "0"))
STRATEGY = {"height": ttnn.ShardStrategy.HEIGHT, "block": ttnn.ShardStrategy.BLOCK,
            "width": ttnn.ShardStrategy.WIDTH}
INTERLEAVED = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}
DRAM_SHARDED = {"dram_height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "dram_width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                "dram_block": ttnn.TensorMemoryLayout.BLOCK_SHARDED}
root = pathlib.Path(os.environ.get("TT_METAL_LOGS_PATH") or os.environ.get("TT_METAL_HOME") or os.getcwd())
profile_csv = root / "generated" / "profiler" / ".logs" / "profile_log_device.csv"
if (EXPECT_NEOS or EXPECT_DMS) and profile_csv.exists():
    profile_csv.unlink()


def memory_config(strategy, shard_shape, rng):
    if strategy in INTERLEAVED:
        return INTERLEAVED[strategy]
    if strategy in DRAM_SHARDED:
        # The shard grid of a DRAM tensor is in DRAM-bank coordinates, not Tensix cores.
        g = device.dram_grid_size()
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        spec = ttnn.ShardSpec(banks, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(DRAM_SHARDED[strategy], ttnn.BufferType.DRAM, spec)
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
        differ = got.contiguous().view(torch.int16) != golden.contiguous().view(torch.int16)
        if differ.any():
            h, w = differ.shape[-2:]
            wrong = int(differ.reshape(-1, h // 32, 32, w // 32, 32).any(dim=4).any(dim=2).sum())
            mismatches.append("{}: {} of {} tiles differ".format(name, wrong, differ.numel() // 1024))
finally:
    ttnn.close_device(device)

sources = [ln.split("source:", 1)[1].strip()
           for ln in (root / "generated" / "inspector" / "kernels.yaml").read_text().splitlines()
           if "source:" in ln]
# Every binary_ng kernel tree counts: the descriptor binds kernels/ and kernels_ng/, which neither the
# native nor the Metal 2.0 factory uses.
binary_ng = [s for s in sources if "/binary_ng/device/kernels" in s]
qsr = [s for s in binary_ng if "kernels_qsr/" in s]
dfb = [s for s in binary_ng if "kernels_dfb/" in s]
other = len(binary_ng) - len(qsr) - len(dfb)
if EXPECT == "qsr" and (dfb or other or not qsr):
    failures.append("expected every op on kernels_qsr, bound qsr={} dfb={} other={}".format(len(qsr), len(dfb), other))
if EXPECT == "dfb" and (qsr or other or not dfb):
    failures.append("expected every op on kernels_dfb, bound qsr={} dfb={} other={}".format(len(qsr), len(dfb), other))
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
    print("FAIL")
    for f in failures + mismatches:
        print("  " + f)
    # 2 means only the output is wrong: every op ran and routed as expected.
    sys.exit(1 if failures else 2)
print("OK over {} shards".format(len(SHARDS)))
"""


def _run_shard_arm(shards, expect, timeout, knobs=None):
    """Run one configuration of _SHARD_ARM_SRC in its own process.

    Inherited TTNN_QSR_* variables are stripped, as in _run_arm, so the arm runs exactly `knobs` and the
    rule's value of every knob that `knobs` leaves unset. With the device profiler on, inherited DPRINT and
    streaming-profiler variables are stripped too, because the runtime refuses to start with either of
    them next to it. A build without Tracy skips the test.
    """
    profiling = (knobs or {}).get("TT_METAL_DEVICE_PROFILER") == "1"
    stripped = ("TTNN_QSR_", "TT_METAL_DPRINT_", "TT_METAL_STREAMING_PROFILER") if profiling else ("TTNN_QSR_",)
    env = {k: v for k, v in os.environ.items() if not k.startswith(stripped)}
    env.update(
        {
            "TTNN_QSR_NATIVE": "1",
            "ARM_SHARDS": json.dumps(list(shards)),
            "ARM_EXPECT": expect,
            **(knobs or {}),
        }
    )
    p = subprocess.run([sys.executable, "-c", _SHARD_ARM_SRC], env=env, capture_output=True, text=True, timeout=timeout)
    if profiling and "requires a Tracy-enabled build" in p.stdout + p.stderr:
        pytest.skip("the device profiler needs a Tracy-enabled build of tt-metal")
    return p


# Every borrowed shard above is even: each core computes one whole shard, and 4 divides it. So the rule's own
# configuration runs them too, and its compute batch of 8 at ring stride 4 leaves the predicted tiles wrong.
_BORROWED_SHARD_CASES = [(entry, False, [_shard_tiles(entry)] * _shard_cores(entry)) for entry in _BORROWED_SHARDS]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("name, knobs", _config_params({**_BORROWED_CONFIGS, "rule": {}}, _BORROWED_SHARD_CASES))
def test_native_borrowed_shards_match_the_prediction(name, knobs):
    """Height, block and width shards must run native and give the predicted output in every configuration.

    A wrong credit count hangs into the timeout, credits on the wrong counter fail the bit comparison,
    and a cached program that keeps the first call's shard addresses fails the repeat.
    """
    p = _run_shard_arm(_BORROWED_SHARDS, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"borrowed shards, {name}", expect_wrong=_expect_wrong(_BORROWED_SHARD_CASES, knobs))


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("name, knobs", list(_BORROWED_CONFIGS.items()), ids=list(_BORROWED_CONFIGS))
def test_native_borrowed_uneven_shards_are_bit_exact(name, knobs):
    """Shards whose last clusters hold fewer real tiles must run native and bit-exact in every configuration.

    Every role processes the full shard's tile count. Roles that disagree on a boundary cluster's count
    hang into the timeout.
    """
    p = _run_shard_arm(_UNEVEN_SHARDS, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"uneven shards, {name}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("name, knobs", list(_BORROWED_CONFIGS_C2.items()), ids=list(_BORROWED_CONFIGS_C2))
def test_native_borrowed_indivisible_shards_stay_native(name, knobs):
    """A shard whose tile count does not divide by the compute count must stay native and bit-exact.

    A borrowed ring sized to the whole shard dies on the DFB host assertion, and a tail copied from the
    wrong address or packed into the wrong ring fails the bit comparison. R1C2W1-n1 runs two-entry tail
    rings. At 1,1,1 every shard divides, and 9 and 10 tiles leave a remainder after one compute batch of 8.
    """
    p = _run_shard_arm(_SMALL_SHARDS, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"indivisible shards, {name}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("shard", _FOUR_NEO_SHARDS, ids=[s[0] for s in _FOUR_NEO_SHARDS])
def test_native_borrowed_indivisible_shards_use_every_compute_thread(shard):
    """With no thread knob set, a borrowed shard must run on all four Neos, however few tiles it has.

    The rule's compute count is 4. The borrowed part keeps the largest multiple of 4, which is none below 4
    tiles, and the leftover tiles go through owned rings of one entry per compute thread. A factory that
    drops to fewer compute threads instead fails the Neo count, which the child reads from the device
    profiler; wrong output or a fallback fails as usual.
    """
    p = _run_shard_arm(
        [shard],
        expect="qsr",
        timeout=900,
        knobs={**_COMPUTE_BATCH_1, "TT_METAL_DEVICE_PROFILER": "1", "ARM_EXPECT_NEOS": "4"},
    )
    _check_arm(p, f"{shard[0]} on four Neos")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
def test_native_borrowed_mismatched_shard_specs_fall_back():
    """Inputs and output on one grid with different shard specs must not be borrowed, and must be right.

    A predicate that compares grids alone borrows the shards in place and fails the bit comparison. The
    fallback reshards over the NoC, so both checks hold only when the predicate compares the full spec.
    """
    p = _run_shard_arm(_MISMATCHED_SHARDS, expect="dfb", timeout=900)
    _check_arm(p, "mismatched shard specs")


# 64 tiles per core: at the rule's 1,4,1 each compute thread takes 16, two full batches of 8 at ring stride 4.
_BATCHED_BORROWED_SHARDS = [
    ("block64", "block", [8 * 32, 8 * 32], (0, 0, 1, 1), (2 * 8 * 32, 2 * 8 * 32)),
]
_BATCHED_BORROWED_CASES = [
    (entry, False, [_shard_tiles(entry)] * _shard_cores(entry)) for entry in _BATCHED_BORROWED_SHARDS
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@_PACK_STRIDE_XFAIL
def test_native_borrowed_tiles_per_cycle_above_stride_one():
    """With every knob unset a borrowed shard runs the rule's 1,4,1 at a compute batch of 8, and its output
    is wrong until the pack path is fixed.

    This is the test that becomes a plain pass when that fix lands. An exact output fails it too: the rule
    then does not batch.
    """
    p = _run_shard_arm(_BATCHED_BORROWED_SHARDS, expect="qsr", timeout=900, knobs={})
    _check_arm(p, "the rule on a borrowed shard", expect_wrong=_expect_wrong(_BATCHED_BORROWED_CASES, {}))


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

# Where the rule runs each L1-interleaved case, and its tile count per cluster. At C = 4 a slice is borrowed
# when 4 divides its tiles per bank, else every operand moves over the NoC. The pad cases are left out: a
# pad slot takes a batch position but is not in the output.
_L1_INTERLEAVED_RULE_SPLITS = {
    "l1_128": (False, [4] * 32),
    "l1_256": (False, [8] * 32),
    "l1_128_again": (False, [4] * 32),
    "l1_128_in_place": (False, [4] * 32),
    "l1_160": (True, _noc_split(160)),
    "l1_64": (True, _noc_split(64)),
    "l1_8_idle": (True, _noc_split(8)),
    "l1_128_sub_grid": (True, _noc_split(128, clusters=16)),
    "l1_b_in_dram": (True, _noc_split(128)),
    "l1_out_in_dram": (True, _noc_split(128)),
    "dram_a": (True, _noc_split(128)),
}
_L1_INTERLEAVED_RULE_CASES = [
    (entry, *_L1_INTERLEAVED_RULE_SPLITS[entry[0]])
    for entry in _L1_INTERLEAVED + _L1_INTERLEAVED_NOC
    if entry[0] in _L1_INTERLEAVED_RULE_SPLITS
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("name, knobs", list(_BORROWED_CONFIGS_C2.items()), ids=list(_BORROWED_CONFIGS_C2))
def test_native_l1_interleaved_is_bit_exact(name, knobs):
    """L1-interleaved operands must run native and bit-exact in every configuration, borrowed or not.

    Credits on the wrong counter or a slice read at the wrong offset fail the bit comparison, and a cached
    program that keeps the first call's slice addresses fails the repeat.
    """
    p = _run_shard_arm(_L1_INTERLEAVED + _L1_INTERLEAVED_NOC, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"L1-interleaved, {name}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("name, knobs", _config_params({"rule": {}}, _L1_INTERLEAVED_RULE_CASES))
def test_native_l1_interleaved_matches_the_rule_prediction(name, knobs):
    """With every knob unset, L1-interleaved operands must give the rule's predicted output.

    A borrowed slice of 4 tiles gives each compute thread a batch of one, which packs right; 8 tiles give
    it a batch of two, which loses one tile at ring stride 4. The NoC cases split their tiles over the
    worker clusters as usual.
    """
    p = _run_shard_arm([case[0] for case in _L1_INTERLEAVED_RULE_CASES], expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"L1-interleaved, {name}", expect_wrong=_expect_wrong(_L1_INTERLEAVED_RULE_CASES, knobs))


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("threads", ["rule", "R4C4W2"])
@pytest.mark.parametrize("case, cores", _L1_INTERLEAVED_DM_BORROWED, ids=[c[0][0] for c in _L1_INTERLEAVED_DM_BORROWED])
def test_native_l1_interleaved_borrows_in_place(case, cores, threads):
    """L1-interleaved operands whose tile count per cluster divides by 4 must be borrowed, and run 1,4,1, with
    no thread knob set or with 4,4,2 set.

    The borrowed and the NoC path bind the same kernel sources, so the routing check cannot tell them
    apart. The thread counts can: a borrowed program runs one reader and one writer thread whatever is set,
    and the NoC path runs four and two. The child reads the DM cores and the Neos from the device profiler.
    """
    p = _run_shard_arm(
        [case],
        expect="qsr",
        timeout=900,
        knobs={
            **(_threads(4, 4, 2) if threads == "R4C4W2" else {}),
            **_COMPUTE_BATCH_1,
            "TT_METAL_DEVICE_PROFILER": "1",
            "ARM_EXPECT_DMS": "2",
            "ARM_EXPECT_NEOS": "4",
            "ARM_EXPECT_CORES": str(cores),
        },
    )
    _check_arm(p, f"{case[0]} borrowed at 1,4,1, threads {threads}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("case, cores", _L1_INTERLEAVED_DM_NOC, ids=[c[0][0] for c in _L1_INTERLEAVED_DM_NOC])
def test_native_l1_interleaved_keeps_the_noc_path(case, cores):
    """An indivisible tile count per cluster, a partial grid, or an operand outside L1 must keep the NoC path,
    and run the rule's 4,4,2 there.

    A borrowed ring must divide by the compute count, and the L1-interleaved path has no tail rings, so a
    borrowed ring of 5, 2 or 1 tiles at C = 4 dies on the DFB host assertion. On a grid that is not the
    bank cores, the pages of the banks outside it would never be computed. A DRAM operand has no slice in
    the cluster's L1 to borrow.
    """
    p = _run_shard_arm(
        [case],
        expect="qsr",
        timeout=900,
        knobs={
            **_COMPUTE_BATCH_1,
            "TT_METAL_DEVICE_PROFILER": "1",
            "ARM_EXPECT_DMS": "6",
            "ARM_EXPECT_NEOS": "4",
            "ARM_EXPECT_CORES": str(cores),
        },
    )
    _check_arm(p, f"{case[0]} on the NoC path at the rule's 4,4,2")


# DRAM-sharded operands, alone and beside interleaved ones. The shapes need at least two DRAM banks; the
# simulator has two, so each case makes two shards, and the uneven ones leave the second shard partial. A
# wrong page-to-bank map that a, b and the output share would still give c = a + b, so every layout also
# appears beside an operand with another placement. dram_width_1024 holds 32 tiles per cluster, enough for
# full DM batches of 8 at every arm. dram_width_again repeats dram_width while its tensors are alive, a
# cache hit that must rebind.
_DRAM_SHARDED = [
    ("dram_width", "dram_width", [4 * 32, 8 * 32], None, (4 * 32, 16 * 32)),
    ("dram_height", "dram_height", [32 * 32, 2 * 32], None, (64 * 32, 2 * 32)),
    ("dram_block", "dram_block", [4 * 32, 4 * 32], None, (4 * 32, 8 * 32)),
    ("dram_height_uneven", "dram_height", [4 * 32, 32], None, (7 * 32, 32)),
    ("dram_width_uneven", "dram_width", [32, 3 * 32], None, (32, 5 * 32)),
    ("dram_a_dram_b", "dram_width", [4 * 32, 8 * 32], None, (4 * 32, 16 * 32), {"b": ("dram", None, None)}),
    (
        "dram_a_l1_b_dram_out",
        "dram_height",
        [32 * 32, 2 * 32],
        None,
        (64 * 32, 2 * 32),
        {"b": ("l1", None, None), "out": ("dram", None, None)},
    ),
    ("dram_block_dram_out", "dram_block", [4 * 32, 4 * 32], None, (4 * 32, 8 * 32), {"out": ("dram", None, None)}),
    ("dram_height_uneven_dram_b", "dram_height", [4 * 32, 32], None, (7 * 32, 32), {"b": ("dram", None, None)}),
    ("dram_width_uneven_l1_b", "dram_width", [32, 3 * 32], None, (32, 5 * 32), {"b": ("l1", None, None)}),
    (
        "dram_height_dram_width_out",
        "dram_height",
        [2 * 32, 16 * 32],
        None,
        (4 * 32, 16 * 32),
        {"out": ("dram_width", [4 * 32, 8 * 32], None)},
    ),
    (
        "dram_in_dram_sharded_out",
        "dram",
        None,
        None,
        (64 * 32, 2 * 32),
        {"out": ("dram_height", [32 * 32, 2 * 32], None)},
    ),
    (
        "l1_in_dram_sharded_out",
        "l1",
        None,
        None,
        (4 * 32, 16 * 32),
        {"out": ("dram_width", [4 * 32, 8 * 32], None)},
    ),
    ("dram_width_1024", "dram_width", [32 * 32, 16 * 32], None, (32 * 32, 32 * 32), {"b": ("dram", None, None)}),
    ("dram_width_again", "dram_width", [4 * 32, 8 * 32], None, (4 * 32, 16 * 32)),
    ("dram_width_in_place", "dram_width", [4 * 32, 8 * 32], None, (4 * 32, 16 * 32), {"in_place": True}),
]
# Every DRAM-sharded op takes the NoC path, which splits its tiles over the 32 worker clusters. The per-tile
# configurations read one page per transfer; the others keep the rule's DM batch of 8. Under the rule only
# dram_width_1024 gives a compute thread more than one tile.
_DRAM_SHARDED_CONFIGS = {
    **{f"R{r}C{c}W{w}-per-tile": {**_threads(r, c, w), **_PER_TILE} for r, c, w in [(1, 1, 1), (1, 4, 1), (4, 4, 2)]},
    "R1C1W1": _threads(1, 1, 1),
    "R1C4W1-n1": {**_threads(1, 4, 1), **_COMPUTE_BATCH_1},
    "rule-n1": _COMPUTE_BATCH_1,
    "rule": {},
}
_DRAM_SHARDED_CASES = [(entry, True, _noc_split(_tensor_tiles(entry))) for entry in _DRAM_SHARDED]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("name, knobs", _config_params(_DRAM_SHARDED_CONFIGS, _DRAM_SHARDED_CASES))
def test_native_dram_sharded_matches_the_prediction(name, knobs):
    """DRAM-sharded operands must run native and give the predicted output, alone or beside interleaved
    operands.

    Their fallback is the descriptor, which does not run on Quasar, so a refusing gate fails the test. A
    page read from the wrong bank or offset fails the bit comparison, and a cached program that keeps the
    first call's addresses fails the repeat.
    """
    p = _run_shard_arm(_DRAM_SHARDED, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"DRAM-sharded, {name}", expect_wrong=_expect_wrong(_DRAM_SHARDED_CASES, knobs))


# One case per child, as above. Each case has at least 32 tiles, one per worker cluster. dram_width_in_place
# supplies the output tensor, which takes another path to the worker grid.
_DRAM_SHARDED_PLACEMENT = [
    s for s in _DRAM_SHARDED if s[0] in ("dram_width", "dram_in_dram_sharded_out", "dram_width_in_place")
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("case", _DRAM_SHARDED_PLACEMENT, ids=[c[0] for c in _DRAM_SHARDED_PLACEMENT])
def test_native_dram_sharded_spreads_over_the_worker_grid(case):
    """A DRAM-sharded op must run the rule's threads on every worker cluster, not on its shard grid.

    The shard grid of a DRAM tensor is in DRAM-bank coordinates. The NoC path splits the output tiles over
    the worker grid, so at the rule's 4,4,2 all 32 clusters run 6 DM cores and 4 Neos. A placement taken
    from the DRAM shard grid would run on one cluster per DRAM bank.
    """
    p = _run_shard_arm(
        [case],
        expect="qsr",
        timeout=900,
        knobs={
            **_COMPUTE_BATCH_1,
            "TT_METAL_DEVICE_PROFILER": "1",
            "ARM_EXPECT_DMS": "6",
            "ARM_EXPECT_NEOS": "4",
            "ARM_EXPECT_CORES": "32",
        },
    )
    _check_arm(p, f"{case[0]} on the worker grid at the rule's 4,4,2")


# --- The rule and its overrides ----------------------------------------------------------------------------

_DRAM_128 = ("dram_128", "dram", None, None, (4 * 32, 32 * 32))


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
def test_native_rule_follows_a_set_compute_count():
    """With only the compute count set, the rule must give the reader and writer counts that match it.

    Where the operands move over the NoC, the rule runs a reader thread and a writer thread per compute
    thread, within the 6 DM cores. So C = 2 runs 2 reader and 2 writer threads: 4 DM cores and 2 Neos on
    every cluster.
    """
    p = _run_shard_arm(
        [_DRAM_128],
        expect="qsr",
        timeout=900,
        knobs={
            "TTNN_QSR_COMPUTE_THREADS": "2",
            **_COMPUTE_BATCH_1,
            "TT_METAL_DEVICE_PROFILER": "1",
            "ARM_EXPECT_DMS": "4",
            "ARM_EXPECT_NEOS": "2",
            "ARM_EXPECT_CORES": "32",
        },
    )
    _check_arm(p, "C = 2 on the NoC path")


# Each case: the knobs, the values the settings line must name with their sources, and whether the output
# must be exact. C = 1 alone gives 1,1,1 by the rule. R = 2 beside C = 1 makes each compute thread own two
# tile counters of the input rings, so the rule's compute batch drops to 1. That case
# checks the line only: its output is exact only on craq-sim builds without the defect that corrupts a run
# whose DM side outnumbers its Tensix side. A hang still fails it.
_LOG_CASES = {
    "W1-n1": (
        {"TTNN_QSR_WRITER_THREADS": "1", **_COMPUTE_BATCH_1},
        [
            "R=4 (rule)",
            "C=4 (rule)",
            "W=1 (TTNN_QSR_WRITER_THREADS)",
            "entries_per_thread=16 (rule)",
            "dm_batch=8 (rule)",
            "tiles_per_cycle=1 (TTNN_QSR_TILES_PER_CYCLE)",
        ],
        True,
    ),
    "C1": (
        {"TTNN_QSR_COMPUTE_THREADS": "1"},
        ["R=1 (rule)", "C=1 (TTNN_QSR_COMPUTE_THREADS)", "W=1 (rule)", "tiles_per_cycle=8 (rule)"],
        True,
    ),
    "R2C1": (
        {"TTNN_QSR_READER_THREADS": "2", "TTNN_QSR_COMPUTE_THREADS": "1"},
        ["R=2 (TTNN_QSR_READER_THREADS)", "C=1 (TTNN_QSR_COMPUTE_THREADS)", "W=1 (rule)", "tiles_per_cycle=1 (rule)"],
        False,
    ),
}


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
@pytest.mark.parametrize("name", list(_LOG_CASES))
def test_native_tuning_log_names_each_source(name):
    """The line that reports the native settings must name the source of each value, the knob or the rule.

    An A/B log is then self-describing: a value the rule chose cannot be read as a value someone set. The
    line is an info message, so the arm sets the logger to info.
    """
    knobs, expected, exact = _LOG_CASES[name]
    p = _run_arm([32], knobs={**knobs, "TT_LOGGER_LEVEL": "info"}, timeout=300)
    if exact:
        _check_arm(p, f"the logged configuration, {name}")
    else:
        assert p.returncode in (0, 2), f"{name} failed:\n{p.stdout}\n{p.stderr[-2000:]}"
    log = p.stdout + p.stderr
    missing = [token for token in expected if token not in log]
    assert not missing, f"the native settings line lacks {missing}:\n{log[-3000:]}"


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
def test_native_illegal_thread_knobs_fall_back():
    """Set thread counts that break a ring's STRIDED ratio must send the op to the fallback factory.

    R = 3 beside C = 4 gives the input rings max(R, C) % min(R, C) = 1, which the DFB host refuses. The gate
    checks the ratio on the NoC path's counts, so the op runs on the Metal 2.0 factory, and runs right.
    """
    p = _run_shard_arm([_DRAM_128], expect="dfb", timeout=900, knobs=_threads(3, 4, 2))
    _check_arm(p, "R=3 beside C=4")


@pytest.mark.skipif(not _native_enabled(), reason="native factory not enabled (TTNN_QSR_NATIVE)")
def test_native_prediction_tables_cover_their_cases():
    """Each prediction must describe its whole case, and every case must have one. No device.

    A split that misses tiles or counts a pad slot, a borrowed shard that 4 does not divide, or an
    L1-interleaved case that is neither in the rule table nor a pad case would make a prediction wrong, or
    silently leave a case out of the rule column.
    """
    cases = _BORROWED_SHARD_CASES + _BATCHED_BORROWED_CASES + _L1_INTERLEAVED_RULE_CASES + _DRAM_SHARDED_CASES
    uncovered = [entry[0] for entry, _, split in cases if sum(split) != _tensor_tiles(entry)]
    assert not uncovered, f"these splits do not cover their output: {uncovered}"
    indivisible = [entry[0] for entry in _BORROWED_SHARDS if _shard_tiles(entry) % _RULE_COMPUTE_THREADS]
    assert not indivisible, f"these borrowed shards leave tail tiles under the rule: {indivisible}"
    left_out = {entry[0] for entry in _L1_INTERLEAVED + _L1_INTERLEAVED_NOC} - set(_L1_INTERLEAVED_RULE_SPLITS)
    assert left_out == {"l1_100_pad", "l1_250_pad"}, f"L1-interleaved cases in no prediction table: {left_out}"
