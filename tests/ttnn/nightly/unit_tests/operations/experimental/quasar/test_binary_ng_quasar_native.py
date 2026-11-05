# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Quasar-native binary_ng: proof that the native factory ran, and that it computed the predicted answer.

Routing is read from the inspector's kernels.yaml, which names the kernel SOURCES actually bound. That
is the only observable that separates the two factories at this stage -- they are a mechanical copy of
each other, so cycles, values and the active-core set are identical by construction. The native tests
compare bits with the bf16 oracles; the fallback test uses the shared _run helper (golden + PCC floor +
constant-output guard).

The native factory is the default on Quasar, and TTNN_QSR_NATIVE=0 turns it off. The tests read that state
from the environment rather than parametrize it: the program cache resolves the factory from a cached index, so
flipping the knob in-process would silently re-run the previous program. So the native tests run by default, and
test_fallback_factory_is_engaged runs in a process with TTNN_QSR_NATIVE=0. Off Quasar the native tests skip.

Every TTNN_QSR_* tuning knob left unset takes the rule's value (resolve_native_config in
binary_ng_utils.cpp): 4 compute threads; where the operands move over the NoC, a reader thread per compute
thread and the writer threads that the 6 DM cores leave, and one of each where they are borrowed. With every
operand in L1: batches of 8 tiles in the DM kernels, in rings of 16 entries per thread; in the compute, where each
compute thread owns one tile counter of each ring, 4 tiles on the SFPU kernel (2 in fp32) and 8 on the FPU kernel at
1,1,1 only. Beside a DRAM operand: batches of 1, in rings of 2 entries per thread, except where every operand is
interleaved and a compute thread draws 4 tiles or more. The tests in this process use DRAM tensors. Each test
predicts its output from its configuration: the tests in this process from the invocation's knobs, the arms from
their own. A test that predicts wrong output is a strict expected failure, and it passes only on exactly the
predicted wrong tiles.
"""

import json
import math
import os
import pathlib
import re
import subprocess
import sys
import time

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.binary_ng_quasar_test_utils import (
    _add_goldens,
    _matches_none,
    _run,
)
from tt_py_test_utils_common.utility_functions import is_quasar

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
    # On Quasar unless TTNN_QSR_NATIVE=0, matching native_tuning() in binary_ng_utils.cpp, which refuses any value but
    # 0 and 1. Off Quasar the gate refuses every op, so the native tests skip there.
    return is_quasar() and os.environ.get("TTNN_QSR_NATIVE", "") != "0"


# The rule's value of every knob left unset. Mirrors resolve_native_config in binary_ng_utils.cpp: a test whose
# prediction disagrees with the device is how a change on either side shows up.
_RULE_COMPUTE_THREADS = 4
# The FPU kernel's rule batch is 8 at R = C = W = 1 only, until batching the other thread splits is measured.
_RULE_TILES_PER_CYCLE = 8
# The SFPU kernel's rule batch is 4: it holds a tile in two DST slots, so 4 tiles fill a 16-bit DST section. In fp32
# the rule runs it at 2; no prediction reads that batch, as SFPU batches are exact, and the log test checks it.
_RULE_SFPU_TILES_PER_CYCLE = 4
_USER_DM_CORES = 6

# The worker clusters of the simulator's 8x4 grid, which every test here assumes.
_CLUSTERS = 32


# Where a program's operands are, as (an input moves over the NoC, the output moves over the NoC, an operand is
# in DRAM): all of them over the NoC and in L1, all of them over the NoC beside a DRAM operand, or all of them
# borrowed in place. _path gives the others.
_NOC_L1 = (True, True, False)
_NOC_DRAM = (True, True, True)
_BORROWED = (False, False, False)


def _resolve(knobs, path, sfpu=False):
    """(R, C, W, compute batch) of a program run with `knobs`, a mapping of TTNN_QSR_* names to values.

    `path` is _NOC_L1, _NOC_DRAM, _BORROWED or one from _path. A borrowed pair of inputs or a borrowed output
    runs one reader or writer thread whatever is set. Beside a DRAM operand it models batch 1, not the compute
    thread's share: nothing that reads the batch predicts a wrong tile. `sfpu` is True for an op on the SFPU
    kernel, whose rule batch is smaller. An empty value counts as unset, as in native_tuning().
    """
    inputs_move, output_moves, dram = path

    def knob(name, rule):
        value = knobs.get(name, "")
        return int(value) if value else rule

    c = knob("TTNN_QSR_COMPUTE_THREADS", _RULE_COMPUTE_THREADS)
    r = knob("TTNN_QSR_READER_THREADS", c) if inputs_move else 1
    w = knob("TTNN_QSR_WRITER_THREADS", min(c, max(1, _USER_DM_CORES - r))) if output_moves else 1
    fpu_batch = _RULE_TILES_PER_CYCLE if r == c == w == 1 else 1
    rule_batch = 1 if dram else (_RULE_SFPU_TILES_PER_CYCLE if sfpu else fpu_batch)
    batch = knob("TTNN_QSR_TILES_PER_CYCLE", rule_batch if r <= c and w <= c else 1)
    return r, c, w, batch


def _path(borrowed, dram):
    """The path of a program that borrows `borrowed`, a string of the operands "a", "b" and "c" it holds in place."""
    return ("a" not in borrowed or "b" not in borrowed, "c" not in borrowed, dram)


def _route(case, knobs):
    """(DM cores, clusters) that run the program of `case` under `knobs`: R + W, and the clusters of its split."""
    _, path, split = case
    r, _, w, _ = _resolve(knobs, path)
    return r + w, len(split)


# No tile is wrong: the binary unpack steps a batch by the ring stride (#61774). Kept so a new defect can be modeled.
def _batched_wrong_tiles(tiles_per_cluster, knobs, path, sfpu=False):
    return 0


def _noc_split(tiles, clusters=_CLUSTERS):
    """Each cluster's tile count on the NoC path, which split_work_to_cores spreads over the worker clusters."""
    per_cluster, extra = divmod(tiles, clusters)
    return [per_cluster + 1] * extra + [per_cluster] * (clusters - extra)


def _wrong_tiles(got, goldens):
    """The 32x32 tiles of a 2D output in which any element matches none of the oracles."""
    differ = _matches_none(got, goldens)
    height, width = differ.shape[-2:]
    return int(differ.reshape(-1, height // 32, 32, width // 32, 32).any(dim=4).any(dim=2).sum())


def _threads(r, c, w):
    return {"TTNN_QSR_READER_THREADS": str(r), "TTNN_QSR_COMPUTE_THREADS": str(c), "TTNN_QSR_WRITER_THREADS": str(w)}


# Compute batching off. Every other knob keeps the rule's value.
_COMPUTE_BATCH_1 = {"TTNN_QSR_TILES_PER_CYCLE": "1"}

# One tile per DM transfer and per compute batch, in rings of two entries per thread, so that a thread's ring
# wraps every two tiles.
_PER_TILE = {"TTNN_QSR_DM_BATCH": "1", "TTNN_QSR_ENTRIES_PER_THREAD": "2", "TTNN_QSR_TILES_PER_CYCLE": "1"}

# The invocation's knobs, which every test in this process runs with.
_PARENT_KNOBS = {k: v for k, v in os.environ.items() if k.startswith("TTNN_QSR_")}


class _WrongOutput(Exception):
    """An op ran to completion on the expected factory, and its output differs from the bf16 oracles."""


# A test whose prediction has wrong tiles is a strict expected failure; _batched_wrong_tiles predicts none today.
# strict=True: when the batched output changes, the counts move, and the run fails until the predictions follow.
_BATCHED_REASON = "compute batching: a modeled defect leaves these tiles wrong"
_BATCHED_XFAIL = pytest.mark.xfail(raises=_WrongOutput, strict=True, reason=_BATCHED_REASON)


def _expectation_marks(expected_wrong):
    """A test that predicts wrong output is a strict expected failure."""
    return [_BATCHED_XFAIL] if expected_wrong > 0 else []


def _matches_prediction(got, goldens, expected_wrong, what):
    """Assert that exactly the predicted number of output tiles differ from the oracles; 0 means exact.

    Returns True when the output is wrong as predicted.
    """
    wrong = _wrong_tiles(got, goldens)
    total = got.numel() // (32 * 32)
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
_BENCHMARK_WRONG = _batched_wrong_tiles(_noc_split(32 * 40), _PARENT_KNOBS, _NOC_DRAM)


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.xfail(_BENCHMARK_WRONG > 0, raises=_WrongOutput, strict=True, reason=_BATCHED_REASON)
def test_native_output_matches_the_prediction(device):
    """bf16 add on this slice is exact: the output equals one of _FPU_ADD_ROUNDINGS bit for bit, except the
    tiles that a set batch leaves wrong."""
    out, ta, tb = _add_returning_operands(device)
    if _matches_prediction(ttnn.to_torch(out), _add_goldens(ta, tb), _BENCHMARK_WRONG, "32x40 tiles"):
        raise _WrongOutput("32x40 tiles: wrong as predicted")


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
def test_native_factory_is_engaged(device):
    """The benchmark add binds the three kernels_qsr sources and no kernels_dfb source.

    Routing only: test_native_output_matches_the_prediction checks the values of the same add.
    """
    _add_returning_operands(device)
    sources = bound_kernel_sources()
    qsr = [s for s in sources if "kernels_qsr/" in s]
    dfb = [s for s in sources if "kernels_dfb/" in s]
    assert len(qsr) == 3, f"expected 3 kernels_qsr sources, got {qsr}"
    assert dfb == [], f"native run still bound kernels_dfb sources: {dfb}"


@pytest.mark.skipif(_native_enabled(), reason="negative control runs with TTNN_QSR_NATIVE=0")
def test_fallback_factory_is_engaged(device):
    assert is_quasar() == (device.arch() == ttnn.device.Arch.QUASAR), "ttnn.get_arch_name() disagrees with the device"
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
import hashlib, json, os, pathlib, sys
import torch, ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.binary_ng_quasar_test_utils import (
    _add_goldens, _matches_none)

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
        got = ttnn.to_torch(out)
        # Output bits, to compare two arms.
        bits = got.contiguous().view(torch.int16).numpy().tobytes()
        print("digest {}: {}".format(tiles, hashlib.sha256(bits).hexdigest()))
        differ = _matches_none(got, _add_goldens(ta, tb))
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
    number of output tiles a modeled defect leaves wrong. Only exactly those counts raise
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
        raise _WrongOutput(f"{what}: output differs from the bf16 oracles as predicted\n{p.stdout}")
    assert p.returncode == 0, f"{what} failed:\n{p.stdout}\n{p.stderr[-2000:]}"


def _add_one_row(device, tiles, seed):
    """One (1 x tiles) add. Returns (output, bf16 oracles).

    A 1-tile-tall shape is a legitimate one-dimensional sweep here because the native reader walks
    page = start_tile_id + k -- a pure linear index -- so behaviour depends on the tile COUNT alone and
    not on how it factors into height x width.

    The goldens are rounded to bf16, not left in fp32: the device packs bf16, and the sum of two bf16
    values needs 9 mantissa bits where bf16 has 8, so an fp32 comparison reports ~54% mismatch on
    random data purely from rounding.
    """
    out, ta, tb = _add_returning_operands(device, h_tiles=1, w_tiles=tiles, seed=seed)
    return ttnn.to_torch(out), _add_goldens(ta, tb)


def _parent_prediction(tiles):
    return _batched_wrong_tiles(_noc_split(tiles), _PARENT_KNOBS, _NOC_DRAM)


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize(
    "tiles", [pytest.param(t, marks=_expectation_marks(_parent_prediction(t))) for t in _RAGGED_TILE_COUNTS], ids=str
)
def test_native_uneven_tile_counts_match_the_prediction(device, tiles):
    before = set(bound_kernel_sources())
    got, goldens = _add_one_row(device, tiles, seed=tiles)
    wrong_as_predicted = _matches_prediction(got, goldens, _parent_prediction(tiles), f"{tiles} tiles")

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


# Even, ragged across clusters, and ragged across threads. Under the rule the output is exact, so a wrong tile
# can only come from the program cache.
_CACHE_SEQUENCE = (128, 33, 128, 32, 128)
_CACHE_WRONG = max(_parent_prediction(t) for t in _CACHE_SEQUENCE)


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.xfail(_CACHE_WRONG > 0, raises=_WrongOutput, strict=True, reason=_BATCHED_REASON)
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
# in two-entry rings, so the rings wrap at the low tile counts. batched sets a DM batch of 8 at every tile count,
# in the 16-entry rings that the rule derives for it, with the compute batch at 1 so that only the DM batch
# changes.
_DM_BATCH_8 = {"TTNN_QSR_DM_BATCH": "8"}
_BATCHING = {"per-tile": _PER_TILE, "batched": {**_DM_BATCH_8, **_COMPUTE_BATCH_1}}


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
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


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("rcw", _DM_BATCH_ARMS, ids=lambda t: "R{}C{}W{}".format(*t))
def test_native_dm_batch_above_one_counter_is_bit_exact(rcw):
    """A DM batch of 8 on a role that owns more than one tile counter must stay bit-exact.

    The batch is set by its knob. The rule's depth for it, 16, is the
    smallest legal one: a batch occupies n consecutive slots and the ring wraps only afterwards, so the
    factory demands entries_per_thread >= 2n and entries_per_thread % n == 0. The compute batch is 1, so only
    the DM batch is under test.

    One process per arm, as above. The timeout is the failure detector for a broken pairing: a counter
    asked for more tiles than it receives hangs rather than corrupts.
    """
    p = _run_arm(_DM_BATCH_TILE_COUNTS, knobs={**_threads(*rcw), **_DM_BATCH_8, **_COMPUTE_BATCH_1}, timeout=600)
    _check_arm(p, f"dm_batch=8 arm R={rcw[0]} C={rcw[1]} W={rcw[2]}")


# --- Compute batching --------------------------------------------------------------------------------
#
# An FPU-kernel compute batch of 8 runs eight tiles per tile_regs_acquire. The output is exact at any ring stride:
# the binary unpack and the pack both step a batch by the stride. Per-cluster tile counts on the 32-cluster grid:
# 1 runs a single short batch and no full one; 9 runs one full batch of 8 then a tail of 1; 64 runs eight full
# batches, wrapping the 16-deep ring three times, and no tail; 65 adds a tail of 1 that starts exactly on a wrap
# boundary.
_TILES_PER_CYCLE_TILE_COUNTS = [32, 288, 2048, 2080]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("dm_batch", [1, 8], ids=["compute-batch-only", "both-batches"])
def test_native_tiles_per_cycle_is_bit_exact(dm_batch):
    """A compute batch of 8 at 1,1,1 must stay bit-exact, alone and together with a DM batch of 8.

    The compute kernel splits a thread's share into full chunks of num_tiles_per_cycle and one remainder
    chunk; a truncating divide there once dropped the tail and hung. The tile counts above reach every
    branch of that split. Both batches are set by their knobs.
    both-batches measured 1.81x the throughput of no batching at 1,1,1 on craq-sim (marginal cycles per
    tile), which models no DRAM time.
    """
    knobs = {**_threads(1, 1, 1), "TTNN_QSR_TILES_PER_CYCLE": "8", "TTNN_QSR_DM_BATCH": str(dm_batch)}
    p = _run_arm(_TILES_PER_CYCLE_TILE_COUNTS, knobs=knobs, timeout=600)
    _check_arm(p, f"tiles_per_cycle=8 dm_batch={dm_batch}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
def test_native_tiles_per_cycle_above_stride_one():
    """With the compute batch set to 8 and every other knob unset, the NoC path runs the rule's 4,4,2 and both
    batches of 8 in L1, and its output is exact: the binary unpack steps a batch by the ring stride (#61774).

    L1-interleaved tensors of 288 and 2080 tiles hold 9 and 65 tiles per bank, which 4 does not divide, so they
    take the NoC path. The ring stride is 4 on every DFB. 288 tiles give each compute thread two or three
    tiles, one short batch; 2080 give it sixteen or seventeen, two full batches of 8 and a tail.
    """
    knobs = {"TTNN_QSR_TILES_PER_CYCLE": "8"}
    cases = [_anchored((f"l1_{t}", "l1", None, None, (32, t * 32)), None, "") for t in (288, 2080)]
    p = _run_shard_arm([case[0] for case in cases], expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, "a compute batch of 8 on the L1 NoC path", expect_wrong=_expect_wrong(cases, knobs))


# Batching on and off must give the same output bits. 2080 tiles are 65 per cluster, at least 16 per compute thread,
# so with batching on both batches are 8, and with it off 1.
_DRAM_BATCH_SPLITS = [
    ("rule", {}),
    ("R4C4W1", _threads(4, 4, 1)),
    ("R1C1W1", _threads(1, 1, 1)),
]
_DRAM_BATCH_TILE_COUNTS = _RAGGED_TILE_COUNTS + [2080]
# The factory's line for each program beside DRAM: its tile count, DM batch and compute batch.
_PROGRAM_BATCHES = r"(\d+) tiles beside DRAM, \d+ per compute thread: dm_batch=(\d+) .*?tiles_per_cycle=(\d+)"


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("split", _DRAM_BATCH_SPLITS, ids=[s[0] for s in _DRAM_BATCH_SPLITS])
def test_native_dram_batch_is_bit_exact(split):
    label, knobs = split
    digests = {}
    for state, switch, batch in (("ON", {}, 8), ("OFF", {"TTNN_QSR_DRAM_BATCH": "0"}, 1)):
        p = _run_arm(_DRAM_BATCH_TILE_COUNTS, knobs={**knobs, **switch, "TT_LOGGER_LEVEL": "info"}, timeout=1200)
        _check_arm(p, f"batching beside DRAM {state}, {label}")
        log = p.stdout + p.stderr
        assert f"batching beside DRAM {state}" in log, f"{label}: the log does not report batching {state}"
        # The batches each program resolved, from the factory's line per program.
        batches = {int(m.group(1)): (int(m.group(2)), int(m.group(3))) for m in re.finditer(_PROGRAM_BATCHES, log)}
        assert batches.get(2080) == (batch, batch), f"{label}: with batching {state}, 2080 tiles resolve {batches}"
        digests[state] = dict(re.findall(r"^digest (\d+): (\w+)$", p.stdout, re.MULTILINE))
    assert len(digests["ON"]) == len(_DRAM_BATCH_TILE_COUNTS), f"{label}: missing digests {digests['ON']}"
    differ = sorted(int(t) for t in digests["ON"] if digests["ON"][t] != digests["OFF"].get(t))
    assert not differ, f"{label}: batching changes the output at {differ} tiles"


# Each guard, the knob values that trip it, the text its TT_FATAL carries, and optionally the _SHARD_ARM_SRC
# entries to run it on instead of a DRAM add. A batch needs a ring at least twice as deep (double buffering)
# and a depth that is a multiple of the batch (a batch occupies consecutive slots and the ring wraps only after
# it). A compute batch also needs each compute thread to own one tile
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
        # The rule batches only when every operand is in L1: an L1-interleaved tensor of 5 tiles per bank
        # takes the NoC path at the rule's 4 compute threads, where the rule's DM batch is 8 and its FPU batch 1.
        "rule-batch-depth",
        {"TTNN_QSR_ENTRIES_PER_THREAD": "8"},
        "dm_batch=8 (rule) needs entries_per_thread >= 16 for double buffering, "
        "got entries_per_thread=8 (TTNN_QSR_ENTRIES_PER_THREAD)",
        [("l1_160", "l1", None, None, (5 * 32, 32 * 32))],
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
    (
        # A batch must fit one DST section: the SFPU kernel runs at most 4 tiles in a 16-bit DST, so a set batch of 8
        # is refused for multiply.
        "sfpu-batch",
        {"TTNN_QSR_TILES_PER_CYCLE": "8"},
        "tiles_per_cycle=8 (TTNN_QSR_TILES_PER_CYCLE) exceeds 4, the most the SFPU kernel runs in a 16-bit DST",
        [("dram_128/multiply", "dram", None, None, (4 * 32, 32 * 32), {"op": "multiply"})],
    ),
    (
        # In a 32-bit DST the SFPU kernel runs at most 2 tiles, so a set batch of 4 is refused for fp32 add.
        "sfpu-batch-fp32",
        {"TTNN_QSR_TILES_PER_CYCLE": "4"},
        "tiles_per_cycle=4 (TTNN_QSR_TILES_PER_CYCLE) exceeds 2, the most the SFPU kernel runs in a 32-bit DST",
        [("dram_128/fp32", "dram", None, None, (4 * 32, 32 * 32), {"dtype": "float32"})],
    ),
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("arm", _GUARD_ARMS, ids=[a[0] for a in _GUARD_ARMS])
def test_native_knob_guards_refuse(arm):
    """A knob setting the factory cannot run must fail with its own message, not run and hang.

    The message is asserted, not only a non-zero exit: an arm that died for another reason -- no
    simulator, a build break -- would otherwise pass as a refusal.
    """
    _, knobs, message, *cases = arm
    if cases:
        p = _run_shard_arm(cases[0], expect="qsr", timeout=300, knobs=knobs)
    else:
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
# same column over a grid of four clusters, two of which hold no shard, and the program does not run there.
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
    """Tiles in the tensor of a _SHARD_ARM_SRC entry, of any rank. A partial tile counts as a whole one."""
    *outer, height, width = entry[4]
    return math.prod(outer) * math.ceil(height / 32) * math.ceil(width / 32)


def _shard_cores(entry):
    """Cores in the inclusive core range of a _SHARD_ARM_SRC entry."""
    x0, y0, x1, y1 = entry[3]
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def _runs_sfpu(entry):
    """True when the op of a _SHARD_ARM_SRC entry runs the SFPU kernel: multiply out of its fast mode, divide, or any
    op on fp32 operands."""
    opts = entry[5] if len(entry) > 5 else {}
    return opts.get("op") in ("multiply", "divide") or opts.get("dtype") == "float32"


def _expect_wrong(cases, knobs):
    """The prediction of each case that `knobs` leaves wrong: {name: wrong output tiles}.

    A case is (entry, path, tiles per cluster), with path as in _resolve. The tile counts must cover the whole
    output; a case with pad slots must predict no wrong tile in any configuration.
    test_native_prediction_tables_cover_their_cases checks both.
    """
    predicted = {entry[0]: _batched_wrong_tiles(split, knobs, path, _runs_sfpu(entry)) for entry, path, split in cases}
    return {name: wrong for name, wrong in predicted.items() if wrong > 0}


def _config_params(configs, cases):
    """One pytest param per configuration, a strict expected failure where the prediction is wrong output."""
    return [
        pytest.param(name, knobs, marks=_expectation_marks(sum(_expect_wrong(cases, knobs).values())), id=name)
        for name, knobs in configs.items()
    ]


# Same shape as _ARM_SRC: a fixed child script, parameters as JSON in the environment. ARM_EXPECT names the factory
# every op in the child must have bound: "qsr" for native, "dfb" for the fallback. The strategy "l1" or "dram" places a
# tensor interleaved, and then the shard shape and range are None. The strategies "dram_height", "dram_width" and
# "dram_block" shard it over all DRAM banks, and then the range is None. A shard entry may carry a sixth element, a dict
# of options: "out" gives the output its own placement (otherwise it takes the inputs'), "b" does the same for b,
# "output_tensor" supplies a preallocated output with its own placement, "in_place" adds b into a with add_, "grid" (x0,
# y0, x1, y1) runs the op on that sub-core grid, "op" runs "subtract", "multiply", "divide" or "multiply_fast" (multiply
# in fast and approximate mode) instead of add, "dtype": "float32" runs it on fp32 operands and output (with "op" add,
# subtract, multiply or divide, and "act" None, "lhs", "rhs" or "relu"), and "act" chains SQUARE on a ("lhs"), on b ("rhs"), on both ("both") or
# on the result ("post"), a lone RELU on the result ("relu"), RELU on a ("relu_lhs"), or SQUARE on a and RELU on the
# result ("relu_chain"). "out" may also be None, so that the op derives the output's memory config, or
# "height_specless", an L1 height-sharded config without a shard spec, which the op completes; then "resolved_out" names
# the placement the output must take. A placement in an option may add a fourth element, "col", for column-major shard
# order. An entry's tensors stay alive to the end when a later entry repeats its spec, so the repeat lands at new
# addresses and must hit the program cache; the other entries' tensors are freed, so that many cases fit in L1 one after
# another. With ARM_EXPECT_NEOS or ARM_EXPECT_DMS (and the device profiler on), every core must show that many Neos
# running the compute, or DM cores running a reader or writer thread; ARM_EXPECT_CORES names how many cores must report.
_SHARD_ARM_SRC = """
import collections, csv, json, os, pathlib, sys
import torch, ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.binary_ng_quasar_test_utils import (
    _FPU_ADD_ROUNDINGS, _ULP, _matches_none)

SHARDS = json.loads(os.environ["ARM_SHARDS"])
EXPECT = os.environ["ARM_EXPECT"]
EXPECT_NEOS = int(os.environ.get("ARM_EXPECT_NEOS", "0"))
EXPECT_DMS = int(os.environ.get("ARM_EXPECT_DMS", "0"))
EXPECT_CORES = int(os.environ.get("ARM_EXPECT_CORES", "0"))
STRATEGY = {"height": ttnn.ShardStrategy.HEIGHT, "block": ttnn.ShardStrategy.BLOCK,
            "width": ttnn.ShardStrategy.WIDTH}
LAYOUT = {"height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED, "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
          "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED}
INTERLEAVED = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}
DRAM_SHARDED = {"dram_height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                "dram_width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                "dram_block": ttnn.TensorMemoryLayout.BLOCK_SHARDED}
root = pathlib.Path(os.environ.get("TT_METAL_LOGS_PATH") or os.environ.get("TT_METAL_HOME") or os.getcwd())


def round_to_bf16(x):
    return x.to(torch.bfloat16)


def truncate_to_bf16(x):
    # Quasar's SFPU multiply drops the low 16 bits of the fp32 result today, where the Wormhole and Blackhole LLKs
    # round to nearest even.
    return (x.contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


# op -> (the op, its extra arguments, the kernel's bf16 results from the operands; an element is right when it
# matches one). The FPU kernel adds and subtracts as _FPU_ADD_ROUNDINGS says, and multiplies with rounding to nearest
# even. The SFPU kernel runs multiply out of fast mode, which truncates, and divide, which rounds to nearest even.
OPS = {"add": (ttnn.experimental.quasar.add, {}, _FPU_ADD_ROUNDINGS),
       "subtract": (ttnn.experimental.quasar.subtract, {},
                    tuple(lambda a, b, rounding=rounding: rounding(a, -b) for rounding in _FPU_ADD_ROUNDINGS)),
       "multiply": (ttnn.experimental.quasar.multiply, {}, (lambda a, b: truncate_to_bf16(torch.multiply(a, b)),)),
       "multiply_fast": (ttnn.experimental.quasar.multiply, {"fast_and_approximate_mode": True},
                         (lambda a, b: round_to_bf16(torch.multiply(a, b)),)),
       "divide": (ttnn.experimental.quasar.divide, {}, (lambda a, b: round_to_bf16(torch.divide(a, b)),))}


def square(x):
    # SQUARE runs on the SFPU on both kernels, and truncates as the SFPU multiply does.
    return truncate_to_bf16(x * x).float()


# act ->(the op's activation arguments, the output from the operands and one of the op's roundings). SQUARE
# chains on a, on b, on both or on the result; a lone RELU on the result runs in the packer, on the rounded result.
# RELU on a, and RELU on the result beside a chain, run on the SFPU.
SQUARE, RELU = ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE), ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
ACTS = {None: ({}, lambda a, b, op: op(a, b)),
        "lhs": ({"input_tensor_a_activations": [SQUARE]}, lambda a, b, op: op(square(a), b)),
        "rhs": ({"input_tensor_b_activations": [SQUARE]}, lambda a, b, op: op(a, square(b))),
        "both": ({"input_tensor_a_activations": [SQUARE], "input_tensor_b_activations": [SQUARE]},
                 lambda a, b, op: op(square(a), square(b))),
        "post": ({"activations": [SQUARE]}, lambda a, b, op: square(op(a, b).float()).to(torch.bfloat16)),
        "relu": ({"activations": [RELU]}, lambda a, b, op: torch.relu(op(a, b))),
        "relu_lhs": ({"input_tensor_a_activations": [RELU]}, lambda a, b, op: op(torch.relu(a), b)),
        "relu_chain": ({"input_tensor_a_activations": [SQUARE], "activations": [RELU]},
                       lambda a, b, op: torch.relu(op(square(a), b)))}


# On fp32 operands every op runs on the SFPU kernel and keeps the fp32 result: within _ULP of torch's, the bound the
# suites check too.
OPS_FP32 = {"add": (ttnn.experimental.quasar.add, {}, (torch.add,)),
            "subtract": (ttnn.experimental.quasar.subtract, {}, (torch.subtract,)),
            "multiply": (ttnn.experimental.quasar.multiply, {}, (torch.multiply,)),
            "divide": (ttnn.experimental.quasar.divide, {}, (torch.divide,))}
ACTS_FP32 = {None: ACTS[None],
             "lhs": ({"input_tensor_a_activations": [SQUARE]}, lambda a, b, op: op(a * a, b)),
             "rhs": ({"input_tensor_b_activations": [SQUARE]}, lambda a, b, op: op(a, b * b)),
             "relu": ({"activations": [RELU]}, lambda a, b, op: torch.relu(op(a, b)))}
DTYPES = {"bfloat16": ttnn.bfloat16, "float32": ttnn.float32}
profile_csv = root / "generated" / "profiler" / ".logs" / "profile_log_device.csv"
if (EXPECT_NEOS or EXPECT_DMS) and profile_csv.exists():
    profile_csv.unlink()


def memory_config(strategy, shard_shape, rng, orientation="row"):
    if strategy in INTERLEAVED:
        return INTERLEAVED[strategy]
    if strategy in DRAM_SHARDED:
        # The shard grid of a DRAM tensor is in DRAM-bank coordinates, not Tensix cores.
        g = device.dram_grid_size()
        banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
        spec = ttnn.ShardSpec(banks, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(DRAM_SHARDED[strategy], ttnn.BufferType.DRAM, spec)
    x0, y0, x1, y1 = rng
    grid = ttnn.CoreRangeSet({ttnn.CoreRange((x0, y0), (x1, y1))})
    if orientation == "col":
        # From a ShardSpec: create_sharded_memory_config transposes the shard shape for column-major.
        spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
        return ttnn.MemoryConfig(LAYOUT[strategy], ttnn.BufferType.L1, spec)
    return ttnn.create_sharded_memory_config(
        shard_shape, core_grid=grid, strategy=STRATEGY[strategy], use_height_and_width_as_shard_shape=True
    )


mismatches, failures = [], []
keep, seen = [], set()
specs = [json.dumps(entry[1:]) for entry in SHARDS]
device = ttnn.open_device(device_id=0)
try:
    for i, entry in enumerate(SHARDS):
        name, strategy, shard_shape, rng, shape = entry[:5]
        opts = entry[5] if len(entry) > 5 else {}
        torch.manual_seed(i)
        mem = memory_config(strategy, shard_shape, rng)
        if "out" in opts and opts["out"] is None:
            out_mem = None
        elif opts.get("out") == "height_specless":
            out_mem = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        else:
            out_mem = memory_config(*opts["out"]) if "out" in opts else mem
        b_mem = memory_config(*opts["b"]) if "b" in opts else mem
        fp32 = opts.get("dtype") == "float32"
        dtype = DTYPES[opts.get("dtype", "bfloat16")]
        cfg = dict(dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)
        ta = ttnn.from_torch(torch.randn(shape, dtype=torch.float32), memory_config=mem, **cfg)
        tb = ttnn.from_torch(torch.randn(shape, dtype=torch.float32), memory_config=b_mem, **cfg)
        op_fn, op_extra, results = (OPS_FP32 if fp32 else OPS)[opts.get("op", "add")]
        act_kwargs, model = (ACTS_FP32 if fp32 else ACTS)[opts.get("act")]
        a, b = ttnn.to_torch(ta).float(), ttnn.to_torch(tb).float()
        goldens = [model(a, b, op) for op in results]
        cached = device.num_program_cache_entries()
        op_kwargs = {}
        if "grid" in opts:
            x0, y0, x1, y1 = opts["grid"]
            op_kwargs["sub_core_grids"] = ttnn.CoreRangeSet({ttnn.CoreRange((x0, y0), (x1, y1))})
        if opts.get("in_place"):
            if opts.get("op", "add") != "add" or act_kwargs:
                failures.append("{}: in place runs add only, with no activation".format(name))
            out = ttnn.experimental.quasar.add_(ta, tb)
        else:
            if "output_tensor" in opts:
                op_kwargs["output_tensor"] = ttnn.from_torch(
                    torch.zeros(shape), memory_config=memory_config(*opts["output_tensor"]), **cfg)
            if out_mem is not None:
                op_kwargs["memory_config"] = out_mem
            out = op_fn(ta, tb, dtype=dtype, **op_extra, **act_kwargs, **op_kwargs)
        if "resolved_out" in opts and out.memory_config() != memory_config(*opts["resolved_out"]):
            failures.append("{}: the output took {}".format(name, out.memory_config()))
        spec = json.dumps(entry[1:])
        if spec in seen and device.num_program_cache_entries() != cached:
            failures.append("{}: repeats an earlier spec but missed the program cache".format(name))
        seen.add(spec)
        if spec in specs[i + 1:]:
            keep.extend([ta, tb, out])
        got = ttnn.to_torch(out)
        # Integer ULP distance per element, as production measures it: the sign of a zero does not count. A bf16
        # golden is one of the kernel's exact roundings; fp32 allows the op's measured bound.
        bound = _ULP[ttnn.float32][opts.get("op", "add")] if fp32 else 0
        differ = _matches_none(got, goldens, bound)
        if differ.any():
            # Pad a partial last tile row or column with matches, so that the mask holds whole tiles.
            differ = torch.nn.functional.pad(differ, (0, -differ.shape[-1] % 32, 0, -differ.shape[-2] % 32))
            h, w = differ.shape[-2:]
            wrong = int(differ.reshape(-1, h // 32, 32, w // 32, 32).any(dim=4).any(dim=2).sum())
            mismatches.append("{}: {} of {} tiles differ".format(name, wrong, differ.numel() // 1024))
        # Free this entry's tensors before the next entry allocates its own; `keep` holds the ones a repeat needs.
        del ta, tb, out, op_kwargs
finally:
    ttnn.close_device(device)

sources = [ln.split("source:", 1)[1].strip()
           for ln in (root / "generated" / "inspector" / "kernels.yaml").read_text().splitlines()
           if "source:" in ln]
# Every binary_ng kernel tree counts, so a kernel source from any tree but kernels_qsr/ and kernels_dfb/
# fails the check.
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
# configuration runs them too, at the FPU kernel's compute batch of 1, which is exact.
_BORROWED_SHARD_CASES = [(entry, _BORROWED, [_shard_tiles(entry)] * _shard_cores(entry)) for entry in _BORROWED_SHARDS]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", _config_params({**_BORROWED_CONFIGS, "rule": {}}, _BORROWED_SHARD_CASES))
def test_native_borrowed_shards_match_the_prediction(name, knobs):
    """Height, block and width shards must run native and give the predicted output in every configuration.

    A wrong credit count hangs into the timeout, credits on the wrong counter fail the bit comparison,
    and a cached program that keeps the first call's shard addresses fails the repeat.
    """
    p = _run_shard_arm(_BORROWED_SHARDS, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"borrowed shards, {name}", expect_wrong=_expect_wrong(_BORROWED_SHARD_CASES, knobs))


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", list(_BORROWED_CONFIGS.items()), ids=list(_BORROWED_CONFIGS))
def test_native_borrowed_uneven_shards_are_bit_exact(name, knobs):
    """Shards whose last clusters hold fewer real tiles must run native and bit-exact in every configuration.

    Every role processes the full shard's tile count. Roles that disagree on a boundary cluster's count
    hang into the timeout.
    """
    p = _run_shard_arm(_UNEVEN_SHARDS, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"uneven shards, {name}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", list(_BORROWED_CONFIGS_C2.items()), ids=list(_BORROWED_CONFIGS_C2))
def test_native_borrowed_indivisible_shards_stay_native(name, knobs):
    """A shard whose tile count does not divide by the compute count must stay native and bit-exact.

    A borrowed ring sized to the whole shard dies on the DFB host assertion, and a tail copied from the
    wrong address or packed into the wrong ring fails the bit comparison. R1C2W1-n1 runs two-entry tail
    rings. At 1,1,1 every shard divides, and 9 and 10 tiles leave a remainder after one compute batch of 8.
    """
    p = _run_shard_arm(_SMALL_SHARDS, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"indivisible shards, {name}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
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


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
def test_native_borrowed_mismatched_output_borrows_only_the_inputs():
    """An output on the inputs' grid with another shard spec must not be borrowed, and must be right.

    A predicate that compares grids alone borrows the output in place and fails the bit comparison. The
    inputs' anchor borrows them, and the output moves over the NoC in their order: one reader thread and the
    rule's 4 writer threads on each of the 4 clusters, which the device profiler shows. An op that borrowed
    the output too would run 1 and 1.
    """
    p = _run_shard_arm(
        _MISMATCHED_SHARDS,
        expect="qsr",
        timeout=900,
        knobs={
            **_COMPUTE_BATCH_1,
            "TT_METAL_DEVICE_PROFILER": "1",
            "ARM_EXPECT_DMS": "5",
            "ARM_EXPECT_NEOS": "4",
            "ARM_EXPECT_CORES": "4",
        },
    )
    _check_arm(p, "an output shard spec unlike the inputs'")


# 64 tiles per core: at 1,4,1 and a set batch of 8, each compute thread takes 16, two full batches at ring stride 4.
_BATCHED_BORROWED_SHARDS = [
    ("block64", "block", [8 * 32, 8 * 32], (0, 0, 1, 1), (2 * 8 * 32, 2 * 8 * 32)),
]
_BATCHED_BORROWED_CASES = [
    (entry, _BORROWED, [_shard_tiles(entry)] * _shard_cores(entry)) for entry in _BATCHED_BORROWED_SHARDS
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
def test_native_borrowed_tiles_per_cycle_above_stride_one():
    """With the compute batch set to 8 and every other knob unset, a borrowed shard runs the rule's 1,4,1 at a
    compute batch of 8, and its output is exact: the binary unpack steps a batch by the ring stride (#61774).
    """
    knobs = {"TTNN_QSR_TILES_PER_CYCLE": "8"}
    p = _run_shard_arm(_BATCHED_BORROWED_SHARDS, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(
        p, "a compute batch of 8 on a borrowed shard", expect_wrong=_expect_wrong(_BATCHED_BORROWED_CASES, knobs)
    )


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
# divide; l1_128_sub_grid runs on 16 of the 32 bank cores. At C = 1 the first three borrow, and l1_8_idle
# leaves 24 clusters with a pad slot only.
_L1_INTERLEAVED_NOC = [
    ("l1_160", "l1", None, None, (5 * 32, 32 * 32)),
    ("l1_64", "l1", None, None, (2 * 32, 32 * 32)),
    ("l1_8_idle", "l1", None, None, (2 * 32, 4 * 32)),
    ("l1_128_sub_grid", "l1", None, None, (4 * 32, 32 * 32), {"grid": (0, 0, 3, 3)}),
]

# Slices beside a DRAM operand: each cluster borrows its banks' slices, and the DRAM operand moves in bank
# order. l1_b_in_dram reads b, l1_out_in_dram writes the output and dram_a reads a.
_L1_INTERLEAVED_PARTIAL = [
    ("l1_b_in_dram", "l1", None, None, (4 * 32, 32 * 32), {"b": ("dram", None, None)}),
    ("l1_out_in_dram", "l1", None, None, (4 * 32, 32 * 32), {"out": ("dram", None, None)}),
    ("dram_a", "dram", None, None, (4 * 32, 32 * 32), {"b": ("l1", None, None), "out": ("l1", None, None)}),
]
_L1_INTERLEAVED_PARTIAL_BORROWED = {"l1_b_in_dram": "ac", "l1_out_in_dram": "ab", "dram_a": "bc"}

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
    "l1_128": (_BORROWED, [4] * 32),
    "l1_256": (_BORROWED, [8] * 32),
    "l1_128_again": (_BORROWED, [4] * 32),
    "l1_128_in_place": (_BORROWED, [4] * 32),
    "l1_160": (_NOC_L1, _noc_split(160)),
    "l1_64": (_NOC_L1, _noc_split(64)),
    "l1_8_idle": (_NOC_L1, _noc_split(8)),
    "l1_128_sub_grid": (_NOC_L1, _noc_split(128, clusters=16)),
    **{name: (_path(borrowed, True), [4] * 32) for name, borrowed in _L1_INTERLEAVED_PARTIAL_BORROWED.items()},
}
_L1_INTERLEAVED_RULE_CASES = [
    (entry, *_L1_INTERLEAVED_RULE_SPLITS[entry[0]])
    for entry in _L1_INTERLEAVED + _L1_INTERLEAVED_NOC + _L1_INTERLEAVED_PARTIAL
    if entry[0] in _L1_INTERLEAVED_RULE_SPLITS
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", list(_BORROWED_CONFIGS_C2.items()), ids=list(_BORROWED_CONFIGS_C2))
def test_native_l1_interleaved_is_bit_exact(name, knobs):
    """L1-interleaved operands must run native and bit-exact in every configuration, borrowed or not.

    Credits on the wrong counter or a slice read at the wrong offset fail the bit comparison, and a cached
    program that keeps the first call's slice addresses fails the repeat.
    """
    p = _run_shard_arm(
        _L1_INTERLEAVED + _L1_INTERLEAVED_NOC + _L1_INTERLEAVED_PARTIAL, expect="qsr", timeout=900, knobs=knobs
    )
    _check_arm(p, f"L1-interleaved, {name}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", _config_params({"rule": {}}, _L1_INTERLEAVED_RULE_CASES))
def test_native_l1_interleaved_matches_the_rule_prediction(name, knobs):
    """With every knob unset, L1-interleaved operands must give the rule's predicted output.

    The rule runs the FPU kernel at a compute batch of 1 at ring stride 4, so every case is exact. The NoC cases
    split their tiles over the worker clusters as usual.
    """
    p = _run_shard_arm([case[0] for case in _L1_INTERLEAVED_RULE_CASES], expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"L1-interleaved, {name}", expect_wrong=_expect_wrong(_L1_INTERLEAVED_RULE_CASES, knobs))


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
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


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("case, cores", _L1_INTERLEAVED_DM_NOC, ids=[c[0][0] for c in _L1_INTERLEAVED_DM_NOC])
def test_native_l1_interleaved_keeps_the_noc_path(case, cores):
    """An indivisible tile count per cluster or a partial grid must keep the NoC path, and run the rule's 4,4,2
    there.

    A borrowed ring must divide by the compute count, and the L1-interleaved path has no tail rings, so a
    borrowed ring of 5, 2 or 1 tiles at C = 4 dies on the DFB host assertion. On a grid that is not the
    bank cores, the pages of the banks outside it would never be computed.
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


_L1_INTERLEAVED_PARTIAL_ROUTES = [
    (entry, *_route((entry, _path(_L1_INTERLEAVED_PARTIAL_BORROWED[entry[0]], True), [4] * 32), _COMPUTE_BATCH_1))
    for entry in _L1_INTERLEAVED_PARTIAL
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize(
    "case, dms, cores", _L1_INTERLEAVED_PARTIAL_ROUTES, ids=[r[0][0] for r in _L1_INTERLEAVED_PARTIAL_ROUTES]
)
def test_native_l1_interleaved_borrows_beside_dram(case, dms, cores):
    """Slices beside a DRAM operand must be borrowed, and the DRAM operand must move in bank order.

    A program that borrows them runs one reader thread where both inputs are borrowed, and one writer thread
    where the output is; the NoC path runs four and two. The child reads them from the device profiler.
    """
    p = _run_shard_arm(
        [case],
        expect="qsr",
        timeout=900,
        knobs={
            **_COMPUTE_BATCH_1,
            "TT_METAL_DEVICE_PROFILER": "1",
            "ARM_EXPECT_DMS": str(dms),
            "ARM_EXPECT_NEOS": "4",
            "ARM_EXPECT_CORES": str(cores),
        },
    )
    _check_arm(p, f"{case[0]} with its slices borrowed")


# DRAM-sharded operands, alone and beside interleaved ones. The shapes need at least two DRAM banks; the
# simulator has two, so each case makes two shards, and the uneven ones leave the second shard partial. A
# wrong page-to-bank map that a, b and the output share would still give c = a + b, so every layout also
# appears beside an operand with another placement. dram_width_1024 holds 32 tiles per cluster, enough for
# full DM batches of 8 at the batched arms. dram_width_again repeats dram_width while its tensors are alive, a
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
# A DRAM-sharded operand always moves over the NoC. dram_a_l1_b_dram_out borrows b's slices on the 32 bank
# clusters; every other case here splits its tiles over the 32 worker clusters, with 4 tiles each. Beside DRAM
# the rule's program is the per-tile one: one page per transfer, a compute batch of 1, rings of 2 entries per
# thread. So the per-tile arms run the rule at other thread counts, and the batched arms set a DM batch of 8,
# which keeps the batched walk tested over DRAM-sharded accessors.
_DRAM_SHARDED_CONFIGS = {
    **{f"R{r}C{c}W{w}-per-tile": {**_threads(r, c, w), **_PER_TILE} for r, c, w in [(1, 1, 1), (1, 4, 1)]},
    "R1C1W1-batched": {**_threads(1, 1, 1), **_DM_BATCH_8, "TTNN_QSR_TILES_PER_CYCLE": "8"},
    "R1C4W1-dm8-n1": {**_threads(1, 4, 1), **_DM_BATCH_8, **_COMPUTE_BATCH_1},
    "rule-dm8-n1": {**_DM_BATCH_8, **_COMPUTE_BATCH_1},
    "rule": {},
}
_DRAM_SHARDED_CASES = [(entry, _NOC_DRAM, _noc_split(_tensor_tiles(entry))) for entry in _DRAM_SHARDED]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
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


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
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


# --- Mixed layouts ---------------------------------------------------------------------------------------
#
# Each operand may be interleaved (L1 or DRAM), L1-sharded or DRAM-sharded, on its own. The factory picks an anchor, a
# placement in which some operands already sit, and borrows them in place. A shard anchor is the memory config of an
# L1-sharded operand: it borrows every operand with exactly that memory config, and cluster i computes shard i. The
# slice anchor is the L1 banks: it borrows every L1-interleaved operand, and cluster i computes the pages of bank i. An
# anchor must run on at least as many clusters as the legacy binary_ng uses for the op: the clusters that hold the
# output's shards where the legacy binary_ng borrows the output, else its split of the output tiles over the 32
# clusters. Among those, the factory takes the anchor that borrows the most operands; on a tie, the one that borrows the
# output, then the one on more clusters, then a's before b's. Under the rule a borrowed ring must divide by 4, and a
# shard anchor's clusters must lie in the worker grid. Every operand that is not borrowed moves over the NoC in the
# anchor's tile order, and skips the anchor's pad slots. A placement is (strategy, shard shape, core range[, "col"]), as
# in _SHARD_ARM_SRC.
# The legacy binary_ng is the op in eltwise/binary_ng, not the descriptor or the fallback factory of this op.
_IL1 = ("l1", None, None)
_IDRAM = ("dram", None, None)
_H16 = ("height", [4 * 32, 16 * 32], (0, 0, 0, 3))  # on 16x16 tiles: 64 tiles per core down column 0
_B16 = ("block", [4 * 32, 4 * 32], (0, 0, 3, 3))  # 16 tiles per core on a 4x4 grid
_W16 = ("width", [16 * 32, 4 * 32], (0, 0, 3, 0))  # 64 tiles per core across row 0
_B16_COL = ("block", [4 * 32, 4 * 32], (0, 0, 3, 1), "col")  # on 16x8 tiles: column-major core order
_B16_SLABS = ("block", [4 * 32, 4 * 32], (0, 0, 3, 1))  # on (2, 2, 64, 512): each shard spans two (N, C) slabs
_H16_IDLE = ("height", [4 * 32, 16 * 32], (0, 0, 1, 3))  # 4 shards on 8 cores
_B10_UNEVEN = ("block", [4 * 32, 4 * 32], (0, 0, 2, 2))  # on 10x10 tiles: 3x3 shards, the last row and column short
_H32 = ("height", [32, 40 * 32], (0, 0, 7, 3))  # on 32x40 tiles: one tile row per cluster, on all 32 clusters
_B32 = ("block", [4 * 32, 4 * 32], (0, 0, 7, 3))  # on 16x32 tiles: 16 tiles per cluster, on all 32 clusters
_W32 = ("width", [4 * 32, 2 * 32], (0, 0, 7, 3))  # on 4x64 tiles: 8 tiles per cluster, on all 32 clusters
_B12 = ("block", [4 * 32, 4 * 32], (0, 0, 3, 2))  # on 12x16 tiles: 16 tiles per core on a 4x3 grid
_B32_COL = ("block", [4 * 32, 4 * 32], (0, 0, 7, 3), "col")  # on 32x16 tiles: column-major order on all 32 clusters
_SQUARE16 = (16 * 32, 16 * 32)
_PARTIAL16 = (16 * 32 - 7, 16 * 32 - 13)  # 16x16 tiles, the last tile row and column partial


def _mixed(name, a, b, out, shape=_SQUARE16, **opts):
    """A _SHARD_ARM_SRC entry for c = a + b with one placement per operand, and any further options."""
    return (name, *a, shape, {"b": b, "out": out, **opts})


def _placement(entry, operand):
    """The placement of operand "a", "b" or "c" of a _SHARD_ARM_SRC entry. A supplied output tensor's decides,
    and an in-place add writes into a, whatever "out" says. Where the op resolves the output, "resolved_out"
    names the result."""
    opts = entry[5] if len(entry) > 5 else {}
    if operand == "b":
        return opts.get("b", entry[1:4])
    if operand == "c" and not opts.get("in_place"):
        return opts.get("resolved_out", opts.get("output_tensor", opts.get("out", entry[1:4])))
    return entry[1:4]


def _has_dram(entry):
    """True when a, b or the output of a _SHARD_ARM_SRC entry is in DRAM, interleaved or sharded."""
    return any(_placement(entry, operand)[0].startswith("dram") for operand in "abc")


def _anchor_split(entry, anchor):
    """Tiles per cluster under `anchor`, pad slots included: "a", "b" or "c" puts one shard of that operand's
    placement on each core that holds one, "slice" puts each L1 bank's reservation on its cluster, and None
    splits the tiles over the worker clusters as the NoC path does."""
    tiles = _tensor_tiles(entry)
    if anchor is None:
        x0, y0, x1, y1 = entry[5].get("grid", (0, 0, 7, 3)) if len(entry) > 5 else (0, 0, 7, 3)
        return _noc_split(tiles, clusters=(x1 - x0 + 1) * (y1 - y0 + 1))
    if anchor == "slice":
        return [math.ceil(tiles / _CLUSTERS)] * _CLUSTERS
    _, (height, width), *_ = _placement(entry, anchor)
    *outer, rows, cols = entry[4]
    shard_rows, shard_cols = height // 32, width // 32
    # Each (N, C) slab rounds its rows up to whole tiles on its own.
    height_tiles = math.prod(outer) * math.ceil(rows / 32)
    shards = math.ceil(height_tiles / shard_rows) * math.ceil(math.ceil(cols / 32) / shard_cols)
    return [shard_rows * shard_cols] * shards


def _anchored(entry, anchor, borrowed):
    """(entry, path, tiles per cluster) of a case whose anchor borrows the operands named in `borrowed`."""
    return (entry, _path(borrowed, _has_dram(entry)), _anchor_split(entry, anchor))


# (entry, anchor, the operands it borrows), as the rule picks them at its compute count of 4; a set count can
# pick another anchor. The anchor is "a", "b" or "c" (that operand's shard placement), "slice" (the L1 banks) or
# None (the NoC path).
_MIXED = [
    # Two L1-interleaved inputs beside an L1-sharded output. The legacy binary_ng borrows the output on its shard grid.
    # The slices borrow two operands on all 32 clusters, which is not fewer, so they win, and the output moves in bank
    # order. IL1.IL1.B_again repeats IL1.IL1.B while its tensors are alive, a cache hit that must rebind.
    # IL1.IL1.Bsupplied asks for an L1-interleaved output but supplies a block-sharded tensor, whose own placement
    # decides. In IL1.IL1.B12 a bank holds 6 pages, which 4 does not divide, so the output's anchor borrows it on its 12
    # clusters, as the legacy binary_ng does, and the inputs move in its block order.
    (_mixed("IL1.IL1.H", _IL1, _IL1, _H16), "slice", "ab"),
    (_mixed("IL1.IL1.B", _IL1, _IL1, _B16), "slice", "ab"),
    (_mixed("IL1.IL1.W", _IL1, _IL1, _W16), "slice", "ab"),
    (_mixed("IL1.IL1.B_again", _IL1, _IL1, _B16), "slice", "ab"),
    (_mixed("IL1.IL1.Bcol", _IL1, _IL1, _B16_COL, shape=(16 * 32, 8 * 32)), "slice", "ab"),
    (_mixed("IL1.IL1.Hidle", _IL1, _IL1, _H16_IDLE), "slice", "ab"),
    (_mixed("IL1.IL1.Bsupplied", _IL1, _IL1, _IL1, output_tensor=_B16), "slice", "ab"),
    (_mixed("IL1.IL1.B12", _IL1, _IL1, _B12, shape=(12 * 32, 16 * 32)), "c", "c"),
    # An output shard beside DRAM inputs. The legacy binary_ng borrows nothing, so it uses all 32 clusters, and an
    # anchor on the output's fewer cores is not taken: every operand moves over the NoC, the output through its
    # sharding-aware accessor. I.I.B4dx crosses (N, C) slabs; I.I.Hidle's grid holds 4 shards on 8 cores. An output
    # shard on all 32 clusters stays the anchor, and the inputs move in its order: a width walk, a block walk across
    # (N, C) slabs, a block walk in column-major core order, and a block walk that skips pad slots.
    (_mixed("I.I.H", _IDRAM, _IDRAM, _H16), None, ""),
    (_mixed("I.I.B", _IDRAM, _IDRAM, _B16), None, ""),
    (_mixed("I.I.W", _IDRAM, _IDRAM, _W16), None, ""),
    (_mixed("I.I.Bcol", _IDRAM, _IDRAM, _B16_COL, shape=(16 * 32, 8 * 32)), None, ""),
    (_mixed("I.I.B4d", _IDRAM, _IDRAM, _B16, shape=(2, 2, 4 * 32, 16 * 32)), None, ""),
    (_mixed("I.I.B4dx", _IDRAM, _IDRAM, _B16_SLABS, shape=(2, 2, 2 * 32, 16 * 32)), None, ""),
    (_mixed("I.I.Hidle", _IDRAM, _IDRAM, _H16_IDLE), None, ""),
    (_mixed("I.I.W32", _IDRAM, _IDRAM, _W32, shape=(4 * 32, 64 * 32)), "c", "c"),
    (_mixed("I.I.B32x4d", _IDRAM, _IDRAM, _B32, shape=(2, 4, 2 * 32, 32 * 32)), "c", "c"),
    (_mixed("I.I.B32col", _IDRAM, _IDRAM, _B32_COL, shape=(32 * 32, 16 * 32)), "c", "c"),
    (_mixed("I.I.B32u", _IDRAM, _IDRAM, _B32, shape=(14 * 32, 30 * 32)), "c", "c"),
    # One input shares the output's shard config. The legacy binary_ng borrows nothing, as a and b differ, so the
    # program keeps all 32 clusters: an L1-interleaved input's slices are borrowed, and every other operand moves over
    # the NoC. H.I.H is an L1-sharded a beside a DRAM b, whose default output takes a's memory config. Halt.H.H holds
    # a's shards on another column; Dw.H.H reads a DRAM shard; I.H.H_again repeats I.H.H; H.I.H_in_place adds into a. On
    # all 32 clusters the shared config stays the anchor: H32.I.H32 borrows a and the output.
    (_mixed("I.H.H", _IDRAM, _H16, _H16), None, ""),
    (_mixed("H.I.H", _H16, _IDRAM, _H16), None, ""),
    (_mixed("IL1.H.H", _IL1, _H16, _H16), "slice", "a"),
    (_mixed("IL1.B.B", _IL1, _B16, _B16), "slice", "a"),
    (_mixed("Halt.H.H", ("height", [4 * 32, 16 * 32], (1, 0, 1, 3)), _H16, _H16), None, ""),
    (_mixed("I.W.W", _IDRAM, _W16, _W16), None, ""),
    (_mixed("Dw.H.H", ("dram_width", [16 * 32, 8 * 32], None), _H16, _H16), None, ""),
    (_mixed("I.H.H_again", _IDRAM, _H16, _H16), None, ""),
    (_mixed("H.I.H_in_place", _H16, _IDRAM, _H16, in_place=True), None, ""),
    (_mixed("H32.I.H32", _H32, _IDRAM, _H32, shape=(32 * 32, 40 * 32)), "c", "ac"),
    # The inputs share a shard config. Where the output is sharded on their grid, the legacy binary_ng runs there, and
    # the inputs' anchor borrows them while the output moves in their order. In H.H.B@same-spec one ShardSpec places the
    # height inputs and the block output on other cores. Elsewhere the legacy binary_ng uses all 32 clusters: W.W.I
    # moves all three, and H.H.IL1 borrows the output's slices. On all 32 clusters the inputs stay the anchor, and the
    # output moves in their order: a block walk, and a block walk that skips pad slots.
    (_mixed("H.H.W@same-grid", _H16, _H16, ("width", [16 * 32, 4 * 32], (0, 0, 0, 3))), "a", "ab"),
    (
        _mixed(
            "H.H.B@same-spec",
            ("height", [8 * 32, 16 * 32], (0, 0, 1, 1)),
            ("height", [8 * 32, 16 * 32], (0, 0, 1, 1)),
            ("block", [8 * 32, 16 * 32], (0, 0, 1, 1)),
        ),
        "a",
        "ab",
    ),
    (_mixed("W.W.I", _W16, _W16, _IDRAM), None, ""),
    (_mixed("H.H.IL1", _H16, _H16, _IL1), "slice", "c"),
    (_mixed("B32.B32.I", _B32, _B32, _IDRAM, shape=(16 * 32, 32 * 32)), "a", "ab"),
    (_mixed("B32u.B32u.I", _B32, _B32, _IDRAM, shape=(14 * 32, 30 * 32)), "a", "ab"),
    # Three placements: the legacy binary_ng borrows nothing, so all three move over the NoC on 32 clusters.
    (_mixed("H.W.B", _H16, _W16, _B16), None, ""),
    (_mixed("W.B.H", _W16, _B16, _H16), None, ""),
    (_mixed("B.H.W", _B16, _H16, _W16), None, ""),
    (
        _mixed(
            "H@g0.W@g1.B@g2",
            _H16,
            ("width", [16 * 32, 4 * 32], (0, 1, 3, 1)),
            ("block", [8 * 32, 8 * 32], (2, 2, 3, 3)),
        ),
        None,
        "",
    ),
    # An L1-interleaved output beside inputs that are not slices: the output's slices are borrowed on all 32
    # clusters, and both inputs move in bank order.
    (_mixed("I.I.IL1", _IDRAM, _IDRAM, _IL1), "slice", "c"),
    # A sharded input beside an interleaved output. The legacy binary_ng borrows nothing, so a shard on fewer than 32
    # clusters is not an anchor, and all three move over the NoC: H.I.I, W.I.I, and H.H'.I, whose inputs hold their
    # shards on two columns. In H.I.I@sub-grid a's shards lie outside the requested grid, so the op takes the NoC path
    # on that grid.
    (_mixed("H.I.I", ("height", [8 * 32, 40 * 32], (0, 0, 0, 3)), _IDRAM, _IDRAM, shape=(32 * 32, 40 * 32)), None, ""),
    (_mixed("W.I.I", ("width", [32 * 32, 10 * 32], (0, 0, 3, 0)), _IDRAM, _IDRAM, shape=(32 * 32, 40 * 32)), None, ""),
    (
        _mixed(
            "H.H'.I",
            ("height", [8 * 32, 32 * 32], (0, 0, 0, 3)),
            ("height", [8 * 32, 32 * 32], (1, 0, 1, 3)),
            _IDRAM,
            shape=(32 * 32, 32 * 32),
        ),
        None,
        "",
    ),
    (
        _mixed(
            "H.I.I@sub-grid",
            ("height", [8 * 32, 40 * 32], (4, 0, 4, 3)),
            _IDRAM,
            _IDRAM,
            shape=(32 * 32, 40 * 32),
            grid=(0, 0, 3, 3),
        ),
        None,
        "",
    ),
    # No memory_config: the op derives the output's, as the legacy binary_ng does. It takes a's config (adjusted to the
    # output shape), or b's where a is in DRAM and b is sharded, or where b's shard grid has more cores. A
    # height-sharded config without a shard spec takes a's shard spec, or, beside interleaved inputs, a generated one.
    # The anchor rule then applies as with an explicit config.
    (_mixed("H.I.derived", _H16, _IDRAM, None, resolved_out=_H16), None, ""),
    (_mixed("I.H.derived", _IDRAM, _H16, None, resolved_out=_H16), None, ""),
    (_mixed("IL1.H.derived", _IL1, _H16, None, resolved_out=_IL1), "slice", "ac"),
    (_mixed("H.B.derived", _H16, _B16, None, resolved_out=_B16), None, ""),
    (_mixed("IL1.IL1.derived", _IL1, _IL1, None, resolved_out=_IL1), "slice", "abc"),
    (
        _mixed("I.I.specless", _IDRAM, _IDRAM, "height_specless", resolved_out=("height", [32, 16 * 32], (0, 0, 7, 1))),
        None,
        "",
    ),
    (_mixed("H.I.specless", _H16, _IDRAM, "height_specless", resolved_out=_H16), None, ""),
    # IL1.IL1.B9: the legacy binary_ng borrows the output on its 9 clusters, but 4 divides neither its 9 tiles per
    # cluster nor the 3 per bank. The DFB cannot borrow such a ring beside operands that move, so all three move over
    # the NoC, on 32 clusters.
    (_mixed("IL1.IL1.B9", _IL1, _IL1, ("block", [3 * 32, 3 * 32], (0, 0, 2, 2)), shape=(9 * 32, 9 * 32)), None, ""),
    # Pad slots beside operands that move, which the walks must skip. A pad row maps to a page past the tensor,
    # and a pad column of a block or width shard to a REAL page of the next row, so a write there would replace
    # a right tile. B32u.B32u.I and I.I.B32u, above, skip the pad rows and pad columns of a block shard on all 32
    # clusters, in the writer and in the reader. IL1.IL1.Bu leaves 28 of its banks one page short, beside a block
    # output with short corner shards. In the _short cases 28 banks hold 3 pages of 4, beside a DRAM operand that
    # moves. Hu.I.I, Bu.I.I, I.I.Bu and Wu.I.I hold their uneven shards on fewer than 32 clusters, so all three
    # operands move over the NoC.
    (_mixed("IL1.IL1.Bu", _IL1, _IL1, _B10_UNEVEN, shape=(10 * 32, 10 * 32)), "slice", "ab"),
    (_mixed("Hu.I.I", _H16, _IDRAM, _IDRAM, shape=(14 * 32, 16 * 32)), None, ""),
    (_mixed("Bu.I.I", _B10_UNEVEN, _IDRAM, _IDRAM, shape=(10 * 32, 10 * 32)), None, ""),
    (_mixed("I.I.Bu", _IDRAM, _IDRAM, _B10_UNEVEN, shape=(10 * 32, 10 * 32)), None, ""),
    (_mixed("Wu.I.I", ("width", [4 * 32, 4 * 32], (0, 0, 2, 0)), _IDRAM, _IDRAM, shape=(4 * 32, 10 * 32)), None, ""),
    (_mixed("IL1.I.IL1_short", _IL1, _IDRAM, _IL1, shape=(10 * 32, 10 * 32)), "slice", "ac"),
    (_mixed("IL1.IL1.I_short", _IL1, _IL1, _IDRAM, shape=(10 * 32, 10 * 32)), "slice", "ab"),
    (_mixed("I.I.IL1_short", _IDRAM, _IDRAM, _IL1, shape=(10 * 32, 10 * 32)), "slice", "c"),
    # Shapes that are not a multiple of 32, so the last tile row or tile column is partial. The op counts, places
    # and walks the tiles of the padded shape, and derives the output config from it, so each case takes the
    # anchor of its padded shape. The cases cover each anchor kind and each walk, in the reader and in the writer.
    # In I.I.B4d_partial every (N, C) slab ends in a partial tile row, inside the tensor. B32u.B32u.I_partial also
    # has pad slots.
    (_mixed("I.I.I_partial", _IDRAM, _IDRAM, _IDRAM, shape=_PARTIAL16), None, ""),
    (_mixed("IL1.IL1.IL1_partial", _IL1, _IL1, _IL1, shape=(16 * 32 - 1, 16 * 32 - 1)), "slice", "abc"),
    (_mixed("IL1.IL1.I_partial", _IL1, _IL1, _IDRAM, shape=_PARTIAL16), "slice", "ab"),
    (_mixed("I.I.IL1_partial", _IDRAM, _IDRAM, _IL1, shape=_PARTIAL16), "slice", "c"),
    (_mixed("B.B.B_partial", _B16, _B16, _B16, shape=_PARTIAL16), "c", "abc"),
    (_mixed("B32.B32.I_partial", _B32, _B32, _IDRAM, shape=(16 * 32, 32 * 32 - 31)), "a", "ab"),
    (_mixed("I.I.W32_partial", _IDRAM, _IDRAM, _W32, shape=(4 * 32 - 31, 64 * 32)), "c", "c"),
    (_mixed("I.I.B4d_partial", _IDRAM, _IDRAM, _B16, shape=(2, 2, 4 * 32 - 5, 16 * 32 - 9)), None, ""),
    (_mixed("B32u.B32u.I_partial", _B32, _B32, _IDRAM, shape=(14 * 32 - 5, 30 * 32 - 9)), "a", "ab"),
    (_mixed("H.I.derived_partial", _H16, _IDRAM, None, shape=_PARTIAL16, resolved_out=_H16), None, ""),
    (
        _mixed(
            "I.I.specless_partial",
            _IDRAM,
            _IDRAM,
            "height_specless",
            shape=_PARTIAL16,
            resolved_out=("height", [32, 16 * 32], (0, 0, 7, 1)),
        ),
        None,
        "",
    ),
]
_MIXED_CASES = [_anchored(entry, anchor, borrowed) for entry, anchor, borrowed in _MIXED]

# The mixed cases with pad slots: an uneven shard or a short bank in the anchor. The split of every other case must
# hold exactly its output tiles.
_MIXED_PAD_SLOTS = {
    "I.I.B32u",
    "B32u.B32u.I",
    "IL1.IL1.Bu",
    "IL1.I.IL1_short",
    "IL1.IL1.I_short",
    "I.I.IL1_short",
    "B32u.B32u.I_partial",
}

# R1C4W1-n1 puts one reader thread on four tile counters, so the shard walk takes the counter-major steps.
# Beside DRAM the rule batches only all-interleaved operands, so rule-dm8-n1 sets a DM batch of 8 for these cases.
_MIXED_CONFIGS = {
    "rule-n1": _COMPUTE_BATCH_1,
    "rule-dm8-n1": {**_DM_BATCH_8, **_COMPUTE_BATCH_1},
    "R1C1W1": _threads(1, 1, 1),
    "R1C4W1-n1": {**_threads(1, 4, 1), **_COMPUTE_BATCH_1},
    "R4C4W2-per-tile": {**_threads(4, 4, 2), **_PER_TILE},
    "rule": {},
}


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", _config_params(_MIXED_CONFIGS, _MIXED_CASES))
def test_native_mixed_layouts_match_the_prediction(name, knobs):
    """Every mix of placements must run native and give the predicted output.

    The fallback binds kernels_dfb, so a refusing gate fails the routing check. A tile read or written in
    the wrong order, or a borrowed ring paired with a NoC ring of another size in the wrong order, fails the
    comparison, which is exact. A cached program that keeps the first call's addresses fails the repeat.
    """
    p = _run_shard_arm([case[0] for case in _MIXED_CASES], expect="qsr", timeout=1800, knobs=knobs)
    _check_arm(p, f"mixed layouts, {name}", expect_wrong=_expect_wrong(_MIXED_CASES, knobs))


# The ops other than add. subtract, and multiply in fast mode, run the FPU kernel, as add does; multiply and divide
# run the SFPU kernel. Each op runs one mix of each kind: the output's shard walked by the reader, the inputs' shard
# walked by the writer, the bank walk in the reader and in the writer, pad slots in the reader and in the writer, all
# three borrowed as slices with partial tiles, all three borrowed as shards with 4 tiles per compute thread, and the
# NoC path beside DRAM. Under the rule, the shards hold 4 tiles per compute thread, so the SFPU kernel runs one full
# batch per thread. Every output is exact.
_OTHER_OPS = ("subtract", "multiply", "multiply_fast", "divide")
_OP_MIXES = (
    "I.I.W32",
    "B32.B32.I",
    "I.I.IL1",
    "IL1.IL1.H",
    "I.I.B32u",
    "B32u.B32u.I",
    "IL1.IL1.IL1_partial",
    "B.B.B_partial",
    "I.I.I_partial",
)


def _with_op(entry, op):
    """A _SHARD_ARM_SRC entry that runs `op`, named after it."""
    opts = entry[5] if len(entry) > 5 else {}
    return (f"{entry[0]}/{op}", *entry[1:5], {**opts, "op": op})


_OP_CASES = [
    (_with_op(entry, op), path, split)
    for op in _OTHER_OPS
    for entry, path, split in _MIXED_CASES
    if entry[0] in _OP_MIXES
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", _config_params(_MIXED_CONFIGS, _OP_CASES))
def test_native_other_ops_match_the_prediction(name, knobs):
    """subtract, multiply and divide must run native in every kind of mix and give the predicted output, exactly as
    each kernel rounds (OPS in _SHARD_ARM_SRC), so a wrong op define or DST slot fails."""
    p = _run_shard_arm([case[0] for case in _OP_CASES], expect="qsr", timeout=1800, knobs=knobs)
    _check_arm(p, f"subtract, multiply and divide, {name}", expect_wrong=_expect_wrong(_OP_CASES, knobs))


# Borrowed shards that 4 does not divide, so that their last tiles go through the tail rings: block_6 has a
# borrowed part and a tail, height1 only a tail. At 1,1,1 height_5 runs a full SFPU batch and a short one.
_OP_TAIL_SHARDS = [
    _with_op(entry, op)
    for op in _OTHER_OPS
    for entry in _SMALL_SHARDS
    if entry[0] in ("block_6", "height1", "height_5")
]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", list(_BORROWED_CONFIGS_C2.items()), ids=list(_BORROWED_CONFIGS_C2))
def test_native_other_ops_on_the_tail_rings_are_bit_exact(name, knobs):
    """subtract, multiply and divide must stay native and exact where a shard's last tiles go through the tail
    rings."""
    p = _run_shard_arm(_OP_TAIL_SHARDS, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"subtract, multiply and divide on the tail rings, {name}")


# Activations: SQUARE on a, on b or on the result; a lone RELU on the result, which the packer applies; and RELU on a,
# or on the result beside a chain, which runs on the SFPU. Each runs on the FPU kernel (add) and on the SFPU kernel
# (multiply), in one mix of each kind of borrowing: nothing borrowed, beside DRAM with partial tiles; all three
# borrowed; the inputs borrowed beside an output that moves; the output borrowed beside inputs that move. SQUARE on both
# inputs runs two self-loop rings on each Neo. block_6's shards hold 6 tiles, which 4 does not divide. The tail rings
# carry no activation, so at the rule's compute count block_6 takes the NoC path.
# SQUARE and RELU have exact models; GELU and TANH do not. GELU also shifts the SFPU results of the later programs on
# the same device, on both factories.
_ACTIVATIONS = ("lhs", "rhs", "post", "relu", "relu_lhs", "relu_chain")
_ACT_MIXES = ("I.I.I_partial", "B.B.B_partial", "B32.B32.I", "I.I.W32")
_BLOCK_6 = next(entry for entry in _SMALL_SHARDS if entry[0] == "block_6")


def _with_act(entry, act):
    """A _SHARD_ARM_SRC entry that chains the activation `act`, named after it."""
    opts = entry[5] if len(entry) > 5 else {}
    return (f"{entry[0]}/{act}", *entry[1:5], {**opts, "act": act})


_ACT_CASES = (
    [
        (_with_act(_with_op(entry, op), act), path, split)
        for op in ("add", "multiply")
        for act in _ACTIVATIONS
        for entry, path, split in _MIXED_CASES
        if entry[0] in _ACT_MIXES
    ]
    + [
        (_with_act(_with_op(entry, op), "both"), path, split)
        for op in ("add", "multiply")
        for entry, path, split in _MIXED_CASES
        if entry[0] == "B.B.B_partial"
    ]
    + [
        (_with_act(_with_op(_BLOCK_6, op), act), _NOC_L1, _noc_split(_tensor_tiles(_BLOCK_6)))
        for op in ("add", "multiply")
        for act in ("lhs", "relu")
    ]
)

# The activations change only the compute side: one compute thread at the rule's batch, four at batch 1, and the
# rule, which batches the SFPU kernel at 4 and the FPU kernel at 1, both exact.
_ACT_CONFIGS = {name: _MIXED_CONFIGS[name] for name in ("R1C1W1", "rule-n1", "rule")}


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", _config_params(_ACT_CONFIGS, _ACT_CASES))
def test_native_activations_match_the_prediction(name, knobs):
    """Activations on a, on b and on the result must run native and give the predicted output.

    The oracle applies each activation as the kernels do: SQUARE truncates its fp32 result to bf16, and RELU clips
    the rounded result. A self-loop ring too small for its compute threads fails the program, and a tail ring
    beside an activation fails to compile.
    """
    p = _run_shard_arm([case[0] for case in _ACT_CASES], expect="qsr", timeout=1800, knobs=knobs)
    _check_arm(p, f"activations, {name}", expect_wrong=_expect_wrong(_ACT_CASES, knobs))


# fp32: add, subtract, multiply and divide all run the SFPU kernel, in a 32-bit DST, where the rule's compute batch is 2.
# They run the op mixes, the shards whose last tiles go through the tail rings, and three activations: SQUARE on a and
# SQUARE on b, which run the Float32 activation rings of a and of b, and a lone RELU on the result, in the packer.
_FP32_OPS = ("add", "subtract", "multiply", "divide")


def _with_fp32(entry):
    """A _SHARD_ARM_SRC entry on fp32 operands and output, named after it."""
    opts = entry[5] if len(entry) > 5 else {}
    return (f"{entry[0]}/fp32", *entry[1:5], {**opts, "dtype": "float32"})


_FP32_CASES = [
    (_with_fp32(_with_op(entry, op)), path, split)
    for op in _FP32_OPS
    for entry, path, split in _MIXED_CASES
    if entry[0] in _OP_MIXES
]
_FP32_TAIL_SHARDS = [
    _with_fp32(_with_op(entry, op))
    for op in _FP32_OPS
    for entry in _SMALL_SHARDS
    if entry[0] in ("block_6", "height1", "height_5")
]
_FP32_ACT_CASES = [
    (_with_act(_with_fp32(_with_op(entry, op)), act), path, split)
    for op in ("add", "multiply")
    for act in ("lhs", "rhs", "relu")
    for entry, path, split in _MIXED_CASES
    if entry[0] in _ACT_MIXES
]
_FP32_CONFIGS = {name: _MIXED_CONFIGS[name] for name in ("R1C1W1", "rule-n1", "rule")}
# The configurations that have tail rings: at C = 1 every shard divides, and R4C4W2-n1 builds rule-n1's program.
_FP32_TAIL_CONFIGS = {name: _BORROWED_CONFIGS_C2[name] for name in ("rule-n1", "R1C2W1-n1")}


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", _config_params(_FP32_CONFIGS, _FP32_CASES))
def test_native_fp32_ops_match_the_prediction(name, knobs):
    """fp32 add, subtract, multiply and divide must run native in every kind of mix and give the predicted output.

    The oracle is torch's fp32 result, within _ULP. A batch larger than a
    32-bit DST section holds overwrites the section the packer reads.
    """
    p = _run_shard_arm([case[0] for case in _FP32_CASES], expect="qsr", timeout=1800, knobs=knobs)
    _check_arm(p, f"fp32 ops, {name}", expect_wrong=_expect_wrong(_FP32_CASES, knobs))


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", list(_FP32_TAIL_CONFIGS.items()), ids=list(_FP32_TAIL_CONFIGS))
def test_native_fp32_on_the_tail_rings_match_the_oracle(name, knobs):
    """fp32 ops must stay native and match the fp32 oracle where a shard's last tiles go through the tail rings."""
    p = _run_shard_arm(_FP32_TAIL_SHARDS, expect="qsr", timeout=900, knobs=knobs)
    _check_arm(p, f"fp32 ops on the tail rings, {name}")


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name, knobs", _config_params(_ACT_CONFIGS, _FP32_ACT_CASES))
def test_native_fp32_activations_match_the_prediction(name, knobs):
    """Activations on fp32 operands must run native and give the predicted output."""
    p = _run_shard_arm([case[0] for case in _FP32_ACT_CASES], expect="qsr", timeout=1800, knobs=knobs)
    _check_arm(p, f"fp32 activations, {name}", expect_wrong=_expect_wrong(_FP32_ACT_CASES, knobs))


# One case per child, as above, and one case of each kind of anchor and walk. The thread counts tell the
# programs apart where the clusters do not, as both bind the same kernel sources.
_MIXED_ROUTED = (
    "IL1.IL1.H",
    "IL1.IL1.Hidle",
    "I.I.H",
    "I.I.B",
    "I.I.Hidle",
    "I.H.H",
    "H.I.H",
    "IL1.B.B",
    "H.H.W@same-grid",
    "H.H.B@same-spec",
    "W.W.I",
    "H.W.B",
    "H.I.I",
    "W.I.I",
    "H.H'.I",
    "IL1.IL1.B9",
    "IL1.IL1.Bu",
    "Hu.I.I",
    "Bu.I.I",
    "I.I.Bu",
    "IL1.IL1.I_short",
    "I.I.IL1",
    "H.I.I@sub-grid",
    "H.I.derived",
    "I.H.derived",
    "IL1.H.derived",
    "H.B.derived",
    "IL1.IL1.derived",
    "I.I.specless",
    "H.I.specless",
    "I.I.I_partial",
    "IL1.IL1.IL1_partial",
    "IL1.IL1.I_partial",
    "I.I.IL1_partial",
    "B.B.B_partial",
    "B32.B32.I_partial",
    "I.I.W32_partial",
    "I.I.B4d_partial",
    "B32u.B32u.I_partial",
    "H.I.derived_partial",
    "I.I.specless_partial",
    "IL1.IL1.B12",
    "I.I.W32",
    "I.I.B32x4d",
    "I.I.B32col",
    "I.I.B32u",
    "H32.I.H32",
    "B32.B32.I",
    "B32u.B32u.I",
    "IL1.H.H",
    "H.H.IL1",
)
_MIXED_ROUTES = [(case[0], *_route(case, _COMPUTE_BATCH_1)) for case in _MIXED_CASES if case[0][0] in _MIXED_ROUTED]


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("case, dms, cores", _MIXED_ROUTES, ids=[r[0][0] for r in _MIXED_ROUTES])
def test_native_mixed_layouts_route_by_the_anchor(case, dms, cores):
    """Each mix must run where its anchor puts it, with the threads of what it moves.

    A program on other clusters than the anchor's fails, and so does one that borrows fewer operands than the
    anchor allows where that adds reader or writer threads. A borrowed input beside one that moves adds none, so
    there only the clusters show. The child reads the DM cores and the Neos per cluster, and the number of
    clusters, from the device profiler.
    """
    p = _run_shard_arm(
        [case],
        expect="qsr",
        timeout=900,
        knobs={
            **_COMPUTE_BATCH_1,
            "TT_METAL_DEVICE_PROFILER": "1",
            "ARM_EXPECT_DMS": str(dms),
            "ARM_EXPECT_NEOS": "4",
            "ARM_EXPECT_CORES": str(cores),
        },
    )
    _check_arm(p, f"{case[0]} with {dms} DM cores on {cores} clusters")


# --- The rule and its overrides ----------------------------------------------------------------------------

_DRAM_128 = ("dram_128", "dram", None, None, (4 * 32, 32 * 32))


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
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


# Each case: the knobs, the values each program shape's part of the settings line must name with their
# sources, and whether the output must be exact. The line gives five shapes: every operand over the NoC and in
# L1, the same beside a DRAM operand (batch 1 in the line), the output borrowed beside an input that
# moves, the inputs borrowed beside an output that moves, all of them borrowed. With no knob set, the rule runs the
# FPU kernel at batch 1 above ring stride 1, and the SFPU kernel at 4 and 2. C = 1 alone gives 1,1,1 by the
# rule. R = 2 beside C = 1 makes each compute thread own two tile counters of the input rings, so the rule's
# compute batch drops to 1 where the inputs move. That case checks the line only: its output is exact only on
# craq-sim builds without the defect that corrupts a run whose DM side outnumbers its Tensix side. A hang still
# fails it.
_NOC_SHAPE = "the operands move over the NoC, all in L1"
_DRAM_SHAPE = "an operand is in DRAM"
_OUTPUT_SHAPE = "the output is borrowed and an input moves"
_INPUTS_SHAPE = "the inputs are borrowed and the output moves"
_ALL_SHAPE = "all of them are borrowed"
_SFPU_SHAPE = "the SFPU kernel runs, all in L1 over the NoC"
_LOG_CASES = {
    "W1-n1": (
        {"TTNN_QSR_WRITER_THREADS": "1", **_COMPUTE_BATCH_1},
        {
            _NOC_SHAPE: [
                "R=4 (rule)",
                "C=4 (rule)",
                "W=1 (TTNN_QSR_WRITER_THREADS)",
                "entries_per_thread=16 (rule)",
                "dm_batch=8 (rule)",
                "tiles_per_cycle=1 (TTNN_QSR_TILES_PER_CYCLE)",
            ],
            _DRAM_SHAPE: ["entries_per_thread=2 (rule) dm_batch=1 (rule) tiles_per_cycle=1 (TTNN_QSR_TILES_PER_CYCLE)"],
            _OUTPUT_SHAPE: ["R=4 (rule) W=1 tiles_per_cycle=1 (TTNN_QSR_TILES_PER_CYCLE)"],
            _INPUTS_SHAPE: ["R=1 W=1 (TTNN_QSR_WRITER_THREADS) tiles_per_cycle=1 (TTNN_QSR_TILES_PER_CYCLE)"],
            _ALL_SHAPE: ["R=1 W=1 tiles_per_cycle=1 (TTNN_QSR_TILES_PER_CYCLE)"],
            _SFPU_SHAPE: [
                "tiles_per_cycle=1 (TTNN_QSR_TILES_PER_CYCLE) in a 16-bit DST",
                "tiles_per_cycle=1 (TTNN_QSR_TILES_PER_CYCLE) in a 32-bit DST",
            ],
        },
        True,
    ),
    "rule": (
        {},
        {
            _NOC_SHAPE: [
                "R=4 (rule)",
                "C=4 (rule)",
                "W=2 (rule)",
                "entries_per_thread=16 (rule)",
                "dm_batch=8 (rule)",
                "tiles_per_cycle=1 (rule)",
            ],
            _DRAM_SHAPE: ["entries_per_thread=2 (rule) dm_batch=1 (rule) tiles_per_cycle=1 (rule)"],
            _OUTPUT_SHAPE: ["R=4 (rule) W=1 tiles_per_cycle=1 (rule)"],
            _INPUTS_SHAPE: ["R=1 W=4 (rule) tiles_per_cycle=1 (rule)"],
            _ALL_SHAPE: ["R=1 W=1 tiles_per_cycle=1 (rule)"],
            _SFPU_SHAPE: ["tiles_per_cycle=4 (rule) in a 16-bit DST", "tiles_per_cycle=2 (rule) in a 32-bit DST"],
        },
        True,
    ),
    "C1": (
        {"TTNN_QSR_COMPUTE_THREADS": "1"},
        {
            _NOC_SHAPE: ["R=1 (rule)", "C=1 (TTNN_QSR_COMPUTE_THREADS)", "W=1 (rule)", "tiles_per_cycle=8 (rule)"],
            _DRAM_SHAPE: ["entries_per_thread=2 (rule) dm_batch=1 (rule) tiles_per_cycle=1 (rule)"],
            _OUTPUT_SHAPE: ["R=1 (rule) W=1 tiles_per_cycle=8 (rule)"],
            _INPUTS_SHAPE: ["R=1 W=1 (rule) tiles_per_cycle=8 (rule)"],
            _ALL_SHAPE: ["R=1 W=1 tiles_per_cycle=8 (rule)"],
            _SFPU_SHAPE: ["tiles_per_cycle=4 (rule) in a 16-bit DST", "tiles_per_cycle=2 (rule) in a 32-bit DST"],
        },
        True,
    ),
    "R2C1": (
        {"TTNN_QSR_READER_THREADS": "2", "TTNN_QSR_COMPUTE_THREADS": "1"},
        {
            _NOC_SHAPE: [
                "R=2 (TTNN_QSR_READER_THREADS)",
                "C=1 (TTNN_QSR_COMPUTE_THREADS)",
                "W=1 (rule)",
                "tiles_per_cycle=1 (rule)",
            ],
            _DRAM_SHAPE: ["entries_per_thread=2 (rule) dm_batch=1 (rule) tiles_per_cycle=1 (rule)"],
            _OUTPUT_SHAPE: ["R=2 (TTNN_QSR_READER_THREADS) W=1 tiles_per_cycle=1 (rule)"],
            _INPUTS_SHAPE: ["R=1 W=1 (rule) tiles_per_cycle=8 (rule)"],
            _ALL_SHAPE: ["R=1 W=1 tiles_per_cycle=8 (rule)"],
            _SFPU_SHAPE: ["tiles_per_cycle=1 (rule) in a 16-bit DST", "tiles_per_cycle=1 (rule) in a 32-bit DST"],
        },
        False,
    ),
}


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
@pytest.mark.parametrize("name", list(_LOG_CASES))
def test_native_tuning_log_names_each_source(name):
    """The line that reports the native settings must name the source of each value, the knob or the rule.

    An A/B log is then self-describing: a value the rule chose cannot be read as a value someone set. Each
    program shape's values are checked in its own part of the line, so one shape cannot satisfy another's.
    The line is an info message, so the arm sets the logger to info.
    """
    knobs, expected, exact = _LOG_CASES[name]
    p = _run_arm([32], knobs={**knobs, "TT_LOGGER_LEVEL": "info"}, timeout=300)
    if exact:
        _check_arm(p, f"the logged configuration, {name}")
    else:
        assert p.returncode in (0, 2), f"{name} failed:\n{p.stdout}\n{p.stderr[-2000:]}"
    log = p.stdout + p.stderr
    line = next((ln for ln in log.splitlines() if "Quasar-native ENABLED" in ln), "")
    parts = {part.split(":", 1)[0]: part for part in line.split("Where ")[1:]}
    missing = [
        (shape, token) for shape, tokens in expected.items() for token in tokens if token not in parts.get(shape, "")
    ]
    assert not missing, f"the native settings line lacks {missing}:\n{line}"


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
def test_native_illegal_thread_knobs_fall_back():
    """Set thread counts that break a ring's STRIDED ratio must send the op to the fallback factory.

    R = 3 beside C = 4 gives the input rings max(R, C) % min(R, C) = 1, which the DFB host refuses. The gate
    checks the ratio on the NoC path's counts, so the op runs on the Metal 2.0 factory, and runs right.
    """
    p = _run_shard_arm([_DRAM_128], expect="dfb", timeout=900, knobs=_threads(3, 4, 2))
    _check_arm(p, "R=3 beside C=4")


@pytest.mark.skipif(not _native_enabled(), reason="native factory off: TTNN_QSR_NATIVE=0, or not Quasar")
def test_native_prediction_tables_cover_their_cases():
    """Each prediction must describe its whole case, and every case must have one. No device.

    A split that misses tiles or counts a pad slot, a borrowed shard that 4 does not divide, or an
    L1-interleaved case that is neither in the rule table nor a pad case would make a prediction wrong, or
    silently leave a case out of the rule column.
    """
    cases = _BORROWED_SHARD_CASES + _BATCHED_BORROWED_CASES + _L1_INTERLEAVED_RULE_CASES + _DRAM_SHARDED_CASES
    # A pad slot takes a batch position but is not in the output, so a mixed case with pad slots must be exact in
    # every configuration: at most one tile per compute thread, or a DRAM operand.
    padded = [case for case in _MIXED_CASES if case[0][0] in _MIXED_PAD_SLOTS]
    unknown = _MIXED_PAD_SLOTS - {case[0][0] for case in _MIXED_CASES}
    assert not unknown, f"these cases with pad slots are not in the mixed table: {unknown}"
    no_pads = [case[0][0] for case in padded if sum(case[2]) == _tensor_tiles(case[0])]
    assert not no_pads, f"these cases are listed with pad slots, but their split has none: {no_pads}"
    cases += [case for case in _MIXED_CASES if case not in padded]
    inexact = [case[0][0] for case in padded for knobs in _MIXED_CONFIGS.values() if _expect_wrong([case], knobs)]
    assert not inexact, f"these cases with pad slots have a wrong-tile prediction: {inexact}"
    # The op table reuses mixed placements, with the SFPU kernel's own batch, so the same rules hold for it.
    op_padded = [case for case in _OP_CASES if case[0][0].split("/")[0] in _MIXED_PAD_SLOTS]
    op_inexact = [case[0][0] for case in op_padded for knobs in _MIXED_CONFIGS.values() if _expect_wrong([case], knobs)]
    assert not op_inexact, f"these op cases with pad slots have a wrong-tile prediction: {op_inexact}"
    missing_ops = set(_OP_MIXES) - {case[0][0].split("/")[0] for case in _OP_CASES}
    assert not missing_ops, f"these op mixes are not in the mixed table: {missing_ops}"
    cases += [case for case in _OP_CASES if case not in op_padded]
    # The activation table reuses mixed placements without pad slots, and block_6 on the NoC path.
    missing_acts = set(_ACT_MIXES) - {case[0][0].split("/")[0] for case in _ACT_CASES}
    assert not missing_acts, f"these activation mixes are not in the mixed table: {missing_acts}"
    padded_acts = set(_ACT_MIXES) & _MIXED_PAD_SLOTS
    assert not padded_acts, f"these activation mixes have pad slots: {padded_acts}"
    cases += _ACT_CASES
    # The fp32 tables reuse the op and activation placements; the ones with pad slots stay out of the split check.
    cases += [case for case in _FP32_CASES if case[0][0].split("/")[0] not in _MIXED_PAD_SLOTS] + _FP32_ACT_CASES
    # In place, a wrong-tile prediction is not verified above a batch of 1: a batch may read other threads'
    # slots, which in place hold their results, not their inputs.
    in_place = [case for case in _MIXED_CASES if len(case[0]) > 5 and case[0][5].get("in_place")]
    racy = [case[0][0] for case in in_place for knobs in _MIXED_CONFIGS.values() if _expect_wrong([case], knobs)]
    assert not racy, f"these in-place cases have a wrong-tile prediction: {racy}"
    uncovered = [entry[0] for entry, _, split in cases if sum(split) != _tensor_tiles(entry)]
    assert not uncovered, f"these splits do not cover their output: {uncovered}"
    misplaced = [entry[0] for entry, path, _ in cases if path[2] != _has_dram(entry)]
    assert not misplaced, f"these paths name the wrong DRAM placement: {misplaced}"
    indivisible = [entry[0] for entry in _BORROWED_SHARDS if _shard_tiles(entry) % _RULE_COMPUTE_THREADS] + [
        entry[0]
        for (entry, anchor, _), (_, _, split) in zip(_MIXED, _MIXED_CASES)
        if anchor and split[0] % _RULE_COMPUTE_THREADS
    ]
    assert not indivisible, f"these borrowed shards leave tail tiles under the rule: {indivisible}"
    # A shard anchor borrows the operands placed exactly as it is, and the slice anchor the L1-interleaved ones.
    unlike = [
        entry[0]
        for entry, anchor, borrowed in _MIXED
        if anchor is not None
        and any(
            _placement(entry, operand) != (_IL1 if anchor == "slice" else _placement(entry, anchor))
            for operand in borrowed
        )
    ]
    assert not unlike, f"these cases borrow an operand that their anchor does not hold: {unlike}"
    left_out = {entry[0] for entry in _L1_INTERLEAVED + _L1_INTERLEAVED_NOC + _L1_INTERLEAVED_PARTIAL} - set(
        _L1_INTERLEAVED_RULE_SPLITS
    )
    assert left_out == {"l1_100_pad", "l1_250_pad"}, f"L1-interleaved cases in no prediction table: {left_out}"
