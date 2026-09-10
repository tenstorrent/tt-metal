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

import os
import pathlib
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
    # the rest of the suite rather than rolling its own weaker check. bf16 add on the phase-1 slice is
    # in fact bit-exact against torch; Task 3's oracle asserts that stronger property.
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


# --- F1 / Milestone 1.1: uneven tile counts -------------------------------------------------------
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
_ARM_SRC = """
import os, pathlib, sys
import torch, ttnn

R, C, W = {rcw}
TILE_COUNTS = {tile_counts}

failures = []
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
            failures.append("{{}} tiles: {{}} of {{}} elements differ".format(
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

if failures:
    print("FAIL R={{}} C={{}} W={{}}".format(R, C, W))
    for f in failures:
        print("  " + f)
    sys.exit(1)
print("OK R={{}} C={{}} W={{}} over {{}} shapes".format(R, C, W, len(TILE_COUNTS)))
"""


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
    """The whole point of F1 only exists above one thread, so run the ragged sweep on real arms.

    Every test above this one runs at whatever thread counts the invocation happens to carry, and the
    default is 1,1,1 -- where my_tiles == num_tiles, every strided loop has stride 1, and no thread can
    draw zero tiles. Those tests would pass on a build that had never implemented uneven splitting.
    This one sets the counts explicitly, so an empty thread and a ragged tail are actually reached.

    Requests no `device` fixture on purpose: each arm opens its own device inside its own process.
    """
    env = {
        **os.environ,
        "TTNN_QSR_NATIVE": "1",
        "TTNN_QSR_READER_THREADS": str(rcw[0]),
        "TTNN_QSR_COMPUTE_THREADS": str(rcw[1]),
        "TTNN_QSR_WRITER_THREADS": str(rcw[2]),
    }
    src = _ARM_SRC.format(rcw=repr(tuple(rcw)), tile_counts=repr(_RAGGED_TILE_COUNTS))
    p = subprocess.run([sys.executable, "-c", src], env=env, capture_output=True, text=True, timeout=3600)
    assert p.returncode == 0, f"arm R={rcw[0]} C={rcw[1]} W={rcw[2]} failed:\n{p.stdout}\n{p.stderr[-2000:]}"
