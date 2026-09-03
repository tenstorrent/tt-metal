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
