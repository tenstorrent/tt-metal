# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf harness for nomic_sdpa: one call per (shape, compute config) for scripts/run_safe_pytest.sh --profile.

Device kernel time is read from the profiler CSV (one generic_op row per call, in parametrize order). Correctness is
the acceptance test's job; this file only checks the output is finite so a broken variant is not timed silently.
"""

import os

import pytest
import torch
import ttnn

import importlib
from eval.golden_tests.nomic_sdpa.helpers import make_qkv, reserve_l1
from ttnn.operations.nomic_sdpa import nomic_sdpa

nomic_sdpa_module = importlib.import_module("ttnn.operations.nomic_sdpa.nomic_sdpa")


@pytest.fixture(autouse=True)
def _knob_overrides(monkeypatch):
    """NOMIC_SDPA_KNOBS="KV_MCAST=0,KV_DEPTH_L1=1" patches the op module's block knobs for an A/B profile run."""
    for item in filter(None, (s.strip() for s in os.environ.get("NOMIC_SDPA_KNOBS", "").split(","))):
        name, value = item.split("=")
        monkeypatch.setattr(nomic_sdpa_module, name, int(value))
    monkeypatch.setattr(nomic_sdpa_module, "_CALL_CACHE", {})


CONFIGS = {  # name: (fidelity, math_approx_mode, fp32_dest_acc_en[, dst_full_sync_en])
    # bf16 DEST + approximate exp is refused at S = 512 (misses the bar); the exact exp is its honored bf16-DEST form.
    "bf16dest_hifi2_exactexp": (ttnn.MathFidelity.HiFi2, False, False),
    "fp32dest_hifi2": (ttnn.MathFidelity.HiFi2, True, True),
    "fp32dest_hifi3": (ttnn.MathFidelity.HiFi3, True, True),
    # fp32 DEST in full-sync mode: 8 DEST tiles instead of 4 (bigger matmul subblocks, no math/pack DEST overlap).
    "fp32dest_hifi2_fullsync": (ttnn.MathFidelity.HiFi2, True, True, True),
}


@pytest.mark.parametrize("buffer", ["l1", "dram"])
@pytest.mark.parametrize("config", list(CONFIGS))
@pytest.mark.parametrize(
    "shape",
    [(2, 512), (4, 512), (8, 512), (1, 128), (1, 512), (8, 384), (16, 256), (32, 128)],
    ids=lambda s: f"{s[0]}x{s[1]}",
)
def test_nomic_sdpa_perf(device, shape, config, buffer):
    batch, seqlen = shape
    # NOMIC_SDPA_SHAPES="8x512,4x512": profile only these shapes (--profile cannot forward a multi-word -k).
    only = [x for x in os.environ.get("NOMIC_SDPA_SHAPES", "").split(",") if x]
    if only and f"{batch}x{seqlen}" not in only:
        pytest.skip("not in NOMIC_SDPA_SHAPES")
    # NOMIC_SDPA_CONFIGS="fp32dest_hifi2" / NOMIC_SDPA_BUFFERS="l1": the same filter for the other two axes.
    for env, value in (("NOMIC_SDPA_CONFIGS", config), ("NOMIC_SDPA_BUFFERS", buffer)):
        allowed = [x for x in os.environ.get(env, "").split(",") if x]
        if allowed and value not in allowed:
            pytest.skip(f"not in {env}")
    fidelity, approx, fp32_dest, *rest = CONFIGS[config]
    full_sync = bool(rest and rest[0])
    compute = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32_dest, math_approx_mode=approx, dst_full_sync_en=full_sync
    )
    q, k, v = make_qkv((batch, 12, seqlen, 64), "real", seed=0)
    # NOMIC_SDPA_RESERVE_KB=176: the perf-flagged golden loose case's live-L1 reservation, allocated before the inputs.
    filler = reserve_l1(device, int(os.environ.get("NOMIC_SDPA_RESERVE_KB", "0")))
    mem = ttnn.L1_MEMORY_CONFIG if buffer == "l1" else ttnn.DRAM_MEMORY_CONFIG
    tq, tk, tv = (
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem)
        for t in (q, k, v)
    )
    # NOMIC_SDPA_REPS=3: repeat the call (one profiler row each) for a median on the ~2-3% noise band.
    for _ in range(int(os.environ.get("NOMIC_SDPA_REPS", "1"))):
        out = nomic_sdpa(tq, tk, tv, scale=1.0, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=compute)
        if not os.environ.get("NOMIC_SDPA_ABLATE"):  # ablation stubs payloads: outputs are garbage by design
            assert torch.isfinite(ttnn.to_torch(out).float()).all()
        ttnn.deallocate(out)
    if filler is not None:
        ttnn.deallocate(filler)
