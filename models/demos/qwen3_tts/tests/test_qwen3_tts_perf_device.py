# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-block performance gate for qwen3_tts.

Runs the full TTS demo (warmup + trace capture + inference) and asserts
two timings from the inference pass within ``MARGIN`` of an arch-specific
golden:

  prefill_ms          (Talker prefill: ICL -> first decode token)
  steady_ms_per_frame (mean AR step time, excluding the first decode step)

Warmup and trace-capture cost are excluded — only the inference numbers
are checked.

Goldens are arch-specific (``ARCH_GOLDENS`` below). Selection is driven by the
``arch`` field in the demo's result dict (the demo emits ``device.arch().name``).
For Wormhole N150 set ``MESH_DEVICE=N150`` so tt-metal selects the 8x8 worker
grid required by the sharded layouts (see demo_full_ttnn_tts.py).

Run (BH P150):
    pytest -s -v models/demos/qwen3_tts/tests/test_qwen3_tts_perf_device.py

Run (WH N150):
    MESH_DEVICE=N150 pytest -s -v models/demos/qwen3_tts/tests/test_qwen3_tts_perf_device.py
"""
from pathlib import Path

import pytest

# Steady AR-step ms is checked against a tight bidirectional bound.
# Prefill ms varies a lot run-to-run with trace-cache state and bucket warmth
# (observed 15-25 ms swing on otherwise-identical runs on BH), so we only
# enforce an upper bound — catches regressions, ignores favorable variance.
ARCH_GOLDENS = {
    "BLACKHOLE": {
        # BH P150: 2CQ is faster (no fabric overhead, H2D overlap helps).
        "steady_ms_per_frame": 59.2,
        "steady_margin": 0.05,
        "prefill_ms_upper": 30.0,
        "use_2cq": True,
    },
    "WORMHOLE_B0": {
        # WH: 1CQ is faster than 2CQ (event sync overhead > H2D overlap benefit).
        # N150 TP=1 1CQ: ~103 ms/frame.  N300 TP=2 1CQ: ~91-95 ms/frame.
        # Golden covers the N300 TP=2 path (primary test target); N150 is close
        # to the upper bound.
        # Prefill: 23-32 ms typical (40 ms cap for cold-cache outliers).
        "steady_ms_per_frame": 97.0,
        "steady_margin": 0.10,
        "prefill_ms_upper": 40.0,
        "use_2cq": False,
    },
}

REPO_ROOT = Path(__file__).resolve().parents[4]
REF_AUDIO = str(REPO_ROOT / "models" / "demos" / "qwen3_tts" / "demo" / "jim_reference.wav")
REF_TEXT = "Jason, can we take a look at the review slides"
TARGET_TEXT = (
    "Good morning. Today is a beautiful day for a walk in the park, with bright sun "
    "and a gentle breeze through the trees."
)
OUTPUT_WAV = "/tmp/qwen3_tts_perf_device.wav"


@pytest.fixture(scope="module")
def demo_run():
    """Run run_full_ttnn_tts once and return the timing dict.

    use_2cq is read from ARCH_GOLDENS after a brief device probe so each arch
    gets its optimal CQ mode: BH uses 2CQ (H2D overlap helps on single chip),
    WH uses 1CQ (event sync overhead > H2D benefit on both N150 and N300).
    """
    # Detect arch to pick the right CQ mode without running the full demo twice.
    import os

    import ttnn
    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import run_full_ttnn_tts

    mesh_shape_map = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}
    mesh_shape = mesh_shape_map.get(os.environ.get("MESH_DEVICE"), (1, 1))
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    probe = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))
    arch_name = probe.arch().name
    ttnn.close_mesh_device(probe)
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    g = ARCH_GOLDENS.get(arch_name, ARCH_GOLDENS["WORMHOLE_B0"])
    _use_2cq = g.get("use_2cq", False)

    result = run_full_ttnn_tts(
        text=TARGET_TEXT,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        output_path=OUTPUT_WAV,
        seed=42,
        use_2cq=_use_2cq,
    )
    assert isinstance(result, dict), f"run_full_ttnn_tts must return a dict; got {type(result)}"
    return result


def _goldens_for(demo_run):
    arch = demo_run.get("arch")
    if arch not in ARCH_GOLDENS:
        pytest.skip(f"No perf goldens defined for arch={arch!r}; add an entry to ARCH_GOLDENS")
    return arch, ARCH_GOLDENS[arch]


def test_prefill_ms(demo_run):
    arch, g = _goldens_for(demo_run)
    measured = demo_run["prefill_ms"]
    upper = g["prefill_ms_upper"]
    print(f"[prefill_ms] arch={arch} measured = {measured:.2f} ms  upper bound = {upper:.2f} ms")
    assert measured < upper, f"[prefill_ms] {measured:.2f} ms >= {upper:.2f} ms upper bound (arch={arch})"


def test_steady_ms_per_frame(demo_run):
    arch, g = _goldens_for(demo_run)
    measured = demo_run["steady_ms_per_frame"]
    expected = g["steady_ms_per_frame"]
    margin = g["steady_margin"]
    lower = expected * (1 - margin)
    upper = expected * (1 + margin)
    print(
        f"[steady_ms_per_frame] arch={arch} measured = {measured:.2f} ms  expected = {expected:.2f} ms  "
        f"bounds = [{lower:.2f}, {upper:.2f}]"
    )
    assert lower <= measured <= upper, (
        f"[steady_ms_per_frame] {measured:.2f} ms outside [{lower:.2f}, {upper:.2f}] "
        f"(arch={arch}, expected {expected:.2f}, margin {margin:.0%})"
    )
