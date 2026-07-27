# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full-model smoke test: coherence + decode-latency ceiling.

The rest of this directory tests components (layer 0, MoE, embedding, MLA
boundaries). Nothing covered the full 47-layer model end to end, and nothing
covered decode *latency* at all -- so the whole 74.8 -> 51.3 ms optimization
stack landed with no automated guard. This is that guard.

It drives `scripts/debug_run_full_tt_greedy.py` in a subprocess rather than
calling the model API directly, deliberately:

  - it exercises the exact validated path (perf_defaults applied at import,
    traced sampling decode) instead of a parallel reimplementation that can
    drift from what we actually benchmark;
  - subprocess isolation gives clean device teardown, so a failure here cannot
    leave a wedged mesh behind for the next test;
  - it mirrors `scripts/run_sweep_isl_batch.py`, which already parses this same
    stdout contract.

One run, two assertions (the run costs minutes, so the fixture is module-scoped
and both tests consume it).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

RUNNER = "models/experimental/glm4_moe_lite/scripts/debug_run_full_tt_greedy.py"

# Known-answer prompt. Must not itself contain the expected answer, or the
# coherence check would pass on the echoed prompt alone.
PROMPT = "What is the capital city of Australia? Answer with just the city name."
EXPECTED_ANSWER = "Canberra"

# Decode-latency thresholds, ms/token, bs=1 traced steady-state decode.
#
# CALIBRATION (measured, this exact test, 32-chip WH Galaxy 4x8, 2026-07-27):
#   mean 50.1 ms/token, min/max spread 0.8 ms.
# For reference the ISL=128 sweep cell reports 51.3 ms; this test uses a real
# short prompt rather than simulated context, which lands slightly faster.
#
# Two levels, because a single lost optimization sits uncomfortably close to the
# machine-variation band. Projected regressions from the documented stack:
#   lose BF8 dense (~7%)        -> ~53.6 ms
#   lose MoE layout ops (~4.5%) -> ~52.4 ms
#   lose sharded RMSNorm (~4%)  -> ~52.1 ms
# So the hard ceiling is set to catch a >=10% / compound regression reliably
# without flaking, and the advisory catches any single-optimization loss and
# reports it without failing the build.
#
# Calibrated on ONE machine. Tighten the ceiling toward the advisory once there
# is multi-machine, multi-build history. Override per-environment with
# GLM4_MOE_LITE_SMOKE_MAX_DECODE_MS.
DEFAULT_MAX_DECODE_MS = 55.0
ADVISORY_DECODE_MS = 52.0

_SUBSEQUENT_RE = re.compile(r"subsequent:\s+mean=\s*([\d.]+)\s+min=\s*([\d.]+)\s+max=\s*([\d.]+)")
# loguru / ttnn log lines that interleave with the generated text on stdout. Levels
# appear in both cases ("| DEBUG |" and "| info  "), and the trailing pipe is not
# always present, so anchor on the leading ISO timestamp as well.
_LOG_LINE_RE = re.compile(
    r"\|\s*(?i:debug|info|warning|error|critical|always)\s*\|" r"|^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"
)

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("TT_ENABLE_HW_TESTS") != "1",
        reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
    ),
    pytest.mark.skipif(
        os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
        reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads the full 47-layer model).",
    ),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class SmokeResult:
    """Parsed outcome of one full-model greedy run."""

    def __init__(self, stdout: str, stderr: str, returncode: int) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    @property
    def decode_mean_ms(self) -> float | None:
        m = _SUBSEQUENT_RE.search(self.stdout)
        return float(m.group(1)) if m else None

    @property
    def decode_spread_ms(self) -> float | None:
        m = _SUBSEQUENT_RE.search(self.stdout)
        return (float(m.group(3)) - float(m.group(2))) if m else None

    @property
    def generated_text(self) -> str:
        """Everything the runner printed after the latency block, minus log lines."""
        m = _SUBSEQUENT_RE.search(self.stdout)
        tail = self.stdout[m.end() :] if m else self.stdout
        lines = [ln for ln in tail.splitlines() if ln.strip() and not _LOG_LINE_RE.search(ln)]
        return "\n".join(lines).strip()


@pytest.fixture(scope="module")
def smoke_run() -> SmokeResult:
    """Run the full model once, greedily, with the validated default flag set."""
    root = _repo_root()
    mesh_rows = os.environ.get("GLM4_MOE_LITE_SMOKE_MESH_ROWS", "4")
    mesh_cols = os.environ.get("GLM4_MOE_LITE_SMOKE_MESH_COLS", "8")
    timeout_s = int(_env_float("GLM4_MOE_LITE_SMOKE_TIMEOUT_S", 1800.0))

    cmd = [
        sys.executable,
        RUNNER,
        "--prompt",
        PROMPT,
        "--max-new-tokens",
        "24",
        "--batch-size",
        "1",
        "--mesh-rows",
        mesh_rows,
        "--mesh-cols",
        mesh_cols,
        "--kv-cache-dtype",
        "bf16",
        "--min-cache-tokens",
        "256",
        "--phase",
        "both",
        "--enable-trace",
        "--trace-mode",
        "sampling",
    ]
    # Inherit the environment so the runner applies its own perf defaults; do not
    # pin flags here, or this test would stop reflecting the shipping config.
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    result = SmokeResult(proc.stdout, proc.stderr, proc.returncode)

    if proc.returncode != 0:
        pytest.fail(
            f"runner exited {proc.returncode}\n"
            f"--- last 40 lines of stdout ---\n{os.linesep.join(result.stdout.splitlines()[-40:])}\n"
            f"--- last 20 lines of stderr ---\n{os.linesep.join(result.stderr.splitlines()[-20:])}"
        )
    return result


def test_full_model_decode_latency_under_ceiling(smoke_run: SmokeResult) -> None:
    """Guard the decode-latency regression the component tests cannot see."""
    ceiling = _env_float("GLM4_MOE_LITE_SMOKE_MAX_DECODE_MS", DEFAULT_MAX_DECODE_MS)
    mean_ms = smoke_run.decode_mean_ms

    assert mean_ms is not None, (
        "could not parse the decode-latency line from runner stdout. Expected a "
        "'subsequent: mean=... min=... max=...' line; the runner's output contract "
        f"may have changed.\n--- tail ---\n{os.linesep.join(smoke_run.stdout.splitlines()[-30:])}"
    )

    # Always report, pass or fail -- the measured number is the useful artifact.
    spread = smoke_run.decode_spread_ms
    print(f"\n[smoke] decode mean={mean_ms:.1f} ms/token  spread={spread:.1f} ms  ceiling={ceiling:.1f} ms")

    if mean_ms > ADVISORY_DECODE_MS:
        print(
            f"[smoke] ADVISORY: {mean_ms:.1f} ms is above the {ADVISORY_DECODE_MS:.1f} ms advisory "
            "level. Still passing, but the regression margin is shrinking -- worth profiling."
        )

    assert mean_ms <= ceiling, (
        f"decode latency regressed: {mean_ms:.1f} ms/token > {ceiling:.1f} ms ceiling.\n"
        "The optimization stack that produced ~51.3 ms is documented in README.md; a jump "
        "here usually means one of those flags stopped taking effect. Check the flags the "
        "runner actually applied (perf_defaults.apply_perf_defaults) before assuming a code "
        "regression, and see overridden_defaults() for environment overrides."
    )


def test_full_model_output_is_coherent(smoke_run: SmokeResult) -> None:
    """Guard numerical regressions that leave the model running but incoherent.

    Perf work on this model has repeatedly produced changes that run at full speed
    and emit garbage (FUSED_KV_BRANCH is the standing example). A latency ceiling
    alone would pass all of them, so pair it with a known-answer check.
    """
    text = smoke_run.generated_text
    assert text, "runner produced no generated text"

    print(f"\n[smoke] generated: {text[:200]!r}")

    assert EXPECTED_ANSWER.lower() in text.lower(), (
        f"known-answer check failed: expected {EXPECTED_ANSWER!r} in the generated text.\n"
        f"Got: {text[:500]!r}\n"
        "This is the signature of a numerical regression (wrong dtype, a bad fusion, or a "
        "CCL corruption) rather than a crash."
    )

    # Degenerate repetition is the other common failure signature: the model stays
    # "coherent" token-by-token but emits one token forever.
    words = text.split()
    if len(words) >= 8:
        assert len(set(words)) > 1, f"generated text is a single repeated token: {text[:200]!r}"
