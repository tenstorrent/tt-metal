# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Suite-wide report hooks and fixtures.

The device tests skip when `tests/golden` lacks the goldens (`scripts/gen_golden.py`) or the
weight exports (`scripts/export_weights.py`), and pytest reports a run in which every one of
them skipped as green. The header names what is absent before the run; the summary counts the
tests it cost after it, so such a run does not read as a pass.

`unchecked_prepared_weights` lists the vocoder convolutions that run a prepared weight whose
geometry was never checked (`TtConv1d._verify_prepared`).
"""
from __future__ import annotations

import glob
import os

import pytest

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
WEIGHT_EXPORTS = ("hift_weights.npz", "flow_weights.npz", "llm_weights.npz")
# Words the absent-asset skip reasons across the suite share.
ASSET_SKIP_WORDS = ("gen_golden", "export_weights", "goldens first", "weights first", "prepare_inputs")


def _missing_assets() -> list[str]:
    missing = [w for w in WEIGHT_EXPORTS if not os.path.exists(os.path.join(GOLDEN_DIR, w))]
    goldens = [p for p in glob.glob(os.path.join(GOLDEN_DIR, "*.npz")) if os.path.basename(p) not in WEIGHT_EXPORTS]
    if not goldens:
        missing.insert(0, "the goldens")
    return missing


def pytest_report_header(config):
    missing = _missing_assets()
    if missing:
        return (
            f"CosyVoice: tests/golden lacks {', '.join(missing)}; the tests that need them will skip. "
            "Run scripts/gen_golden.py and scripts/export_weights.py first (README.md, Quick start, step 3)."
        )
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    skipped = [
        r
        for r in terminalreporter.stats.get("skipped", [])
        if isinstance(r.longrepr, tuple) and any(w in str(r.longrepr[-1]) for w in ASSET_SKIP_WORDS)
    ]
    if not skipped:
        return
    terminalreporter.section("CosyVoice: tests skipped for absent goldens, weights or inputs", red=True, bold=True)
    terminalreporter.line(
        f"{len(skipped)} tests did not run because tests/golden or the prompt inputs are absent "
        f"({', '.join(_missing_assets()) or 'the prompt inputs'}). A green result here is not a pass for them: "
        "generate the goldens with scripts/gen_golden.py and export the weights with scripts/export_weights.py, "
        "then run again.",
        red=True,
        bold=True,
    )


@pytest.fixture
def unchecked_prepared_weights(monkeypatch):
    """`watch(hift)` returns a list that fills, from then on, with every call of one of
    `hift`'s convolutions that ran a prepared weight its geometry never had checked.

    A prepared weight can be silently wrong at some lengths (`tt/hifigan/conv.py`,
    `prepare_weights_default`), so the vocoder checks each geometry once before trusting
    it. This counts the calls that trusted one unchecked, which is the order a test can
    assert on any board, not only on one where the unchecked weight happens to be wrong.
    Skips under `COSYVOICE_CONV_PREPARE=1`, which runs prepared weights unchecked by design.
    """
    from models.demos.cosyvoice.tt.hifigan.conv import TtConv1d

    calls: list[str] = []
    watched: set[int] = set()
    call = TtConv1d.__call__

    def recording(self, x, input_length: int, batch_size: int = 1):
        out = call(self, x, input_length, batch_size)
        if id(self) in watched:
            key = (input_length, batch_size)
            weight = self._prep_cache.get(key, (self.weight, None))[0]
            if weight is not self.weight and key not in self._verified:
                calls.append(
                    f"Conv1d({self.in_channels}->{self.out_channels}, k={self.kernel_size}) at length {input_length}"
                )
        return out

    def watch(hift):
        if os.environ.get("COSYVOICE_CONV_PREPARE") == "1":
            pytest.skip("COSYVOICE_CONV_PREPARE=1 runs prepared weights unchecked by design")
        watched.update(id(c) for c in hift._convs())
        return calls

    monkeypatch.setattr(TtConv1d, "__call__", recording)
    return watch
