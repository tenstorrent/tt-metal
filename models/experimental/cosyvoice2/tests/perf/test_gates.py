# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""tests/perf/gates.py's enforcement logic, on the host with a stand-in device: a met target is asserted, an unmet
one must stay inside its band in both directions, and an unrecorded one is refused."""
from __future__ import annotations

from models.experimental.cosyvoice2.tests.perf import gates


class _Wormhole:
    def arch(self):
        return "Arch.WORMHOLE_B0"


def test_enforce_meets_misses_and_unrecorded(monkeypatch, expect_error):
    dev = _Wormhole()
    monkeypatch.setitem(gates.EXPECTATIONS, "wormhole", {"rtf_nonstreaming": gates.Meets()})
    assert "PASS" in gates.enforce("rtf_nonstreaming", 0.8, dev)
    with expect_error(AssertionError, "target not met"):
        gates.enforce("rtf_nonstreaming", 1.2, dev)

    monkeypatch.setitem(gates.EXPECTATIONS, "wormhole", {"rtf_nonstreaming": gates.Misses(2.0, 0.25, "bucketing")})
    assert "in band" in gates.enforce("rtf_nonstreaming", 2.2, dev)
    with expect_error(AssertionError, "outside the recorded band"):
        gates.enforce("rtf_nonstreaming", 3.0, dev)  # slower: a regression
    with expect_error(AssertionError, "outside the recorded band"):
        gates.enforce("rtf_nonstreaming", 1.2, dev)  # faster: the published figure is stale
    with expect_error(AssertionError, "is now met"):
        gates.enforce("rtf_nonstreaming", 0.9, dev)

    monkeypatch.setitem(gates.EXPECTATIONS, "wormhole", {})
    with expect_error(AssertionError, "records no verdict"):
        gates.enforce("rtf_nonstreaming", 0.9, dev)
    assert gates.GATES["speaker_similarity"].passes(0.61) and not gates.GATES["wer"].passes(5.0)
