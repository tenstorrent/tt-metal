# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The bounty's numeric targets (tenstorrent/tt-metal#54104), and the code that enforces them.

`GATES` quotes every numeric target in the issue. `EXPECTATIONS` records, per architecture, the verdict for each
target measured so far, and grows as measurements arrive. A target nobody has measured has no entry, and
`enforce` refuses it rather than guessing.

* `Meets()`: the target is met; `enforce` asserts the target itself, so a regression fails.
* `Misses(recorded, tol, lever)`: the target is not met. `enforce` asserts the measurement stays inside
  `recorded * (1 +- tol)` in both directions: worse is a regression, better means the published figure
  (docs/VALIDATION.md) is stale. Nothing is `xfail`-ed; an unmet target keeps a number, a band and a named lever.

This is CosyVoice1's scheme (models/experimental/cosyvoice/tests/perf/gates.py). As there, a `recorded` value is
the centre of a band, not the last run's figure.
"""
from __future__ import annotations

from dataclasses import dataclass

ABOVE, BELOW = "above", "below"


@dataclass(frozen=True)
class Gate:
    key: str
    quote: str  # the issue's wording
    stage: str
    target: float
    direction: str
    unit: str = ""

    def passes(self, measured: float) -> bool:
        return measured > self.target if self.direction == ABOVE else measured < self.target

    def describe(self) -> str:
        return f"{self.key} {'>' if self.direction == ABOVE else '<'} {self.target}{self.unit}"


GATES: dict[str, Gate] = {
    g.key: g
    for g in (
        Gate("token_accuracy", "Token-level accuracy > 95% against PyTorch reference for LLM output", "Stage 1", 95.0,
             ABOVE, " %"),
        Gate("wer", "Audio quality: WER < 5.0 ... on a representative test set", "Stage 1", 5.0, BELOW, " %"),
        Gate("speaker_similarity", "speaker similarity > 0.60 cosine sim on a representative test set", "Stage 1",
             0.60, ABOVE),
        # Measured as the worst per-utterance RTF over the corpus's distinct utterances (scripts/corpus.py), after
        # warmup_buckets() (the Stage 1 protocol); see tests/perf/test_pipeline_perf.py.
        Gate("rtf_nonstreaming", "RTF < 1.0 for non-streaming, whole-utterance synthesis", "Stage 1", 1.0, BELOW),
        Gate("ttfp_ms", "Streaming inference support with time-to-first-packet < 500ms", "Stage 3", 500.0, BELOW,
             " ms"),
        Gate("rtf_streaming", "RTF < 0.4 for streaming synthesis", "Stage 3", 0.4, BELOW),
    )
}  # fmt: skip


@dataclass(frozen=True)
class Meets:
    """The target is met on this architecture; assert the target itself."""


@dataclass(frozen=True)
class Misses:
    """Not met: assert the measurement stays within `recorded * (1 +- tol)`. `lever` names what would close it."""

    recorded: float
    tol: float
    lever: str


# Wormhole: N150, the board every figure so far comes from (docs/VALIDATION.md).
WORMHOLE: dict = {
    # Stage 1 protocol, 2026-09-28: warmup_buckets() first (542 s on a warm kernel cache), then the six distinct
    # corpus utterances. RTF 0.428-0.633 each, aggregate 0.481. Before bucketing, a distinct utterance ran at RTF
    # 21-75 on a cold kernel cache.
    "rtf_nonstreaming": Meets(),
    # Teacher-forced top-1 (tests/e2e/test_token_accuracy.py), fp32-logit head: 95.94 % over 5,003 positions of
    # 27 sequences, 4 speakers (the corpus plus its token-accuracy extension); 96.37 % on the first seven (bf16
    # logits: 90.66 %). A bf16 PyTorch run of the same model reaches 96.45 %, or 98.58 % with an fp32 head.
    "token_accuracy": Meets(),
    # scripts/eval_wer_sim.py in the reference venv, on the bucketed Stage 1 audio (2026-09-28): corpus WER 0.68 %
    # and WavLM-base-plus-sv SIM 95.88 (cosine x 100), the PyTorch reference 0.68 % and 95.21. Recorded, not
    # enforced by a device test.
    "wer": Meets(),
    "speaker_similarity": Meets(),
}
EXPECTATIONS = {"wormhole": WORMHOLE}


def arch_key(device) -> str:
    arch = str(device.arch()).upper()
    for key in ("wormhole", "blackhole"):
        if key.upper() in arch:
            return key
    raise AssertionError(f"unknown architecture {arch!r}")


def enforce(key: str, measured: float, device, *, extra: str = "") -> str:
    """Assert `measured` against `GATES[key]` under this architecture's recorded verdict; return the verdict
    line. Fails when a met target regresses, when an unmet one leaves its band either way, and when nothing is
    recorded for `key` on this architecture yet (record the first measurement in `EXPECTATIONS`)."""
    gate, arch = GATES[key], arch_key(device)
    suffix = f"  [{extra}]" if extra else ""
    verdict = EXPECTATIONS.get(arch, {}).get(key)
    assert verdict is not None, (
        f"{gate.describe()}: measured {measured:.3f} on {arch}, but tests/perf/gates.py records no verdict for it "
        f"there yet. Record Meets() or Misses(...) in EXPECTATIONS, with the figure in docs/VALIDATION.md."
    )
    if isinstance(verdict, Meets):
        assert gate.passes(measured), (
            f"{gate.stage} target not met on {arch}: {gate.describe()}, measured {measured:.3f}; recorded as met in "
            f"tests/perf/gates.py, so this is a regression or a run that is not comparable"
        )
        return f"{gate.describe():<32} measured {measured:8.3f}   PASS{suffix}"
    lo, hi = verdict.recorded * (1 - verdict.tol), verdict.recorded * (1 + verdict.tol)
    assert not gate.passes(measured), (
        f"{gate.describe()} is now met on {arch} (measured {measured:.3f}) but recorded as missed at "
        f"{verdict.recorded}; promote it to Meets() and update docs/VALIDATION.md"
    )
    assert lo <= measured <= hi, (
        f"{gate.describe()} on {arch}: measured {measured:.3f}, outside the recorded band [{lo:.3f}, {hi:.3f}]; "
        f"{'a regression' if gate.direction == BELOW and measured > hi else 'update docs/VALIDATION.md and this table'}"
    )
    return f"{gate.describe():<32} measured {measured:8.3f}   MISS, in band [{lo:.3f}, {hi:.3f}]{suffix}\n    lever: {verdict.lever}"


def report(lines: list[str], title: str) -> None:
    print(f"\n  {title}")
    for line in lines:
        print(f"    {line}")
