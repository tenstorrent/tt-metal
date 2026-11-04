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
    # Stage 1 protocol on the masked HiFT (2026-09-30): warmup_buckets() first (185 s on a warm kernel cache), then the
    # six distinct corpus utterances. RTF 0.441-0.654 each, aggregate 0.483; the perf test's own run, worst 0.675.
    # (09-28, silence padding: 0.433-0.628.)
    # Before bucketing, a distinct utterance ran at RTF 21-75 on a cold kernel cache.
    "rtf_nonstreaming": Meets(),
    # Teacher-forced top-1 (tests/e2e/test_token_accuracy.py), fp32-logit head: 95.94 % over 5,003 positions of
    # 27 sequences, 4 speakers (the corpus plus its token-accuracy extension); 96.37 % on the first seven (bf16
    # logits: 90.66 %). A bf16 PyTorch run of the same model reaches 96.45 %, or 98.58 % with an fp32 head.
    "token_accuracy": Meets(),
    # scripts/eval_draws.py in the reference venv: the corpus scorer over five vocoder noise draws with the tokens
    # fixed (2026-09-30, masked HiFT). Corpus WER 0.68 % in every draw and WavLM-base-plus-sv SIM 95.88 (95.84-95.92,
    # cosine x 100); the PyTorch reference 0.68 % and 95.22. Recorded, not
    # enforced by a device test.
    "wer": Meets(),
    "speaker_similarity": Meets(),
    # Streaming (demo.py --stream): warmup_buckets() and warmup_streaming() first, then the six distinct corpus
    # utterances, in fresh processes; two runs on 2026-09-29 and two on the masked HiFT on 2026-09-30. The figure is
    # the worst utterance, as for rtf_nonstreaming. Enforced by tests/perf/test_pipeline_perf.py's streaming test
    # (2026-09-30: worst 1,469 ms and RTF 1.110).
    # Time to first packet: worst 1,455 and 1,479 ms (09-29), 1,502 and 1,432 ms (09-30); best 1,313. The first chunk
    # is 0.37-0.47 s of text and LLM until its 25 or 32 tokens and 3 look-ahead, then the flow over the prompt and the
    # chunk, 0.81-0.92 s, and HiFT, 0.12-0.13 s.
    "ttfp_ms": Misses(
        1470.0,
        0.15,
        "the first chunk's flow: the CFM's 10 Euler steps (67-73 ms each) over the prompt plus the chunk; with a free "
        "flow, first audio would be 0.51-0.59 s (the LLM's first 28-35 tokens, then HiFT)",
    ),
    # Streaming RTF: worst 1.057, 1.122, 1.121 and 1.103, each on the 3.8 s utterance (aggregate 0.836-0.853). Every
    # chunk reruns the flow over the whole prefix, and the final chunk runs it non-streaming over every token, as
    # upstream does.
    "rtf_streaming": Misses(
        1.09,
        0.2,
        "the flow per chunk (the CFM over the whole prefix, 10 Euler steps); short utterances carry the ~1.4 s first "
        "chunk over little audio",
    ),
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
