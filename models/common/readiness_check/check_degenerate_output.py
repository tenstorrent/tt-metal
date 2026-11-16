# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Machine check for degenerate generated text in readiness artifacts.

This is a runner-side verification, not a quality judgement. It flags
mechanically broken generation — the kind produced by decode-loop bugs
(stale token/position feedback, stale trace inputs) — while staying
agnostic to model style, checkpoint type (base vs instruct), and content.

The key signal is adjacent-token duplication: a healthy model, base or
instruct, does not emit nearly every word twice while the text continues
to advance ("the the difference difference between between ..."). Phrase
-level looping, by contrast, is common in base checkpoints under greedy
decoding and is reported only as advisory.

Checked artifacts (discovered under one or more roots):

  - ``readiness_vllm/vllm_qualitative_outputs.json`` written by
    ``run_vllm_server`` (list of {prompt, greedy_completion,
    sampled_completion}).
  - ``autoregressive_meta.json`` written by ``run_autoregressive``
    ({hf: {token_ids}, tt: {token_ids}, ...}) plus the sibling
    ``tt_completion.txt`` when present.

Exit codes: 0 = clean, 1 = advisory findings only, 2 = critical findings,
3 = checker-internal error (never a model verdict).

Invoke by file path rather than ``python -m`` so the package ``__init__``
(which imports torch) is not pulled in:

    python models/common/readiness_check/check_degenerate_output.py \\
        [--model-dir models/autoports/<model>] [--hf-model <hf-model-id>] \\
        [--root models/autoports] [--scope all|vllm|autoregressive] \\
        [--missing-artifacts advisory|critical] [--json report.json]

Scoping precedence: explicit paths, then ``--model-dir``, then ``--hf-model``
fuzzy resolution against autoport directory names, then the unscoped
``--root``. Scoped runs ensure stale artifacts from another model on the
same machine can neither pass nor fail this check. Unreadable artifacts are
critical findings (the stage owns its evidence); exit 3 is reserved for
checker/environment faults.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# A completion shorter than this many words is too small for a stable
# duplication rate; it is skipped rather than judged.
MIN_WORDS_FOR_DUPLICATION = 20
# Critical: fraction of consecutive word pairs that are identical.
# Calibration on experiment archives: healthy serving outputs measure
# <= 0.8% (max over completions); the known token-feedback bug measures
# 54% mean / 94% max. Anything above 10% is mechanical, not stylistic.
ADJACENT_DUPLICATION_CRITICAL = 0.10
# Advisory: fraction of the completion covered by repeats of its most
# common trigram. Base checkpoints legitimately loop phrases under greedy
# decoding, so this never fails the check on its own.
TRIGRAM_LOOP_ADVISORY = 0.50
MIN_WORDS_FOR_LOOP = 50
# Advisory: a near-empty completion when many tokens were requested.
NEAR_EMPTY_CHARS = 5

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class Finding:
    severity: str  # "critical" | "advisory"
    artifact: str
    label: str
    metric: str
    value: float
    threshold: float
    detail: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    measured: list[dict[str, Any]] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if any(f.severity == "critical" for f in self.findings):
            return 2
        if self.findings:
            return 1
        return 0


def words_of(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def adjacent_duplication(tokens: Sequence[Any]) -> float:
    """Fraction of consecutive token pairs that are identical."""
    if len(tokens) < 2:
        return 0.0
    dup = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    return dup / (len(tokens) - 1)


def trigram_loop_fraction(tokens: Sequence[Any]) -> float:
    """Fraction of the sequence covered by non-overlapping repeats of its most common trigram."""
    if len(tokens) < 3:
        return 0.0
    counts: dict[tuple[Any, ...], int] = {}
    for i in range(len(tokens) - 2):
        gram = tuple(tokens[i : i + 3])
        counts[gram] = counts.get(gram, 0) + 1
    top_gram = max(counts, key=counts.get)  # type: ignore[arg-type]
    covered = 0
    i = 0
    while i <= len(tokens) - 3:
        if tuple(tokens[i : i + 3]) == top_gram:
            covered += 3
            i += 3
        else:
            i += 1
    return covered / len(tokens)


def check_completion(
    report: Report,
    *,
    artifact: Path,
    label: str,
    text: str | None,
    token_ids: Sequence[int] | None = None,
) -> None:
    """Apply degeneracy metrics to one completion (text and/or token ids)."""
    tokens: Sequence[Any] | None = None
    source = None
    if text is not None and text.strip():
        tokens = words_of(text)
        source = "words"
    elif token_ids:
        # No text available: token-id duplication is a weaker but still
        # meaningful signal (repeated layout tokens make it noisier).
        tokens = list(token_ids)
        source = "token_ids"

    measured: dict[str, Any] = {"artifact": str(artifact), "label": label, "source": source}

    if text is not None and len(text.strip()) < NEAR_EMPTY_CHARS:
        report.findings.append(
            Finding(
                severity="advisory",
                artifact=str(artifact),
                label=label,
                metric="near_empty_completion",
                value=float(len(text.strip())),
                threshold=float(NEAR_EMPTY_CHARS),
                detail="Completion is empty or whitespace; verify EOS handling and sampler output.",
            )
        )
        measured["near_empty"] = True
        report.measured.append(measured)
        return

    if tokens is None:
        report.measured.append(measured)
        return

    dup = adjacent_duplication(tokens)
    loop = trigram_loop_fraction(tokens)
    measured.update(
        {"num_tokens": len(tokens), "adjacent_duplication": round(dup, 4), "trigram_loop_fraction": round(loop, 4)}
    )
    report.measured.append(measured)

    if len(tokens) >= MIN_WORDS_FOR_DUPLICATION and dup > ADJACENT_DUPLICATION_CRITICAL:
        report.findings.append(
            Finding(
                severity="critical",
                artifact=str(artifact),
                label=label,
                metric="adjacent_duplication",
                value=round(dup, 4),
                threshold=ADJACENT_DUPLICATION_CRITICAL,
                detail=(
                    "Adjacent-token duplication this high while the text advances is a "
                    "decode-loop input bug signature (stale token/position/trace feedback), "
                    "not a model-quality property. Compare the same prompt against the HF "
                    "reference before classifying this as a model limitation."
                ),
            )
        )
    elif len(tokens) >= MIN_WORDS_FOR_LOOP and loop > TRIGRAM_LOOP_ADVISORY:
        report.findings.append(
            Finding(
                severity="advisory",
                artifact=str(artifact),
                label=label,
                metric="trigram_loop_fraction",
                value=round(loop, 4),
                threshold=TRIGRAM_LOOP_ADVISORY,
                detail=(
                    "Completion is dominated by one repeating phrase. This can be normal for "
                    "base checkpoints under greedy decoding; verify against the HF reference "
                    "on the same prompt."
                ),
            )
        )


def _load_artifact(report: Report, path: Path) -> Any | None:
    """Parse an artifact, recording unreadable files as critical findings.

    A required artifact that cannot be read is failed stage evidence, not a
    checker fault, so it goes through the normal finding path rather than
    exiting 3.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        report.findings.append(
            Finding(
                severity="critical",
                artifact=str(path),
                label="artifact parse",
                metric="unreadable_artifact",
                value=0.0,
                threshold=1.0,
                detail=f"Artifact could not be read or parsed ({exc}). Regenerate it from the documented runner.",
            )
        )
        return None


def check_vllm_qualitative(report: Report, path: Path) -> None:
    items = _load_artifact(report, path)
    if items is None:
        return
    for i, item in enumerate(items):
        prompt = str(item.get("prompt", ""))[:60]
        for key in ("greedy_completion", "sampled_completion"):
            if key in item:
                check_completion(
                    report,
                    artifact=path,
                    label=f"prompt[{i}] {key} ({prompt!r})",
                    text=item.get(key) or "",
                )


def check_autoregressive_meta(report: Report, path: Path) -> None:
    meta = _load_artifact(report, path)
    if meta is None:
        return
    tt = meta.get("tt", {})
    text_path = path.parent / "tt_completion.txt"
    text = text_path.read_text(encoding="utf-8") if text_path.exists() else None
    check_completion(
        report,
        artifact=path,
        label="tt free-running completion",
        text=text,
        token_ids=tt.get("token_ids"),
    )
    hf_ids = meta.get("hf", {}).get("token_ids")
    tt_ids = tt.get("token_ids")
    if hf_ids and tt_ids:
        match = sum(1 for a, b in zip(hf_ids, tt_ids) if a == b)
        report.measured.append(
            {
                "artifact": str(path),
                "label": "hf/tt token agreement (informational)",
                "matching_tokens": match,
                "compared_tokens": min(len(hf_ids), len(tt_ids)),
            }
        )


def discover(roots: Iterable[Path], scope: str) -> tuple[list[Path], list[Path]]:
    vllm_files: list[Path] = []
    meta_files: list[Path] = []
    for root in roots:
        if root.is_file():
            if root.name == "autoregressive_meta.json":
                meta_files.append(root)
            else:
                vllm_files.append(root)
            continue
        if scope in ("all", "vllm"):
            vllm_files.extend(sorted(root.rglob("vllm_qualitative_outputs.json")))
        if scope in ("all", "autoregressive"):
            meta_files.extend(sorted(root.rglob("autoregressive_meta.json")))
    return vllm_files, meta_files


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def resolve_model_dirs(root: Path, hf_model: str) -> tuple[list[Path], str]:
    """Find the autoport directory (or directories) belonging to hf_model.

    Agents choose their own autoport directory names (observed variants for
    one model include `llama31_8b_instruct` and `meta_llama_Llama_3_1_8B_Instruct`),
    so match on alphanumeric-squashed containment between the directory path
    relative to root and the HF model id, in either direction.
    """
    if not root.is_dir():
        return [], f"{root} does not exist"
    target = _squash(hf_model)
    markers = ("tt", "doc", "readiness_vllm")
    candidates = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_dir()
        and len(path.relative_to(root).parts) <= 3
        and any((path / marker).is_dir() for marker in markers)
    ]
    matches = []
    for path in candidates:
        squashed = _squash(str(path.relative_to(root)))
        if squashed and (squashed in target or target in squashed):
            matches.append(path)
    # Prefer the deepest matching directories; a parent that only matched
    # because its child did adds nothing but duplicate scanning.
    matches = [m for m in matches if not any(other != m and other.is_relative_to(m) for other in matches)]
    if matches:
        return matches, ""
    if candidates:
        listing = ", ".join(str(c.relative_to(root)) for c in candidates)
        return [], f"no autoport directory under {root} matches {hf_model!r} (found: {listing})"
    return [], f"no autoport directories found under {root}"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # noqa: D102 - argparse override
        # argparse exits 2 on misuse, which stage gates would misread as a
        # critical model verdict. CLI misuse is a checker/environment fault.
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(3)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _Parser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Artifact files or directories to scan. Default: --model-dir, --hf-model resolution, or --root.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("models/autoports"),
        help="Directory scanned when no paths are given (default: models/autoports).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Explicit autoport directory to scope the scan to. Takes precedence over --hf-model.",
    )
    parser.add_argument(
        "--hf-model",
        help="Scope the scan to the autoport directory matching this HF model id (fuzzy fallback when --model-dir is not given).",
    )
    parser.add_argument("--scope", choices=("all", "vllm", "autoregressive"), default="all")
    parser.add_argument(
        "--missing-artifacts",
        choices=("advisory", "critical"),
        default="advisory",
        help=(
            "Severity when no generation artifacts (or no matching model "
            "directory) are found. Stage gates should use 'critical': by "
            "stages 5+, missing generation evidence is a failed completion "
            "requirement, not a soft warning."
        ),
    )
    parser.add_argument("--json", type=Path, help="Write the machine-readable report here.")
    args = parser.parse_args(argv)

    report = Report()

    if args.paths:
        roots: list[Path] = args.paths
    elif args.model_dir:
        if args.model_dir.is_dir():
            roots = [args.model_dir]
        else:
            roots = []
            report.findings.append(
                Finding(
                    severity=args.missing_artifacts,
                    artifact=str(args.model_dir),
                    label="model directory resolution",
                    metric="missing_model_dir",
                    value=0.0,
                    threshold=1.0,
                    detail=(
                        f"--model-dir {args.model_dir} does not exist. The stage requires "
                        "generation evidence under the model's autoport directory."
                    ),
                )
            )
    elif args.hf_model:
        roots, why_empty = resolve_model_dirs(args.root, args.hf_model)
        if not roots:
            report.findings.append(
                Finding(
                    severity=args.missing_artifacts,
                    artifact=str(args.root),
                    label="model directory resolution",
                    metric="missing_model_dir",
                    value=0.0,
                    threshold=1.0,
                    detail=(
                        f"{why_empty}. The stage requires generation evidence under the "
                        "model's autoport directory; create it at the documented location."
                    ),
                )
            )
        else:
            print(f"scoped to: {', '.join(str(r) for r in roots)}")
    else:
        roots = [args.root]

    vllm_files, meta_files = discover(roots, args.scope)

    if roots and not vllm_files and not meta_files and not report.findings:
        report.findings.append(
            Finding(
                severity=args.missing_artifacts,
                artifact=", ".join(str(r) for r in roots),
                label="artifact discovery",
                metric="missing_artifacts",
                value=0.0,
                threshold=1.0,
                detail=(
                    f"No generation artifacts found (scope={args.scope}). If this stage "
                    "requires generation evidence, produce it at the documented location "
                    "before reporting complete."
                ),
            )
        )
    for path in vllm_files:
        check_vllm_qualitative(report, path)
    for path in meta_files:
        check_autoregressive_meta(report, path)

    for m in report.measured:
        compact = {k: v for k, v in m.items() if k not in ("artifact", "label")}
        print(f"measured: {m.get('label')} [{m.get('artifact')}] {compact}")

    if report.findings:
        print(f"\n{len(report.findings)} finding(s):")
        for f in report.findings:
            print(f"\n[{f.severity.upper()}] {f.metric}={f.value} (threshold {f.threshold})")
            print(f"  artifact: {f.artifact}")
            print(f"  where:    {f.label}")
            print(f"  {f.detail}")
    else:
        print("\nNo degenerate output detected.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "findings": [asdict(f) for f in report.findings],
                    "measured": report.measured,
                    "exit_code": report.exit_code,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    return report.exit_code


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 - exit 3 marks checker-internal errors, never a model verdict
        import traceback

        traceback.print_exc()
        sys.exit(3)
