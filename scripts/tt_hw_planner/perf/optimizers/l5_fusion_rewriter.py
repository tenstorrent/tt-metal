# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L5 fusion_rewriter — match known fusion patterns; v1 emits a diff for review."""

from __future__ import annotations

from typing import List, Sequence

from ..join import JoinedRow
from .base import (
    Finding,
    ModelSource,
    Patch,
    PatchKind,
    Severity,
    SourceLocation,
    VerificationResult,
)


# Pattern: tuple of op codes that occur in order in the same block.
PATTERNS = [
    (("ttnn.matmul", "ttnn.add", "ttnn.gelu"), "ffn_matmul_bias_gelu", "ttnn.linear_with_bias_and_gelu"),
    (("ttnn.matmul", "ttnn.add"), "matmul_bias", "ttnn.linear_with_bias"),
    (("ttnn.mul", "ttnn.rsqrt", "ttnn.mul"), "rmsnorm", "ttnn.rms_norm"),
]


def _find_pattern(seq: Sequence[str], pattern: Sequence[str]) -> List[int]:
    starts: List[int] = []
    n = len(pattern)
    for i in range(len(seq) - n + 1):
        if tuple(seq[i : i + n]) == tuple(pattern):
            starts.append(i)
    return starts


class FusionRewriter:
    name = "fusion_rewriter"
    level = 5
    requires: List[str] = ["program_config_tuner"]

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box) -> List[Finding]:  # noqa: ARG002
        findings: List[Finding] = []
        # Group by block to keep pattern scope coherent.
        block_to_rows = {}
        for r in joined:
            block_to_rows.setdefault(r.block_path, []).append(r)
        for block, rows in block_to_rows.items():
            seq = [r.op_code for r in rows]
            for pat, label, fused_op in PATTERNS:
                starts = _find_pattern(seq, pat)
                for s in starts:
                    chain = rows[s : s + len(pat)]
                    cost_ms = sum((r.device_kernel_ns or 0) for r in chain) / 1e6
                    findings.append(
                        Finding(
                            block_name=self.name,
                            cluster_id=",".join(r.cluster_id or "?" for r in chain)[:96],
                            block_path=block,
                            severity=Severity.INFO,
                            cost_ms=cost_ms,
                            evidence={"pattern": label, "fused_op": fused_op, "chain": list(pat)},
                            suggestion=(
                                f"{len(pat)}-op pattern matches `{label}`; replace with `{fused_op}` "
                                "(v1 emits diff for manual apply)."
                            ),
                            source_locations=[SourceLocation(path=source.demo_test_path, func=block)],
                            expected_gain_ms=cost_ms * 0.3,
                        )
                    )
        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:  # noqa: ARG002
        patches: List[Patch] = []
        for f in findings:
            patches.append(
                Patch(
                    kind=PatchKind.SOURCE_REWRITE,
                    target=SourceLocation(path=source.demo_test_path, func=f.block_path),
                    diff_text=(
                        f"# v1: fusion {f.evidence.get('pattern')} -> {f.evidence.get('fused_op')}\n"
                        f"# Manual step: replace the {len(f.evidence.get('chain', []))}-op chain in "
                        f"{f.block_path} with `{f.evidence.get('fused_op')}`.\n"
                    ),
                    rationale=(
                        f"Pattern {f.evidence.get('pattern')} found; expected "
                        f"-{f.expected_gain_ms:.2f} ms after fusion"
                    ),
                    extra={"finding": f.cluster_id, "v1_manual": True},
                )
            )
        return patches

    def verify(
        self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource
    ) -> VerificationResult:  # noqa: ARG002
        b = sum(r.device_kernel_ns or 0 for r in before)
        a = sum(r.device_kernel_ns or 0 for r in after)
        delta_ms = (a - b) / 1e6
        return VerificationResult(
            ok=delta_ms < 0,
            delta_ms=delta_ms,
            accuracy_held=True,
            reason=("fusion helped" if delta_ms < 0 else "no improvement; revert manually"),
        )
