# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L8 async_dispatch_inserter — close host bubbles in the waterfall."""

from __future__ import annotations

from typing import List

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


BUBBLE_NS_THRESHOLD = 5_000  # 5 us


class AsyncDispatchInserter:
    name = "async_dispatch_inserter"
    level = 8
    requires: List[str] = ["trace_capturer", "tp_strategy_picker"]

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box) -> List[Finding]:  # noqa: ARG002
        findings: List[Finding] = []
        bubble_total_ns = 0
        bubble_count = 0
        for r in joined:
            gap = r.op_to_op_latency_ns or 0
            kernel = r.device_kernel_ns or 0
            if gap > BUBBLE_NS_THRESHOLD and (kernel == 0 or gap > 0.5 * kernel):
                bubble_total_ns += gap
                bubble_count += 1
        if bubble_count < 5:
            return findings
        findings.append(
            Finding(
                block_name=self.name,
                cluster_id="waterfall",
                block_path="root",
                severity=Severity.WARN,
                cost_ms=bubble_total_ns / 1e6,
                evidence={"bubble_count": bubble_count, "bubble_total_ns": bubble_total_ns},
                suggestion=(
                    f"{bubble_count} host bubbles >5us totaling "
                    f"{bubble_total_ns/1e6:.2f} ms. Insert async dispatch wrappers "
                    "and double-buffer the CB to hide the dispatch pipe."
                ),
                source_locations=[SourceLocation(path=source.demo_test_path, func="iteration")],
                expected_gain_ms=bubble_total_ns / 1e6 * 0.6,
            )
        )
        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:  # noqa: ARG002
        patches: List[Patch] = []
        for f in findings:
            patches.append(
                Patch(
                    kind=PatchKind.SOURCE_REWRITE,
                    target=SourceLocation(path=source.demo_test_path, func="iteration"),
                    diff_text=(
                        "# v1: async dispatch + CB double-buffer\n"
                        "# Manual step: wrap iteration body in `with ttnn.async_dispatch(...):`\n"
                    ),
                    rationale=f"close {f.evidence.get('bubble_count')} bubbles; expected -{f.expected_gain_ms:.2f} ms",
                    extra={"finding": f.cluster_id, "v1_manual": True},
                )
            )
        return patches

    def verify(
        self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource
    ) -> VerificationResult:  # noqa: ARG002
        def bubbles(rows):
            return sum((r.op_to_op_latency_ns or 0) for r in rows if (r.op_to_op_latency_ns or 0) > BUBBLE_NS_THRESHOLD)

        b = bubbles(before)
        a = bubbles(after)
        delta_ms = (a - b) / 1e6
        return VerificationResult(
            ok=delta_ms < 0,
            delta_ms=delta_ms,
            accuracy_held=True,
            reason=("bubbles shrank" if delta_ms < 0 else "bubbles did not shrink"),
        )
