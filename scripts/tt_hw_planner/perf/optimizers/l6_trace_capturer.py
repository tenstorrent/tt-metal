# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L6 trace_capturer — wrap the inner loop in a metal trace.

Diagnoses Region F clusters (dispatch-bound) and clusters with low
program_cache_hit, then proposes a KWARG_REPLACE that flips the demo's
`--enable_trace` flag on. The patch lives in the run dir so revert is
single-file deletion.
"""

from __future__ import annotations

from typing import List

from ..cluster import Cluster
from ..join import JoinedRow
from .base import (
    Finding,
    ModelSource,
    OptimizerBlock,
    Patch,
    PatchKind,
    Severity,
    SourceLocation,
    VerificationResult,
)


class TraceCapturer:
    name = "trace_capturer"
    level = 6
    requires: List[str] = []

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box) -> List[Finding]:  # noqa: ARG002
        findings: List[Finding] = []
        # group by cluster id to dedupe
        seen = set()
        for r in joined:
            if r.cluster_id in seen:
                continue
            seen.add(r.cluster_id)
            if r.region != "F":
                continue
            cost_ms = (r.op_to_op_latency_ns or 0) / 1e6
            cache_rate = 1.0 if r.program_cache_hit else 0.0
            findings.append(
                Finding(
                    block_name=self.name,
                    cluster_id=r.cluster_id or "?",
                    block_path=r.block_path,
                    severity=Severity.WARN,
                    cost_ms=cost_ms,
                    evidence={
                        "op_to_op_latency_ns": r.op_to_op_latency_ns,
                        "device_kernel_ns": r.device_kernel_ns,
                        "program_cache_hit_rate": cache_rate,
                        "region_reason": r.region_reason,
                    },
                    suggestion=(
                        "Wrap the iteration loop in `--enable_trace` (and a "
                        "create_device_trace context) so dispatch overhead "
                        "is amortized across iterations."
                    ),
                    source_locations=[
                        SourceLocation(path=source.demo_test_path, func="iteration", variable="enable_trace")
                    ],
                    expected_gain_ms=cost_ms * 0.6,
                )
            )
        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:
        if not findings:
            return []
        target = SourceLocation(path=source.demo_test_path, variable="--enable_trace")
        rationale = (
            f"Region F detected on {len(findings)} clusters; enabling metal "
            f"trace removes per-op dispatch overhead. "
            f"Expected aggregate gain: {sum(f.expected_gain_ms for f in findings):.1f} ms."
        )
        return [
            Patch(
                kind=PatchKind.KWARG_REPLACE,
                target=target,
                new_kwargs={"enable_trace": True, "trace_region_size_hint": "100MB"},
                diff_text=(
                    "--- demo invocation\n+++ demo invocation\n"
                    "@@ pytest args @@\n"
                    "- --disable_trace\n"
                    "+ --enable_trace\n"
                ),
                rationale=rationale,
                extra={"findings": [f.cluster_id for f in findings]},
            )
        ]

    def verify(
        self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource
    ) -> VerificationResult:  # noqa: ARG002
        before_total = sum((r.device_kernel_ns or 0) + (r.op_to_op_latency_ns or 0) for r in before)
        after_total = sum((r.device_kernel_ns or 0) + (r.op_to_op_latency_ns or 0) for r in after)
        delta_ms = (after_total - before_total) / 1e6
        ok = delta_ms < 0
        return VerificationResult(
            ok=ok,
            delta_ms=delta_ms,
            accuracy_held=True,
            reason=("trace enabled" if ok else "no improvement; reverting"),
        )
