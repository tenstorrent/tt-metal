from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from ..join import JoinedRow
from .base import Finding, ModelSource, Patch, Severity, SourceLocation, VerificationResult


class OpCoverageGuard:
    name = "op_coverage_guard"
    level = 0
    requires: List[str] = []

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box) -> List[Finding]:
        findings: List[Finding] = []
        by_op_missing_kernel: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0.0, "cost_ms": 0.0})

        tracer_present = any(r.tracer_op_name for r in joined)
        tracer_missing_rows: List[JoinedRow] = []

        for r in joined:
            missing_kernel = (
                not (r.compute_kernel_hash or "").strip()
                and not (r.dm_kernel_hash or "").strip()
                and not (r.compute_kernel_source or "").strip()
                and not (r.dm_kernel_source or "").strip()
            )
            if missing_kernel:
                key = r.op_code or "unknown_op"
                by_op_missing_kernel[key]["count"] += 1.0
                by_op_missing_kernel[key]["cost_ms"] += float(r.device_kernel_ns or 0.0) / 1e6

            if tracer_present and not r.tracer_op_name:
                tracer_missing_rows.append(r)

        for op_code, stats in sorted(by_op_missing_kernel.items(), key=lambda kv: kv[1]["count"], reverse=True):
            findings.append(
                Finding(
                    block_name=self.name,
                    cluster_id=f"coverage:{op_code}",
                    block_path="root",
                    severity=Severity.BLOCKER,
                    cost_ms=float(stats["cost_ms"]),
                    evidence={
                        "kind": "missing_kernel_signature",
                        "op_code": op_code,
                        "rows": int(stats["count"]),
                    },
                    suggestion=(
                        f"{op_code}: {int(stats['count'])} row(s) missing kernel identity; "
                        "add kernel support or decompose into supported TT ops."
                    ),
                    source_locations=[SourceLocation(path=source.demo_test_path)],
                    expected_gain_ms=0.0,
                )
            )

        if tracer_missing_rows:
            grouped: Dict[Tuple[str, str], int] = defaultdict(int)
            for r in tracer_missing_rows:
                grouped[(r.op_code or "unknown_op", r.module_path or r.block_path or "root")] += 1
            for (op_code, scope), count in sorted(grouped.items(), key=lambda kv: kv[1], reverse=True)[:32]:
                findings.append(
                    Finding(
                        block_name=self.name,
                        cluster_id=f"coverage-tracer:{op_code}:{scope}",
                        block_path=scope,
                        severity=Severity.WARN,
                        cost_ms=0.0,
                        evidence={
                            "kind": "tracer_unmapped",
                            "op_code": op_code,
                            "scope": scope,
                            "rows": count,
                        },
                        suggestion=(
                            f"{op_code} in {scope}: {count} row(s) not mapped by model_tracer; "
                            "check op naming parity or tracer coverage before tuning."
                        ),
                        source_locations=[SourceLocation(path=source.demo_test_path)],
                        expected_gain_ms=0.0,
                    )
                )

        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:
        return []

    def verify(self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource) -> VerificationResult:
        return VerificationResult(ok=True, delta_ms=0.0, accuracy_held=True, reason="coverage_guard is diagnostic only")
