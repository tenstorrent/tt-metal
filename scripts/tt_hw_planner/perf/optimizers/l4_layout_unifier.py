# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L4 layout_unifier — kill redundant reshard/tilize/untilize chains."""

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


_LAYOUT_OPS = {"ttnn.reshard", "ttnn.tilize", "ttnn.untilize", "ttnn.layout"}


class LayoutUnifier:
    name = "layout_unifier"
    level = 4
    requires: List[str] = ["dram_l1_promoter", "program_config_tuner"]

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box) -> List[Finding]:  # noqa: ARG002
        findings: List[Finding] = []
        run_count = 0
        cluster_set = set()
        cost_ms = 0.0
        for i, r in enumerate(joined):
            if r.op_code in _LAYOUT_OPS:
                run_count += 1
                cluster_set.add(r.cluster_id or f"?_{i}")
                cost_ms += (r.device_kernel_ns or 0) / 1e6
            else:
                if run_count >= 2:
                    findings.append(
                        Finding(
                            block_name=self.name,
                            cluster_id=",".join(sorted(cluster_set))[:64],
                            block_path=r.block_path,
                            severity=Severity.INFO,
                            cost_ms=cost_ms,
                            evidence={
                                "consecutive_layout_ops": run_count,
                                "clusters": sorted(cluster_set),
                            },
                            suggestion=(
                                f"Chain of {run_count} consecutive "
                                "reshard/tilize/untilize ops; unify the layout "
                                "between the producer and consumer."
                            ),
                            source_locations=[
                                SourceLocation(
                                    path="models/tt_transformers/tt/model_config.py", variable="layout_chain"
                                )
                            ],
                            expected_gain_ms=cost_ms * 0.7,
                        )
                    )
                run_count = 0
                cluster_set = set()
                cost_ms = 0.0
        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:  # noqa: ARG002
        patches: List[Patch] = []
        for f in findings:
            patches.append(
                Patch(
                    kind=PatchKind.TUNING_TABLE_ENTRY,
                    target=SourceLocation(path="models/tt_transformers/tt/model_config.py", variable="layout_chain"),
                    new_kwargs={"layout": "tile_unified", "clusters": f.evidence.get("clusters", [])},
                    rationale=f"unify {f.evidence.get('consecutive_layout_ops')}-op layout chain (-{f.expected_gain_ms:.2f} ms)",
                    extra={"finding": f.cluster_id},
                )
            )
        return patches

    def verify(
        self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource
    ) -> VerificationResult:  # noqa: ARG002
        def layout_total(rows):
            return sum((r.device_kernel_ns or 0) for r in rows if r.op_code in _LAYOUT_OPS)

        b = layout_total(before)
        a = layout_total(after)
        delta_ms = (a - b) / 1e6
        return VerificationResult(
            ok=delta_ms < 0,
            delta_ms=delta_ms,
            accuracy_held=True,
            reason=("layout-op time shrank" if delta_ms < 0 else "layout time did not shrink; reverting"),
        )
