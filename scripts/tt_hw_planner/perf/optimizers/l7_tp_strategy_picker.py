# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""L7 tp_strategy_picker — pick TP factor + collective topology."""

from __future__ import annotations

from typing import List

from ..ceilings import BoxSpec
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


CCL_OPS = {"ttnn.all_gather", "ttnn.reduce_scatter", "ttnn.all_reduce"}


class TPStrategyPicker:
    name = "tp_strategy_picker"
    level = 7
    requires: List[str] = ["program_config_tuner", "layout_unifier"]

    def diagnose(self, joined: List[JoinedRow], source: ModelSource, box: BoxSpec) -> List[Finding]:  # noqa: ARG002
        findings: List[Finding] = []
        ccl_ns = sum((r.device_kernel_ns or 0) for r in joined if r.op_code in CCL_OPS)
        total_ns = sum((r.device_kernel_ns or 0) for r in joined)
        if total_ns == 0:
            return findings
        ccl_frac = ccl_ns / total_ns
        if box.total_chips > 1 and ccl_frac > 0.20:
            findings.append(
                Finding(
                    block_name=self.name,
                    cluster_id="mesh",
                    block_path="root",
                    severity=Severity.WARN,
                    cost_ms=ccl_ns / 1e6,
                    evidence={
                        "ccl_fraction": ccl_frac,
                        "total_chips": box.total_chips,
                        "current_mesh_shape": list(box.mesh_shape),
                    },
                    suggestion=(
                        f"Collectives account for {ccl_frac*100:.0f}% of device time on a "
                        f"{box.mesh_shape[0]}x{box.mesh_shape[1]} mesh. Consider a hierarchical "
                        "topology or a different TP factor."
                    ),
                    source_locations=[SourceLocation(path=source.demo_test_path, variable="MESH_DEVICE")],
                    expected_gain_ms=ccl_ns / 1e6 * 0.3,
                )
            )
        return findings

    def propose(self, findings: List[Finding], source: ModelSource) -> List[Patch]:  # noqa: ARG002
        patches: List[Patch] = []
        for f in findings:
            patches.append(
                Patch(
                    kind=PatchKind.TUNING_TABLE_ENTRY,
                    target=SourceLocation(path=source.demo_test_path, variable="MESH_DEVICE"),
                    new_kwargs={"collective_topology": "hierarchical"},
                    rationale=(
                        f"CCL frac {f.evidence.get('ccl_fraction')*100:.1f}%; "
                        f"expected -{f.expected_gain_ms:.2f} ms with hierarchical topology"
                    ),
                    extra={"finding": f.cluster_id},
                )
            )
        return patches

    def verify(
        self, before: List[JoinedRow], after: List[JoinedRow], source: ModelSource
    ) -> VerificationResult:  # noqa: ARG002
        def ccl_total(rows):
            return sum((r.device_kernel_ns or 0) for r in rows if r.op_code in CCL_OPS)

        b = ccl_total(before)
        a = ccl_total(after)
        delta_ms = (a - b) / 1e6
        return VerificationResult(
            ok=delta_ms < 0,
            delta_ms=delta_ms,
            accuracy_held=True,
            reason=("collective time shrank" if delta_ms < 0 else "no improvement; reverting"),
        )
