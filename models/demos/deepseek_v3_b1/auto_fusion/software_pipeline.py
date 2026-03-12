# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Software pipelining for auto-fusion graphs.

Computes the Minimum Initiation Interval (MII) and determines whether
overlapping reader/compute/writer across loop iterations is beneficial.

Key concepts:
- ResMII (resource-bound MII): max cycles any single RISC is busy
- RecMII (recurrence-bound MII): longest cycle in the dependence graph
- MII = max(ResMII, RecMII)
- Prologue: initial iterations to fill the pipeline
- Steady state: overlapped execution at MII rate
- Epilogue: drain remaining iterations

For single-core compute-only ops (like gated_local_reduce), pipelining
is NOT beneficial because all work is on TRISC with no reader/writer overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from models.demos.deepseek_v3_b1.auto_fusion.types import OpNode


@dataclass
class PipelineSchedule:
    """Result of software pipelining analysis."""

    mii: int  # Minimum Initiation Interval
    res_mii: int  # Resource-bound MII
    rec_mii: int  # Recurrence-bound MII
    is_beneficial: bool  # Whether pipelining helps
    prologue_length: int = 0  # Number of prologue iterations
    epilogue_length: int = 0  # Number of epilogue iterations
    steady_state_ops: List[str] = field(default_factory=list)  # Ops in steady state
    risc_utilization: Dict[str, float] = field(default_factory=dict)  # RISC -> utilization %
    reason: str = ""  # Why pipelining is/isn't beneficial


class SoftwarePipeliner:
    """
    Analyzes whether software pipelining is beneficial for a fusion graph.

    Software pipelining overlaps iterations of the fused kernel loop body
    so that reader (NCRISC), compute (TRISC), and writer (BRISC) can
    work on different iterations simultaneously.

    This is beneficial when:
    1. Multiple RISCs have significant work (not all noop)
    2. The ops form a pipeline (reader -> compute -> writer)
    3. The MII is less than the sequential execution time
    """

    def __init__(self, graph, ilp_result=None, sdf_result=None):
        """
        Args:
            graph: FusionGraph instance
            ilp_result: Optional ILPResult for schedule and latencies
            sdf_result: Optional SDFAnalyzer for buffer sizing
        """
        self._graph = graph
        self._nodes = {n.id: n for n in graph.nodes}
        self._edges = graph.edges
        self._schedule = graph.get_schedule()
        self._ilp = ilp_result
        self._sdf = sdf_result

    def _get_risc_latency(self, node: OpNode, risc: str) -> int:
        """Get estimated latency for a specific RISC on an op."""
        if node.spec.risc_latency:
            return node.spec.risc_latency.get(risc, 0)
        # Default estimates
        contract = getattr(node.spec, risc, None)
        if contract and contract.is_noop:
            return 0
        return 100

    def compute_mii(self) -> int:
        """
        Compute the Minimum Initiation Interval.

        MII = max(ResMII, RecMII)

        ResMII = max over all RISCs of (sum of that RISC's latencies)
        RecMII = longest cycle weight in the dependence graph (0 if acyclic)
        """
        res_mii = self._compute_res_mii()
        rec_mii = self._compute_rec_mii()
        return max(res_mii, rec_mii, 1)

    def _compute_res_mii(self) -> int:
        """Resource-bound MII: max RISC utilization across one iteration."""
        risc_totals = {"ncrisc": 0, "brisc": 0, "trisc": 0}
        for nid in self._schedule:
            node = self._nodes[nid]
            for risc in risc_totals:
                risc_totals[risc] += self._get_risc_latency(node, risc)
        return max(risc_totals.values()) if risc_totals else 1

    def _compute_rec_mii(self) -> int:
        """Recurrence-bound MII: longest cycle in dependence graph.

        For DAGs (no back-edges), RecMII = 0.
        For loops with carried dependencies, RecMII = cycle weight.
        """
        # Auto-fusion graphs are DAGs within a single invocation
        # Back-edges only exist in loop-carried dependencies
        # For now, return 0 (no loop-carried deps)
        return 0

    def is_beneficial(self) -> bool:
        """
        Determine whether software pipelining would improve throughput.

        Pipelining is NOT beneficial when:
        - All ops run on the same RISC (no overlap possible)
        - Only one RISC has meaningful work
        - The graph is too small (overhead exceeds benefit)
        """
        # Count RISCs with meaningful work
        risc_work = {"ncrisc": 0, "brisc": 0, "trisc": 0}
        for nid in self._schedule:
            node = self._nodes[nid]
            for risc in risc_work:
                lat = self._get_risc_latency(node, risc)
                if lat > 0:
                    risc_work[risc] += lat

        active_riscs = sum(1 for v in risc_work.values() if v > 0)

        # Need at least 2 active RISCs for overlap
        if active_riscs < 2:
            return False

        # Check if MII < sequential time
        mii = self.compute_mii()
        sequential = sum(risc_work.values())
        return mii < sequential * 0.8  # 20% improvement threshold

    def generate_schedule(self) -> PipelineSchedule:
        """
        Generate the full pipelining analysis.

        Returns:
            PipelineSchedule with MII, utilization, and benefit assessment.
        """
        res_mii = self._compute_res_mii()
        rec_mii = self._compute_rec_mii()
        mii = max(res_mii, rec_mii, 1)
        beneficial = self.is_beneficial()

        # Compute RISC utilization
        risc_totals = {"ncrisc": 0, "brisc": 0, "trisc": 0}
        for nid in self._schedule:
            node = self._nodes[nid]
            for risc in risc_totals:
                risc_totals[risc] += self._get_risc_latency(node, risc)

        utilization = {}
        for risc, total in risc_totals.items():
            utilization[risc] = (total / mii * 100) if mii > 0 else 0.0

        # Determine prologue/epilogue
        n_stages = len(self._schedule)
        prologue = n_stages - 1 if beneficial else 0
        epilogue = n_stages - 1 if beneficial else 0

        if not beneficial:
            reason = self._explain_not_beneficial(risc_totals)
        else:
            reason = (
                f"MII={mii} (ResMII={res_mii}, RecMII={rec_mii}). "
                f"RISC utilization: {', '.join(f'{r}={u:.0f}%' for r, u in utilization.items())}"
            )

        return PipelineSchedule(
            mii=mii,
            res_mii=res_mii,
            rec_mii=rec_mii,
            is_beneficial=beneficial,
            prologue_length=prologue,
            epilogue_length=epilogue,
            steady_state_ops=list(self._schedule),
            risc_utilization=utilization,
            reason=reason,
        )

    def _explain_not_beneficial(self, risc_totals: Dict[str, int]) -> str:
        """Generate human-readable explanation for why pipelining isn't beneficial."""
        active = [r for r, t in risc_totals.items() if t > 0]
        if len(active) <= 1:
            risc = active[0] if active else "none"
            return f"All work on {risc.upper()}, no cross-RISC overlap possible"
        mii = max(risc_totals.values())
        seq = sum(risc_totals.values())
        return f"MII={mii} vs sequential={seq}: insufficient speedup (need >20%)"
