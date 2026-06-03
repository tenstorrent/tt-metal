from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional, TypedDict


class PatternKind(str, Enum):
    CHAIN = "chain"  # contiguous op-subsequence
    HORIZONTAL_MERGE = "horizontal_merge"  # N sibling branches sharing one input


def _resolve(value, dims: dict[str, int]):
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        return dims[value[1:-1]]
    return value


@dataclass(eq=True)
class KBEntry:
    id: str
    fused_op: str
    category: str
    pattern_kind: PatternKind
    torch_pattern: list[str]
    signature: dict[str, Any]
    config_template: dict[str, Any]
    weight_transform: Optional[str]
    source: str
    usage_examples: list[str] = field(default_factory=list)
    applicability_notes: str = ""
    status: str = "in_use"  # in_use | supported_unused
    accumulation_sensitive: bool = False
    pattern_source: str = "unit_test"  # golden | unit_test | llm — provenance of torch_pattern
    confidence: str = "high"  # high (tier1 golden / tier2 unit-test) | low (tier3 llm)
    unit_test_refs: list = field(default_factory=list)  # op unit test(s) to reuse/parameterize at bring-up
    placement_observations: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pattern_kind"] = self.pattern_kind.value
        d["placement_observations"] = [
            o.to_dict() if isinstance(o, PlacementObservation) else o for o in self.placement_observations
        ]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "KBEntry":
        d = dict(d)
        d["pattern_kind"] = PatternKind(d["pattern_kind"])
        # Keep placement_observations as raw dicts for portability; callers that need
        # PlacementObservation objects call PlacementObservation.from_dict() themselves.
        d["placement_observations"] = list(d.get("placement_observations", []))
        return cls(**d)


@dataclass(eq=True)
class FusionProposal:
    entry_id: str
    fused_op: str
    matched_nodes: list[str]
    config: dict[str, Any]
    weight_transform: Optional[str]
    rationale: str
    source: str

    def resolve(self, dims: dict[str, int]) -> "FusionProposal":
        return FusionProposal(
            entry_id=self.entry_id,
            fused_op=self.fused_op,
            matched_nodes=list(self.matched_nodes),
            config={k: _resolve(v, dims) for k, v in self.config.items()},
            weight_transform=self.weight_transform,
            rationale=self.rationale,
            source=self.source,
        )


@dataclass(eq=True)
class Diagnosis:
    node: str
    axis: str  # "per_block_pcc" | "long_decode_drift"
    measured: float
    config_tried: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.axis in ("per_block_pcc", "long_decode_drift"), self.axis


class BringupState(TypedDict, total=False):
    model: str
    graph_summary: list
    proposals: list
    applied: list
    placements: dict
    per_block_pcc: dict
    full_pcc: float
    drift: dict
    perf: dict
    perf_warnings: list
    diagnosis: dict
    iteration: int
    max_iterations: int
    status: str  # "running" | "pass" | "handoff"
    run_dir: str


@dataclass(eq=True)
class PlacementObservation:
    op: str
    tensor_role: str
    size_descriptor: dict  # {"dims","dtype","bytes_expr"}
    memory_config: dict  # {"buffer":"L1"|"DRAM","layout":..., "shard_spec_template":...}
    program_config: Optional[str]
    condition: Optional[dict]  # {"var","op","value"} or None
    source: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PlacementObservation":
        return cls(**d)


@dataclass(eq=True)
class MemoryPlacement:
    buffer: str  # "L1" | "DRAM"
    layout: str = "interleaved"  # interleaved | width_sharded | block_sharded
    shard_spec: Optional[dict] = None
