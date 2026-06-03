from dataclasses import dataclass, field
from pathlib import Path


def _repo_root() -> Path:
    # this file is <repo>/models/experimental/opt_transfer/config.py
    return Path(__file__).resolve().parents[3]


@dataclass
class Config:
    repo_root: Path = field(default_factory=_repo_root)
    kb_source_roots: tuple = (
        "models/tt_transformers",
        "models/tt_dit",
        "models/demos",
        "tests/ttnn/unit_tests/operations",
        "tech_reports",
    )
    kb_dir: Path = field(default=None)
    cache_dir: Path = field(default=None)
    run_dir: Path = field(default=None)
    matcher_model: str = "claude-opus-4-8"
    models: dict = field(
        default_factory=lambda: {
            "seamless_m4t_v2": {
                "embed_dim": 1024,
                "num_heads": 16,
                "head_dim": 64,
                "reference": "models.experimental.opt_transfer.references.seamless_m4t_v2",
            },
        }
    )
    gates: dict = field(
        default_factory=lambda: {
            "per_block_pcc": 0.99,
            "full_pcc": 0.99,
            "drift_first_divergence_min_frac": 0.9,
            "min_perf_gain_pct": 2.0,
            "max_iterations": 3,
            "placement_min_gain_pct": 2.0,
        }
    )
    l1_budgets: dict = field(
        default_factory=lambda: {
            # per-core usable L1 (conservative starting values; tuned against the perf gate)
            "wormhole_b0": {"per_core_bytes": 1024 * 1024, "num_cores": 64},
            "blackhole": {"per_core_bytes": 1400 * 1024, "num_cores": 130},
        }
    )

    def __post_init__(self):
        base = self.repo_root / "models/experimental/opt_transfer"
        self.kb_dir = self.kb_dir or base / "kb" / "records"
        self.cache_dir = self.cache_dir or base / ".cache"
        self.run_dir = self.run_dir or base / "runs"


CONFIG = Config()
