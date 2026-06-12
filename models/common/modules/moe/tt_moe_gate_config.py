# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for ``TTMoEGate`` (the routing front-end).

Parsed from the same per-model YAML that ``TTMoEDecodeConfig`` reads — the gate-specific keys
(``n_group``/``score_func``/``routed_scaling_factor``/``score_correction_bias``/``router_bias``/
``gate_matmul_compute``) are exactly the extra keys ``TTMoEDecodeConfig`` ignores. Mirrors the
``TTMoEDecode(mesh_device, config, torch_weights...)`` entry-point convention: ``TTMoEGate`` takes this
config + the torch gate weight/bias(es), and builds everything on device internally.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel, ConfigDict

# Gate-relevant keys read from a model YAML; every other key (decode/compute/reduce/...) is ignored.
_GATE_YAML_KEYS = (
    "num_routed_experts",
    "select_experts_k",
    "hidden_size",
    "batch_per_device",
    "n_group",
    "score_func",
    "routed_scaling_factor",
    "score_correction_bias",
    "router_bias",
    "gate_matmul_compute",
    "gate_matmul_program_config",
)


class TTMoEGateConfig(BaseModel):
    """Gate shape + routing semantics for ``TTMoEGate``.

    ``batch_per_device`` is OPTIONAL: when set, ``TTMoEGate`` sizes its per-token op buffers to it, so the
    per-forward ``ttnn.slice`` of those buffers is a free full-tensor slice (zero perf cost — decode path).
    When ``None``, the buffers are sized to the device core count and the slice trims ``cores →
    batch_per_iter`` (the variable-/large-batch fallback). Either way the result is identical.
    """

    model_config = ConfigDict(frozen=True)

    # --- shape ---
    num_routed_experts: int
    select_experts_k: int
    hidden_size: int
    batch_per_device: int | None = None  # set → buffers sized to it (free slice); None → sized to cores

    # --- gate semantics (see TTMoEGate) ---
    n_group: int = 1  # 1 → generalized (ungrouped) op; 8 → deepseek (grouped) op
    score_func: str = "softmax"  # "softmax" | "sigmoid" | "sqrtsoftplus"
    routed_scaling_factor: float = 1.0
    score_correction_bias: bool = False  # deepseek/noaux_tc: e_score_correction_bias (selection-only)
    router_bias: bool = False  # gpt-oss: router LINEAR bias (into logits → selection AND weights)
    gate_matmul_compute: dict | None = None  # per-model router-matmul compute kernel config
    # per-model router-matmul program config (kernel grid / blocking) — a tuned 2D-mcast config can
    # materially cut the gate matmul's latency. `type` names the ttnn program-config class (default
    # MatmulMultiCoreReuseMultiCastProgramConfig); remaining keys are its kwargs. None → ttnn auto-picks.
    gate_matmul_program_config: dict | None = None
    eps: float = 1e-20

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "TTMoEGateConfig":
        """Build from a model YAML, picking only the gate-relevant keys (defaults fill the rest)."""
        raw = yaml.safe_load(yaml_text)
        return cls(**{k: raw[k] for k in _GATE_YAML_KEYS if k in raw})
