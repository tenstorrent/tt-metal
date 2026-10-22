# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for ``TTMoEGate`` (the routing front-end).

Parsed from the same per-model YAML that ``TTMoEDecodeConfig`` reads — the gate-specific keys
(``n_group``/``score_func``/``routed_scaling_factor``/``score_correction_bias``/``gate_proj_bias``/
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
    "softmax_position",
    "routed_scaling_factor",
    "score_correction_bias",
    "gate_proj_bias",
    "gate_matmul_high_fidelity",
    "gate_matmul_compute",
    "gate_matmul_auto_program_config",
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
    # softmax ONLY (ignored otherwise) — where the softmax is applied relative to the top-k. With no score-
    # correction bias both are mathematically equal (the global Z cancels under renorm), differing only in
    # how the exp is computed (a selection bias would make them diverge, but softmax models are bias-free):
    #   "post" (default) — rank by the raw logit, then softmax OVER THE SELECTED top-k (the kernel op's
    #                      exp-over-selected / ttnn.softmax(sel) in the fallback). The original behavior.
    #   "pre"            — softmax over ALL experts UP FRONT (same slot as sqrtsoftplus), then LINEAR renorm
    #                      over the selected top-k. The exp is the numerically-stable full softmax
    #                      (max-subtracted over all experts) instead of the kernel's exp-over-raw-logit.
    softmax_position: str = "post"  # "post" | "pre"
    routed_scaling_factor: float = 1.0
    score_correction_bias: bool = False  # deepseek/noaux_tc: e_score_correction_bias (selection-only)
    gate_proj_bias: bool = (
        False  # gpt-oss: model has a router LINEAR/projection bias b (logits = Wx + b → selection AND weights)
    )
    # router-matmul COMPUTE kernel fidelity. Default HIGH fidelity (HiFi2 + fp32 accumulate): the deep gate
    # reduction drifts under the ttnn default (LoFi + bf16 accumulate) and flips near-tied experts at the
    # top-k boundary, so every gate gets high fidelity (the matmul is tiny → the cost is negligible). Set
    # `gate_matmul_high_fidelity: false` to opt out (ttnn's default fidelity).
    gate_matmul_high_fidelity: bool = True
    # OPTIONAL full override of the compute kernel: a dict (keys math_fidelity / math_approx_mode /
    # fp32_dest_acc_en / packer_l1_acc override the HiFi2 defaults). Wins over gate_matmul_high_fidelity
    # when set; None → use the flag above. See TTMoEGate._matmul_compute_config.
    gate_matmul_compute: dict | None = None
    # router-matmul PROGRAM config (kernel grid / 2D-mcast blocking). Default AUTO: derived from the gate
    # matmul shape + device grid (TTMoEGate._derive_program_config) — a tuned config materially cuts the gate
    # matmul latency vs ttnn's auto pick. Set `gate_matmul_auto_program_config: false` for ttnn's auto-select.
    gate_matmul_auto_program_config: bool = True
    # OPTIONAL full override of the program config: a dict whose `type` names the ttnn program-config class
    # (default MatmulMultiCoreReuseMultiCastProgramConfig) and remaining keys are its kwargs
    # (compute_with_storage_grid_size auto-filled). Wins over the auto flag when set; None → auto / ttnn.
    gate_matmul_program_config: dict | None = None
    eps: float = 1e-20

    @classmethod
    def from_yaml(cls, yaml_text: str) -> "TTMoEGateConfig":
        """Build from a model YAML, picking only the gate-relevant keys (defaults fill the rest)."""
        raw = yaml.safe_load(yaml_text)
        return cls(**{k: raw[k] for k in _GATE_YAML_KEYS if k in raw})
