# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for one GLM-5.2 MTP module.

Values are read straight out of the HF checkout's config.json: glm_moe_dsa is not
AutoConfig-loadable on the transformers versions here.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MTPConfig:
    """Geometry and layout for MTP prefill.

    ``num_weight_modules`` comes from the checkpoint and counts weight modules, not prediction levels;
    ``num_levels`` is the serving choice of how many levels to run over those weights.
    """

    hidden_size: int = 6144
    rms_norm_eps: float = 1e-5
    mtp_layer_idx: int = 78
    num_weight_modules: int = 1
    num_levels: int = 1
    index_share_for_mtp_iteration: bool = True
    first_k_dense_replace: int = 3

    def __post_init__(self) -> None:
        assert self.hidden_size > 0, f"hidden_size must be positive, got {self.hidden_size}"
        assert self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        assert self.num_weight_modules >= 1, f"num_weight_modules must be >= 1, got {self.num_weight_modules}"

    @property
    def is_moe_layer(self) -> bool:
        """Whether the MTP layer is an MoE layer -- the same test ``TtPrefillBlock`` applies."""
        return self.mtp_layer_idx >= self.first_k_dense_replace

    @property
    def concat_dim(self) -> int:
        """Width of the concatenated ``[enorm(embed), hnorm(hidden)]`` activation (12288)."""
        return 2 * self.hidden_size

    @classmethod
    def from_hf_config(cls, c, *, num_levels: int | None = None) -> "MTPConfig":
        """Build from an already-loaded HF config, either a mapping or an attribute object.

        ``num_levels`` defaults to 1, not to ``num_nextn_predict_layers``, which counts weight modules.
        """
        get = c.get if isinstance(c, dict) else (lambda k, d=None: getattr(c, k, d))
        d = cls()
        num_hidden_layers = int(get("num_hidden_layers", d.mtp_layer_idx))
        return cls(
            hidden_size=int(get("hidden_size", d.hidden_size)),
            rms_norm_eps=float(get("rms_norm_eps", d.rms_norm_eps)),
            mtp_layer_idx=num_hidden_layers,
            num_weight_modules=int(get("num_nextn_predict_layers", d.num_weight_modules)),
            num_levels=int(num_levels) if num_levels is not None else d.num_levels,
            index_share_for_mtp_iteration=bool(get("index_share_for_mtp_iteration", d.index_share_for_mtp_iteration)),
            first_k_dense_replace=int(get("first_k_dense_replace", d.first_k_dense_replace)),
        )

    @classmethod
    def from_pretrained(cls, path: str, *, num_levels: int | None = None) -> "MTPConfig":
        """Build from a HF checkpoint directory and confirm the MTP weights are really there.

        ``mtp_layer_idx`` is derived from the layer count, then checked against the checkpoint's index.
        """
        with open(os.path.join(path, "config.json")) as f:
            cfg = json.load(f)
        out = cls.from_hf_config(cfg, num_levels=num_levels)
        out.assert_weights_present(path)
        return out

    def assert_weights_present(self, path: str) -> None:
        """Assert the checkpoint at ``path`` actually carries ``eh_proj`` on :attr:`mtp_layer_idx`."""
        from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import _resolve_weight_map

        key = f"model.layers.{self.mtp_layer_idx}.eh_proj.weight"
        weight_map, is_sharded = _resolve_weight_map(path)
        if not is_sharded:
            from safetensors import safe_open

            with safe_open(os.path.join(path, "model.safetensors"), framework="pt") as f:
                present = key in set(f.keys())
        else:
            present = key in weight_map
        assert present, (
            f"{path} carries no MTP weights: expected {key}. "
            f"num_hidden_layers={self.mtp_layer_idx} implies MTP lives on layer {self.mtp_layer_idx}; "
            "either this is not an MTP-carrying checkout or the layout differs."
        )
