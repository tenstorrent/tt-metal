# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Config for one GLM-5.2 MTP (Multi-Token-Prediction) module.

Values come from the GLM-5.2 HF checkout's ``config.json`` (architecture ``GlmMoeDsaForCausalLM``,
model_type ``glm_moe_dsa``). Read directly as JSON rather than through ``AutoConfig``:
``glm_moe_dsa`` is not AutoConfig-loadable on the transformers versions on this box, which is why
``runners/adapters/glm_5_2.py`` also hand-rolls its config load.

Deliberately small. It carries only what the MTP module itself needs; the decoder layer inside the
module is configured by the ordinary GLM config/model_config pair, unchanged. See issue #53533.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MTPConfig:
    """Geometry and layout for MTP prefill.

    Attributes:
        hidden_size: model hidden dim (6144). ``eh_proj`` is ``[hidden_size, 2 * hidden_size]``.
        rms_norm_eps: epsilon for ``enorm`` / ``hnorm`` / ``shared_head.norm``.
        mtp_layer_idx: layer index the MTP weights live on. GLM-5.2 stores them at
            ``model.layers.78.*``, i.e. one past the last trunk layer (``num_hidden_layers`` is 78,
            layers 0..77). Verified against the checkpoint index, not assumed.
        num_weight_modules: how many distinct sets of MTP *weights* the checkpoint carries, from
            ``num_nextn_predict_layers``. For GLM-5.2 this is **1**, and it is a trap: it does NOT
            mean one prediction level. See :attr:`num_levels`.
        num_levels: how many prediction levels to *run*. This is a serving choice, not a checkpoint
            fact — GLM-5.2 ships one weight module and MTP4 runs it at four levels (the
            DeepSeek-V3 paper scheme: K levels at ONE position, K KV caches, one shared weight
            module; not EAGLE-style autoregressive drafting). Stage 1 runs 1.
        index_share_for_mtp_iteration: from ``config.json``; True for GLM-5.2. The MTP layer's
            lightning-indexer top-k is computed once and reused across MTP levels rather than
            recomputed per level. A no-op at ``num_levels == 1``; load-bearing from level 2 on.
        first_k_dense_replace: dense-layer count (3). ``mtp_layer_idx >= first_k_dense_replace``, so
            the MTP layer is MoE — layer 78 really does carry 256 routed experts in fp8.
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
        """Whether the MTP layer is an MoE layer — the same test ``TtPrefillBlock`` applies."""
        return self.mtp_layer_idx >= self.first_k_dense_replace

    @property
    def concat_dim(self) -> int:
        """Width of the concatenated ``[enorm(embed), hnorm(hidden)]`` activation (12288)."""
        return 2 * self.hidden_size

    def shares_weights_across_levels(self) -> bool:
        """True when fewer weight modules than levels, i.e. one module is replayed per level."""
        return self.num_weight_modules < self.num_levels

    @classmethod
    def from_hf_config(cls, c, *, num_levels: int | None = None) -> "MTPConfig":
        """Build from an already-loaded HF config (attribute object or plain dict).

        Args:
            c: the GLM-5.2 config, as either a mapping or an attribute-style object
                (``glm_5_2_hf_config()`` returns a ``SimpleNamespace``).
            num_levels: prediction levels to run. Defaults to 1 — deliberately NOT
                ``num_nextn_predict_layers``, which counts weight modules and would silently pin
                MTP4 to one level.
        """
        get = c.get if isinstance(c, dict) else (lambda k, d=None: getattr(c, k, d))
        d = cls()
        num_hidden_layers = int(get("num_hidden_layers", d.mtp_layer_idx))
        return cls(
            hidden_size=int(get("hidden_size", d.hidden_size)),
            rms_norm_eps=float(get("rms_norm_eps", d.rms_norm_eps)),
            # MTP weights sit one past the last trunk layer. from_pretrained() verifies this against
            # the checkpoint index; from a bare config object it is arithmetic and stays unchecked.
            mtp_layer_idx=num_hidden_layers,
            num_weight_modules=int(get("num_nextn_predict_layers", d.num_weight_modules)),
            num_levels=int(num_levels) if num_levels is not None else d.num_levels,
            index_share_for_mtp_iteration=bool(get("index_share_for_mtp_iteration", d.index_share_for_mtp_iteration)),
            first_k_dense_replace=int(get("first_k_dense_replace", d.first_k_dense_replace)),
        )

    @classmethod
    def from_pretrained(cls, path: str, *, num_levels: int | None = None) -> "MTPConfig":
        """Build from a HF checkpoint directory (``$GLM52_HF_MODEL``) and verify the weights exist.

        ``mtp_layer_idx`` is derived as ``num_hidden_layers``, then checked against the checkpoint's
        own tensor index — a checkout with no MTP weights fails here with a readable message rather
        than at the first mesh-mapper call.
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
