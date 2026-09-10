# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Indexed safetensors access with layer-local canonical names.

Canonical (layer-local) names produced by :meth:`KimiCheckpoint.layer_state_dict`:
  attention (KDA):  q_proj.weight k_proj.weight v_proj.weight q_conv1d.weight k_conv1d.weight v_conv1d.weight A_log dt_bias
                    f_a_proj.weight f_b_proj.weight g_a_proj.weight g_b_proj.weight b_proj.weight o_norm.weight o_proj.weight
  attention (MLA):  q_proj.weight kv_a_proj_with_mqa.weight kv_a_layernorm.weight kv_b_proj.weight o_proj.weight
  norms:            input_layernorm.weight post_attention_layernorm.weight
  dense MLP:        mlp.gate_proj.weight mlp.up_proj.weight mlp.down_proj.weight
  MoE:              moe.gate.weight moe.gate.e_score_correction_bias moe.shared.{gate,up,down}_proj.weight
                    moe.experts.gate [E, I, H]  moe.experts.up [E, I, H]  moe.experts.down [E, H, I]   (stacked, torch weight layout)
Top level: embed_tokens.weight, norm.weight, lm_head.weight.
The HF checkpoint's own prefixes are discovered from the index, so w1/w3/w2 vs gate/up/down naming differences are absorbed here.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open

from models.autoports.moonshotai_kimi_linear_48b_a3b_instruct.reference.config import KimiLinearConfig

KDA_NAMES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "A_log",
    "dt_bias",
    "f_a_proj.weight",
    "f_b_proj.weight",
    "g_a_proj.weight",
    "g_b_proj.weight",
    "b_proj.weight",
    "o_norm.weight",
    "o_proj.weight",
)
MLA_NAMES = ("q_proj.weight", "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight", "o_proj.weight")
# expert projection aliases in checkpoint order -> canonical
_EXPERT_ALIASES = {"gate": ("w1", "gate_proj"), "up": ("w3", "up_proj"), "down": ("w2", "down_proj")}
_SHARED_ALIASES = {"gate_proj": ("gate_proj", "w1"), "up_proj": ("up_proj", "w3"), "down_proj": ("down_proj", "w2")}


class KimiCheckpoint:
    """Lazy reader over the sharded safetensors of one snapshot directory."""

    def __init__(self, snapshot_dir: str | Path, config: KimiLinearConfig | None = None):
        self.dir = Path(snapshot_dir)
        self.config = config or KimiLinearConfig.from_snapshot(self.dir)
        index = json.load(open(self.dir / "model.safetensors.index.json"))
        self.weight_map: dict[str, str] = dict(index["weight_map"])
        self._files: dict[str, object] = {}
        self.layer_prefix = self._detect_layer_prefix()

    # ---- raw access ------------------------------------------------------------------------
    def _file(self, shard: str):
        f = self._files.get(shard)
        if f is None:
            f = safe_open(self.dir / shard, framework="pt", device="cpu")
            self._files[shard] = f
        return f

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def tensor(self, name: str) -> torch.Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError as e:
            raise KeyError(
                f"{name} not in checkpoint index; similar: {self.keys_like(name.rsplit('.', 2)[0])[:8]}"
            ) from e
        return self._file(shard).get_tensor(name)

    def keys_like(self, prefix: str) -> list[str]:
        return sorted(k for k in self.weight_map if k.startswith(prefix))

    def close(self) -> None:
        self._files.clear()

    # ---- naming -----------------------------------------------------------------------------
    def _detect_layer_prefix(self) -> str:
        for cand in ("model.layers.", "language_model.model.layers.", "model.language_model.layers."):
            if any(k.startswith(cand) for k in self.weight_map):
                return cand
        raise ValueError("could not find a decoder layer prefix in the safetensors index")

    def _lp(self, layer_idx: int) -> str:
        return f"{self.layer_prefix}{layer_idx}."

    def _first_existing(self, candidates: Iterable[str]) -> str:
        for c in candidates:
            if c in self.weight_map:
                return c
        raise KeyError(f"none of {list(candidates)} in checkpoint index")

    @property
    def top_prefix(self) -> str:
        return self.layer_prefix[: -len("layers.")]  # e.g. "model."

    # ---- canonical loaders ------------------------------------------------------------------
    def embedding(self) -> torch.Tensor:
        return self.tensor(self._first_existing([f"{self.top_prefix}embed_tokens.weight", "model.embed_tokens.weight"]))

    def final_norm(self) -> torch.Tensor:
        return self.tensor(self._first_existing([f"{self.top_prefix}norm.weight", "model.norm.weight"]))

    def lm_head(self) -> torch.Tensor:
        if self.config.tie_word_embeddings:
            return self.embedding()
        return self.tensor(self._first_existing(["lm_head.weight", f"{self.top_prefix}lm_head.weight"]))

    def attention_state_dict(self, layer_idx: int) -> dict[str, torch.Tensor]:
        p = self._lp(layer_idx) + "self_attn."
        names = KDA_NAMES if self.config.is_kda_layer(layer_idx) else MLA_NAMES
        return {n: self.tensor(p + n) for n in names}

    def norms_state_dict(self, layer_idx: int) -> dict[str, torch.Tensor]:
        p = self._lp(layer_idx)
        return {n: self.tensor(p + n) for n in ("input_layernorm.weight", "post_attention_layernorm.weight")}

    def _moe_prefix(self, layer_idx: int) -> str:
        p = self._lp(layer_idx)
        for cand in ("block_sparse_moe.", "mlp."):
            if any(k.startswith(p + cand + "experts.") for k in self.weight_map):
                return p + cand
        raise KeyError(f"layer {layer_idx}: no MoE experts under {p}(block_sparse_moe|mlp).experts.")

    def dense_mlp_state_dict(self, layer_idx: int) -> dict[str, torch.Tensor]:
        p = self._lp(layer_idx) + "mlp."
        out = {}
        for canon, aliases in _SHARED_ALIASES.items():
            out[f"mlp.{canon}.weight"] = self.tensor(self._first_existing([p + a + ".weight" for a in aliases]))
        return out

    def moe_state_dict(self, layer_idx: int, experts: bool = True) -> dict[str, torch.Tensor]:
        cfg = self.config
        p = self._moe_prefix(layer_idx)
        out = {
            "moe.gate.weight": self.tensor(p + "gate.weight"),
            "moe.gate.e_score_correction_bias": self.tensor(p + "gate.e_score_correction_bias"),
        }
        if cfg.num_shared_experts:
            sp = p + "shared_experts."
            for canon, aliases in _SHARED_ALIASES.items():
                out[f"moe.shared.{canon}.weight"] = self.tensor(
                    self._first_existing([sp + a + ".weight" for a in aliases])
                )
        if experts:
            stacked = {k: [] for k in _EXPERT_ALIASES}
            for e in range(cfg.num_experts):
                ep = f"{p}experts.{e}."
                for canon, aliases in _EXPERT_ALIASES.items():
                    stacked[canon].append(self.tensor(self._first_existing([ep + a + ".weight" for a in aliases])))
            for canon in _EXPERT_ALIASES:
                out[f"moe.experts.{canon}"] = torch.stack(stacked[canon], dim=0)  # [E, out, in] torch layout
        return out

    def layer_state_dict(self, layer_idx: int, experts: bool = True) -> dict[str, torch.Tensor]:
        sd = {}
        sd.update(self.norms_state_dict(layer_idx))
        sd.update(self.attention_state_dict(layer_idx))
        if self.config.is_moe_layer(layer_idx):
            sd.update(self.moe_state_dict(layer_idx, experts=experts))
        else:
            sd.update(self.dense_mlp_state_dict(layer_idx))
        return sd

    def expert_tensor(self, layer_idx: int, expert: int, which: str) -> torch.Tensor:
        """Single expert projection ``which`` in {gate, up, down}; torch layout [out, in]."""
        ep = f"{self._moe_prefix(layer_idx)}experts.{expert}."
        return self.tensor(self._first_existing([ep + a + ".weight" for a in _EXPERT_ALIASES[which]]))

    def describe_layer(self, layer_idx: int) -> list[str]:
        p = self._lp(layer_idx)
        return sorted(
            {re.sub(r"experts\.\d+\.", "experts.N.", k[len(p) :]) for k in self.weight_map if k.startswith(p)}
        )


@lru_cache(maxsize=4)
def open_checkpoint(snapshot_dir: str) -> KimiCheckpoint:
    return KimiCheckpoint(snapshot_dir)
