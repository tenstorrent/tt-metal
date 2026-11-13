# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4Config: every dimension of tencent/Hy4-preview (HYV4ForCausalLM), read from the checkpoint ``config.json``.

CPU only (no ttnn import). One object replaces the module-level literals (5120 / 2304 / 384 / ...) of the DeepSeek-V4.1-Flash tree.
Anything the checkpoint config does not state is derived here and marked UNVERIFIED where a formula is a convention not yet confirmed by the oracle.

Facts below that come straight from config.json / tensor shapes are exact. Mesh facts (4x8 Galaxy, 8 experts per chip, TP8 columns) come from the port plan.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DEFAULT_CKPT = "/mnt/tt-data/Hy4-Preview"
TILE = 32

# Galaxy mesh (port plan section 2): 4 rows x 8 columns, 32 chips.
MESH_ROWS = 4
MESH_COLS = 8
N_CHIPS = MESH_ROWS * MESH_COLS

FULL = "full"
SHARED = "shared"
DENSE = "dense"
MOE = "moe"


@dataclass(frozen=True)
class LayerSpec:
    """One decoder layer. ``index`` 0..num_hidden_layers-1; the MTP layer has index == num_hidden_layers and ``is_mtp``."""

    index: int
    mlp: str  # DENSE | MOE
    indexer: str  # FULL | SHARED
    indexer_source: int  # layer index whose top-k ids this layer uses (itself when FULL)
    is_mtp: bool = False

    @property
    def has_indexer_weights(self) -> bool:
        return self.indexer == FULL

    @property
    def has_hc(self) -> bool:
        """The checkpoint carries hc_attn_layer / hc_mlp_layer tensors for the 78 main layers only (no hc tensors under model.mtp_layers.0)."""
        return not self.is_mtp


class Hy4ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Hy4Config:
    # --- identity
    architecture: str = "HYV4ForCausalLM"
    model_type: str = "hy_v4"
    hf_id: str = "tencent/Hy4-preview"
    # --- core dims
    hidden_size: int = 6144
    num_hidden_layers: int = 78
    vocab_size: int = 120832
    intermediate_size: int = 18432  # dense layers (layer 0)
    moe_intermediate_size: int = 2048
    rms_norm_eps: float = 1e-5
    # --- attention (gated MLA)
    num_attention_heads: int = 64
    num_key_value_heads: int = 8  # present in config.json; no tensor shape uses it (port plan 0.1, UNVERIFIED)
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    qk_head_dim: int = 256
    v_head_dim: int = 256
    head_dim: int = 64  # config.json field; equals qk_rope_head_dim here
    gated_mla: bool = True
    gating_type: str = "elementwise"
    learnable_sink: bool = True
    learnable_sink_init: float = 0.0
    # --- rope
    rope_theta: float = 1e7
    rope_type: str = "default"
    max_position_embeddings: int = 1048576
    # --- indexer (DSA + IndexCache)
    use_dsa: bool = True
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    indexer_types: Tuple[str, ...] = ()
    # --- iHC
    enable_ihc: bool = True
    hc_mult: int = 4
    hc_eps: float = 1e-6
    hc_magnitude: float = 2.0
    # --- MoE
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.827
    swiglu_limit: float = 10.0
    mlp_layer_types: Tuple[str, ...] = ()  # "dense" | "sparse" as in config.json
    # --- MTP
    num_nextn_predict_layers: int = 1
    # --- head / tokens
    tie_word_embeddings: bool = False
    enable_lm_head_fp32: bool = True
    bos_token_id: int = 120000
    eos_token_id: int = 120025
    pad_token_id: int = 120002
    # --- placement (port plan; mesh is not in config.json)
    mesh_rows: int = MESH_ROWS
    mesh_cols: int = MESH_COLS

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_dict(cls, d: dict) -> "Hy4Config":
        rope = d.get("rope_parameters") or {}
        kw = dict(
            architecture=(d.get("architectures") or ["HYV4ForCausalLM"])[0],
            model_type=d.get("model_type", "hy_v4"),
            hidden_size=d["hidden_size"],
            num_hidden_layers=d["num_hidden_layers"],
            vocab_size=d["vocab_size"],
            intermediate_size=d["intermediate_size"],
            moe_intermediate_size=d["moe_intermediate_size"],
            rms_norm_eps=d["rms_norm_eps"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            q_lora_rank=d["q_lora_rank"],
            kv_lora_rank=d["kv_lora_rank"],
            qk_nope_head_dim=d["qk_nope_head_dim"],
            qk_rope_head_dim=d["qk_rope_head_dim"],
            qk_head_dim=d["qk_head_dim"],
            v_head_dim=d["v_head_dim"],
            head_dim=d.get("head_dim", d["qk_rope_head_dim"]),
            gated_mla=d.get("gated_mla", False),
            gating_type=d.get("gating_type", "elementwise"),
            learnable_sink=d.get("learnable_sink", False),
            learnable_sink_init=d.get("learnable_sink_init", 0.0),
            rope_theta=float(rope.get("rope_theta", d.get("rope_theta", 1e7))),
            rope_type=rope.get("rope_type", "default"),
            max_position_embeddings=d["max_position_embeddings"],
            use_dsa=d.get("use_dsa", False),
            index_n_heads=d["index_n_heads"],
            index_head_dim=d["index_head_dim"],
            index_topk=d["index_topk"],
            indexer_types=tuple(d["indexer_types"]),
            enable_ihc=d.get("enable_ihc", False),
            hc_mult=d["hc_mult"],
            hc_eps=d["hc_eps"],
            hc_magnitude=d["hc_magnitude"],
            n_routed_experts=d["n_routed_experts"],
            n_shared_experts=d["n_shared_experts"],
            num_experts_per_tok=d["num_experts_per_tok"],
            n_group=d["n_group"],
            topk_group=d["topk_group"],
            norm_topk_prob=d["norm_topk_prob"],
            routed_scaling_factor=d["routed_scaling_factor"],
            swiglu_limit=d["swiglu_limit"],
            mlp_layer_types=tuple(d["mlp_layer_types"]),
            num_nextn_predict_layers=d.get("num_nextn_predict_layers", 0),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            enable_lm_head_fp32=d.get("enable_lm_head_fp32", False),
            bos_token_id=d["bos_token_id"],
            eos_token_id=d["eos_token_id"],
            pad_token_id=d["pad_token_id"],
        )
        cfg = cls(**kw)
        cfg.validate()
        return cfg

    @classmethod
    def from_checkpoint(cls, ckpt_dir: Optional[str] = None) -> "Hy4Config":
        ckpt_dir = ckpt_dir or os.environ.get("HY4_CKPT", DEFAULT_CKPT)
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            return cls.from_dict(json.load(f))

    # ------------------------------------------------------------------ validation
    def validate(self) -> None:
        errs: List[str] = []

        def need(cond: bool, msg: str) -> None:
            if not cond:
                errs.append(msg)

        L = self.num_hidden_layers
        need(
            len(self.indexer_types) == L, f"indexer_types has {len(self.indexer_types)} entries, num_hidden_layers={L}"
        )
        need(
            len(self.mlp_layer_types) == L,
            f"mlp_layer_types has {len(self.mlp_layer_types)} entries, num_hidden_layers={L}",
        )
        need(set(self.indexer_types) <= {FULL, SHARED}, f"unknown indexer type in {set(self.indexer_types)}")
        need(set(self.mlp_layer_types) <= {"dense", "sparse"}, f"unknown mlp layer type in {set(self.mlp_layer_types)}")
        if self.indexer_types:
            need(
                self.indexer_types[0] == FULL,
                "layer 0 must have a full indexer (shared layers take ids from an earlier full layer)",
            )
        need(self.qk_head_dim == self.qk_nope_head_dim + self.qk_rope_head_dim, "qk_head_dim != nope + rope")
        need(self.head_dim == self.qk_rope_head_dim, "head_dim (config) != qk_rope_head_dim")
        need(self.num_attention_heads % self.mesh_cols == 0, "heads not divisible by TP columns")
        need(self.hidden_size % (TILE * self.mesh_cols) == 0, "hidden_size not tile-divisible per TP column")
        need(
            self.moe_intermediate_size % (TILE * self.mesh_cols) == 0,
            "moe_intermediate_size not tile-divisible per TP column",
        )
        need(
            self.intermediate_size % (TILE * self.mesh_cols) == 0,
            "dense intermediate_size not tile-divisible per TP column",
        )
        need(self.vocab_size % (TILE * self.mesh_cols) == 0, "vocab_size not tile-divisible per TP column")
        need(
            self.n_routed_experts % self.n_chips == 0,
            f"{self.n_routed_experts} experts do not divide over {self.n_chips} chips",
        )
        need(self.num_experts_per_tok <= self.n_routed_experts, "topk > experts")
        need(
            self.n_group == 1 and self.topk_group == 1,
            "grouped routing (n_group/topk_group != 1) not supported by this port",
        )
        need(self.gating_type == "elementwise", f"gating_type {self.gating_type!r}: only elementwise is planned")
        need(self.rope_type == "default", f"rope_type {self.rope_type!r}: only default (no YaRN) is planned")
        need(self.hc_mult == 4, "hc_mult != 4: the iHC kernels are planned for 4 streams")
        need(self.enable_ihc, "enable_ihc false: plain residual path not planned")
        need(self.index_topk <= self.max_position_embeddings, "index_topk > max position")
        need(self.num_nextn_predict_layers in (0, 1), "more than one MTP layer is not planned")
        need(not self.tie_word_embeddings, "tied embeddings not planned (checkpoint has a separate lm_head)")
        need(self.max_position_embeddings % 512 == 0, "max context is not a multiple of the 512 prefill chunk")
        if self.mlp_layer_types:
            need(self.mlp_layer_types[0] == "dense", "layer 0 expected dense")
        if errs:
            raise Hy4ConfigError("; ".join(errs))

    # ------------------------------------------------------------------ mesh
    @property
    def n_chips(self) -> int:
        return self.mesh_rows * self.mesh_cols

    @property
    def experts_per_chip(self) -> int:
        """256 / 32 = 8."""
        return self.n_routed_experts // self.n_chips

    def expert_chip(self, e: int) -> int:
        """Linear chip id (row * mesh_cols + col) that holds routed expert ``e``. Contiguous blocks of ``experts_per_chip``: UNVERIFIED placement convention (DSV4 moe_place.py may differ)."""
        if not 0 <= e < self.n_routed_experts:
            raise IndexError(e)
        return e // self.experts_per_chip

    def expert_mesh_coord(self, e: int) -> Tuple[int, int]:
        c = self.expert_chip(e)
        return divmod(c, self.mesh_cols)

    def chip_experts(self, chip: int) -> range:
        n = self.experts_per_chip
        return range(chip * n, (chip + 1) * n)

    # ------------------------------------------------------------------ TP8 column shards (per column)
    @property
    def tp(self) -> int:
        return self.mesh_cols

    @property
    def hidden_per_col(self) -> int:
        return self.hidden_size // self.tp

    @property
    def heads_per_col(self) -> int:
        """8 local heads per column."""
        return self.num_attention_heads // self.tp

    @property
    def shared_inter_per_col(self) -> int:
        """Shared expert (moe_intermediate 2048) / 8 = 256 = 8 tiles."""
        return self.moe_intermediate_size // self.tp

    @property
    def dense_inter_per_col(self) -> int:
        """Dense layer 0: 18432 / 8 = 2304 = 72 tiles."""
        return self.intermediate_size // self.tp

    @property
    def vocab_per_col(self) -> int:
        """120832 / 8 = 15104 = 472 tiles."""
        return self.vocab_size // self.tp

    # ------------------------------------------------------------------ derived tensor dims
    @property
    def q_b_out(self) -> int:
        return self.num_attention_heads * self.qk_head_dim  # 16384

    @property
    def kv_a_out(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim  # 576

    @property
    def kv_b_out(self) -> int:
        return self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim)  # 28672

    @property
    def attn_out_dim(self) -> int:
        """Width of the attention output before o_proj, and of linear_gate (elementwise gate): heads * v_head_dim = 16384."""
        return self.num_attention_heads * self.v_head_dim

    @property
    def indexer_q_out(self) -> int:
        return self.index_n_heads * self.index_head_dim  # 4096 (wq_b)

    @property
    def hc_fn_rows(self) -> int:
        """pre (hc_mult) + post (hc_mult) = 8; no comb matrix (port plan 0.4)."""
        return 2 * self.hc_mult

    @property
    def hc_flat_dim(self) -> int:
        return self.hc_mult * self.hidden_size  # 24576

    @property
    def kv_row_width(self) -> int:
        """Absorbed-MLA latent row 576 (UNVERIFIED vs oracle)."""
        return self.kv_a_out

    # ------------------------------------------------------------------ tile counts
    def tiles(self, n: int) -> int:
        if n % TILE:
            raise Hy4ConfigError(f"{n} is not a multiple of {TILE}")
        return n // TILE

    # ------------------------------------------------------------------ per-layer tables
    @property
    def n_total_layers(self) -> int:
        """Main layers plus the MTP layer(s)."""
        return self.num_hidden_layers + self.num_nextn_predict_layers

    def layer(self, i: int) -> LayerSpec:
        return self.layer_table[i]

    @property
    def layer_table(self) -> Tuple[LayerSpec, ...]:
        return _build_layer_table(self)

    @property
    def main_layers(self) -> Tuple[LayerSpec, ...]:
        return self.layer_table[: self.num_hidden_layers]

    @property
    def mtp_layers(self) -> Tuple[LayerSpec, ...]:
        return self.layer_table[self.num_hidden_layers :]

    @property
    def full_indexer_layers(self) -> Tuple[int, ...]:
        """21 main layers with their own indexer weights (0, 1, 5, 9, ..., 77)."""
        return tuple(l.index for l in self.main_layers if l.indexer == FULL)

    @property
    def dense_layers(self) -> Tuple[int, ...]:
        return tuple(l.index for l in self.main_layers if l.mlp == DENSE)

    @property
    def moe_layers(self) -> Tuple[int, ...]:
        return tuple(l.index for l in self.main_layers if l.mlp == MOE)

    def indexer_source(self, i: int) -> int:
        return self.layer_table[i].indexer_source

    # ------------------------------------------------------------------ misc
    @property
    def router_scale_attr(self) -> float:
        return self.routed_scaling_factor

    def summary(self) -> Dict[str, object]:
        return {
            "layers": self.num_hidden_layers,
            "dense_layers": list(self.dense_layers),
            "full_indexer_layers": len(self.full_indexer_layers),
            "experts": f"{self.n_routed_experts}/{self.n_chips} = {self.experts_per_chip} per chip",
            "heads_per_col": self.heads_per_col,
        }


def _build_layer_table(cfg: Hy4Config) -> Tuple[LayerSpec, ...]:
    cache = _TABLES.get(cfg)
    if cache is not None:
        return cache
    rows: List[LayerSpec] = []
    last_full = -1
    for i in range(cfg.num_hidden_layers):
        ind = cfg.indexer_types[i]
        if ind == FULL:
            last_full = i
        rows.append(
            LayerSpec(
                index=i,
                mlp=DENSE if cfg.mlp_layer_types[i] == "dense" else MOE,
                indexer=ind,
                indexer_source=last_full if ind == SHARED else i,
            )
        )
    for k in range(cfg.num_nextn_predict_layers):
        i = cfg.num_hidden_layers + k
        # MTP layer: own full indexer (indexer.* tensors present), MoE FFN, no hc tensors.
        rows.append(LayerSpec(index=i, mlp=MOE, indexer=FULL, indexer_source=i, is_mtp=True))
    out = tuple(rows)
    _TABLES[cfg] = out
    return out


_TABLES: Dict["Hy4Config", Tuple[LayerSpec, ...]] = {}
