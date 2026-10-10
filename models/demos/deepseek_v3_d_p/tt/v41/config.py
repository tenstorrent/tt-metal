# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash model config for the prefill, read from the checkpoint's own ``inference/config.json`` (the file
``inference/model.py`` builds ``ModelArgs`` from), plus the per-layer CSA2 roles.

The role rules are the reference's (``model.py`` ``Attention`` / ``Indexer`` constructors; mirrored by tt-blaze
``blaze/models/deepseek_v4_1_flash/layer_schedule.py``):

* ``compress_ratios[l]``: 0 = sliding window only; r > 0 = also one compressed KV entry per r tokens.
* KV sources compute a compressed KV (and the index keys); every later compressing layer reads the latest source's cache.
* Index sources run an indexer and publish top-k; the others reuse the latest index source's top-k.
* Layer 20 (the candidate source) is the first DECODER layer: its compressor projects the encoder output into the ratio-1
  KV every decoder layer shares. The prefill runs layers 0..19 in full and layer 20 KV-only (DS41F-0037 M1).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_MODEL_DIR = os.environ.get("DEEPSEEK_V4_V41_HF_MODEL", "/mnt/tt-data/sdawle/models/DeepSeek-V4.1-Flash")
SWA, FULL, REINDEX, REUSE = "swa", "full", "reindex", "reuse"


@dataclass(frozen=True)
class LayerRole:
    layer: int
    compress_ratio: int
    mode: str  # swa | full | reindex | reuse
    kv_source: int | None  # whose compressed KV this layer attends to (itself for Full)
    index_source: int | None  # whose top-k this layer uses (itself for Full / Reindex)
    is_candidate_source: bool
    uses_candidates: bool
    engram: bool


@dataclass(frozen=True)
class V41Config:
    dim: int
    n_layers: int
    n_heads: int
    head_dim: int
    rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    n_routed_experts: int
    n_activated_experts: int
    moe_inter_dim: int
    route_scale: float
    score_func: str
    swiglu_limit: float
    window_size: int
    compress_ratios: tuple
    kv_source_layers: tuple
    index_source_layers: tuple
    candidate_source_layer: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    rope_theta: float
    compress_rope_theta: float
    original_seq_len: int
    rope_factor: float
    beta_fast: float
    beta_slow: float
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    norm_eps: float
    vocab_size: int
    engram_layer_ids: tuple
    raw: dict = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def load(cls, model_dir: str = DEFAULT_MODEL_DIR) -> "V41Config":
        c = json.load(open(os.path.join(model_dir, "inference", "config.json")))
        n = int(c["n_layers"])
        kw = {k: c[k] for k in cls.__dataclass_fields__ if k not in ("raw",) and k in c}
        kw["compress_ratios"] = tuple(
            int(r) for r in c["compress_ratios"][:n]
        )  # the list carries the DSpark layers too
        for k in ("kv_source_layers", "index_source_layers", "engram_layer_ids"):
            kw[k] = tuple(int(v) for v in c[k])
        return cls(**kw, raw=c)

    def role(self, layer: int) -> LayerRole:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer {layer} is not a backbone layer (0..{self.n_layers - 1})")
        ratio = self.compress_ratios[layer]

        def latest(src):
            prior = [s for s in src if s <= layer]
            return prior[-1] if prior else None

        if ratio == 0:
            mode, kv_src, idx_src = SWA, None, None
        else:
            kv_src, idx_src = latest(self.kv_source_layers), latest(self.index_source_layers)
            mode = FULL if layer in self.kv_source_layers else (REINDEX if layer in self.index_source_layers else REUSE)
            if kv_src is None or idx_src is None or self.compress_ratios[kv_src] != ratio:
                raise ValueError(f"layer {layer}: inconsistent KV / index sources ({kv_src}, {idx_src})")
        return LayerRole(
            layer=layer,
            compress_ratio=ratio,
            mode=mode,
            kv_source=kv_src,
            index_source=idx_src,
            is_candidate_source=layer == self.candidate_source_layer,
            uses_candidates=ratio > 0
            and 0 <= self.candidate_source_layer < layer
            and layer in self.index_source_layers,
            engram=layer in self.engram_layer_ids,
        )

    @property
    def first_decoder_layer(self) -> int:
        return self.candidate_source_layer

    @property
    def encoder_layers(self) -> range:
        return range(self.first_decoder_layer)
