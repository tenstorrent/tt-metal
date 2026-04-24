# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


KeyCategory = Literal["expert", "non_expert"]

_EXPERT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.ffn\.experts\.(?P<expert>\d+)\.(?P<projection>w1|w2|w3)\.(?P<kind>weight|scale)$"
)

_SEGMENT_MAP = {
    "embed_tokens": "embed",
    "self_attn": "attn",
    "mlp": "ffn",
    "input_layernorm": "attn_norm",
    "post_attention_layernorm": "ffn_norm",
    "q_proj": "wq",
    "q_a_proj": "wq_a",
    "q_a_layernorm": "q_norm",
    "q_b_proj": "wq_b",
    "kv_a_proj_with_mqa": "wkv_a",
    "kv_a_layernorm": "kv_norm",
    "kv_b_proj": "wkv_b",
    "o_proj": "wo",
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
    "lm_head": "head",
    "weight_scale_inv": "scale",
    "e_score_correction_bias": "bias",
}


@dataclass(frozen=True)
class MappedKey:
    source: str
    canonical: str
    category: KeyCategory
    layer: int | None = None
    expert: int | None = None
    projection: str | None = None
    tensor_kind: str | None = None


def normalize_hf_key(key: str) -> MappedKey:
    if not key:
        raise ValueError("Weight key must be non-empty")
    canonical = _normalize_key_string(key)
    expert_match = _EXPERT_RE.match(canonical)
    if expert_match is not None:
        return MappedKey(
            source=key,
            canonical=canonical,
            category="expert",
            layer=int(expert_match.group("layer")),
            expert=int(expert_match.group("expert")),
            projection=expert_match.group("projection"),
            tensor_kind=expert_match.group("kind"),
        )
    return MappedKey(source=key, canonical=canonical, category="non_expert")


def expert_packed_key(canonical_expert_key: str) -> str:
    mapped = normalize_hf_key(canonical_expert_key)
    if mapped.category != "expert" or mapped.tensor_kind != "weight":
        raise ValueError(f"Expected routed expert weight key, got {canonical_expert_key}")
    return canonical_expert_key.removesuffix(".weight") + ".weight_packed"


def _normalize_key_string(key: str) -> str:
    if key == "lm_head.weight":
        return "head.weight"
    if key.startswith("model."):
        key = key[len("model.") :]
    if key == "lm_head.weight":
        return "head.weight"

    key = key.replace("weight_scale_inv", "scale")
    key = key.replace("_scale_inv", ".scale")
    parts = key.split(".")
    normalized = [_SEGMENT_MAP.get(part, part) for part in parts]
    return ".".join(normalized)
