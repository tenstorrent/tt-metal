# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Token ids → the exact input of Kimi-K3 ``model.layers.0``'s KDA module.

The first decoder layer receives ``input_layernorm(embed_tokens[token_ids])``, so real tokens can drive it from
two checkpoint tensors without recorded activations. Deeper layers' inputs follow earlier layers, MoE and
AttnRes, so only ``layers.0`` is built here. (``KIMI_K3_FIRST_KDA_LAYER = 1`` elsewhere counts KDA layers from
checkpoint ``layers.1``; this is checkpoint ``layers.0``.)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import (
    kda_state_dict_sha256,
    load_checkpoint_tensor,
    resolve_model_root,
)

# Names what ``FirstLayerInput`` computes, for golden identities: a golden built another way is a different input.
FIRST_LAYER_INPUT = "input_layernorm(embed_tokens), vLLM rounding (FP32, one BF16 rounding)"


class FirstLayerInput:
    """``input_layernorm(embed_tokens[ids])`` for ``model.layers.0``, as the BF16 tensor the layer receives.

    Loads the embedding table (2.3 GB BF16) and the norm weight once, so a long sequence can be built segment
    by segment without holding all of it.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        self.eps = config.get("text_config", config)["rms_norm_eps"]
        root = resolve_model_root(checkpoint_dir)
        self.embedding = load_checkpoint_tensor(checkpoint_dir, f"{root}embed_tokens.weight")
        self.norm_weight = load_checkpoint_tensor(checkpoint_dir, f"{root}layers.0.input_layernorm.weight")

    def weights_sha256(self) -> str:
        """Content identity of the two tensors this input is built from (embedding table and norm weight)."""
        return kda_state_dict_sha256(
            {"embed_tokens.weight": self.embedding, "layers.0.input_layernorm.weight": self.norm_weight}
        )

    def __call__(self, token_ids: torch.Tensor | list[int]) -> torch.Tensor:
        """``[1, T, hidden]`` in the embedding dtype (BF16), for 1-D ``token_ids`` of length T."""
        embeddings = self.embedding[torch.as_tensor(token_ids, dtype=torch.long)]
        # vLLM's RMSNorm: normalize and scale in FP32, then round once to BF16. (HF KimiRMSNorm rounds the
        # normalized value before scaling; on recorded vLLM rows that differs in 26% of elements by one ulp.)
        hidden = embeddings.float()
        hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.eps) * self.norm_weight.float()
        return hidden.to(embeddings.dtype).unsqueeze(0)
