# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HF reference for the full multimodal model (CPU): `EmbeddingGemma2Model` + ST Pooling(mean) + Normalize.

Needs transformers >= 5.20 (overlay on PYTHONPATH, see hf_reference.py). The model is the unmodified
HF class loaded with every tower (`from_pretrained`, fp32 by default). `embed` feeds it the processor
output of one input (`tt.embedding_model.process_item`, the sentence-transformers formatting) and
returns the sentence-transformers embedding: masked mean over every token (include_prompt) then L2
normalize, optionally MRL-truncated and re-normalized. ST itself was checked against this path on
the test inputs (doc/full_model/README.md, "Reference").
"""

from __future__ import annotations

import torch

from models.demos.embedding_gemma2.reference.hf_reference import _check_transformers, sentence_embedding

MODEL_INPUTS = (
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_position_ids",
    "pixel_values_videos",
    "video_position_ids",
    "num_frames_per_video",
    "input_features",
    "input_features_mask",
)


def load_reference_full_model(snapshot, dtype=torch.float32):
    """Default attention implementation per sub-model, as sentence-transformers loads it (sdpa for the
    text and vision towers; the audio tower has eager only)."""
    _check_transformers()
    from transformers import AutoModel

    model = AutoModel.from_pretrained(str(snapshot), dtype=dtype)
    if model.vision_tower is None or model.audio_tower is None:
        raise RuntimeError("reference model was built without a vision or audio tower")
    return model.eval()


def model_inputs(features: dict, dtype=torch.float32) -> dict:
    out = {}
    for k in MODEL_INPUTS:
        if k in features:
            v = features[k]
            out[k] = v.to(dtype) if torch.is_floating_point(v) else v
    return out


def truncate(emb: torch.Tensor, dim: int | None) -> torch.Tensor:
    """MRL: keep the first `dim` values and re-normalize (model card, Best Practices 3)."""
    if dim is None or dim == emb.shape[-1]:
        return emb
    return torch.nn.functional.normalize(emb[..., :dim], p=2, dim=-1)


@torch.no_grad()
def embed(model, features: dict, truncate_dim: int | None = None) -> torch.Tensor:
    """[B, d] float32 ST embedding of processor `features`."""
    dtype = next(model.parameters()).dtype
    out = model(**model_inputs(features, dtype))
    emb = sentence_embedding(out.last_hidden_state.float(), features["attention_mask"])
    return truncate(emb, truncate_dim)
