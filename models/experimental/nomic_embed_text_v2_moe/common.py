# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pinned revision, checkpoint resolution, contracts and test-input helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass(frozen=True)
class ModelReference:
    model_id: str
    revision: str


@dataclass(frozen=True)
class CheckpointContract:
    n_tensors: int
    n_parameters: int


@dataclass(frozen=True)
class TokenizerContract:
    length: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int


@dataclass(frozen=True)
class ModelCardExample:
    sentences: tuple[str, str]
    cosine_similarity: float
    tolerance: float


@dataclass(frozen=True)
class ParityThresholds:
    pcc: float
    max_abs: float


MODEL = ModelReference(
    model_id="nomic-ai/nomic-embed-text-v2-moe",
    revision="1066b6599d099fbb93dfcb64f9c37a7c9e503e85",
)

# Asserted by tests/pcc/test_checkpoint_contract.py.
CHECKPOINT = CheckpointContract(
    n_tensors=148,
    n_parameters=475_292_928,
)

# length is smaller than config.vocab_size, which is padded up to pad_vocab_size_multiple,
# so the embedding table has unreachable trailing rows.
TOKENIZER = TokenizerContract(
    length=250002,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
)

# The model card's worked example: cosine similarity between the passage-prefixed embeddings
# of these two sentences. The card prints 0.9118; the reference reproduces 0.911788.
MODEL_CARD = ModelCardExample(
    sentences=("Hello!", "¡Hola!"),
    cosine_similarity=0.9118,
    tolerance=1e-4,
)

# Thresholds against upstream. The reference is bit-exact in practice (max-abs 0.0), so these
# leave room only for fp32 non-determinism. Loosening one is a regression, not a tolerance
# adjustment.
PARITY = ParityThresholds(
    pcc=0.9999999,
    max_abs=1e-4,
)


def resolve_checkpoint(allow_download: bool = True) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=MODEL.model_id,
            filename="model.safetensors",
            revision=MODEL.revision,
            local_files_only=not allow_download,
        )
    )


def resolve_config(allow_download: bool = True) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=MODEL.model_id,
            filename="config.json",
            revision=MODEL.revision,
            local_files_only=not allow_download,
        )
    )


def checkpoint_is_cached() -> bool:
    try:
        resolve_checkpoint(allow_download=False)
        return True
    except Exception:
        return False


def load_tokenizer():
    """AutoTokenizer is safe here even though AutoModel is not: tokenizer_config.json's
    explicit tokenizer_class outranks the nomic_bert model-type mapping."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL.model_id, revision=MODEL.revision)


def capture_hidden_states(model: torch.nn.Module, module_paths: list[str]) -> tuple[dict, list]:
    """Register forward hooks on module_paths, returning (captures, handles).

    End-to-end PCC can hide compensating errors, so parity is checked at every layer boundary
    instead. The caller must remove the handles.
    """
    captures: dict[str, torch.Tensor] = {}
    handles = []
    named = dict(model.named_modules())

    for path in module_paths:
        if path not in named:
            raise KeyError(f"no module at {path!r}")

        def make_hook(name: str):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if isinstance(tensor, torch.Tensor):
                    captures[name] = tensor.detach().clone()

            return hook

        handles.append(named[path].register_forward_hook(make_hook(path)))

    return captures, handles


def layer_ladder_paths(num_hidden_layers: int) -> list[str]:
    """Capture points for the parity ladder: post-embedding norm, then each block."""
    return ["emb_ln"] + [f"encoder.layers.{i}" for i in range(num_hidden_layers)]


def synthetic_state_dict(config, seed: int = 0) -> dict[str, torch.Tensor]:
    """A deterministic state dict matching the real key/shape contract.

    Lets the structural tests run with no network and no 1.8 GB download at the model's real
    dimensions. The model is never shrunk; only the weights are synthetic. Norm weights are
    ones and biases zeros so norms start as identity.
    """
    from models.experimental.nomic_embed_text_v2_moe.reference.loader import expected_checkpoint_keys

    generator = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {}
    for key, shape in expected_checkpoint_keys(config).items():
        is_norm = "norm" in key or "emb_ln" in key
        if is_norm:
            state[key] = torch.zeros(shape) if key.endswith(".bias") else torch.ones(shape)
        elif key.endswith(".bias") and not key.endswith("experts.bias"):
            state[key] = torch.zeros(shape)
        else:
            state[key] = torch.randn(shape, generator=generator) * 0.02
    return state


def build_synthetic_model(config, seed: int = 0):
    from models.experimental.nomic_embed_text_v2_moe.reference.loader import load_reference_model

    return load_reference_model(config, synthetic_state_dict(config, seed=seed))


def random_input_ids(
    batch: int,
    seqlen: int,
    config,
    seed: int = 0,
    pad_lengths: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (input_ids, attention_mask). pad_lengths right-pads row b by that many tokens."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, config.vocab_size, (batch, seqlen), generator=generator)
    attention_mask = torch.ones((batch, seqlen), dtype=torch.long)
    if pad_lengths is not None:
        for row, n_pad in enumerate(pad_lengths):
            if n_pad > 0:
                input_ids[row, seqlen - n_pad :] = config.pad_token_id
                attention_mask[row, seqlen - n_pad :] = 0
    return input_ids, attention_mask
