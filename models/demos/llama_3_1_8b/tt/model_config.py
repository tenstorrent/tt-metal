# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint resolution and the safetensors loader (P1).

**Weight resolution** follows recipe §4 step 0, in order: ``PREFILL_HF_MODEL`` / ``HF_MODEL`` if set,
then the shared store ``/mnt/models/meta-llama/Llama-3.1-8B-Instruct``, then fail. Nothing downloads
weights and nothing substitutes a synthetic checkpoint — random weights are for the component tests,
never for an end-to-end number.

**The loader** is a plain safetensors walk. There is no dequantization step: the checkpoint has no
``quantization_config`` and stores bf16 tensors directly, which
``tests/torch_ref/test_reference_llama.py::test_checkpoint_is_not_quantized`` asserts rather than
assumes. The QKV fusion and the Meta-RoPE permutation the TT modules need are NOT done here — they
live in ``tt/attention/weights.py``, next to the sharding that depends on them, so the state dict
this returns is still the checkpoint's own and can be fed straight to the torch reference.

``LLAMA_LOAD_NLAYERS=K`` loads only the shards holding layers ``0..K-1`` plus the non-layer tensors.
That is a debugging aid for isolation runs (the full checkpoint is 16 GB across four shards); any
number produced under it is a reduced run and must be labelled as such.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

SHARED_STORE = "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"

# Tensors this model has. Anything else in the checkpoint is a surprise worth failing on rather than
# dropping silently — a renamed key is how a "why is layer 7 wrong" afternoon starts.
_EXPECTED_NON_LAYER = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
_EXPECTED_PER_LAYER = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}


def resolve_checkpoint(path: Optional[str] = None) -> Path:
    """Resolve the real checkpoint directory, or raise with what was tried."""
    candidates = [path, os.getenv("PREFILL_HF_MODEL"), os.getenv("HF_MODEL"), SHARED_STORE]
    tried = []
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        tried.append(str(p))
        if (p / "model.safetensors.index.json").exists() or list(p.glob("*.safetensors")):
            return p
    raise FileNotFoundError(
        "no real Llama-3.1-8B-Instruct checkpoint found. Tried: "
        + ", ".join(tried or ["(nothing)"])
        + ". Set PREFILL_HF_MODEL / HF_MODEL, or stage the checkpoint under "
        + SHARED_STORE
        + ". This pipeline does not download weights and does not substitute a synthetic checkpoint."
    )


def load_state_dict(weights_path=None, *, num_layers: Optional[int] = None, dtype=torch.bfloat16) -> dict:
    """Read the checkpoint into a flat HF-key state dict (``model.*`` / ``lm_head.weight``).

    ``num_layers`` (or ``LLAMA_LOAD_NLAYERS``) restricts the load to layers ``0..num_layers-1``, and
    only the shards that hold them are opened.
    """
    from safetensors.torch import load_file

    path = resolve_checkpoint(weights_path)
    index_path = path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    limit = num_layers if num_layers is not None else (int(os.getenv("LLAMA_LOAD_NLAYERS", "0")) or None)

    def keep_key(key: str) -> bool:
        if not key.startswith("model.layers."):
            return True
        return limit is None or int(key.split(".")[2]) < limit

    shards = sorted({shard for key, shard in weight_map.items() if keep_key(key)})
    logger.info(f"loading {len(shards)} of {len(set(weight_map.values()))} safetensors shards from {path}")

    state_dict = {}
    for shard in shards:
        for k, v in load_file(str(path / shard)).items():
            if not keep_key(k):
                continue
            state_dict[k] = v.to(dtype) if v.dtype != dtype else v

    validate_state_dict(state_dict, num_layers=limit)
    return state_dict


def validate_state_dict(state_dict: dict, *, num_layers: Optional[int] = None) -> None:
    """Fail loudly on a missing or unexpected key instead of discovering it as a PCC drop."""
    layers = set()
    non_layer = set()
    for key in state_dict:
        if key.startswith("model.layers."):
            layers.add(int(key.split(".")[2]))
        else:
            non_layer.add(key)
    assert non_layer == _EXPECTED_NON_LAYER, (
        f"unexpected non-layer keys: extra={sorted(non_layer - _EXPECTED_NON_LAYER)}, "
        f"missing={sorted(_EXPECTED_NON_LAYER - non_layer)}"
    )
    if num_layers is not None:
        assert layers == set(range(num_layers)), f"expected layers 0..{num_layers - 1}, got {sorted(layers)}"
    for i in sorted(layers):
        prefix = f"model.layers.{i}."
        got = {k[len(prefix) :] for k in state_dict if k.startswith(prefix)}
        assert got == _EXPECTED_PER_LAYER, (
            f"layer {i}: extra={sorted(got - _EXPECTED_PER_LAYER)}, missing={sorted(_EXPECTED_PER_LAYER - got)}"
        )


def random_state_dict(cfg, *, num_layers: Optional[int] = None, dtype=torch.bfloat16, seed: int = 0) -> dict:
    """A checkpoint-shaped state dict of random tensors, for the component tests ONLY.

    Every module test up to P1 runs on random weights that are identical on both sides (recipe §4);
    this builds them once so the TT modules and the torch reference cannot be handed different ones.
    """
    g = torch.Generator().manual_seed(seed)
    n = cfg.num_hidden_layers if num_layers is None else num_layers
    d, h, i = cfg.head_dim, cfg.hidden_size, cfg.intermediate_size

    def rnd(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(dtype)

    sd = {
        "model.embed_tokens.weight": rnd(cfg.vocab_size, h),
        "model.norm.weight": rnd(h),
        "lm_head.weight": rnd(cfg.vocab_size, h),
    }
    for L in range(n):
        p = f"model.layers.{L}."
        sd[p + "input_layernorm.weight"] = rnd(h)
        sd[p + "post_attention_layernorm.weight"] = rnd(h)
        sd[p + "self_attn.q_proj.weight"] = rnd(cfg.num_attention_heads * d, h)
        sd[p + "self_attn.k_proj.weight"] = rnd(cfg.num_key_value_heads * d, h)
        sd[p + "self_attn.v_proj.weight"] = rnd(cfg.num_key_value_heads * d, h)
        sd[p + "self_attn.o_proj.weight"] = rnd(h, cfg.num_attention_heads * d)
        sd[p + "mlp.gate_proj.weight"] = rnd(i, h)
        sd[p + "mlp.up_proj.weight"] = rnd(i, h)
        sd[p + "mlp.down_proj.weight"] = rnd(h, i)
    validate_state_dict(sd, num_layers=n)
    return sd
