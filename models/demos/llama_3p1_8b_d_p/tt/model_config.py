# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B ModelArgs (weights). Mirrors ``gpt_oss_d_p/tt/model_config.py``.

Loads the bf16 safetensors and converts the q/k projections to Meta format for the on-device
(indexed) RoPE.

**The q/k conversion is not optional.** HF ships Llama's q_proj/k_proj permuted so that its
``rotate_half`` (half-split) RoPE reproduces Meta's interleaved rotation. blaze decode writes K in
the **Meta-interleaved** frame (``blaze/ops/rope/kernels/op.hpp``, and ``make_cos_sin`` builds the
table as ``stack((cos,cos),-1).flatten(-2)`` -- each frequency duplicated *adjacently*). If prefill
writes the HF frame instead, KV migration copies bytes faithfully and decode reads a permutation:
the byte-compare gate passes and the output is fluent garbage.

The conversion happens in exactly ONE place: ``tt/attention.py``'s ``hf_to_meta_head_frame``, at
weight-load time. This module's loader deliberately returns raw HF-frame tensors. Converting here
as well would apply the permutation twice, which is not a no-op and not the HF frame either — it is
a third frame that no RoPE convention matches, and the symptom would be a q/k-only PCC failure with
v and o intact. ``test_attention_vs_ref`` pins the single conversion.

Note the split of responsibilities: the **config** comes from the repo-bundled ``config.json`` named
by the adapter's ``hf_model_default`` (no mount, no network), while the **weights** come from the
checkpoint named here. Keeping them separate is what lets ``load_hf_config`` run in the H2D producer
and in device-free tests.
"""

import os
from pathlib import Path

from loguru import logger

from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig

# Full checkpoint: config.json + the 4 safetensors shards + tokenizer. Overridden by
# PREFILL_HF_MODEL (the engine's knob) or LLAMA31_8B_HF_MODEL.
DEFAULT_WEIGHTS_PATH = "/mnt/models/meta-llama/Llama-3.1-8B-Instruct"


def resolve_weights_path() -> str:
    """Where the safetensors live. Env wins, in the runner's precedence order."""
    return os.environ.get("PREFILL_HF_MODEL") or os.environ.get("LLAMA31_8B_HF_MODEL") or DEFAULT_WEIGHTS_PATH


def cross_check_hf_config(hf_config) -> None:
    """Assert the loaded HF config agrees with our dim SSOT.

    Cheap, and it catches the failure mode where someone points PREFILL_HF_MODEL at a different
    Llama (3.2-3B, 3.1-70B) and every later PCC is mysteriously wrong at the wrong shape.
    """
    expected = {
        "hidden_size": Llama31_8BConfig.EMB_SIZE,
        "intermediate_size": Llama31_8BConfig.INTERMEDIATE_SIZE,
        "num_hidden_layers": Llama31_8BConfig.NUM_LAYERS,
        "num_attention_heads": Llama31_8BConfig.NUM_ATTENTION_HEADS,
        "num_key_value_heads": Llama31_8BConfig.NUM_KEY_VALUE_HEADS,
        "vocab_size": Llama31_8BConfig.VOCAB_SIZE,
    }
    mismatches = {
        key: (getattr(hf_config, key, None), want)
        for key, want in expected.items()
        if getattr(hf_config, key, None) != want
    }
    # head_dim is derived on some Llama configs rather than stored.
    head_dim = getattr(hf_config, "head_dim", None) or (
        getattr(hf_config, "hidden_size", 0) // max(getattr(hf_config, "num_attention_heads", 1), 1)
    )
    if head_dim != Llama31_8BConfig.HEAD_DIM:
        mismatches["head_dim"] = (head_dim, Llama31_8BConfig.HEAD_DIM)
    if mismatches:
        raise ValueError(
            "loaded HF config does not match Llama31_8BConfig: "
            + ", ".join(f"{k}: got {got!r}, want {want!r}" for k, (got, want) in sorted(mismatches.items()))
        )


def _wanted_keys(num_layers: int, first_layer_idx: int) -> set:
    """HF checkpoint keys this rank needs: its own layers, plus the embedding and the final norm.

    A rank loads only its slice so the host copy scales with the slice and not with the whole 16 GB
    checkpoint. ``embed_tokens`` and ``model.norm`` are pulled unconditionally rather than gated on
    the rank's role: the loader does not know whether this rank is first or last, and two unused
    4096-wide tensors are not worth the coupling.
    """
    keys = {"model.embed_tokens.weight", "model.norm.weight"}
    per_layer = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    for layer_idx in range(first_layer_idx, first_layer_idx + num_layers):
        for name in per_layer:
            keys.add(f"model.layers.{layer_idx}.{name}.weight")
        keys.add(f"model.layers.{layer_idx}.input_layernorm.weight")
        keys.add(f"model.layers.{layer_idx}.post_attention_layernorm.weight")
    return keys


def load_llama_state_dict(
    weights_path=None,
    *,
    num_layers: int = Llama31_8BConfig.NUM_LAYERS,
    first_layer_idx: int = 0,
) -> dict:
    """Read the sharded bf16 safetensors into a state dict, keeping only this rank's layers.

    Returns tensors under their **HuggingFace names in the HuggingFace frame**, unconverted. The
    q/k Meta-frame conversion the module docstring describes happens exactly once, in
    ``tt/attention.py`` at weight-load time (``hf_to_meta_head_frame``) — doing it here as well
    would apply the permutation twice and land back in a third frame that is neither.

    Reads shard-by-shard with a key filter instead of ``load_file`` per shard, so a rank owning 4 of
    32 layers never materialises the other 28.
    """
    from safetensors import safe_open

    weights_path = Path(weights_path or resolve_weights_path())
    shards = sorted(weights_path.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(
            f"no .safetensors under {weights_path}; point PREFILL_HF_MODEL or LLAMA31_8B_HF_MODEL "
            f"at a Llama-3.1-8B checkpoint"
        )

    wanted = _wanted_keys(num_layers, first_layer_idx)
    state_dict = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if key in wanted:
                    state_dict[key] = f.get_tensor(key)

    missing = wanted - state_dict.keys()
    if missing:
        raise KeyError(
            f"checkpoint at {weights_path} is missing {len(missing)} expected tensor(s), " f"e.g. {sorted(missing)[:3]}"
        )
    logger.info(
        f"Loaded {len(state_dict)} tensors for layers "
        f"{first_layer_idx}..{first_layer_idx + num_layers - 1} from {len(shards)} shard(s)"
    )
    return state_dict


class ModelArgs:
    """Llama-3.1-8B ModelArgs: carries the resolved weights path."""

    def __init__(self, mesh_device=None, max_seq_len: int = 2048):
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len
        self.weights_path = Path(resolve_weights_path())
        logger.info(f"Llama-3.1-8B ModelArgs: weights_path={self.weights_path}")

    @staticmethod
    def load_state_dict(weights_path, **kwargs):
        """Thin alias for ``load_llama_state_dict``, for parity with the other models' ModelArgs."""
        return load_llama_state_dict(weights_path, **kwargs)
