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
the byte-compare gate passes and the output is fluent garbage. ``convert_hf_qkv_to_meta_format``
is the shared helper that does this, and is what gpt_oss_d_p uses for the same reason.

Note the split of responsibilities: the **config** comes from the repo-bundled ``config.json`` named
by the adapter's ``hf_model_default`` (no mount, no network), while the **weights** come from the
checkpoint named here. Keeping them separate is what lets ``load_hf_config`` run in the H2D producer
and in device-free tests.

Scaffold status: the weight-loading body lands with #4149 (runner integration); the path resolution
and the dim cross-check are live now so later ops can import this module.
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


class ModelArgs:
    """Llama-3.1-8B ModelArgs.

    Scaffold: carries the resolved weights path. Weight loading (``load_state_dict``) lands with
    #4149.
    """

    def __init__(self, mesh_device=None, max_seq_len: int = 2048):
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len
        self.weights_path = Path(resolve_weights_path())
        logger.info(f"Llama-3.1-8B ModelArgs: weights_path={self.weights_path}")

    @staticmethod
    def load_state_dict(weights_path, convert_to_meta_format: bool = True):
        """Load the bf16 safetensors and convert q/k to Meta format.

        Lands with #4149. See the module docstring for why ``convert_to_meta_format`` must stay on:
        the frame has to match what blaze decode writes, or migration silently permutes K.
        """
        raise NotImplementedError(
            "Llama-3.1-8B prefill weight loading lands with tt-blaze#4149 (runner integration). "
            "Use models.tt_transformers.tt.load_checkpoints.convert_hf_qkv_to_meta_format for the "
            "q/k frame conversion, as gpt_oss_d_p does."
        )
