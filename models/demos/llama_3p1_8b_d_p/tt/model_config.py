# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint path/config helpers retained for the prefill adapter scaffold.

The implemented model uses ``tt.weights.CheckpointWeights`` for incremental raw-HF loading.
That loader does not permute Q/K. ``QKVProjection`` performs the single required HF-to-Meta
conversion when it constructs device weights, so the resulting K frame matches Blaze decode.

The adapter config remains import-light and independent of checkpoint availability. Do not use
this module's unsupported bulk-loader scaffold for the full model; use CheckpointWeights.layer
and release each host layer mapping after construction.
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

    Carries the resolved checkpoint path for the adapter scaffold. The implemented model uses
    CheckpointWeights directly for incremental loading.
    """

    def __init__(self, mesh_device=None, max_seq_len: int = 2048):
        self.mesh_device = mesh_device
        self.max_seq_len = max_seq_len
        self.weights_path = Path(resolve_weights_path())
        logger.info(f"Llama-3.1-8B ModelArgs: weights_path={self.weights_path}")

    @staticmethod
    def load_state_dict(weights_path, convert_to_meta_format: bool = True):
        """Unsupported legacy bulk-loader signature; use incremental raw CheckpointWeights.

        The signature is retained for compatibility with the scaffold only. Its historical
        convert_to_meta_format default must not be applied before QKVProjection.
        """
        raise NotImplementedError(
            "Use models.demos.llama_3p1_8b_d_p.tt.weights.CheckpointWeights for incremental raw-HF "
            "loading. Do not pre-convert Q/K: QKVProjection performs that conversion exactly once."
        )
