# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared model creation and weight-loading helpers for Qwen3 scripts."""

import time

import ttml

from model_qwen3 import (
    Qwen3ForCausalLM,
    create_qwen3_config_from_hf,
    load_weights_from_hf,
)
from model_qwen3_distributed import (
    DistributedQwen3ForCausalLM,
    load_weights_from_hf_distributed,
)
from utils.sharded_loss import sharded_cross_entropy_loss
from utils.context_managers import empty_init


def build_mode_str(dp_size: int, tp_size: int) -> str:
    parts = []
    if dp_size > 1:
        parts.append(f"DP={dp_size}")
    if tp_size > 1:
        parts.append(f"TP={tp_size}")
    return "distributed " + ", ".join(parts) if parts else "single device"


def create_ttml_model(
    hf_config,
    max_seq_len,
    *,
    dp_size=1,
    tp_size=1,
    checkpoint=False,
    scatter_intermediates=False,
    track_memory=False,
    sharded_loss=False,
):
    """Instantiate a ttml Qwen3 model (single-device or TP).

    Returns ``(model, config, tie, shard_dim, mode_str)``.
    Weight loading is left to the caller (see :func:`load_hf_weights`).
    """
    use_tp = tp_size > 1
    config = create_qwen3_config_from_hf(hf_config, max_seq_len)
    tie = getattr(hf_config, "tie_word_embeddings", False)
    mode_str = build_mode_str(dp_size, tp_size)

    print(f"\nCreating ttml model ({mode_str}, max_seq_len={max_seq_len})...")
    t0 = time.time()

    if sharded_loss and not use_tp:
        print("  WARNING: --sharded_loss requires TP mode, ignoring.")
        sharded_loss = False

    with empty_init():
        if use_tp:
            shard_dim = 1
            model = DistributedQwen3ForCausalLM(
                config,
                tie_word_embeddings=tie,
                shard_dim=shard_dim,
                use_checkpoint=checkpoint,
                scatter_intermediates=scatter_intermediates,
                track_memory=track_memory,
                sharded_loss=sharded_loss,
            )
            if checkpoint:
                ckpt_msg = "ENABLED"
                if scatter_intermediates:
                    ckpt_msg += " (scatter_intermediates=True)"
                print(f"  Gradient checkpointing: {ckpt_msg}")
            if sharded_loss:
                print("  Sharded loss: ENABLED (LM head gather_output=False)")
        else:
            shard_dim = None
            model = Qwen3ForCausalLM(
                config,
                tie_word_embeddings=tie,
                track_memory=track_memory,
                use_checkpoint=checkpoint,
            )
            if checkpoint:
                print("  Gradient checkpointing: ENABLED")
            if scatter_intermediates:
                print(
                    "  WARNING: --scatter_intermediates requires "
                    "--checkpoint and TP mode, ignoring."
                )

    print(f"  Model created in {time.time() - t0:.2f}s")
    return model, config, tie, shard_dim, mode_str


def load_hf_weights(
    model, hf_state_dict, config, *, tie=False, tp=False, shard_dim=None
):
    """Load a HuggingFace state-dict into a ttml model, printing timing."""
    print("\nLoading HF weights into ttml model...")
    t0 = time.time()
    if tp:
        load_weights_from_hf_distributed(
            model,
            hf_state_dict,
            config,
            tie_word_embeddings=tie,
            shard_dim=shard_dim,
        )
    else:
        load_weights_from_hf(model, hf_state_dict, config, tie_word_embeddings=tie)
    print(f"  Weight loading took {time.time() - t0:.2f}s")
