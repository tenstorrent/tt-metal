# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared shape constants for the Mistral-Medium-3.5 block tests.

Production dims from ``configs/Mistral-Medium-3.5-128B/config.json``. The three HOST-ONLY tests
(rope / checkpoint / reference-model) import no ttnn and run with ``--noconftest`` on a dev box with
no TT runtime built; everything else needs a device.
"""

# Model dims.
HIDDEN = 12288
N_Q = 96
N_KV = 8
HEAD_DIM = 128
FFN = 28672
EPS = 1e-5
N_LAYERS = 88
VOCAB = 131072

# Hardware target: (8,4) Blackhole Galaxy, TP=4 (cols) x SP=8 (rows). See ../../config.py.
TARGET_MESH = (8, 4)
TARGET_TP = 4


def per_chip(tp: int) -> dict:
    """Per-chip shapes at a given TP. All tile-aligned at the TP=4 target."""
    return {
        "hidden": HIDDEN // tp,  # 3072 at TP=4
        "ffn": FFN // tp,  # 7168
        "n_q": N_Q // tp,  # 24
        "n_kv": N_KV // tp,  # 2  <- the one value no other GQA model in the repo uses
        "qkv": (N_Q // tp + 2 * (N_KV // tp)) * HEAD_DIM,  # 3584
        "gate_up": 2 * (FFN // tp),  # 14336
    }


# YaRN kwargs for tt/rope_tables builders (verified against transformers at 0 ULP).
YARN = dict(
    rope_theta=1000000.0,
    yarn_factor=64.0,
    yarn_orig_max_pos=4096,
    yarn_beta_fast=4.0,
    yarn_beta_slow=1.0,
    yarn_mscale=1.0,
    yarn_mscale_all_dim=0.0,
    yarn_truncate=True,
)


class HFConfigStub:
    """Minimal duck-typed HF config for block tests (no ``transformers`` import needed)."""

    def __init__(self, **overrides):
        self.hidden_size = HIDDEN
        self.intermediate_size = FFN
        self.num_hidden_layers = N_LAYERS
        self.num_attention_heads = N_Q
        self.num_key_value_heads = N_KV
        self.head_dim = HEAD_DIM
        self.hidden_act = "silu"
        self.rms_norm_eps = EPS
        self.vocab_size = VOCAB
        self.sliding_window = None
        self.tie_word_embeddings = False
        self.rotary_dim = HEAD_DIM
        self.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 64.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 4.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "llama_4_scaling_beta": 0,
        }
        for k, v in overrides.items():
            setattr(self, k, v)
