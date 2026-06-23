# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The Qwen3.5 model implementation (ttnn, Blackhole P150).

Read in this order:
  * model.py        — Qwen35Model: the top-level model. Assembles embedding -> N x
                      Qwen35DecoderLayer -> final RMSNorm -> LM head, and owns the
                      prefill / decode / generate drivers. Start here.
  * layer.py        — Qwen35DecoderLayer: one hybrid decoder layer. Each layer is either
                      a full (softmax) attention layer or a Gated DeltaNet layer, chosen
                      per index by args.is_full_attention_layer(layer_num); both share the
                      same RMSNorm + residual + SwiGLU-MLP wiring.
  * model_config.py — Qwen35ModelArgs: every config value, derived from the HF checkpoint
                      named by the HF_MODEL env var. Subclasses tt_transformers.ModelArgs.

Token mixers and MLP each get their own subpackage (each with its own __init__ + unit test):
  * attention/ — Qwen35Attention            (full softmax attention)
  * gdn/       — Qwen35GatedDeltaNet         (Gated DeltaNet / linear attention)
  * mlp/       — Qwen35MLP                   (SwiGLU MLP)

Supporting modules:
  * rms_norm.py       — Qwen35RMSNorm.
  * tp_common.py      — tensor-parallel sharding/replication helpers (shard_w, replicate,
                        prepare_*_qkv, prepare_conv_taps).
  * weight_mapping.py — remap HF checkpoint keys to the internal scheme (remap_qwen35_state_dict,
                        submodule_state_dict).

The classes below are re-exported so callers can ``from ...tt import Qwen35Model``.
"""
from .layer import Qwen35DecoderLayer
from .model import Qwen35Model
from .model_config import Qwen35ModelArgs

__all__ = ["Qwen35Model", "Qwen35DecoderLayer", "Qwen35ModelArgs"]
