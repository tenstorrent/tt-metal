# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mandatory shard-advisor target for one rewritten Qwen3-32B dense block."""

import os
import sys

import torch
import ttnn

MODEL_DIR = os.environ.get("SHARD_ADVISE_MODEL_DIR", "/home/mvasiljevic/tt-metal")
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "32"))
MAX_CACHE_LEN = int(os.environ.get("SHARD_ADVISE_SEQ", "128"))
CURRENT_POS = int(os.environ.get("SHARD_ADVISE_CURRENT_POS", "17"))

_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None


def _build(device):
    # Append so the advisor environment's real ttnn package remains authoritative.
    if MODEL_DIR not in sys.path:
        sys.path.append(MODEL_DIR)
    from models.autoports.qwen_qwen3_32b.tests.test_functional_decoder import (
        _config,
        _synthetic_state,
    )
    from models.autoports.qwen_qwen3_32b.tt.functional_decoder import REPRESENTATIVE_LAYER
    from models.autoports.qwen_qwen3_32b.tt.optimized_decoder import OptimizedDecoder

    class CaptureDecoder(OptimizedDecoder):
        """Tracer-only explicit movement boundaries; production code is unchanged."""

        def _move_owned(self, tensor, memory_config):
            return ttnn.to_memory_config(tensor, memory_config)

        def _decode_norm(self, residual, weight):
            norm_input = ttnn.to_memory_config(residual, self.residual_memory_config)
            return ttnn.rms_norm(
                norm_input,
                epsilon=self.rms_norm_eps,
                weight=weight,
                program_config=self.norm_program_config,
                compute_kernel_config=self.norm_compute_config,
                memory_config=self.residual_memory_config,
            )

    config = _config()
    state = _synthetic_state(config)
    decoder = CaptureDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=REPRESENTATIVE_LAYER,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=MAX_CACHE_LEN,
        precision_policy="mlp_bfp4_lofi",
        decode_target_cores=40,
    )
    hidden = ttnn.from_torch(
        torch.randn((1, BATCH, 1, config.hidden_size), dtype=torch.bfloat16),
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    key_cache, value_cache = decoder.allocate_kv_cache()
    return decoder, key_cache, value_cache, hidden


def decode(hidden):
    return _DECODER.decode_forward(
        hidden,
        _KEY_CACHE,
        _VALUE_CACHE,
        current_pos=CURRENT_POS,
    )


def make_inputs(device):
    global _DECODER, _KEY_CACHE, _VALUE_CACHE
    _DECODER, _KEY_CACHE, _VALUE_CACHE, hidden = _build(device)
    return (hidden,)
