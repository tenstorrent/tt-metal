# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

OPS = ("rms_norm_residual_decode",)
REQUIRES = ("rms_norm_residual_decode",)

_seen = {}


def reset():
    _seen.clear()


def _in_table(hidden_states, residual, weight, attn_norm_config):
    key = (
        tuple(hidden_states.padded_shape),
        hidden_states.is_sharded(),
        tuple(residual.padded_shape),
        residual.is_sharded(),
    )
    if key not in _seen:
        _seen[key] = ttnn.experimental.llama31_8b.rms_norm_residual_decode_supported(
            hidden_states,
            residual,
            weight,
            program_config=attn_norm_config["sharded_program_config"],
            memory_config=attn_norm_config["sharded_output_config"],
        )
    return _seen[key]


def covers(next_norm, activation_dtype, attn_norm_config, hidden_states=None, residual=None):
    return (
        next_norm is not None
        and (activation_dtype or ttnn.bfloat16) == ttnn.bfloat16
        and attn_norm_config is not None
        and attn_norm_config.get("sharded_program_config") is not None
        and attn_norm_config.get("sharded_output_config") is not None
        and hidden_states is not None
        and residual is not None
        and _in_table(hidden_states, residual, next_norm.weight, attn_norm_config)
    )


def add_next_norm(hidden_states, residual, next_norm, attn_norm_config):
    normed = ttnn.experimental.llama31_8b.rms_norm_residual_decode(
        hidden_states,
        residual,
        next_norm.weight,
        next_norm.eps,
        program_config=attn_norm_config["sharded_program_config"],
        memory_config=attn_norm_config["sharded_output_config"],
        compute_kernel_config=next_norm.compute_kernel_config_hifi2,
    )
    if attn_norm_config.get("output_mem_config") is not None:
        normed = ttnn.to_memory_config(normed, attn_norm_config["output_mem_config"])
    return normed
