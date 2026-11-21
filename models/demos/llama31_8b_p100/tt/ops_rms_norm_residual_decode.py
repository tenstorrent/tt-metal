# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

OPS = ("rms_norm_residual_decode",)


_seen = {}


def reset():
    _seen.clear()


def _same_sharding(x, residual, out_mc):
    if not (x.is_sharded() and residual.is_sharded()):
        return False
    r = residual.memory_config()

    def same(m):
        return (
            m.memory_layout == r.memory_layout
            and m.shard_spec is not None
            and m.shard_spec.grid == r.shard_spec.grid
            and tuple(m.shard_spec.shape) == tuple(r.shard_spec.shape)
        )

    return same(x.memory_config()) and same(out_mc)


def _usable(x, residual, weight, norm_config):
    key = (
        tuple(x.padded_shape),
        x.is_sharded(),
        x.dtype,
        tuple(residual.padded_shape),
        residual.is_sharded(),
        residual.dtype,
    )
    if key not in _seen:
        _seen[key] = (
            key[4]
            and key[5] == ttnn.bfloat16
            and key[2] == ttnn.bfloat16
            and not _same_sharding(x, residual, norm_config["sharded_output_config"])
        ) and ttnn.experimental.llama31_8b.rms_norm_residual_decode_supported(
            x,
            residual,
            weight,
            program_config=norm_config["sharded_program_config"],
            memory_config=norm_config["sharded_output_config"],
        )
    return _seen[key]


def covers(block, attn_out, residual, ff_norm_config):
    return (
        block.pre_ff_norm is None
        and block.post_ff_norm is None
        and block.ff_norm.norm is not None
        and ff_norm_config is not None
        and ff_norm_config.get("sharded_program_config") is not None
        and ff_norm_config.get("sharded_output_config") is not None
        and _usable(attn_out, residual, block.ff_norm.norm.weight, ff_norm_config)
    )


def add_norm(x, residual, norm, norm_config):
    out = ttnn.experimental.llama31_8b.rms_norm_residual_decode(
        x,
        residual,
        norm.weight,
        norm.eps,
        program_config=norm_config["sharded_program_config"],
        memory_config=norm_config["sharded_output_config"],
        compute_kernel_config=norm.compute_kernel_config_hifi2,
    )
    if norm_config.get("output_mem_config") is not None:
        out = ttnn.to_memory_config(out, norm_config["output_mem_config"])
    return out
