# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture of one Qwen3-32B TP=4 rank-local dense graph."""

import torch
import ttnn


_QKV_WEIGHT = None
_PACKED_GATE_UP_WEIGHT = None
_O_WEIGHT = None
_DOWN_WEIGHT = None


def _weight(shape, device):
    return ttnn.from_torch(
        torch.zeros(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def make_inputs(device):
    global _QKV_WEIGHT, _PACKED_GATE_UP_WEIGHT, _O_WEIGHT, _DOWN_WEIGHT
    _QKV_WEIGHT = _weight((5120, 2560), device)
    _PACKED_GATE_UP_WEIGHT = _weight((5120, 12800), device)
    _O_WEIGHT = _weight((2048, 5120), device)
    _DOWN_WEIGHT = _weight((6400, 5120), device)

    def activation(shape):
        return ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return (
        activation((1, 1, 32, 5120)),
        activation((1, 1, 32, 2048)),
        activation((1, 1, 32, 6400)),
    )


def decode_qkv(full_hidden, local_attention, local_intermediate):
    del local_attention, local_intermediate
    return ttnn.matmul(full_hidden, _QKV_WEIGHT, dtype=ttnn.bfloat16)


def decode_packed_gate_up(full_hidden, local_attention, local_intermediate):
    del local_attention, local_intermediate
    return ttnn.matmul(full_hidden, _PACKED_GATE_UP_WEIGHT, dtype=ttnn.bfloat16)


def decode_o(full_hidden, local_attention, local_intermediate):
    del full_hidden, local_intermediate
    return ttnn.matmul(local_attention, _O_WEIGHT, dtype=ttnn.bfloat16)


def decode_down(full_hidden, local_attention, local_intermediate):
    del full_hidden, local_attention
    return ttnn.matmul(local_intermediate, _DOWN_WEIGHT, dtype=ttnn.bfloat16)
