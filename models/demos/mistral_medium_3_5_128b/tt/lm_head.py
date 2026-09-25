# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LM head: full-width normed hidden -> logits, vocab column-parallel over TP (131072 / 4 = 32768 per
chip, already tile- and power-of-two aligned, so no padding). From minimax_m3 ``tt/model.py`` (bf8
weight, column-parallel vocab); the spec's default weight dtype applies, compute is HiFi4 + fp32 acc."""

import torch

import ttnn

from .common import cache_name, compute_config, dtype_tag


class LMHead:
    def __init__(self, mesh_device, mesh_config, weight, *, dtype=ttnn.bfloat8_b, tensor_cache_path=None):
        """``weight``: HF ``[vocab, hidden]`` (None when loading from ``tensor_cache_path``)."""
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        torch_weight = None if weight is None else weight.to(torch.bfloat16).t()[None, None]  # [1, 1, hidden, vocab]
        self.weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
            cache_file_name=cache_name(tensor_cache_path, f"weight_tp{mesh_config.tp}_{dtype_tag(dtype)}"),
        )
        self.compute_kernel_config = compute_config(mesh_device)

    def __call__(self, x):
        """``x`` full-width ``[1, 1, s, hidden]`` -> per chip ``[1, 1, s, vocab/tp]`` bf16 logits."""
        return ttnn.linear(
            x,
            self.weight,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
