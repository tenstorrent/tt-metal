# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstral2_small.devstral_utils.fp8_dequantize_compat import apply_fp8_dequantize_compat
import ttnn
import torch

apply_fp8_dequantize_compat()

TILE = ttnn.TILE_SIZE
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def _patch_merge_m_cap() -> int:
    raw = os.environ.get("PIXTRAL_PATCH_MERGE_WS_M_CAP", "512")
    return max(TILE, int(raw))


class TTMistral3PatchMerger(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path=None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = mesh_device
        self.spatial_merge_size = 2
        self.patch_size = args.vision_patch_size
        self.args = args
        self._merge_compute_cfg = args.compute_kernel_config_hifi2

        def get_weight(name):
            return torch.transpose(state_dict[f"{state_dict_prefix}{name}.weight"], -2, -1)

        def as_tensor_data(tensor_data, dtype, inner_h, inner_w):
            # Tilize on host then upload; device= in from_torch/as_tensor runs Tilize on device trace.
            host_tt = ttnn.from_torch(
                tensor_data,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                pad_value=0.0,
            )
            return ttnn.to_device(host_tt, mesh_device, memory_config=_DRAM)

        merging_weights = get_weight("merging_layer")
        input_dim, output_dim = merging_weights.shape
        hidden_dim = input_dim // self.spatial_merge_size**2
        merging_weights = merging_weights.reshape(
            hidden_dim,
            self.spatial_merge_size,
            self.spatial_merge_size,
            output_dim,
        )
        self.merging_weights = [
            as_tensor_data(merging_weights[:, inner_h, inner_w, :].contiguous(), dtype, inner_h, inner_w)
            for inner_h in range(self.spatial_merge_size)
            for inner_w in range(self.spatial_merge_size)
        ]

    def _pad_m_to_tile(
        self,
        patch: ttnn.Tensor,
        m_rows: int,
    ) -> tuple[ttnn.Tensor, int]:
        padded = ((m_rows + TILE - 1) // TILE) * TILE
        if padded == m_rows:
            return patch, m_rows
        pad_rows = padded - m_rows
        zeros = ttnn.zeros(
            (pad_rows, patch.shape[-1]),
            dtype=patch.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_DRAM,
        )
        return ttnn.concat([patch, zeros], dim=0, memory_config=_DRAM), padded

    def _merge_linear(
        self,
        patch: ttnn.Tensor,
        weight_index: int,
        m_rows: int,
        feature_dim: int,
    ) -> ttnn.Tensor:
        m_cap = _patch_merge_m_cap()
        if m_rows <= m_cap:
            return self._merge_linear_chunk(patch, weight_index, m_rows, feature_dim)

        parts = []
        for start in range(0, m_rows, m_cap):
            end = min(start + m_cap, m_rows)
            sl = ttnn.slice(
                patch,
                (start, 0),
                (end, feature_dim),
                memory_config=_DRAM,
            )
            parts.append(self._merge_linear_chunk(sl, weight_index, end - start, feature_dim))
            ttnn.deallocate(sl)

        out = parts[0]
        for part in parts[1:]:
            prev = out
            out = ttnn.concat([prev, part], dim=0, memory_config=_DRAM)
            ttnn.deallocate(prev)
            ttnn.deallocate(part)
        return out

    def _merge_linear_chunk(
        self,
        patch: ttnn.Tensor,
        weight_index: int,
        m_rows: int,
        feature_dim: int,
    ) -> ttnn.Tensor:
        patch_in, mm_seq = self._pad_m_to_tile(patch, m_rows)
        out = ttnn.linear(
            patch_in,
            self.merging_weights[weight_index],
            dtype=ttnn.bfloat16,
            memory_config=_DRAM,
            compute_kernel_config=self._merge_compute_cfg,
        )
        if patch_in is not patch:
            ttnn.deallocate(patch_in)
        if mm_seq != m_rows:
            out = ttnn.slice(
                out,
                (0, 0),
                (m_rows, feature_dim),
                memory_config=_DRAM,
            )
        return out

    def forward(self, image_features: ttnn.Tensor, image_sizes) -> ttnn.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        outputs = []
        for image_index, image_tokens in enumerate(ttnn.split(image_features, tokens_per_image, dim=0)):
            h, w = image_sizes[image_index]
            merged_h = h // self.spatial_merge_size
            merged_w = w // self.spatial_merge_size
            m_rows = merged_h * merged_w

            if image_tokens.memory_config().buffer_type != ttnn.BufferType.DRAM:
                image_tokens = ttnn.to_memory_config(image_tokens, _DRAM)

            grid = ttnn.reshape(image_tokens, (1, h, w, d), memory_config=_DRAM)

            merged = None
            weight_index = 0
            for inner_h in range(self.spatial_merge_size):
                for inner_w in range(self.spatial_merge_size):
                    patch = ttnn.slice(
                        grid,
                        (0, inner_h, inner_w, 0),
                        (1, h, w, d),
                        (1, self.spatial_merge_size, self.spatial_merge_size, 1),
                        memory_config=_DRAM,
                    )
                    patch = ttnn.reshape(patch, (m_rows, d), memory_config=_DRAM)
                    projected = self._merge_linear(patch, weight_index, m_rows, d)
                    ttnn.deallocate(patch)
                    if merged is None:
                        merged = projected
                    else:
                        prev_merged = merged
                        merged = ttnn.add(merged, projected, memory_config=_DRAM)
                        ttnn.deallocate(prev_merged)
                        ttnn.deallocate(projected)
                    weight_index += 1

            ttnn.deallocate(grid)
            outputs.append(merged)

        if len(outputs) == 1:
            return outputs[0]
        return ttnn.concat(outputs, dim=0)


__all__ = ["TTMistral3PatchMerger"]
