# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstarl2_small.devstral_utils.dram_sharded_matmul import (
    TILE,
    width_sharded_l1_linear_keep_sharded,
)
from models.experimental.devstarl2_small.devstral_utils.pixtral_seq_chunk import vision_slice_memcfg
import ttnn
import torch


def _patch_merge_ws_m_cap() -> int:
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
            cache_name = None
            if weight_cache_path is not None:
                cache_name = weight_cache_path / f"{state_dict_prefix}merging_layer.weight.{inner_h}_{inner_w}.tile"
            return ttnn.as_tensor(
                tensor_data,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name,
            )

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

    def _ensure_tile(self, tensor: ttnn.Tensor, mem_cfg: ttnn.MemoryConfig) -> ttnn.Tensor:
        if tensor.get_layout() != ttnn.TILE_LAYOUT:
            return ttnn.to_layout(tensor, ttnn.TILE_LAYOUT, memory_config=mem_cfg)
        if mem_cfg.buffer_type == ttnn.BufferType.L1 and tensor.memory_config().buffer_type != ttnn.BufferType.L1:
            return ttnn.to_memory_config(tensor, mem_cfg)
        return tensor

    def _merge_linear_ws(
        self,
        patch: ttnn.Tensor,
        weight_index: int,
        m_rows: int,
        feature_dim: int,
    ) -> ttnn.Tensor:
        """Width-sharded merge linear for one M-chunk (m_rows <= ws cap, tile-padded if needed)."""
        mm_seq = m_rows
        patch_in = patch
        if m_rows % TILE != 0:
            padded = ((m_rows + TILE - 1) // TILE) * TILE
            patch_in = ttnn.pad(
                patch,
                padding=[(0, 0), (0, 0), (0, padded - m_rows), (0, 0)],
                value=0.0,
            )
            mm_seq = padded

        out = width_sharded_l1_linear_keep_sharded(
            self.args,
            patch_in,
            self.merging_weights[weight_index],
            m_seq=mm_seq,
            k_dim=feature_dim,
            n_dim=feature_dim,
            fuse_batch=True,
            compute_kernel_config=self._merge_compute_cfg,
        )
        if mm_seq != m_rows:
            out = ttnn.slice(
                out,
                (0, 0, 0, 0),
                (1, 1, m_rows, out.shape[-1]),
                memory_config=out.memory_config(),
            )
        return out

    def _merge_linear(self, patch: ttnn.Tensor, weight_index: int, m_rows: int, feature_dim: int) -> ttnn.Tensor:
        ws_cap = _patch_merge_ws_m_cap()
        if m_rows <= ws_cap:
            return self._merge_linear_ws(patch, weight_index, m_rows, feature_dim)

        # Large merged grids (e.g. 1540px → 3025 rows): chunk WS matmuls; single DRAM M=3025 hurts PCC.
        parts = []
        feat = int(patch.shape[-1])
        for start in range(0, m_rows, ws_cap):
            end = min(start + ws_cap, m_rows)
            sl = ttnn.slice(
                patch,
                (0, 0, start, 0),
                (1, 1, end, feat),
                memory_config=patch.memory_config(),
            )
            chunk_out = self._merge_linear_ws(sl, weight_index, end - start, feature_dim)
            ttnn.deallocate(sl)
            if chunk_out.is_sharded():
                chunk_il = ttnn.sharded_to_interleaved(chunk_out, ttnn.DRAM_MEMORY_CONFIG)
                chunk_out.deallocate(True)
            else:
                chunk_il = chunk_out
            parts.append(chunk_il)

        out = parts[0]
        for part in parts[1:]:
            prev = out
            out = ttnn.concat([prev, part], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(prev)
            ttnn.deallocate(part)
        return out

    def forward(self, image_features: ttnn.Tensor, image_sizes) -> ttnn.Tensor:
        image_sizes = [
            (image_size[0] // self.patch_size, image_size[1] // self.patch_size) for image_size in image_sizes
        ]

        tokens_per_image = [h * w for h, w in image_sizes]
        d = image_features.shape[-1]

        permuted_tensor = []
        for image_index, image_tokens in enumerate(ttnn.split(image_features, tokens_per_image, dim=0)):
            h, w = image_sizes[image_index]
            merged_h = h // self.spatial_merge_size
            merged_w = w // self.spatial_merge_size

            slice_mem_cfg = vision_slice_memcfg(h * w)
            image_tokens = self._ensure_tile(image_tokens, slice_mem_cfg)

            grid = ttnn.reshape(
                image_tokens,
                (merged_h, self.spatial_merge_size, merged_w, self.spatial_merge_size, d),
            )
            if (
                slice_mem_cfg.buffer_type == ttnn.BufferType.L1
                and grid.memory_config().buffer_type != ttnn.BufferType.L1
            ):
                grid = ttnn.to_memory_config(grid, slice_mem_cfg)
            m_rows = merged_h * merged_w
            merged = None
            merge_mem_cfg = None
            weight_index = 0
            for inner_h in range(self.spatial_merge_size):
                for inner_w in range(self.spatial_merge_size):
                    patch = ttnn.slice(
                        grid,
                        (0, inner_h, 0, inner_w, 0),
                        (merged_h, inner_h + 1, merged_w, inner_w + 1, d),
                        memory_config=slice_mem_cfg,
                    )
                    patch = ttnn.reshape(patch, (1, 1, m_rows, d))
                    projected = self._merge_linear(patch, weight_index, m_rows, d)
                    ttnn.deallocate(patch)
                    if merge_mem_cfg is None:
                        merge_mem_cfg = projected.memory_config()
                    if merged is None:
                        merged = projected
                    else:
                        prev_merged = merged
                        merged = ttnn.add(merged, projected, memory_config=merge_mem_cfg)
                        ttnn.deallocate(prev_merged)
                        ttnn.deallocate(projected)
                    weight_index += 1

            ttnn.deallocate(grid)
            if merged.is_sharded():
                merged_out = ttnn.sharded_to_interleaved(merged, ttnn.DRAM_MEMORY_CONFIG)
                merged.deallocate(True)
            else:
                merged_out = merged
            permuted_tensor.append(ttnn.reshape(merged_out, (m_rows, d)))

        image_features = ttnn.concat(permuted_tensor, dim=0)

        return image_features


__all__ = ["TTMistral3PatchMerger"]
