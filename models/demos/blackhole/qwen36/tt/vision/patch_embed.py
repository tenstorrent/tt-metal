# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""On-device patch embed (Conv3d folded to a linear) and interpolated positional embedding.
Output is [1, 1, seq_len, dim_local] bf16 TILE DRAM, zero past n_patches.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def from_torch_host_tiled(t, device, mesh_mapper, dtype=ttnn.bfloat16):
    """Tilize on host, then DMA — avoids a device Tilize / TilizeWithValPadding."""
    host = ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mesh_mapper)
    return ttnn.to_device(host, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class VisionEmbed(LightweightModule):
    """patch_embed + interpolated pos_embed, entirely on device."""

    def __init__(self, mesh_device, args, reference_model, dtype=ttnn.bfloat16, weight_cache_path=None):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.tp = args.cluster_shape[1]
        # Replicated when TP cannot split dim into whole tiles; otherwise fractured on the hidden dim.
        self.replicated_acts = getattr(args, "vision_replicated_acts", False)
        self._hidden_mapper = (
            ttnn.ReplicateTensorToMesh(mesh_device)
            if self.replicated_acts
            else ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=args.cluster_shape)
        )

        cache = (
            (lambda name: None)
            if (args.dummy_weights or weight_cache_path is None)
            else (lambda name: weight_cache_path / f"visual.{name}.tp{self.tp}")
        )

        proj = reference_model.patch_embed.proj
        embed_dim = proj.weight.shape[0]
        # Flatten [embed_dim, in_ch, T, P, P] to [in_ch*T*P*P, embed_dim]; order matches the processor.
        w = proj.weight.reshape(embed_dim, -1).t().contiguous()
        self.patch_dim = w.shape[0]
        self.embed_dim = embed_dim
        self.proj_weight = ttnn.as_tensor(
            w.unsqueeze(0).unsqueeze(0),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=self._hidden_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache("patch_embed_proj_w"),
        )
        self.proj_bias = ttnn.as_tensor(
            proj.bias.reshape(1, 1, 1, embed_dim),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=self._hidden_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache("patch_embed_proj_b"),
        )

        # ROW_MAJOR: ttnn.embedding untilizes a TILE table on every call. Cache key `_rm`.
        self.pos_table = ttnn.as_tensor(
            reference_model.pos_embed.weight.contiguous(),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=self._hidden_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache("pos_embed_w_rm"),
        )
        self.num_positions = reference_model.pos_embed.weight.shape[0]

        self._row_mask_cache = {}

    def _row_mask(self, rows, valid):
        """Zero padded rows; a biased matmul would otherwise leave the bias there."""
        key = (rows, valid)
        cached = self._row_mask_cache.get(key)
        if cached is not None:
            return cached
        idx = ttnn.arange(0, rows, 1, dtype=ttnn.float32, device=self.mesh_device)
        idx = ttnn.reshape(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), (1, 1, rows, 1))
        mask = ttnn.lt(idx, float(valid))
        ttnn.deallocate(idx)
        mask = ttnn.typecast(mask, ttnn.bfloat16)
        self._row_mask_cache[key] = mask
        return mask

    def forward(self, pixel_values, bilinear_indices, bilinear_weights, seq_len):
        n = pixel_values.shape[0]
        assert n <= seq_len, f"{n} patches exceed the padded seq_len {seq_len}"
        assert (
            pixel_values.shape[1] == self.patch_dim
        ), f"expected patch dim {self.patch_dim}, got {pixel_values.shape[1]}"
        assert bilinear_indices.shape[1] == n, "one bilinear index column per patch"

        # Tile-align the upload; pad the rest to seq_len on device.
        rows = ((n + 31) // 32) * 32

        x = pixel_values.to(torch.bfloat16)
        if rows != n:
            x = torch.nn.functional.pad(x, (0, 0, 0, rows - n))
        replicate = ttnn.ReplicateTensorToMesh(self.mesh_device)
        x_tt = ttnn.from_torch(
            x.reshape(1, 1, rows, self.patch_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        plan = self.args.vision_mm_plan(
            "patch_embed",
            rows=rows,
            k=self.patch_dim,
            n=self.embed_dim if self.replicated_acts else self.embed_dim // self.tp,
            in0_dtype=x_tt.dtype,
            in1_dtype=self.proj_weight.dtype,
            out_dtype=ttnn.bfloat16,
        )
        if plan.chunk != rows:
            x_tt = ttnn.reshape(x_tt, [1, rows // plan.chunk, plan.chunk, self.patch_dim])
        h = ttnn.linear(
            x_tt,
            self.proj_weight,
            bias=self.proj_bias,
            compute_kernel_config=plan.compute_kernel_config,
            memory_config=plan.memory_config,
            program_config=plan.program_config,
        )
        ttnn.deallocate(x_tt)
        if plan.chunk != rows:
            h = ttnn.reshape(h, [1, 1, rows, -1])

        idx = bilinear_indices.to(torch.int32)
        wts = bilinear_weights.to(torch.bfloat16)
        if rows != n:
            # Pad rows look up entry 0 with weight 0, contributing exactly nothing.
            idx = torch.nn.functional.pad(idx, (0, rows - n))
            wts = torch.nn.functional.pad(wts, (0, rows - n))
        idx_tt = ttnn.from_torch(
            idx,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=replicate,
        )
        wts_tt = from_torch_host_tiled(wts.reshape(4, rows, 1), self.mesh_device, replicate)
        pos = ttnn.embedding(idx_tt, self.pos_table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.deallocate(idx_tt)
        pos = ttnn.multiply(pos, wts_tt)
        ttnn.deallocate(wts_tt)
        pos_sum = ttnn.sum(pos, dim=0, keepdim=True)
        ttnn.deallocate(pos)
        pos_sum = ttnn.reshape(pos_sum, (1, 1, rows, pos_sum.shape[-1]))

        out = ttnn.add(h, pos_sum)
        ttnn.deallocate(h)
        ttnn.deallocate(pos_sum)

        # Rows past the real patch count must be exactly zero (the bias would otherwise leak in).
        if rows != n:
            out = ttnn.multiply(out, self._row_mask(rows, n))

        if seq_len != rows:
            out = ttnn.pad(out, [(0, 0), (0, 0), (0, seq_len - rows), (0, 0)], value=0.0)
        return ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
