# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""On-device vision patch embedding + interpolated positional embedding.

Replaces the two host-torch stages that used to run on the HF reference model once per image
(``reference_model.patch_embed`` and ``reference_model.fast_pos_embed_interpolate``):

    patch_input = patch_embed(pixel_values)            # nn.Conv3d on CPU
    pos_embeds  = fast_pos_embed_interpolate(grid_thw) # 4 CPU gathers of [n_patches, dim]
    x           = patch_input + pos_embeds             # then F.pad + upload

``patch_embed`` is an ``nn.Conv3d`` whose stride equals its kernel, applied to a tensor that the
processor has *already* patchified to ``[n_patches, in_ch*T*P*P]``. So it is exactly a linear:
flatten the conv weight to ``[in_ch*T*P*P, embed_dim]`` and matmul. No convolution op needed.

The positional embedding is a bilinear interpolation of a learned ``[num_positions, dim]`` table:
four corner lookups combined with per-token weights. The corner *indices and weights* are integer
/ fractional grid arithmetic over ``[4, n_patches]`` — vectorized, ~microseconds, and they stay on
host. The expensive half — four ``[n_patches, dim]`` gathers, the weighting and the sum — moves to
``ttnn.embedding`` + ``multiply`` + ``sum``.

Output contract: identical to ``VisionModelArgs.prepare_residual_tensor_prefill`` —
``[1, 1, seq_len, dim_local]`` bf16 TILE in DRAM, replicated when ``vision_replicated_acts`` and
otherwise fractured along the hidden dim across cluster axis 1. Rows past ``n_patches`` are exactly
zero, as the host ``F.pad`` produced them.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class VisionEmbed(LightweightModule):
    """patch_embed + interpolated pos_embed, entirely on device."""

    def __init__(self, mesh_device, args, reference_model, dtype=ttnn.bfloat16, weight_cache_path=None):
        """
        Args:
            mesh_device: the mesh the vision tower runs on.
            args (VisionModelArgs): supplies ``cluster_shape`` and ``vision_replicated_acts``.
            reference_model: the HF vision module, read once for ``patch_embed.proj`` and
                ``pos_embed`` weights. Not retained.
            dtype: weight dtype. bf16 by default — this projection is small (1536x1152) and
                feeds every downstream block, so it is not worth bfp8 error here.
            weight_cache_path: optional ttnn weight cache dir.
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.tp = args.cluster_shape[1]
        # Match the activation distribution the vision blocks expect (see
        # VisionModelArgs.prepare_residual_tensor_prefill): replicated when TP cannot split `dim`
        # into whole tiles, otherwise fractured along the hidden dim.
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

        # ---- patch projection: Conv3d(stride == kernel) folded to a [K, dim] linear ----
        proj = reference_model.patch_embed.proj
        embed_dim = proj.weight.shape[0]
        # [embed_dim, in_ch, T, P, P] -> [in_ch*T*P*P, embed_dim]. The flatten order matches the
        # `hidden_states.view(-1, in_ch, T, P, P)` the reference does before the conv, which is the
        # layout the image processor already emits, so no permutation is involved.
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

        # ---- learned positional embedding table, sharded the same way ----
        # ttnn.embedding needs the table 2D [num_positions, dim]; the hidden shard is its last dim,
        # so the same mapper applies.
        self.pos_table = ttnn.as_tensor(
            reference_model.pos_embed.weight.contiguous(),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            mesh_mapper=self._hidden_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache("pos_embed_w"),
        )
        self.num_positions = reference_model.pos_embed.weight.shape[0]

        # Row masks are keyed by (padded rows, real rows) and reused across images of the same
        # shape — a request usually sends several tiles of one size.
        self._row_mask_cache = {}

    # ------------------------------------------------------------------ helpers

    def _row_mask(self, rows, valid):
        """[1, 1, rows, 1] bf16, 1.0 below `valid` and 0.0 at/above it — built on device.

        Needed only because the projection has a bias: a zero-padded input row still comes out of
        the matmul as `bias`, where the host `F.pad` used to leave an exact zero.
        """
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

    # ------------------------------------------------------------------ forward

    def forward(self, pixel_values, bilinear_indices, bilinear_weights, seq_len):
        """
        Args:
            pixel_values (torch.Tensor): ``[n_patches, in_ch*T*P*P]`` patchified pixels, host.
            bilinear_indices (torch.Tensor): ``[4, n_patches]`` int corner indices into the
                positional table (from ``transformers.vision_utils`` — host index arithmetic).
            bilinear_weights (torch.Tensor): ``[4, n_patches]`` float corner weights.
            seq_len (int): padded sequence length the vision blocks run at.

        Returns:
            ttnn.Tensor: ``[1, 1, seq_len, dim_local]`` bf16 TILE, DRAM.
        """
        n = pixel_values.shape[0]
        assert n <= seq_len, f"{n} patches exceed the padded seq_len {seq_len}"
        assert (
            pixel_values.shape[1] == self.patch_dim
        ), f"expected patch dim {self.patch_dim}, got {pixel_values.shape[1]}"
        assert bilinear_indices.shape[1] == n, "one bilinear index column per patch"

        # Round the uploaded rows up to a tile so the matmul and the embedding agree; the rest of
        # the pad to seq_len is a tile-aligned device pad below (nothing to transfer for it).
        rows = ((n + 31) // 32) * 32

        # --- patch projection ---
        x = pixel_values
        if rows != n:
            x = torch.nn.functional.pad(x, (0, 0, 0, rows - n))
        x_tt = ttnn.from_torch(
            x.reshape(1, 1, rows, self.patch_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
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

        # --- interpolated positional embedding: sum_c w_c * table[idx_c] ---
        idx = bilinear_indices.to(torch.int32)
        wts = bilinear_weights.to(torch.float32)
        if rows != n:
            # Pad rows look up entry 0 with weight 0, contributing exactly nothing.
            idx = torch.nn.functional.pad(idx, (0, rows - n))
            wts = torch.nn.functional.pad(wts, (0, rows - n))
        idx_tt = ttnn.from_torch(
            idx,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        pos = ttnn.embedding(idx_tt, self.pos_table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        ttnn.deallocate(idx_tt)
        wts_tt = ttnn.from_torch(
            wts.reshape(4, rows, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        pos = ttnn.multiply(pos, wts_tt)  # broadcasts over the hidden dim
        ttnn.deallocate(wts_tt)
        pos_sum = ttnn.sum(pos, dim=0, keepdim=True)  # [1, rows, dim_local]
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
