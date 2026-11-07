# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Gemma-4 vision tower.

Mirrors HF ``Gemma4VisionModel``: the full image encoder that maps raw patch pixels to pooled
soft tokens. The pipeline is

    patch embed (``VisionPatchEmbedder``) -> transformer encoder  (``VisionTransformer``)
    -> spatial average pooling                (``VisionPooler``)
    -> (optional) ``standardize`` affine ``(x - std_bias) * std_scale`` per hidden dim

This module holds those pieces (the rotary embedding, the patch embedder, the pooler, the encoder and the
tower that composes them; the transformer block lives in ``vision_block.py``). ``standardize`` is enabled for Gemma-4 (``config.standardize``
is ``True``); the per-hidden-dim ``std_bias`` / ``std_scale`` buffers are loaded as ttnn tensors
and applied on device. The affine is per-token independent, so applying it to the full pooled
tensor (then stripping padded soft tokens) is identical to the reference, which strips first.

Interface: like ``Gemma4VisionModel.forward``, this takes the host (torch) ``pixel_values`` and
``pixel_position_ids`` and does the host->device upload internally. The encoder runs on device;
the pooler builds its (tiny, integer) pooling matrix on host from the position metadata and runs
the heavy contraction on device.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.tt.vision.vision_block import VisionBlock
from models.tt_transformers.tt.common import get_rot_transformation_mat

# ── VisionRotaryEmbedding ─────────────────────────────────────────────────────────────────────────────────────────
# TTNN implementation of the Gemma-4 vision rotary embedding (cos/sin generation).
#
# Mirrors HF ``Gemma4VisionRotaryEmbedding``: 2D (x/y) RoPE where each spatial axis applies
# 1D RoPE over ``head_dim // 2`` channels, and the per-axis ``cos``/``sin`` are concatenated
# into a ``head_dim``-wide table.
#
# The reference computes, per axis ``i``::
#
#     freqs_i = position_ids[..., i] (outer) inv_freq          # [.., spatial_dim//2]
#     emb_i   = cat(freqs_i, freqs_i)                           # [.., spatial_dim]
#     cos = cat(cos(emb_0), cos(emb_1));  sin = cat(sin(emb_0), sin(emb_1))   # [.., head_dim]
#
# This is algebraically a single matmul: ``emb = position_ids @ freq_proj`` where
# ``freq_proj`` is the ``[2, head_dim]`` block-diagonal matrix whose row 0 holds the per-axis
# ``inv_freq`` in the first ``spatial_dim`` columns and row 1 holds it in the last ``spatial_dim``
# columns. Then ``cos = cos(emb)``, ``sin = sin(emb)`` reproduce the concatenated tables. Folding
# the per-axis duplication and concatenation into ``freq_proj`` (host-built) avoids the
# non-tile-aligned on-device concats that the literal formulation would require.
#
# Output layout: this produces the **Meta interleaved** convention (pairwise-duplicated:
# ``[c0, c0, c1, c1, ...]`` per spatial block), which is what the on-device attention's
# ``rotary_embedding_llama`` / ``apply_multidimensional_rope`` consumes. This is obtained for free
# by duplicating ``inv_freq`` with ``repeat_interleave`` (rather than the reference's half-stacked
# ``cat(inv_freq, inv_freq)``), and equals ``convert_rope_style_hf_to_meta_md`` applied to the HF
# cos/sin.
#
# Note: no typecasts are inserted; cos/sin are produced in the working dtype. The reference forces
# float32 for the trig (RoPE frequencies need precision), so if PCC is low the fix is to compute
# ``emb``/``cos``/``sin`` in float32.


class VisionRotaryEmbedding(LightweightModule):
    def __init__(self, mesh_device, args, dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.dtype = dtype

        vision_config = args.hf_config.vision_config
        head_dim = getattr(vision_config, "head_dim", None) or (
            vision_config.hidden_size // vision_config.num_attention_heads
        )
        base = vision_config.rope_parameters["rope_theta"]
        spatial_dim = head_dim // 2
        self.head_dim = head_dim
        self.attention_scaling = 1.0  # "default" rope_type

        # inv_freq over spatial_dim // 2 channels, identical for each spatial axis (matches reference).
        exponents = torch.arange(0, spatial_dim, 2, dtype=torch.int64).to(torch.float32) / spatial_dim
        inv_freq = 1.0 / (base**exponents)  # [spatial_dim // 2]
        # repeat_interleave (not cat) -> pairwise-duplicated freqs == Meta interleaved cos/sin layout.
        inv_freq_dup = torch.repeat_interleave(inv_freq, 2)  # [spatial_dim]

        # Block-diagonal [2, head_dim]: row 0 -> first spatial_dim cols (x axis), row 1 -> last (y axis).
        freq_proj = torch.zeros(2, head_dim, dtype=torch.float32)
        freq_proj[0, :spatial_dim] = inv_freq_dup
        freq_proj[1, spatial_dim:] = inv_freq_dup

        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        self.freq_proj = ttnn.as_tensor(
            freq_proj,
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None,
        )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, position_ids):
        """Generate the rotary cos/sin tables for the given patch positions.

        Args:
            position_ids: ttnn.Tensor ``[batch, num_patches, 2]`` patch (x, y) positions in the
                working dtype (e.g. bfloat16), TILE layout.

        Returns:
            cos, sin: ttnn.Tensor ``[1, batch, num_patches, head_dim]`` rotary tables.
        """
        pos = ttnn.unsqueeze_to_4D(position_ids)  # [1, batch, num_patches, 2]
        emb = ttnn.matmul(
            pos,
            self.freq_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
        )  # [1, batch, num_patches, head_dim]

        cos = ttnn.cos(emb)
        sin = ttnn.sin(emb)
        ttnn.deallocate(emb)

        if self.attention_scaling != 1.0:
            cos = ttnn.multiply(cos, self.attention_scaling)
            sin = ttnn.multiply(sin, self.attention_scaling)

        return cos, sin


# ── VisionPatchEmbedder ───────────────────────────────────────────────────────────────────────────────────────────
# TTNN implementation of the Gemma-4 vision patch embedder.
#
# Mirrors HF ``Gemma4VisionPatchEmbedder``:
#   * ``input_proj``: a (bias-free) linear projection of the flattened patch pixels
#     ``[batch, num_patches, 3*patch_size^2] -> [..., hidden_size]`` -- a real matmul on
#     real pixel data, run on device.
#   * position embeddings: the reference computes ``one_hot(position_ids) @ position_embedding_table``
#     summed over the x/y spatial axes, which is exactly two embedding-table lookups, so we use
#     ``ttnn.embedding`` against the two ``[position_embedding_size, hidden_size]`` tables and add.
#   * padding patches are zeroed and the projected patches and position embeddings are summed.
#
# All inputs (``pixel_values``, ``pixel_position_ids``, ``padding_positions``) are on-device
# ``ttnn`` tensors: the position-id clamp/slice, embedding lookups, padding mask, matmul and add
# all run on device with no host round-trip.


class VisionPatchEmbedder(LightweightModule):
    def __init__(
        self, mesh_device, args, state_dict, state_dict_prefix="", weight_cache_path=None, dtype=ttnn.bfloat16
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.dtype = dtype

        vision_config = args.hf_config.vision_config
        self.hidden_size = vision_config.hidden_size
        self.patch_size = vision_config.patch_size
        self.position_embedding_size = vision_config.position_embedding_size

        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"
        mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None

        def cache_name(name):
            if args.dummy_weights or weight_cache_path is None:
                return None
            return weight_cache_path / f"{state_dict_prefix}{name}"

        # input_proj weight is [hidden_size, in_dim]; transpose to [in_dim, hidden_size] for x @ W.
        proj_weight = state_dict[f"{state_dict_prefix}input_proj.weight"]
        self.input_proj_weight = ttnn.as_tensor(
            proj_weight.transpose(-2, -1).contiguous(),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
            cache_file_name=cache_name("input_proj"),
        )

        # position_embedding_table is [2, position_embedding_size, hidden_size]: one table per spatial axis.
        position_embedding_table = state_dict[f"{state_dict_prefix}position_embedding_table"]
        self.position_tables = []
        for axis in range(2):
            self.position_tables.append(
                ttnn.as_tensor(
                    position_embedding_table[axis].contiguous(),
                    dtype=dtype,
                    device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mesh_mapper,
                    cache_file_name=cache_name(f"position_embedding_table_{axis}"),
                )
            )

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, pixel_values, pixel_position_ids, padding_positions):
        """Embed flattened image patches and add 2D positional embeddings.

        Args:
            pixel_values: ttnn.Tensor ``[1, batch, num_patches, 3*patch_size^2]`` flattened patch pixels.
            pixel_position_ids: ttnn.Tensor (int32, ROW_MAJOR) ``[batch, num_patches, 2]`` patch (x, y)
                positions (padding patches are ``(-1, -1)``).
            padding_positions: ttnn.Tensor (uint32/int32, ROW_MAJOR) ``[batch, num_patches]``
                (nonzero = padding patch).

        Returns:
            ttnn.Tensor ``[1, batch, num_patches, hidden_size]`` patch embeddings.
        """
        batch, num_patches, _ = pixel_position_ids.shape

        # Gemma4 applies no normalization, just a [-1, 1] rescale, then projects.
        pixel_values = ttnn.subtract(pixel_values, 0.5)
        pixel_values = ttnn.multiply(pixel_values, 2.0)
        hidden_states = ttnn.matmul(
            pixel_values,
            self.input_proj_weight,
            compute_kernel_config=self.compute_kernel_config,
            dtype=self.dtype,
        )

        # one_hot(position_ids) @ table, summed over the x/y axes == two embedding lookups + add.
        # Clamp the -1 padding ids to 0 so they form valid (uint32) gather indices; the resulting
        # padding rows are zeroed out afterwards via the validity mask.
        clamped_positions = ttnn.clamp(pixel_position_ids, min=0)
        pos_emb = None
        for axis in range(2):
            ids = ttnn.slice(clamped_positions, [0, 0, axis], [batch, num_patches, axis + 1])  # [b, p, 1]
            ids = ttnn.reshape(ids, [batch, num_patches])
            ids = ttnn.typecast(ids, ttnn.uint32)
            axis_emb = ttnn.embedding(ids, self.position_tables[axis], layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
            ttnn.deallocate(ids)
            pos_emb = axis_emb if pos_emb is None else ttnn.add(pos_emb, axis_emb)
        ttnn.deallocate(clamped_positions)

        pos_emb = ttnn.unsqueeze_to_4D(pos_emb)  # [batch, num_patches, hidden] -> [1, batch, num_patches, hidden]

        # Zero out the position embeddings of padding patches (reference uses torch.where).
        # padding_positions is {0, 1}, so the validity mask is 1 - padding. (logical_not has no
        # uint8 kernel, so build it arithmetically in bf16.)
        valid = ttnn.reshape(padding_positions, [1, batch, num_patches, 1])  # ROW_MAJOR
        valid = ttnn.to_layout(valid, ttnn.TILE_LAYOUT)
        valid = ttnn.typecast(valid, self.dtype)
        valid = ttnn.multiply(valid, -1.0)
        valid = ttnn.add(valid, 1.0)
        pos_emb = ttnn.multiply(pos_emb, valid)
        ttnn.deallocate(valid)

        out = ttnn.add(hidden_states, pos_emb)
        ttnn.deallocate(pos_emb)
        return out


# ── VisionPooler ──────────────────────────────────────────────────────────────────────────────────────────────────
# TTNN implementation of the Gemma-4 vision pooler.
#
# Mirrors HF ``Gemma4VisionPooler``: spatial (``k x k``) average pooling of the encoder
# patch features down to ``output_length`` soft tokens, followed by a ``sqrt(hidden_size)``
# scaling.
#
# Split of work:
#   * The pooling matrix and validity mask are derived purely from the (small, integer)
#     ``pixel_position_ids`` / ``padding_positions`` metadata via index ops (``one_hot``,
#     ``max``, integer division) that have no device equivalent, so they are built on host
#     in torch -- identically to the reference.
#   * The expensive ``weights^T @ hidden_states`` contraction and the ``sqrt(hidden_size)``
#     scaling run on device with ``ttnn`` ops.
#
# Note on dtype: the reference computes the scaling in float32 because ``sqrt(hidden_size)``
# can push activations past the float16 range (max 65504). bfloat16 shares float32's 8-bit
# exponent, so it does not overflow; the pooled features are produced in the working dtype
# and the caller (``Gemma4VisionModel``) standardizes and casts as needed.


class VisionPooler(LightweightModule):
    def __init__(self, mesh_device, args, dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        vision_config = args.hf_config.vision_config
        self.hidden_size = vision_config.hidden_size
        self.pooling_kernel_size = vision_config.pooling_kernel_size
        self.root_hidden_size = self.hidden_size**0.5
        self.dtype = dtype

        self.is_mesh_device = mesh_device.__class__.__name__ == "MeshDevice"

        # HiFi4 + fp32 accumulation keeps the small (1/k^2) pooling weights accurate.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _pooling_weights(self, pixel_position_ids, padding_positions, length):
        """Host-side construction of the pooling matrix and validity mask.

        Mirrors ``Gemma4VisionPooler._avg_pool_by_positions`` but returns the matrix already
        transposed for ``weights @ hidden_states`` and with the ``masked_fill`` of padding
        patches folded in (their columns are zeroed) so the device matmul needs no separate
        masking step.

        Args:
            pixel_position_ids: torch.LongTensor ``[batch, seq, 2]`` patch (x, y) positions
                (padding patches are ``(-1, -1)``).
            padding_positions: torch.BoolTensor ``[batch, seq]`` (True = padding patch).
            length: int target number of soft tokens (``output_length``).

        Returns:
            weights_t: torch.Tensor ``[batch, length, seq]`` pooling matrix.
            mask: torch.BoolTensor ``[batch, length]`` (True = valid output token).
        """
        input_seq_len = pixel_position_ids.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool seq_len={input_seq_len} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )

        # Clamp padding positions (which are -1) to 0 so they don't break one_hot.
        clamped_positions = pixel_position_ids.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = torch.nn.functional.one_hot(kernel_idxs.long(), length).float() / k_squared  # [batch, seq, length]

        # Validity mask is computed from the unmasked weights (matches the reference, where
        # padding patches still map to a kernel cell via the clamp).
        mask = torch.logical_not((weights == 0).all(dim=1))  # [batch, length]

        # Fold the reference's `hidden_states.masked_fill(padding_positions, 0)` into the
        # matrix: zeroing the columns of padding patches yields the same pooled result.
        weights = weights.masked_fill(padding_positions.unsqueeze(-1), 0.0)

        weights_t = weights.transpose(1, 2).contiguous()  # [batch, length, seq]
        return weights_t, mask

    def _to_device(self, torch_tensor):
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device) if self.is_mesh_device else None
        return ttnn.from_torch(
            torch_tensor,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length):
        """Pool encoder patch features to soft tokens and scale by ``sqrt(hidden_size)``.

        Args:
            hidden_states: ttnn.Tensor ``[1, batch, seq, hidden_size]`` encoder output
                (replicated across the mesh), with the padding tokens already stripped to
                the true patch count (``seq == pixel_position_ids.shape[1]``).
            pixel_position_ids: torch.LongTensor ``[batch, seq, 2]`` patch positions.
            padding_positions: torch.BoolTensor ``[batch, seq]`` (True = padding patch).
            output_length: int target number of soft tokens.

        Returns:
            pooled: ttnn.Tensor ``[1, batch, output_length, hidden_size]`` scaled soft tokens.
            mask: torch.BoolTensor ``[batch, output_length]`` (True = valid token), for the
                caller to strip padded soft tokens (``hidden_states[mask]``).
        """
        seq_len = hidden_states.shape[-2]
        if output_length > seq_len:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are patches ({seq_len})."
            )

        if seq_len != output_length:
            weights_t, mask = self._pooling_weights(pixel_position_ids, padding_positions, output_length)
            # [batch, length, seq] -> [1, batch, length, seq] to batch-matmul against hidden_states.
            weights_tt = self._to_device(weights_t.unsqueeze(0))
            pooled = ttnn.matmul(
                weights_tt,
                hidden_states,
                compute_kernel_config=self.compute_kernel_config,
                dtype=self.dtype,
            )
            ttnn.deallocate(weights_tt)
        else:
            # No pooling: just zero out the padding patches (the reference's masked_fill).
            # The mask marks the VALID patches, like the pooled branch and the callers
            # (which strip ``hidden_states[~mask]``) expect.
            mask = torch.logical_not(padding_positions)
            valid = torch.logical_not(padding_positions).to(torch.float32).unsqueeze(-1)  # [batch, seq, 1]
            valid_tt = self._to_device(valid.unsqueeze(0))  # [1, batch, seq, 1]
            pooled = ttnn.multiply(hidden_states, valid_tt)
            ttnn.deallocate(valid_tt)

        pooled = ttnn.multiply(pooled, self.root_hidden_size)
        return pooled, mask


# ── VisionTransformer ─────────────────────────────────────────────────────────────────────────────────────────────
# The encoder: 2D rotary cos/sin generation -> transformer blocks (``VisionBlock``).


class VisionTransformer(LightweightModule):
    """
    Gemma-4 vision encoder.

    Runs the on-device transformer stack: 2D rotary cos/sin generation -> transformer blocks.
    Patch embedding is done upstream in ``VisionTower``; patch merging / pooling is done
    downstream. This module operates on already-embedded patch hidden states.
    """

    def __init__(
        self,
        args,
        dtype,
        state_dict,
        tt_ccl,
        weight_cache_path,
    ):
        """
        Initialize the Vision Transformer model.

        Args:
            args (VisionModelArgs): Model arguments
            dtype (ttnn.dtype): Data type for computations
            mesh_device (ttnn.mesh_device): Mesh device for the model
            state_dict (dict): State dictionary containing model weights
            weight_cache_path (str): Path to weight cache
        """
        super().__init__()
        self.args = args
        self.dtype = dtype
        self.weight_cache_path = weight_cache_path

        # Create transformation matrix for RoPE QK prefill
        transformation_mat_torch = get_rot_transformation_mat(
            args.head_dim
        )  # todo)) args.head_dim is ignored inside the function
        self.transformation_mats = {
            "prefill": ttnn.as_tensor(
                transformation_mat_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=args.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(args.mesh_device),
            )
        }

        # Rotary cos/sin generator (patch embedding lives upstream in ``VisionTower``).
        self.rotary_embedding = VisionRotaryEmbedding(
            mesh_device=args.mesh_device,
            args=args,
            dtype=ttnn.bfloat16,
        )

        # Create vision blocks
        self.blocks = []
        for i in range(args.hf_config.vision_config.num_hidden_layers):
            block = VisionBlock(
                mesh_device=args.mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                tt_ccl=tt_ccl,
                dtype=dtype,
                transformation_mats=self.transformation_mats,
                args=args,
            )
            self.blocks.append(block)

    def forward(
        self,
        inputs_embeds,
        pixel_position_ids,
        unpadded_seq_len,
        seq_len,
        num_valid_patches=None,
    ):
        """
        Vision encoder forward: rotary cos/sin -> transformer blocks.

        Args:
            inputs_embeds (ttnn.Tensor): Embedded patch hidden states ``[1, batch, num_patches, hidden_dim]``
                (produced upstream by ``VisionTower``'s patch embedder).
            pixel_position_ids (ttnn.Tensor): Patch (x, y) positions ``[batch, num_patches, 2]``
                (int32, ROW_MAJOR; padding patches are ``(-1, -1)``).
            unpadded_seq_len (int): True number of patches (output is sliced back to this).
            seq_len (int): Padded sequence length the blocks run at.

        Returns:
            ttnn.Tensor: Encoder output ``[1, batch, unpadded_seq_len, hidden_dim]``.
        """
        num_patches = pixel_position_ids.shape[1]

        # Pad the embedded sequence seq -> seq_len. When padding, ``ttnn.pad`` allocates a new
        # tensor, so free the caller's input here; in the no-pad case the first block frees it.
        x = inputs_embeds
        if seq_len > num_patches:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, seq_len - num_patches), (0, 0)], value=0.0)
            ttnn.deallocate(inputs_embeds)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # Rotary cos/sin (Meta interleaved). The rotary matmul needs a float position tensor, so we
        # apply a functional int->bf16 cast here (the rotary module itself stays typecast-free).
        pos_tile = ttnn.to_layout(pixel_position_ids, ttnn.TILE_LAYOUT)
        pos_bf16 = ttnn.typecast(pos_tile, ttnn.bfloat16)
        ttnn.deallocate(pos_tile)
        cos, sin = self.rotary_embedding(pos_bf16)  # [1, B, num_patches, head_dim]
        ttnn.deallocate(pos_bf16)
        if seq_len > num_patches:
            # Pad the sequence with an identity rotation (cos=1, sin=0) for the padding tokens.
            cos = ttnn.pad(cos, [(0, 0), (0, 0), (0, seq_len - num_patches), (0, 0)], value=1.0)
            sin = ttnn.pad(sin, [(0, 0), (0, 0), (0, seq_len - num_patches), (0, 0)], value=0.0)
        rot_mats = [cos, sin]

        # Bidirectional padding mask, as window boundaries. ``Gemma4VisionModel``
        # passes ``attention_mask=~padding_positions`` to its encoder; without an
        # equivalent here every real patch attends to the (-1,-1) padding patches
        # in all 27 blocks. ``num_valid_patches`` counts the real ones; the rest
        # of ``seq_len`` is padding (image padding first, then any tile padding),
        # and both are a contiguous suffix, so a single split is the whole mask.
        cu_window_seqlens = None
        if num_valid_patches is not None and 0 < num_valid_patches < seq_len:
            cu_window_seqlens = ttnn.from_torch(
                torch.tensor([0, int(num_valid_patches), int(seq_len)], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.args.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.args.mesh_device)
                if self.args.mesh_device.get_num_devices() > 1
                else None,
            )

        for block in self.blocks:
            x = block(
                x,
                rot_mats=rot_mats,
                cu_window_seqlens=cu_window_seqlens,
            )

        x = x[:, :, :unpadded_seq_len, :]
        return x


# ── VisionTower ───────────────────────────────────────────────────────────────────────────────────────────────────
# The composed tower (HF ``Gemma4VisionModel``).


class VisionTower(LightweightModule):
    def __init__(
        self,
        args,
        dtype,
        state_dict,
        tt_ccl,
        weight_cache_path,
    ):
        """
        Args:
            args (VisionModelArgs): Model arguments.
            dtype (ttnn.dtype): Data type for the transformer blocks.
            state_dict (dict): State dictionary containing model weights.
            tt_ccl (TT_CCL): Collective-communication helper for the encoder blocks.
            weight_cache_path: Path to the weight cache.
        """
        super().__init__()
        self.args = args
        self.mesh_device = args.mesh_device

        vision_config = args.hf_config.vision_config
        self.pooling_kernel_size = vision_config.pooling_kernel_size

        # Patch embedder (input_proj + 2D positional embeddings): projects raw patch pixels to
        # hidden states before the transformer encoder.
        self.patch_embedder = VisionPatchEmbedder(
            mesh_device=self.mesh_device,
            args=args,
            state_dict=state_dict,
            state_dict_prefix=f"{args.get_state_dict_prefix('VisionTransformer')}.patch_embedder.",
            weight_cache_path=weight_cache_path,
            dtype=ttnn.bfloat16,
        )
        self.encoder = VisionTransformer(
            args=args,
            dtype=dtype,
            state_dict=state_dict,
            tt_ccl=tt_ccl,
            weight_cache_path=weight_cache_path,
        )
        self.pooler = VisionPooler(mesh_device=self.mesh_device, args=args, dtype=ttnn.bfloat16)

        self.is_mesh_device = self.mesh_device.__class__.__name__ == "MeshDevice"

        # Optional post-pool standardization: (x - std_bias) * std_scale per hidden dim.
        self.standardize = getattr(vision_config, "standardize", False)
        self.std_bias = None
        self.std_scale = None
        if self.standardize:
            prefix = args.get_state_dict_prefix("VisionTransformer")  # "visual"
            self.std_bias = self._load_affine(state_dict[f"{prefix}.std_bias"], weight_cache_path, "std_bias")
            self.std_scale = self._load_affine(state_dict[f"{prefix}.std_scale"], weight_cache_path, "std_scale")

    def _load_affine(self, weight, weight_cache_path, name):
        """Load a ``[hidden_size]`` standardize buffer as a ``[1, 1, 1, hidden_size]`` ttnn tensor."""
        cache_name = None
        if not self.args.dummy_weights and weight_cache_path is not None:
            cache_name = weight_cache_path / f"visual.{name}"
        return ttnn.as_tensor(
            weight.reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate_mapper(),
            cache_file_name=cache_name,
        )

    def _replicate_mapper(self):
        return ttnn.ReplicateTensorToMesh(self.mesh_device) if self.is_mesh_device else None

    def forward(self, pixel_values, pixel_position_ids, seq_len):
        """Encode ONE image's patches to pooled soft tokens.

        The tower is tensor-parallel across the whole mesh (see ``VisionAttention`` /
        ``Gemma4VisionMLP``), so it always runs a single image and every device holds
        the full activation. Callers with more than one image loop over users.

        Args:
            pixel_values (torch.Tensor): Flattened patch pixels ``[1, num_patches, 3*patch_size^2]``.
            pixel_position_ids (torch.LongTensor): Patch (x, y) positions ``[1, num_patches, 2]``
                (padding patches are ``(-1, -1)``).
            seq_len (int): Padded sequence length the encoder blocks run at (>= num_patches).

        Returns:
            pooled (ttnn.Tensor): ``[1, 1, output_length, hidden_size]`` scaled soft tokens,
                replicated across the mesh.
            mask (torch.BoolTensor): ``[1, output_length]`` (True = valid token); use it to strip
                padded soft tokens (``pooled[mask]``), matching ``Gemma4VisionModel``.
        """
        batch = pixel_position_ids.shape[0]
        if batch != 1:
            raise ValueError(
                f"VisionTower is tensor-parallel and runs batch=1 only (got batch={batch}). "
                "Loop over users and call it once per image."
            )
        padding_positions = (pixel_position_ids == -1).all(dim=-1)  # [1, num_patches]
        num_patches = pixel_position_ids.shape[1]
        output_length = num_patches // (self.pooling_kernel_size**2)

        replicate = self._replicate_mapper()
        pixel_values_tt = ttnn.from_torch(
            pixel_values.unsqueeze(0),  # [1, 1, num_patches, in_dim]
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        position_ids_tt = ttnn.from_torch(
            pixel_position_ids.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        padding_positions_tt = ttnn.from_torch(
            padding_positions.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )

        # Patch embedding (projected patches + 2D positional embeddings).
        inputs_embeds = self.patch_embedder(pixel_values_tt, position_ids_tt, padding_positions_tt)
        ttnn.deallocate(pixel_values_tt)
        ttnn.deallocate(padding_positions_tt)

        # Encoder: rotary + transformer blocks. Consumes ``inputs_embeds`` (freed internally).
        # Output is sliced back to the true patch count (padding patches included; the pooler
        # masks them out).
        # Real patch count for the encoder's bidirectional padding mask. Note this
        # is a DIFFERENT padding from ``unpadded_seq_len``: that one is tile
        # padding (num_patches -> seq_len), this one is the (-1,-1) image padding
        # inside num_patches. Only the former was handled before.
        num_valid_patches = int((~padding_positions).sum().item())
        encoder_output = self.encoder(
            inputs_embeds,
            position_ids_tt,
            unpadded_seq_len=num_patches,
            seq_len=seq_len,
            num_valid_patches=num_valid_patches,
        )
        ttnn.deallocate(position_ids_tt)

        # Pooler builds its weights on host from the (torch) position metadata.
        pooled, mask = self.pooler(
            encoder_output,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )
        ttnn.deallocate(encoder_output)

        # Standardize: (x - std_bias) * std_scale, broadcast over the per-hidden-dim buffers.
        if self.standardize:
            pooled = ttnn.subtract(pooled, self.std_bias)
            pooled = ttnn.multiply(pooled, self.std_scale)

        return pooled, mask
