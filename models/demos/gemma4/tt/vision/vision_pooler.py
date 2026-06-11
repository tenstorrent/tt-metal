# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Gemma-4 vision pooler.

Mirrors HF ``Gemma4VisionPooler``: spatial (``k x k``) average pooling of the encoder
patch features down to ``output_length`` soft tokens, followed by a ``sqrt(hidden_size)``
scaling.

Split of work:
  * The pooling matrix and validity mask are derived purely from the (small, integer)
    ``pixel_position_ids`` / ``padding_positions`` metadata via index ops (``one_hot``,
    ``max``, integer division) that have no device equivalent, so they are built on host
    in torch -- identically to the reference.
  * The expensive ``weights^T @ hidden_states`` contraction and the ``sqrt(hidden_size)``
    scaling run on device with ``ttnn`` ops.

Note on dtype: the reference computes the scaling in float32 because ``sqrt(hidden_size)``
can push activations past the float16 range (max 65504). bfloat16 shares float32's 8-bit
exponent, so it does not overflow; the pooled features are produced in the working dtype
and the caller (``Gemma4VisionModel``) standardizes and casts as needed.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


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
        return ttnn.from_torch(
            torch_tensor,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if self.is_mesh_device else None,
        )

    def forward(self, hidden_states, pixel_position_ids, padding_positions, output_length):
        """Pool encoder patch features to soft tokens and scale by ``sqrt(hidden_size)``.

        Args:
            hidden_states: ttnn.Tensor ``[1, batch, seq, hidden_size]`` encoder output
                (replicated across the mesh), with the padding tokens already stripped to the
                true patch count (``seq == pixel_position_ids.shape[1]``).
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
            mask = padding_positions
            valid = torch.logical_not(padding_positions).to(torch.float32).unsqueeze(-1)  # [batch, seq, 1]
            valid_tt = self._to_device(valid.unsqueeze(0))  # [1, batch, seq, 1]
            pooled = ttnn.multiply(hidden_states, valid_tt)
            ttnn.deallocate(valid_tt)

        pooled = ttnn.multiply(pooled, self.root_hidden_size)
        return pooled, mask
