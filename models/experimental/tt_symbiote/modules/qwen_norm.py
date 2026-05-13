# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""TTNN ports of Qwen3.5 / Qwen3.6 normalization layers.

Currently provides:

- ``TTNNQwen3MoeRMSNorm``: drop-in replacement for transformers'
  ``Qwen3_5MoeRMSNorm``. The HF op is a zero-centered RMS norm with a learned
  residual scale:

      output = x * rsqrt(mean(x^2) + eps) * (1.0 + weight)

  where ``weight`` is zero-initialized. We pre-fold the ``+1`` constant into
  the TTNN weight at staging time so the captured trace is a single
  ``ttnn.rms_norm`` (single device) or the standard distributed
  pre-/all-gather/post pattern (multi-device, T3K and similar).
"""

import torch
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.run_config import trace_enabled


@trace_enabled
class TTNNQwen3MoeRMSNorm(TTNNModule):
    """TTNN drop-in for ``transformers.models.qwen3_5_moe.Qwen3_5MoeRMSNorm``.

    Replaces the per-decoder-layer torch fallbacks (input_layernorm and
    post_attention_layernorm) plus the final ``model.norm`` so they execute
    natively on device. Without this port each call round-trips activations
    through ``unwrap_to_torch`` / ``wrap_to_torch_ttnn_tensor``, which on
    Qwen3.6-35B-A3B costs roughly ~150 ms per generated token across the 40
    layers (see ``qwen3_6_35b_a3b_timing_stats_pivot.csv``).

    The residual stream feeding this norm is column-sharded on the hidden
    dimension across the cluster (the previous attention/MoE block returns a
    reduce-scatter output), so when running on a multi-device mesh we use the
    distributed pre/all-gather/post split: each device computes a partial
    sum-of-squares, the small statistics tensor is all-gathered, and the
    finalization multiplies the per-shard ``(1 + weight)`` slice with the
    per-shard input. The output stays column-sharded, matching the input
    contract that ``TTNNQwen3FullAttention`` / ``TTNNQwen3LinearAttention`` /
    ``TTNNQwen3MoE`` already expect.

    On a single-device mesh we fall back to a plain ``ttnn.rms_norm`` over a
    replicated weight tensor.
    """

    @classmethod
    def from_torch(cls, qwen_rms_norm):
        """Wrap a ``Qwen3_5MoeRMSNorm`` (or any module with ``.weight`` and ``.eps``).

        The class is intentionally permissive about the source attribute names
        because Qwen3.5 uses ``eps`` while several adjacent norms in
        transformers use ``variance_epsilon``.
        """
        if not hasattr(qwen_rms_norm, "weight") or qwen_rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {qwen_rms_norm} has no weight; leaving torch fallback in place.")
            return qwen_rms_norm
        instance = cls()
        instance._fallback_torch_layer = qwen_rms_norm
        return instance

    @property
    def _is_distributed(self) -> bool:
        return self.device is not None and self.device.get_num_devices() > 1

    @staticmethod
    def _resolve_eps(torch_layer) -> float:
        return getattr(torch_layer, "eps", getattr(torch_layer, "variance_epsilon", 1e-6))

    def preprocess_weights_impl(self):
        """Stage ``(1 + weight)`` once on the host and capture the eps value.

        We fold the ``+1`` constant into the host tensor here so that downstream
        device staging just shards / replicates a normal RMSNorm weight. Padded
        positions get ``1.0`` (no-op scaling) since the input shard's matching
        positions are zero-padded by ``rms_norm_post_all_gather``.
        """
        weight = self._fallback_torch_layer.weight.detach().to(torch.bfloat16)
        # `(1 + w)` is the effective RMSNorm scale used by Qwen3_5MoeRMSNorm.
        self._scale_torch_host = (weight + 1.0).contiguous()
        self._eps = self._resolve_eps(self._fallback_torch_layer)
        self._dim = int(self._scale_torch_host.shape[0])
        # Pad to a multiple of TILE_SIZE so the device tensor has whole tiles
        # along the hidden dim. Padded positions use 1.0 because their
        # corresponding input positions are zeros.
        tile = ttnn.TILE_SIZE
        self._padded_dim = ((self._dim + tile - 1) // tile) * tile
        if self._padded_dim != self._dim:
            self._scale_torch_host = torch.nn.functional.pad(
                self._scale_torch_host, (0, self._padded_dim - self._dim), value=1.0
            )

    def move_weights_to_device_impl(self):
        """Stage the folded scale tensor on device.

        Two paths:

        - **Distributed (mesh > 1 device)**: shard the scale on the hidden dim
          to match the input's column-sharding. We mirror the layout used by
          ``TTNNDistributedRMSNorm`` (``[1, 1, padded_dim/32, 32]`` reshape +
          ``ShardTensor2dMesh``) so the result is compatible with
          ``ttnn.rms_norm_post_all_gather``.

        - **Single device**: replicate the weight as a flat ``[1, padded_dim]``
          tile-laid-out tensor for the standard ``ttnn.rms_norm`` op.
        """
        scale_bf16 = self._scale_torch_host
        padded_dim = self._padded_dim

        if self._is_distributed:
            # Shard the folded scale along the hidden dim across the mesh's
            # column axis so each device holds the slice that aligns with its
            # column-sharded input. Layout matches TTNNDistributedRMSNorm.
            staged = (
                scale_bf16.unsqueeze(0)
                .view(1, 1, padded_dim)
                .reshape(1, 1, padded_dim // ttnn.TILE_SIZE, ttnn.TILE_SIZE)
            )
            self._scale_dev = ttnn.as_tensor(
                staged,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.device, dims=(None, 2), mesh_shape=list(self.device.shape)),
            )
            self._scale_dev = ttnn.to_device(self._scale_dev, self.device)
        else:
            # Single-device path: replicated 2D tile weight, suitable for the
            # fused `ttnn.rms_norm` kernel.
            self._scale_dev = ttnn.from_torch(
                scale_bf16.unsqueeze(0),  # [1, padded_dim]
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        # Free the host copy now that the device tensor is staged.
        self._scale_torch_host = None

    def forward(self, x):
        """Normalize ``x`` over its last dimension.

        Accepts 3D ``(B, T, H)`` or 4D ``(B, 1, T, H)`` tensors and preserves
        the original rank in the returned tensor.
        """
        original_rank = len(x.shape)
        if original_rank == 3:
            x = ttnn.unsqueeze(x, 1)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if self._is_distributed:
            out = self._forward_distributed(x)
        else:
            out = ttnn.rms_norm(x, weight=self._scale_dev, epsilon=self._eps)

        if original_rank == 3 and len(out.shape) == 4:
            out = ttnn.reshape(out, [out.shape[0], out.shape[2], out.shape[3]])
        return out

    @run_on_devices(DeviceArch.T3K, DeviceArch.QB2)
    def _forward_distributed(self, x):
        # Pre-all-gather: per-device partial statistics (sum of squares).
        stats = ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.bfloat16)
        # All-gather the small statistics tensor (Ring topology for trace
        # compatibility, matching TTNNDistributedRMSNorm).
        stats = ttnn.all_gather(stats, dim=-1, num_links=1, topology=ttnn.Topology.Ring)
        # Post-all-gather: complete the normalization with the per-shard
        # `(1 + weight)` slice we pre-folded in preprocess.
        out = ttnn.rms_norm_post_all_gather(
            x,
            stats,
            epsilon=self._eps,
            weight=self._scale_dev,
        )
        stats.deallocate(True)
        return out
