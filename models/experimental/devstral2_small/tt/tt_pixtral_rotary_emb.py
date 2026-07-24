# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.devstral2_small.devstral_utils.multimodal_demo_helpers import resolve_rope_parameters
from models.experimental.devstral2_small.devstral_utils.pixtral_seq_chunk import vision_rope_memcfg
from models.tt_transformers.tt.common import precompute_mistral_vision_freqs


class TtPixtralRotaryEmbedding(LightweightModule):
    """TT version of HF ``PixtralRotaryEmbedding``; cos/sin are gathered host-side and uploaded directly."""

    def __init__(self, mesh_device, config, datatype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.config = config
        self.datatype = datatype
        self.is_mesh_device = isinstance(mesh_device, ttnn._ttnn.multi_device.MeshDevice)

        rope_parameters = resolve_rope_parameters(config)
        rope_type = rope_parameters["rope_type"]
        if rope_type != "default":
            raise ValueError(f"TtPixtralRotaryEmbedding supports rope_type='default' only, got {rope_type!r}.")

        image_size = config.image_size[0] if isinstance(config.image_size, (tuple, list)) else int(config.image_size)
        patch_size = config.patch_size[0] if isinstance(config.patch_size, (tuple, list)) else int(config.patch_size)
        max_patches_per_side = image_size // patch_size

        self.head_dim = int(getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads))
        rope_theta = float(rope_parameters["rope_theta"])
        cos_torch, sin_torch = precompute_mistral_vision_freqs(
            dim=self.head_dim,
            max_patches_per_side=int(max_patches_per_side),
            theta=rope_theta,
            scale_factor=None,
            orig_context_len=None,
        )
        # Keep tables on host so position-id gather runs on host (no device embedding/untilize/tilize).
        self._cos_host = cos_torch
        self._sin_host = sin_torch
        self._mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.is_mesh_device else None

        self._cached_key: tuple | None = None
        self._cached_cos: ttnn.Tensor | None = None
        self._cached_sin: ttnn.Tensor | None = None

    def forward(self, x: ttnn.Tensor, position_ids: torch.Tensor):
        """``position_ids`` is a torch tensor on host. cos/sin are gathered host-side and uploaded directly."""
        if not isinstance(position_ids, torch.Tensor):
            raise TypeError(f"position_ids must be a torch.Tensor (host); got {type(position_ids).__name__}")
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        seq_len = int(position_ids.shape[-1])
        rope_mem_cfg = vision_rope_memcfg(seq_len, self.head_dim)

        idx = position_ids.flatten().to(torch.long)
        cache_key = (seq_len, self.head_dim, idx.cpu().numpy().tobytes())
        if cache_key != self._cached_key:
            # Cast to the target dtype on HOST so from_torch skips the device FP32->BF16 typecast
            # and tilizes the smaller bf16 tensor (the freq tables are precomputed in fp32). One-time
            # (cached below). bfloat8_b can't be represented host-side, so fall back to device convert.
            torch_dtype = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}.get(self.datatype)
            cos_sel = self._cos_host[idx].reshape(1, seq_len, self.head_dim)
            sin_sel = self._sin_host[idx].reshape(1, seq_len, self.head_dim)
            if torch_dtype is not None:
                cos_sel = cos_sel.to(torch_dtype)
                sin_sel = sin_sel.to(torch_dtype)
            cos_sel = cos_sel.contiguous()
            sin_sel = sin_sel.contiguous()

            # Tilize on HOST (from_torch without device=), then upload. Passing device= tilizes the
            # row-major upload ON DEVICE (two TilizeDeviceOperation ops on the trace). The tables are
            # one-time/cached, so the host tilize is paid once and never hits the device timeline.
            def _upload(host_torch: torch.Tensor) -> ttnn.Tensor:
                host_tile = ttnn.from_torch(
                    host_torch,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=self.datatype,
                    mesh_mapper=self._mesh_mapper,
                )
                return ttnn.to_device(host_tile, self.mesh_device, memory_config=rope_mem_cfg)

            self._cached_cos = _upload(cos_sel)
            self._cached_sin = _upload(sin_sel)
            self._cached_key = cache_key

        cos, sin = self._cached_cos, self._cached_sin
        if x.dtype != cos.dtype:
            cos = ttnn.typecast(cos, dtype=x.dtype)
            sin = ttnn.typecast(sin, dtype=x.dtype)
        return cos, sin


__all__ = ["TtPixtralRotaryEmbedding"]
