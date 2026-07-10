# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Per-layer K/V cache for HunyuanImage-3.0 autoregressive decode on TTNN.
# Mirrors upstream HunyuanStaticCache (modeling_hunyuan_image_3.py) at a minimal
# subset: append-only keys/values per decoder layer, used after a prefix prefill.

from __future__ import annotations

import ttnn


class HunyuanTtKvCache:
    """Device-resident KV tensors per layer (pre-GQA-expansion layout)."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.keys: list[ttnn.Tensor | None] = [None] * num_layers
        self.values: list[ttnn.Tensor | None] = [None] * num_layers
        self.seq_len: int = 0
        # Trace replay: fixed-shape [B, kv_heads, max_len, head_dim] buffers + runtime write index.
        self.trace_fixed: bool = False
        self.max_len: int = 0
        self.write_pos_tt: ttnn.Tensor | None = None

    def get(self, layer_idx: int):
        return self.keys[layer_idx], self.values[layer_idx]

    def replace(self, layer_idx: int, key: ttnn.Tensor, value: ttnn.Tensor) -> None:
        if self.trace_fixed:
            raise RuntimeError("replace() invalid when trace_fixed KV buffers are active")
        old_k, old_v = self.keys[layer_idx], self.values[layer_idx]
        if old_k is not None:
            ttnn.deallocate(old_k)
        if old_v is not None:
            ttnn.deallocate(old_v)
        self.keys[layer_idx] = key
        self.values[layer_idx] = value

    def promote_to_trace_buffers(self, device, max_len: int) -> None:
        """Pad per-layer K/V to ``max_len`` for in-place decode writes during trace replay."""
        import torch

        multi = hasattr(device, "get_num_devices") and device.get_num_devices() > 1
        composer = ttnn.ConcatMeshToTensor(device, dim=0) if multi else None
        mapper = ttnn.ReplicateTensorToMesh(device) if multi else None

        def _to_host(t: ttnn.Tensor) -> torch.Tensor:
            if composer is not None:
                return ttnn.to_torch(t, mesh_composer=composer)[:1].float()
            return ttnn.to_torch(t).float()

        def _to_device(t: torch.Tensor) -> ttnn.Tensor:
            kwargs = dict(
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if mapper is not None:
                kwargs["mesh_mapper"] = mapper
            return ttnn.from_torch(t.to(torch.bfloat16), **kwargs)

        for i in range(self.num_layers):
            k, v = self.keys[i], self.values[i]
            if k is None or v is None:
                continue
            k_h = _to_host(k)
            v_h = _to_host(v)
            cur = int(k_h.shape[2])
            if cur > max_len:
                raise ValueError(f"KV len {cur} exceeds trace max_len {max_len}")
            if cur < max_len:
                pad = max_len - cur
                k_h = torch.nn.functional.pad(k_h, (0, 0, 0, pad))
                v_h = torch.nn.functional.pad(v_h, (0, 0, 0, pad))
            fixed_k = _to_device(k_h)
            fixed_v = _to_device(v_h)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            self.keys[i] = fixed_k
            self.values[i] = fixed_v
        self.trace_fixed = True
        self.max_len = max_len

    def set_write_pos_tensor(self, pos_tt: ttnn.Tensor) -> None:
        self.write_pos_tt = pos_tt

    def clear(self) -> None:
        for i in range(self.num_layers):
            if self.keys[i] is not None:
                ttnn.deallocate(self.keys[i])
            if self.values[i] is not None:
                ttnn.deallocate(self.values[i])
            self.keys[i] = None
            self.values[i] = None
        self.trace_fixed = False
        self.max_len = 0
        self.write_pos_tt = None
        self.seq_len = 0
