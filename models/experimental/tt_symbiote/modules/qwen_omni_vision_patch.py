# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-Omni vision patch stem: HF ``Conv3d`` patch embed and ``Qwen3OmniMoeVisionPatchMerger`` on TTNN.

``Qwen3OmniMoeVisionPatchEmbed`` uses a strided ``Conv3d`` whose kernel equals the patch volume; it is
numerically equivalent to ``Linear(in_channels * T * H * W, embed_dim)`` on flattened patches.
<<<<<<< HEAD
=======

``TTNNQwen3OmniVisionPatchMerger`` matches ``Qwen3OmniMoeVisionPatchMerger.forward`` in HuggingFace; on mesh,
vision block outputs may be **width-sharded** (e.g. 1152 → 144 per device).  All-gather to full
``config.hidden_size`` **before** ``view(-1, merged_dim)`` or reshape to ``merged_dim`` will fail volume checks.
>>>>>>> ign/qwen3_omni_audio_issue
"""

from __future__ import annotations

import torch
from torch import nn
import ttnn

from models.experimental.tt_symbiote.core.module import TTNNModule, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.tt_symbiote.core.utils import tree_map
from models.experimental.tt_symbiote.modules.linear import (
    TTNNLinear,
    TTNNLinearIColShardedWAllReduced,
    TTNNLinearIReplicatedWColSharded,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNQwenLayerNorm


def _replicate_mapper(device):
    if device is None or device.get_num_devices() <= 1:
        return None
    return ttnn.ReplicateTensorToMesh(device)


def _ensure_ttnn(x, device, *, mesh_mapper=None):
    if isinstance(x, ttnn.Tensor):
        return x
    if isinstance(x, TorchTTNNTensor):
        if x.ttnn_tensor is not None:
            return x.ttnn_tensor
        if x.elem is not None:
            return ttnn.from_torch(
                x.elem.contiguous().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=mesh_mapper,
            )
    if isinstance(x, torch.Tensor):
        return ttnn.from_torch(
            x.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=mesh_mapper,
        )
    raise TypeError(f"_ensure_ttnn: unsupported type {type(x)}")


class TTNNQwen3OmniVisionPatchEmbed(TTNNModule):
    """TTNN patch embedding: flattened patch volume → ``ttnn.linear`` (Conv3d-equivalent)."""

    @classmethod
    def from_torch(cls, pe):
        m = cls()
        m._fallback_torch_layer = pe
        m.patch_size = int(pe.patch_size)
        m.temporal_patch_size = int(pe.temporal_patch_size)
        m.in_channels = int(pe.in_channels)
        m.embed_dim = int(pe.embed_dim)
        m._flat_dim = m.in_channels * m.temporal_patch_size * m.patch_size * m.patch_size

        w = pe.proj.weight.data.clone().to(torch.bfloat16)
        b = pe.proj.bias.data.clone().to(torch.bfloat16) if pe.proj.bias is not None else None
        w_flat = w.reshape(m.embed_dim, m._flat_dim)
        lin = nn.Linear(m._flat_dim, m.embed_dim, bias=b is not None)
        lin.weight.data.copy_(w_flat)
        if b is not None:
            lin.bias.data.copy_(b)
        m.linear = TTNNLinear.from_torch(lin)
        return m

    def preprocess_weights_impl(self):
        self.linear.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.linear.move_weights_to_device()

    def deallocate_weights_impl(self):
        self.linear.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        """Match vision hidden width after mesh readback (same idea as ``TTNNQwen3OmniVisionMLP``)."""
        if self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        def _materialize_one_replica(e):
            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.embed_dim)
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            if pt.shape[-1] > h:
                pt = pt[..., :h]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        return tree_map(_materialize_one_replica, output_tensors)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        mapper = _replicate_mapper(self.device)
        x = _ensure_ttnn(hidden_states, self.device, mesh_mapper=mapper)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Match HF ``view(-1, in_ch, T, P, P)`` then flatten patch volume for ``ttnn.linear``.
        x = ttnn.reshape(
            x,
            (-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size),
        )
        x = ttnn.reshape(x, (-1, self._flat_dim))

        return self.linear(x)


def _all_gather_width_until(h, *, target_w: int, max_steps: int = 16):
    """Increase last dim toward ``target_w`` (``TTNNQwen3OmniVisionMLP`` pattern; ring may need multiple steps)."""
    in_w = int(h.shape[-1])
    if in_w > target_w:
        rank = len(h.shape)
        starts = [0] * rank
        ends = [int(s) for s in h.shape]
        ends[-1] = target_w
        return ttnn.slice(h, starts, ends)
    for _ in range(max_steps):
        if int(h.shape[-1]) >= target_w:
            break
        h = ttnn.all_gather(
            h,
            dim=-1,
            cluster_axis=1,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
    in_w = int(h.shape[-1])
    if in_w < target_w:
        raise RuntimeError(
            f"TTNNQwen3OmniVisionPatchMerger: last dim {in_w} still < target width {target_w} after all_gather"
        )
    if in_w > target_w:
        rank = len(h.shape)
        starts = [0] * rank
        ends = [int(s) for s in h.shape]
        ends[-1] = target_w
        h = ttnn.slice(h, starts, ends)
    return h


class TTNNQwen3OmniVisionPatchMerger(TTNNModule):
    """TTNN ``Qwen3OmniMoeVisionPatchMerger``: LayerNorm + 2× Linear + GELU (TP pattern like vision MLP)."""

    @property
    def hidden_size(self) -> int:
        """HF name: merged width ``config.hidden_size * spatial_merge_size**2``."""
        return int(self.merged_dim)

    @classmethod
    def from_torch(cls, merger):
        m = cls()
        m._fallback_torch_layer = merger
        m.use_postshuffle_norm = bool(merger.use_postshuffle_norm)
        m.merged_dim = int(merger.hidden_size)
        m.out_hidden_size = int(merger.mlp[2].out_features)

        ln_dim = int(merger.ln_q.normalized_shape[0])
        if merger.use_postshuffle_norm:
            if ln_dim != m.merged_dim:
                raise ValueError(
                    "TTNNQwen3OmniVisionPatchMerger: postshuffle merger expects ln_q over merged width "
                    f"(got normalized_shape={merger.ln_q.normalized_shape}, merged_dim={m.merged_dim})"
                )
            cfg = getattr(merger, "config", None)
            sm = int(getattr(cfg, "spatial_merge_size", 2)) if cfg is not None else 2
            spatial_merge_sq = sm * sm
            m.vision_patch_hidden = m.merged_dim // spatial_merge_sq
            if m.vision_patch_hidden * spatial_merge_sq != m.merged_dim:
                raise ValueError(
                    f"TTNNQwen3OmniVisionPatchMerger: merged_dim={m.merged_dim} not divisible by spatial_merge_size^2={spatial_merge_sq}"
                )
        else:
            m.vision_patch_hidden = ln_dim

        m.ln_q = TTNNQwenLayerNorm.from_torch(merger.ln_q)
        if isinstance(m.ln_q, TTNNQwenLayerNorm) and m.use_postshuffle_norm:
            m.ln_q._symbiote_force_gather_layernorm = True
        m.lin1 = TTNNLinearIReplicatedWColSharded.from_torch(merger.mlp[0])
        m.lin2 = TTNNLinearIColShardedWAllReduced.from_torch(merger.mlp[2])

        # Bypass module_run TorchTTNNTensor wrapping on sub-modules so the MLP
        # path (lin1 → gelu → lin2) stays as raw ttnn.Tensor throughout.  Without
        # bypass, lin1 returns a TorchTTNNTensor that ttnn.gelu cannot properly
        # handle, corrupting the col-sharded feature data.
        if isinstance(m.ln_q, TTNNModule):
            m.ln_q._bypass_tensor_wrapping = True
        m.lin1._bypass_tensor_wrapping = True
        m.lin2._bypass_tensor_wrapping = True

        return m

    def preprocess_weights_impl(self):
        if isinstance(self.ln_q, TTNNQwenLayerNorm):
            self.ln_q.preprocess_weights()
        self.lin1.preprocess_weights()
        self.lin2.preprocess_weights()

    def move_weights_to_device_impl(self):
        if isinstance(self.ln_q, TTNNQwenLayerNorm):
            self.ln_q.move_weights_to_device()
        self.lin1.move_weights_to_device()
        self.lin2.move_weights_to_device()

    def deallocate_weights_impl(self):
        if isinstance(self.ln_q, TTNNQwenLayerNorm):
            self.ln_q.deallocate_weights()
        self.lin1.deallocate_weights()
        self.lin2.deallocate_weights()

    def set_output_tensors_config_impl(self, output_tensors):
        if self.device is None or self.device.get_num_devices() <= 1:
            return super().set_output_tensors_config_impl(output_tensors)

        def _materialize_one_replica(e):
            if not isinstance(e, TorchTTNNTensor) or e.ttnn_tensor is None:
                return e
            t = e.ttnn_tensor
            n = int(t.shape[0])
            h = int(self.out_hidden_size)
            pt = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
            if pt.shape[0] > n:
                pt = pt[:n]
            if pt.shape[-1] > h:
                pt = pt[..., :h]
            e.elem = pt.contiguous()
            e.ttnn_tensor = None
            if getattr(e, "_distributed_tensor_config", None) is not None:
                e._distributed_tensor_config = None
            return e

        return tree_map(_materialize_one_replica, output_tensors)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden):
        """Match HF ``Qwen3OmniMoeVisionPatchMerger.forward`` exactly.

        HF PyTorch reference::

            hidden = self.ln_q(
                hidden.view(-1, self.hidden_size) if self.use_postshuffle_norm else hidden
            ).view(-1, self.hidden_size)
            for layer in self.mlp:        # Linear → GELU → Linear
                hidden = layer(hidden)

        Sub-modules have ``_bypass_tensor_wrapping=True`` so all intermediate
        tensors are raw ``ttnn.Tensor`` — no ``TorchTTNNTensor`` wrapping between
        ``lin1 → gelu → lin2`` which would corrupt col-sharded feature data.
        """
        mapper = _replicate_mapper(self.device)
        h = _ensure_ttnn(hidden, self.device, mesh_mapper=mapper)

        vph = int(self.vision_patch_hidden)
        mh = int(self.merged_dim)

        # --- Step 1: gather width-sharded vision hidden to full per-token width ---
        if self.device is not None and self.device.get_num_devices() > 1:
            h = _all_gather_width_until(h, target_w=vph)
        else:
            in_w = int(h.shape[-1])
            if in_w > vph:
                rank = len(h.shape)
                starts = [0] * rank
                ends = [int(s) for s in h.shape]
                ends[-1] = vph
                h = ttnn.slice(h, starts, ends)

        h = ttnn.reshape(h, (-1, vph))

        # --- Step 2: postshuffle merges patches BEFORE LayerNorm ---
        if self.use_postshuffle_norm:
            h = ttnn.reshape(h, (-1, mh))

        if h.layout != ttnn.TILE_LAYOUT:
            h = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # --- Step 3: LayerNorm (bypass returns raw ttnn.Tensor) ---
        if isinstance(self.ln_q, TTNNQwenLayerNorm):
            h = self.ln_q(h)
        else:
            th = ttnn.to_torch(h)
            th = self.ln_q(th)
            h = ttnn.from_torch(
                th.contiguous().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=mapper,
            )

        # Ensure raw ttnn.Tensor after LN (bypass may still return TorchTTNNTensor
        # if LN is a plain nn.Module; TTNNQwenLayerNorm with bypass returns raw).
        if isinstance(h, TorchTTNNTensor):
            h = h.ttnn_tensor if h.ttnn_tensor is not None else _ensure_ttnn(h, self.device, mesh_mapper=mapper)
        if h.layout != ttnn.TILE_LAYOUT:
            h = ttnn.to_layout(h, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # --- Step 4: non-postshuffle merges patches AFTER LayerNorm ---
        h = ttnn.reshape(h, (-1, mh))

        # --- Step 5: MLP — Linear → GELU → Linear (all raw ttnn.Tensor) ---
        h = self.lin1(h)
        # Ensure raw ttnn.Tensor for ttnn.gelu (lin1 bypass should already return raw).
        if isinstance(h, TorchTTNNTensor):
            h = h.ttnn_tensor if h.ttnn_tensor is not None else _ensure_ttnn(h, self.device, mesh_mapper=mapper)
        h = ttnn.gelu(h, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h = self.lin2(h)

        # Trim output width if lin2 all-reduce produced wider than out_hidden_size.
        out_w = int(h.shape[-1])
        if out_w > int(self.out_hidden_size):
            rank = len(h.shape)
            starts = [0] * rank
            ends = [int(s) for s in h.shape]
            ends[-1] = int(self.out_hidden_size)
            h = ttnn.slice(h, starts, ends)

        return h
