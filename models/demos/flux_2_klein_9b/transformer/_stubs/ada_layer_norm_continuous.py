# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `AdaLayerNormContinuous` (`norm_out`) of
`black-forest-labs/FLUX.2-klein-9B (subfolder `transformer`)` (FLUX.2 Klein 9B).

Reference (diffusers `models/normalization.py::AdaLayerNormContinuous`)::

    emb          = self.linear(self.silu(conditioning_embedding).to(x.dtype))
    scale, shift = emb.chunk(2, dim=1)
    x            = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]

For this checkpoint the inner norm is `nn.LayerNorm(4096, eps=1e-6,
elementwise_affine=False, bias=False)` -- no gamma/beta -- and `self.linear` is
`nn.Linear(4096, 8192, bias=False)`. Shapes seen live: x `(1, S, 4096)`,
conditioning_embedding `(1, 4096)`.

The forward is pure ttnn: no torch math and no device->host readback. torch is
used only in `__init__`, to transpose the checkpoint weight into ttnn's
`[in, out]` layout once before staging it on device.
"""

from __future__ import annotations

import torch

import ttnn


def _mesh_shape(device):
    """(rows, cols) of `device` as a mesh, or None if it is a single device."""
    shape = getattr(device, "shape", None)
    if shape is None:
        return None
    try:
        dims = [int(d) for d in shape]
    except TypeError:
        return None
    if len(dims) == 1:
        dims = [1, dims[0]]
    if len(dims) != 2:
        return None
    return dims[0], dims[1]


def _num_devices(device) -> int:
    shape = _mesh_shape(device)
    return shape[0] * shape[1] if shape is not None else 1


def _replicate_mapper(device):
    return ttnn.ReplicateTensorToMesh(device) if _num_devices(device) > 1 else None


class TtAdaLayerNormContinuous:
    def __init__(self, device, torch_module) -> None:
        if torch_module is None:
            raise RuntimeError("ada_layer_norm_continuous needs the torch reference module for weights")

        self.device = device
        linear = torch_module.linear
        norm = torch_module.norm

        self.cond_dim = int(linear.in_features)
        self.embedding_dim = int(linear.out_features) // 2
        self.eps = float(getattr(norm, "eps", 1e-6) or 1e-6)

        # torch keeps [out, in]; ttnn wants [in, out].
        w = linear.weight.detach().to(torch.float32).t().contiguous()
        self.weight = ttnn.from_torch(
            w.reshape(1, 1, self.cond_dim, 2 * self.embedding_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(device),
        )
        bias = getattr(linear, "bias", None)
        self.bias = (
            None
            if bias is None
            else ttnn.from_torch(
                bias.detach().to(torch.float32).reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=_replicate_mapper(device),
            )
        )

        # `elementwise_affine=False` for this checkpoint, but honour gamma/beta
        # if a future config turns them on.
        self.norm_weight = self._stage_norm_param(getattr(norm, "weight", None))
        self.norm_bias = self._stage_norm_param(getattr(norm, "bias", None))

        self.kernel_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _stage_norm_param(self, param):
        if param is None:
            return None
        return ttnn.from_torch(
            param.detach().to(torch.float32).reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(self.device),
        )

    @classmethod
    def build(cls, device, torch_module):
        return cls(device, torch_module)

    def _to_device(self, tensor):
        """Accept an already-marshalled `ttnn.Tensor`, or stage a host one.

        The PCC harness and the pipeline both hand device tensors in; the
        `from_torch` branch only exists so the module is still callable with a
        plain torch tensor."""
        if isinstance(tensor, ttnn.Tensor):
            return tensor
        return ttnn.from_torch(
            tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=_replicate_mapper(self.device),
        )

    def __call__(self, x, conditioning_embedding=None, **kwargs):
        if conditioning_embedding is None:
            raise RuntimeError("ada_layer_norm_continuous requires `conditioning_embedding`")

        shape = list(x.shape)
        dim = int(shape[-1])
        seq = int(shape[-2])
        batch = int(shape[0]) if len(shape) >= 3 else 1
        x4 = x if len(shape) == 4 else ttnn.reshape(x, [batch, 1, seq, dim])

        cond = ttnn.reshape(self._to_device(conditioning_embedding), [batch, 1, 1, self.cond_dim])

        emb = ttnn.linear(
            ttnn.silu(cond),
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.kernel_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # `chunk(2, dim=1)` on the (batch, 2 * embedding_dim) projection is a
        # split of the feature axis: first half scale, second half shift.
        scale = ttnn.slice(emb, [0, 0, 0, 0], [batch, 1, 1, self.embedding_dim])
        shift = ttnn.slice(emb, [0, 0, 0, self.embedding_dim], [batch, 1, 1, 2 * self.embedding_dim])
        ttnn.deallocate(emb)

        normed = ttnn.layer_norm(
            x4,
            epsilon=self.eps,
            weight=self.norm_weight,
            bias=self.norm_bias,
            compute_kernel_config=self.kernel_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        scaled = ttnn.mul(normed, ttnn.add(scale, 1.0))
        out = ttnn.add(scaled, shift)
        ttnn.deallocate(normed)
        ttnn.deallocate(scaled)
        ttnn.deallocate(scale)
        ttnn.deallocate(shift)

        if len(shape) != 4:
            out = ttnn.reshape(out, shape)
        return out


def build(device, torch_module=None):
    return TtAdaLayerNormContinuous.build(device, torch_module)


def ada_layer_norm_continuous(device, torch_module=None):
    return TtAdaLayerNormContinuous.build(device, torch_module)
