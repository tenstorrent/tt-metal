# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `final_layer_f_p32`
(meituan-longcat/LongCat-Video's `dit.final_layer`, class `FinalLayer_FP32`
in the vendored `longcat_video/modules/blocks.py`):

    norm_final = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
    linear = Linear(hidden_size, num_patch * out_channels, bias=True)
    adaLN_modulation = Sequential(SiLU(), Linear(adaln_tembed_dim, 2 * hidden_size, bias=True))

    forward(x, t, latent_shape):        # x: [B, N, hidden_size], t: [B, T, adaln_tembed_dim]
        T, _, _ = latent_shape
        shift, scale = adaLN_modulation(t).unsqueeze(2).chunk(2, dim=-1)   # [B, T, 1, hidden_size]
        x = (norm_final(x.view(B, T, -1, hidden_size)) * (scale + 1) + shift).view(B, N, hidden_size)
        return linear(x)

TP scheme: `adaLN_modulation`'s Linear feeds an elementwise modulation of the
FULL (replicated) hidden state, so its weight stays REPLICATED (splitting it
would require an all_gather before the modulation anyway, for no benefit --
its output width, `2 * hidden_size`, is comparatively small). `linear` is the
network's terminal projection with no on-chip consumer, so it graduates
DIRECTLY tensor-parallel column-wise: shard its OUTPUT features across the
mesh, then all_gather to reassemble the full, replicated result the golden
expects.

Per-frame (T > 1) modulation is a real feature of this checkpoint, but this
bring-up's synthetic PCC input always uses B=1, T=1 (see
`tests/pcc/test_final_layer_f_p32.py::_make_arg_for`) -- one global
(shift, scale) pair applied uniformly across every token. That degenerates
exactly to a standard affine LayerNorm, so the whole norm+modulate step is
one fused `ttnn.layer_norm(x, weight=scale + 1, bias=shift)` call -- no
separate broadcast/multiply/add, and no host round-trip.
"""

from __future__ import annotations

import torch

import ttnn


class TtFinalLayerFP32:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        self.dtype = ttnn.bfloat16
        self.eps = torch_module.norm_final.eps

        state = torch_module.state_dict()
        hidden_size = torch_module.hidden_size

        adaln_w = state["adaLN_modulation.1.weight"]  # [2*hidden_size, adaln_tembed_dim]
        adaln_b = state["adaLN_modulation.1.bias"]  # [2*hidden_size]
        # `chunk(2, dim=-1)` on the Linear's OUTPUT is equivalent to splitting
        # its weight/bias rows in half -- avoids any on-device chunk/split.
        shift_w, scale_w = adaln_w[:hidden_size], adaln_w[hidden_size:]
        shift_b, scale_b = adaln_b[:hidden_size], adaln_b[hidden_size:]

        def _replicated(t):
            return ttnn.from_torch(
                t,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        self.shift_w = _replicated(shift_w.transpose(0, 1).contiguous())
        self.shift_b = _replicated(shift_b.reshape(1, -1))
        self.scale_w = _replicated(scale_w.transpose(0, 1).contiguous())
        self.scale_b = _replicated(scale_b.reshape(1, -1))

        # Terminal projection: column-parallel (split OUTPUT features).
        linear_w = state["linear.weight"].transpose(0, 1).contiguous()
        linear_b = state["linear.bias"].reshape(1, -1)
        self.linear_w = ttnn.from_torch(
            linear_w,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )
        self.linear_b = ttnn.from_torch(
            linear_b,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        )

        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x: ttnn.Tensor, t: torch.Tensor, latent_shape) -> ttnn.Tensor:
        if t.shape[1] > 1:
            # The fused ttnn.layer_norm affine path applies one (shift, scale) pair uniformly
            # across the whole sequence -- exact (not an approximation) when every frame's
            # timestep embedding is identical, which is what run_t2v's uniform-noise-level T2V
            # generation always produces (see long_cat_video_transformer3_d_model.py's
            # _timestep_embed, which broadcasts one scalar timestep to every frame). A caller
            # with genuinely different per-frame timesteps (e.g. rolling/progressive denoising)
            # needs a real per-frame-modulated implementation, not this fused one -- fail loud
            # rather than silently apply only frame 0's modulation to every frame.
            assert torch.allclose(t, t[:, :1, :].expand_as(t)), (
                "per-frame (T>1) timestep modulation with genuinely DIFFERENT values per frame "
                "is not supported by this fused ttnn.layer_norm affine path."
            )
        t = t[:, :1, :]  # representative frame (identical to every other frame; see above)
        t_tt = ttnn.from_torch(
            t.to(torch.float32).reshape(1, -1),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        silu_t = ttnn.silu(t_tt)
        shift = ttnn.linear(silu_t, self.shift_w, bias=self.shift_b, compute_kernel_config=self.ckc)
        scale = ttnn.linear(silu_t, self.scale_w, bias=self.scale_b, compute_kernel_config=self.ckc)

        x_mod = ttnn.layer_norm(x, epsilon=self.eps, weight=scale + 1.0, bias=shift)

        out = ttnn.linear(x_mod, self.linear_w, bias=self.linear_b, compute_kernel_config=self.ckc)
        return ttnn.all_gather(out, dim=-1, topology=ttnn.Topology.Linear)


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtFinalLayerFP32:
    return TtFinalLayerFP32(mesh_device, torch_module)
