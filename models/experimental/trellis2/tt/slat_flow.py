# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TRELLIS.2 structured-latent (SLat) flow DiT, shape or texture, on one device.

Mirrors `reference/slat_flow_host.functional_forward`. Same 30 blocks as the structure DiT; tokens are the active
voxels padded to a bucket length, and self-attention runs block-diagonal over [real | padding] through SDPA's
`cu_window_seqlens`, so padded rows never reach a real token (DESIGN D07).

Per-request tables (RoPE, cross mask, windows) are built once by `prepare` from the voxel coords; per-step adaLN
vectors are computed once per timestep and kept on device (`modulation`), so a call moves no data from the host.
"""

from __future__ import annotations

import torch

import ttnn
from models.experimental.trellis2.reference import slat_flow_host as slat
from models.experimental.trellis2.reference import structure_flow_host as host
from models.experimental.trellis2.tt.structure_flow import Block, DeviceTables, Precision, _hifi4, _norm_in, hi_lo_split
from models.tt_dit.layers.linear import Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import LayerNorm
from models.tt_dit.utils.tensor import bf16_tensor, float32_tensor


class TtSLatFlow(Module):
    def __init__(self, *, mesh_device: ttnn.MeshDevice, in_channels: int, precision: Precision = Precision()) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.precision = precision
        d = host.MODEL_CHANNELS
        io = _hifi4(mesh_device) if precision.io_hifi4 else None
        k_in = in_channels * (2 if precision.fp32_input else 1)
        self.input_layer = Linear(k_in, d, mesh_device=mesh_device, compute_kernel_config=io)
        self.blocks = ModuleList(Block(mesh_device=mesh_device, precision=precision) for _ in range(host.NUM_BLOCKS))
        self.final_norm = LayerNorm(
            d, norm_eps=host.FINAL_NORM_EPS, norm_elementwise_affine=False, mesh_device=mesh_device
        )
        self.out_layer = Linear(d, slat.OUT_CHANNELS, mesh_device=mesh_device, compute_kernel_config=io)
        self._host_state: dict[str, torch.Tensor] = {}
        self._mods: dict[float, list[list[ttnn.Tensor]]] = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        self._host_state = {k: v for k, v in state.items() if k.startswith(host.HOST_KEYS)}
        self._host_state.update({k: v for k, v in state.items() if k.endswith(".modulation")})
        for k in list(state):
            if k.startswith(host.HOST_KEYS):
                del state[k]
        if self.precision.fp32_input and "input_layer.weight" in state:
            state["input_layer.weight"] = torch.cat([state["input_layer.weight"]] * 2, dim=1)

    def prepare(self, coords: torch.Tensor, n_kv: int, n_pad: int | None = None) -> DeviceTables:
        """Per-request device tables from voxel coords [N, 4] and the conditioning length."""
        n = coords.shape[0]
        n_pad = n_pad or slat.bucket_len(n)
        cos, sin = slat.rope_tables(coords, n_pad)
        expand = lambda t: t.reshape(1, 1, *t.shape).expand(1, host.NUM_HEADS, *t.shape).contiguous()
        dev = self.mesh_device
        rope_tensor = float32_tensor if self.precision.fp32_qk else bf16_tensor
        return DeviceTables(
            cos=rope_tensor(expand(cos), device=dev),
            sin=rope_tensor(expand(sin), device=dev),
            rot=rope_tensor(host.pair_rotation(), device=dev),
            cross_mask=bf16_tensor(host.cross_mask(n_pad, n_kv), device=dev),
            self_windows=ttnn.from_torch(
                torch.tensor(slat.window_bounds(n, n_pad), dtype=torch.int32),
                device=dev,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=ttnn.uint32,
            ),
        )

    def modulation(self, t: torch.Tensor) -> list[list[ttnn.Tensor]]:
        """adaLN vectors for timestep t (x1000), computed on the host once per distinct t and kept on device."""
        key = float(t[0])
        if key not in self._mods:
            m = host.modulation(self._host_state, t)
            parts = (m.shift_msa, m.scale_msa, m.gate_msa, m.shift_mlp, m.scale_mlp, m.gate_mlp)
            to_device = float32_tensor if self.precision.fp32_modulate else bf16_tensor
            self._mods[key] = [
                [to_device(p[i].reshape(1, 1, 1, -1), device=self.mesh_device) for p in parts]
                for i in range(host.NUM_BLOCKS)
            ]
        return self._mods[key]

    def forward_tokens(
        self, x: ttnn.Tensor, t: torch.Tensor, context: ttnn.Tensor, tables: DeviceTables
    ) -> ttnn.Tensor:
        """Device in, device out: x [1, 1, n_pad, C_in] -> [1, 1, n_pad, 32]; padded rows are don't-care."""
        h = self.input_layer(x)
        if self.precision.fp32_residual:
            h = ttnn.typecast(h, ttnn.float32)
        for block, mod in zip(self.blocks, self.modulation(t)):
            h = block(h, mod, context, tables)
        out_dtype = ttnn.float32 if self.precision.fp32_output else None
        h = self.final_norm(h)
        return self.out_layer(h if self.precision.fp32_out_input else _norm_in(h), dtype=out_dtype)

    def forward(self, feats: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, tables: DeviceTables) -> torch.Tensor:
        """Torch in, torch out (for tests): feats [N, C_in], t [1], cond [1, L, 1024] -> [N, 32]."""
        n = feats.shape[0]
        n_pad = tables.cos.shape[2]
        dev = self.mesh_device
        tokens = slat.feats_to_tokens(feats.float(), n_pad)
        if self.precision.fp32_input:
            tokens = hi_lo_split(tokens)
        x = bf16_tensor(tokens, device=dev)
        context = bf16_tensor(host.pad_context(cond.float()), device=dev)
        out = self.forward_tokens(x, t, context, tables)
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()[0, 0, :n]
