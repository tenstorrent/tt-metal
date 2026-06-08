# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""S2V auxiliary blocks (mirrors ``wan/modules/s2v/auxi_blocks.py``):
CausalConv1d + the 3-stage MotionEncoder_tc built on top of it."""

from __future__ import annotations

import torch

import ttnn

from .....layers.linear import Linear
from .....layers.module import Module, Parameter
from .....parallel.manager import CCLManager
from .....utils.conv3d import get_conv3d_config
from .....utils.tensor import local_device_to_torch


class CausalConv1d(Module):
    """Conv1d with causal (left-only) temporal padding via ``ttnn.experimental.conv3d``."""

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        tp_mesh_axis: int | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.mesh_device = mesh_device
        self.dtype = dtype
        # TP-shard ``chan_out`` (ColParallel-style); all-gather at the end re-replicates.
        self.tp_mesh_axis = tp_mesh_axis
        self.ccl_manager = ccl_manager
        if tp_mesh_axis is not None:
            tp_factor = self.mesh_device.shape[tp_mesh_axis]
            assert ccl_manager is not None, "TP-sharded CausalConv1d requires ccl_manager for all-gather"
            assert chan_out % tp_factor == 0, f"chan_out={chan_out} must be divisible by tp_factor={tp_factor}"
            weight_mesh_axes = [tp_mesh_axis, None, None, None, None]
            bias_mesh_axes = [None, tp_mesh_axis]
            on_host = False
        else:
            weight_mesh_axes = None
            bias_mesh_axes = None
            on_host = True

        self.conv_config = get_conv3d_config(
            chan_in,
            chan_out // (self.mesh_device.shape[tp_mesh_axis] if tp_mesh_axis is not None else 1),
            (kernel_size, 1, 1),
            dtype,
            grid_size=mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Replicated path: rank-5 on host, conv3d prepares internally. TP-sharded
        # path: pre-prepare to ``[d, chan_out]`` so ``chan_out`` shards cleanly.
        if tp_mesh_axis is not None:
            in_aligned = ((chan_in + 31) // 32) * 32
            d = kernel_size * 1 * 1 * in_aligned
            self.weight = Parameter(
                total_shape=[d, chan_out],
                mesh_axes=[None, tp_mesh_axis],
                device=mesh_device,
                pad_value=0,
                dtype=dtype,
            )
        else:
            self.weight = Parameter(
                total_shape=[chan_out, chan_in, kernel_size, 1, 1],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                pad_value=0,
                dtype=dtype,
                on_host=True,
            )
        self.bias = Parameter(
            total_shape=[1, chan_out],
            mesh_axes=bias_mesh_axes,
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "conv.weight" in state:
            w = state.pop("conv.weight")  # [out, in, k]
            assert w.shape == (self.chan_out, self.chan_in, self.kernel_size), w.shape
            w_5d = w.unsqueeze(-1).unsqueeze(-1).contiguous()  # [out, in, k, 1, 1]
            if self.tp_mesh_axis is not None:
                # Pre-prepare so the Parameter can shard chan_out cleanly.
                weight_tt = ttnn.from_torch(w_5d, dtype=self.dtype, pad_value=0)
                prepared = ttnn.experimental.prepare_conv3d_weights(
                    weight_tensor=weight_tt,
                    C_in_block=self.conv_config.C_in_block,
                    device=self.mesh_device,
                )
                state["weight"] = local_device_to_torch(prepared)  # [d, chan_out]
            else:
                state["weight"] = w_5d
        if "conv.bias" in state:
            state["bias"] = state.pop("conv.bias").reshape(1, -1)

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        local_out_channels = self.chan_out
        if self.tp_mesh_axis is not None:
            tp_factor = self.mesh_device.shape[self.tp_mesh_axis]
            local_out_channels = self.chan_out // tp_factor
        out = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=local_out_channels,
            kernel_size=(self.kernel_size, 1, 1),
            stride=(self.stride, 1, 1),
            padding=(0, 0, 0),
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        if self.tp_mesh_axis is not None:
            out = self.ccl_manager.all_gather_persistent_buffer(out, dim=-1, mesh_axis=self.tp_mesh_axis)
        return out


class MotionEncoder_tc(Module):
    """3-stage causal-conv encoder; ``[B, T, in_dim]`` → ``[B, T//4, num_heads + 1, hidden_dim]``."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        num_heads: int = 4,
        need_global: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        tp_mesh_axis: int | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        assert hidden_dim % 4 == 0, "MotionEncoder hidden_dim must be divisible by 4"
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.need_global = need_global
        self.mesh_device = mesh_device
        self.dtype = dtype

        # All three causal convs ColParallel-shard their out_channels across
        # the TP mesh axis. Eliminates the 32× redundant compute that the
        # previously-replicated weights produced.
        conv_tp = dict(tp_mesh_axis=tp_mesh_axis, ccl_manager=ccl_manager)
        self.conv1_local = CausalConv1d(
            in_dim,
            hidden_dim // 4 * num_heads,
            kernel_size=3,
            stride=1,
            mesh_device=mesh_device,
            dtype=dtype,
            **conv_tp,
        )
        if need_global:
            self.conv1_global = CausalConv1d(
                in_dim,
                hidden_dim // 4,
                kernel_size=3,
                stride=1,
                mesh_device=mesh_device,
                dtype=dtype,
                **conv_tp,
            )
            # Final dense projection used only on the global branch.
            self.final_linear = Linear(hidden_dim, hidden_dim, bias=True, mesh_device=mesh_device)
        self.conv2 = CausalConv1d(
            hidden_dim // 4,
            hidden_dim // 2,
            kernel_size=3,
            stride=2,
            mesh_device=mesh_device,
            dtype=dtype,
            **conv_tp,
        )
        self.conv3 = CausalConv1d(
            hidden_dim // 2,
            hidden_dim,
            kernel_size=3,
            stride=2,
            mesh_device=mesh_device,
            dtype=dtype,
            **conv_tp,
        )

        self.padding_tokens = Parameter(
            total_shape=[1, 1, 1, hidden_dim],
            device=mesh_device,
            dtype=dtype,
        )

        # No-affine LayerNorm runs once between each conv stage; cumulative
        # error over the 3 stages compounds, so use HiFi4 + fp32_dest_acc to
        # keep PCC > 0.99 against the reference at ~5120 inner dim.
        self.layer_norm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _causal_pad_host(self, x_torch: torch.Tensor, kernel_size: int) -> torch.Tensor:
        """Left-replicate-pad temporal dim by ``kernel_size - 1`` frames on host."""
        pad = kernel_size - 1
        if pad == 0:
            return x_torch
        first = x_torch[:, :1, :].expand(-1, pad, -1)
        return torch.cat([first, x_torch], dim=1)

    def _causal_pad_device(self, x_BTHWC: ttnn.Tensor, kernel_size: int) -> ttnn.Tensor:
        """On-device equivalent of ``_causal_pad_host`` for 5D ``[B, T, 1, 1, C]``."""
        pad = kernel_size - 1
        if pad == 0:
            return x_BTHWC
        B = int(x_BTHWC.shape[0])
        C = int(x_BTHWC.shape[-1])
        first = ttnn.slice(x_BTHWC, [0, 0, 0, 0, 0], [B, 1, 1, 1, C])
        return ttnn.concat([first] * pad + [x_BTHWC], dim=1)

    def _conv_stage_BTC(
        self,
        x_tile_BTC: ttnn.Tensor,
        conv: CausalConv1d,
        B_eff: int,
        in_chan: int,
        out_chan: int,
    ) -> ttnn.Tensor:
        """``[B_eff, T, in_chan]`` TILE → on-device causal-pad → conv →
        no-affine LayerNorm → SiLU → ``[B_eff, T_post, out_chan]`` TILE.
        """
        T = int(x_tile_BTC.shape[1])
        x_rm = ttnn.to_layout(x_tile_BTC, ttnn.ROW_MAJOR_LAYOUT)
        x_5d = ttnn.reshape(x_rm, [B_eff, T, 1, 1, in_chan])
        x_padded = self._causal_pad_device(x_5d, kernel_size=3)
        x = conv(x_padded)
        x = ttnn.layer_norm(
            x,
            weight=None,
            bias=None,
            epsilon=1e-6,
            compute_kernel_config=self.layer_norm_compute_kernel_config,
        )
        x = ttnn.silu(x)
        t_post = int(x.shape[1])
        return ttnn.reshape(x, [B_eff, t_post, out_chan])

    def forward(self, x_torch: torch.Tensor) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        """Run the 3-stage encoder on ``x_torch`` ``[B, T, in_dim]`` (CPU).

        Returns ``local_dev`` alone, or ``(global_dev, local_dev)`` when ``need_global=True``.
        """
        B, T, _ = x_torch.shape
        H4 = self.hidden_dim // 4
        H2 = self.hidden_dim // 2
        H = self.hidden_dim

        x_torch_p = self._causal_pad_host(x_torch, kernel_size=3)
        T_in_padded = x_torch_p.shape[1]
        x_5d = x_torch_p.reshape(B, T_in_padded, 1, 1, self.in_dim).contiguous()
        x_dev_in = ttnn.from_torch(x_5d, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        # Local branch: conv1_local → split heads → LN → conv2 → conv3.
        x = self.conv1_local(x_dev_in)
        x_BT_nH_H4 = ttnn.reshape(x, [B, T, self.num_heads, H4])
        x_B_nH_T_H4 = ttnn.permute(x_BT_nH_H4, [0, 2, 1, 3])
        B_local = B * self.num_heads
        x_BTC = ttnn.reshape(x_B_nH_T_H4, [B_local, T, H4])
        x_BTC = ttnn.to_layout(x_BTC, ttnn.TILE_LAYOUT)
        x_dev = ttnn.layer_norm(
            x_BTC,
            weight=None,
            bias=None,
            epsilon=1e-6,
            compute_kernel_config=self.layer_norm_compute_kernel_config,
        )
        x_dev = ttnn.silu(x_dev)
        x_dev = self._conv_stage_BTC(x_dev, self.conv2, B_local, H4, H2)
        x_dev = self._conv_stage_BTC(x_dev, self.conv3, B_local, H2, H)

        # Unmerge heads and append the learned padding token as an extra
        # head/token slot along dim 2.
        T4 = int(x_dev.shape[1])
        x_unmerge = ttnn.reshape(x_dev, [B, self.num_heads, T4, H])
        x_unmerge = ttnn.permute(x_unmerge, [0, 2, 1, 3])
        pad_tokens_BTH = ttnn.expand(self.padding_tokens.data, [B, T4, 1, H])
        local_dev = ttnn.concat([x_unmerge, pad_tokens_BTH], dim=2)

        if not self.need_global:
            return local_dev

        # Global branch: reuses the same upload; no head-split.
        x = self.conv1_global(x_dev_in)
        x = ttnn.reshape(x, [B, T, H4])
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x_dev = ttnn.layer_norm(
            x,
            weight=None,
            bias=None,
            epsilon=1e-6,
            compute_kernel_config=self.layer_norm_compute_kernel_config,
        )
        x_dev = ttnn.silu(x_dev)
        x_dev = self._conv_stage_BTC(x_dev, self.conv2, B, H4, H2)
        x_dev = self._conv_stage_BTC(x_dev, self.conv3, B, H2, H)
        x_dev = self.final_linear(x_dev)
        # Unsqueeze the singleton head axis to match ref ``rearrange('(b n) t c -> b t n c', b=b)`` with n=1.
        return ttnn.unsqueeze(x_dev, 2), local_dev
