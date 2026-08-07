# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""TTNN implementation of LFM2's ``ShortConv`` operator (the non-attention decoder mixer).

Reference (HF ``Lfm2ShortConv.slow_forward``)::

    BCx = in_proj(x).transpose(-1, -2)   # [B, 3H, S]
    B, C, x = BCx.chunk(3, dim=-2)       # each [B, H, S]
    Bx = B * x
    conv_out = depthwise_causal_conv1d(Bx, kernel_size=L_cache)[..., :seqlen]
    y = C * conv_out
    y = out_proj(y.transpose(-1, -2))

``in_proj``/``out_proj``/the depthwise conv all run on-device via ttnn. The causal
depthwise convolution (kernel size 3) is implemented as a small sum of per-tap
elementwise multiplies of shifted copies of ``Bx`` (pad + slice), which is both simple
and numerically exact for such a small kernel -- no im2col/``ttnn.conv1d`` needed.

Decode-mode state (the last ``L_cache - 1`` pre-conv ``Bx`` values per batch slot) is
kept as a plain host ``torch.Tensor`` and the single-token decode step itself is also
computed on host: with ``L_cache=3`` and batch <= 32 this is a handful of floats and is
far simpler/more robust than juggling a persistent, per-slot device tensor across
retraced decode graphs. See ``README.md`` for the known limitations of this approach
(no CUDA-graph-style ttnn trace support for decode, no multi-device sharding).
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode


class TtLfm2ShortConv(LightweightModule):
    def __init__(
        self,
        mesh_device,
        args,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        layer_num,
        dtype,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.hidden_size = args.dim
        self.L_cache = args.conv_L_cache
        self.has_bias = args.conv_bias
        self.max_batch_size = args.max_batch_size
        self.layer_num = layer_num

        # Prefix is typically "layers.N.conv" (no trailing dot); normalize once.
        prefix = state_dict_prefix if state_dict_prefix.endswith(".") else f"{state_dict_prefix}."
        torch_in_proj_w = state_dict[f"{prefix}in_proj.weight"]  # [3H, H]
        torch_out_proj_w = state_dict[f"{prefix}out_proj.weight"]  # [H, H]
        torch_conv_w = state_dict[f"{prefix}conv.weight"]  # [H, 1, K]
        assert torch_conv_w.shape[-1] == self.L_cache, (torch_conv_w.shape, self.L_cache)

        if args.dummy_weights or weight_cache_path is None:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / f"{prefix}{name}"

        self.in_proj = ttnn.as_tensor(
            torch_in_proj_w.transpose(-1, -2).contiguous(),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("in_proj.weight"),
        )
        # out_proj output dim is fractured across the mesh (column-parallel): the layer input is
        # full-width on every device (DistributedNorm all-gathers the sharded residual), so this
        # produces the width-sharded output the tt_transformers residual stream expects -- matching
        # attention/MLP outputs -- without any CCL.
        self.out_proj = ttnn.as_tensor(
            torch_out_proj_w.transpose(-1, -2).contiguous(),
            dtype=dtype,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name("out_proj.weight_fractured"),
        )

        self.in_proj_bias = None
        self.out_proj_bias = None
        self.conv_bias_t = None
        self._conv_bias_host = None
        if self.has_bias:
            in_bias_key = f"{prefix}in_proj.bias"
            out_bias_key = f"{prefix}out_proj.bias"
            conv_bias_key = f"{prefix}conv.bias"
            if in_bias_key in state_dict:
                self.in_proj_bias = ttnn.as_tensor(
                    state_dict[in_bias_key].reshape(1, -1),
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name("in_proj.bias"),
                )
            if out_bias_key in state_dict:
                # Sharded like out_proj's output dim (see above).
                self.out_proj_bias = ttnn.as_tensor(
                    state_dict[out_bias_key].reshape(1, -1),
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=args.cluster_shape),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name("out_proj.bias_fractured"),
                )
            if conv_bias_key in state_dict:
                conv_bias = state_dict[conv_bias_key]
                self._conv_bias_host = conv_bias.float()
                self.conv_bias_t = ttnn.as_tensor(
                    conv_bias.reshape(1, 1, 1, -1),
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_file_name=cache_name("conv.bias"),
                )

        # Per-tap depthwise conv weights, one [1, 1, 1, H] tensor per kernel position (broadcasts
        # over batch/seq in the elementwise multiplies below).
        self.conv_weights = [
            ttnn.as_tensor(
                torch_conv_w[:, 0, k].reshape(1, 1, 1, -1).contiguous(),
                dtype=dtype,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=cache_name(f"conv.weight.tap{k}"),
            )
            for k in range(self.L_cache)
        ]
        # Host copy of the same weights ([H, K]), used by the (host-computed) single-token decode step.
        self._conv_weight_host = torch_conv_w[:, 0, :].float()  # [H, K]

        # Host-resident per-slot decode state: last (L_cache - 1) pre-conv Bx rows per batch slot.
        self._conv_state_host = torch.zeros(self.max_batch_size, max(self.L_cache - 1, 0), self.hidden_size)

    def reset_conv_state(self, batch_indices=None):
        if batch_indices is None:
            self._conv_state_host.zero_()
        else:
            self._conv_state_host[batch_indices] = 0

    def forward(self, x: ttnn.Tensor, mode: Mode, user_id: int = 0) -> ttnn.Tensor:
        """
        Args:
            x: prefill -> [1, 1, S, H] (or [B, 1, S/B, H] for batched prefill);
               decode  -> [B, 1, 1, H] with B == max_batch_size.
            mode: Mode.PREFILL or Mode.DECODE.
            user_id: batch slot whose decode state to seed after a prefill pass (prefill only).
        """
        BCx = ttnn.linear(x, self.in_proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.in_proj_bias is not None:
            BCx = ttnn.add(BCx, self.in_proj_bias)

        H = self.hidden_size
        B_gate = BCx[:, :, :, 0:H]
        C_gate = BCx[:, :, :, H : 2 * H]
        x_gate = BCx[:, :, :, 2 * H : 3 * H]
        Bx = ttnn.multiply(B_gate, x_gate)
        ttnn.deallocate(BCx)

        if mode == Mode.PREFILL:
            conv_out = self._conv_prefill(Bx)
            self._update_state_from_prefill(Bx, user_id)
        else:
            conv_out = self._conv_decode(Bx)

        y = ttnn.multiply(C_gate, conv_out)
        ttnn.deallocate(conv_out)
        y = ttnn.linear(y, self.out_proj, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.out_proj_bias is not None:
            y = ttnn.add(y, self.out_proj_bias)
        return y

    def _conv_prefill(self, Bx: ttnn.Tensor) -> ttnn.Tensor:
        K = self.L_cache
        seq_len = Bx.shape[-2]
        Bx_rm = ttnn.to_layout(Bx, ttnn.ROW_MAJOR_LAYOUT)

        conv_out = None
        for k in range(K):
            shift = K - 1 - k  # tap k=K-1 -> current token (no shift); tap k=0 -> oldest (max shift)
            if shift == 0:
                shifted = Bx_rm
            else:
                padded = ttnn.pad(Bx_rm, padding=((0, 0), (0, 0), (shift, 0), (0, 0)), value=0.0)
                shifted = padded[:, :, :seq_len, :]
            term = ttnn.multiply(ttnn.to_layout(shifted, ttnn.TILE_LAYOUT), self.conv_weights[k])
            conv_out = term if conv_out is None else ttnn.add(conv_out, term)

        if self.conv_bias_t is not None:
            conv_out = ttnn.add(conv_out, self.conv_bias_t)
        return conv_out

    @staticmethod
    def _replicated_to_torch(t: ttnn.Tensor) -> torch.Tensor:
        """Bring a mesh-replicated tensor to host. All shards are identical (inputs and weights are
        replicated across the mesh), so reading shard 0 avoids needing a mesh composer."""
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])

    def _update_state_from_prefill(self, Bx: ttnn.Tensor, user_id: int) -> None:
        K = self.L_cache
        if K <= 1:
            return
        Bx_host = self._replicated_to_torch(Bx).float()
        # Collapse any leading batch/head dims: keep only the trailing [S, H] and treat the whole
        # prefill call as belonging to a single logical sequence (batch=1 prefill, the common case
        # for this bring-up demo/tests).
        Bx_host = Bx_host.reshape(-1, Bx_host.shape[-2], Bx_host.shape[-1])[-1]  # [S, H]
        seq_len = Bx_host.shape[0]
        if seq_len >= K - 1:
            tail = Bx_host[seq_len - (K - 1) :, :]
        else:
            pad = torch.zeros(K - 1 - seq_len, Bx_host.shape[-1])
            tail = torch.cat([pad, Bx_host], dim=0)
        self._conv_state_host[user_id % self.max_batch_size] = tail

    def _conv_decode(self, Bx: ttnn.Tensor) -> ttnn.Tensor:
        K = self.L_cache
        Bx_host = self._replicated_to_torch(Bx).float()
        # tt_transformers decode packs users on the rows dim padded to a tile ([1, 1, 32, H]);
        # the unit test feeds [B, 1, 1, H]. Flatten either layout to rows [num_rows, H]: the
        # first `max_batch_size` rows are the real batch slots, the rest is tile padding.
        orig_shape = tuple(Bx_host.shape)
        rows = Bx_host.reshape(-1, self.hidden_size)
        B = min(self.max_batch_size, rows.shape[0])
        Bx_rows = rows[:B]  # [B, H]

        if K > 1:
            state = self._conv_state_host[:B]  # [B, K-1, H]
            full = torch.cat([state, Bx_rows.unsqueeze(1)], dim=1)  # [B, K, H]
            self._conv_state_host[:B] = full[:, 1:, :]
        else:
            full = Bx_rows.unsqueeze(1)  # [B, 1, H]

        out = torch.zeros(B, self.hidden_size)
        for k in range(K):
            out = out + full[:, k, :] * self._conv_weight_host[:, k]
        if self._conv_bias_host is not None:
            out = out + self._conv_bias_host

        out_rows = torch.zeros_like(rows)
        out_rows[:B] = out
        return ttnn.from_torch(
            out_rows.reshape(orig_shape),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
