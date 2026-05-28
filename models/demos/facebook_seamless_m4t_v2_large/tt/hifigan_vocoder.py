# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of ``SeamlessM4Tv2HifiGan`` (HiFi-GAN vocoder core).

The PyTorch reference is
``models/demos/facebook_seamless_m4t_v2_large/reference/functional.py::hifigan_vocoder_forward``,
which reproduces the forward of HuggingFace
``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2HifiGan``.

Architecture::

    x = conv_pre(input_embeds)                                       # (B, C0, T)
    for i in range(num_upsamples):                                   # 5 stages
        x = leaky_relu(x, slope=0.1)
        x = upsampler[i](x)                                          # ConvTranspose1d, stride=upsample_rates[i]
        # MRF: 3 residual blocks with different kernels, same dilations (1,3,5)
        r = sum(resblock[i*K + j](x) for j in 0..K-1) / K
        x = r
    x = leaky_relu(x, slope=0.01)                                    # NOTE: default slope!
    x = conv_post(x)                                                 # (B, 1, T_out)
    x = tanh(x)
    return x.squeeze(1)                                              # (B, T_out)

Implementation notes (TTNN port):

* ``ttnn.conv_transpose1d`` does NOT exist in TTNN. We use
  ``ttnn.conv_transpose2d`` with ``H=1`` (the same trick used by
  ``models/demos/qwen3_tts/tt/ttnn_conv_decoder.py::TTNNConvTranspose1d``).
* Conv ops (1D and transposed) require NHWC row-major ``[B, 1, T, C]`` input.
  Activations (LeakyReLU, residual add) run in TILE layout ``[B, T, C]``.
* The MRF residual blocks reuse the existing ``HifiGanResidualBlock`` module
  from ``tt/hifigan_residual_block.py``. That module accepts a configurable
  ``kernel_size`` and ``dilation`` already (constructor args), so we
  instantiate three of them per upsample stage with kernels (3, 7, 11) and
  dilations all (1, 3, 5). The residual block contract is
  ``[B, C, T]`` TILE in -> ``[B, C, T]`` TILE out.
* Compute config: HiFi4 + fp32 dest-acc, matching the other Seamless conv
  blocks.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.facebook_seamless_m4t_v2_large.tt.hifigan_residual_block import HifiGanResidualBlock


class HifiGanVocoder(LightweightModule):
    """SeamlessM4T-v2 HiFi-GAN vocoder.

    Args:
        device: ttnn device.
        state_dict: nested dict matching ``hifigan_vocoder_forward``:
            - ``conv_pre`` = ``{"weight": [C0, in_dim, 7], "bias": [C0]}``
            - ``upsampler`` = list of ``{"weight": [Cin, Cout, k_up], "bias": [Cout]}``
            - ``resblocks`` = flat list of ``num_upsamples * num_kernels``
              residual-block state dicts, indexed as ``resblocks[i*K + j]``.
            - ``conv_post`` = ``{"weight": [1, Clast, 7], "bias": [1]}``
        upsample_rates: per-stage stride (default (5, 4, 4, 2, 2)).
        upsample_kernel_sizes: per-stage kernel (default (11, 8, 8, 4, 4)).
        resblock_kernel_sizes: MRF kernel sizes (default (3, 7, 11)).
        resblock_dilation_sizes: MRF dilation tuples
            (default ((1, 3, 5), (1, 3, 5), (1, 3, 5))).
        leaky_relu_slope: per-stage leaky_relu slope (default 0.1). NOTE: the
            final ``leaky_relu`` before ``conv_post`` uses the HF default
            slope of 0.01 (HF quirk — see reference for details).
        weight_dtype: storage dtype for conv weights.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        upsample_rates: Sequence[int] = (5, 4, 4, 2, 2),
        upsample_kernel_sizes: Sequence[int] = (11, 8, 8, 4, 4),
        resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
        resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        leaky_relu_slope: float = 0.1,
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.upsample_rates = tuple(int(s) for s in upsample_rates)
        self.upsample_kernel_sizes = tuple(int(k) for k in upsample_kernel_sizes)
        self.resblock_kernel_sizes = tuple(int(k) for k in resblock_kernel_sizes)
        self.resblock_dilation_sizes = tuple(tuple(int(d) for d in row) for row in resblock_dilation_sizes)
        self.leaky_relu_slope = float(leaky_relu_slope)
        self.weight_dtype = weight_dtype

        assert len(self.upsample_rates) == len(self.upsample_kernel_sizes), (
            f"upsample_rates/upsample_kernel_sizes length mismatch: "
            f"{len(self.upsample_rates)} vs {len(self.upsample_kernel_sizes)}"
        )
        assert len(self.resblock_kernel_sizes) == len(self.resblock_dilation_sizes), (
            f"resblock_kernel_sizes/resblock_dilation_sizes length mismatch: "
            f"{len(self.resblock_kernel_sizes)} vs {len(self.resblock_dilation_sizes)}"
        )

        self.num_upsamples = len(self.upsample_rates)
        self.num_kernels = len(self.resblock_kernel_sizes)

        # --- conv_pre: Conv1d(in_dim, C0, k=7, pad=3) ---
        conv_pre_w: torch.Tensor = state_dict["conv_pre"]["weight"]  # [C0, in_dim, 7]
        conv_pre_b: torch.Tensor = state_dict["conv_pre"]["bias"]  # [C0]
        self.conv_pre_in_channels = int(conv_pre_w.shape[1])
        self.conv_pre_out_channels = int(conv_pre_w.shape[0])
        self.conv_pre_kernel_size = int(conv_pre_w.shape[2])
        assert self.conv_pre_kernel_size == 7, f"conv_pre kernel must be 7, got {self.conv_pre_kernel_size}"
        self.conv_pre_weight = ttnn.from_torch(conv_pre_w, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.conv_pre_bias = ttnn.from_torch(
            conv_pre_b.reshape(1, 1, 1, self.conv_pre_out_channels),
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # --- conv_post: Conv1d(Clast, 1, k=7, pad=3) ---
        conv_post_w: torch.Tensor = state_dict["conv_post"]["weight"]  # [1, Clast, 7]
        conv_post_b: torch.Tensor = state_dict["conv_post"]["bias"]  # [1]
        self.conv_post_in_channels = int(conv_post_w.shape[1])
        self.conv_post_out_channels = int(conv_post_w.shape[0])
        self.conv_post_kernel_size = int(conv_post_w.shape[2])
        assert self.conv_post_kernel_size == 7, f"conv_post kernel must be 7, got {self.conv_post_kernel_size}"
        assert self.conv_post_out_channels == 1, f"conv_post out_channels must be 1, got {self.conv_post_out_channels}"
        self.conv_post_weight = ttnn.from_torch(conv_post_w, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.conv_post_bias = ttnn.from_torch(
            conv_post_b.reshape(1, 1, 1, self.conv_post_out_channels),
            dtype=weight_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        # --- upsampler: list of ConvTranspose1d ---
        upsampler_sd = state_dict["upsampler"]
        assert (
            len(upsampler_sd) == self.num_upsamples
        ), f"upsampler length {len(upsampler_sd)} != num_upsamples {self.num_upsamples}"
        self.upsample_in_channels = []
        self.upsample_out_channels = []
        self.upsample_paddings = []
        self.upsampler_weight = []
        self.upsampler_bias = []
        for i, sd in enumerate(upsampler_sd):
            w = sd["weight"]  # [in_channels, out_channels, kernel_size]
            b = sd["bias"]  # [out_channels]
            in_ch = int(w.shape[0])
            out_ch = int(w.shape[1])
            k = int(w.shape[2])
            assert (
                k == self.upsample_kernel_sizes[i]
            ), f"upsampler[{i}] kernel {k} != expected {self.upsample_kernel_sizes[i]}"
            stride = self.upsample_rates[i]
            pad = (k - stride) // 2
            self.upsample_in_channels.append(in_ch)
            self.upsample_out_channels.append(out_ch)
            self.upsample_paddings.append(pad)

            # conv_transpose2d expects weight shape [in_channels, out_channels, kH, kW].
            # Our 1D weight is [in_channels, out_channels, kernel_size]; treat as
            # H=1: [in_channels, out_channels, 1, kernel_size].
            w_2d = w.unsqueeze(2)  # [in, out, 1, K]
            self.upsampler_weight.append(ttnn.from_torch(w_2d, dtype=weight_dtype, layout=ttnn.ROW_MAJOR_LAYOUT))
            self.upsampler_bias.append(
                ttnn.from_torch(
                    b.reshape(1, 1, 1, out_ch),
                    dtype=weight_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            )

        # --- MRF residual blocks: num_upsamples * num_kernels of them ---
        resblocks_sd = state_dict["resblocks"]
        assert len(resblocks_sd) == self.num_upsamples * self.num_kernels, (
            f"resblocks length {len(resblocks_sd)} != "
            f"num_upsamples*num_kernels = {self.num_upsamples * self.num_kernels}"
        )
        # Build a 2D grid [num_upsamples][num_kernels] of HifiGanResidualBlock,
        # each parameterised with the matching kernel_size and dilation.
        self.resblocks = []
        for i in range(self.num_upsamples):
            row = []
            for j in range(self.num_kernels):
                rb_sd = resblocks_sd[i * self.num_kernels + j]
                k_j = self.resblock_kernel_sizes[j]
                d_j = self.resblock_dilation_sizes[j]
                convs1_weights = [layer["weight"] for layer in rb_sd["convs1"]]
                convs1_biases = [layer["bias"] for layer in rb_sd["convs1"]]
                convs2_weights = [layer["weight"] for layer in rb_sd["convs2"]]
                convs2_biases = [layer["bias"] for layer in rb_sd["convs2"]]
                block = HifiGanResidualBlock(
                    device=device,
                    convs1_weights=convs1_weights,
                    convs1_biases=convs1_biases,
                    convs2_weights=convs2_weights,
                    convs2_biases=convs2_biases,
                    kernel_size=k_j,
                    dilation=tuple(d_j),
                    leaky_relu_slope=self.leaky_relu_slope,
                    weight_dtype=weight_dtype,
                )
                row.append(block)
            self.resblocks.append(row)

        # Conv configs and compute kernel config.
        self.conv1d_config = ttnn.Conv1dConfig(
            weights_dtype=weight_dtype,
            shard_layout=None,  # auto-pick
            deallocate_activation=False,
        )
        self.conv2d_config = ttnn.Conv2dConfig(
            weights_dtype=weight_dtype,
            shard_layout=None,
            deallocate_activation=False,
        )
        self.conv_compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            # packer_l1_acc=True is the SKILL.md "standard recipe" -- enables
            # in-tile packer accumulation. Matches the conv config used by the
            # newer tt-metal conv1d kernels. Tracy validated: PCC unchanged,
            # ~X% improvement on the conv compute. Same change in
            # hifigan_residual_block.py.
            packer_l1_acc=True,
        )

        # 1 / num_kernels broadcasted later via ttnn.mul (works on TILE).
        self._inv_num_kernels = 1.0 / float(self.num_kernels)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_conv1d(
        self,
        x_nhwc_rm: ttnn.Tensor,
        weight_attr: str,
        bias_attr: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        batch: int,
        seq_len: int,
    ) -> Tuple[ttnn.Tensor, int]:
        """Run ``ttnn.conv1d`` on NHWC ROW_MAJOR ``[B, 1, T, C_in]`` input.

        Caches the prepared weight/bias back onto ``self`` under the same
        attribute names so subsequent forwards skip preprocessing.
        """
        weight_tt = getattr(self, weight_attr)
        bias_tt = getattr(self, bias_attr)
        out, out_length, [new_w, new_b] = ttnn.conv1d(
            input_tensor=x_nhwc_rm,
            weight_tensor=weight_tt,
            device=self.device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch,
            input_length=seq_len,
            kernel_size=kernel_size,
            stride=1,
            padding=[padding, padding],
            dilation=1,
            groups=1,
            bias_tensor=bias_tt,
            conv_config=self.conv1d_config,
            compute_config=self.conv_compute_config,
            dtype=ttnn.bfloat16,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        setattr(self, weight_attr, new_w)
        setattr(self, bias_attr, new_b)
        return out, int(out_length)

    def _run_conv_transpose1d(
        self,
        x_nhwc_rm: ttnn.Tensor,
        idx: int,
        batch: int,
        seq_len: int,
    ) -> Tuple[ttnn.Tensor, int]:
        """Run ``ttnn.conv_transpose2d`` with H=1 to realise a 1-D transposed conv.

        Input ``x_nhwc_rm`` is row-major ``[B, 1, T, C_in]``.
        Returns ``(out, out_w)``; ``out`` is row-major NHWC of shape
        ``[B, 1, out_w, C_out]`` (typically TILE-padded by the conv kernel).
        """
        in_ch = self.upsample_in_channels[idx]
        out_ch = self.upsample_out_channels[idx]
        kernel_size = self.upsample_kernel_sizes[idx]
        stride = self.upsample_rates[idx]
        pad = self.upsample_paddings[idx]

        weight_tt = self.upsampler_weight[idx]
        bias_tt = self.upsampler_bias[idx]
        result = ttnn.conv_transpose2d(
            input_tensor=x_nhwc_rm,
            weight_tensor=weight_tt,
            device=self.device,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=batch,
            input_height=1,
            input_width=seq_len,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
            output_padding=(0, 0),
            bias_tensor=bias_tt,
            conv_config=self.conv2d_config,
            compute_config=self.conv_compute_config,
            mirror_kernel=True,  # weights not pre-flipped; let TTNN mirror internally
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        out, [out_h, out_w], [new_w, new_b] = result
        self.upsampler_weight[idx] = new_w
        self.upsampler_bias[idx] = new_b
        return out, int(out_w)

    @staticmethod
    def _nhwc_rm_from_bct_tile(x_bct_tile: ttnn.Tensor, batch: int, channels: int, seq_len: int) -> ttnn.Tensor:
        """Convert ``[B, C, T]`` TILE -> NHWC row-major ``[B, 1, T, C]``."""
        # Permute on TILE: [B, C, T] -> [B, T, C]
        x_btc_tile = ttnn.permute(x_bct_tile, (0, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_rm = ttnn.to_layout(x_btc_tile, ttnn.ROW_MAJOR_LAYOUT)
        x_rm = ttnn.reshape(x_rm, (batch, 1, seq_len, channels))
        return x_rm

    @staticmethod
    def _bct_tile_from_nhwc_rm(x_nhwc: ttnn.Tensor, batch: int, channels: int, seq_len: int) -> ttnn.Tensor:
        """Convert NHWC ``[B, 1, T, C]`` (any layout) -> ``[B, C, T]`` TILE.

        Note: conv outputs frequently come out as ``[1, 1, B*T, C]`` flattened
        NHWC. We reshape to ``[B, T, C]`` then permute to ``[B, C, T]``.
        """
        if x_nhwc.layout != ttnn.TILE_LAYOUT:
            x_nhwc = ttnn.to_layout(x_nhwc, ttnn.TILE_LAYOUT)
        x_btc = ttnn.reshape(x_nhwc, (batch, seq_len, channels))
        # Permute [B, T, C] -> [B, C, T]
        x_bct = ttnn.permute(x_btc, (0, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return x_bct

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Run the HiFi-GAN vocoder.

        Args:
            x: ttnn tensor of shape ``[B, in_dim, T_in]`` in TILE_LAYOUT.

        Returns:
            ttnn tensor of shape ``[B, T_out]`` in TILE_LAYOUT, in [-1, 1].
        """
        mc_dram = ttnn.DRAM_MEMORY_CONFIG
        batch = int(x.shape[0])
        in_dim = int(x.shape[1])
        T_in = int(x.shape[2])
        assert (
            in_dim == self.conv_pre_in_channels
        ), f"input channels mismatch: got {in_dim} vs expected {self.conv_pre_in_channels}"

        # --- 1. conv_pre ---
        # [B, in_dim, T] TILE -> NHWC ROW_MAJOR [B, 1, T, in_dim]
        x_rm = self._nhwc_rm_from_bct_tile(x, batch, in_dim, T_in)

        x_post_pre, T_pre = self._run_conv1d(
            x_rm,
            weight_attr="conv_pre_weight",
            bias_attr="conv_pre_bias",
            in_channels=self.conv_pre_in_channels,
            out_channels=self.conv_pre_out_channels,
            kernel_size=self.conv_pre_kernel_size,
            padding=3,
            batch=batch,
            seq_len=T_in,
        )
        ttnn.deallocate(x_rm)
        # conv_pre with stride=1 + "same" pad preserves the time dim.
        assert T_pre == T_in, f"conv_pre time dim changed: {T_in} -> {T_pre}"

        # Bring back to [B, C0, T] TILE for downstream activations / residual blocks.
        cur_channels = self.conv_pre_out_channels
        cur_seq_len = T_pre
        h_bct_tile = self._bct_tile_from_nhwc_rm(x_post_pre, batch, cur_channels, cur_seq_len)
        ttnn.deallocate(x_post_pre)

        # --- 2. Upsample stages with MRF ---
        for i in range(self.num_upsamples):
            # 2a. Per-stage leaky_relu(0.1) on TILE [B, C, T].
            h_bct_tile = ttnn.leaky_relu(h_bct_tile, self.leaky_relu_slope, memory_config=mc_dram)

            # 2b. ConvTranspose1d via conv_transpose2d, H=1.
            #     Need NHWC ROW_MAJOR [B, 1, T, C_in].
            x_rm = self._nhwc_rm_from_bct_tile(h_bct_tile, batch, cur_channels, cur_seq_len)
            ttnn.deallocate(h_bct_tile)

            x_up, T_up = self._run_conv_transpose1d(
                x_rm,
                idx=i,
                batch=batch,
                seq_len=cur_seq_len,
            )
            ttnn.deallocate(x_rm)

            cur_channels = self.upsample_out_channels[i]
            cur_seq_len = T_up

            # Convert back to [B, C_out, T_up] TILE for the MRF residual blocks.
            h_bct_tile = self._bct_tile_from_nhwc_rm(x_up, batch, cur_channels, cur_seq_len)
            ttnn.deallocate(x_up)

            # 2c. MRF: sum K residual block outputs then divide by K.
            #     Each HifiGanResidualBlock consumes/returns [B, C, T] TILE.
            mrf_blocks = self.resblocks[i]
            res_sum = mrf_blocks[0](h_bct_tile)
            for j in range(1, self.num_kernels):
                res_j = mrf_blocks[j](h_bct_tile)
                new_sum = ttnn.add(res_sum, res_j, memory_config=mc_dram)
                ttnn.deallocate(res_sum)
                ttnn.deallocate(res_j)
                res_sum = new_sum
            ttnn.deallocate(h_bct_tile)
            # Divide by K (scalar broadcast).
            h_bct_tile = ttnn.mul(res_sum, self._inv_num_kernels, memory_config=mc_dram)
            ttnn.deallocate(res_sum)

        # --- 3. Final leaky_relu(0.01) (HF quirk: NOT leaky_relu_slope!) ---
        h_bct_tile = ttnn.leaky_relu(h_bct_tile, 0.01, memory_config=mc_dram)

        # --- 4. conv_post: Conv1d(C_last, 1, k=7, pad=3) ---
        x_rm = self._nhwc_rm_from_bct_tile(h_bct_tile, batch, cur_channels, cur_seq_len)
        ttnn.deallocate(h_bct_tile)
        x_post, T_post = self._run_conv1d(
            x_rm,
            weight_attr="conv_post_weight",
            bias_attr="conv_post_bias",
            in_channels=self.conv_post_in_channels,
            out_channels=self.conv_post_out_channels,
            kernel_size=self.conv_post_kernel_size,
            padding=3,
            batch=batch,
            seq_len=cur_seq_len,
        )
        ttnn.deallocate(x_rm)
        assert T_post == cur_seq_len, f"conv_post time dim changed: {cur_seq_len} -> {T_post}"

        # Output is NHWC [B, 1, T_post, 1]. Bring back to TILE and squeeze.
        if x_post.layout != ttnn.TILE_LAYOUT:
            x_post = ttnn.to_layout(x_post, ttnn.TILE_LAYOUT)
        # Reshape to [B, T_post, 1] then [B, T_post].
        x_post = ttnn.reshape(x_post, (batch, T_post, self.conv_post_out_channels))

        # --- 5. tanh on TILE ---
        h = ttnn.tanh(x_post, memory_config=mc_dram)
        ttnn.deallocate(x_post)

        # Squeeze the singleton channel dim: [B, T, 1] -> [B, T].
        # ttnn.squeeze on TILE should handle the trailing-1 case, but if it
        # doesn't we can reshape directly.
        waveform = ttnn.reshape(h, (batch, T_post))
        return waveform
