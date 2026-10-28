# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioSubsample (HF `Gemma4AudioSubSampleConvProjection`, modeling_gemma4.py:363-418).

    x0 = mel * frame_mask                                          [B, T, 128 mel, 1]   (NHWC)
    h0 = relu( LayerNorm_128( conv0(x0) ) )                         conv0 1 -> 128, k3 s2 p1, no bias
    x1 = h0 * frame_mask[::2]                                      [B, T/2, 64, 128]
    h1 = relu( LayerNorm_32( conv1(x1) ) )                          conv1 128 -> 32, k3 s2 p1, no bias
    out = flatten(h1) @ W_in                                        [B, T/4, 32*32 = 1024] -> 1024, no bias

LayerNorms are over channels, weight only (`bias=False`), eps = rms_norm_eps. `flatten` follows HF
`permute(0, 2, 3, 1).reshape(B, t, f*c)`: index f*32 + c, which is the NHWC conv output's natural order.

Both convs run on device (`ttnn.conv2d`, fp32 mel in for conv0, bf16 weights, HiFi4 + fp32 accumulation).
`ttnn.conv2d` lays out its weights for one input geometry, so the device copies are prepared per
(batch, frame bucket) by `prepare(batch, frames)` at setup (first conv2d call with host weights,
`return_weights_and_bias=True`); the forward only uses prepared weights and raises for an
unprepared shape.

Contract (forward)
    mel         [B, 1, T, 128] fp32 TILE DRAM, T = 4 * S (frame bucket); frames past the clip are 0
    frame_mask  [B, 1, T, 1] fp32 TILE DRAM (1 valid frame / 0 padding), multiplied into mel on device
    mask1       [1, 1, B*(T/2)*64, 1] bf16 TILE DRAM: frame_mask[::2] repeated over the 64 frequency rows
    output      [B, 1, S, 1024] act_dtype (bf16 default) TILE DRAM (rows >= the clip's token count are
                not meaningful)
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.audio.config import SUBSAMPLE, AudioConfig, audio_act_memory_config
from models.demos.blackhole.embedding_gemma2.tt.config import compute_kernel_config_hifi4, compute_kernel_config_norm
from models.demos.blackhole.embedding_gemma2.tt.weights import as_weight, linear_weight

CONV0_L1_BYTES_PER_BANK = 256 * 1024


def _conv_config():
    return ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, output_layout=ttnn.TILE_LAYOUT, config_tensors_in_dram=True)


class AudioSubsample(LightweightModule):
    def __init__(
        self,
        config: AudioConfig,
        device,
        conv_w_host: tuple[ttnn.Tensor, ttnn.Tensor],
        norm_w: tuple[ttnn.Tensor, ttnn.Tensor],
        w_in: ttnn.Tensor,
        act_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.config = config
        self.device = device
        self.conv_w_host = conv_w_host
        self.norm_w = norm_w
        self.w_in = w_in
        self.act_dtype = act_dtype
        self.compute_kernel_config = compute_kernel_config_hifi4()
        self.norm_kernel_config = compute_kernel_config_norm()
        self._conv_w: dict[tuple[int, int], tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AudioConfig,
        device,
        *,
        cache_dir: Path | None = None,
        name: str = "audio.subsample",
        act_dtype=ttnn.bfloat16,
    ) -> "AudioSubsample":
        """`state_dict` holds the HF `subsample_conv_projection.` sub-state. `act_dtype`: output dtype
        of input_proj (the conformer residual stream), see AudioModel."""
        c0, c1 = config.subsampling_channels
        shapes = {
            "layer0.conv.weight": (c0, 1, 3, 3),
            "layer1.conv.weight": (c1, c0, 3, 3),
            "layer0.norm.weight": (c0,),
            "layer1.norm.weight": (c1,),
            "input_proj_linear.weight": (config.hidden_size, config.proj_input_dim),
        }
        for k, shp in shapes.items():
            if tuple(state_dict[k].shape) != shp:
                raise ValueError(f"{name}.{k}: expected {shp}, got {tuple(state_dict[k].shape)}")
        # conv weights stay on the host (OIHW fp32 ttnn tensors); prepare() makes the device layouts
        conv_w = tuple(ttnn.from_torch(state_dict[f"layer{i}.conv.weight"].float(), dtype=ttnn.float32) for i in (0, 1))
        norm_w = tuple(
            as_weight(
                state_dict[f"layer{i}.norm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE),
                device,
                cache_dir=cache_dir,
                name=f"{name}.layer{i}.norm.weight",
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for i in (0, 1)
        )
        w_in = as_weight(
            linear_weight(state_dict["input_proj_linear.weight"]),
            device,
            cache_dir=cache_dir,
            name=f"{name}.input_proj",
        )
        return cls(config, device, conv_w, norm_w, w_in, act_dtype)

    # ------------------------------------------------------------------ conv plumbing
    def _geometry(self, i: int, batch: int, frames: int) -> dict:
        c0, c1 = self.config.subsampling_channels
        mel = self.config.num_mel_bins
        if i == 0:
            return dict(in_channels=1, out_channels=c0, batch_size=batch, input_height=frames, input_width=mel)
        return dict(in_channels=c0, out_channels=c1, batch_size=batch, input_height=frames // 2, input_width=mel // 2)

    def _conv(self, i: int, x: ttnn.Tensor, weight: ttnn.Tensor, batch: int, frames: int, prepare: bool = False):
        return ttnn.conv2d(
            input_tensor=x,
            weight_tensor=weight,
            device=self.device,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            conv_config=_conv_config(),
            compute_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            return_weights_and_bias=prepare,
            **self._geometry(i, batch, frames),
        )

    def prepared(self, batch: int, frames: int) -> bool:
        return (batch, frames) in self._conv_w

    def prepare(self, x: "AudioSubsampleInputs") -> None:
        """Lay out both conv weights for (batch, frames) by running the stack once (setup only)."""
        B, T = x.mel.shape[0], x.mel.shape[2]
        if (B, T) in self._conv_w:
            return
        ws = []
        h = self._mask_mel(x)
        for i in (0, 1):
            out, (w_dev, _) = self._conv(i, h, self.conv_w_host[i], B, T, prepare=True)
            ws.append(w_dev)
            ttnn.deallocate(h)
            h = self._post_conv(i, out, x, B, T)
        ttnn.deallocate(h)
        self._conv_w[(B, T)] = tuple(ws)

    def _mask_mel(self, x: "AudioSubsampleInputs") -> ttnn.Tensor:
        """mel [B,1,T,128] TILE * frame_mask -> [B,T,128,1] RM (the conv0 NHWC input, C = 1).

        E5c: the host uploads mel in tile layout (was [B,T,128,1] row-major, which cost a 0.75 ms
        row-major reshape at 1 x 768 before the multiply)."""
        B, _, T, F = x.mel.shape
        mc = audio_act_memory_config(x.mel.device(), x.mel.shape, ttnn.float32)  # E5c: L1 when it fits
        masked = ttnn.multiply(x.mel, x.frame_mask, memory_config=mc)
        rm = ttnn.to_layout(masked, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        ttnn.deallocate(masked)
        return ttnn.reshape(rm, (B, T, F, 1), memory_config=mc)

    def _post_conv(self, i: int, out: ttnn.Tensor, x: "AudioSubsampleInputs", B: int, T: int) -> ttnn.Tensor:
        """LayerNorm over channels + ReLU (+ the layer-1 input mask after conv0)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        # conv0's [B*T/2*64, 128] output path (S2I, LayerNorm, ReLU * mask -> conv1 input) in L1 up to
        # CONV0_L1_BYTES_PER_BANK (193 KB per bank at 1 x 768). conv1's output stays in DRAM: an L1
        # input_proj input makes the default ttnn.linear pick another config (E5c c20, not bit-identical).
        mc = dram
        if i == 0:
            mc = audio_act_memory_config(self.device, out.shape, bytes_per_bank=CONV0_L1_BYTES_PER_BANK)
        if out.is_sharded():  # an L1 conv input gives a height-sharded output; layer_norm needs interleaved
            o = ttnn.sharded_to_interleaved(out, mc)
            ttnn.deallocate(out)
            out = o
        ln = ttnn.layer_norm(
            out,
            epsilon=self.config.rms_norm_eps,
            weight=self.norm_w[i],
            memory_config=mc,
            compute_kernel_config=self.norm_kernel_config,
        )
        ttnn.deallocate(out)
        if i == 0:
            # relu as the SFPU pre-activation of the layer-1 input mask multiply
            relu = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
            m = ttnn.multiply(ln, x.mask1, input_tensor_a_activations=[relu], memory_config=mc)
            ttnn.deallocate(ln)
            return m
        r = ttnn.relu(ln, memory_config=dram)
        ttnn.deallocate(ln)
        return r

    # ------------------------------------------------------------------ forward
    def conv_stack(self, x: "AudioSubsampleInputs") -> ttnn.Tensor:
        """Masked conv0 / LN / ReLU / mask / conv1 / LN / ReLU, flattened: [B, 1, S, 1024] bf16 (HF
        input_proj_linear input)."""
        B, T = x.mel.shape[0], x.mel.shape[2]
        if (B, T) not in self._conv_w:
            raise RuntimeError(f"conv weights not prepared for (batch, frames) {(B, T)}; call prepare() at setup")
        ws = self._conv_w[(B, T)]
        h = self._mask_mel(x)
        for i in (0, 1):
            out = self._conv(i, h, ws[i], B, T)
            ttnn.deallocate(h)
            h = self._post_conv(i, out, x, B, T)
        return ttnn.reshape(h, (B, 1, T // SUBSAMPLE, self.config.proj_input_dim))

    def input_proj(self, flat: ttnn.Tensor) -> ttnn.Tensor:
        """[B, 1, S, 1024] -> [B, 1, S, 1024] act_dtype, `input_proj_linear` (no bias)."""
        return ttnn.linear(
            flat,
            self.w_in,
            dtype=self.act_dtype,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: "AudioSubsampleInputs") -> ttnn.Tensor:
        flat = self.conv_stack(x)
        out = self.input_proj(flat)
        ttnn.deallocate(flat)
        return out


class AudioSubsampleInputs:
    """Duck type for the three subsampling inputs (audio_model.AudioDeviceInputs provides them)."""

    mel: ttnn.Tensor
    frame_mask: ttnn.Tensor
    mask1: ttnn.Tensor
