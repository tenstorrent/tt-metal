# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of the XTTS-v2 HiFi-GAN vocoder generator (Block 4).

Reference: models/experimental/xtts_v2/reference/xtts_hifigan_ref.py (PCC 1.0 vs coqui).

Block boundary: input GPT-latents-derived tensor z [1,1024,L] + d-vector g [1,512,1] ->
24 kHz waveform [1,1,L*256]. (The two host `interpolate`s that build z from the GPT latents
stay on CPU, as in the reference; this module is the generator proper.)

Layout convention (whole net): the 1D signal `[1, C, L]` is carried as NHWC `[1, 1, L, C]`
(height 1, width = time L, channels last). This makes every op natural:
  - `ttnn.conv1d` wants `[N, H, W, C] = [1,1,L,C]`, input_length=L, weight [out,in,k].
  - the 4 upsamples are `ttnn.conv_transpose2d` with a height-1 kernel (1,k), stride (1,s),
    padding (0,p) (there is no conv_transpose1d); weight reshaped to IOHW [in,out,1,k].
  - per-layer conditioning `conds[i](g)` / `cond_layer(g)` is a 1x1 conv on the length-1
    d-vector g == a plain linear -> a [1,1,1,C] vector, broadcast-added over the time axis.

Long sequences: activations grow to [1,1,101376,32] (396*256). The conv sliding-window/halo
config would blow L1 at those widths, so convs whose input length exceeds SLICE_L use a DRAM
width-slice (`Conv2dSliceConfig(Conv2dDRAMSliceWidth, num_slices=0)` -> auto).

Precision: **fp32 activations** (default). bf16 tops out at waveform PCC ~0.96 on this
oscillatory output — the 32->1 `conv_post` reduction amplifies the bf16 error accumulated
across the 12 ResBlocks (per-stage transpose outputs are still ~0.999 in bf16, but the final
waveform is not). fp32 clears the 0.99 gate at ~0.998. The fp32 conv config/halo tensor needs
a larger L1_SMALL than bf16 — open the device with `l1_small_size=65536` (bf16's 32768 OOMs
in fp32). See CLAUDE_XTTS_BUGS.md BUG-3.
"""

import ttnn

import torch

from models.experimental.xtts_v2.reference.xtts_hifigan_ref import (
    UPS,
    RES_K,
    RES_D,
    _pad,
    load_hifigan_state,
)

LRELU = 0.1  # LRELU_SLOPE used everywhere except the final activation
FINAL_LRELU = 0.01  # coqui uses torch's DEFAULT leaky_relu slope before conv_post (GOTCHA)
UP_CH = [256, 128, 64, 32]  # output channels of each upsample
# Convs whose input length is longer than this run with a DRAM width-slice to fit L1.
SLICE_L = 8192


def _compute_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def preprocess_hifigan_parameters(device, ckpt_path=None, conv_dtype=ttnn.float32, lin_dtype=ttnn.float32):
    """Build TTNN host/device weights from the (already weight-norm folded) generator state."""
    w = load_hifigan_state(ckpt_path) if ckpt_path else load_hifigan_state()

    def conv_w(name):  # Conv1d weight [out,in,k] -> ttnn host row-major (conv1d takes it as-is)
        return ttnn.from_torch(w[name], dtype=conv_dtype)

    def convT_w(name):  # ConvTranspose1d weight [in,out,k] -> IOHW [in,out,1,k]
        t = w[name]
        return ttnn.from_torch(t.reshape(t.shape[0], t.shape[1], 1, t.shape[2]), dtype=conv_dtype)

    def bias4(name):  # conv bias [out] -> [1,1,1,out]
        return ttnn.from_torch(w[name].reshape(1, 1, 1, -1), dtype=conv_dtype)

    def cond_linT(name):  # 1x1 conv weight [out,in,1] -> linear weight [in,out] on device (TILE)
        return ttnn.from_torch(
            w[name].squeeze(-1).t().contiguous(), dtype=lin_dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def vec(name):  # bias [out] -> [1,out] on device (TILE)
        return ttnn.from_torch(w[name].reshape(1, -1), dtype=lin_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    p = {
        "conv_pre_w": conv_w("conv_pre.weight"),
        "conv_pre_b": bias4("conv_pre.bias"),
        "cond_layer_w": cond_linT("cond_layer.weight"),
        "cond_layer_b": vec("cond_layer.bias"),
        "conv_post_w": conv_w("conv_post.weight"),  # no bias
        "ups": [],
        "conds_w": [],
        "conds_b": [],
        "resblocks": [],
    }
    for i in range(4):
        p["ups"].append({"w": convT_w(f"ups.{i}.weight"), "b": bias4(f"ups.{i}.bias")})
        p["conds_w"].append(cond_linT(f"conds.{i}.weight"))
        p["conds_b"].append(vec(f"conds.{i}.bias"))
    for r in range(12):  # 4 upsamples * 3 resblocks
        rb = {"convs1": [], "convs2": []}
        for j in range(3):
            rb["convs1"].append(
                {"w": conv_w(f"resblocks.{r}.convs1.{j}.weight"), "b": bias4(f"resblocks.{r}.convs1.{j}.bias")}
            )
            rb["convs2"].append(
                {"w": conv_w(f"resblocks.{r}.convs2.{j}.weight"), "b": bias4(f"resblocks.{r}.convs2.{j}.bias")}
            )
        p["resblocks"].append(rb)
    return p


class TTNNHifiganGenerator:
    def __init__(self, device, parameters, dtype=ttnn.float32, weights_dtype=None):
        self.device = device
        self.p = parameters
        self.cc = _compute_config()
        self.dtype = dtype  # activation dtype
        self.weights_dtype = weights_dtype or dtype

    def _conv_config(self):
        # Fresh Conv2dConfig per call: ttnn conv ops may write auto-selected sharding back into
        # the config, so a shared instance can leak conv1d's sharding into conv_transpose2d.
        return ttnn.Conv2dConfig(weights_dtype=self.weights_dtype)

    # ---- primitives ----------------------------------------------------------
    def _slice_cfg(self, L):
        if L > SLICE_L:
            return ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0)
        return None

    def _post(self, out, L, C):
        """conv output -> interleaved DRAM, NHWC [1,1,L,C], TILE layout."""
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (1, 1, L, C))
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        return out

    def _conv1d(self, x, w, b, in_ch, out_ch, L, k, pad, dil):
        """x: [1,1,L,in_ch] (any layout) -> [1,1,Lout,out_ch] TILE. Length-preserving 'same' pad."""
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        out, lo, _ = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=w,
            bias_tensor=b,
            device=self.device,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=1,
            input_length=L,
            kernel_size=k,
            stride=1,
            padding=pad,
            dilation=dil,
            groups=1,
            conv_config=self._conv_config(),
            compute_config=self.cc,
            dtype=self.dtype,
            slice_config=self._slice_cfg(L),
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return self._post(out, lo, out_ch), lo

    def _convT(self, x, w, b, in_ch, out_ch, L, k, s, pad):
        """Upsample via height-1 conv_transpose2d. -> [1,1,Lout,out_ch] TILE."""
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        slice_cfg = self._slice_cfg(L)
        kwargs = {}
        if slice_cfg is not None:
            kwargs["dram_slice_config"] = slice_cfg
        out, [ho, wo] = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=w,
            bias_tensor=b,
            device=self.device,
            in_channels=in_ch,
            out_channels=out_ch,
            batch_size=1,
            input_height=1,
            input_width=L,
            kernel_size=(1, k),
            stride=(1, s),
            padding=(0, pad),
            output_padding=(0, 0),
            dilation=(1, 1),
            conv_config=self._conv_config(),
            compute_config=self.cc,
            groups=1,
            return_output_dim=True,
            dtype=self.dtype,
            **kwargs,
        )
        return self._post(out, ho * wo, out_ch), wo

    def _cond(self, g2d, lin_w, lin_b, out_ch):
        """conds[i](g) / cond_layer(g): 1x1 conv on length-1 g == linear -> [1,1,1,out_ch]."""
        c = ttnn.linear(g2d, lin_w, bias=lin_b, compute_kernel_config=self.cc)  # [1,out_ch]
        return ttnn.reshape(c, (1, 1, 1, out_ch))

    def _resblock(self, x, rb, L, ch, k, dils):
        for j in range(3):
            xt = ttnn.leaky_relu(x, negative_slope=LRELU)
            xt, _ = self._conv1d(
                xt, rb["convs1"][j]["w"], rb["convs1"][j]["b"], ch, ch, L, k, _pad(k, dils[j]), dils[j]
            )
            xt = ttnn.leaky_relu(xt, negative_slope=LRELU)
            xt, _ = self._conv1d(xt, rb["convs2"][j]["w"], rb["convs2"][j]["b"], ch, ch, L, k, _pad(k, 1), 1)
            x = ttnn.add(xt, x)
        return x

    # ---- forward -------------------------------------------------------------
    def __call__(self, z, g, return_intermediates=False):
        """z: ttnn [1,1,L,1024] NHWC (TILE/RM); g: ttnn [1,512] -> waveform [1,1,L*256,1]."""
        inter = {}
        p = self.p
        _, _, L, _ = z.shape

        def cap(key, t):
            # Capture to HOST immediately: a device intermediate stashed in `inter` has its DRAM
            # buffer reused by later ops and reads as garbage after the forward (the forward path
            # itself is unaffected). So per-stage oracles must snapshot to torch at capture time.
            if return_intermediates:
                inter[key] = ttnn.to_torch(t).to(torch.float32)

        # 1) conv_pre + cond_layer(g)
        o, L = self._conv1d(z, p["conv_pre_w"], p["conv_pre_b"], 1024, 512, L, 7, 3, 1)
        cap("conv_pre", o)  # oracle dbg_conv_pre is conv_pre only (pre-cond)
        o = ttnn.add(o, self._cond(g, p["cond_layer_w"], p["cond_layer_b"], 512))

        # 2) upsample blocks
        for i in range(4):
            k, s, pad = UPS[i]
            ch = UP_CH[i]
            o = ttnn.leaky_relu(o, negative_slope=LRELU)
            o, L = self._convT(
                o, p["ups"][i]["w"], p["ups"][i]["b"], (512 if i == 0 else UP_CH[i - 1]), ch, L, k, s, pad
            )
            cap(f"ups{i}", o)  # oracle dbg_ups{i} is the transpose-conv output (pre-cond, pre-MRF)
            o = ttnn.add(o, self._cond(g, p["conds_w"][i], p["conds_b"][i], ch))
            # MRF: mean of 3 ResBlock1 (kernels 3,7,11; dils [1,3,5])
            acc = None
            for j in range(3):
                r = self._resblock(o, p["resblocks"][i * 3 + j], L, ch, RES_K[j], RES_D[j])
                acc = r if acc is None else ttnn.add(acc, r)
            o = ttnn.multiply(acc, 1.0 / 3.0)

        # 3) final activation (DEFAULT slope 0.01) -> conv_post (no bias) -> tanh
        o = ttnn.leaky_relu(o, negative_slope=FINAL_LRELU)
        o, L = self._conv1d(o, p["conv_post_w"], None, 32, 1, L, 7, 3, 1)
        o = ttnn.tanh(o)
        if return_intermediates:
            return o, inter
        return o
