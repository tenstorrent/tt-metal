# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Persistent TTNN runtime modules for RVC inference.

These modules preprocess and upload weights ONCE, then reuse them
across forward calls. This solves the device memory accumulation
problem where re-creating weights every forward causes L1 OOM.

Weight lifecycle:
    1. from_checkpoint() — load from safetensors, preprocess, upload to device
    2. forward()         — execute with persistent weights (no re-upload)
    3. deallocate()      — explicitly free device memory

Usage:
    flow = TTNNFlowDecoder.from_checkpoint(state_dict, device)
    gen = TTNNGeneratorNSF.from_checkpoint(state_dict, device)

    for batch in data:
        z = flow(z_p, g)
        audio = gen(z, har_source, g)

    flow.deallocate()
    gen.deallocate()
"""

import math
import torch
import torch.nn.functional as F
import ttnn

from models.demos.rvc.ttnn.utils import (
    to_device, to_host, preprocess_linear_weight, preprocess_linear_bias,
    preprocess_conv1d_weight, DEFAULT_DTYPE, DEFAULT_MEMORY_CONFIG,
)
from models.demos.rvc.ttnn.ops.conv_transpose1d import TTNNConvTranspose1d


# =====================================================================
# Config constants (v2 48k)
# =====================================================================

HIDDEN_CH = 192
HALF_CH = 96
KERNEL_SIZE = 5
DILATION_RATE = 1
NUM_WN_LAYERS = 3
N_FLOWS = 4
GIN_CHANNELS = 256

UPSAMPLE_RATES = [12, 10, 2, 2]
UPSAMPLE_KERNELS = [24, 20, 4, 4]
UPSAMPLE_INITIAL_CH = 512
RESBLOCK_DILATIONS = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
NUM_UPSAMPLES = 4       # len(UPSAMPLE_RATES)
NUM_KERNELS = 3         # ResBlocks per upsample stage (kernels 3/7/11)
LRELU_SLOPE = 0.1


def _conv1d_to_torch(result, out_channels):
    """Convert ttnn.conv1d result to torch tensor.

    Handles sharded→interleaved conversion and reshaping.
    This is the common postprocess path for all conv1d dispatches.
    """
    out_tt = result[0]
    out_len = result[1]
    try:
        out_tt = ttnn.sharded_to_interleaved(out_tt)
    except RuntimeError:
        pass
    out = ttnn.to_torch(ttnn.from_device(out_tt)).float()
    out = out.reshape(1, 1, out_len, -1)[:, :, :, :out_channels].squeeze(1)
    return out, out_len


def _linear_channel_first(x, weight, bias):
    """Matrix multiply for channels-first: weight @ x + bias."""
    y = torch.matmul(weight, x)
    if bias is not None:
        y = y + bias.view(1, -1, 1)
    return y


# =====================================================================
# Persistent Flow Decoder
# =====================================================================

class TTNNFlowDecoder:
    """Persistent TTNN flow decoder with device-resident weights.

    Weights are preprocessed and uploaded to device ONCE during from_checkpoint().
    forward() reuses the same device tensors without re-upload.

    Device tensor inventory (per flow × 4 flows):
        - pre_w, pre_b      : Linear(96 → 192)
        - post_w, post_b    : Linear(192 → 96)
        - cond_w, cond_b    : Linear(256 → 1152)
        - rsl_ws[3], rsl_bs[3] : res_skip Linear weights
        Total persistent device tensors: 4 × (6 + 6) = 48

    Transient per-forward:
        - conv weights (host tensors, uploaded by ttnn.conv1d internally)
        - intermediate activations (freed after each op)
    """

    def __init__(self):
        self._flows = []  # List of per-flow weight dicts
        self._conv_weights = []  # Host-side conv weights (not device-resident)
        self._device = None

    @classmethod
    def from_checkpoint(cls, state_dict, device):
        """Load flow weights from checkpoint, preprocess, upload to device."""
        obj = cls()
        obj._device = device

        for f in range(N_FLOWS):
            prefix = f"flow.flows.{f}"

            # Device-resident linear weights (uploaded once)
            fw = {
                "pre_w": preprocess_linear_weight(state_dict[f"{prefix}.pre_linear.weight"].float(), device),
                "pre_b": preprocess_linear_bias(state_dict[f"{prefix}.pre_linear.bias"].float(), device),
                "post_w": preprocess_linear_weight(state_dict[f"{prefix}.post_linear.weight"].float(), device),
                "post_b": preprocess_linear_bias(state_dict[f"{prefix}.post_linear.bias"].float(), device),
                "cond_w": preprocess_linear_weight(state_dict[f"{prefix}.enc.cond_layer.weight"].float(), device),
                "cond_b": preprocess_linear_bias(state_dict[f"{prefix}.enc.cond_layer.bias"].float(), device),
                "rsl_ws": [],
                "rsl_bs": [],
            }
            for i in range(NUM_WN_LAYERS):
                fw["rsl_ws"].append(
                    preprocess_linear_weight(state_dict[f"{prefix}.enc.res_skip_layers.{i}.weight"].float(), device))
                fw["rsl_bs"].append(
                    preprocess_linear_bias(state_dict[f"{prefix}.enc.res_skip_layers.{i}.bias"].float(), device))
            obj._flows.append(fw)

            # Host-side conv weights (ttnn.conv1d handles device transfer internally)
            conv = {
                "ws": [],
                "bs": [],
            }
            for i in range(NUM_WN_LAYERS):
                conv["ws"].append(
                    preprocess_conv1d_weight(state_dict[f"{prefix}.enc.in_layers.{i}.weight"].float()))
                conv["bs"].append(
                    state_dict[f"{prefix}.enc.in_layers.{i}.bias"].float())
            obj._conv_weights.append(conv)

        return obj

    def _conditioned_wn(self, x_cl, g_proj_cl, flow_idx, seq_len):
        """Conditioned WN using persistent weights."""
        fw = self._flows[flow_idx]
        conv = self._conv_weights[flow_idx]
        output_acc = torch.zeros(x_cl.shape[0], seq_len, HIDDEN_CH)

        for i in range(NUM_WN_LAYERS):
            d = DILATION_RATE ** i
            padding = d * (KERNEL_SIZE - 1) // 2

            # Conv1d (conv weight is host tensor, ttnn.conv1d manages device)
            x_tt = ttnn.from_torch(x_cl, dtype=DEFAULT_DTYPE)
            result = ttnn.conv1d(
                input_tensor=x_tt, weight_tensor=conv["ws"][i], device=self._device,
                in_channels=HIDDEN_CH, out_channels=2 * HIDDEN_CH, batch_size=1,
                input_length=seq_len, kernel_size=KERNEL_SIZE, stride=1,
                padding=padding, dilation=d, groups=1,
                dtype=DEFAULT_DTYPE, return_output_dim=True,
            )
            conv_torch, _ = _conv1d_to_torch(result, 2 * HIDDEN_CH)
            conv_torch = conv_torch + conv["bs"][i].unsqueeze(0).unsqueeze(0)

            # Conditioning
            cond_offset = i * 2 * HIDDEN_CH
            g_l = g_proj_cl[:, :, cond_offset:cond_offset + 2*HIDDEN_CH]
            conv_torch = conv_torch + g_l

            # Gating on device
            tanh_tt = to_device(conv_torch[:, :, :HIDDEN_CH], self._device)
            sig_tt = to_device(conv_torch[:, :, HIDDEN_CH:], self._device)
            gated = ttnn.multiply(ttnn.tanh(tanh_tt), ttnn.sigmoid(sig_tt))

            # res_skip linear (persistent weights)
            rs_out = ttnn.linear(gated, fw["rsl_ws"][i], bias=fw["rsl_bs"][i],
                                  memory_config=DEFAULT_MEMORY_CONFIG)
            rs_torch = to_host(rs_out)[:1, :seq_len, :]

            # Deallocate transient device tensors
            ttnn.deallocate(tanh_tt)
            ttnn.deallocate(sig_tt)
            ttnn.deallocate(gated)
            ttnn.deallocate(rs_out)

            if i < NUM_WN_LAYERS - 1:
                x_cl = x_cl + rs_torch[:, :, :HIDDEN_CH]
                output_acc = output_acc + rs_torch[:, :, HIDDEN_CH:]
            else:
                output_acc = output_acc + rs_torch[:, :, :HIDDEN_CH]

        return output_acc

    def forward(self, z_p, g):
        """Execute 4-flow decoder with persistent weights.

        Args:
            z_p: [1, 192, T] channels-first latent from TextEncoder
            g:   [1, 256, 1] channels-first speaker embedding

        Returns:
            z: [1, 192, T] channels-first decoded latent
        """
        assert z_p.shape[0] == 1, f"Only batch=1 supported, got {z_p.shape[0]}"
        seq_len = z_p.shape[2]
        x_cl = z_p.permute(0, 2, 1)  # [1, T, C]

        for f in range(N_FLOWS):
            fw = self._flows[f]

            # Channel flip
            x_cl = torch.flip(x_cl, [2])
            x0_cl = x_cl[:, :, :HALF_CH]
            x1_cl = x_cl[:, :, HALF_CH:]

            # pre_linear (persistent weight)
            x0_tt = to_device(x0_cl, self._device)
            h_tt = ttnn.linear(x0_tt, fw["pre_w"], bias=fw["pre_b"],
                                memory_config=DEFAULT_MEMORY_CONFIG)
            h_cl = to_host(h_tt)[:1, :seq_len, :HIDDEN_CH]
            ttnn.deallocate(x0_tt)
            ttnn.deallocate(h_tt)

            # Project g (persistent weight)
            g_cl = g.permute(0, 2, 1)
            g_tt = to_device(g_cl, self._device)
            g_proj_tt = ttnn.linear(g_tt, fw["cond_w"], bias=fw["cond_b"],
                                     memory_config=DEFAULT_MEMORY_CONFIG)
            g_proj_cl = to_host(g_proj_tt)[:1, :1, :2*HIDDEN_CH*NUM_WN_LAYERS]
            ttnn.deallocate(g_tt)
            ttnn.deallocate(g_proj_tt)

            # Conditioned WN
            wn_out = self._conditioned_wn(h_cl, g_proj_cl, f, seq_len)

            # post_linear (persistent weight)
            wn_tt = to_device(wn_out, self._device)
            stats_tt = ttnn.linear(wn_tt, fw["post_w"], bias=fw["post_b"],
                                    memory_config=DEFAULT_MEMORY_CONFIG)
            stats_cl = to_host(stats_tt)[:1, :seq_len, :HALF_CH]
            ttnn.deallocate(wn_tt)
            ttnn.deallocate(stats_tt)

            # Affine + concat
            x1_cl = x1_cl - stats_cl
            x_cl = torch.cat([x0_cl, x1_cl], dim=-1)

        return x_cl.permute(0, 2, 1)  # [1, C, T]

    def __call__(self, z_p, g):
        """Enable callable syntax: flow(z_p, g) instead of flow.forward(z_p, g)."""
        return self.forward(z_p, g)

    def deallocate(self):
        """Explicitly free all device-resident weights."""
        for fw in self._flows:
            for key in ["pre_w", "pre_b", "post_w", "post_b", "cond_w", "cond_b"]:
                try:
                    ttnn.deallocate(fw[key])
                except (RuntimeError, ValueError):
                    pass
            for w in fw["rsl_ws"]:
                try:
                    ttnn.deallocate(w)
                except (RuntimeError, ValueError):
                    pass
            for b in fw["rsl_bs"]:
                try:
                    ttnn.deallocate(b)
                except (RuntimeError, ValueError):
                    pass
        self._flows = []
        self._conv_weights = []


# =====================================================================
# Persistent Generator
# =====================================================================

class TTNNGeneratorNSF:
    """Persistent TTNN generator with device-resident weights.

    Architecture: conv_pre → cond → 4×(upsample + noise + 3×ResBlock) → conv_post → tanh

    Device-resident tensors:
        - cond_w, cond_b: Linear(256 → 512)
        - 4 × TTNNConvTranspose1d (each holds weight + bias on host for conv_transpose2d)

    Host-resident tensors (ttnn.conv1d manages device transfer):
        - conv_pre weight
        - conv_post weight
        - 12 ResBlocks × 6 conv weights = 72 conv weight tensors
        - 4 noise_conv weights

    The conv1d weights are host tensors. ttnn.conv1d handles device upload
    internally with its own caching. By reusing the SAME host tensor objects,
    we allow ttnn's internal cache to work instead of creating new entries.
    """

    def __init__(self):
        self._device = None
        # Device-resident
        self._cond_w = None
        self._cond_b = None
        # Host-resident (persistent objects for ttnn.conv1d cache)
        self._conv_pre_w = None
        self._conv_pre_b = None
        self._conv_post_w = None
        self._ups = []  # TTNNConvTranspose1d objects
        self._noise_convs = []  # dicts with torch weights
        self._resblocks = []  # dicts with host ttnn tensors
        self._gen_torch = None  # torch weights for ops that stay on host

    @classmethod
    def from_checkpoint(cls, state_dict, device):
        """Load generator weights from checkpoint."""
        obj = cls()
        obj._device = device
        sd = state_dict

        # conv_pre: host tensor (for ttnn.conv1d)
        obj._conv_pre_w = preprocess_conv1d_weight(sd["dec.conv_pre.weight"].float())
        obj._conv_pre_b = sd["dec.conv_pre.bias"].float()

        # conv_post: host tensor (for ttnn.conv1d)
        obj._conv_post_w = preprocess_conv1d_weight(sd["dec.conv_post.weight"].float())

        # cond_linear: device-resident
        obj._cond_w = preprocess_linear_weight(sd["dec.cond_linear.weight"].float(), device)
        obj._cond_b = preprocess_linear_bias(sd["dec.cond_linear.bias"].float(), device)

        # Upsample ConvTranspose1d: created once, weights loaded once
        for i in range(NUM_UPSAMPLES):
            s = UPSAMPLE_RATES[i]
            k = UPSAMPLE_KERNELS[i]
            p = (k - s) // 2
            in_ch = UPSAMPLE_INITIAL_CH // (2 ** i)
            out_ch = UPSAMPLE_INITIAL_CH // (2 ** (i + 1))
            ct = TTNNConvTranspose1d(device, in_ch, out_ch, k, stride=s, padding=p)
            ct.load_weights(sd[f"dec.ups.{i}.weight"].float(),
                            sd[f"dec.ups.{i}.bias"].float())
            obj._ups.append(ct)

        # Noise convs: torch weights (used via F.conv1d or linear_channel_first)
        for i in range(NUM_UPSAMPLES):
            obj._noise_convs.append({
                "w": sd[f"dec.noise_convs.{i}.weight"].float(),
                "b": sd[f"dec.noise_convs.{i}.bias"].float(),
            })

        # ResBlocks: host ttnn tensors (persistent for ttnn.conv1d cache)
        # Bias is preprocessed as ttnn tensor for direct use in conv1d bias_tensor param
        def _prep_bias_for_conv1d(bias_1d):
            """Reshape [out_ch] bias to [1,1,1,out_ch] ttnn tensor for conv1d."""
            return ttnn.from_torch(
                bias_1d.reshape(1, 1, 1, -1), dtype=DEFAULT_DTYPE
            )

        for rb in range(12):
            block = {"convs1": [], "convs2": []}
            for d in range(3):
                block["convs1"].append({
                    "w": preprocess_conv1d_weight(sd[f"dec.resblocks.{rb}.convs1.{d}.weight"].float()),
                    "b_tt": _prep_bias_for_conv1d(sd[f"dec.resblocks.{rb}.convs1.{d}.bias"].float()),
                    "kernel": sd[f"dec.resblocks.{rb}.convs1.{d}.weight"].shape[2],
                })
                block["convs2"].append({
                    "w": preprocess_conv1d_weight(sd[f"dec.resblocks.{rb}.convs2.{d}.weight"].float()),
                    "b_tt": _prep_bias_for_conv1d(sd[f"dec.resblocks.{rb}.convs2.{d}.bias"].float()),
                    "kernel": sd[f"dec.resblocks.{rb}.convs2.{d}.weight"].shape[2],
                })
            block["channels"] = sd[f"dec.resblocks.{rb}.convs1.0.weight"].shape[0]
            obj._resblocks.append(block)

        return obj

    def _conv1d_persistent(self, x_cl, w_tt, b_torch, in_ch, out_ch, k, seq_len, dilation=1):
        """Conv1d using persistent host weight tensor (legacy path for conv_pre/conv_post)."""
        padding = dilation * (k - 1) // 2
        x_tt = ttnn.from_torch(x_cl, dtype=DEFAULT_DTYPE)
        result = ttnn.conv1d(
            input_tensor=x_tt, weight_tensor=w_tt, device=self._device,
            in_channels=in_ch, out_channels=out_ch, batch_size=1,
            input_length=seq_len, kernel_size=k, stride=1,
            padding=padding, dilation=dilation, groups=1,
            dtype=DEFAULT_DTYPE, return_output_dim=True,
        )
        out, out_len = _conv1d_to_torch(result, out_ch)
        if b_torch is not None:
            out = out + b_torch.unsqueeze(0).unsqueeze(0)
        return out, out_len

    def _conv1d_device(self, x_in, w_tt, b_tt, in_ch, out_ch, k, seq_len,
                       dilation=1):
        """Conv1d that returns a device tensor in interleaved DRAM.

        Used by _resblock1_device. Input must be ROW_MAJOR — ttnn.conv1d at
        certain configs (notably k=11/d=5/ch=128/seq=7200) rejects TILE
        inputs with `program.cpp:1403: tt::exception`, regardless of
        storage or memory_config. The caller must convert with
        `ttnn.to_layout(..., ROW_MAJOR_LAYOUT)` before calling this.
        Output is forced to interleaved DRAM (sharded L1 outputs accumulate
        bank pressure across chained calls).
        """
        padding = dilation * (k - 1) // 2
        result = ttnn.conv1d(
            input_tensor=x_in, weight_tensor=w_tt, device=self._device,
            in_channels=in_ch, out_channels=out_ch, batch_size=1,
            input_length=seq_len, kernel_size=k, stride=1,
            padding=padding, dilation=dilation, groups=1,
            dtype=DEFAULT_DTYPE, return_output_dim=True,
            bias_tensor=b_tt,
        )
        out_tt = result[0]
        try:
            out_tt = ttnn.sharded_to_interleaved(out_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except RuntimeError:
            out_tt = ttnn.to_memory_config(out_tt, ttnn.DRAM_MEMORY_CONFIG)
        return out_tt

    def _resblock1_device(self, x_cf, block_idx, dilations, seq_len):
        """Single ResBlock1, device-resident (one `from_torch` in, one
        `to_torch` out).

        Thin wrapper around `_resblock_core_device`: uploads the input,
        runs the three dilation iterations of
            leaky_relu → conv1 → leaky_relu → conv2 → add(residual)
        on device, then downloads. Retained as the per-block reference path
        and exercised directly by `tests/test_production_shapes.py`. The
        generator's `forward` no longer calls this per ResBlock — it uses
        `_resblock_group_resident`, which shares one device-resident input
        across all three ResBlocks of a stage (Phase 3A.1/3A.2a).

        Two layout constraints must be honored on every step:

          - `ttnn.leaky_relu` requires `Layout.TILE` for its compute kernel.
          - `ttnn.conv1d` rejects `Layout.TILE` input with
            `program.cpp:1403: tt::exception` at certain
            (kernel_size, dilation, channels, input_width) combinations —
            confirmed at (k=11, d=5, ch=128, seq=7200), independent of
            storage or memory_config.

        The fix is a `ttnn.to_layout(x, ROW_MAJOR_LAYOUT)` between every
        leaky_relu output and the conv1d that follows it. Cheap, device-side.

        See README "How the optimization works" for the why; see
        `tests/test_production_shapes.py` for the regression guard.
        """
        x_nhwc = x_cf.permute(0, 2, 1).unsqueeze(1)
        x_dev = ttnn.from_torch(
            x_nhwc.float(),
            dtype=DEFAULT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_dev = self._resblock_core_device(x_dev, block_idx, dilations, seq_len)
        ttnn.deallocate(x_dev)
        out = ttnn.to_torch(out_dev).float().squeeze(1).permute(0, 2, 1)
        ttnn.deallocate(out_dev)
        return out

    def _resblock_core_device(self, x_shared, block_idx, dilations, seq_len):
        """Device-resident ResBlock1 inner loop on an already-uploaded tensor.

        Runs the three dilation iterations of
            leaky_relu → conv1 → leaky_relu → conv2 → add(residual)
        entirely on device and returns a NEW device tensor. `x_shared` (the
        block input, NHWC/TILE) is treated as read-only and is NOT deallocated
        here: _resblock_group_resident feeds the same `x_shared` to all three
        kernels of a stage, so its lifetime is owned by the caller.

        Layout discipline is the same as documented on `_resblock1_device`:
        leaky_relu needs TILE; conv1d rejects TILE at certain configs, so a
        `to_layout(ROW_MAJOR)` sits between each leaky_relu and the conv1d
        that follows it.
        """
        block = self._resblocks[block_idx]
        ch = block["channels"]

        x_dev = x_shared
        owns_x = False  # iteration 0's x_dev is the caller-owned x_shared
        for idx in range(3):
            d = dilations[idx]
            c1 = block["convs1"][idx]
            c2 = block["convs2"][idx]

            # leaky_relu reads TILE
            lr1 = ttnn.leaky_relu(x_dev, negative_slope=LRELU_SLOPE)
            # conv1d rejects TILE at some configs → relayout
            lr1_rm = ttnn.to_layout(lr1, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(lr1)
            c1_out = self._conv1d_device(
                lr1_rm, c1["w"], c1["b_tt"], ch, ch, c1["kernel"], seq_len,
                dilation=d)
            ttnn.deallocate(lr1_rm)

            # conv1d outputs TILE; leaky_relu accepts TILE
            lr2 = ttnn.leaky_relu(c1_out, negative_slope=LRELU_SLOPE)
            ttnn.deallocate(c1_out)
            lr2_rm = ttnn.to_layout(lr2, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(lr2)
            c2_out = self._conv1d_device(
                lr2_rm, c2["w"], c2["b_tt"], ch, ch, c2["kernel"], seq_len)
            ttnn.deallocate(lr2_rm)

            new_x = ttnn.add(x_dev, c2_out)
            ttnn.deallocate(c2_out)
            if owns_x:
                ttnn.deallocate(x_dev)  # our own intermediate, never x_shared
            owns_x = True
            x_dev = new_x

        return x_dev

    def _resblock_group_resident(self, x_shared, stage, seq_len):
        """ResBlock-group body on an already-resident device tensor.

        Takes `x_shared` (NHWC/TILE device tensor), runs all NUM_KERNELS
        ResBlocks reading it, sums their outputs and averages
        (×1/NUM_KERNELS) on device, and returns the result — no host
        transfer. Consumes (deallocates) `x_shared`.

        Phase 3A.1 introduced device-resident per-stage ResBlock grouping
        (Σ and average on device); Phase 3A.2a feeds it the conv_transpose
        output directly from `forward` so the whole stage body stays resident.
        Transient device tensors (per-kernel outputs, the running accumulator,
        the shared input) are deallocated as soon as they are consumed, so
        device residency does not grow across the group or across chunks.
        """
        acc = None
        for j in range(NUM_KERNELS):
            rb_idx = stage * NUM_KERNELS + j
            rb_out = self._resblock_core_device(
                x_shared, rb_idx, RESBLOCK_DILATIONS[j], seq_len)
            if acc is None:
                acc = rb_out
            else:
                new_acc = ttnn.add(acc, rb_out)
                ttnn.deallocate(acc)
                ttnn.deallocate(rb_out)
                acc = new_acc

        ttnn.deallocate(x_shared)
        avg = ttnn.multiply(acc, 1.0 / NUM_KERNELS)
        ttnn.deallocate(acc)
        return avg

    def forward(self, z, har_source, g):
        """Execute full generator with persistent weights.

        Args:
            z:          [1, 192, T] latent from flow decoder
            har_source: [1, 1, T*480] harmonic source from SineGen
            g:          [1, 256, 1] speaker embedding

        Returns:
            audio: [1, 1, T*480] generated waveform in [-1, 1]
        """
        seq_len = z.shape[2]

        # conv_pre (persistent host weight)
        x_cl = z.permute(0, 2, 1)
        x_cl, _ = self._conv1d_persistent(
            x_cl, self._conv_pre_w, self._conv_pre_b, 192, 512, 7, seq_len)
        x_cf = x_cl.permute(0, 2, 1)

        # Conditioning (persistent device weight)
        g_cl = g.permute(0, 2, 1)
        g_tt = to_device(g_cl, self._device)
        cond_tt = ttnn.linear(g_tt, self._cond_w, bias=self._cond_b,
                               memory_config=DEFAULT_MEMORY_CONFIG)
        cond_cl = to_host(cond_tt)[:1, :1, :512]
        ttnn.deallocate(g_tt)
        ttnn.deallocate(cond_tt)
        x_cf = x_cf + cond_cl.permute(0, 2, 1)

        for i in range(NUM_UPSAMPLES):
            ct = self._ups[i]

            # --- Within-stage device residency (Phase 3A.2a) ---
            # Upload the stage activation ONCE; leaky_relu, conv_transpose,
            # noise-add and the ResBlock group all run device-resident. The
            # stage boundary (the download below) is intentionally retained —
            # cross-stage residency is 3A.2b.
            x_nhwc = x_cf.permute(0, 2, 1).unsqueeze(1)
            x_dev = ttnn.from_torch(
                x_nhwc.float(), dtype=DEFAULT_DTYPE, layout=ttnn.TILE_LAYOUT,
                device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

            # pre-upsample leaky_relu (device, TILE); conv_transpose wants ROW_MAJOR
            lr = ttnn.leaky_relu(x_dev, negative_slope=LRELU_SLOPE)
            ttnn.deallocate(x_dev)
            lr_rm = ttnn.to_layout(lr, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(lr)

            # ConvTranspose1d output stays resident (NHWC/TILE/DRAM); the old
            # postprocess_output host download is dropped — probe-verified
            # bit-identical to the on-device tensor.
            ct_out, out_len = ct(lr_rm, batch_size=1, input_length=seq_len)
            ttnn.deallocate(lr_rm)
            seq_len = out_len

            # Noise injection: generated on host (conservative, unchanged math),
            # uploaded as NHWC/TILE, added on device.
            nc = self._noise_convs[i]
            if i < NUM_UPSAMPLES - 1:
                stride_f0 = math.prod(UPSAMPLE_RATES[i + 1:])
                x_source = F.conv1d(har_source, nc["w"], nc["b"],
                                     stride=stride_f0, padding=stride_f0 // 2)
            else:
                x_source = _linear_channel_first(har_source, nc["w"], nc["b"])
            src_nhwc = x_source[:, :, :seq_len].permute(0, 2, 1).unsqueeze(1)
            src_dev = ttnn.from_torch(
                src_nhwc.float(), dtype=DEFAULT_DTYPE, layout=ttnn.TILE_LAYOUT,
                device=self._device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x_dev = ttnn.add(ct_out, src_dev)
            ttnn.deallocate(ct_out)
            ttnn.deallocate(src_dev)

            # ResBlock group on the resident tensor; download to host CF at the
            # stage boundary (retained for 3A.2a).
            avg = self._resblock_group_resident(x_dev, i, seq_len)
            x_cf = ttnn.to_torch(avg).float().squeeze(1).permute(0, 2, 1)
            ttnn.deallocate(avg)

        # conv_post + tanh (persistent host weight)
        x_cf = F.leaky_relu(x_cf)
        x_cl = x_cf.permute(0, 2, 1)
        x_cl, _ = self._conv1d_persistent(
            x_cl, self._conv_post_w, None, 32, 1, 7, seq_len)
        return torch.tanh(x_cl.permute(0, 2, 1))

    def __call__(self, z, har_source, g):
        """Enable callable syntax: gen(z, har, g) instead of gen.forward(z, har, g)."""
        return self.forward(z, har_source, g)

    def deallocate(self):
        """Explicitly free device-resident weights."""
        for tensor in [self._cond_w, self._cond_b]:
            if tensor is not None:
                try:
                    ttnn.deallocate(tensor)
                except (RuntimeError, ValueError):
                    pass
        self._cond_w = None
        self._cond_b = None
        self._ups = []
        self._resblocks = []
        self._noise_convs = []
