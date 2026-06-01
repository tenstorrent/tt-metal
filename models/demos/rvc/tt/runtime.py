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

from models.demos.rvc.tt.utils import (
    to_device, to_host, preprocess_linear_weight, preprocess_linear_bias,
    preprocess_conv1d_weight, DEFAULT_DTYPE, DEFAULT_MEMORY_CONFIG,
)
from models.demos.rvc.tt.ops.conv_transpose1d import TTNNConvTranspose1d


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


def _conv1d_to_torch(result, out_channels, batch=1):
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
    out = out.reshape(batch, 1, out_len, -1)[:, :, :, :out_channels].squeeze(1)
    return out, out_len
NUM_KERNELS = 3
NUM_UPSAMPLES = 4
LRELU_SLOPE = 0.1

# Fused activation config: eliminates separate leaky_relu dispatch per conv
_FUSED_LRELU_CONFIG = ttnn.Conv2dConfig(
    activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, LRELU_SLOPE)
)

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
        # Lazy cache of prepared conv1d weight+bias keyed by (flow_idx, layer_idx, seq_len).
        # prepare_conv_weights does the per-call internal weight tilization once,
        # returning a device-resident tensor — skips that write on every conv1d call.
        self._prep_cache = {}

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

            # Host-side conv weights + ttnn-tensor biases for native conv1d bias_tensor= path
            # (matches Generator's _prep_bias_for_conv1d pattern; eliminates host-side bias add)
            conv = {
                "ws": [],
                "bs_tt": [],
            }
            for i in range(NUM_WN_LAYERS):
                conv["ws"].append(
                    preprocess_conv1d_weight(state_dict[f"{prefix}.enc.in_layers.{i}.weight"].float()))
                bias_1d = state_dict[f"{prefix}.enc.in_layers.{i}.bias"].float()
                conv["bs_tt"].append(
                    ttnn.from_torch(bias_1d.reshape(1, 1, 1, -1), dtype=DEFAULT_DTYPE))
            obj._conv_weights.append(conv)

        return obj

    def _ensure_prepared_conv(self, flow_idx, layer_idx, seq_len, batch=1):
        """Get prepared conv1d (weight, bias, cfg) for this shape; prep on miss.

        Cache key includes batch_size so different batches get their own
        prepared tensors (ttnn.prepare_conv_weights bakes the batch in).
        """
        key = (flow_idx, layer_idx, seq_len, batch)
        cached = self._prep_cache.get(key)
        if cached is not None:
            return cached
        conv = self._conv_weights[flow_idx]
        d = DILATION_RATE ** layer_idx
        padding = d * (KERNEL_SIZE - 1) // 2
        cfg = ttnn.Conv2dConfig(weights_dtype=DEFAULT_DTYPE)
        common = dict(
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            in_channels=HIDDEN_CH, out_channels=2 * HIDDEN_CH, batch_size=batch,
            input_height=1, input_width=seq_len,
            kernel_size=(1, KERNEL_SIZE), stride=(1, 1),
            padding=(0, padding), dilation=(1, d), groups=1,
            device=self._device, input_dtype=DEFAULT_DTYPE,
            conv_config=cfg,
        )
        w_p = ttnn.prepare_conv_weights(
            weight_tensor=conv["ws"][layer_idx],
            weights_format="OIHW", has_bias=True, **common)
        b_p = ttnn.prepare_conv_bias(
            bias_tensor=conv["bs_tt"][layer_idx], **common)
        self._prep_cache[key] = (w_p, b_p, cfg)
        return self._prep_cache[key]

    def _conditioned_wn_device(self, h_tt, g_proj_4d, flow_idx, seq_len, batch=1):
        """Device-resident WN inner loop.

        Inputs are TTNN device tensors; output is a TTNN device tensor.
        Eliminates the per-layer to_torch/to_device roundtrip pair that
        the host-routed path paid 48× per inference.

        Args:
            h_tt:        [1, T, HIDDEN_CH] TILE — pre_linear output (device-resident)
            g_proj_4d:   [1, 1, 1, 2*HIDDEN*NUM_LAYERS] TILE — cond_linear output reshaped
            flow_idx:    which of N_FLOWS
            seq_len:     T

        Returns:
            output_acc:  [1, 1, T, HIDDEN_CH] device tensor (TILE)
        """
        fw = self._flows[flow_idx]
        conv = self._conv_weights[flow_idx]

        # Reshape h_tt to 4D NHWC-ish for conv1d: [batch, 1, T, HIDDEN]
        x_dev = ttnn.reshape(h_tt, (batch, 1, seq_len, HIDDEN_CH))
        output_acc = None

        for i in range(NUM_WN_LAYERS):
            d = DILATION_RATE ** i
            padding = d * (KERNEL_SIZE - 1) // 2

            # conv1d rejects TILE input at the WN config -> ROW_MAJOR
            x_rm = ttnn.to_layout(x_dev, ttnn.ROW_MAJOR_LAYOUT)
            w_p, b_p, cfg = self._ensure_prepared_conv(flow_idx, i, seq_len, batch)
            result = ttnn.conv1d(
                input_tensor=x_rm, weight_tensor=w_p, device=self._device,
                in_channels=HIDDEN_CH, out_channels=2 * HIDDEN_CH, batch_size=batch,
                input_length=seq_len, kernel_size=KERNEL_SIZE, stride=1,
                padding=padding, dilation=d, groups=1,
                dtype=DEFAULT_DTYPE, return_output_dim=True,
                bias_tensor=b_p,
                conv_config=cfg,
            )
            ttnn.deallocate(x_rm)
            conv_out = result[0]
            try:
                conv_out = ttnn.sharded_to_interleaved(conv_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            except RuntimeError:
                conv_out = ttnn.to_memory_config(conv_out, ttnn.DRAM_MEMORY_CONFIG)
            # conv1d may flatten [B,1,T,C] -> [1,1,B*T,C]; reshape back so
            # downstream slices on T and broadcast-add against [B,1,1,C] g_l
            # align correctly per batch row.
            if batch > 1:
                conv_out = ttnn.reshape(conv_out, (batch, 1, seq_len, 2 * HIDDEN_CH))

            # Slice conditioning from g_proj_4d on device and broadcast-add
            cond_offset = i * 2 * HIDDEN_CH
            g_l = ttnn.slice(g_proj_4d, [0, 0, 0, cond_offset],
                              [batch, 1, 1, cond_offset + 2 * HIDDEN_CH])
            cond_in = ttnn.add(conv_out, g_l)
            ttnn.deallocate(conv_out)
            ttnn.deallocate(g_l)

            # Split gating halves on device
            tanh_half = ttnn.slice(cond_in, [0, 0, 0, 0],
                                    [batch, 1, seq_len, HIDDEN_CH])
            sig_half = ttnn.slice(cond_in, [0, 0, 0, HIDDEN_CH],
                                   [batch, 1, seq_len, 2 * HIDDEN_CH])
            ttnn.deallocate(cond_in)

            gated = ttnn.mul(ttnn.tanh(tanh_half), ttnn.sigmoid(sig_half))
            ttnn.deallocate(tanh_half)
            ttnn.deallocate(sig_half)

            # res_skip linear on device
            rs_out = ttnn.linear(gated, fw["rsl_ws"][i], bias=fw["rsl_bs"][i],
                                  memory_config=DEFAULT_MEMORY_CONFIG)
            ttnn.deallocate(gated)

            if i < NUM_WN_LAYERS - 1:
                # rs_out: [batch, 1, T, 2*HIDDEN] — first half residual, second half skip
                res_part = ttnn.slice(rs_out, [0, 0, 0, 0],
                                       [batch, 1, seq_len, HIDDEN_CH])
                skip_part = ttnn.slice(rs_out, [0, 0, 0, HIDDEN_CH],
                                        [batch, 1, seq_len, 2 * HIDDEN_CH])
                ttnn.deallocate(rs_out)

                new_x = ttnn.add(x_dev, res_part)
                ttnn.deallocate(x_dev)
                ttnn.deallocate(res_part)
                x_dev = new_x

                if output_acc is None:
                    output_acc = skip_part
                else:
                    new_acc = ttnn.add(output_acc, skip_part)
                    ttnn.deallocate(output_acc)
                    ttnn.deallocate(skip_part)
                    output_acc = new_acc
            else:
                # Last layer: rs_out shape [1, 1, T, HIDDEN] = skip only
                if output_acc is None:
                    output_acc = rs_out
                else:
                    new_acc = ttnn.add(output_acc, rs_out)
                    ttnn.deallocate(output_acc)
                    ttnn.deallocate(rs_out)
                    output_acc = new_acc

        ttnn.deallocate(x_dev)
        return output_acc

    def forward(self, z_p, g):
        """Execute 4-flow decoder with persistent weights.

        Args:
            z_p: [B, 192, T] channels-first latent from TextEncoder
            g:   [B, 256, 1] channels-first speaker embedding

        Returns:
            z: [B, 192, T] channels-first decoded latent
        """
        batch = z_p.shape[0]
        seq_len = z_p.shape[2]
        x_cl = z_p.permute(0, 2, 1)  # [B, T, C]

        for f in range(N_FLOWS):
            fw = self._flows[f]

            # Channel flip
            x_cl = torch.flip(x_cl, [2])
            x0_cl = x_cl[:, :, :HALF_CH]
            x1_cl = x_cl[:, :, HALF_CH:]

            # pre_linear (persistent weight) — output stays on device
            x0_tt = to_device(x0_cl, self._device)
            h_tt = ttnn.linear(x0_tt, fw["pre_w"], bias=fw["pre_b"],
                                memory_config=DEFAULT_MEMORY_CONFIG)
            ttnn.deallocate(x0_tt)

            # Project g (persistent weight) — output stays on device
            g_cl = g.permute(0, 2, 1)
            g_tt = to_device(g_cl, self._device)
            g_proj_tt = ttnn.linear(g_tt, fw["cond_w"], bias=fw["cond_b"],
                                     memory_config=DEFAULT_MEMORY_CONFIG)
            ttnn.deallocate(g_tt)
            # Reshape to 4D for broadcast-add against [B,1,T,2*HIDDEN] conv outputs
            g_proj_4d = ttnn.reshape(g_proj_tt, (batch, 1, 1, 2 * HIDDEN_CH * NUM_WN_LAYERS))

            # Conditioned WN — fully device-resident
            wn_out = self._conditioned_wn_device(h_tt, g_proj_4d, f, seq_len, batch)
            ttnn.deallocate(g_proj_4d)

            # post_linear (persistent weight) — input on device
            # Reshape wn_out from [B,1,T,HIDDEN] back to [B,T,HIDDEN] for linear
            wn_3d = ttnn.reshape(wn_out, (batch, seq_len, HIDDEN_CH))
            ttnn.deallocate(wn_out)
            stats_tt = ttnn.linear(wn_3d, fw["post_w"], bias=fw["post_b"],
                                    memory_config=DEFAULT_MEMORY_CONFIG)
            stats_cl = to_host(stats_tt)[:batch, :seq_len, :HALF_CH]
            ttnn.deallocate(wn_3d)
            ttnn.deallocate(stats_tt)

            # Affine + concat
            x1_cl = x1_cl - stats_cl
            x_cl = torch.cat([x0_cl, x1_cl], dim=-1)

        return x_cl.permute(0, 2, 1)  # [B, C, T]

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
        # Conditioning projection cache: cond_linear(g) only depends on the
        # speaker embedding g, which is constant across chunks of a single
        # inference. Cache cond_cl keyed by g so we don't repeat the linear
        # + two host roundtrips on every chunk.
        self._cond_cache_g = None
        self._cond_cache_cl = None
        # Lazy cache of prepared ResBlock conv1d weight+bias keyed by shape.
        # ttnn.conv1d re-tilizes weight on every call by default;
        # prepare_conv_weights does that once and returns a device-resident
        # tensor — eliminates the per-call write that dominates Generator
        # inner-loop overhead (288 conv1d calls per inference).
        self._prep_cache = {}

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

    def _conv1d_persistent(self, x_cl, w_tt, b_torch, in_ch, out_ch, k, seq_len, dilation=1, batch=1):
        """Conv1d using persistent host weight tensor (legacy path for conv_pre/conv_post).

        At batch>1, pin act_block_h_override=32 — auto-picker otherwise busts
        L1 (static CB clashes with prior L1 buffers); 32 makes all shapes fit.
        """
        padding = dilation * (k - 1) // 2
        x_tt = ttnn.from_torch(x_cl, dtype=DEFAULT_DTYPE)
        conv_kwargs = dict(
            input_tensor=x_tt, weight_tensor=w_tt, device=self._device,
            in_channels=in_ch, out_channels=out_ch, batch_size=batch,
            input_length=seq_len, kernel_size=k, stride=1,
            padding=padding, dilation=dilation, groups=1,
            dtype=DEFAULT_DTYPE, return_output_dim=True,
        )
        if batch > 1:
            conv_kwargs["conv_config"] = ttnn.Conv2dConfig(
                weights_dtype=DEFAULT_DTYPE, act_block_h_override=32)
        result = ttnn.conv1d(**conv_kwargs)
        out, out_len = _conv1d_to_torch(result, out_ch, batch=batch)
        if b_torch is not None:
            out = out + b_torch.unsqueeze(0).unsqueeze(0)
        return out, out_len

    def _conv1d_fused(self, x_tt_host, w_tt, b_tt, in_ch, out_ch, k, seq_len,
                      dilation=1, fuse_relu=True):
        """Conv1d with fused LeakyReLU and native bias — minimal host roundtrips.

        Takes a ttnn host tensor, returns a ttnn host tensor.
        LeakyReLU is fused into the conv kernel (zero extra dispatch).
        Bias is handled by conv1d natively (zero extra dispatch).
        """
        padding = dilation * (k - 1) // 2
        config = _FUSED_LRELU_CONFIG if fuse_relu else None
        result = ttnn.conv1d(
            input_tensor=x_tt_host, weight_tensor=w_tt, device=self._device,
            in_channels=in_ch, out_channels=out_ch, batch_size=1,
            input_length=seq_len, kernel_size=k, stride=1,
            padding=padding, dilation=dilation, groups=1,
            dtype=DEFAULT_DTYPE, return_output_dim=True,
            bias_tensor=b_tt,
            conv_config=config,
        )
        out_tt = result[0]
        try:
            out_tt = ttnn.sharded_to_interleaved(out_tt)
        except RuntimeError:
            pass
        return ttnn.from_device(out_tt), result[1]

    def _resblock1(self, x_cf, block_idx, dilations, seq_len):
        """ResBlock1 — optimized with native conv1d bias.

        Keeps leaky_relu on host (torch) since device relu + roundtrip
        costs more than it saves. Uses conv1d native bias parameter
        to eliminate separate host bias add.
        """
        block = self._resblocks[block_idx]
        ch = block["channels"]

        for idx in range(3):
            d = dilations[idx]
            c1 = block["convs1"][idx]
            c2 = block["convs2"][idx]

            xt = F.leaky_relu(x_cf, LRELU_SLOPE)
            xt_cl = xt.permute(0, 2, 1).unsqueeze(1)  # [1, 1, T, C]
            xt_tt = ttnn.from_torch(xt_cl, dtype=DEFAULT_DTYPE)
            # conv1 fuses LeakyReLU on its output, replacing the host LRELU
            # that would otherwise sit between conv1 and conv2.
            xt_tt, _ = self._conv1d_fused(
                xt_tt, c1["w"], c1["b_tt"], ch, ch, c1["kernel"], seq_len,
                dilation=d, fuse_relu=True)
            xt = ttnn.to_torch(xt_tt).float().squeeze(1).permute(0, 2, 1)

            xt_cl = xt.permute(0, 2, 1).unsqueeze(1)
            xt_tt = ttnn.from_torch(xt_cl, dtype=DEFAULT_DTYPE)
            # conv2 keeps raw output — the following op is a residual add,
            # not another activation, so no fusion here.
            xt_tt, _ = self._conv1d_fused(
                xt_tt, c2["w"], c2["b_tt"], ch, ch, c2["kernel"], seq_len,
                fuse_relu=False)
            xt = ttnn.to_torch(xt_tt).float().squeeze(1).permute(0, 2, 1)

            x_cf = xt + x_cf
        return x_cf

    def _ensure_prepared_conv(self, w_host, b_host, in_ch, out_ch, k, seq_len, dilation, batch=1):
        """Lazy prepared-weight cache for Generator conv1d shapes.

        B=1: HEIGHT_SHARDED where it fits per-core L1 (40/48 shapes), else
        DEFAULT — tuned for single-stream RTF.

        B>1: pinned act_block_h_override=32 universally. ttnn.conv1d's
        auto-picker at batch>=2 statically allocates a CB region that
        clashes with prior L1 buffers (at byte 1118848) for 24/48 shapes
        (k>=7 at upsampled seq_len>=9000). Forcing act_block_h=32 keeps
        the CB small enough that all 48 shapes fit at B=2..8.

        Cache key includes batch — prepared weights bake batch_size in.
        """
        key = (id(w_host), in_ch, out_ch, k, seq_len, dilation, batch)
        cached = self._prep_cache.get(key)
        if cached is not None:
            return cached
        padding = dilation * (k - 1) // 2

        def _build(cfg):
            common = dict(
                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                in_channels=in_ch, out_channels=out_ch, batch_size=batch,
                input_height=1, input_width=seq_len,
                kernel_size=(1, k), stride=(1, 1),
                padding=(0, padding), dilation=(1, dilation), groups=1,
                device=self._device, input_dtype=DEFAULT_DTYPE,
                conv_config=cfg,
            )
            w_p = ttnn.prepare_conv_weights(weight_tensor=w_host, weights_format="OIHW",
                                              has_bias=True, **common)
            b_p = ttnn.prepare_conv_bias(bias_tensor=b_host, **common)
            return w_p, b_p, cfg

        if batch > 1:
            cfg = ttnn.Conv2dConfig(weights_dtype=DEFAULT_DTYPE,
                                     act_block_h_override=32)
            self._prep_cache[key] = _build(cfg)
            return self._prep_cache[key]

        # B=1 path: whitelist HEIGHT_SHARDED for shapes that fit.
        sharded_safe = not (in_ch >= 128 and k >= 11)
        if sharded_safe:
            try:
                cfg = ttnn.Conv2dConfig(weights_dtype=DEFAULT_DTYPE,
                                         shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
                self._prep_cache[key] = _build(cfg)
                return self._prep_cache[key]
            except Exception:
                pass
        cfg = ttnn.Conv2dConfig(weights_dtype=DEFAULT_DTYPE)
        self._prep_cache[key] = _build(cfg)
        return self._prep_cache[key]

    def _conv1d_device(self, x_in, w_tt, b_tt, in_ch, out_ch, k, seq_len,
                       dilation=1, batch=1):
        """Conv1d that returns a device tensor in interleaved DRAM.

        Used by ``_resblock1_device``. The input must already be on device
        in ROW_MAJOR layout — ``ttnn.conv1d`` at the HiFi-GAN ResBlock
        config (e.g. k=11/d=5/ch=128/seq=7200) rejects TILE inputs with
        ``program.cpp:1403: tt::exception``, regardless of storage or
        memory_config. The caller is responsible for the ROW_MAJOR
        relayout. Output is forced to interleaved DRAM because sharded
        L1 outputs accumulate bank pressure across chained calls and
        eventually break subsequent conv halo allocations.

        Weight + bias are passed through prepare_conv_weights/bias on first
        call per shape (cached) so the per-call internal tilization write
        is paid once at model warmup, not on every conv1d dispatch.
        """
        padding = dilation * (k - 1) // 2
        w_p, b_p, cfg = self._ensure_prepared_conv(w_tt, b_tt, in_ch, out_ch, k, seq_len, dilation, batch)
        result = ttnn.conv1d(
            input_tensor=x_in, weight_tensor=w_p, device=self._device,
            in_channels=in_ch, out_channels=out_ch, batch_size=batch,
            input_length=seq_len, kernel_size=k, stride=1,
            padding=padding, dilation=dilation, groups=1,
            dtype=DEFAULT_DTYPE, return_output_dim=True,
            bias_tensor=b_p,
            conv_config=cfg,
        )
        out_tt = result[0]
        try:
            out_tt = ttnn.sharded_to_interleaved(out_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        except RuntimeError:
            out_tt = ttnn.to_memory_config(out_tt, ttnn.DRAM_MEMORY_CONFIG)
        # conv1d may flatten [B,1,T,C] -> [1,1,B*T,C]; reshape back so
        # downstream eltwise ops broadcast against x_dev [B,1,T,C] correctly.
        if batch > 1:
            out_tt = ttnn.reshape(out_tt, (batch, 1, seq_len, out_ch))
        return out_tt

    def _resblock1_device(self, x_cf, block_idx, dilations, seq_len, batch=1):
        """ResBlock1 inner loop with device-resident activations.

        Sequence per dilation iteration:
            leaky_relu (device) -> to_layout(ROW_MAJOR) -> conv1 (device)
            -> leaky_relu (device) -> to_layout(ROW_MAJOR) -> conv2 (device)
            -> add(residual) (device)

        The ROW_MAJOR conversion between every leaky_relu and conv1d is
        mandatory: ``ttnn.conv1d`` rejects TILE-layout input at the
        (k=11, d=5, ch=128, seq=7200) configuration, while
        ``ttnn.leaky_relu`` requires TILE for its compute kernel. The
        per-step relayout is cheap on device and satisfies both.

        One ``from_torch`` at entry, one ``to_torch`` at exit, regardless
        of the three dilation iterations. All intermediates explicitly
        deallocated to keep L1/DRAM pressure stable across chained
        ResBlocks at long seq_len.
        """
        block = self._resblocks[block_idx]
        ch = block["channels"]

        x_nhwc = x_cf.permute(0, 2, 1).unsqueeze(1)
        x_dev = ttnn.from_torch(
            x_nhwc.float(),
            dtype=DEFAULT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=self._device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        for idx in range(3):
            d = dilations[idx]
            c1 = block["convs1"][idx]
            c2 = block["convs2"][idx]

            # leaky_relu reads TILE
            lr1 = ttnn.leaky_relu(x_dev, negative_slope=LRELU_SLOPE)
            # conv1d rejects TILE at the HiFi-GAN config -> relayout
            lr1_rm = ttnn.to_layout(lr1, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(lr1)
            c1_out = self._conv1d_device(
                lr1_rm, c1["w"], c1["b_tt"], ch, ch, c1["kernel"], seq_len,
                dilation=d, batch=batch)
            ttnn.deallocate(lr1_rm)

            # conv1d outputs TILE; leaky_relu accepts TILE
            lr2 = ttnn.leaky_relu(c1_out, negative_slope=LRELU_SLOPE)
            ttnn.deallocate(c1_out)
            lr2_rm = ttnn.to_layout(lr2, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(lr2)
            c2_out = self._conv1d_device(
                lr2_rm, c2["w"], c2["b_tt"], ch, ch, c2["kernel"], seq_len, batch=batch)
            ttnn.deallocate(lr2_rm)

            new_x = ttnn.add(x_dev, c2_out)
            ttnn.deallocate(c2_out)
            ttnn.deallocate(x_dev)
            x_dev = new_x

        out = ttnn.to_torch(x_dev).float().squeeze(1).permute(0, 2, 1)
        ttnn.deallocate(x_dev)
        return out

    def forward(self, z, har_source, g):
        """Execute full generator with persistent weights — TTNN-native B>=1.

        Args:
            z:          [B, 192, T] latent from flow decoder
            har_source: [B, 1, T*480] harmonic source from SineGen
            g:          [B, 256, 1] speaker embedding (rows usually identical
                        for the "convert N inputs to 1 target speaker"
                        pattern; cond cache assumes so and uses g[:1])

        Returns:
            audio: [B, 1, T*480] generated waveform in [-1, 1]
        """
        batch = z.shape[0]
        seq_len = z.shape[2]

        # conv_pre (persistent host weight)
        x_cl = z.permute(0, 2, 1)
        x_cl, _ = self._conv1d_persistent(
            x_cl, self._conv_pre_w, self._conv_pre_b, 192, 512, 7, seq_len, batch=batch)
        x_cf = x_cl.permute(0, 2, 1)

        # Conditioning projection. cond_cl depends only on g and is
        # constant across chunks — cache keyed by g[:1] so a batch of
        # same-target conversions reuses one linear pass.
        g_for_cond = g[:1] if batch > 1 else g
        if self._cond_cache_g is None or not torch.equal(g_for_cond, self._cond_cache_g):
            g_cl = g_for_cond.permute(0, 2, 1)
            g_tt = to_device(g_cl, self._device)
            cond_tt = ttnn.linear(g_tt, self._cond_w, bias=self._cond_b,
                                   memory_config=DEFAULT_MEMORY_CONFIG)
            cond_cl = to_host(cond_tt)[:1, :1, :512]
            ttnn.deallocate(g_tt)
            ttnn.deallocate(cond_tt)
            self._cond_cache_g = g_for_cond.clone()
            self._cond_cache_cl = cond_cl
        x_cf = x_cf + self._cond_cache_cl.permute(0, 2, 1)

        for i in range(NUM_UPSAMPLES):
            x_cf = F.leaky_relu(x_cf, LRELU_SLOPE)

            # Upsample (persistent TTNNConvTranspose1d)
            ct = self._ups[i]
            out_ch = UPSAMPLE_INITIAL_CH // (2 ** (i + 1))
            x_nhwc = x_cf.permute(0, 2, 1).unsqueeze(1)
            x_tt = ttnn.from_torch(x_nhwc.float(), dtype=DEFAULT_DTYPE)
            out_tt, out_len = ct(x_tt, batch_size=batch, input_length=seq_len)
            x_cf = TTNNConvTranspose1d.postprocess_output(out_tt, batch, out_len, out_ch)
            seq_len = out_len

            # Noise injection (torch — small ops)
            nc = self._noise_convs[i]
            if i < NUM_UPSAMPLES - 1:
                stride_f0 = math.prod(UPSAMPLE_RATES[i + 1:])
                x_source = F.conv1d(har_source, nc["w"], nc["b"],
                                     stride=stride_f0, padding=stride_f0 // 2)
            else:
                x_source = _linear_channel_first(har_source, nc["w"], nc["b"])
            x_cf = x_cf + x_source[:, :, :seq_len]

            # ResBlocks (persistent conv weights)
            xs = None
            for j in range(NUM_KERNELS):
                rb_idx = i * NUM_KERNELS + j
                rb_out = self._resblock1_device(x_cf, rb_idx, RESBLOCK_DILATIONS[j], seq_len, batch=batch)
                xs = rb_out if xs is None else xs + rb_out
            x_cf = xs / NUM_KERNELS

        # conv_post + tanh (persistent host weight)
        x_cf = F.leaky_relu(x_cf)
        x_cl = x_cf.permute(0, 2, 1)
        x_cl, _ = self._conv1d_persistent(
            x_cl, self._conv_post_w, None, 32, 1, 7, seq_len, batch=batch)
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
        # Drop the host-side conditioning cache so we don't hold a stale
        # cond_cl referencing weights that no longer exist on device.
        self._cond_cache_g = None
        self._cond_cache_cl = None
