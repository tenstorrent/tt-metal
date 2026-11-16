# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Qwen3-TTS speaker encoder (ECAPA-TDNN).

Reference: models/demos/audio/qwen3_tts/reference/qwen3_speaker_ref.py

Block boundary: input = log-mel [1, T, 128] (the STFT and mel filterbank stay on host,
neither being a TTNN op); output = speaker embedding [1, 2048], unnormalised, which the
talker consumes as one position of its prompt.

Everything here works time-major, [1, T, C] with channels last, which is the layout
`ttnn.conv1d` wants. The reference works channel-first, so each intermediate is the
transpose of its counterpart there.

    mel [1,T,128]
      -> blocks.0   TDNN k5 d1      128 -> 512
      -> blocks.1-3 SE-Res2Net      512 -> 512, k3, dilation 2/3/4
      -> concat blocks 1..3                 -> 1536
      -> mfa        1x1             1536 -> 1536
      -> asp        attentive statistics pooling -> 3072
      -> fc         1x1             3072 -> 2048

Three implementation notes, each a deliberate departure from a literal transcription:

  1. **Reflect padding is built by hand.** Every upstream convolution is
     `padding="same", padding_mode="reflect"`, and `ttnn.conv1d` takes zero padding only.
     Kernel-1 convolutions are unaffected; the rest are padded here by slicing mirrored
     columns and concatenating. Pad widths are 2 to 4 columns, so the cost is small.
  2. **1x1 convolutions run as matmuls.** A kernel-1 convolution over [1, T, C] is a
     pointwise projection, so it goes through `ttnn.linear` rather than the convolution
     path. That covers every layer except `blocks.0` and the 21 grouped Res2Net
     convolutions.
  3. **Attentive statistics pooling folds the concatenation into its weights.** Upstream
     concatenates the features with a broadcast mean and std, then applies a 1x1
     convolution over 3*1536 channels. A linear map over a concatenation is the sum of the
     linear maps over its parts, so the weight is split three ways and the [1, T, 4608]
     intermediate is never built.

Pooling and the output projection accumulate in fp32: upstream clamps the variance at
1e-12 before the square root, which bf16 cannot represent.
"""

import torch

import ttnn
from models.demos.audio.qwen3_tts import weights as checkpoint

VARIANCE_EPS = 1e-12  # AttentiveStatisticsPooling.eps


def _compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _same_padding(kernel, dilation):
    """Columns to add on each side so a dilated convolution keeps its length."""
    return dilation * (kernel - 1) // 2


def preprocess_speaker_parameters(device, config=None, state=None, dtype=ttnn.bfloat16):
    """Checkpoint weights -> the tensors the encoder runs with.

    Convolution weights stay on host in row-major: `ttnn.conv1d` prepares them for the
    parallelisation it picks, and the prepared result is cached on first use.
    """
    cfg = dict(config or checkpoint.speaker_encoder_config())
    state = checkpoint.load_speaker_state() if state is None else state

    channels = cfg["enc_channels"]
    kernels = cfg["enc_kernel_sizes"]
    dilations = cfg["enc_dilations"]
    scale = cfg["enc_res2net_scale"]

    def conv_weight(name):
        """Conv1d weight [out, in, k], left on host for conv1d to prepare."""
        return ttnn.from_torch(state[f"{name}.weight"], dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def conv_bias(name):
        return ttnn.from_torch(state[f"{name}.bias"].reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    def linear_weight(tensor):
        """[out, in] -> device [in, out] for ttnn.linear."""
        return ttnn.from_torch(
            tensor.t().contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def pointwise(name):
        """A kernel-1 Conv1d [out, in, 1] as a matmul plus bias."""
        return {
            "weight": linear_weight(state[f"{name}.weight"].squeeze(-1)),
            "bias": ttnn.from_torch(
                state[f"{name}.bias"].reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
            ),
        }

    params = {
        "config": cfg,
        "stem": {
            "weight": conv_weight("blocks.0.conv"),
            "bias": conv_bias("blocks.0.conv"),
            "in_channels": cfg["mel_dim"],
            "out_channels": channels[0],
            "kernel": kernels[0],
            "dilation": dilations[0],
        },
        "blocks": [],
    }

    for i in range(1, len(channels) - 1):
        name = f"blocks.{i}"
        width = channels[i] // scale
        params["blocks"].append(
            {
                "index": i,
                "width": width,
                "scale": scale,
                "channels": channels[i],
                "kernel": kernels[i],
                "dilation": dilations[i],
                "tdnn1": pointwise(f"{name}.tdnn1.conv"),
                "tdnn2": pointwise(f"{name}.tdnn2.conv"),
                "se1": pointwise(f"{name}.se_block.conv1"),
                "se2": pointwise(f"{name}.se_block.conv2"),
                "res2net": [
                    {
                        "weight": conv_weight(f"{name}.res2net_block.blocks.{j}.conv"),
                        "bias": conv_bias(f"{name}.res2net_block.blocks.{j}.conv"),
                    }
                    for j in range(scale - 1)
                ],
            }
        )

    params["mfa"] = pointwise("mfa.conv")

    # Split the pooling projection across [features | mean | std] (note 3 above). The
    # upstream weight is [attention, 3 * channels, 1]; each third multiplies one part.
    attention_weight = state["asp.tdnn.conv.weight"].squeeze(-1)
    width = channels[-1]
    params["asp"] = {
        "features": linear_weight(attention_weight[:, 0:width]),
        "mean": linear_weight(attention_weight[:, width : 2 * width]),
        "std": linear_weight(attention_weight[:, 2 * width : 3 * width]),
        "bias": ttnn.from_torch(
            state["asp.tdnn.conv.bias"].reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        ),
        "score": pointwise("asp.conv"),
    }
    params["fc"] = pointwise("fc")
    return params


class TtSpeakerEncoder:
    def __init__(self, device, parameters):
        self.device = device
        self.p = parameters
        self.cfg = parameters["config"]
        self.compute_config = _compute_config(device)
        self.conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16)
        self._prepared = {}  # conv1d prepares weights for its chosen parallelisation; reuse them

    # ── primitives ──────────────────────────────────────────────────────────

    def _reflect_pad(self, x, pad):
        """[1, T, C] -> [1, T + 2*pad, C], mirroring torch's reflect padding on time.

        There is no flip op, so the mirrored columns are sliced one at a time. Reflect
        excludes the edge sample itself: the left block is x[pad] .. x[1] and the right
        block is x[T-2] .. x[T-1-pad].
        """
        if pad == 0:
            return x
        _, length, width = x.shape
        if pad >= length:
            raise ValueError(f"reflect padding of {pad} needs more than {length} frames")

        parts = [ttnn.slice(x, [0, i, 0], [1, i + 1, width]) for i in range(pad, 0, -1)]
        parts.append(x)
        parts.extend(ttnn.slice(x, [0, length - 2 - j, 0], [1, length - 1 - j, width]) for j in range(pad))
        return ttnn.concat(parts, dim=1)

    def _conv1d(self, x, params, cache_key, in_channels, out_channels, kernel, dilation):
        """Same-length dilated convolution over time, with upstream's reflect padding."""
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = self._reflect_pad(x, _same_padding(kernel, dilation))
        length = x.shape[1]

        # conv1d prepares the weight for the parallelisation it picks, which depends on the
        # input length; a weight prepared at one length decodes to garbage at another
        # without raising. The length is therefore part of the key.
        cache_key = (cache_key, length)
        weight, bias = self._prepared.get(cache_key, (params["weight"], params["bias"]))
        out, out_length, (weight, bias) = ttnn.conv1d(
            input_tensor=ttnn.reshape(x, (1, length, 1, in_channels)),
            weight_tensor=weight,
            bias_tensor=bias,
            device=self.device,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=1,
            input_length=length,
            kernel_size=kernel,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=1,
            dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        self._prepared[cache_key] = (weight, bias)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.reshape(out, (1, out_length, out_channels))

    def _linear(self, x, params):
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return ttnn.linear(x, params["weight"], bias=params["bias"], compute_kernel_config=self.compute_config)

    def _tdnn_pointwise(self, x, params):
        """TimeDelayNetBlock with kernel 1: a projection followed by ReLU."""
        return ttnn.relu(self._linear(x, params))

    # ── blocks ──────────────────────────────────────────────────────────────

    def _res2net(self, x, block):
        """Split into `scale` channel groups and convolve all but the first, in sequence.

        Group k reads group k plus the previous group's output, which is what makes this a
        serial chain rather than one grouped convolution.
        """
        width = block["width"]
        length = x.shape[1]
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        outputs = []
        previous = None
        for group in range(block["scale"]):
            part = ttnn.slice(x, [0, 0, group * width], [1, length, (group + 1) * width])
            if group == 0:
                current = part
            else:
                if group > 1:
                    part = ttnn.add(ttnn.to_layout(part, ttnn.TILE_LAYOUT), ttnn.to_layout(previous, ttnn.TILE_LAYOUT))
                current = self._conv1d(
                    part,
                    block["res2net"][group - 1],
                    f"res2net.{block['index']}.{group - 1}",
                    width,
                    width,
                    block["kernel"],
                    block["dilation"],
                )
                current = ttnn.relu(ttnn.to_layout(current, ttnn.TILE_LAYOUT))
            outputs.append(current)
            previous = current

        outputs = [ttnn.to_layout(part, ttnn.ROW_MAJOR_LAYOUT) for part in outputs]
        return ttnn.concat(outputs, dim=2)

    def _squeeze_excitation(self, x, block):
        """Channel gating from the time-averaged signal."""
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        pooled = ttnn.mean(x, dim=1, keepdim=True)  # [1, 1, C]
        gate = ttnn.relu(self._linear(pooled, block["se1"]))
        gate = ttnn.sigmoid(self._linear(gate, block["se2"]))
        return ttnn.multiply(x, gate)  # broadcasts over time

    def _se_res2net_block(self, x, block):
        residual = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        out = self._tdnn_pointwise(x, block["tdnn1"])
        out = self._res2net(out, block)
        out = self._tdnn_pointwise(out, block["tdnn2"])
        out = self._squeeze_excitation(out, block)
        return ttnn.add(out, residual)

    def _attentive_statistics_pooling(self, x):
        """Weighted mean and std over time, with the weights predicted from the signal."""
        params = self.p["asp"]
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        # Uniform statistics over the whole utterance. Upstream builds a length mask, but
        # at batch 1 with no padding it is all ones, so the weights reduce to 1/T and its
        # masked_fill of -inf never fires.
        mean = ttnn.mean(x, dim=1, keepdim=True)
        centered = ttnn.subtract(x, mean)
        variance = ttnn.mean(ttnn.multiply(centered, centered), dim=1, keepdim=True)
        std = ttnn.sqrt(ttnn.clamp(variance, min=VARIANCE_EPS))

        scores = ttnn.linear(x, params["features"], bias=params["bias"], compute_kernel_config=self.compute_config)
        scores = ttnn.add(scores, ttnn.matmul(mean, params["mean"], compute_kernel_config=self.compute_config))
        scores = ttnn.add(scores, ttnn.matmul(std, params["std"], compute_kernel_config=self.compute_config))
        scores = ttnn.tanh(ttnn.relu(scores))
        scores = self._linear(scores, params["score"])  # [1, T, C]

        # Softmax over time. Written out rather than calling ttnn.softmax, which reduces
        # over the last dimension; time is the middle one in this layout.
        shifted = ttnn.subtract(scores, ttnn.max(scores, dim=1, keepdim=True))
        exponentiated = ttnn.exp(shifted)
        attention = ttnn.divide(exponentiated, ttnn.sum(exponentiated, dim=1, keepdim=True))

        weighted_mean = ttnn.sum(ttnn.multiply(attention, x), dim=1, keepdim=True)
        deviation = ttnn.subtract(x, weighted_mean)
        weighted_variance = ttnn.sum(ttnn.multiply(attention, ttnn.multiply(deviation, deviation)), dim=1, keepdim=True)
        weighted_std = ttnn.sqrt(ttnn.clamp(weighted_variance, min=VARIANCE_EPS))
        return ttnn.concat([weighted_mean, weighted_std], dim=2)  # [1, 1, 2C]

    # ── forward ─────────────────────────────────────────────────────────────

    def __call__(self, mel, return_intermediates=False):
        """mel [1, T, 128] -> embedding [1, 2048]."""
        intermediates = {}
        stem = self.p["stem"]
        x = self._conv1d(
            mel,
            stem,
            "stem",
            stem["in_channels"],
            stem["out_channels"],
            stem["kernel"],
            stem["dilation"],
        )
        x = ttnn.relu(ttnn.to_layout(x, ttnn.TILE_LAYOUT))
        intermediates["blocks.0"] = x

        outputs = []
        for block in self.p["blocks"]:
            x = self._se_res2net_block(x, block)
            intermediates[f"blocks.{block['index']}"] = x
            outputs.append(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT))

        x = ttnn.concat(outputs, dim=2)
        x = self._tdnn_pointwise(x, self.p["mfa"])
        intermediates["mfa"] = x

        pooled = self._attentive_statistics_pooling(x)
        intermediates["asp"] = pooled

        embedding = self._linear(pooled, self.p["fc"])
        intermediates["fc"] = embedding
        embedding = ttnn.reshape(embedding, (1, self.cfg["enc_dim"]))

        if return_intermediates:
            return embedding, intermediates
        return embedding


def speaker_embedding(device, mel, parameters=None):
    """One-shot helper: host mel [1, T, 128] -> host embedding [1, 2048]."""
    parameters = parameters or preprocess_speaker_parameters(device)
    encoder = TtSpeakerEncoder(device, parameters)
    mel_tt = ttnn.from_torch(mel, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return ttnn.to_torch(encoder(mel_tt)).to(torch.float32)
