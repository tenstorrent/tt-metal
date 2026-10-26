# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Qwen3-TTS codec decoder: codes in, waveform out.

Reference: models/demos/audio/qwen3_tts/reference/qwen3_codec_ref.py

    codes [1, 16, T]
      -> quantizer.decode                          [1, T, 512]
      -> pre_conv, causal k=3                      [1, T, 1024]
      -> pre_transformer, 8 layers at hidden 512   [1, T, 1024]
      -> 2 x (transposed conv stride 2 + ConvNeXt) [1, 4T, 1024]
      -> conv k=7                                  [1, 4T, 1536]
      -> 4 decoder blocks, rates 8, 5, 4, 3        [1, 1920T, 96]
      -> SnakeBeta, conv k=7, clamp                [1, 1920T, 1]

Everything works time-major, [1, T, C] with channels last, which is what the convolutions
want. The reference works channel-first, so each intermediate is the transpose of its
counterpart there.

Five things worth knowing about this block:

  1. **Every convolution is causal.** Upstream pads on the left by the whole receptive
     field and adds whatever right padding the stride needs, so no output depends on a
     future sample. That is what lets the codec stream, and it means the padding cannot be
     treated as a symmetric "same".
  2. **Transposed convolutions run through `ttnn.conv_transpose2d`** with height 1, since
     there is no 1D form. Validated against `torch.nn.ConvTranspose1d` at PCC 0.99998.
     Upstream trims `kernel - stride` samples off the right afterwards.
  3. **SnakeBeta folds its exponentials into the weights.** Upstream computes
     `x + 1/(exp(beta) + 1e-9) * sin(x * exp(alpha))^2` every call; `exp(alpha)` and
     `1/(exp(beta) + 1e-9)` depend only on parameters, so both are computed once at load.
  4. **The quantizer's codebooks are a quotient.** They are stored as `embedding_sum` and
     `cluster_usage` rather than a table, so the usable codebook is built at load. Lookup
     stays on host: it is a gather, and the result is 512 wide against a 2048-entry table.
  5. **Attention is windowed.** Each position sees at most `sliding_window` (72) previous
     positions. Below that length the mask is plainly causal, which is why a short test
     would not exercise the window at all.

Correctness first: nothing traced or sharded, and the whole clip decodes in one pass rather
than the chunked path upstream uses for long audio.
"""

import math

import torch

import ttnn
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import MASK_FILL, _compute_config

SNAKE_EPS = 1e-9
# Frame counts the decoder rounds up to. Every distinct length compiles its own convolution
# programs, and tt-metal keeps each one's L1_SMALL scratch for the life of the device: three
# different lengths filled the 64 KB region and the next block to ask for scratch failed.
# Rounding up to a multiple of this decodes a few frames of padding, which the causal
# convolutions let us trim off exactly, and keeps the program count flat.
LENGTH_BUCKET = 32
# `EuclideanCodebook.epsilon`: the floor on a codebook's running count before it divides.
# Every count in this checkpoint is at least 0.022, so it never binds; it is here because
# the quotient is upstream's and the constant should say which one.
CODEBOOK_EPS = 1e-5


# ── host-side tables ────────────────────────────────────────────────────────


def windowed_causal_mask(length, window):
    """Additive mask [1, 1, T, T]: a position sees itself and `window - 1` before it."""
    positions = torch.arange(length)
    distance = positions.reshape(-1, 1) - positions.reshape(1, -1)
    allowed = (distance >= 0) & (distance < window)
    return torch.where(allowed, 0.0, MASK_FILL).reshape(1, 1, length, length)


def rotary_tables(config, length):
    """Plain RoPE cos and sin as [1, 1, T, head_dim] for the codec's transformer."""
    head_dim = config["head_dim"]
    inverse = 1.0 / (
        config["rope_theta"] ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(length, dtype=torch.float32).reshape(1, 1, length)
    frequencies = (inverse.reshape(1, -1, 1) @ positions).transpose(1, 2)
    embedded = torch.cat((frequencies, frequencies), dim=-1)
    return embedded.cos().unsqueeze(1), embedded.sin().unsqueeze(1)


def codebooks(state, config):
    """Usable codebooks per quantizer, built from `embedding_sum` and `cluster_usage`.

    Mirrors `EuclideanCodebook`, which divides the running sum by the running count. The
    first quantizer is the semantic one and the remaining 15 are acoustic, stored under
    separate prefixes upstream.
    """
    tables = []
    for prefix, count in (("quantizer.rvq_first", 1), ("quantizer.rvq_rest", config["num_quantizers"] - 1)):
        for index in range(count):
            base = f"{prefix}.vq.layers.{index}._codebook"
            usage = state[f"{base}.cluster_usage"].reshape(-1, 1)
            tables.append(state[f"{base}.embedding_sum"] / usage.clamp(min=CODEBOOK_EPS))
    return tables


def quantizer_decode(codes, state, config):
    """codes [1, 16, T] -> latents [1, T, 512], on host.

    Upstream sums each quantizer's residual, projects the semantic and acoustic halves
    separately, then adds them. Both projections are 1x1 convolutions, so they are matmuls.
    """
    tables = codebooks(state, config)
    semantic_count = config["num_semantic_quantizers"]

    def residual_sum(indices):
        total = None
        for index in indices:
            vectors = tables[index][codes[0, index]]  # [T, 256]
            total = vectors if total is None else total + vectors
        return total

    def project(latents, prefix):
        # input_proj/output_proj are Conv1d with kernel 1; only output_proj is used here.
        weight = state[f"{prefix}.output_proj.weight"].squeeze(-1)  # [512, 256]
        return latents @ weight.t()

    semantic = project(residual_sum(range(semantic_count)), "quantizer.rvq_first")
    acoustic = project(residual_sum(range(semantic_count, config["num_quantizers"])), "quantizer.rvq_rest")
    return (semantic + acoustic).unsqueeze(0)


# ── shared with the encoder ─────────────────────────────────────────────────
#
# Both halves of the codec convolve the same way, so the padding arithmetic and the
# length-keyed weight cache live here and `ttnn_qwen3_codec_encoder` imports them.


def conv_parameters(state, name, dtype=ttnn.bfloat16, depthwise=False):
    """Conv1d weight [out, in/groups, k] plus bias, left on host for conv1d to prepare."""
    weight = state[f"{name}.weight"]
    bias = state.get(f"{name}.bias")
    return {
        "weight": ttnn.from_torch(weight, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
        "bias": (
            None
            if bias is None
            else ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        ),
        "out_channels": weight.shape[0],
        "in_channels": weight.shape[1] * (weight.shape[0] if depthwise else 1),
        "kernel": weight.shape[2],
        "groups": weight.shape[0] if depthwise else 1,
    }


def replicate_pad(x, left, right):
    """[1, T, C] -> [1, left + T + right, C], repeating the edge frames.

    `torch.nn.functional.pad(mode="replicate")`, built by hand: `ttnn.pad` fills with a
    constant only. One convolution in the codec needs this, the encoder's `downsample`,
    and zero padding there costs 0.009 of PCC on the tensor the codes come from.
    """
    if not left and not right:
        return x
    _, length, width = x.shape
    parts = [ttnn.slice(x, [0, 0, 0], [1, 1, width])] * left
    parts.append(x)
    parts.extend([ttnn.slice(x, [0, length - 1, 0], [1, length, width])] * right)
    return ttnn.concat(parts, dim=1)


def causal_conv1d(
    device,
    x,
    params,
    prepared,
    key,
    compute_config,
    conv_config,
    stride=1,
    dilation=1,
    pad_mode="constant",
    dtype=ttnn.bfloat16,
):
    """Upstream's causal convolution: pad left by the receptive field, then convolve.

    `_get_extra_padding_for_conv1d` adds whatever the stride needs on the right so the
    output length comes out as ceil. Both paddings are applied here explicitly, since
    `ttnn.conv1d` takes a symmetric amount and this is deliberately asymmetric.

    `pad_mode` follows the module's own: every convolution in this codec pads with zeros
    except the encoder's `downsample`, which replicates.

    `prepared` is the caller's weight cache, keyed by name *and padded length*: `ttnn.conv1d`
    prepares the weight for the parallelisation it picks, and that depends on the input
    length. A weight prepared at one length convolves to garbage at another without raising.
    Measured on the decoder: one instance reused across 4 and 8 frames took the second from
    0.995 to 0.104.
    """
    kernel = (params["kernel"] - 1) * dilation + 1
    left = kernel - stride
    length = x.shape[1]
    frames = (length - kernel + left) / stride + 1
    ideal = (math.ceil(frames) - 1) * stride + (kernel - left)
    right = max(0, ideal - length)

    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    if pad_mode == "replicate":
        x = replicate_pad(x, left, right)
    elif left or right:
        x = ttnn.pad(x, [(0, 0), (left, right), (0, 0)], value=0.0)
    padded = x.shape[1]

    cache_key = (key, padded)
    weight, bias = prepared.get(cache_key, (params["weight"], params["bias"]))
    out, out_length, (weight, bias) = ttnn.conv1d(
        input_tensor=ttnn.reshape(x, (1, padded, 1, params["in_channels"])),
        weight_tensor=weight,
        bias_tensor=bias,
        device=device,
        in_channels=params["in_channels"],
        out_channels=params["out_channels"],
        batch_size=1,
        input_length=padded,
        kernel_size=params["kernel"],
        stride=stride,
        padding=0,
        dilation=dilation,
        groups=params["groups"],
        dtype=dtype,
        conv_config=conv_config,
        compute_config=compute_config,
        return_output_dim=True,
        return_weights_and_bias=True,
    )
    prepared[cache_key] = (weight, bias)
    out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(out, (1, out_length, params["out_channels"]))


# ── parameters ──────────────────────────────────────────────────────────────


def preprocess_codec_parameters(device, config=None, state=None, dtype=ttnn.bfloat16):
    """Checkpoint weights -> the tensors the decoder runs with."""
    cfg = dict(config or checkpoint.codec_decoder_config())
    state = checkpoint.load_codec_decoder_state() if state is None else state

    def conv(name, depthwise=False):
        return conv_parameters(state, name, dtype=dtype, depthwise=depthwise)

    def trans_conv(name):
        """ConvTranspose1d weight [in, out, k] -> conv_transpose2d's (C, O/G, 1, K)."""
        weight = state[f"{name}.weight"]
        return {
            "weight": ttnn.from_torch(weight.unsqueeze(2), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT),
            "bias": ttnn.from_torch(
                state[f"{name}.bias"].reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            "in_channels": weight.shape[0],
            "out_channels": weight.shape[1],
            "kernel": weight.shape[2],
        }

    def linear(name, transpose=True):
        tensor = state[f"{name}.weight"]
        tensor = tensor.t().contiguous() if transpose else tensor.contiguous()
        return ttnn.from_torch(tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def vector(name, transform=None):
        tensor = state[name]
        if transform is not None:
            tensor = transform(tensor)
        return ttnn.from_torch(tensor.reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def snake(name):
        """Fold the exponentials in: they depend only on the parameters (note 3 above)."""
        return {
            "alpha": vector(f"{name}.alpha", torch.exp),
            "beta_inverse": vector(f"{name}.beta", lambda b: 1.0 / (torch.exp(b) + SNAKE_EPS)),
        }

    def convnext(name):
        return {
            "dwconv": conv(f"{name}.dwconv.conv", depthwise=True),
            "norm_weight": vector(f"{name}.norm.weight"),
            "norm_bias": vector(f"{name}.norm.bias"),
            "pwconv1": linear(f"{name}.pwconv1"),
            "pwconv1_bias": vector(f"{name}.pwconv1.bias"),
            "pwconv2": linear(f"{name}.pwconv2"),
            "pwconv2_bias": vector(f"{name}.pwconv2.bias"),
            "gamma": vector(f"{name}.gamma"),
        }

    def residual_unit(name):
        return {
            "act1": snake(f"{name}.act1"),
            "conv1": conv(f"{name}.conv1.conv"),
            "act2": snake(f"{name}.act2"),
            "conv2": conv(f"{name}.conv2.conv"),
        }

    transformer = {
        "input_proj": linear("pre_transformer.input_proj"),
        "input_proj_bias": vector("pre_transformer.input_proj.bias"),
        "output_proj": linear("pre_transformer.output_proj"),
        "output_proj_bias": vector("pre_transformer.output_proj.bias"),
        "norm": vector("pre_transformer.norm.weight"),
        "layers": [
            {
                "input_layernorm": vector(f"pre_transformer.layers.{index}.input_layernorm.weight"),
                "post_attention_layernorm": vector(f"pre_transformer.layers.{index}.post_attention_layernorm.weight"),
                "q_proj": linear(f"pre_transformer.layers.{index}.self_attn.q_proj"),
                "k_proj": linear(f"pre_transformer.layers.{index}.self_attn.k_proj"),
                "v_proj": linear(f"pre_transformer.layers.{index}.self_attn.v_proj"),
                "o_proj": linear(f"pre_transformer.layers.{index}.self_attn.o_proj"),
                "attn_scale": vector(f"pre_transformer.layers.{index}.self_attn_layer_scale.scale"),
                "gate_proj": linear(f"pre_transformer.layers.{index}.mlp.gate_proj"),
                "up_proj": linear(f"pre_transformer.layers.{index}.mlp.up_proj"),
                "down_proj": linear(f"pre_transformer.layers.{index}.mlp.down_proj"),
                "mlp_scale": vector(f"pre_transformer.layers.{index}.mlp_layer_scale.scale"),
            }
            for index in range(cfg["num_hidden_layers"])
        ],
    }

    blocks = []
    for index, rate in enumerate(cfg["upsample_rates"]):
        prefix = f"decoder.{index + 1}.block"
        blocks.append(
            {
                "rate": rate,
                "act": snake(f"{prefix}.0"),
                "trans_conv": trans_conv(f"{prefix}.1.conv"),
                "units": [residual_unit(f"{prefix}.{2 + unit}") for unit in range(3)],
            }
        )

    return {
        "config": cfg,
        "state": state,  # the quantizer tables stay on host
        "pre_conv": conv("pre_conv.conv"),
        "transformer": transformer,
        "upsample": [
            {"trans_conv": trans_conv(f"upsample.{index}.0.conv"), "convnext": convnext(f"upsample.{index}.1")}
            for index in range(len(cfg["upsampling_ratios"]))
        ],
        "head_conv": conv("decoder.0.conv"),
        "blocks": blocks,
        "final_act": snake(f"decoder.{len(cfg['upsample_rates']) + 1}"),
        "final_conv": conv(f"decoder.{len(cfg['upsample_rates']) + 2}.conv"),
    }


class TtCodecDecoder:
    def __init__(self, device, parameters):
        self.device = device
        self.p = parameters
        self.config = parameters["config"]
        self.compute_config = _compute_config(device)
        # Config tensors in DRAM: in L1_SMALL, Wormhole hung at 64 frames and up (README).
        self.conv_config = ttnn.Conv1dConfig(weights_dtype=ttnn.bfloat16, config_tensors_in_dram=True)
        self.trans_conv_config = ttnn.Conv2dConfig(config_tensors_in_dram=True)
        self.heads = self.config["num_attention_heads"]
        self.kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.config["head_dim"]
        self.scale = self.head_dim**-0.5
        # upsampling_ratios then upsample_rates: 4 x 480, so one frame is 1920 samples.
        self.upsample = int(
            torch.tensor(self.config["upsample_rates"] + self.config["upsampling_ratios"], dtype=torch.long).prod()
        )
        self._prepared = {}
        self._compiled = set()  # lengths whose programs are resident; see `forget_programs`

    # ── program room ────────────────────────────────────────────────────────

    def padded_frames(self, frames, bucket=LENGTH_BUCKET):
        """What `decode` will actually run: `frames` rounded up to a bucket."""
        return frames if bucket <= 1 else -(-frames // bucket) * bucket

    def program_room_needed(self, frames, bucket=LENGTH_BUCKET):
        """Does this decode need the device's program cache dropped before it runs?

        True when the length is one the decoder has not compiled since the last drop. Each frame
        count holds its own L1_SMALL scratch until then: 16, 32, 50, 58 KB after four lengths,
        then a failed allocation. A new length gets an empty region rather than a spare-capacity
        check, since the footprint is neither constant nor proportional to the length.

        The caller drops the cache, not this object: no captured trace may be live when the
        programs it was built from go away.
        """
        return bool(self._compiled) and self.padded_frames(frames, bucket) not in self._compiled

    def forget_programs(self):
        """Forget which lengths are resident, after the caller cleared the program cache."""
        self._compiled.clear()

    # ── primitives ──────────────────────────────────────────────────────────

    def _causal_conv(self, x, params, key, stride=1, dilation=1):
        return causal_conv1d(
            self.device, x, params, self._prepared, key, self.compute_config, self.conv_config, stride, dilation
        )

    def _trans_conv(self, x, params, key, stride):
        """Transposed convolution, then upstream's right trim of `kernel - stride`."""
        length = x.shape[1]
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        out, [_, width] = ttnn.conv_transpose2d(
            input_tensor=ttnn.reshape(x, (1, 1, length, params["in_channels"])),
            weight_tensor=params["weight"],
            bias_tensor=params["bias"],
            device=self.device,
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
            batch_size=1,
            input_height=1,
            input_width=length,
            kernel_size=(1, params["kernel"]),
            stride=(1, stride),
            padding=(0, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            conv_config=self.trans_conv_config,
            return_output_dim=True,
        )
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (1, width, params["out_channels"]))
        trim = params["kernel"] - stride
        if trim > 0:
            out = ttnn.slice(out, [0, 0, 0], [1, width - trim, params["out_channels"]])
        return out

    def _snake(self, x, params):
        """x + (1/exp(beta)) * sin(x * exp(alpha))^2, with the exponentials pre-folded."""
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        wave = ttnn.sin(ttnn.multiply(x, params["alpha"]))
        return ttnn.add(x, ttnn.multiply(ttnn.multiply(wave, wave), params["beta_inverse"]))

    # ── blocks ──────────────────────────────────────────────────────────────

    def _convnext(self, x, params, key):
        residual = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        out = self._causal_conv(x, params["dwconv"], f"{key}.dwconv")
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        out = ttnn.layer_norm(out, weight=params["norm_weight"], bias=params["norm_bias"], epsilon=1e-6)
        out = ttnn.linear(
            out, params["pwconv1"], bias=params["pwconv1_bias"], compute_kernel_config=self.compute_config
        )
        out = ttnn.gelu(out)
        out = ttnn.linear(
            out, params["pwconv2"], bias=params["pwconv2_bias"], compute_kernel_config=self.compute_config
        )
        return ttnn.add(residual, ttnn.multiply(out, params["gamma"]))

    def _residual_unit(self, x, params, key, dilation):
        residual = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        out = self._snake(x, params["act1"])
        out = self._causal_conv(out, params["conv1"], f"{key}.conv1", dilation=dilation)
        out = self._snake(out, params["act2"])
        out = self._causal_conv(out, params["conv2"], f"{key}.conv2")
        return ttnn.add(ttnn.to_layout(out, ttnn.TILE_LAYOUT), residual)

    def _attention(self, x, params, cos, sin, mask, length):
        def split(tensor, heads):
            return ttnn.permute(ttnn.reshape(tensor, (1, length, heads, self.head_dim)), (0, 2, 1, 3))

        query = split(ttnn.linear(x, params["q_proj"], compute_kernel_config=self.compute_config), self.heads)
        key = split(ttnn.linear(x, params["k_proj"], compute_kernel_config=self.compute_config), self.kv_heads)
        value = split(ttnn.linear(x, params["v_proj"], compute_kernel_config=self.compute_config), self.kv_heads)

        def rotate(tensor):
            half = self.head_dim // 2
            shape = tensor.shape
            first = ttnn.slice(tensor, [0, 0, 0, 0], [shape[0], shape[1], shape[2], half])
            second = ttnn.slice(tensor, [0, 0, 0, half], [shape[0], shape[1], shape[2], self.head_dim])
            spun = ttnn.concat([ttnn.neg(second), first], dim=-1)
            return ttnn.add(ttnn.multiply(tensor, cos), ttnn.multiply(spun, sin))

        query, key = rotate(query), rotate(key)
        if self.heads != self.kv_heads:
            repeats = self.heads // self.kv_heads
            key = ttnn.repeat_interleave(key, repeats, dim=1)
            value = ttnn.repeat_interleave(value, repeats, dim=1)

        scores = ttnn.matmul(query, ttnn.permute(key, (0, 1, 3, 2)), compute_kernel_config=self.compute_config)
        scores = ttnn.add(ttnn.multiply(scores, self.scale), mask)
        weights = ttnn.softmax(scores, dim=-1, compute_kernel_config=self.compute_config)

        attended = ttnn.matmul(weights, value, compute_kernel_config=self.compute_config)
        attended = ttnn.reshape(ttnn.permute(attended, (0, 2, 1, 3)), (1, length, self.heads * self.head_dim))
        return ttnn.linear(attended, params["o_proj"], compute_kernel_config=self.compute_config)

    def _transformer(self, x, cos, sin, mask):
        params = self.p["transformer"]
        length = x.shape[1]
        eps = self.config["rms_norm_eps"]

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.linear(
            x, params["input_proj"], bias=params["input_proj_bias"], compute_kernel_config=self.compute_config
        )

        for layer in params["layers"]:
            normed = ttnn.rms_norm(x, weight=layer["input_layernorm"], epsilon=eps)
            attended = self._attention(normed, layer, cos, sin, mask, length)
            x = ttnn.add(x, ttnn.multiply(attended, layer["attn_scale"]))

            normed = ttnn.rms_norm(x, weight=layer["post_attention_layernorm"], epsilon=eps)
            gate = ttnn.linear(normed, layer["gate_proj"], compute_kernel_config=self.compute_config)
            up = ttnn.linear(normed, layer["up_proj"], compute_kernel_config=self.compute_config)
            projected = ttnn.linear(
                ttnn.multiply(ttnn.silu(gate), up), layer["down_proj"], compute_kernel_config=self.compute_config
            )
            x = ttnn.add(x, ttnn.multiply(projected, layer["mlp_scale"]))

        x = ttnn.rms_norm(x, weight=params["norm"], epsilon=eps)
        return ttnn.linear(
            x, params["output_proj"], bias=params["output_proj_bias"], compute_kernel_config=self.compute_config
        )

    # ── forward ─────────────────────────────────────────────────────────────

    def host_inputs(self, length):
        """Rotation tables and the windowed mask for a prompt of this many frames."""
        cos, sin = rotary_tables(self.config, length)
        return cos, sin, windowed_causal_mask(length, self.config["sliding_window"])

    def latents(self, codes):
        """codes [1, 16, T] -> latents [1, T, 512] on host."""
        return quantizer_decode(codes, self.p["state"], self.config)

    def __call__(self, latents, cos, sin, mask, return_intermediates=False):
        """latents [1, T, 512] -> waveform [1, 1920 * T, 1]."""
        intermediates = {}

        x = self._causal_conv(latents, self.p["pre_conv"], "pre_conv")
        if return_intermediates:
            intermediates["pre_conv"] = x

        x = self._transformer(x, cos, sin, mask)
        if return_intermediates:
            intermediates["pre_transformer"] = x

        for index, stage in enumerate(self.p["upsample"]):
            x = self._trans_conv(x, stage["trans_conv"], f"upsample.{index}", stride=stage["trans_conv"]["kernel"])
            x = self._convnext(x, stage["convnext"], f"upsample.{index}")
            if return_intermediates:
                intermediates[f"upsample.{index}.1"] = x

        x = self._causal_conv(x, self.p["head_conv"], "head_conv")
        if return_intermediates:
            intermediates["decoder.0"] = x

        for index, block in enumerate(self.p["blocks"]):
            x = self._snake(x, block["act"])
            x = self._trans_conv(x, block["trans_conv"], f"block.{index}", stride=block["rate"])
            for unit, dilation in enumerate((1, 3, 9)):
                x = self._residual_unit(x, block["units"][unit], f"block.{index}.{unit}", dilation)
            if return_intermediates:
                intermediates[f"decoder.{index + 1}"] = x

        x = self._snake(x, self.p["final_act"])
        if return_intermediates:
            intermediates[f"decoder.{len(self.p['blocks']) + 1}"] = x

        x = self._causal_conv(x, self.p["final_conv"], "final_conv")
        x = ttnn.clamp(ttnn.to_layout(x, ttnn.TILE_LAYOUT), min=-1.0, max=1.0)
        if return_intermediates:
            intermediates[f"decoder.{len(self.p['blocks']) + 2}"] = x
            return x, intermediates
        return x

    def decode(self, codes, bucket=LENGTH_BUCKET):
        """codes [1, 16, T] -> waveform [1, 1, 1920 * T] on host, matching the reference.

        **Decoded at a bucketed length and trimmed back.** Every distinct frame count
        compiles its own convolution programs, and tt-metal keeps each program's L1_SMALL
        scratch until the device closes: a server speaking utterances of three different
        lengths filled the 64 KB region, and the next block that wanted scratch could not
        allocate. Rounding the length up to a multiple of `bucket` holds the program count
        flat, so only the first utterance in each bucket compiles.

        Trimming is exact rather than approximate. Every convolution here is causal and the
        attention is windowed backwards, so no output sample depends on a later frame, and
        the padding frames cannot change the samples in front of them.
        `test_bucketing_does_not_change_the_samples_it_keeps` measures that.

        Pass `bucket=1` to decode exactly the frames given, which is what the PCC tests do
        so their measurements are not reading padded audio.
        """
        frames = codes.shape[-1]
        padded = self.padded_frames(frames, bucket)
        self._compiled.add(padded)
        if padded > frames:
            # The last frame repeated: a valid code, in distribution, and thrown away after.
            codes = torch.cat([codes, codes[..., -1:].expand(-1, -1, padded - frames)], dim=-1)

        latents = self.latents(codes)
        cos, sin, mask = self.host_inputs(latents.shape[1])
        to_device = lambda tensor: ttnn.from_torch(
            tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        try:
            waveform = self(to_device(latents), to_device(cos), to_device(sin), to_device(mask))
            samples = ttnn.to_torch(waveform).float().reshape(1, -1)
            return samples[:, : frames * self.upsample].unsqueeze(1)
        finally:
            self._prepared.clear()
