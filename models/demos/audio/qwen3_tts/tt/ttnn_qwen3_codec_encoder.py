# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the Qwen3-TTS codec encoder: waveform in, codes out.

Reference: models/demos/audio/qwen3_tts/reference/qwen3_codec_encoder_ref.py

    waveform [1, N, 1] at 24 kHz
      -> conv stack, strides 4, 5, 6, 8       [1, N/960, 512]
      -> transformer, 8 layers at hidden 512  [1, N/960, 512]
      -> downsample conv, stride 2            [1, N/1920, 512]
      -> 16 residual quantizer steps          [1, 16, N/1920]

Voice cloning needs this block: the prompt carries a reference clip as codes on the codec
track, and nothing else produces them.

Everything works time-major, [1, T, C], which is what the convolutions want. The reference
works channel-first, so each intermediate is the transpose of its counterpart there. The
causal padding and the length-keyed weight cache are shared with the decoder; see
`causal_conv1d` in `ttnn_qwen3_codec`.

Four things worth knowing:

  1. **The transformer runs at 25 Hz, the codes at 12.5.** The stride-2 `downsample`
     convolution sits between them, so the attention sees twice as many positions as there
     are codes. A 3 s clip is 75 positions and 38 codes.
  2. **This transformer is not the talker's.** LayerNorm with a bias rather than RMSNorm,
     a plain two-matmul MLP with exact GELU rather than a SwiGLU, and both residual
     branches scaled by a learned per-channel vector (`layer_scale`, initialised at 0.01,
     which is why the residual stream stays close to the conv stack's output).
  3. **The quantizer search is an argmax, not a matmul chain.** Nearest neighbour under
     Euclidean distance is `argmax(x . e - |e|^2 / 2)` once the `|x|^2` term is dropped,
     which it can be because it does not depend on the codebook entry. The two halves of
     the split quantizer both project the *same* embeddings; the acoustic half does not
     continue the semantic half's residual.
  4. **The semantic and acoustic codebooks are quotients**, stored as `embed_sum` and
     `cluster_usage`, so the usable table is built once at load. Same convention as the
     decoder.

Correctness first: nothing traced or sharded, and the whole clip encodes in one pass.
"""

import torch

import ttnn
from models.demos.audio.qwen3_tts import weights as checkpoint
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import (
    CODEBOOK_EPS,
    causal_conv1d,
    conv_parameters,
    rotary_tables,
    windowed_causal_mask,
)
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_talker import _compute_config

# ── host-side structure ─────────────────────────────────────────────────────


def encoder_layer_plan(config):
    """What each index of `encoder.layers` is, derived from the config.

    Mirrors `MimiEncoder.__init__`, which interleaves activations into the same list as the
    convolutions: one input convolution, then per downsampling ratio a residual block, an
    ELU and a strided convolution, then a final ELU and convolution. For this checkpoint
    that is 15 entries and four ratios.
    """
    plan = [("conv", 0, {"stride": 1})]
    index = 1
    for ratio in reversed(config["upsampling_ratios"]):
        for step in range(config["num_residual_layers"]):
            plan.append(("resnet", index, {"dilation": config["dilation_growth_rate"] ** step}))
            index += 1
        plan.append(("elu", index, {}))
        index += 1
        plan.append(("conv", index, {"stride": ratio}))
        index += 1
    plan.append(("elu", index, {}))
    index += 1
    plan.append(("conv", index, {"stride": 1}))
    return plan


def codebooks(state, config, quantizers):
    """The first `quantizers` usable codebooks, in the order the talker speaks them.

    Index 0 is the semantic quantizer, the rest acoustic. Each table is
    `embed_sum / cluster_usage`, mirroring `MimiEuclideanCodebook.embed`.
    """
    tables = []
    semantic = config["num_semantic_quantizers"]
    for index in range(quantizers):
        if index < semantic:
            base = f"quantizer.semantic_residual_vector_quantizer.layers.{index}.codebook"
        else:
            base = f"quantizer.acoustic_residual_vector_quantizer.layers.{index - semantic}.codebook"
        usage = state[f"{base}.cluster_usage"].reshape(-1, 1)
        tables.append(state[f"{base}.embed_sum"] / usage.clamp(min=CODEBOOK_EPS))
    return tables


# ── parameters ──────────────────────────────────────────────────────────────


def preprocess_codec_encoder_parameters(device, config=None, state=None, dtype=ttnn.float32):
    """Checkpoint weights -> the tensors the encoder runs with.

    **fp32 by default, unlike every other block here.** This one ends in a nearest-neighbour
    search over 2048 entries, repeated 16 times down a residual chain, so a small error in
    the latents becomes a different code. Measured on a 3 s clip: bf16 puts the latents at
    PCC 0.9984 and agrees with the reference on 454 of 608 codes; fp32 puts them at 0.99989
    and agrees on 545. The block runs once per reference clip, and the extra cost is 1.5 s.
    """
    cfg = dict(config or checkpoint.codec_encoder_config())
    state = checkpoint.load_codec_encoder_state() if state is None else state
    quantizers = checkpoint.codec_valid_quantizers()

    def conv(name):
        return conv_parameters(state, name, dtype=dtype)

    def to_device(tensor, tensor_dtype=None):
        return ttnn.from_torch(
            tensor.contiguous(),
            dtype=tensor_dtype or dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def linear(name):
        return to_device(state[f"{name}.weight"].t())

    def vector(name):
        return to_device(state[name].reshape(1, 1, -1))

    stack = []
    for kind, index, options in encoder_layer_plan(cfg):
        entry = {"kind": kind, "key": f"layers.{index}", **options}
        if kind == "conv":
            entry["conv"] = conv(f"encoder.layers.{index}.conv")
        elif kind == "resnet":
            entry["conv1"] = conv(f"encoder.layers.{index}.block.1.conv")
            entry["conv2"] = conv(f"encoder.layers.{index}.block.3.conv")
        stack.append(entry)

    layers = []
    for index in range(cfg["num_hidden_layers"]):
        prefix = f"encoder_transformer.layers.{index}"
        layers.append(
            {
                "input_layernorm": vector(f"{prefix}.input_layernorm.weight"),
                "input_layernorm_bias": vector(f"{prefix}.input_layernorm.bias"),
                "post_attention_layernorm": vector(f"{prefix}.post_attention_layernorm.weight"),
                "post_attention_layernorm_bias": vector(f"{prefix}.post_attention_layernorm.bias"),
                "q_proj": linear(f"{prefix}.self_attn.q_proj"),
                "k_proj": linear(f"{prefix}.self_attn.k_proj"),
                "v_proj": linear(f"{prefix}.self_attn.v_proj"),
                "o_proj": linear(f"{prefix}.self_attn.o_proj"),
                "attn_scale": vector(f"{prefix}.self_attn_layer_scale.scale"),
                "fc1": linear(f"{prefix}.mlp.fc1"),
                "fc2": linear(f"{prefix}.mlp.fc2"),
                "mlp_scale": vector(f"{prefix}.mlp_layer_scale.scale"),
            }
        )

    def projection(half):
        # input_proj is a bias-free 1x1 convolution, so [256, 512, 1] -> a [512, 256] matmul.
        weight = state[f"quantizer.{half}_residual_vector_quantizer.input_proj.weight"].squeeze(-1)
        return to_device(weight.t())

    tables = codebooks(state, cfg, quantizers)
    return {
        "config": cfg,
        "dtype": dtype,
        "quantizers": quantizers,
        "stack": stack,
        "layers": layers,
        "downsample": conv("downsample.conv"),
        "semantic_proj": projection("semantic"),
        "acoustic_proj": projection("acoustic"),
        # Each codebook is carried twice: transposed for the score matmul, upright for the
        # gather that builds the residual.
        "codebooks": [to_device(table.t()) for table in tables],
        # `ttnn.embedding` takes a bfloat16 table and nothing else, so the gather side stays
        # bf16 whatever the rest of the module runs in. It costs nothing measurable: the
        # codebooks are the input to a subtraction, not to a long accumulation.
        "codebook_rows": [to_device(table, ttnn.bfloat16) for table in tables],
        # -|e|^2 / 2, the only part of the distance that depends on the entry alone.
        "codebook_bias": [to_device((-0.5 * (table * table).sum(-1)).reshape(1, 1, -1)) for table in tables],
    }


class TtCodecEncoder:
    def __init__(self, device, parameters):
        self.device = device
        self.p = parameters
        self.config = parameters["config"]
        self.compute_config = _compute_config(device)
        self.dtype = parameters["dtype"]
        self.conv_config = ttnn.Conv1dConfig(weights_dtype=self.dtype)
        self.heads = self.config["num_attention_heads"]
        self.kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.config["head_dim"]
        self.scale = self.head_dim**-0.5
        self.quantizers = parameters["quantizers"]
        # The conv stack strides multiply to 960 and `downsample` halves again, so a code
        # is one 1920-sample frame: 12.5 Hz at 24 kHz.
        self.stack_rate = int(torch.tensor(self.config["upsampling_ratios"], dtype=torch.long).prod())
        self.downsample_rate = 2 * self.stack_rate
        self._prepared = {}

    # ── primitives ──────────────────────────────────────────────────────────

    def _causal_conv(self, x, params, key, stride=1, dilation=1, pad_mode="constant"):
        return causal_conv1d(
            self.device,
            x,
            params,
            self._prepared,
            key,
            self.compute_config,
            self.conv_config,
            stride,
            dilation,
            pad_mode,
            self.dtype,
        )

    def _resnet(self, x, params):
        """ELU, conv k=3, ELU, conv k=1, plus the input. The shortcut is an identity here."""
        residual = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        out = ttnn.elu(residual, alpha=1.0)
        out = self._causal_conv(out, params["conv1"], f"{params['key']}.conv1", dilation=params["dilation"])
        out = ttnn.elu(ttnn.to_layout(out, ttnn.TILE_LAYOUT), alpha=1.0)
        out = self._causal_conv(out, params["conv2"], f"{params['key']}.conv2")
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

    def _transformer(self, x, cos, sin, mask, intermediates=None):
        length = x.shape[1]
        eps = self.config["norm_eps"]
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        for index, layer in enumerate(self.p["layers"]):
            normed = ttnn.layer_norm(
                x, weight=layer["input_layernorm"], bias=layer["input_layernorm_bias"], epsilon=eps
            )
            attended = self._attention(normed, layer, cos, sin, mask, length)
            x = ttnn.add(x, ttnn.multiply(attended, layer["attn_scale"]))

            normed = ttnn.layer_norm(
                x, weight=layer["post_attention_layernorm"], bias=layer["post_attention_layernorm_bias"], epsilon=eps
            )
            hidden = ttnn.linear(normed, layer["fc1"], compute_kernel_config=self.compute_config)
            hidden = ttnn.gelu(hidden)
            hidden = ttnn.linear(hidden, layer["fc2"], compute_kernel_config=self.compute_config)
            x = ttnn.add(x, ttnn.multiply(hidden, layer["mlp_scale"]))
            if intermediates is not None:
                intermediates[f"encoder_transformer.layers.{index}"] = x
        return x

    # ── forward ─────────────────────────────────────────────────────────────

    def frames(self, samples):
        """How many codes a clip of this many samples produces."""
        return -(-int(samples) // self.downsample_rate)

    def positions(self, samples):
        """How many transformer positions a clip of this many samples produces.

        Not twice `frames`: the conv stack rounds up to 960 and `downsample` rounds up
        again, so an odd position count is normal (75 positions give 38 codes).
        """
        return -(-int(samples) // self.stack_rate)

    def host_inputs(self, positions):
        """Rotation tables and the windowed mask for this many transformer positions."""
        cos, sin = rotary_tables(self.config, positions)
        return cos, sin, windowed_causal_mask(positions, self.config["sliding_window"])

    def __call__(self, audio, cos, sin, mask, return_intermediates=False):
        """audio [1, N, 1] -> latents [1, N/1920, 512], the embeddings the quantizer reads."""
        intermediates = {} if return_intermediates else None

        x = audio
        for entry in self.p["stack"]:
            if entry["kind"] == "elu":
                x = ttnn.elu(ttnn.to_layout(x, ttnn.TILE_LAYOUT), alpha=1.0)
            elif entry["kind"] == "conv":
                x = self._causal_conv(x, entry["conv"], entry["key"], stride=entry["stride"])
            else:
                x = self._resnet(x, entry)
            if return_intermediates:
                intermediates[f"encoder.{entry['key']}"] = x

        if return_intermediates:
            intermediates["encoder"] = x

        x = self._transformer(x, cos, sin, mask, intermediates)
        if return_intermediates:
            intermediates["encoder_transformer"] = x

        # The one convolution in the codec that replicates its padding rather than
        # zeroing it. `MimiModel.__init__` passes `pad_mode="replicate"` here and nowhere
        # else; zeros cost 0.009 of PCC on exactly the tensor the codes come from.
        x = self._causal_conv(x, self.p["downsample"], "downsample", stride=2, pad_mode="replicate")
        if return_intermediates:
            intermediates["downsample"] = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            return x, intermediates
        return x

    def quantize(self, latents, prefix=None):
        """latents [1, T, 512] -> codes [1, 16, T] on host.

        Residual vector quantisation: pick the nearest codebook entry, subtract it, repeat.

        `prefix` [k, T] feeds the *reference's* codes into the residual chain while still
        returning this encoder's own picks. One flipped code changes the residual every
        later codebook sees, so a free run measures the cascade; forcing the prefix measures
        the implementation. The code predictor is scored the same way and for the same
        reason.
        """
        length = latents.shape[1]
        latents = ttnn.to_layout(latents, ttnn.TILE_LAYOUT)
        semantic = self.config["num_semantic_quantizers"]
        picked = []

        for index in range(self.quantizers):
            if index in (0, semantic):
                # Each half of the split quantizer projects the embeddings itself, so the
                # acoustic half starts over rather than inheriting the semantic residual.
                half = "semantic_proj" if index < semantic else "acoustic_proj"
                residual = ttnn.linear(latents, self.p[half], compute_kernel_config=self.compute_config)

            scores = ttnn.linear(residual, self.p["codebooks"][index], compute_kernel_config=self.compute_config)
            scores = ttnn.add(scores, self.p["codebook_bias"][index])
            best = ttnn.argmax(ttnn.reshape(scores, (1, 1, length, -1)), dim=-1)
            ttnn.deallocate(scores)
            codes = ttnn.to_torch(best).reshape(-1).long()
            ttnn.deallocate(best)
            picked.append(codes)

            if index + 1 < self.quantizers:
                chosen = codes
                if prefix is not None and index < len(prefix):
                    chosen = torch.as_tensor(prefix[index], dtype=torch.long).reshape(-1)
                chosen = ttnn.from_torch(
                    chosen.reshape(1, -1).to(torch.uint32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                )
                vectors = ttnn.embedding(chosen, self.p["codebook_rows"][index], layout=ttnn.TILE_LAYOUT)
                if self.dtype != ttnn.bfloat16:
                    vectors = ttnn.typecast(vectors, self.dtype)
                residual = ttnn.subtract(residual, vectors)
        return torch.stack(picked).unsqueeze(0)

    def encode(self, waveform, prefix=None):
        """waveform [N], [1, N] or [1, 1, N] -> codes [1, 16, T] on host."""
        audio = torch.as_tensor(waveform).reshape(1, -1, 1).float()
        samples = audio.shape[1]
        cos, sin, mask = self.host_inputs(self.positions(samples))
        to_device = lambda tensor: ttnn.from_torch(
            tensor, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
        )
        try:
            latents = self(
                ttnn.from_torch(audio, dtype=self.dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device),
                to_device(cos),
                to_device(sin),
                to_device(mask),
            )
            codes = self.quantize(latents, prefix=prefix)
            return codes[:, :, : self.frames(samples)]
        finally:
            # Same reason as the decoder's: keyed by length while the clip runs, dropped
            # afterwards so they stop holding L1. See `TtCodecDecoder.decode`.
            self._prepared.clear()
