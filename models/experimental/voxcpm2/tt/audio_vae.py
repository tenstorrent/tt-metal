# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 deterministic, nonstreaming AudioVAE2 operations.

Architecture follows OpenBMB/VoxCPM at
f0c787f0937dc1c9a8f4f64d9a332d9c5da2e629. Device tensors use
[batch, 1, time, channels], including the public encode/decode boundary.
Host Torch is used only to materialize checkpoint weights. All activation
math uses TTNN. This implementation still requires real-device PCC validation.
"""

from dataclasses import dataclass, fields
from math import ceil, prod


@dataclass(frozen=True)
class AudioVAEConfig:
    encoder_dim: int = 128
    encoder_rates: tuple = (2, 5, 8, 8)
    latent_dim: int = 64
    decoder_dim: int = 2048
    decoder_rates: tuple = (8, 6, 5, 2, 2, 2)
    depthwise: bool = True
    sample_rate: int = 16000
    out_sample_rate: int = 48000
    use_noise_block: bool = False
    sr_bin_boundaries: tuple | None = (20000, 30000, 40000)
    cond_type: str = "scale_bias"
    cond_dim: int = 128
    cond_out_layer: bool = False

    @classmethod
    def from_mapping(cls, values):
        names = {field.name for field in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"Unknown AudioVAE configuration fields: {sorted(unknown)}")
        result = cls(**values)
        result.validate()
        return result

    def validate(self):
        rates = tuple(self.encoder_rates) + tuple(self.decoder_rates)
        if not rates or not self.encoder_rates or not self.decoder_rates or any(rate < 1 for rate in rates):
            raise ValueError("Encoder and decoder rates must contain positive strides")
        if min(self.encoder_dim, self.decoder_dim, self.latent_dim, self.sample_rate, self.out_sample_rate) < 1:
            raise ValueError("AudioVAE dimensions and sample rates must be positive")
        if self.decoder_dim % (2 ** len(self.decoder_rates)):
            raise ValueError("decoder_dim must be divisible by 2**len(decoder_rates)")
        if self.use_noise_block:
            raise NotImplementedError("AudioVAE noise blocks require an explicit device RNG implementation")
        if self.cond_type not in ("scale_bias", "scale_bias_init", "add"):
            raise NotImplementedError(f"AudioVAE conditioning {self.cond_type!r} is not implemented")
        if self.sr_bin_boundaries is not None:
            if any(boundary < 1 for boundary in self.sr_bin_boundaries):
                raise ValueError("Sample-rate boundaries must be positive")
            if tuple(sorted(set(self.sr_bin_boundaries))) != tuple(self.sr_bin_boundaries):
                raise ValueError("Sample-rate boundaries must be strictly increasing")
        # The upstream causal transpose slicing uses :-trim; trim=0 is empty.
        if any(rate == 1 for rate in self.decoder_rates):
            raise NotImplementedError("Stride-one causal transpose blocks are not supported")

    @property
    def chunk_size(self):
        return prod(self.encoder_rates)

    @property
    def decode_chunk_size(self):
        return prod(self.decoder_rates)

    def sample_rate_bucket(self, sample_rate):
        # torch.bucketize(..., right=False): equality stays in the lower bucket.
        if self.sr_bin_boundaries is None:
            return None
        return sum(sample_rate > boundary for boundary in self.sr_bin_boundaries)


@dataclass(frozen=True)
class CausalConvSpec:
    kernel: int
    stride: int = 1
    dilation: int = 1
    padding: int = 0
    output_padding: int = 0
    groups: int = 1
    transpose: bool = False

    def __post_init__(self):
        if min(self.kernel, self.stride, self.dilation, self.groups) < 1:
            raise ValueError("Convolution geometry must be positive")
        if self.padding < 0 or self.output_padding < 0 or self.left_pad < 0:
            raise ValueError("Causal padding must be nonnegative")
        if self.transpose and self.left_pad == 0:
            raise NotImplementedError("Upstream zero-trim transpose convolution produces an empty slice")

    @property
    def left_pad(self):
        return 2 * self.padding - self.output_padding

    def output_length(self, length):
        if length < 1:
            raise ValueError("Audio sequence must be nonempty")
        if self.transpose:
            return (length - 1) * self.stride + self.dilation * (self.kernel - 1) + 1 - self.left_pad
        return (length + self.left_pad - self.dilation * (self.kernel - 1) - 1) // self.stride + 1


def materialize_weight(state, prefix):
    """Remove legacy weight_norm(dim=0) once at checkpoint load, never in inference."""
    import torch

    if f"{prefix}.weight" in state:
        return state[f"{prefix}.weight"].detach().float().cpu().contiguous()
    vector = state[f"{prefix}.weight_v"].detach().float().cpu()
    gain = state[f"{prefix}.weight_g"].detach().float().cpu()
    dimensions = tuple(range(1, vector.ndim))
    norm = torch.linalg.vector_norm(vector, dim=dimensions, keepdim=True)
    if torch.any(norm == 0):
        raise ValueError(f"Zero weight_norm vector at {prefix}")
    return (vector * (gain / norm)).contiguous()


class TtCausalConv1d:
    def __init__(self, state, prefix, device, dtype, spec, compute_config=None):
        import ttnn

        if compute_config is None:
            from .ops import compute_config as make_compute_config

            compute_config = make_compute_config(device)
        self.device, self.dtype, self.spec, self.compute_config = device, dtype, spec, compute_config
        weight = materialize_weight(state, prefix)
        if weight.ndim != 3 or weight.shape[-1] != spec.kernel:
            raise ValueError(f"Unexpected convolution weight shape for {prefix}: {tuple(weight.shape)}")
        if spec.transpose:
            self.in_channels = weight.shape[0]
            self.out_channels = weight.shape[1] * spec.groups
        else:
            self.in_channels = weight.shape[1] * spec.groups
            self.out_channels = weight.shape[0]
        self.weight = ttnn.from_torch(weight.unsqueeze(2), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        bias = state.get(f"{prefix}.bias")
        self.bias = None if bias is None else ttnn.from_torch(
            bias.detach().float().cpu().reshape(1, 1, 1, -1), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        self.prepared = {}

    def __call__(self, x):
        import ttnn

        batch, height, length, channels = tuple(x.shape)
        if height != 1 or channels != self.in_channels:
            raise ValueError(f"Expected [B,1,T,{self.in_channels}], got {tuple(x.shape)}")
        spec = self.spec
        prepared_weight, prepared_bias = self.prepared.get((batch, length), (self.weight, self.bias))
        kwargs = dict(
            input_tensor=ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT),
            weight_tensor=prepared_weight,
            bias_tensor=prepared_bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch,
            groups=spec.groups,
            dtype=self.dtype,
            conv_config=ttnn.Conv2dConfig(weights_dtype=self.dtype),
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        if spec.transpose:
            if length > 8192:
                kwargs["dram_slice_config"] = ttnn.Conv2dSliceConfig(
                    slice_type=ttnn.Conv2dDRAMSliceWidth, num_slices=0
                )
            out, (_, raw_length), prepared = ttnn.conv_transpose2d(
                **kwargs,
                input_height=1,
                input_width=length,
                kernel_size=(1, spec.kernel),
                stride=(1, spec.stride),
                padding=(0, 0),
                output_padding=(0, 0),
                dilation=(1, spec.dilation),
            )
            output_length = raw_length - spec.left_pad
        else:
            out, output_length, prepared = ttnn.conv1d(
                **kwargs,
                input_length=length,
                kernel_size=spec.kernel,
                stride=spec.stride,
                padding=(spec.left_pad, 0),
                dilation=spec.dilation,
            )
        self.prepared[(batch, length)] = prepared
        if output_length != spec.output_length(length):
            raise RuntimeError("TTNN convolution returned an unexpected causal output length")
        out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (batch, 1, raw_length if spec.transpose else output_length, self.out_channels))
        if spec.transpose:
            out = ttnn.slice(out, (0, 0, 0, 0), (batch, 1, output_length, self.out_channels))
        return ttnn.to_layout(out, ttnn.TILE_LAYOUT)


class TtSnake1d:
    def __init__(self, state, prefix, device, dtype):
        import ttnn

        alpha = state[f"{prefix}.alpha"].detach().float().cpu().reshape(1, 1, 1, -1)
        self.alpha = ttnn.from_torch(alpha, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.inverse_alpha = ttnn.from_torch(
            (alpha + 1e-9).reciprocal(), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT
        )

    def __call__(self, x):
        import ttnn

        sine = ttnn.sin(ttnn.multiply(x, self.alpha))
        return ttnn.add(x, ttnn.multiply(ttnn.multiply(sine, sine), self.inverse_alpha))


class TtResidualUnit:
    def __init__(self, state, prefix, device, dtype, dilation, groups, compute_config=None):
        self.snake1 = TtSnake1d(state, f"{prefix}.block.0", device, dtype)
        self.conv1 = TtCausalConv1d(
            state, f"{prefix}.block.1", device, dtype,
            CausalConvSpec(7, dilation=dilation, padding=3 * dilation, groups=groups), compute_config
        )
        self.snake2 = TtSnake1d(state, f"{prefix}.block.2", device, dtype)
        self.conv2 = TtCausalConv1d(state, f"{prefix}.block.3", device, dtype, CausalConvSpec(1), compute_config)

    def __call__(self, x):
        import ttnn

        return ttnn.add(x, self.conv2(self.snake2(self.conv1(self.snake1(x)))))


class TtSampleRateCondition:
    def __init__(self, state, prefix, device, dtype, config, compute_config=None):
        self.state, self.prefix, self.device, self.dtype, self.config = state, prefix, device, dtype, config
        self.cache = {}
        self.out = None
        if config.cond_out_layer:
            self.out = (
                TtSnake1d(state, f"{prefix}.out_layer.0", device, dtype),
                TtCausalConv1d(state, f"{prefix}.out_layer.1", device, dtype, CausalConvSpec(1), compute_config),
            )

    def __call__(self, x, sample_rate):
        import ttnn

        bucket = self.config.sample_rate_bucket(sample_rate)
        if bucket not in self.cache:
            names = ("cond_embed",) if self.config.cond_type == "add" else ("scale_embed", "bias_embed")
            self.cache[bucket] = tuple(
                ttnn.from_torch(
                    self.state[f"{self.prefix}.{name}.weight"][bucket].detach().float().cpu().reshape(1, 1, 1, -1),
                    device=self.device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT
                )
                for name in names
            )
        parameters = self.cache[bucket]
        x = ttnn.add(x, parameters[0]) if len(parameters) == 1 else ttnn.add(ttnn.multiply(x, parameters[0]), parameters[1])
        if self.out is not None:
            x = self.out[1](self.out[0](x))
        return x


class TtAudioVAE:
    """Nonstream encode/decode graph; real hardware correctness is not yet qualified."""

    def __init__(self, state, prefix, device, dtype, config=None, compute_config=None):
        self.config = config or AudioVAEConfig()
        self.config.validate()
        self.chunk_size = self.config.chunk_size
        self.decode_chunk_size = self.config.decode_chunk_size
        self.sample_rate = self.config.sample_rate
        self.out_sample_rate = self.config.out_sample_rate
        self.encoder = []
        self.decoder = []
        self.conditions = {}
        base = f"{prefix}." if prefix else ""

        def conv(name, spec):
            return TtCausalConv1d(state, base + name, device, dtype, spec, compute_config)

        def snake(name):
            return TtSnake1d(state, base + name, device, dtype)

        self.encoder.append(conv("encoder.block.0", CausalConvSpec(7, padding=3)))
        channels = self.config.encoder_dim
        for index, stride in enumerate(self.config.encoder_rates, 1):
            path = f"encoder.block.{index}.block"
            groups = channels if self.config.depthwise else 1
            for residual, dilation in enumerate((1, 3, 9)):
                self.encoder.append(TtResidualUnit(state, base + f"{path}.{residual}", device, dtype, dilation, groups, compute_config))
            self.encoder.extend((snake(f"{path}.3"), conv(f"{path}.4", CausalConvSpec(2 * stride, stride, padding=ceil(stride / 2), output_padding=stride % 2))))
            channels *= 2
        self.encoder_mu = conv("encoder.fc_mu", CausalConvSpec(3, padding=1))
        self.encoder_logvar = conv("encoder.fc_logvar", CausalConvSpec(3, padding=1))

        if self.config.depthwise:
            self.decoder.extend((conv("decoder.model.0", CausalConvSpec(7, padding=3, groups=self.config.latent_dim)), conv("decoder.model.1", CausalConvSpec(1))))
        else:
            self.decoder.append(conv("decoder.model.0", CausalConvSpec(7, padding=3)))
        channels = self.config.decoder_dim
        for stride in self.config.decoder_rates:
            index = len(self.decoder)
            path = f"decoder.model.{index}.block"
            if self.config.sr_bin_boundaries is not None:
                self.conditions[index] = TtSampleRateCondition(state, base + f"decoder.sr_cond_model.{index}", device, dtype, self.config, compute_config)
            layers = [snake(f"{path}.0"), conv(f"{path}.1", CausalConvSpec(2 * stride, stride, padding=ceil(stride / 2), output_padding=stride % 2, transpose=True))]
            channels //= 2
            groups = channels if self.config.depthwise else 1
            for residual, dilation in enumerate((1, 3, 9), 2):
                layers.append(TtResidualUnit(state, base + f"{path}.{residual}", device, dtype, dilation, groups, compute_config))
            self.decoder.append(tuple(layers))
        index = len(self.decoder)
        self.decoder.extend((snake(f"decoder.model.{index}"), conv(f"decoder.model.{index + 1}", CausalConvSpec(7, padding=3))))

    def encode(self, audio, sample_rate=None, *, return_distribution=False):
        import ttnn

        if sample_rate not in (None, self.sample_rate):
            raise ValueError(f"Encoder requires {self.sample_rate} Hz audio")
        batch, height, length, channels = tuple(audio.shape)
        if height != 1 or channels != 1:
            raise ValueError("Encoder audio must have shape [B,1,T,1]")
        right_pad = (-length) % self.chunk_size
        if right_pad:
            audio = ttnn.pad(ttnn.to_layout(audio, ttnn.ROW_MAJOR_LAYOUT), ((0, 0), (0, 0), (0, right_pad), (0, 0)), value=0.0)
        for layer in self.encoder:
            audio = layer(audio)
        mu = self.encoder_mu(audio)
        if return_distribution:
            return {"hidden_state": audio, "mu": mu, "logvar": self.encoder_logvar(audio)}
        return mu

    def decode(self, latent, sample_rate=None):
        import ttnn

        if tuple(latent.shape)[-1] != self.config.latent_dim:
            raise ValueError("Decoder latent channel count does not match configuration")
        sample_rate = self.out_sample_rate if sample_rate is None else sample_rate
        for index, layer in enumerate(self.decoder):
            if index in self.conditions:
                latent = self.conditions[index](latent, sample_rate)
            if isinstance(layer, tuple):
                for operation in layer:
                    latent = operation(latent)
            else:
                latent = layer(latent)
        return ttnn.tanh(latent)

    def streaming_decode(self):
        raise NotImplementedError("Stateful streaming AudioVAE has not been ported")
