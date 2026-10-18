# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the Qwen3-TTS codec encoder: waveform in, codes out.

The other half of the codec. Voice cloning needs it: the Base checkpoint conditions on a
reference clip, and the prompt carries that clip as codes on the codec track, so without
this block there is nothing to put there.

    waveform [1, 1, N] at 24 kHz
      -> conv stack, strides 4, 5, 6, 8       [1, 512, N/960]
      -> transformer, 8 layers at hidden 512  [1, N/960, 512]
      -> downsample conv, stride 2            [1, 512, N/1920]
      -> 16 residual quantizer steps          [1, 16, N/1920]

**No vendored copy here, unlike the decoder.** Upstream's `Qwen3TTSTokenizerV2Encoder` is a
three line `MimiModel` subclass that sets `upsample`, `decoder_transformer` and `decoder` to
None; everything it runs is `transformers`' own Mimi. So this file builds a `MimiModel` from
the checkpoint's `encoder_config`, nulls the same three attributes, and loads the 225
`encoder.*` tensors with `strict=True`. Key parity is exact, which is the check that this is
really the same module upstream instantiates.

The wrapper semantics come from `Qwen3TTSTokenizerV2Model.encode`: encode with all 32
codebooks, keep the first `encoder_valid_num_quantizers` (16), and trim to
`ceil(samples / 1920)` frames. Upstream then transposes to `[T, 16]`; this returns
`[1, 16, T]` to match the shape the decoder in this directory takes.

**Fidelity is measured, not asserted.** The same 3 s clip through the genuine `qwen-tts`
package under transformers 4.57.3, in a separate venv, against this file under 5.12.1:
pre-quantizer latents identical to the last bit (max absolute difference 0.0) and all 608
codes the same. That check cannot live in the suite, since the two versions cannot share an
environment, so `test_the_encoder_is_exactly_transformers_mimi` guards the thing that would
break it: the key sets diverging.

One number to hold on to: the conv stack downsamples by 960 and the final conv by 2, so the
frame rate is 12.5 Hz and a 3 s clip is 38 frames. The transformer runs at 25 Hz, before the
downsample, so it sees about twice as many positions as there are codes: 75 for those 38,
since both stages round up independently.
"""

import functools

import torch
from transformers import MimiConfig
from transformers.models.mimi.modeling_mimi import MimiModel

from models.demos.audio.qwen3_tts import weights


class CodecEncoderReference:
    """`transformers`' Mimi encoder, loaded from the codec checkpoint and frozen in eval mode."""

    def __init__(self, config=None, dtype=torch.float32):
        cfg = dict(config or weights.codec_encoder_config())
        self.config = MimiConfig(**cfg)
        self.quantizers = weights.codec_valid_quantizers()
        self.downsample = weights.codec_config()["encode_downsample_rate"]
        self.sample_rate = weights.codec_config()["input_sample_rate"]

        self.model = MimiModel(self.config)
        # The decode half has no weights in this checkpoint, and upstream drops it the same way.
        self.model.upsample = None
        self.model.decoder_transformer = None
        self.model.decoder = None
        self.model.load_state_dict(weights.load_codec_encoder_state(dtype=dtype), strict=True)
        self.model.to(dtype).eval()

    def frames(self, samples):
        """How many codes a clip of this many samples produces."""
        return -(-int(samples) // self.downsample)

    @torch.inference_mode()
    def __call__(self, waveform, return_intermediates=False):
        """waveform [N], [1, N] or [1, 1, N] -> codes [1, 16, T]."""
        audio = torch.as_tensor(waveform)
        while audio.dim() < 3:
            audio = audio.unsqueeze(0)
        samples = audio.shape[-1]

        if not return_intermediates:
            codes = self.model.encode(audio, num_quantizers=self.config.num_quantizers).audio_codes
            return codes[:, : self.quantizers, : self.frames(samples)]

        captured = {}
        handles = []

        def capture(name):
            def hook(_module, _args, output):
                tensor = output
                if hasattr(tensor, "last_hidden_state"):
                    tensor = tensor.last_hidden_state
                elif isinstance(tensor, tuple):
                    tensor = tensor[0]
                captured[name] = tensor.detach().clone()

            return hook

        modules = dict(self.model.named_modules())
        watched = [f"encoder.layers.{index}" for index in range(len(self.model.encoder.layers))]
        watched += ["encoder", "encoder_transformer", "downsample"]
        watched += [f"encoder_transformer.layers.{index}" for index in range(self.config.num_hidden_layers)]
        for name in watched:
            handles.append(modules[name].register_forward_hook(capture(name)))
        try:
            codes = self.model.encode(audio, num_quantizers=self.config.num_quantizers).audio_codes
        finally:
            for handle in handles:
                handle.remove()
        return codes[:, : self.quantizers, : self.frames(samples)], captured

    @torch.inference_mode()
    def latents(self, waveform):
        """The embeddings the quantizer sees: waveform -> [1, 512, T], before any codebook.

        Split out because it is the only thing a TTNN port has to match in floating point.
        Everything after it is a nearest-neighbour search, which either picks the same code
        or does not.
        """
        audio = torch.as_tensor(waveform)
        while audio.dim() < 3:
            audio = audio.unsqueeze(0)
        hidden = self.model.encoder(audio)
        hidden = self.model.encoder_transformer(hidden.transpose(1, 2))[0].transpose(1, 2)
        return self.model.downsample(hidden)


@functools.lru_cache(maxsize=1)
def reference_model(dtype=torch.float32):
    return CodecEncoderReference(dtype=dtype)
