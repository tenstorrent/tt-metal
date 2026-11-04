# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the Qwen3-TTS codec decoder: codes in, waveform out.

This is the last block before audio. It takes the 16 codebooks per frame that the talker
and code predictor produce and upsamples them by 1920 into 24 kHz samples, so one frame at
12.5 Hz becomes 1920 samples.

    codes [1, 16, T]
      -> quantizer.decode                          [1, 512, T]
      -> pre_conv, causal k=3                      [1, 1024, T]
      -> pre_transformer, 8 layers at hidden 512   [1, T, 1024]
      -> 2 x (transposed conv stride 2 + ConvNeXt) [1, 1024, 4T]
      -> conv k=7                                  [1, 1536, 4T]
      -> 4 decoder blocks, rates 8, 5, 4, 3        [1, 96, 1920T]
      -> SnakeBeta, conv k=7, clamp                [1, 1, 1920T]

Where the 1920 comes from: upsampling_ratios [2, 2] then upsample_rates [8, 5, 4, 3], so
4 x 480. The encoder reaches the same ratio by a different route, which is why the two
halves list different factors.

Two details worth naming. Every convolution is causal, padded on the left by the full
receptive field so no output depends on a future sample, which is what lets the codec
stream. And the quantizer stores its codebooks as `embedding_sum` and `cluster_usage`
rather than a plain table: the usable codebook is their quotient, computed at load.
"""

import functools

import torch

from models.demos.audio.qwen3_tts import weights
from models.demos.audio.qwen3_tts.reference.qwen import transformers_compat  # noqa: F401  (registers the shims)
from models.demos.audio.qwen3_tts.reference.qwen.codec_decoder import (
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2DecoderConfig,
)


class CodecDecoderReference:
    """The upstream decoder, loaded from the codec checkpoint and frozen in eval mode."""

    def __init__(self, config=None, dtype=torch.float32):
        cfg = dict(config or weights.codec_decoder_config())
        cfg.setdefault("pad_token_id", None)
        self.config = cfg
        self.quantizers = cfg["num_quantizers"]
        self.upsample = int(torch.tensor(cfg["upsample_rates"] + cfg["upsampling_ratios"], dtype=torch.long).prod())

        self.model = Qwen3TTSTokenizerV2Decoder(Qwen3TTSTokenizerV2DecoderConfig(**cfg))
        self.model.load_state_dict(weights.load_codec_decoder_state(dtype=dtype), strict=True)
        self.model.to(dtype).eval()

    @torch.inference_mode()
    def __call__(self, codes, return_intermediates=False):
        """codes [1, 16, T] -> waveform [1, 1, 1920 * T]."""
        if not return_intermediates:
            return self.model(codes)

        captured = {}
        handles = []

        def capture(name):
            def hook(_module, _args, output):
                tensor = output
                if hasattr(tensor, "last_hidden_state"):  # the transformer returns a ModelOutput
                    tensor = tensor.last_hidden_state
                elif isinstance(tensor, tuple):
                    tensor = tensor[0]
                captured[name] = tensor.detach().clone()

            return hook

        modules = dict(self.model.named_modules())
        watched = ["quantizer", "pre_conv", "pre_transformer"]
        watched += [f"upsample.{index}.1" for index in range(len(self.config["upsampling_ratios"]))]
        watched += [f"decoder.{index}" for index in range(len(self.config["upsample_rates"]) + 3)]
        for name in watched:
            if name in modules:
                handles.append(modules[name].register_forward_hook(capture(name)))
        try:
            waveform = self.model(codes)
        finally:
            for handle in handles:
                handle.remove()
        return waveform, captured


@functools.lru_cache(maxsize=1)
def reference_model(dtype=torch.float32):
    return CodecDecoderReference(dtype=dtype)
