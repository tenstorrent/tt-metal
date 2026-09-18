# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""`ConvRNNF0Predictor`: mel -> f0 (Hz), the piece `generator.py` documents as
deferred ("`f0` is therefore an explicit external input" -- see that module's
docstring, right above `TorchHiFTGeneratorInferenceRef`).

Confirmed against real upstream source directly (fetched from
`FunAudioLLM/CosyVoice`'s `cosyvoice/hifigan/f0_predictor.py`), not assumed
from the class name: **despite "RNN" in the name, `ConvRNNF0Predictor` has NO
recurrent layer anywhere** -- it is five plain `Conv1d(kernel_size=3,
padding=1)` layers, each wrapped in `weight_norm`, each followed by `ELU`,
then one `Linear` classifier, then `abs()`:

    condnet = Sequential(
        weight_norm(Conv1d(80, 512, k=3, pad=1)), ELU(),
        weight_norm(Conv1d(512, 512, k=3, pad=1)), ELU(),
        weight_norm(Conv1d(512, 512, k=3, pad=1)), ELU(),
        weight_norm(Conv1d(512, 512, k=3, pad=1)), ELU(),
        weight_norm(Conv1d(512, 512, k=3, pad=1)), ELU(),
    )
    classifier = Linear(512, 1)

    def forward(x):        # x: [B, 80, T] channel-first
        x = condnet(x)              # [B, 512, T]
        x = x.transpose(1, 2)       # [B, T, 512]
        return abs(classifier(x).squeeze(-1))   # [B, T]

(There IS a `CausalConvRNNF0Predictor` in the same upstream file, built from
`CausalConv1d` instead -- a different class, not used here. CosyVoice2's own
`examples/libritts/cosyvoice2/conf/cosyvoice2.yaml` instantiates the plain,
non-causal `ConvRNNF0Predictor` for `hift.f0_predictor` with
`num_class=1, in_channels=80, cond_channels=512` -- exactly the defaults used
below -- confirming this is the right class for CosyVoice2's (non-causal)
`HiFTGenerator`, matching `generator.py`'s own confirmed choice of
`HiFTGenerator` over `CausalHiFTGenerator`.)

**Interface contract into `HiFTGenerator.inference`**, confirmed from real
upstream source: `f0 = self.f0_predictor(speech_feat)` -- `speech_feat` flows
in with NO transform in between, so whatever channel convention `speech_feat`
already has at that call site is what this module must accept. Real upstream
calls it channel-first (`inference`'s own `speech_feat` argument, `[B, 80,
T]`). This port's mel is channels-last (`[B, T, 80]`) everywhere else (see
`generator.py`'s module docstring), so `TtConvRNNF0Predictor` takes
channels-last mel directly, matching every other Tt module in this package,
while `TorchConvRNNF0PredictorRef` takes channel-first mel, matching real
upstream exactly -- so its weights and math stay checkable against a real
`ConvRNNF0Predictor` instance token-for-token if one is ever loaded, the same
split `generator.py` already draws for `TorchHiFTDecodeRef`/`TtHiFTDecoder`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # pragma: no cover - matches upstream's own fallback
    from torch.nn.utils import weight_norm

import ttnn

from .conv import TtConv1d

COND_CHANNELS = 512
NUM_CONV_LAYERS = 5


class TorchConvRNNF0PredictorRef(nn.Module):
    """Exact real upstream `ConvRNNF0Predictor`, channel-first. See module docstring."""

    def __init__(
        self,
        in_channels: int = 80,
        cond_channels: int = COND_CHANNELS,
        num_class: int = 1,
        seed: int = 0,
    ):
        super().__init__()
        # NOT `TorchHiFTDecodeRef`'s own std=0.02-reinit pattern (generator.py):
        # that overwrites `weight_norm`'s `g`/`v` independently from a fixed
        # std, severing `g`'s usual init relationship to `||v||`. It is safe
        # there because that stack is residual end to end; measured here (a
        # plain, non-residual 5-conv stack) it collapses device-vs-torch PCC to
        # ~0.03 under bf16 rounding -- a real numerical-stability difference,
        # not a device-op bug. So this keeps each module's own default init
        # (kaiming-uniform-based, which sets `g = ||v||` via `weight_norm`'s
        # construction hook) and only seeds torch's global RNG for
        # reproducibility.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            layers = []
            ch = in_channels
            for _ in range(NUM_CONV_LAYERS):
                layers.append(weight_norm(nn.Conv1d(ch, cond_channels, kernel_size=3, padding=1)))
                layers.append(nn.ELU())
                ch = cond_channels
            self.condnet = nn.Sequential(*layers)
            self.classifier = nn.Linear(cond_channels, num_class)

    @classmethod
    def from_checkpoint(cls, f0_predictor_state_dict: dict, **kwargs) -> "TorchConvRNNF0PredictorRef":
        """Real weights from `hift.pt`'s `f0_predictor.*` keys (see
        `tt/checkpoint.py`'s `sub_state_dict` to pull them out of the flat
        checkpoint first). Zero renaming needed -- confirmed empirically:
        `f0_predictor.condnet.{0,2,4,6,8}` and `f0_predictor.classifier.*`
        match this class's own `state_dict()` keys 1:1 (17/17), because
        `condnet`'s `[Conv1d, ELU] * 5` `Sequential` indexing lands the five
        convs at exactly those indices on both sides."""
        ref = cls(**kwargs)
        ref.load_state_dict(f0_predictor_state_dict, strict=True)
        return ref

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, T] channel-first -> f0 [B, T] Hz (non-negative)."""
        x = self.condnet(x)
        x = x.transpose(1, 2)
        return torch.abs(self.classifier(x).squeeze(-1))


class TtConvRNNF0Predictor:
    """`TorchConvRNNF0PredictorRef` on device, channels-last. See module docstring."""

    def __init__(self, device, ref: TorchConvRNNF0PredictorRef, dtype=ttnn.bfloat16, high_fidelity: bool = True):
        self.device = device
        self.dtype = dtype
        self.convs = [
            TtConv1d.from_module(device, layer, dtype=dtype, high_fidelity=high_fidelity)
            for layer in ref.condnet
            if isinstance(layer, nn.Conv1d)
        ]
        assert len(self.convs) == NUM_CONV_LAYERS, len(self.convs)

        w = ref.classifier.weight.detach().float().t().contiguous()  # [cond_channels, num_class]
        b = ref.classifier.bias.detach().float().reshape(1, 1, -1)
        self.classifier_weight = ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.classifier_bias = ttnn.from_torch(b, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def __call__(self, mel, mel_frames: int, batch_size: int = 1):
        """mel: ttnn [B, T_mel, 80] channels-last -> ttnn [B, T_mel] Hz (non-negative).

        OWNERSHIP: frees only the intermediates it creates, never `mel` (matches
        `TtResBlock.__call__`'s convention -- see that class's docstring).
        """
        x, length = mel, mel_frames
        for i, conv in enumerate(self.convs):
            nxt, length = conv(x, length, batch_size)
            if x is not mel:
                ttnn.deallocate(x)
            x = ttnn.elu(nxt, alpha=1.0)
            ttnn.deallocate(nxt)
        logits = ttnn.linear(ttnn.typecast(x, self.classifier_weight.dtype), self.classifier_weight, bias=self.classifier_bias)
        if x is not mel:
            ttnn.deallocate(x)
        f0 = ttnn.abs(logits)
        ttnn.deallocate(logits)
        return ttnn.reshape(f0, (batch_size, length))
