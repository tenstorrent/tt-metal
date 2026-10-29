# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioModel: device forward of the audio path (HF `Gemma4AudioModel` + `embed_audio`).

    h      = subsample(mel, frame_mask, mask1)                 [B,1,S,1024]   two convs + input_proj
    for i in 0..11:  h = conformer_layer_i(h, attn_mask)
    hidden = output_proj(h)                                   [B,1,S,1536] fp32 (HF last_hidden_state)
    soft   = embed_audio(hidden)                              [B,1,S,512]  bf16 (audio soft tokens)

Contract (forward)
    `AudioDeviceInputs` built by `AudioEncoder.prepare_inputs` and uploaded (see audio_encoder.py):
        mel [B,1,4S,128] fp32 TILE, frame_mask [B,1,4S,1] fp32 TILE, mask1 [1,1,B*2S*64,1] bf16
        TILE, attn_mask [B,1,S,64] fp32 TILE. S is a token bucket (config.TOKEN_BUCKETS).
    Returns (soft [B,1,S,512] bf16, hidden [B,1,S,1536] fp32). Clip b's soft tokens are rows
    0..n_b-1 (n_b = valid tokens). forward() issues device ops only, so it can be captured in a trace,
    once `prepare(inputs)` has laid out the conv weights for that (batch, bucket).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.audio.config import AudioConfig
from models.demos.embedding_gemma2.tt.audio.conformer_layer import AudioConformerLayer
from models.demos.embedding_gemma2.tt.audio.embedder import AudioEmbedder, AudioOutputProj
from models.demos.embedding_gemma2.tt.audio.light_conv import load_causal_shifts
from models.demos.embedding_gemma2.tt.audio.subsample import AudioSubsample
from models.demos.embedding_gemma2.tt.precision import PrecisionConfig, resolve_precision
from models.demos.embedding_gemma2.tt.weights import substate

SHIFT_IDX_ONCE = os.environ.get("EG2_AUDIO_SHIFT_IDX_ONCE", "1") == "1"


@dataclass
class AudioDeviceInputs:
    mel: ttnn.Tensor
    frame_mask: ttnn.Tensor
    mask1: ttnn.Tensor
    attn_mask: ttnn.Tensor

    def tensors(self) -> tuple[ttnn.Tensor, ...]:
        return (self.mel, self.frame_mask, self.mask1, self.attn_mask)


class AudioModel(LightweightModule):
    def __init__(
        self,
        config: AudioConfig,
        device,
        subsample: AudioSubsample,
        layers: list[AudioConformerLayer],
        output_proj: AudioOutputProj,
        embedder: AudioEmbedder,
        precision: PrecisionConfig | None = None,
    ):
        super().__init__()
        self.precision = precision or resolve_precision(None)
        self.config = config
        self.device = device
        self.subsample = subsample
        self.layers = layers
        self.output_proj = output_proj
        self.embedder = embedder

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: AudioConfig,
        device,
        *,
        cache_dir: Path | None = None,
        num_layers: int | None = None,
        act_dtype=None,
        precision=None,
    ) -> "AudioModel":
        """`state_dict`: `audio_tower.*` and `embed_audio.*` keys (weights.load_audio_state_dict).

        num_layers < config.num_layers builds a truncated stack (debug / per-layer tests only).
        precision: None (default) loads `doc/datatype_sweep/selected_precision_config.json` (or the
        file named by EMBEDDING_GEMMA2_PRECISION); a path, dict or PrecisionConfig otherwise. Its
        `audio` section sets every weight dtype, compute config and projection output dtype.
        act_dtype (None: from the policy) overrides the output dtype of the activation-stream
        linears (precision.AUDIO_ACT_GROUPS: input_proj, the FFN / lconv / attention-output
        linears), i.e. the dtype of the conformer activations between ops. bf16 or fp32; the fp32
        variant is a measured precision lever (doc/functional_audio/README.md).
        """
        precision = resolve_precision(precision)
        ap = precision.audio
        n = config.num_layers if num_layers is None else num_layers
        if act_dtype not in (None, ttnn.bfloat16, ttnn.float32):
            raise ValueError(f"act_dtype must be None, bfloat16 or float32, got {act_dtype}")
        tower = substate(state_dict, "audio_tower")
        sub = AudioSubsample.from_state_dict(
            substate(tower, "subsample_conv_projection"),
            config,
            device,
            cache_dir=cache_dir,
            act_dtype=act_dtype,
            precision=ap,
        )
        shifts = load_causal_shifts(device, config.conv_kernel_size, cache_dir)  # shared by the lconv layers
        layers = [
            AudioConformerLayer.from_state_dict(
                substate(tower, f"layers.{i}"),
                config,
                i,
                device,
                cache_dir=cache_dir,
                act_dtype=act_dtype,
                shifts=shifts,
                precision=ap,
            )
            for i in range(n)
        ]
        out_proj = AudioOutputProj.from_state_dict(
            substate(tower, "output_proj"), config, device, cache_dir=cache_dir, precision=ap
        )
        embed = AudioEmbedder.from_state_dict(
            substate(state_dict, "embed_audio"), config, device, cache_dir=cache_dir, precision=ap
        )
        return cls(config, device, sub, layers, out_proj, embed, precision)

    def prepare(self, x: AudioDeviceInputs) -> None:
        """Setup for one (batch, bucket): lays out the conv weights (runs the subsampling stack once)."""
        self.subsample.prepare(x)

    def forward_hidden(self, x: AudioDeviceInputs) -> ttnn.Tensor:
        """Subsampling + conformer stack. [B,1,S,1024] act_dtype (padding rows not meaningful)."""
        h = self.subsample(x)
        # the rel-shift gather index is the same in every layer: one repeat per forward instead of one per
        # layer (E10.1 topology; EG2_AUDIO_SHIFT_IDX_ONCE=0 restores the per-layer build for A/B)
        # (no index when the rel-shift runs as matmuls: attention.REL_SHIFT_MM_TUNED, E10.2)
        idx = None
        if SHIFT_IDX_ONCE and self.layers:
            attn = self.layers[0].attn
            if attn.needs_shift_index(h.shape[0], h.shape[2], attn.qkv_policy.out_ttnn_dtype):
                idx = attn.shift_index_for(h.shape[0], h.shape[2])
        for layer in self.layers:
            h_next = layer(h, x.attn_mask, idx)
            ttnn.deallocate(h)
            h = h_next
        if idx is not None:
            ttnn.deallocate(idx)
        return h

    def forward(self, x: AudioDeviceInputs) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        h = self.forward_hidden(x)
        hidden = self.output_proj(h)
        ttnn.deallocate(h)
        soft = self.embedder(hidden)
        return soft, hidden
