# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""AudioEncoder: host-side orchestration around AudioModel (features, masks, upload, trace, readback).

Host work lives here and only here. From the processor outputs (`input_features` [B, T, 128] log-mel,
`input_features_mask` [B, T], True = valid frame; `extract_features` runs the HF feature extractor
on raw 16 kHz waveforms) `prepare_inputs` builds, per forward:
    mel         log-mel frames zero padded to the frame bucket 4S              [B,4S,128,1] fp32 RM
    frame_mask  1 for valid frames                                             [B,1,4S,1]   fp32
    mask1       frame_mask[::2] repeated over the 64 conv0 frequency rows       [1,1,B*2S*64,1] bf16
    attn_mask   additive band + key-validity mask (attention.attention_mask)   [B,1,S,64]   fp32
and the per-clip token counts (HF output mask = input mask[::2][::2]) used to strip padded rows
after readback. The relative-position table, per-dim query scale and clip bounds are load-time
constants inside the modules. The device forward (`AudioModel.forward`) issues no host ops.

Trace policy (same as Text/VisionEncoder): one trace per (batch, token bucket), all captured by one
`capture_all(shapes)` call that allocates every persistent input, prepares the conv weights and
compiles every shape before the first capture. `encode(..., traced=True)` refreshes the persistent
inputs with `copy_host_to_device_tensor` and replays. Any clip length in the same bucket reuses
the trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.audio.attention import attention_mask
from models.demos.blackhole.embedding_gemma2.tt.audio.audio_model import AudioDeviceInputs, AudioModel
from models.demos.blackhole.embedding_gemma2.tt.audio.config import SUBSAMPLE, AudioConfig, token_bucket_for
from models.demos.blackhole.embedding_gemma2.tt.audio.weights import load_audio_config_dicts, load_audio_state_dict
from models.demos.blackhole.embedding_gemma2.tt.config import default_weight_cache_dir
from models.demos.blackhole.embedding_gemma2.tt.weights import resolve_snapshot


@dataclass
class AudioHostInputs:
    tensors: AudioDeviceInputs  # host (not on device) ttnn tensors
    batch: int
    seq_len: int  # token bucket S
    num_frames: int  # logical T from the processor
    num_tokens: list[int]  # valid soft tokens per clip


@dataclass
class _Trace:
    trace_id: int
    inputs: AudioDeviceInputs
    soft: ttnn.Tensor
    hidden: ttnn.Tensor


def token_mask(frame_mask: torch.Tensor) -> torch.Tensor:
    """HF `Gemma4AudioSubSampleConvProjection` output mask: frame mask [B, T] -> [B, ceil(ceil(T/2)/2)]."""
    return frame_mask[:, ::2][:, ::2]


class AudioEncoder(LightweightModule):
    def __init__(self, model: AudioModel, device, snapshot: Path | None = None):
        super().__init__()
        self.model = model
        self.device = device
        self.config: AudioConfig = model.config
        self.snapshot = snapshot
        self._traces: dict[tuple[int, int], _Trace] = {}
        self._compiled: set[tuple[int, int]] = set()
        self._feature_extractor = None

    @classmethod
    def from_pretrained(
        cls,
        device,
        model_id_or_path: str = "google/embeddinggemma-2",
        *,
        cache_dir: Path | None = None,
        state_dict: dict[str, torch.Tensor] | None = None,
        act_dtype=ttnn.bfloat16,
    ) -> "AudioEncoder":
        snapshot = resolve_snapshot(model_id_or_path)
        acfg, text_hidden, mel_bins = load_audio_config_dicts(snapshot)
        config = AudioConfig.from_hf(acfg, text_hidden, mel_bins)
        if state_dict is None:
            state_dict = load_audio_state_dict(snapshot)
        cache_dir = Path(cache_dir) if cache_dir is not None else default_weight_cache_dir() / snapshot.name
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = AudioModel.from_state_dict(state_dict, config, device, cache_dir=cache_dir, act_dtype=act_dtype)
        return cls(model, device, snapshot)

    # ---------------------------------------------------------------- host side
    def extract_features(self, waveforms: list, sampling_rate: int = 16000) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw mono waveforms -> (input_features [B,T,128] fp32, input_features_mask [B,T] bool), via the
        checkpoint's HF feature extractor (the processor; host only)."""
        if self._feature_extractor is None:
            if self.snapshot is None:
                raise RuntimeError("extract_features needs the snapshot path (use from_pretrained)")
            from transformers import AutoFeatureExtractor

            self._feature_extractor = AutoFeatureExtractor.from_pretrained(str(self.snapshot))
        out = self._feature_extractor(
            [w.numpy() if isinstance(w, torch.Tensor) else w for w in waveforms],
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        return out["input_features"].float(), out["input_features_mask"].bool()

    def prepare_inputs(self, input_features: torch.Tensor, input_features_mask: torch.Tensor) -> AudioHostInputs:
        """Processor outputs -> host ttnn tensors for one forward (no device work)."""
        cfg = self.config
        if input_features.dim() != 3 or input_features.shape[-1] != cfg.num_mel_bins:
            raise ValueError(f"input_features must be [B, T, {cfg.num_mel_bins}], got {tuple(input_features.shape)}")
        B, T, F = input_features.shape
        if tuple(input_features_mask.shape) != (B, T):
            raise ValueError(f"input_features_mask must be [{B}, {T}], got {tuple(input_features_mask.shape)}")
        tmask = token_mask(input_features_mask.bool())
        S = token_bucket_for(tmask.shape[1])
        frames = SUBSAMPLE * S
        fm = torch.zeros((B, frames), dtype=torch.float32)
        fm[:, :T] = input_features_mask.float()
        mel = torch.zeros((B, frames, F, 1), dtype=torch.float32)
        mel[:, :T, :, 0] = input_features.float()
        mask1 = fm[:, ::2].reshape(B, frames // 2, 1).expand(B, frames // 2, F // 2).reshape(1, 1, -1, 1)
        # valid tokens form a prefix (right padding); attention keys are masked by count
        num_tokens = tmask.sum(-1).tolist()
        if not all(bool(tmask[b, :n].all()) for b, n in enumerate(num_tokens)):
            raise ValueError("audio token mask is not a prefix (the processor pads on the right)")
        amask = attention_mask(num_tokens, S, cfg.attention)

        def tile(t, dt):
            return ttnn.from_torch(t.contiguous(), dtype=dt, layout=ttnn.TILE_LAYOUT)

        tensors = AudioDeviceInputs(
            mel=ttnn.from_torch(mel, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
            frame_mask=tile(fm.reshape(B, 1, frames, 1), ttnn.float32),
            mask1=tile(mask1, ttnn.bfloat16),
            attn_mask=tile(amask, ttnn.float32),
        )
        return AudioHostInputs(tensors=tensors, batch=B, seq_len=S, num_frames=T, num_tokens=num_tokens)

    def to_device(self, host: AudioHostInputs) -> AudioDeviceInputs:
        dram = ttnn.DRAM_MEMORY_CONFIG
        return AudioDeviceInputs(*(ttnn.to_device(t, self.device, memory_config=dram) for t in host.tensors.tensors()))

    @staticmethod
    def read_rows(t: ttnn.Tensor, counts: list[int]) -> list[torch.Tensor]:
        """[B,1,S,D] device -> per-clip [counts[b], D] float32 (padded rows stripped)."""
        x = ttnn.to_torch(t).float()
        x = x.reshape(x.shape[0], x.shape[-2], x.shape[-1])
        return [x[b, :n] for b, n in enumerate(counts)]

    # ---------------------------------------------------------------- eager
    def prepare(self, x: AudioDeviceInputs) -> None:
        """Setup for the shape of `x` (conv weight layout). Must run before any trace capture."""
        key = (x.mel.shape[0], x.mel.shape[1] // SUBSAMPLE)
        if self._traces and not self.model.subsample.prepared(x.mel.shape[0], x.mel.shape[1]):
            raise RuntimeError(f"shape {key} was not prepared before the traces were captured; use capture_all()")
        self.model.prepare(x)

    def forward(self, x: AudioDeviceInputs) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Device-only forward on already uploaded inputs (the measured unit)."""
        return self.model(x)

    def encode(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        *,
        traced: bool = False,
        return_hidden: bool = False,
    ):
        """Soft tokens per clip: list of [n_b, 512] float32 (n_b = valid tokens)."""
        host = self.prepare_inputs(input_features, input_features_mask)
        if traced:
            soft, hidden = self.replay(host)
        else:
            key = (host.batch, host.seq_len)
            if self._traces and key not in self._compiled:
                raise RuntimeError(
                    f"eager shape {key} was not compiled before the traces were captured; its new "
                    "program-cache buffers could overlap trace buffers. Include it in capture_all()."
                )
            x = self.to_device(host)
            self.prepare(x)
            soft, hidden = self.forward(x)
            self._compiled.add(key)
        out = self.read_rows(soft, host.num_tokens)
        if return_hidden:
            return out, self.read_rows(hidden, host.num_tokens)
        return out

    # ---------------------------------------------------------------- trace
    def _dummy_host(self, batch: int, seq_len: int) -> AudioHostInputs:
        """A full-length input of shape (batch, bucket) for allocation / compile / capture."""
        frames = SUBSAMPLE * seq_len
        g = torch.Generator().manual_seed(0)
        feats = torch.randn((batch, frames, self.config.num_mel_bins), generator=g) - 4.0
        return self.prepare_inputs(feats, torch.ones((batch, frames), dtype=torch.bool))

    def capture_all(self, shapes) -> None:
        """Capture one trace per (batch, token bucket) in `shapes`.

        As in VisionEncoder.capture_all: allocate every persistent input, prepare the conv weights
        and compile every shape with one eager pass before the first capture, so no allocation made
        after a capture can land in that trace's buffers. Call once per session; eager runs
        afterwards are allowed only for shapes listed here. Trace outputs are rewritten by their own
        replay; read them before replaying another trace.
        """
        if self._traces:
            raise RuntimeError("capture_all() may run once, before any other trace is captured")
        shapes = sorted(set((int(b), int(s)) for b, s in shapes))
        inputs = {key: self.to_device(self._dummy_host(*key)) for key in shapes}
        for key in shapes:
            self.model.prepare(inputs[key])
        for key in shapes:
            soft, hidden = self.forward(inputs[key])
            ttnn.synchronize_device(self.device)
            ttnn.deallocate(soft)
            ttnn.deallocate(hidden)
            self._compiled.add(key)
        from ttnn.tools.trace_allocation_tracker import acknowledge_corruptible

        for key in shapes:
            trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
            soft, hidden = self.forward(inputs[key])
            ttnn.end_trace_capture(self.device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.device)
            acknowledge_corruptible(soft)
            acknowledge_corruptible(hidden)
            self._traces[key] = _Trace(trace_id, inputs[key], soft, hidden)

    def replay(self, host: AudioHostInputs, *, blocking: bool = False) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        key = (host.batch, host.seq_len)
        if key not in self._traces:
            raise KeyError(f"no trace for (batch, bucket) {key}; captured: {sorted(self._traces)}")
        tr = self._traces[key]
        for src, dst in zip(host.tensors.tensors(), tr.inputs.tensors()):
            ttnn.copy_host_to_device_tensor(src, dst)
        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=blocking)
        return tr.soft, tr.hidden

    def execute_captured(self, batch: int, seq_len: int, *, blocking: bool = False):
        """Replay a captured trace on whatever its persistent inputs hold (perf loop helper)."""
        tr = self._traces[(batch, seq_len)]
        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=blocking)
        return tr.soft, tr.hidden

    def release_traces(self) -> None:
        for tr in self._traces.values():
            ttnn.release_trace(self.device, tr.trace_id)
        self._traces.clear()
