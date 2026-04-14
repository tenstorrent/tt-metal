# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os

import librosa
import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from scipy import signal

import ttnn
from models.demos.rvc.torch_impl.audio import load_audio
from models.demos.rvc.torch_impl.configs.config import Config
from models.demos.rvc.tt_impl.vc.synthesizer import SynthesizerTrnMsNSF, SynthesizerTrnMsNSF_nono
from models.demos.rvc.tt_impl.vc.utils import load_hubert

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


def _frame_signal(
    y: np.ndarray,
    *,
    frame_length: int,
    hop_length: int,
    center: bool,
    pad_mode: str,
) -> np.ndarray:
    if y.ndim != 1:
        raise ValueError(f"Expected a 1D audio signal, got shape {y.shape}")
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if hop_length <= 0:
        raise ValueError("hop_length must be positive")

    if center:
        pad = frame_length // 2
        y = np.pad(y, (pad, pad), mode=pad_mode)
    elif y.shape[0] < frame_length:
        raise ValueError("len(y) must be at least frame_length when center=False")

    if y.shape[0] < frame_length:
        raise ValueError("Padded signal is shorter than frame_length")

    n_frames = 1 + (y.shape[0] - frame_length) // hop_length
    frame_starts = hop_length * np.arange(n_frames)
    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[frame_starts]
    return frames


def rms_numpy(
    y,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = "constant",
    dtype=np.float32,
) -> np.ndarray:
    """NumPy reimplementation of ``librosa.feature.rms(y=...)`` for 1D audio.

    Returns an array of shape ``(1, num_frames)`` to match librosa.
    """

    y = np.asarray(y, dtype=dtype)
    frames = _frame_signal(
        y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=center,
        pad_mode=pad_mode,
    )
    power = np.mean(np.square(frames), axis=1, dtype=np.float64)
    rms = np.sqrt(power, dtype=np.float64).astype(dtype, copy=False)
    return rms[np.newaxis, :]


def _interpolate_1d(
    x: ttnn.Tensor,
    scale_factor: int | float,
    mode: str = "nearest",
) -> ttnn.Tensor:
    # 1D upsample for [N, L, C] via 2D NHWC upsample with height fixed to 1.
    if mode not in ("nearest", "linear"):
        raise ValueError(f"Unsupported 1D interpolate mode: {mode}")
    upsample_mode = "nearest" if mode == "nearest" else "bilinear"
    x_nhwc = ttnn.unsqueeze(x, dim=1)
    y_nhwc = ttnn.upsample(
        x_nhwc,
        [1, scale_factor],
        mode=upsample_mode,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return ttnn.squeeze(y_nhwc, dim=1)


def _get_configs_dir() -> str:
    configs_dir = os.getenv("RVC_CONFIGS_DIR")
    if not configs_dir:
        raise OSError("RVC_CONFIGS_DIR is not set. Set it to the directory containing v1/ and v2/ config folders.")
    if not os.path.isdir(configs_dir):
        raise FileNotFoundError(f"RVC_CONFIGS_DIR does not exist: {configs_dir}")
    return configs_dir


def _get_assets_dir() -> str:
    assets_dir = os.getenv("RVC_ASSETS_DIR")
    if not assets_dir:
        raise OSError("RVC_ASSETS_DIR is not set. Set it to the directory containing pretrained model weights.")
    if not os.path.isdir(assets_dir):
        raise FileNotFoundError(f"RVC_ASSETS_DIR does not exist: {assets_dir}")
    return assets_dir


def _build_path_mapping() -> dict[tuple[str, str, bool], tuple[str, str]]:
    configs_dir = _get_configs_dir()
    assets_dir = _get_assets_dir()
    pretrained_dir = os.path.join(assets_dir, "pretrained")
    return {
        ("v1", "32k", True): (
            os.path.join(pretrained_dir, "f0G32k.safetensors"),
            os.path.join(configs_dir, "v1/32k.json"),
        ),
        ("v1", "40k", True): (
            os.path.join(pretrained_dir, "f0G40k.safetensors"),
            os.path.join(configs_dir, "v1/40k.json"),
        ),
        ("v1", "48k", True): (
            os.path.join(pretrained_dir, "f0G48k.safetensors"),
            os.path.join(configs_dir, "v1/48k.json"),
        ),
        ("v1", "48k", False): (
            os.path.join(pretrained_dir, "G48k.safetensors"),
            os.path.join(configs_dir, "v1/48k.json"),
        ),
    }


def _get_hubert_paths() -> tuple[str, str]:
    configs_dir = _get_configs_dir()
    assets_dir = _get_assets_dir()
    return (
        os.path.join(configs_dir, "hubert_cfg.json"),
        os.path.join(assets_dir, "hubert.safetensors"),
    )


path_mapping = _build_path_mapping()


def _get_model_and_config_paths(version: str, num: str, if_f0: bool) -> tuple[str, str]:
    key = (version, num, if_f0)
    if key not in path_mapping:
        raise KeyError(f"Unsupported path mapping key: {key}")
    return path_mapping[key]


def _change_rms(source_audio, source_sr, target_audio, target_sr, rate):
    source_rms = rms_numpy(source_audio, frame_length=source_sr // 2 * 2, hop_length=source_sr // 2)
    target_rms = rms_numpy(target_audio, frame_length=target_sr // 2 * 2, hop_length=target_sr // 2)
    source_rms = torch.from_numpy(source_rms)
    source_rms = F.interpolate(source_rms.unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    target_rms = torch.from_numpy(target_rms)
    target_rms = F.interpolate(target_rms.unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
    target_rms = torch.max(target_rms, torch.zeros_like(target_rms) + 1e-6)
    target_audio *= torch.pow(source_rms, torch.tensor(1 - rate)) * torch.pow(target_rms, torch.tensor(rate - 1))
    return target_audio


def _load_synthesizer(
    tt_device: ttnn.MeshDevice,
    if_f0: bool,
    version: str,
    num: str = "32k",
) -> tuple[SynthesizerTrnMsNSF | SynthesizerTrnMsNSF_nono, dict]:
    synthesizer_path, synthesizer_cfg_path = _get_model_and_config_paths(version, num, if_f0)
    synthesizer_state = load_file(synthesizer_path)

    with open(synthesizer_cfg_path) as f:
        synthesizer_config = json.load(f)
    synthesizer_config["model"]["embedding_dims"] = 256 if version == "v1" else 768
    synthesizer_config["model"]["sr"] = synthesizer_config["data"]["sampling_rate"]

    synthesizer_class = {
        1: SynthesizerTrnMsNSF,
        0: SynthesizerTrnMsNSF_nono,
    }
    synthesizer = synthesizer_class[if_f0](device=tt_device, **synthesizer_config["model"])
    synthesizer.load_state_dict(state_dict=synthesizer_state)
    return synthesizer, synthesizer_config["data"]


class Pipeline:
    """Hybrid TT VC pipeline: Torch huBERT frontend + TT synthesizer backend."""

    def __init__(
        self,
        tt_device: ttnn.MeshDevice,
        if_f0: bool = True,
        version: str = "v1",
        num: str = "48k",
        config: Config | None = None,
        speaker_id: int = 0,
        f0_up_key: int = 0,
        f0_method: str = "pm",
        index_rate: float = 0.75,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ):
        hubert_cfg_path, hubert_path = _get_hubert_paths()
        if not os.path.exists(hubert_path):
            raise FileNotFoundError("hubert_path not found.")

        self.config = config or Config()
        self.tt_device = tt_device
        self.if_f0 = if_f0
        self.version = version
        self.num = num
        self.speaker_id = speaker_id
        self.f0_up_key = f0_up_key
        self.f0_method = f0_method
        self.index_rate = index_rate
        self.resample_sr = resample_sr
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect

        self.synthesizer, data_cfg = _load_synthesizer(self.tt_device, self.if_f0, self.version, self.num)
        self.tgt_sr = data_cfg["sampling_rate"]
        self.hubert_model = load_hubert(self.config, hubert_path, hubert_cfg_path, self.tt_device)
        self._init_timing(self.tgt_sr, self.config)

    def _init_timing(self, tgt_sr: int, config: Config) -> None:
        x_pad, x_query, x_center, x_max = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * x_pad
        self.t_pad_tgt = tgt_sr * x_pad
        self.t_query = self.sr * x_query
        self.t_center = self.sr * x_center
        self.t_max = self.sr * x_max
        self.device = config.device

    def _get_f0(self, audio, num_frames):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * math.log(1 + f0_min / 700)
        f0_mel_max = 1127 * math.log(1 + f0_max / 700)
        if self.f0_method == "pm":
            f0 = (
                parselmouth.Sound(audio, self.sr)
                .to_pitch_ac(
                    time_step=self.window / self.sr,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            f0 = torch.from_numpy(f0)
            pad_size = (num_frames - len(f0) + 1) // 2
            if pad_size > 0 or num_frames - len(f0) - pad_size > 0:
                f0 = F.pad(f0, (pad_size, num_frames - len(f0) - pad_size), mode="constant")
        else:
            raise ValueError("f0_method must be 'pm'.")

        f0 *= pow(2, self.f0_up_key / 12)
        f0_continuous = f0.clone()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel)

        f0_coarse = f0_coarse[:num_frames]
        f0_continuous = f0_continuous[:num_frames]
        f0_coarse = f0_coarse.unsqueeze(0).long()
        f0_continuous = f0_continuous.unsqueeze(0).float()
        return f0_coarse, f0_continuous

    def _get_time_stamps(self, audio, num_frames):
        audio_padded = F.pad(audio.unsqueeze(0), (self.window // 2, self.window // 2), mode="reflect").squeeze(0)
        opt_ts = []
        if audio_padded.shape[0] > self.t_max:
            audio_avg = F.avg_pool1d(
                audio_padded.abs().view(1, 1, -1),
                kernel_size=self.window,
                stride=1,
            ).view(-1)
            for t in range(self.t_center, audio.shape[0], self.t_center):
                segment = audio_avg[t - self.t_query : t + self.t_query]
                opt_ts.append(t - self.t_query + torch.argmin(segment).item())
        s = 0
        idx_list = []
        for t in opt_ts:
            idx_list.append((s // self.window, (t + self.t_pad * 2) // self.window))
            s = t
        idx_list.append((s // self.window, num_frames))

        return idx_list

    def _vc(
        self,
        audio,
        pitch,
        pitchf,
        index,
        big_npy,
    ):
        speaker_id = torch.tensor(self.speaker_id, device=self.device).unsqueeze(0).long()
        speaker_id = ttnn.from_torch(
            speaker_id,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        assert audio.dim() == 1, audio.dim()
        hubert_input = ttnn.from_torch(
            audio.view(1, -1, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.tt_device,
        )
        logits = self.hubert_model(
            source=hubert_input,
            output_layer=9 if self.version == "v1" else 12,
        )
        feats = self.hubert_model.final_proj(logits) if self.version == "v1" else logits

        feats = ttnn.to_layout(feats, ttnn.ROW_MAJOR_LAYOUT)
        num_frames = feats.shape[1] * 2

        if self.protect < 0.5 and pitch is not None and pitchf is not None:
            protected_features = _interpolate_1d(feats, scale_factor=2, mode="linear")
        if index is not None and big_npy is not None and self.index_rate != 0:
            index_features = feats[0].cpu().numpy()
            scores, indices = index.search(index_features, k=8)
            scores, indices = torch.from_numpy(scores), torch.from_numpy(indices)
            weights = torch.square(1 / scores)
            weights /= weights.sum(dim=1, keepdim=True)
            index_features = torch.sum(big_npy[indices] * weights.unsqueeze(2), dim=1)
            feats = index_features * self.index_rate + (1 - self.index_rate) * feats

        feats = _interpolate_1d(feats, scale_factor=2, mode="linear")

        if pitch is not None and pitchf is not None:
            pitch = pitch[:, :num_frames]
            pitchf = pitchf[:, :num_frames]
            pitch = ttnn.from_torch(
                pitch,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.tt_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            pitchf = ttnn.from_torch(
                pitchf,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.tt_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            if self.protect < 0.5:
                pitchff = self.protect + (1 - self.protect) * (pitchf >= 1)
                pitchff = ttnn.unsqueeze(pitchff, -1)
                feats = feats * pitchff + protected_features * ttnn.rsub(pitchff, 1)
                ttnn.deallocate(protected_features)
            feats = ttnn.reallocate(feats)
            pitch = ttnn.reallocate(pitch)
            pitchf = ttnn.reallocate(pitchf)
            speaker_id = ttnn.reallocate(speaker_id)
            output = self.synthesizer(feats, pitch, pitchf, speaker_id)
        else:
            output = self.synthesizer(feats, speaker_id)

        output_torch = ttnn.to_torch(output).to(torch.float32)
        output = output_torch[0, :, 0].contiguous()
        return output

    def _run_pipeline(
        self,
        audio,
    ):
        index = big_npy = None
        audio_padded = F.pad(audio.unsqueeze(0), (self.t_pad, self.t_pad), mode="reflect").squeeze(0)
        num_frames = audio_padded.shape[0] // self.window
        idx_list = self._get_time_stamps(audio, num_frames)
        pitch, pitchf = None, None
        if self.if_f0:
            pitch, pitchf = self._get_f0(audio_padded, num_frames)

        audio_output = []
        for idx_s, idx_e in idx_list:
            if self.if_f0:
                chunk_end = min(idx_e, pitch.shape[1])
                pitch_slice = pitch[:, idx_s:chunk_end]
                pitchf_slice = pitchf[:, idx_s:chunk_end]
            else:
                pitch_slice = None
                pitchf_slice = None
            audio_output.append(
                self._vc(
                    audio_padded[idx_s * self.window : idx_e * self.window],
                    pitch_slice,
                    pitchf_slice,
                    index,
                    big_npy,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )

        audio_output = torch.cat(audio_output)

        # post-process
        if self.rms_mix_rate != 1:
            audio_output = _change_rms(audio, 16000, audio_output, self.tgt_sr, self.rms_mix_rate)
        if self.tgt_sr != self.resample_sr and self.resample_sr >= 16000:
            audio_output_np = audio_output.numpy()
            audio_output_np = librosa.resample(audio_output_np, orig_sr=self.tgt_sr, target_sr=self.resample_sr)
            audio_output = torch.from_numpy(audio_output_np)
        audio_max = torch.abs(audio_output).max().item() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        # use torch
        audio_output = (audio_output * max_int16).to(torch.int16)

        return audio_output

    def infer(self, audio_path: str):
        if not os.path.exists(audio_path):
            raise FileNotFoundError("input_audio_path not found.")

        audio = load_audio(audio_path, 16000)
        # preprocess
        audio_max = torch.abs(audio).max().item() / 0.95
        if audio_max > 1:
            audio /= audio_max

        audio = signal.filtfilt(bh, ah, audio)
        audio = torch.from_numpy(audio.copy())

        return self._run_pipeline(audio)
