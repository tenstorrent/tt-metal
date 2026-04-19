# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import math
import os

import numpy as np
import pyworld as pw
import torch
import torch.nn.functional as F
from pysptk import sptk
from safetensors.torch import load_file
from scipy import signal

from models.demos.rvc.torch_impl.rmve import RMVEPitchAlgorithm
from models.demos.rvc.torch_impl.vc.hubert import HubertModel
from models.demos.rvc.torch_impl.vc.synthesizer import SynthesizerTrnMsNSF, SynthesizerTrnMsNSF_nono
from models.demos.rvc.utils.audio import load_audio
from models.demos.rvc.utils.config import (
    Config,
    HubertPretrainingConfig,
    HubertPretrainingTask,
    get_hubert_paths,
    get_model_and_config_paths,
)
from models.demos.rvc.utils.f0 import F0Method

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


def _rms_numpy(
    y,
    *,
    frame_length: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    pad_mode: str = "constant",
    dtype=np.float32,
) -> np.ndarray:
    """NumPy reimplementation of ``librosa.feature.rms(y=...)`` for batched audio (2D).

    Returns an array of shape ``(batch_size, num_frames)`` to match librosa.
    """
    y = np.asarray(y, dtype=dtype)
    frames_list = []
    for i in range(y.shape[0]):
        frames = _frame_signal(
            y[i],
            frame_length=frame_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
        )
        frames_list.append(frames)
    frames = np.stack(frames_list, axis=0)

    power = np.mean(np.square(frames), axis=1, dtype=np.float64)
    rms = np.sqrt(power, dtype=np.float64).astype(dtype, copy=False)
    return rms


def change_rms(source_audio, source_sr, target_audio, target_sr, rate):
    source_rms = _rms_numpy(source_audio, frame_length=source_sr // 2 * 2, hop_length=source_sr // 2)
    target_rms = _rms_numpy(target_audio, frame_length=target_sr // 2 * 2, hop_length=target_sr // 2)
    source_rms = torch.from_numpy(source_rms)
    source_rms = F.interpolate(source_rms.unsqueeze(1), size=target_audio.shape[1], mode="linear").squeeze(1)
    target_rms = torch.from_numpy(target_rms)
    target_rms = F.interpolate(target_rms.unsqueeze(1), size=target_audio.shape[1], mode="linear").squeeze(1)
    target_rms = torch.max(target_rms, torch.zeros_like(target_rms) + 1e-6)
    target_audio *= torch.pow(source_rms, torch.tensor(1 - rate)) * torch.pow(target_rms, torch.tensor(rate - 1))
    return target_audio


def _load_synthesizer(
    config: Config,
    if_f0: bool,
    version: str,
    num: str = "48k",
):
    synthesizer_path, synthesizer_cfg_path = get_model_and_config_paths(version, num, if_f0)
    synthesizer_state = load_file(synthesizer_path)

    with open(synthesizer_cfg_path) as f:
        synthesizer_config = json.load(f)
    synthesizer_config["model"]["embedding_dims"] = 256 if version == "v1" else 768
    synthesizer_config["model"]["sr"] = synthesizer_config["data"]["sampling_rate"]
    synthesizer_class = {
        1: SynthesizerTrnMsNSF,
        0: SynthesizerTrnMsNSF_nono,
    }
    synthesizer = synthesizer_class[if_f0](**synthesizer_config["model"])
    synthesizer.load_state_dict(synthesizer_state, strict=True)
    synthesizer.eval().to(config.device)
    synthesizer = synthesizer.float()
    return synthesizer, synthesizer_config["data"]


def _load_hubert(config, hubert_path: str, hubert_cfg_path: str):
    task = HubertPretrainingTask(HubertPretrainingConfig())
    with open(hubert_cfg_path) as f:
        cfg = json.load(f)
    hubert_model = HubertModel(cfg=cfg["model"], task_cfg=task.cfg)
    hubert_state_safetensors = load_file(hubert_path)
    hubert_model.load_state_dict(hubert_state_safetensors, strict=True)
    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.float()
    return hubert_model.eval()


class Pipeline:
    def __init__(
        self,
        if_f0: bool = True,
        version: str = "v1",
        num: str = "48k",
        config: Config | None = None,
        speaker_id: int = 0,
        f0_up_key: int = 0,
        f0_method: F0Method = F0Method.RAPT,
        index_rate: float = 0.75,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ):
        hubert_cfg_path, hubert_path = get_hubert_paths()
        if not os.path.exists(hubert_path):
            raise FileNotFoundError("hubert_path not found.")
        self.config = config or Config()
        self.if_f0 = if_f0
        self.version = version
        self.num = num
        self.f0_up_key = f0_up_key
        self.f0_method = f0_method
        self.index_rate = index_rate
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect
        self.speaker_id = speaker_id
        self._rmve_pitch_algorithm: RMVEPitchAlgorithm | None = None

        self.synthesizer, data_cfg = _load_synthesizer(self.config, self.if_f0, self.version, self.num)
        self.tgt_sr = data_cfg["sampling_rate"]
        self.hubert_model = _load_hubert(self.config, hubert_path, hubert_cfg_path)
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
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * x_query
        self.t_center = self.sr * x_center
        self.t_max = self.sr * x_max
        self.device = config.device

    def _get_rmve_pitch_algorithm(self) -> RMVEPitchAlgorithm:
        if self._rmve_pitch_algorithm is None:
            device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
            self._rmve_pitch_algorithm = RMVEPitchAlgorithm(sample_rate=self.sr, hop_size=self.window)
        return self._rmve_pitch_algorithm

    def _get_f0(self, audio, num_frames):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * math.log(1 + f0_min / 700)
        f0_mel_max = 1127 * math.log(1 + f0_max / 700)
        if self.f0_method is F0Method.RAPT:
            audio_np = audio.detach().cpu().reshape(-1).numpy().astype(np.float32)
            audio_scaled = np.clip(audio_np * 32767, -32768, 32767)
            rapt_threshold = 0.525
            voice_bias = -0.6 + rapt_threshold * (0.7 - (-0.6))
            f0 = sptk.rapt(
                audio_scaled,
                self.sr,
                self.window,
                min=f0_min,
                max=f0_max,
                voice_bias=voice_bias,
                otype="f0",
            )
            f0 = torch.from_numpy(f0)
            f0 = f0.unsqueeze(0)
        elif self.f0_method is F0Method.DIO:
            audio_np = audio.detach().cpu().reshape(-1).numpy().astype(np.float64)
            frame_period = self.window / self.sr * 1000.0
            dio_threshold = 0.444
            allowed_range = 0.02 + dio_threshold * (0.2 - 0.02)
            f0, t = pw.dio(
                audio_np,
                self.sr,
                f0_floor=f0_min,
                f0_ceil=f0_max,
                frame_period=frame_period,
                allowed_range=allowed_range,
            )
            f0 = pw.stonemask(audio_np, f0, t, self.sr)
            f0 = torch.from_numpy(f0.astype(np.float32))
            f0 = f0.unsqueeze(0)
        elif self.f0_method is F0Method.RMVE:
            f0, _ = self._get_rmve_pitch_algorithm().extract_continuous_periodicity(audio)

        else:
            raise ValueError(f"Unsupported f0_method: {self.f0_method}, must be one of {list(F0Method)}")

        f0_len = f0.shape[1]
        if f0_len < num_frames:
            pad_size = (num_frames - f0_len + 1) // 2
            f0 = F.pad(f0, (pad_size, num_frames - f0_len - pad_size), mode="constant")
        else:
            f0 = f0[:, :num_frames]

        f0 *= pow(2, self.f0_up_key / 12)
        f0_continuous = f0.clone()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel = torch.clamp(f0_mel, min=1, max=255)
        f0_coarse = torch.round(f0_mel)

        return f0_coarse.to(torch.int64), f0_continuous.to(torch.float32)

    def _get_time_stamps(self, audio, num_frames):
        batch_size = audio.shape[0]
        audio_padded = F.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = [[] for _ in range(batch_size)]
        if audio_padded.shape[1] > self.t_max:
            audio_avg = F.avg_pool1d(
                audio_padded.abs().unsqueeze(1),
                kernel_size=self.window,
                stride=1,
            ).squeeze(1)
            for t in range(self.t_center, audio.shape[1], self.t_center):
                segment = audio_avg[:, t - self.t_query : t + self.t_query]
                argmin_indices = torch.argmin(segment, dim=1)
                for b_idx in range(batch_size):
                    opt_ts[b_idx].append((t - self.t_query + argmin_indices[b_idx].item()))

        idx_list = []
        for b_idx in range(batch_size):
            s = 0
            b_idx_list = []
            for t in opt_ts[b_idx]:
                b_idx_list.append((s // self.window, (t + self.t_pad * 2) // self.window))
                s = t
            b_idx_list.append((s // self.window, num_frames))
            idx_list.append(b_idx_list)
        return idx_list

    def _vc(
        self,
        audio,
        pitch,
        pitchf,
        index,
        big_npy,
    ):
        assert audio.dim() == 2, audio.dim()
        speaker_id = torch.tensor(self.speaker_id, device=self.device).unsqueeze(0).long()
        with torch.no_grad():
            logits = self.hubert_model(
                source=audio,
                output_layer=9 if self.version == "v1" else 12,
            )
            feats = self.hubert_model.final_proj(logits) if self.version == "v1" else logits

        if self.protect < 0.5 and pitch is not None and pitchf is not None:
            protected_features = feats.clone()
        if index is not None and big_npy is not None and self.index_rate != 0:
            index_features = feats[0].cpu().numpy()
            scores, indices = index.search(index_features, k=8)
            scores, indices = torch.from_numpy(scores), torch.from_numpy(indices)
            weights = torch.square(1 / scores)
            weights /= weights.sum(dim=1, keepdim=True)
            index_features = torch.sum(big_npy[indices] * weights.unsqueeze(2), dim=1)
            feats = index_features * self.index_rate + (1 - self.index_rate) * feats

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if self.protect < 0.5 and pitch is not None and pitchf is not None:
            protected_features = F.interpolate(protected_features.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        num_frames = feats.shape[1]

        if pitch is not None and pitchf is not None:
            pitch = pitch[:, :num_frames]
            pitchf = pitchf[:, :num_frames]
        if self.protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = self.protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + protected_features * (1 - pitchff)
            feats = feats.to(protected_features.dtype)

        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                output = self.synthesizer(feats, pitch, pitchf, speaker_id)
            else:
                output = self.synthesizer(feats, speaker_id)
        output = output[:, 0, :].contiguous()
        return output

    def _run_pipeline(self, audio):
        assert audio.dim() == 2, audio.dim()
        # if (
        #     file_index
        #     and file_index != ""
        #     and os.path.exists(file_index)
        #     and index_rate != 0
        # ):
        #     try:
        #         import faiss

        #         index = faiss.read_index(file_index)
        #         # big_npy = np.load(file_big_npy)
        #         big_npy = index.reconstruct_n(0, index.ntotal)
        #     except:
        #         traceback.print_exc()
        #         index = big_npy = None
        # else:
        index = big_npy = None
        audio_padded = F.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        num_frames = audio_padded.shape[1] // self.window
        opt_ts = []
        idx_list = self._get_time_stamps(audio, num_frames)
        pitch, pitchf = None, None
        if self.if_f0:
            pitch, pitchf = self._get_f0(audio_padded, num_frames)

        audio_output = []
        for idx_s, idx_e in idx_list[0]:
            if self.if_f0:
                chunk_end = min(idx_e, pitch.shape[1])
                pitch_slice = pitch[:, idx_s:chunk_end]
                pitchf_slice = pitchf[:, idx_s:chunk_end]
            else:
                pitch_slice = None
                pitchf_slice = None
            audio_output.append(
                self._vc(
                    audio_padded[:, idx_s * self.window : idx_e * self.window],
                    pitch_slice,
                    pitchf_slice,
                    index,
                    big_npy,
                )[:, self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_output = torch.cat(audio_output, dim=1)

        # post-process
        if self.rms_mix_rate != 1:
            audio_output = change_rms(audio, self.sr, audio_output, self.tgt_sr, self.rms_mix_rate)
        audio_max = torch.abs(audio_output).max().item() / 0.99
        max_int16 = 32768
        scale = max_int16 / audio_max if audio_max > 1 else 1
        audio_output = audio_output * scale
        return audio_output

    def infer(self):
        audio = load_audio(self.sr)
        audio_max = torch.abs(audio).max().item()
        if audio_max > 1:
            audio /= audio_max
        audio = signal.filtfilt(bh, ah, audio)
        audio = torch.from_numpy(audio.copy()).unsqueeze(0).to(torch.float32)
        return self._run_pipeline(audio)[0]
