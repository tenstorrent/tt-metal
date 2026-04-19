# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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

import ttnn
from models.demos.rvc.torch_impl.rmve import RMVEPitchAlgorithm
from models.demos.rvc.torch_impl.vc.pipeline import change_rms
from models.demos.rvc.tt_impl.vc.hubert import HubertModel
from models.demos.rvc.tt_impl.vc.synthesizer import SynthesizerTrnMsNSF, SynthesizerTrnMsNSF_nono
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


def _load_hubert(config, hubert_path: str, hubert_cfg_path: str, tt_device):
    task = HubertPretrainingTask(HubertPretrainingConfig())
    with open(hubert_cfg_path) as f:
        cfg = json.load(f)
    hubert_model = HubertModel(device=tt_device, cfg=cfg["model"], task_cfg=task.cfg)
    hubert_state_safetensors = load_file(hubert_path)
    hubert_model.load_state_dict(hubert_state_safetensors)
    return hubert_model.eval()


def _load_synthesizer(
    tt_device: ttnn.MeshDevice,
    if_f0: bool,
    version: str,
    num: str = "32k",
) -> tuple[SynthesizerTrnMsNSF | SynthesizerTrnMsNSF_nono, dict]:
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
        f0_method: F0Method = F0Method.RAPT,
        index_rate: float = 0.75,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ):
        hubert_cfg_path, hubert_path = get_hubert_paths()
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
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect
        self._rmve_pitch_algorithm: RMVEPitchAlgorithm | None = None

        if self.tt_device.get_num_devices() > 1:
            self.input_mesh_mapper = ttnn.ShardTensorToMesh(self.tt_device, dim=0)
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.tt_device, dim=0)
        else:
            self.input_mesh_mapper = None
            self.output_mesh_composer = None

        self.synthesizer, data_cfg = _load_synthesizer(self.tt_device, self.if_f0, self.version, self.num)
        self.tgt_sr = data_cfg["sampling_rate"]
        self.hubert_model = _load_hubert(self.config, hubert_path, hubert_cfg_path, self.tt_device)
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
        batch_size = audio.shape[0]
        speaker_id = torch.full((batch_size,), self.speaker_id, device=self.device, dtype=torch.long)
        speaker_id = ttnn.from_torch(
            speaker_id,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.tt_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=self.input_mesh_mapper,
        )
        hubert_input = ttnn.from_torch(
            audio.view(batch_size, -1, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.tt_device,
            mesh_mapper=self.input_mesh_mapper,
        )
        logits = self.hubert_model(
            source=hubert_input,
            output_layer=9 if self.version == "v1" else 12,
        )
        feats = self.hubert_model.final_proj(logits) if self.version == "v1" else logits

        feats = ttnn.to_layout(feats, ttnn.ROW_MAJOR_LAYOUT)

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
        num_frames = feats.shape[1]

        if pitch is not None and pitchf is not None:
            pitch = pitch[:, :num_frames]
            pitchf = pitchf[:, :num_frames]
            pitch = ttnn.from_torch(
                pitch,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.tt_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=self.input_mesh_mapper,
            )
            pitchf = ttnn.from_torch(
                pitchf,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.tt_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=self.input_mesh_mapper,
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

        output_torch = ttnn.to_torch(output, mesh_composer=self.output_mesh_composer).to(torch.float32)
        output = output_torch[:, :, 0].contiguous()
        return output

    def _run_pipeline(self, audio):
        assert audio.dim() == 2, audio.dim()
        index = big_npy = None
        audio_padded = F.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        num_frames = audio_padded.shape[1] // self.window
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
        if audio_max > 1:
            max_int16 /= audio_max
        # use torch
        audio_output = (audio_output * max_int16).to(torch.int16)

        return audio_output

    def infer(self):
        audio = load_audio(self.sr)
        # preprocess
        audio_max = torch.abs(audio).max().item() / 0.95
        if audio_max > 1:
            audio /= audio_max

        audio = signal.filtfilt(bh, ah, audio)
        audio = torch.from_numpy(audio.copy()).unsqueeze(0)
        return self._run_pipeline(audio)[0]
