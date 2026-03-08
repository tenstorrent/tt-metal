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
from models.demos.rvc.tt_impl.synthesizer.models import SynthesizerTrnMsNSF, SynthesizerTrnMsNSF_nono
from models.demos.rvc.tt_impl.vc.utils import load_hubert

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


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


def _change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))).numpy()
    return data2


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
    synthesizer.load_parameters(parameters=synthesizer_state)
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
    ):
        hubert_cfg_path, hubert_path = _get_hubert_paths()
        if not os.path.exists(hubert_path):
            raise FileNotFoundError("hubert_path not found.")

        self.config = config or Config()
        self.tt_device = tt_device
        self.if_f0 = if_f0
        self.version = version
        self.num = num

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
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * x_query
        self.t_center = self.sr * x_center
        self.t_max = self.sr * x_max
        self.device = config.device

    def _get_f0(self, x, p_len, f0_up_key, f0_method):
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * math.log(1 + f0_min / 700)
        f0_mel_max = 1127 * math.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=self.window / self.sr,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        else:
            raise ValueError("f0_method must be 'pm'.")

        f0 *= pow(2, f0_up_key / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel)

        f0_coarse = f0_coarse[:p_len]
        f0bak = f0bak[:p_len]
        f0_coarse = torch.tensor(f0_coarse, device=self.device).unsqueeze(0).long()
        f0bak = torch.tensor(f0bak, device=self.device).unsqueeze(0).float()
        return f0_coarse, f0bak

    def _vc(
        self,
        audio,
        speaker_id,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        protect,
    ):
        feats = torch.from_numpy(audio).float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1, 1)
        hubert_input = ttnn.from_torch(
            feats,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.tt_device,
        )
        with torch.no_grad():
            logits = self.hubert_model(
                source=hubert_input,
                output_layer=9 if self.version == "v1" else 12,
            )
            feats = self.hubert_model.final_proj(logits) if self.version == "v1" else logits

        feats = ttnn.to_torch(feats).to(torch.float32)

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if index is not None and big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        p_len = feats.shape[1]
        if pitch is not None and pitchf is not None:
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        with torch.no_grad():
            tt_feats = ttnn.from_torch(
                feats,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.tt_device,
            )
            tt_speaker_id = ttnn.from_torch(
                speaker_id,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.tt_device,
            )
            if pitch is not None and pitchf is not None:
                tt_pitch = ttnn.from_torch(
                    pitch,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.tt_device,
                )
                tt_pitchf = ttnn.from_torch(
                    pitch,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.tt_device,
                )
                tt_output = self.synthesizer(tt_feats, tt_pitch, tt_pitchf, tt_speaker_id)
            else:
                tt_output = self.synthesizer(tt_feats, tt_speaker_id)

            output_torch = ttnn.to_torch(tt_output).to(torch.float32)
            output = output_torch[0, :, 0].contiguous().cpu().numpy()
        return output

    def _run_pipeline(
        self,
        audio,
        speaker_id,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
    ):
        index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_padded = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_padded.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += np.abs(audio_padded[i : i - self.window])
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query : t + self.t_query]
                        == audio_sum[t - self.t_query : t + self.t_query].min()
                    )[0][0]
                )

        audio_output = []
        audio_padded = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_padded.shape[0] // self.window
        s = 0
        idx_list = []
        for t in opt_ts:
            idx_list.append((s // self.window, (t + self.t_pad2) // self.window))
            s = t
        idx_list.append((s // self.window, p_len))
        speaker_id = torch.tensor(speaker_id, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if self.if_f0:
            pitch, pitchf = self._get_f0(audio_padded, p_len, f0_up_key, f0_method)

        for idx_s, idx_e in idx_list:
            chunk_end = min(idx_e + self.t_pad2 // self.window, pitch.shape[1]) if pitch is not None else idx_e
            pitch_slice = pitch[:, idx_s:chunk_end] if pitch is not None else None
            pitchf_slice = pitchf[:, idx_s:chunk_end] if pitchf is not None else None
            audio_output.append(
                self._vc(
                    audio_padded[idx_s * self.window : (idx_e + self.t_pad2 // self.window) * self.window],
                    speaker_id,
                    pitch_slice,
                    pitchf_slice,
                    index,
                    big_npy,
                    index_rate,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )

        audio_output = np.concatenate(audio_output)
        if rms_mix_rate != 1:
            audio_output = _change_rms(audio, 16000, audio_output, self.tgt_sr, rms_mix_rate)
        if self.tgt_sr != resample_sr and resample_sr >= 16000:
            audio_output = librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=resample_sr)
        audio_max = np.abs(audio_output).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_output = (audio_output * max_int16).astype(np.int16)
        return audio_output

    def infer(
        self,
        audio_path: str,
        speaker_id: int,
        f0_up_key: int = 0,
        f0_method: str = "pm",
        index_file: str | None = None,
        index_rate: float = 0.75,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ):
        if not os.path.exists(audio_path):
            raise FileNotFoundError("input_audio_path not found.")

        audio = load_audio(audio_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        return self._run_pipeline(
            audio,
            speaker_id,
            f0_up_key,
            f0_method,
            index_file,
            index_rate,
            resample_sr,
            rms_mix_rate,
            protect,
        )
