import json
import math
import os

import librosa
import numpy as np
import parselmouth
import torch
import torch.nn.functional as F
import torchcrepe
from safetensors.torch import load_file
from scipy import signal

from rvc.audio import load_audio
from rvc.configs.config import Config
from rvc.synthesizer.models import SynthesizerTrnMsNSF, SynthesizerTrnMsNSF_nono
from rvc.vc.utils import load_hubert

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


def _get_hubert_paths():
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
    config: Config,
    if_f0: bool,
    version: str,
    num: str = "48k",
):
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
    synthesizer = synthesizer_class[if_f0](**synthesizer_config["model"])
    synthesizer.load_state_dict(synthesizer_state, strict=True)
    synthesizer.eval().to(config.device)
    synthesizer = synthesizer.float()
    return synthesizer, synthesizer_config["data"]


class Pipeline:
    def __init__(
        self,
        if_f0: bool = True,
        version: str = "v1",
        num: str = "48k",
        config: Config | None = None,
    ):

        hubert_cfg_path, hubert_path = _get_hubert_paths()
        if not os.path.exists(hubert_path):
            raise FileNotFoundError("hubert_path not found.")
        self.config = config or Config()
        self.if_f0 = if_f0
        self.version = version
        self.num = num

        self.synthesizer, data_cfg = _load_synthesizer(self.config, self.if_f0, self.version, self.num)
        self.tgt_sr = data_cfg["sampling_rate"]
        self.hubert_model = load_hubert(self.config, hubert_path, hubert_cfg_path)
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
        elif f0_method == "crepe":
            model = "full"
            batch_size = 512
            audio = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio,
                self.sr,
                self.window,
                f0_min,
                f0_max,
                model,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                from rvc.lib.rmvpe import RMVPE

                self.model_rmvpe = RMVPE(
                    f"{os.environ['rmvpe_root']}/rmvpe.pt",
                    device=self.device,
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        else:
            raise ValueError("f0_method must be 'pm', 'crepe', or 'rmvpe'.")
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
        feats = torch.from_numpy(audio)
        feats = feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        with torch.no_grad():
            logits = self.hubert_model.extract_features(
                source=feats.to(self.device),
                padding_mask=padding_mask,
                output_layer=9 if self.version == "v1" else 12,
            )
            feats = self.hubert_model.final_proj(logits) if self.version == "v1" else logits
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
        p_len = audio.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        p_len = torch.tensor([p_len], device=self.device).long()

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, speaker_id) if hasp else (feats, p_len, speaker_id)
            output = (self.synthesizer(*arg)[0, 0]).data.cpu().float().numpy()
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
                        audio_sum[t - self.t_query : t + self.t_query] == audio_sum[t - self.t_query : t + self.t_query].min()
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
            pitch, pitchf = self._get_f0(
                audio_padded,
                p_len,
                f0_up_key,
                f0_method,
            )
        for idx_s, idx_e in idx_list:
            pitch_slice = pitch[:, idx_s:idx_e] if pitch is not None else None
            pitchf_slice = pitchf[:, idx_s:idx_e] if pitchf is not None else None
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

        result = self._run_pipeline(
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

        return result
