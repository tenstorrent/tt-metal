# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np
import pyworld as pw
import torch
import torch.nn.functional as F
from pysptk import sptk
from safetensors.torch import load_file
from scipy import signal

import ttnn
from models.demos.rvc.torch_impl.vc.pipeline import change_rms
from models.demos.rvc.tt_impl.crepe import CrepePredictor
from models.demos.rvc.tt_impl.rmvpe import RMVPEPitchAlgorithm
from models.demos.rvc.tt_impl.vc.hubert import HubertModel
from models.demos.rvc.tt_impl.vc.synthesizer import SynthesizerTrnMsNSF, SynthesizerTrnMsNSF_nono
from models.demos.rvc.utils.audio import load_audio, load_audio_batch
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
) -> ttnn.Tensor:
    x_nhwc = ttnn.unsqueeze(x, dim=1)
    y_nhwc = ttnn.upsample(
        x_nhwc,
        [1, scale_factor],
        mode="nearest",
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return ttnn.squeeze(y_nhwc, dim=1)


def _load_hubert(config, hubert_path: str, hubert_cfg_path: str, device):
    task = HubertPretrainingTask(HubertPretrainingConfig())
    with open(hubert_cfg_path) as f:
        cfg = json.load(f)
    hubert_model = HubertModel(device=device, cfg=cfg["model"], task_cfg=task.cfg)
    hubert_state_safetensors = load_file(hubert_path)
    hubert_model.load_state_dict(hubert_state_safetensors)
    return hubert_model.eval()


def _load_synthesizer(
    device: ttnn.MeshDevice,
    if_f0: bool,
    version: str,
    num: str = "32k",
    validation: bool = False,
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
    synthesizer = synthesizer_class[if_f0](device=device, **synthesizer_config["model"], validation=validation)
    synthesizer.load_state_dict(state_dict=synthesizer_state)
    return synthesizer, synthesizer_config["data"]


class RVCRunner:
    def __init__(
        self,
        device: ttnn.MeshDevice,
        batch_size: int = 1,
        if_f0: bool = True,
        version: str = "v1",
        num: str = "48k",
        config: Config | None = None,
        index_rate: float = 0.75,
        protect: float = 0.33,
        validation: bool = False,
        performance_runner: bool = False,
    ):
        hubert_cfg_path, hubert_path = get_hubert_paths()
        if not os.path.exists(hubert_path):
            raise FileNotFoundError("hubert_path not found.")
        self.device = device
        self.version = version
        self.index_rate = index_rate
        config = config or Config()
        self.protect = protect

        self.sr = 16000
        self.window = 160
        self.x_center = config.x_center
        self.t_center = self.x_center * self.sr
        self.performance_runner = performance_runner
        self.num_devices = self.device.get_num_devices()
        if batch_size % self.num_devices != 0:
            raise ValueError(f"Batch size {batch_size} must be divisible by the number of devices {self.num_devices}.")

        self.batch_size_per_device = batch_size // self.num_devices

        self.synthesizer, data_cfg = _load_synthesizer(self.device, if_f0, self.version, num, validation=validation)
        self.tgt_sr = data_cfg["sampling_rate"]
        self.hubert_model = _load_hubert(config, hubert_path, hubert_cfg_path, self.device)
        self.torch_input = torch.randn((batch_size, self.t_center // self.batch_size_per_device, 1))
        self.torch_pitch = torch.randint(
            low=0,
            high=255,
            size=(batch_size, (self.t_center // self.batch_size_per_device) // self.window),
            dtype=torch.uint32,
        )
        self.torch_pitchf = torch.randn(
            (batch_size, (self.t_center // self.batch_size_per_device) // self.window), dtype=torch.float32
        )
        self.torch_speaker_id = torch.full((batch_size,), fill_value=0, dtype=torch.long)

        (
            tt_inputs_host,
            tt_pitch_host,
            tt_pitchf_host,
            tt_speaker_id_host,
            input_mem_config,
        ) = self.setup_dram_sharded_input(device)

        self.tt_inputs_device = tt_inputs_host.to(device, ttnn.DRAM_MEMORY_CONFIG)
        self.tt_pitch_device = tt_pitch_host.to(device, ttnn.DRAM_MEMORY_CONFIG)
        self.tt_pitchf_device = tt_pitchf_host.to(device, ttnn.DRAM_MEMORY_CONFIG)
        self.tt_speaker_id_device = tt_speaker_id_host.to(device, ttnn.DRAM_MEMORY_CONFIG)

        if not self.performance_runner:
            self.run(
                self.tt_inputs_device,
                self.tt_pitch_device,
                self.tt_pitchf_device,
                self.tt_speaker_id_device,
                index=None,
                big_npy=None,
            )
            return

        self.op_event = ttnn.record_event(device, 0)

        # First run configures convs JIT
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_inputs_device, 1)
        ttnn.copy_host_to_device_tensor(tt_pitch_host, self.tt_pitch_device, 1)
        ttnn.copy_host_to_device_tensor(tt_pitchf_host, self.tt_pitchf_device, 1)
        ttnn.copy_host_to_device_tensor(tt_speaker_id_host, self.tt_speaker_id_device, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        input_spec = self.tt_inputs_device.spec
        pitch_spec = self.tt_pitch_device.spec
        pitchf_spec = self.tt_pitchf_device.spec
        speaker_id_spec = self.tt_speaker_id_device.spec
        self.op_event = ttnn.record_event(device, 0)
        self.output_tensor = self.run(
            self.tt_inputs_device,
            self.tt_pitch_device,
            self.tt_pitchf_device,
            self.tt_speaker_id_device,
            index=None,
            big_npy=None,
        )
        self.output_tensor.deallocate(force=True)

        # Optimized run
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_inputs_device, 1)
        ttnn.copy_host_to_device_tensor(tt_pitch_host, self.tt_pitch_device, 1)
        ttnn.copy_host_to_device_tensor(tt_pitchf_host, self.tt_pitchf_device, 1)
        ttnn.copy_host_to_device_tensor(tt_speaker_id_host, self.tt_speaker_id_device, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.op_event = ttnn.record_event(device, 0)
        self.output_tensor = self.run(
            self.tt_inputs_device,
            self.tt_pitch_device,
            self.tt_pitchf_device,
            self.tt_speaker_id_device,
            index=None,
            big_npy=None,
        )

        # Capture
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_inputs_device, 1)
        ttnn.copy_host_to_device_tensor(tt_pitch_host, self.tt_pitch_device, 1)
        ttnn.copy_host_to_device_tensor(tt_pitchf_host, self.tt_pitchf_device, 1)
        ttnn.copy_host_to_device_tensor(tt_speaker_id_host, self.tt_speaker_id_device, 1)
        self.write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.op_event = ttnn.record_event(device, 0)
        self.output_tensor.deallocate(force=True)
        trace_input_addr = self.tt_inputs_device.buffer_address()
        trace_pitch_addr = self.tt_pitch_device.buffer_address()
        trace_pitchf_addr = self.tt_pitchf_device.buffer_address()
        trace_speaker_id_addr = self.tt_speaker_id_device.buffer_address()
        self.tid = ttnn.begin_trace_capture(device, cq_id=0)
        self.output_tensor = self.run(
            self.tt_inputs_device,
            self.tt_pitch_device,
            self.tt_pitchf_device,
            self.tt_speaker_id_device,
            index=None,
            big_npy=None,
        )
        self.tt_inputs_device.deallocate(force=True)
        self.tt_pitch_device.deallocate(force=True)
        self.tt_pitchf_device.deallocate(force=True)
        self.tt_speaker_id_device.deallocate(force=True)
        self.input_tensor = ttnn.allocate_tensor_on_device(input_spec, device)
        self.pitch_tensor = ttnn.allocate_tensor_on_device(pitch_spec, device)
        self.pitchf_tensor = ttnn.allocate_tensor_on_device(pitchf_spec, device)
        self.speaker_id_tensor = ttnn.allocate_tensor_on_device(speaker_id_spec, device)
        ttnn.end_trace_capture(device, self.tid, cq_id=0)
        assert trace_input_addr == self.input_tensor.buffer_address()
        assert trace_pitch_addr == self.pitch_tensor.buffer_address()
        assert trace_pitchf_addr == self.pitchf_tensor.buffer_address()
        assert trace_speaker_id_addr == self.speaker_id_tensor.buffer_address()
        self.tt_inputs_device = self.input_tensor
        self.tt_pitch_device = self.pitch_tensor
        self.tt_pitchf_device = self.pitchf_tensor
        self.tt_speaker_id_device = self.speaker_id_tensor

    def setup_l1_sharded_input(self, device, torch_input_tensor=None):
        # torch tensor
        torch_input_tensor = self.torch_input if torch_input_tensor is None else torch_input_tensor
        b, t, c = torch_input_tensor.shape
        assert c == 1, f"Expected input with 1 channel, got {c}"
        if device.get_num_devices() > 1:
            assert b % device.get_num_devices() == 0, "b isn't evenly divided by the available number of devices"
            b = b // device.get_num_devices()
            input_mem_config = ttnn.create_sharded_memory_config(
                [b, t, c],
                ttnn.CoreGrid(x=8, y=4),
                ttnn.ShardStrategy.HEIGHT,
            )
            input_tensor = torch.chunk(torch_input_tensor, device.get_num_devices(), dim=0)
            tt_inputs_host = ttnn.from_host_shards(
                [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for t in input_tensor],
                device.shape,
            )
        else:
            input_mem_config = ttnn.DRAM_MEMORY_CONFIG
            tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        return tt_inputs_host, input_mem_config

    def _from_torch_data_parallel(self, torch_tensor, dtype, device):
        if device.get_num_devices() <= 1:
            return ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        if torch_tensor.shape[0] % device.get_num_devices() != 0:
            raise ValueError(
                f"Tensor batch {torch_tensor.shape[0]} must be divisible by the number of devices {device.get_num_devices()}."
            )
        shards = torch.chunk(torch_tensor, device.get_num_devices(), dim=0)
        return ttnn.from_host_shards(
            [ttnn.from_torch(shard, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT) for shard in shards],
            device.shape,
        )

    def setup_dram_sharded_input(self, device, torch_input_tensor=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(device, torch_input_tensor)
        tt_pitch_host = self._from_torch_data_parallel(self.torch_pitch, ttnn.uint32, device)
        tt_pitchf_host = self._from_torch_data_parallel(self.torch_pitchf, ttnn.bfloat16, device)
        tt_speaker_id_host = self._from_torch_data_parallel(self.torch_speaker_id, ttnn.uint32, device)
        return tt_inputs_host, tt_pitch_host, tt_pitchf_host, tt_speaker_id_host, input_mem_config

    def execute_inference(self, tt_inputs_host, tt_pitch_host, tt_pitchf_host, tt_speaker_id_host):
        if not self.performance_runner:
            ttnn.copy_host_to_device_tensor(tt_inputs_host, self.tt_inputs_device)
            ttnn.copy_host_to_device_tensor(tt_pitch_host, self.tt_pitch_device)
            ttnn.copy_host_to_device_tensor(tt_pitchf_host, self.tt_pitchf_device)
            ttnn.copy_host_to_device_tensor(tt_speaker_id_host, self.tt_speaker_id_device)
            return self.run(
                self.tt_inputs_device,
                self.tt_pitch_device,
                self.tt_pitchf_device,
                self.tt_speaker_id_device,
                index=None,
                big_npy=None,
            )
        ttnn.wait_for_event(1, self.op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, self.input_tensor, 1)
        ttnn.copy_host_to_device_tensor(tt_pitch_host, self.pitch_tensor, 1)
        ttnn.copy_host_to_device_tensor(tt_pitchf_host, self.pitchf_tensor, 1)
        ttnn.copy_host_to_device_tensor(tt_speaker_id_host, self.speaker_id_tensor, 1)
        self.write_event = ttnn.record_event(self.device, 1)
        ttnn.wait_for_event(0, self.write_event)
        self.op_event = ttnn.record_event(self.device, 0)
        ttnn.execute_trace(self.device, self.tid, cq_id=0, blocking=False)
        return self.output_tensor

    def run(
        self,
        audio,
        pitch,
        pitchf,
        speaker_id,
        index,
        big_npy,
    ):
        assert (
            len(audio.shape) == 3 and audio.shape[2] == 1
        ), f"Expected audio shape [batch_size, 1, num_samples], got {audio.shape}"
        logits = self.hubert_model(
            source=audio,
            output_layer=9 if self.version == "v1" else 12,
        )
        feats = self.hubert_model.final_proj(logits) if self.version == "v1" else logits

        feats = ttnn.to_layout(feats, ttnn.ROW_MAJOR_LAYOUT)

        if self.protect < 0.5 and pitch is not None and pitchf is not None:
            protected_features = _interpolate_1d(feats, scale_factor=2)
        if index is not None and big_npy is not None and self.index_rate != 0:
            feats_torch = ttnn.to_torch(feats, mesh_composer=self.output_mesh_composer).to(torch.float32)
            index_features = feats_torch[0].detach().cpu().numpy()
            scores, indices = index.search(index_features, k=8)
            scores = torch.from_numpy(scores).to(torch.float32)
            indices = torch.from_numpy(indices).to(torch.long)
            weights = torch.square(1.0 / torch.clamp(scores, min=1e-6))
            weights /= weights.sum(dim=1, keepdim=True)
            index_features = torch.sum(big_npy[indices] * weights.unsqueeze(2), dim=1)
            index_features = index_features.unsqueeze(0).to(dtype=feats_torch.dtype)
            feats_torch = index_features * self.index_rate + (1 - self.index_rate) * feats_torch
            feats = ttnn.from_torch(
                feats_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=self.input_mesh_mapper,
            )

        feats = _interpolate_1d(feats, scale_factor=2)
        num_frames = feats.shape[1]

        if pitch is not None and pitchf is not None:
            pitch = pitch[:, :num_frames]
            pitchf = pitchf[:, :num_frames]
            if self.protect < 0.5:
                pitchff = self.protect + (1 - self.protect) * (pitchf >= 1)
                pitchff = ttnn.unsqueeze(pitchff, -1)
                feats = feats * pitchff + protected_features * ttnn.rsub(pitchff, 1)
                ttnn.deallocate(protected_features)

            output = self.synthesizer(feats, pitch, pitchf, speaker_id)
        else:
            output = self.synthesizer(feats, speaker_id)

        return output


class Pipeline:
    """Hybrid TT VC pipeline: Torch huBERT frontend + TT synthesizer backend."""

    def __init__(
        self,
        device: ttnn.MeshDevice,
        batch_size: int = 1,
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
        file_index: str | None = None,
        validation: bool = False,
        performance_runner: bool = False,
    ):
        config = config or Config()
        self.device = device
        self.if_f0 = if_f0
        self.speaker_id = speaker_id
        self.f0_up_key = f0_up_key
        self.f0_method = f0_method
        self.rms_mix_rate = rms_mix_rate
        self.file_index = file_index
        self._rmvpe_pitch_algorithm: RMVPEPitchAlgorithm | None = None
        self._crepe_predictor: CrepePredictor | None = None
        self._feature_index: Any | None = None
        self._feature_index_embeddings: torch.Tensor | None = None

        if self.device.get_num_devices() > 1:
            self.input_mesh_mapper = ttnn.ShardTensorToMesh(self.device, dim=0)
            self.output_mesh_composer = ttnn.ConcatMeshToTensor(self.device, dim=0)
        else:
            self.input_mesh_mapper = None
            self.output_mesh_composer = None

        self.runner = RVCRunner(
            device=self.device,
            batch_size=batch_size,
            if_f0=self.if_f0,
            version=version,
            num=num,
            config=config,
            index_rate=index_rate,
            protect=protect,
            validation=validation,
            performance_runner=performance_runner,
        )
        self.tgt_sr = self.runner.tgt_sr
        self._init_timing(self.tgt_sr, config)

    def _init_timing(self, tgt_sr: int, config: Config) -> None:
        x_pad, x_center, x_max = (
            config.x_pad,
            config.x_center,
            config.x_max,
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * x_pad
        self.t_pad_tgt = tgt_sr * x_pad
        self.t_center = self.sr * x_center
        self.t_max = self.sr * x_max

    def _get_rmvpe_pitch_algorithm(self) -> RMVPEPitchAlgorithm:
        if self._rmvpe_pitch_algorithm is None:
            self._rmvpe_pitch_algorithm = RMVPEPitchAlgorithm(
                device=self.device, sample_rate=self.sr, hop_size=self.window
            )
        return self._rmvpe_pitch_algorithm

    def _get_crepe_predictor(self) -> CrepePredictor:
        if self._crepe_predictor is None:
            self._crepe_predictor = CrepePredictor(device=self.device)
        return self._crepe_predictor

    def _load_feature_index(self):
        if not self.file_index:
            return None, None
        if self._feature_index is not None and self._feature_index_embeddings is not None:
            return self._feature_index, self._feature_index_embeddings
        if not os.path.exists(self.file_index):
            raise FileNotFoundError(f"Feature index file not found: {self.file_index}")

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "Feature index support requires faiss. Install faiss-cpu or faiss-gpu to use --file-index."
            ) from exc

        feature_index = faiss.read_index(self.file_index)
        feature_index_embeddings = torch.from_numpy(feature_index.reconstruct_n(0, feature_index.ntotal)).to(
            torch.float32
        )
        self._feature_index = feature_index
        self._feature_index_embeddings = feature_index_embeddings
        return self._feature_index, self._feature_index_embeddings

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
        elif self.f0_method is F0Method.HARVEST:
            audio_np = audio.detach().cpu().reshape(-1).numpy().astype(np.float64)
            frame_period = self.window / self.sr * 1000.0
            f0, _ = pw.harvest(
                audio_np,
                self.sr,
                f0_floor=f0_min,
                f0_ceil=f0_max,
                frame_period=frame_period,
            )
            f0 = torch.from_numpy(f0.astype(np.float32))
            f0 = f0.unsqueeze(0)
        elif self.f0_method is F0Method.CREPE:
            f0 = self._get_crepe_predictor().predict(
                audio.to(torch.float32),
                sample_rate=self.sr,
                hop_length=self.window,
                fmin=f0_min,
                fmax=f0_max,
            )
        elif self.f0_method is F0Method.RMVPE:
            f0 = self._get_rmvpe_pitch_algorithm().extract_pitch(audio)
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
        batch_size_per_device = self.runner.batch_size_per_device
        block_size = self.t_center // batch_size_per_device
        num_segments = (audio.shape[1] + block_size - 1) // block_size
        idx_list = []
        for b_idx in range(batch_size):
            b_idx_list = []
            for i in range(num_segments):
                s = i * block_size
                t = min(s + block_size, num_frames * self.window)
                b_idx_list.append((s // self.window, t // self.window))
            idx_list.append(b_idx_list)
        return idx_list

    def run(self, audio):
        assert audio.dim() == 2, audio.dim()
        batch_size = audio.shape[0]
        batch_size_per_device = self.runner.batch_size_per_device
        speaker_id = ttnn.from_torch(
            torch.full((batch_size,), self.speaker_id, dtype=torch.long),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self.input_mesh_mapper,
        )
        index, big_npy = self._load_feature_index()
        audio_secs = audio.shape[1] / self.sr
        audio_padded = F.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        block_size = self.t_center // batch_size_per_device
        padded_length = ((audio_padded.shape[1] + block_size - 1) // block_size) * block_size
        audio_padded = F.pad(audio_padded, (0, padded_length - audio_padded.shape[1]), mode="constant")
        num_frames = audio_padded.shape[1] // self.window
        idx_list = self._get_time_stamps(audio, num_frames)
        pitch, pitchf = None, None
        if self.if_f0:
            pitch_list = []
            pitchf_list = []
            for b_idx in range(batch_size):
                pitch_b, pitchf_b = self._get_f0(audio_padded[b_idx], num_frames)
                pitch_list.append(pitch_b)
                pitchf_list.append(pitchf_b)
            pitch = torch.cat(pitch_list, dim=0)
            pitchf = torch.cat(pitchf_list, dim=0)

        audio_output = []
        for idx_s, idx_e in idx_list[0]:
            if self.if_f0:
                chunk_end = min(idx_e, pitch.shape[1])
                pitch_slice = ttnn.from_torch(
                    pitch[:, idx_s:chunk_end],
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self.input_mesh_mapper,
                )
                pitchf_slice = ttnn.from_torch(
                    pitchf[:, idx_s:chunk_end],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self.input_mesh_mapper,
                )
            else:
                pitch_slice = None
                pitchf_slice = None

            audio_slice = ttnn.from_torch(
                audio_padded[:, idx_s * self.window : idx_e * self.window].view(batch_size, -1, 1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.input_mesh_mapper,
            )
            out = self.runner.execute_inference(audio_slice, pitch_slice, pitchf_slice, speaker_id)
            out = ttnn.to_torch(out, mesh_composer=self.output_mesh_composer).to(torch.float32)[:, :, 0]
            start_idx = self.t_pad_tgt
            end_idx = out.shape[1] - self.t_pad_tgt
            audio_output.append(out[:, start_idx:end_idx])

        audio_output = torch.cat(audio_output, dim=1)
        audio_output = audio_output[:, : int(audio_secs * self.tgt_sr)]

        # Need to use torch for multiple reasons
        # e.g. ttnn does not support int16 datayeype for tensors
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

    def _prepare_loaded_audio(self, audio: torch.Tensor) -> torch.Tensor:
        # preprocess
        audio_max = torch.abs(audio).max().item()
        if audio_max > 1:
            audio /= audio_max
        audio = signal.filtfilt(bh, ah, audio, axis=-1)
        audio = torch.from_numpy(audio.copy()).to(torch.float32)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        return audio

    def prepare_audio_input(self, num_secs: float | None = None) -> torch.Tensor:
        audio = load_audio(self.sr)
        return self._prepare_loaded_audio(audio)

    def prepare_audio_batch_input(self, num_secs: float | None = None) -> torch.Tensor:
        audio = load_audio_batch(self.sr)
        return self._prepare_loaded_audio(audio)
