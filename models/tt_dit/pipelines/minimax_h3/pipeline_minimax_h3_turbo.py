# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 with a lightx2v Turbo distillation adapter merged into the transformer weights.

`num_inference_steps` counts sigma grid points, so a 4-forward adapter runs at 5 and an 8-forward one at 9.
The 768p files were distilled at video shift 6, not the checkpoint's 12. A HyperFlow file fixes its own
grid (see `hyperflow_minimax_h3`); its float32 time-embedder deltas are fused on host, not via bfloat16 LoRA.

The adapter is merged on host in float32 and rounded to bfloat16 stochastically: its deltas are mostly
below half a bfloat16 ulp of the weights, so a round-to-nearest merge would drop them.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn

from ...experimental.lora.h3_adapter_loader import h3_host_deltas, iter_h3_host_deltas
from ...models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3TimestepEmbedding, MiniMaxH3TwoTime
from .hyperflow_minimax_h3 import MiniMaxH3HyperFlow
from .pipeline_minimax_h3 import AUDIO_SHIFT, VIDEO_SHIFT, MiniMaxH3Pipeline
from .scheduler import MiniMaxH3Scheduler
from .weights_minimax_h3 import LORA_PATH_ENV, resolve_adapter_settings

#: Forward counts the published adapters were distilled for; `num_inference_steps` is one more.
TURBO_NUM_FORWARDS = (4, 8)

#: A HyperFlow file's float32 targets: the base time embedder and the endpoint copy it adds.
HYPERFLOW_HOST_PREFIXES = ("time_embedder.", "endpoint_time_embedder.")
_TIME_EMBEDDER_KEYS = ("linear_1.weight", "linear_1.bias", "linear_2.weight", "linear_2.bias")


def _round_bf16_stochastic(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    bits = tensor.float().contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, generator=generator)
    return ((bits + noise) & ~0xFFFF).view(torch.float32).to(torch.bfloat16)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


class MiniMaxH3TurboPipeline(MiniMaxH3Pipeline):
    """`MiniMaxH3Pipeline` with a Turbo adapter; `lora_strength` multiplies the file's `alpha / rank`."""

    def __init__(
        self,
        *,
        lora_path: str | os.PathLike,
        lora_strength: float = 1.0,
        **kwargs,
    ) -> None:
        self.lora_path: Path = resolve_adapter_settings(
            lora_path=lora_path,
            lora_strength=lora_strength,
            default_video_shift=VIDEO_SHIFT,
            default_audio_shift=AUDIO_SHIFT,
        ).lora_path
        self.lora_strength = float(lora_strength)
        self._lora_digest = _file_digest(self.lora_path)
        # Stays None when the transformer loads from the weight cache, which skips `_read_safetensors`.
        self.merged_targets: int | None = None
        self._time_embedder_states: dict[str, dict[str, torch.Tensor]] | None = None
        with safe_open(str(self.lora_path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        video_shift, audio_shift = kwargs.get("video_shift"), kwargs.get("audio_shift")
        self.hyperflow = MiniMaxH3HyperFlow.from_adapter_metadata(
            metadata,
            video_shift=VIDEO_SHIFT if video_shift is None else float(video_shift),
            audio_shift=AUDIO_SHIFT if audio_shift is None else float(audio_shift),
        )
        super().__init__(**kwargs)

    @classmethod
    def create_pipeline(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        weights_dir: str | os.PathLike | None = None,
        lora_path: str | os.PathLike | None = None,
        lora_strength: float | None = None,
        video_shift: float | None = None,
        audio_shift: float | None = None,
        **kwargs,
    ) -> MiniMaxH3TurboPipeline:
        """Unset adapter arguments fall back to the `MINIMAX_H3_*` environment (see `resolve_adapter_settings`)."""
        settings = resolve_adapter_settings(
            lora_path=lora_path,
            lora_strength=lora_strength,
            video_shift=video_shift,
            audio_shift=audio_shift,
            default_video_shift=VIDEO_SHIFT,
            default_audio_shift=AUDIO_SHIFT,
        )
        if settings.lora_path is None:
            raise ValueError(f"the Turbo pipeline needs an adapter: pass lora_path= or set {LORA_PATH_ENV}")
        return super().create_pipeline(
            mesh_device=mesh_device,
            weights_dir=weights_dir,
            video_shift=settings.video_shift,
            audio_shift=settings.audio_shift,
            lora_path=settings.lora_path,
            lora_strength=settings.lora_strength,
            **kwargs,
        )

    def __call__(self, prompt: str, *, num_inference_steps: int | None = None, **kwargs):
        """`num_inference_steps=None` takes a HyperFlow adapter's grid, else the base default."""
        if num_inference_steps is None and self.hyperflow is not None:
            num_inference_steps = self.hyperflow.num_grid_points
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps
        return super().__call__(prompt, **kwargs)

    def _dit_weight_mode(self) -> str:
        return f"{super()._dit_weight_mode()}_{self.lora_path.stem}_{self._lora_digest[:12]}_x{self.lora_strength!r}"

    def _read_safetensors(self, subfolder: str) -> dict[str, torch.Tensor]:
        state = super()._read_safetensors(subfolder)
        if subfolder != self.transformer_subfolder:
            return state
        host_prefixes = HYPERFLOW_HOST_PREFIXES if self.hyperflow is not None else ()
        generator = torch.Generator().manual_seed(0)
        merged = 0
        for key, delta in iter_h3_host_deltas(str(self.lora_path), ("",), scale=self.lora_strength):
            if key.startswith(host_prefixes):
                continue
            weight = state.get(key)
            if weight is None:
                raise KeyError(f"turbo adapter target {key} is not in the {self.transformer_subfolder} checkpoint")
            if weight.shape != delta.shape:
                raise ValueError(f"turbo adapter target {key}: delta {tuple(delta.shape)} != {tuple(weight.shape)}")
            if weight.dtype == torch.bfloat16:
                state[key] = _round_bf16_stochastic(weight.float() + delta, generator)
            else:
                state[key] = weight + delta.to(weight.dtype)
            merged += 1
        if merged == 0:
            raise RuntimeError(f"turbo adapter {self.lora_path} has no LoRA targets to merge")
        self.merged_targets = merged
        logger.info(f"turbo adapter {self.lora_path.name}: merged {merged} targets, strength {self.lora_strength:g}")
        return state

    def _prepare_transformer(self):
        transformer = super()._prepare_transformer()
        contract = self.hyperflow
        if contract is None:
            return transformer
        if self._time_embedder_states is None:
            contract.assert_supports_task(self.task)
            contract.assert_supports_subfolder(self.transformer_subfolder)
            self._time_embedder_states = self._fused_time_embedder_states()
            endpoint = MiniMaxH3TimestepEmbedding(
                in_channels=transformer.time_embedder.linear_1.in_features,
                hidden_dim=transformer.time_embedder.linear_1.out_features,
                out_dim=transformer.time_embedder.linear_2.out_features,
                mesh_device=self.mesh_device,
            )
            endpoint.load_torch_state_dict(self._time_embedder_states["endpoint_time_embedder"])
            transformer.two_time = MiniMaxH3TwoTime(embedder=endpoint, gate=contract.gate)
            transformer.time_embedder.load_torch_state_dict(self._time_embedder_states["time_embedder"])
            logger.info(
                f"hyperflow: {contract.num_forwards} forwards, gate {contract.gate:g}, "
                "both time embedders fused in float32"
            )
        elif not self.coresident:
            transformer.time_embedder.load_torch_state_dict(self._time_embedder_states["time_embedder"])
        return transformer

    def _fused_time_embedder_states(self) -> dict[str, dict[str, torch.Tensor]]:
        """Base `time_embedder` weights plus each embedder's own float32 delta, as two state dicts."""
        base = self._read_checkpoint_tensors([f"time_embedder.{key}" for key in _TIME_EMBEDDER_KEYS])
        deltas = h3_host_deltas(str(self.lora_path), HYPERFLOW_HOST_PREFIXES, scale=self.lora_strength)
        expected = {
            f"{prefix}{key}"
            for prefix in HYPERFLOW_HOST_PREFIXES
            for key in _TIME_EMBEDDER_KEYS
            if key.endswith("weight")
        }
        if set(deltas) != expected:
            raise RuntimeError(f"hyperflow adapter time-embedder targets {sorted(deltas)}, expected {sorted(expected)}")
        states = {}
        for prefix in HYPERFLOW_HOST_PREFIXES:
            state = {key: base[f"time_embedder.{key}"].float().clone() for key in _TIME_EMBEDDER_KEYS}
            for key in _TIME_EMBEDDER_KEYS:
                if key.endswith("weight"):
                    state[key] += deltas[f"{prefix}{key}"]
            states[prefix.rstrip(".")] = state
        return states

    def _read_checkpoint_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        """A few named tensors from the transformer partition, without reading all 62 GB."""
        directory = self.weights_dir / self.transformer_subfolder
        index = directory / "diffusion_pytorch_model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            shards = {key: directory / weight_map[key] for key in keys}
        else:
            shards = {key: directory / "diffusion_pytorch_model.safetensors" for key in keys}
        tensors = {}
        for key, shard in shards.items():
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                tensors[key] = handle.get_tensor(key)
        return tensors

    def _build_schedulers(self, num_inference_steps: int) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
        contract = self.hyperflow
        if contract is None or self._warming:
            return super()._build_schedulers(num_inference_steps)
        contract.assert_forwards(num_inference_steps)
        scheduler = MiniMaxH3Scheduler(shift=self.video_shift)
        audio_scheduler = MiniMaxH3Scheduler(shift=self.audio_shift)
        scheduler.set_timesteps(sigmas=contract.modality_sigmas(self.video_shift))
        audio_scheduler.set_timesteps(sigmas=contract.modality_sigmas(self.audio_shift))
        return scheduler, audio_scheduler

    def _step_endpoints(
        self, scheduler: MiniMaxH3Scheduler, audio_scheduler: MiniMaxH3Scheduler
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.hyperflow is None:
            return None
        return self.hyperflow.endpoints(scheduler.sigmas), self.hyperflow.endpoints(audio_scheduler.sigmas)


__all__ = ["HYPERFLOW_HOST_PREFIXES", "TURBO_NUM_FORWARDS", "MiniMaxH3TurboPipeline"]
