# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 with a lightx2v Turbo distillation adapter fused into the transformer.

The adapter is a plain low-rank delta over the attention and feed-forward weights, so the generation
path is the base pipeline's: same packing, schedulers, VAE and audio decode. What changes is the
transformer's weights, the step count the caller asks for, and for the 768p files the video shift.
Those three travel together here rather than as switches on the base class.

A Turbo file publishes no sampling contract in its header. The caller supplies the step count, and
`num_inference_steps` counts sigma grid points, so a 4-forward adapter runs at 5 and an 8-forward one
at 9. The model card's shift for the file goes in `video_shift`; the 768p variants were distilled at
6 against the checkpoint's 12, and a wrong shift is a valid schedule over the wrong grid.

A HyperFlow file does publish a contract (see `hyperflow_minimax_h3`): its own sigma grid, and
interval `(t, r)` conditioning through a second, adapted time embedder. The step count then comes
from the file and any other is refused, except during warmup, whose output is discarded and which
keeps its short schedule. Both time embedders are float32 on device and the file ships their adapters
in float32, so they are fused on host rather than through the bfloat16 `register_lora` path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

import ttnn

from ...experimental.lora.h3_adapter_loader import H3AdapterHandle, h3_host_deltas, load_h3_adapter_into
from ...experimental.lora.promote import lora_modules
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


class MiniMaxH3TurboPipeline(MiniMaxH3Pipeline):
    """`MiniMaxH3Pipeline` whose transformer carries a lightx2v Turbo adapter.

    `lora_strength` multiplies the adapter's own `alpha / rank`, which the loader reads from the file;
    1.0 runs the adapter as trained.
    """

    def __init__(
        self,
        *,
        lora_path: str | os.PathLike,
        lora_strength: float = 1.0,
        **kwargs,
    ) -> None:
        # Bound onto the built transformer after the base weights load, never fused into the
        # checkpoint, so the weight cache stays adapter-independent and one cached copy serves every
        # adapter and strength.
        self.lora_path: Path = resolve_adapter_settings(
            lora_path=lora_path,
            lora_strength=lora_strength,
            default_video_shift=VIDEO_SHIFT,
            default_audio_shift=AUDIO_SHIFT,
        ).lora_path
        self.lora_strength = float(lora_strength)
        self._adapter: H3AdapterHandle | None = None
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
        """`lora_path`, `lora_strength`, `video_shift` and `audio_shift` fall back to `MINIMAX_H3_LORA_PATH`,
        `MINIMAX_H3_LORA_STRENGTH`, `MINIMAX_H3_VIDEO_SHIFT` and `MINIMAX_H3_AUDIO_SHIFT`.

        Everything else is `MiniMaxH3Pipeline.create_pipeline`'s.
        """
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

    @property
    def adapter(self) -> H3AdapterHandle | None:
        """What is bound on the transformer, or None before its first residency."""
        return self._adapter

    def _prepare_transformer(self):
        transformer = super()._prepare_transformer()
        contract = self.hyperflow
        if self._adapter is None:
            if contract is not None:
                contract.assert_supports_task(self.task)
                contract.assert_supports_subfolder(self.transformer_subfolder)
            self._adapter = load_h3_adapter_into(
                transformer,
                str(self.lora_path),
                scale=self.lora_strength,
                name=self.lora_path.name,
                host_prefixes=HYPERFLOW_HOST_PREFIXES if contract is not None else (),
            )
            logger.info(
                f"turbo adapter {self._adapter.name}: {len(self._adapter)} targets, strength {self.lora_strength:g}"
            )
            if contract is not None:
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
        else:
            # `coresident=False` evicts the transformer between stages and `cache.load_model` brings
            # back the cached *base* weights, so the fused delta has to be merged again. A no-op while
            # the weights stayed resident.
            for module in lora_modules(transformer):
                module.reapply_after_load()
            if self._time_embedder_states is not None and not self.coresident:
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
        """A few named tensors from the transformer partition, without reading the other 62 GB."""
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
