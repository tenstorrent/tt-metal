# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va`` with a FastVideo PDD student, e.g. FastH3 OmniRef PDD-8.

The export holds only ``transformer_ref/`` and its sampling contract (``fastvideo_inference.json``); the text
encoder, processor and both VAEs come from the base ``weights_dir``, which must be the base revision the export was
distilled against or one whose shared partitions are byte-identical to it.

The widened output heads never reach the device: each forward's block is fused on host once per process and the
transformer runs step ``i`` with step ``i``'s pair of normal-width projections (see ``pdd_minimax_h3``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open

from ...layers.linear import Linear
from ...models.transformers.minimax_h3.transformer_minimax_h3 import MiniMaxH3Transformer3DModel
from ...models.transformers.minimax_h3.vsa_prepass import MiniMaxH3VSAConfig
from .pdd_minimax_h3 import PDD_HEADS, MiniMaxH3PDDContract
from .pipeline_minimax_h3 import MiniMaxH3Pipeline
from .scheduler import MiniMaxH3Scheduler
from .vsa_geometry import VSA_TILE_TOKENS

_GATE_SUFFIX = ".attn.to_gate_compress.weight"


def _export_digest(transformer_dir: Path) -> str:
    """Identifies the export in the weight-cache key, so it never loads another export's (or the base's) cache."""
    identity = transformer_dir.parent / "checkpoint-identity.json"
    if identity.is_file():
        return json.loads(identity.read_text())["aggregate_sha256"]
    return hashlib.sha256((transformer_dir / "diffusion_pytorch_model.safetensors.index.json").read_bytes()).hexdigest()


class MiniMaxH3PDDPipeline(MiniMaxH3Pipeline):
    """`task="ref2va"` only; `pdd_export_dir` is the export, `weights_dir` the base model."""

    def __init__(self, *, pdd_export_dir: str | Path, task: str = "ref2va", **kwargs) -> None:
        if task != "ref2va":
            raise ValueError(f"a PDD export is a ref2va student, got task {task!r}")
        self.pdd_export_dir = Path(pdd_export_dir)
        self.pdd = MiniMaxH3PDDContract.from_export(self.pdd_export_dir)
        for name, value in (("video_shift", self.pdd.video_shift), ("audio_shift", self.pdd.audio_shift)):
            if kwargs.get(name) not in (None, value):
                raise ValueError(f"{name} {kwargs[name]} conflicts with the export's {value}")
            kwargs[name] = value
        self._pdd_transformer_dir = self.pdd_export_dir / self.pdd.transformer_component
        self._pdd_digest = _export_digest(self._pdd_transformer_dir)
        self._pdd_head_states = self.pdd.fused_head_states(self._read_widened_heads())
        self._pdd_heads: list[tuple[Linear, Linear]] | None = None
        super().__init__(task=task, **kwargs)
        if self.vsa_config is not None:
            # The fused routing scores 64-token tiles over the target video alone; references stay dense.
            logger.warning(
                f"VSA runs {VSA_TILE_TOKENS}-token tiles over the target video only; the export was trained on "
                f"{self.pdd.vsa_tile_size}-token tiles with reference videos kept at {self.pdd.vsa_ref_keep_rate}"
            )
            if self.vsa_config.sparsity != self.pdd.vsa_sparsity:
                logger.warning(
                    f"VSA sparsity {self.vsa_config.sparsity} overrides the export's {self.pdd.vsa_sparsity}"
                )
        logger.info(
            f"PDD export {self.pdd_export_dir.name}: {self.pdd.num_forwards} forwards over a {self.pdd.pdd_steps}-head "
            f"grid, blocks {list(self.pdd.step_indices)}, shifts {self.pdd.video_shift:g}/{self.pdd.audio_shift:g}"
        )

    @staticmethod
    def export_vsa_config(export_dir: str | Path) -> MiniMaxH3VSAConfig:
        """The export's trained sparsity."""
        return MiniMaxH3VSAConfig(sparsity=MiniMaxH3PDDContract.from_export(export_dir).vsa_sparsity)

    def __call__(self, prompt: str, *, num_inference_steps: int | None = None, **kwargs):
        """`num_inference_steps` counts sigma grid points and defaults to the export's forwards + 1."""
        if num_inference_steps is None:
            num_inference_steps = self.pdd.num_forwards + 1
        return super().__call__(prompt, num_inference_steps=num_inference_steps, **kwargs)

    # ------------------------------------------------------------------ weights

    def _read_widened_heads(self) -> dict[str, torch.Tensor]:
        index = json.loads((self._pdd_transformer_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
        tensors = {}
        for head in PDD_HEADS.values():
            for name in ("weight", "bias"):
                key = f"{head}.{name}"
                with safe_open(str(self._pdd_transformer_dir / index["weight_map"][key]), framework="pt") as handle:
                    tensors[key] = handle.get_tensor(key)
        return tensors

    def _read_config(self, subfolder: str) -> dict:
        if subfolder != self.transformer_subfolder:
            return super()._read_config(subfolder)
        raw = json.loads((self._pdd_transformer_dir / "config.json").read_text())
        return {k: v for k, v in raw.items() if not k.startswith("_") and k != "pdd_steps"}

    def _read_safetensors(self, subfolder: str) -> dict[str, torch.Tensor]:
        if subfolder != self.transformer_subfolder:
            return super()._read_safetensors(subfolder)
        from safetensors.torch import load_file

        index = json.loads((self._pdd_transformer_dir / "diffusion_pytorch_model.safetensors.index.json").read_text())
        state: dict[str, torch.Tensor] = {}
        for shard in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(self._pdd_transformer_dir / shard)))
        # The cached module carries the first block's heads; every step swaps in its own pair anyway.
        for head, params in self._pdd_head_states[0].items():
            for name, tensor in params.items():
                state[f"{head}.{name}"] = tensor
        if self.vsa_config is None:
            gates = [key for key in state if key.endswith(_GATE_SUFFIX)]
            for key in gates:
                del state[key]
            logger.warning(f"PDD export without vsa_config: {len(gates)} trained VSA gates dropped, attention is dense")
        return state

    def _dit_weight_mode(self) -> str:
        return f"{super()._dit_weight_mode()}_pdd_{self._pdd_digest[:12]}"

    def _step_heads(self) -> list[tuple[Linear, Linear]]:
        if self._pdd_heads is None:
            heads = []
            for states in self._pdd_head_states:
                pair = []
                for head in PDD_HEADS.values():
                    weight = states[head]["weight"]
                    linear = Linear(weight.shape[1], weight.shape[0], bias=True, mesh_device=self.mesh_device)
                    linear.load_torch_state_dict({name: tensor.clone() for name, tensor in states[head].items()})
                    pair.append(linear)
                heads.append(tuple(pair))
            self._pdd_heads = heads
        return self._pdd_heads

    # ------------------------------------------------------------------ sampling

    def _build_schedulers(self, num_inference_steps: int) -> tuple[MiniMaxH3Scheduler, MiniMaxH3Scheduler]:
        if self._warming:
            return super()._build_schedulers(num_inference_steps)
        if num_inference_steps != self.pdd.num_forwards + 1:
            raise ValueError(
                f"the export samples {self.pdd.num_forwards} forwards (num_inference_steps "
                f"{self.pdd.num_forwards + 1}), got num_inference_steps {num_inference_steps}"
            )
        scheduler = MiniMaxH3Scheduler(shift=self.video_shift)
        audio_scheduler = MiniMaxH3Scheduler(shift=self.audio_shift)
        scheduler.set_timesteps(sigmas=self.pdd.node_sigmas("video"))
        audio_scheduler.set_timesteps(sigmas=self.pdd.node_sigmas("audio"))
        return scheduler, audio_scheduler

    def _denoise(
        self, transformer, layout, prompt_embeds, video_rows, audio_rows, scheduler, audio_scheduler, **kwargs
    ):
        # The students are trained from the first node's noise level, not unit noise; conditioning rows stay as drawn.
        video_rows[layout.num_condition_video_rows :] *= float(scheduler.sigmas[0])
        audio_rows[layout.num_condition_audio_rows :] *= float(audio_scheduler.sigmas[0])
        try:
            return super()._denoise(
                transformer, layout, prompt_embeds, video_rows, audio_rows, scheduler, audio_scheduler, **kwargs
            )
        finally:
            transformer.set_output_projections(None)

    def _before_step(self, transformer: MiniMaxH3Transformer3DModel, step: int) -> None:
        transformer.set_output_projections(self._step_heads()[step])


__all__ = ["MiniMaxH3PDDPipeline"]
