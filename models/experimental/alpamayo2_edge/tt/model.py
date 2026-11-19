# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Alpamayo 2 Edge driving inference on device: vision tower -> LM prefill and CoT decode -> expert flow -> trajectory.

Input preparation, token sampling, the Fourier features and Euler update of the flow, and `action_to_traj` run on
the host; `reference/host_ops.md` lists every host step and why it is there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

import ttnn
from models.experimental.alpamayo2_edge.reference.action import action_to_traj
from models.experimental.alpamayo2_edge.reference.config import (
    Alpamayo2EdgeConfig,
    load_config,
    resolve_model_dir,
    validate_supported,
)
from models.experimental.alpamayo2_edge.reference.inputs import ModelInputs
from models.experimental.alpamayo2_edge.reference.model import Prediction
from models.experimental.alpamayo2_edge.reference.vision import vision_cu_seqlens
from models.experimental.alpamayo2_edge.reference.weights import load_weights
from models.experimental.alpamayo2_edge.tt.action import TtActionInProj, TtActionOutProj, sample_actions_tt
from models.experimental.alpamayo2_edge.tt.expert import TtExpert, make_expert_frame
from models.experimental.alpamayo2_edge.tt.generate import CotResult, generate_cot
from models.experimental.alpamayo2_edge.tt.text import AlpamayoCache, TtLanguageModel
from models.experimental.alpamayo2_edge.tt.vision import PosEmbedCache, TtVisionTower
from models.tt_dit.encoders.transformer import LINEAR_DTYPE
from models.tt_dit.layers.linear import ColParallelLinear
from models.tt_dit.utils import tensor

_POS_EMBED_KEY = "embeddings.position_embedding.weight"


@dataclass
class TtRun:
    prediction: Prediction
    cot: CotResult  # logits kept only with `keep_intermediates`
    vision_embeds: torch.Tensor | None  # [V, hidden] read back from the device with `keep_intermediates`
    timings: dict[str, float]  # wall seconds per stage: vision, cot (prefill + decode), flow, total


class Alpamayo2EdgeTT:
    def __init__(
        self,
        cfg: Alpamayo2EdgeConfig,
        *,
        vision: TtVisionTower,
        pos_embeds: PosEmbedCache,
        lm: TtLanguageModel,
        expert: TtExpert,
        action_in_proj: TtActionInProj,
        action_out_proj: TtActionOutProj,
    ) -> None:
        validate_supported(cfg)
        self.cfg = cfg
        self.mesh_device = lm.device
        self.vision = vision
        self.pos_embeds = pos_embeds
        self.lm = lm
        self.expert = expert
        self.action_in_proj = action_in_proj
        self.action_out_proj = action_out_proj

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        cfg: Alpamayo2EdgeConfig | None = None,
        *,
        lm_head_dtype: ttnn.DataType = LINEAR_DTYPE,
    ) -> Alpamayo2EdgeTT:
        """Builds every block at the config's depths and loads the checkpoint into it.

        Args:
            cfg: Defaults to the `nvidia/Alpamayo2-Edge` snapshot, with the `ALPAMAYO_NUM_LAYERS` override.
            lm_head_dtype: Weight dtype of the LM's vocabulary projection; tt_dit's linear default is bf8.
        """
        if cfg is None:
            cfg = load_config(resolve_model_dir())
        md = cfg.model_dir

        visual = load_weights(md, "vlm.model.visual.")
        vision = TtVisionTower(cfg.vision, cfg.projector, mesh_device=mesh_device)
        vision.load_torch_state_dict(TtVisionTower.convert_state(visual, load_weights(md, "vlm.model.projector.")))
        pos_embeds = PosEmbedCache(visual[_POS_EMBED_KEY], merge=cfg.vision.merge, mesh_device=mesh_device)
        del visual

        lm = TtLanguageModel(TtLanguageModel.config_from(cfg.text), device=mesh_device)
        if lm_head_dtype != LINEAR_DTYPE:
            # Parameters are allocated at load, so replacing the head before loading costs nothing.
            lm.final_linear = ColParallelLinear(
                cfg.text.hidden,
                cfg.text.vocab,
                bias=False,
                mesh_device=mesh_device,
                mesh_axis=None,
                dtype=lm_head_dtype,
            )
        lm.load_torch_state_dict(
            TtLanguageModel.convert_state(
                load_weights(md, "vlm.model.language_model."), load_weights(md, "vlm.lm_head.")
            )
        )

        expert = TtExpert(cfg.expert, mesh_device=mesh_device, lm_num_layers=cfg.text.layers)
        expert.load_torch_state_dict(TtExpert.convert_state(load_weights(md, "expert.expert.")))
        in_proj = TtActionInProj(cfg.action, cfg.expert.hidden, mesh_device=mesh_device)
        in_proj.load_torch_state_dict(load_weights(md, "expert.action_in_proj."))
        out_proj = TtActionOutProj(cfg.expert.hidden, cfg.action.action_dim, mesh_device=mesh_device)
        out_proj.load_torch_state_dict(load_weights(md, "expert.action_out_proj."))

        return cls(
            cfg,
            vision=vision,
            pos_embeds=pos_embeds,
            lm=lm,
            expert=expert,
            action_in_proj=in_proj,
            action_out_proj=out_proj,
        )

    def encode_images(self, inp: ModelInputs) -> ttnn.Tensor:
        """`[V, hidden]` device embeddings of the prompt's image tokens, in prompt order."""
        patches = tensor.bf16_tensor(inp.patches, device=self.mesh_device)
        return self.vision.forward(
            patches, self.pos_embeds.get(inp.image_grid_thw), vision_cu_seqlens(inp.image_grid_thw)
        )

    def run(
        self,
        inp: ModelInputs,
        *,
        num_samples: int = 1,
        forced_cot: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        seed: int = 0,
        max_new_tokens: int = 256,
        keep_intermediates: bool = False,
    ) -> TtRun:
        """`infer`, plus the CoT result and stage timings.

        With `keep_intermediates`, the vision embeddings and every step's logits are also read back to the host.
        """
        acfg = self.cfg.action
        if forced_cot is not None:
            if forced_cot.ndim == 1:
                forced_cot = forced_cot.expand(num_samples, -1)
            if forced_cot.shape[0] != num_samples:
                msg = f"forced_cot has {forced_cot.shape[0]} rows for {num_samples} samples"
                raise ValueError(msg)
        noise_shape = (num_samples, acfg.n_waypoints, acfg.action_dim)
        if noise is None:
            noise = torch.randn(*noise_shape, generator=torch.Generator().manual_seed(seed))
        elif tuple(noise.shape) != noise_shape:
            msg = f"noise must be {list(noise_shape)}, got {list(noise.shape)}"
            raise ValueError(msg)

        timings = {}
        start = time.perf_counter()
        vision_embeds = self.encode_images(inp)
        kept_vision = tensor.to_torch(vision_embeds) if keep_intermediates else None
        ttnn.synchronize_device(self.mesh_device)
        timings["vision"] = time.perf_counter() - start

        mark = time.perf_counter()
        cache = AlpamayoCache(
            device=self.mesh_device, size=AlpamayoCache.size_for(inp.input_ids.shape[1]), batch_size=1
        )
        cot = generate_cot(
            self.lm,
            inp,
            vision_embeds,
            tokens=self.cfg.tokens,
            cache=cache,
            num_samples=num_samples,
            forced=forced_cot,
            seed=seed,
            max_new_tokens=max_new_tokens,
            return_logits=keep_intermediates,
        )
        if cot.cache is not cache:
            # The samples continue in a repeated copy; the single-sequence prefill cache is no longer read.
            for layer in range(self.cfg.text.layers):
                for kv in cache.layer_kv(layer):
                    ttnn.deallocate(kv)
        del cache
        timings["cot"] = time.perf_counter() - mark

        mark = time.perf_counter()
        frame = make_expert_frame(
            cot, inp, ecfg=self.cfg.expert, n_waypoints=acfg.n_waypoints, mesh_device=self.mesh_device
        )
        actions = sample_actions_tt(
            self.expert,
            self.action_in_proj,
            self.action_out_proj,
            cache=cot.cache,
            frame=frame,
            noise=noise,
            acfg=acfg,
        )
        timings["flow"] = time.perf_counter() - mark

        xyz, rot = action_to_traj(
            actions,
            inp.ego_history_xyz.repeat_interleave(num_samples, dim=0),
            inp.ego_history_rot.repeat_interleave(num_samples, dim=0),
            acfg,
        )
        timings["total"] = time.perf_counter() - start
        prediction = Prediction(xyz=xyz, rot=rot, cot_tokens=cot.tokens, actions=actions)
        return TtRun(prediction=prediction, cot=cot, vision_embeds=kept_vision, timings=timings)

    def infer(
        self,
        inp: ModelInputs,
        *,
        num_samples: int = 1,
        forced_cot: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        seed: int = 0,
        max_new_tokens: int = 256,
    ) -> Prediction:
        """`num_samples` trajectories of one prompt, each with its own sampled CoT.

        Args:
            inp: A single unpadded prompt from `prepare_inputs`.
            forced_cot: `[T]` ids for every sample, or `[num_samples, T]`, taken in place of sampling; each row must
                contain the anchor.
            noise: `[num_samples, n_waypoints, action_dim]` flow start; drawn from `seed` when omitted.
            seed: Seeds the CoT sampling and the default noise.

        Raises:
            ValueError: A sample has no `<traj_future_start>` anchor after `max_new_tokens` tokens.
        """
        return self.run(
            inp,
            num_samples=num_samples,
            forced_cot=forced_cot,
            noise=noise,
            seed=seed,
            max_new_tokens=max_new_tokens,
        ).prediction
