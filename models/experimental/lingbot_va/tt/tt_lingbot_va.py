# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single-entry TTNN Lingbot-VA wrapper for pipeline-style perf (e.g. `tt_cnn` 2CQ, no trace).

This module mirrors the idea of a staged ``TtDINO``-style model: one ``forward``/``__call__`` used by
``create_pipeline_from_config`` while the real observation and prompt live in ``state`` / ``message``.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import torch
import ttnn


class TtLingbotVA:
    """Orchestrates Lingbot-VA TTNN inference (text → VAE encode → transformer) like ``run_inference``."""

    def __init__(self, models: dict, state: dict, message: dict) -> None:
        self.models = models
        self.state = state
        self.message = message
        self.single_run_inputs: dict | None = None

    @classmethod
    def prepare(
        cls,
        checkpoint_path: str | Path,
        message: dict,
        mesh_device: ttnn.MeshDevice,
        save_dir: str | Path | None = None,
        *,
        num_inference_steps: int | None = None,
        action_num_inference_steps: int | None = None,
        frame_chunk_size: int | None = None,
    ) -> TtLingbotVA:
        """
        Load checkpoints and run the same phases as ``demo.run_inference`` up to (and including) the
        transformer, without executing the final infer chunk.

        Args:
            checkpoint_path: Directory with ``text_encoder``, ``vae``, ``tokenizer``, transformer weights.
            message: Observation + prompt dict (``build_infer_message`` / ``run_inference`` shape).
            mesh_device: Open mesh (e.g. pytest ``mesh_device``); must match the pipeline ``device``.
            save_dir: ``config.save_root`` for caches.
            num_inference_steps / action_num_inference_steps / frame_chunk_size: same as ``run_inference``.
        """
        from models.experimental.lingbot_va.tests.demo import demo as lingbot_demo

        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_path}")

        lingbot_demo._set_seed()
        os.chdir(lingbot_demo._REPO_ROOT)
        config = deepcopy(lingbot_demo.VA_CONFIGS["robotwin"])
        config.wan22_pretrained_model_name_or_path = str(checkpoint_path)
        config.local_rank = 0
        config.rank = 0
        config.world_size = 1
        lingbot_demo.apply_robotwin_inference_overrides(
            config,
            num_inference_steps=num_inference_steps,
            action_num_inference_steps=action_num_inference_steps,
            frame_chunk_size=frame_chunk_size,
        )
        if save_dir is None:
            save_dir = lingbot_demo._SCRIPT_DIR
        config.save_root = str(save_dir)

        models = lingbot_demo._load_models_phase1(config, load_text_encoder=False, mesh_device=mesh_device)
        state: dict = {}
        prompt = message.get("prompt", "")
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        lingbot_demo._load_text_encoder_into_models(models, config)
        prompt_embeds, neg_embeds = lingbot_demo._encode_prompt(
            models,
            state,
            prompt,
            do_classifier_free_guidance=(config.guidance_scale > 1),
            max_sequence_length=512,
        )
        state["prompt_embeds"] = prompt_embeds
        state["negative_prompt_embeds"] = neg_embeds
        state["_prompt_embeds_prompt"] = prompt_list
        lingbot_demo._free_tt_model(models, "text_encoder")

        lingbot_demo._prepare_state_for_vae_encode(state, config)
        lingbot_demo._load_tt_vae_into_models(models, config)
        state["init_latent"] = lingbot_demo._encode_obs(models, state, message)
        lingbot_demo._free_tt_vae_from_models(models, config)

        lingbot_demo._load_transformer_into_models(models, config)

        instance = cls(models, state, message)
        instance.single_run_inputs = instance._prepare_single_run_inputs()
        return instance

    def _prepare_single_run_inputs(self) -> dict:
        """Precompute WanTransformer inputs for single-pass TTNN-only forward path."""
        from models.experimental.lingbot_va.tests.demo import demo as lingbot_demo

        transformer = self.models["transformer"]
        tt_transformer = getattr(transformer, "_tt_model", transformer)

        spatial = self.state["init_latent"]
        prompt = self.state["prompt_embeds"]
        if prompt.dim() == 2:
            prompt = prompt.unsqueeze(0)

        B, _, F, H, W = spatial.shape
        pF, pH, pW = tt_transformer.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        # Build regular (non-action) grid/timestep inputs and run existing preprocessing helpers once.
        grid_id = lingbot_demo.get_mesh_id(patch_F, patch_H, patch_W, 0, 1, 0).to(spatial.device)
        grid_id = grid_id.unsqueeze(0)
        timestep = torch.zeros((B,), dtype=torch.float32, device=spatial.device)

        rope_cos_1HND, rope_sin_1HND, trans_mat = tt_transformer.get_rope_features(grid_id)
        temb_11BD, timestep_proj_1BTD, prompt_1BLP = tt_transformer.prepare_conditioning(
            timestep, prompt, action_mode=False
        )
        spatial_1BNI, N = tt_transformer.preprocess_spatial_input(spatial)
        spatial_1BND = tt_transformer.patch_embedding(spatial_1BNI)

        metadata = {
            "rope_cos": rope_cos_1HND,
            "rope_sin": rope_sin_1HND,
            "trans_mat": trans_mat,
            "N": N,
            "F": F,
            "H": H,
            "W": W,
            "use_per_token": False,
            "action_mode": False,
        }
        return {
            "spatial_1BND": spatial_1BND,
            "prompt_1BLP": prompt_1BLP,
            "temb": temb_11BD,
            "block_temb": timestep_proj_1BTD,
            "metadata": metadata,
        }

    def forward_reset_and_infer(self) -> ttnn.Tensor:
        """Run one preprocessed WanTransformer forward with ``single_run=True``."""
        if self.single_run_inputs is None:
            self.single_run_inputs = self._prepare_single_run_inputs()

        transformer = self.models["transformer"]
        tt_transformer = getattr(transformer, "_tt_model", transformer)
        return tt_transformer.forward(
            spatial=self.single_run_inputs["spatial_1BND"],
            prompt=self.single_run_inputs["prompt_1BLP"],
            timestep=self.single_run_inputs["temb"],
            grid_id=self.single_run_inputs["metadata"],
            action_mode=False,
            update_cache=0,
            cache_name="pos",
            timestep_per_frame=self.single_run_inputs["block_temb"],
            single_run=True,
        )

    def __call__(self, l1_input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Pipeline entrypoint: runs one Lingbot chunk."""
        _ = l1_input_tensor
        return self.forward_reset_and_infer()
