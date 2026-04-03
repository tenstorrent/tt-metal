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
from reference.utils import VA_CONFIGS, apply_robotwin_inference_overrides

from models.experimental.lingbot_va.tests.demo import demo as lingbot_demo
from models.experimental.lingbot_va.tt.utils import get_mesh_id_ttnn


class TtLingbotVA:
    """Orchestrates Lingbot-VA TTNN inference (text → VAE encode → transformer) like ``run_inference``."""

    def __init__(self, models: dict, state: dict, message: dict, init_obs: dict) -> None:
        self.models = models
        self.state = state
        self.message = message
        self._init_obs = init_obs
        self.single_run_inputs: dict | None = None

    @classmethod
    def prepare(
        cls,
        checkpoint_path: str | Path,
        message: dict,
        mesh_device: ttnn.MeshDevice | None = None,
        save_dir: str | Path | None = None,
        *,
        num_inference_steps: int | None = None,
        action_num_inference_steps: int | None = None,
        frame_chunk_size: int | None = None,
    ) -> TtLingbotVA:
        """
        Load checkpoints and run the same phases as ``demo.run_inference`` up to (and including) the
        transformer, without executing the final infer chunk.

        Uses ``_encode_prompt_ttnn`` and ``_encode_obs_ttnn`` exactly as in ``run_inference``.

        Args:
            checkpoint_path: Directory with ``text_encoder``, ``vae``, ``tokenizer``, transformer weights.
            message: Observation + prompt dict (``build_infer_message`` / ``run_inference`` shape).
            mesh_device: Optional opened mesh (e.g. pytest ``mesh_device``); ``None`` opens via demo.
            save_dir: ``config.save_root`` for caches.
            num_inference_steps / action_num_inference_steps / frame_chunk_size: same as ``run_inference``.
        """

        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_path}")

        lingbot_demo._set_seed()
        os.chdir(lingbot_demo._REPO_ROOT)
        config = deepcopy(VA_CONFIGS["robotwin"])
        config.wan22_pretrained_model_name_or_path = str(checkpoint_path)
        config.local_rank = 0
        config.rank = 0
        config.world_size = 1
        config.save_root = str(save_dir)
        config.num_chunks_to_infer = 1
        apply_robotwin_inference_overrides(
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
        tokenizer = models["tokenizer"]
        text_encoder = models["text_encoder"]
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        mesh_dev = text_encoder.mesh_device
        tt_input = ttnn.from_torch(text_input_ids, dtype=ttnn.uint32, device=mesh_dev)
        tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=mesh_dev)
        prompt_embeds, neg_embeds = lingbot_demo._encode_prompt_ttnn(
            models,
            tt_input,
            tt_mask,
            do_classifier_free_guidance=(config.guidance_scale > 1),
            max_sequence_length=512,
        )
        state["prompt_embeds"] = prompt_embeds
        state["negative_prompt_embeds"] = neg_embeds
        state["_prompt_embeds_prompt"] = prompt_list
        lingbot_demo._free_tt_model(models, "text_encoder")

        lingbot_demo._prepare_state_for_vae_encode(state, config)
        init_obs = lingbot_demo._normalize_infer_obs_for_encode(message["obs"])
        lingbot_demo._load_tt_vae_into_models(models)
        state["init_latent"] = lingbot_demo._encode_obs_ttnn(models, state, init_obs)
        lingbot_demo._free_tt_vae_from_models(models)

        lingbot_demo._load_transformer_into_models(models, config)

        lingbot_demo._reset_state(models, state, prompt)

        instance = cls(models, state, message, init_obs)
        instance.single_run_inputs = instance._prepare_single_run_inputs()
        return instance

    def _prepare_single_run_inputs(self) -> dict:
        """Precompute WanTransformer inputs for single-pass TTNN-only forward path."""
        transformer = self.models["transformer"]
        tt_transformer = getattr(transformer, "_tt_model", transformer)
        mesh_device = self.models["mesh_device"]

        spatial = self.state["init_latent"]
        prompt = self.state["prompt_embeds"]
        if not isinstance(prompt, ttnn.Tensor):
            raise TypeError(
                "state['prompt_embeds'] must be a ttnn.Tensor (same as run_inference / _encode_prompt_ttnn)."
            )
        if len(prompt.shape) == 2:
            prompt = ttnn.unsqueeze(prompt, 0)

        B, _c, F, H, W = (int(spatial.shape[i]) for i in range(5))
        pF, pH, pW = tt_transformer.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW

        # Match ``get_mesh_id_ttnn`` + batch dim, and mesh 1-D timestep (not torch on ``.device``).
        grid_id = get_mesh_id_ttnn(mesh_device, patch_F, patch_H, patch_W, 0, 1, 0, action=False)
        grid_id = ttnn.unsqueeze(grid_id, 0)
        # ``WanTransformer.prepare_timestep_conditioning`` typecasts this 1-D tensor before reshape/TILE.
        # ROW_MAJOR length-1 vectors can hit ttnn typecast padding rules (last dim multiple of 32); build as TILE.
        timestep = ttnn.from_torch(
            torch.zeros(B, dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )

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
