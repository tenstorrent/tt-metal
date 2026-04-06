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
import torch
from loguru import logger
from reference.utils import VA_CONFIGS

from models.experimental.lingbot_va.tests.demo import demo as lingbot_demo


class TtLingbotVA:
    """Orchestrates Lingbot-VA TTNN inference like ``run_inference``; forward runs ``demo._infer_impl``."""

    def __init__(
        self,
        models: dict,
        state: dict,
        message: dict,
        init_obs: dict,
        *,
        checkpoint_path: str,
        save_dir: str,
        num_inference_steps: int | None,
        action_num_inference_steps: int | None,
    ) -> None:
        self.models = models
        self.state = state
        self.message = message
        self._init_obs = init_obs
        self._checkpoint_path = checkpoint_path
        self._save_dir = save_dir
        self._num_inference_steps = num_inference_steps
        self._action_num_inference_steps = action_num_inference_steps

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
            num_inference_steps / action_num_inference_steps: same as ``run_inference``.
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
        config.num_chunks_to_infer = 1
        if num_inference_steps is not None:
            config.num_inference_steps = num_inference_steps
        if action_num_inference_steps is not None:
            config.action_num_inference_steps = action_num_inference_steps
        config.frame_chunk_size = 2
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

        return cls(
            models,
            state,
            message,
            init_obs,
            checkpoint_path=str(checkpoint_path),
            save_dir=str(save_dir),
            num_inference_steps=num_inference_steps,
            action_num_inference_steps=action_num_inference_steps,
        )

    def forward_reset_and_infer(self) -> ttnn.Tensor:
        """Run :func:`demo.run_inference` prepared path (same as full infer chunk); re-upload latents for pipeline."""
        result = lingbot_demo.run_inference(
            self.message,
            self._checkpoint_path,
            save_dir=self._save_dir,
            num_inference_steps=self._num_inference_steps,
            action_num_inference_steps=self._action_num_inference_steps,
            prepared=(self.models, self.state, self._init_obs),
            return_latents=True,
        )
        latents_torch = result["latents"]
        mesh_device = self.models["mesh_device"]
        dtype = self.models["dtype"]
        tt_dtype = ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32
        return ttnn.from_torch(
            latents_torch.detach().contiguous(),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=tt_dtype,
        )
        _ = _actions_tt
        return latents_tt

    def __call__(self, l1_input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Pipeline entrypoint: :func:`demo.run_inference` (prepared) then latents re-uploaded as ``ttnn``."""
        _ = l1_input_tensor
        return self.forward_reset_and_infer()


class TtLingbotVAEncodeOnly:
    """Pipeline-style perf for VAE observation encode only (no transformer).

    :meth:`prepare` runs tokenizer, TT text encoder, :func:`demo._encode_prompt_ttnn`, frees the text encoder,
    then loads the TT VAE encoder (same ordering as full ``run_inference`` to stay within DRAM).

    Each ``__call__`` / forward only runs :func:`demo._encode_obs_ttnn` (timed pipeline iterations).
    """

    def __init__(
        self,
        models: dict,
        state: dict,
        init_obs: dict,
    ) -> None:
        self.models = models
        self.state = state
        self._init_obs = init_obs

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
    ) -> TtLingbotVAEncodeOnly:
        """Tokenize, TT text encoder + prompt encode, then TT VAE load; leaves VAE resident for ``__call__``."""

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
        config.num_chunks_to_infer = 1
        if num_inference_steps is not None:
            config.num_inference_steps = num_inference_steps
        if action_num_inference_steps is not None:
            config.action_num_inference_steps = action_num_inference_steps
        config.frame_chunk_size = 2
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

        return cls(models, state, init_obs)

    def __call__(self, l1_input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Observation encode only; prompt was encoded in :meth:`prepare`.

        The tt_cnn pipeline calls this once per host input with no logging inside the executor: each
        call is a full ``_encode_obs_ttnn`` (TT VAE). ``compile`` runs it once; ``enqueue`` runs it
        ``num_iterations`` more times — long gaps with no output are normal, not a hang.
        """
        _ = l1_input_tensor
        logger.info("TtLingbotVAEncodeOnly: running _encode_obs_ttnn (pipeline compile or enqueue step)")
        return lingbot_demo._encode_obs_ttnn(self.models, self.state, self._init_obs)
