# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTVibeVoiceModel — Public API for VibeVoice-1.5B TTNN inference.

Usage:
    model = TTVibeVoiceModel.from_checkpoint(mesh_device, model_path)
    output = model.generate(**processor_batch, tokenizer=processor.tokenizer)
"""

from typing import Optional

import torch

from models.experimental.vibevoice.tt.vibevoice_config import (
    load_vibevoice_model_config,
    VibeVoiceModelConfig,
)
from models.experimental.vibevoice.tt.load_weights import (
    load_vibevoice_state_dict,
    split_submodule_weights,
    remap_lm_keys_to_tt_transformers,
    fold_weight_norm,
    load_speech_scale_bias,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import (
    preprocess_lm_weights,
    TTVibeVoiceLM,
)
from models.experimental.vibevoice.tt.ttnn_speech_connector import (
    preprocess_connector_parameters,
    TTSpeechConnector,
)
from models.experimental.vibevoice.tt.ttnn_diffusion_head import (
    preprocess_diffusion_head_weights,
    TTDiffusionHead,
)
from models.experimental.vibevoice.tt.ttnn_dpm_scheduler import (
    TTDPMSolverMultistepScheduler,
)
from models.experimental.vibevoice.tt.ttnn_vibevoice_generator import (
    TTVibeVoiceGenerator,
    TTVibeVoiceOutput,
)


class TTVibeVoiceModel:
    """Assembled VibeVoice-1.5B TT model — load once, generate repeatedly."""

    def __init__(
        self,
        lm: TTVibeVoiceLM,
        acoustic_connector: TTSpeechConnector,
        semantic_connector: TTSpeechConnector,
        diffusion_head: TTDiffusionHead,
        acoustic_tokenizer,
        semantic_tokenizer,
        scheduler: TTDPMSolverMultistepScheduler,
        config: VibeVoiceModelConfig,
        device,
        speech_scaling_factor: Optional[float] = None,
        speech_bias_factor: Optional[float] = None,
    ):
        self._lm = lm
        self._ac_conn = acoustic_connector
        self._sem_conn = semantic_connector
        self._diff_head = diffusion_head
        self._ac_tok = acoustic_tokenizer
        self._sem_tok = semantic_tokenizer
        self._scheduler = scheduler
        self._config = config
        self._device = device
        self._speech_scaling_factor = speech_scaling_factor
        self._speech_bias_factor = speech_bias_factor

    @classmethod
    def from_checkpoint(
        cls,
        mesh_device,
        model_path: str,
        cfg_scale: float = 1.3,
        num_diffusion_steps: int = 10,
    ) -> "TTVibeVoiceModel":
        """Load all submodule weights and build TT model."""
        config = load_vibevoice_model_config(model_path)
        state_dict = load_vibevoice_state_dict(model_path)
        sub = split_submodule_weights(state_dict)
        speech_scale, speech_bias = load_speech_scale_bias(state_dict)

        lm_state = remap_lm_keys_to_tt_transformers(sub["lm"])
        lm_weights = preprocess_lm_weights(lm_state, mesh_device, config.decoder)
        lm = TTVibeVoiceLM(lm_weights, mesh_device)

        ac_conn_params = preprocess_connector_parameters(sub["acoustic_connector"], mesh_device)
        sem_conn_params = preprocess_connector_parameters(sub["semantic_connector"], mesh_device)
        acoustic_connector = TTSpeechConnector(ac_conn_params)
        semantic_connector = TTSpeechConnector(sem_conn_params)

        diff_cfg = config.diffusion_head
        diff_head_weights = preprocess_diffusion_head_weights(
            sub["diffusion_head"],
            mesh_device,
            hidden_size=diff_cfg.hidden_size,
            latent_size=diff_cfg.latent_size,
            head_ffn_ratio=diff_cfg.head_ffn_ratio,
            norm_eps=diff_cfg.rms_norm_eps,
            num_layers=diff_cfg.head_layers,
        )
        diffusion_head = TTDiffusionHead(diff_head_weights)

        scheduler = TTDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="cosine",
            solver_order=2,
            prediction_type="v_prediction",
        )

        from models.experimental.vibevoice.tt.ttnn_acoustic_tokenizer import (
            TTAcousticTokenizer,
            preprocess_acoustic_tokenizer_weights,
        )
        from models.experimental.vibevoice.tt.ttnn_semantic_tokenizer import (
            TTSemanticTokenizer,
            preprocess_semantic_tokenizer_weights,
        )

        ac_tok_state = fold_weight_norm(sub["acoustic_tokenizer"])
        sem_tok_state = fold_weight_norm(sub["semantic_tokenizer"])

        ac_tok_weights = preprocess_acoustic_tokenizer_weights(ac_tok_state, mesh_device, config.acoustic_tokenizer)
        sem_tok_weights = preprocess_semantic_tokenizer_weights(sem_tok_state, mesh_device, config.semantic_tokenizer)
        acoustic_tokenizer = TTAcousticTokenizer(ac_tok_weights, mesh_device)
        semantic_tokenizer = TTSemanticTokenizer(sem_tok_weights, mesh_device)

        return cls(
            lm=lm,
            acoustic_connector=acoustic_connector,
            semantic_connector=semantic_connector,
            diffusion_head=diffusion_head,
            acoustic_tokenizer=acoustic_tokenizer,
            semantic_tokenizer=semantic_tokenizer,
            scheduler=scheduler,
            config=config,
            device=mesh_device,
            speech_scaling_factor=speech_scale,
            speech_bias_factor=speech_bias,
        )

    def set_speech_scale_bias(self, scaling_factor: float, bias_factor: float) -> None:
        """Set runtime speech scaling (from reference after voice prefill)."""
        self._speech_scaling_factor = scaling_factor
        self._speech_bias_factor = bias_factor

    def _make_generator(
        self,
        tokenizer,
        cfg_scale: float,
        num_diffusion_steps: int,
        max_new_tokens: Optional[int],
        ref_inference=None,
    ) -> TTVibeVoiceGenerator:
        return TTVibeVoiceGenerator(
            lm_tt=self._lm,
            acoustic_connector=self._ac_conn,
            semantic_connector=self._sem_conn,
            diffusion_head=self._diff_head,
            acoustic_tokenizer=self._ac_tok,
            semantic_tokenizer=self._sem_tok,
            scheduler=self._scheduler,
            device=self._device,
            speech_start_id=tokenizer.speech_start_id,
            speech_end_id=tokenizer.speech_end_id,
            speech_diffusion_id=tokenizer.speech_diffusion_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=getattr(tokenizer, "bos_token_id", None),
            cfg_scale=cfg_scale,
            num_diffusion_steps=num_diffusion_steps,
            max_new_tokens=max_new_tokens,
            speech_scaling_factor=self._speech_scaling_factor,
            speech_bias_factor=self._speech_bias_factor,
            acoustic_fix_std=self._config.acoustic_tokenizer.fix_std,
            ref_inference=ref_inference,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.Tensor] = None,
        speech_masks: Optional[torch.Tensor] = None,
        speech_input_mask: Optional[torch.Tensor] = None,
        tokenizer=None,
        cfg_scale: float = 1.3,
        num_diffusion_steps: int = 10,
        prefill_speech_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        rng: Optional[torch.Generator] = None,
        ref_inference=None,
    ) -> TTVibeVoiceOutput:
        """Run VibeVoice TTS generation (processor batch fields + tokenizer).

        Pass ``ref_inference`` (loaded VibeVoiceForConditionalGenerationInference) to
        drive the AR loop with CPU fp32 reference LM hidden states + ref diffusion/fusion
        for parity with HuggingFace generate().
        """
        if tokenizer is None:
            raise ValueError("tokenizer is required (VibeVoiceProcessor.tokenizer)")

        generator = self._make_generator(
            tokenizer, cfg_scale, num_diffusion_steps, max_new_tokens, ref_inference=ref_inference
        )
        return generator.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            speech_tensors=speech_tensors,
            speech_masks=speech_masks,
            speech_input_mask=speech_input_mask,
            prefill_speech_embeds=prefill_speech_embeds,
            max_new_tokens=max_new_tokens,
            rng=rng,
        )
