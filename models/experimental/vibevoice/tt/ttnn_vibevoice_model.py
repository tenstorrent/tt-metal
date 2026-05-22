# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTVibeVoiceModel — Public API for VibeVoice-1.5B TTNN inference.

Usage:
    model = TTVibeVoiceModel.from_checkpoint(mesh_device, model_path)
    output = model.generate(input_ids_ttnn, attention_mask_ttnn, voice_audio_ttnn)

The processor (VibeVoiceProcessor) and waveform writing (scipy/soundfile) stay
outside tt/ — use them in demo_ttnn.py only.
"""

from dataclasses import dataclass
from typing import Optional, List

import torch
import ttnn

from models.experimental.vibevoice.tt.vibevoice_config import (
    load_vibevoice_model_config,
    VibeVoiceModelConfig,
)
from models.experimental.vibevoice.tt.load_weights import (
    load_vibevoice_state_dict,
    split_submodule_weights,
    remap_lm_keys_to_tt_transformers,
    fold_weight_norm,
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


@dataclass
class TTVibeVoiceOutput:
    sequences: torch.Tensor
    speech_outputs: List[torch.Tensor]


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

    @classmethod
    def from_checkpoint(
        cls,
        mesh_device,
        model_path: str,
        cfg_scale: float = 1.3,
        num_diffusion_steps: int = 10,
    ) -> "TTVibeVoiceModel":
        """Load all submodule weights and build TT model.

        Args:
            mesh_device: TTNN mesh device
            model_path:  Path to VibeVoice-1.5B checkpoint directory
            cfg_scale:   Classifier-free guidance scale
            num_diffusion_steps: DPM inference steps
        """
        config = load_vibevoice_model_config(model_path)
        state_dict = load_vibevoice_state_dict(model_path)
        sub = split_submodule_weights(state_dict)

        # ── LM ───────────────────────────────────────────────────────────
        lm_state = remap_lm_keys_to_tt_transformers(sub["lm"])
        lm_weights = preprocess_lm_weights(lm_state, mesh_device, config.decoder)
        lm = TTVibeVoiceLM(lm_weights, mesh_device)

        # ── Connectors ────────────────────────────────────────────────────
        ac_conn_params = preprocess_connector_parameters(sub["acoustic_connector"], mesh_device)
        sem_conn_params = preprocess_connector_parameters(sub["semantic_connector"], mesh_device)
        acoustic_connector = TTSpeechConnector(ac_conn_params)
        semantic_connector = TTSpeechConnector(sem_conn_params)

        # ── Diffusion head ─────────────────────────────────────────────────
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

        # ── DPM Scheduler ─────────────────────────────────────────────────
        scheduler = TTDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_schedule="cosine",
            solver_order=2,
            prediction_type="v_prediction",
        )

        # ── Tokenizers ────────────────────────────────────────────────────
        # Import lazily to avoid circular deps if tokenizers not yet implemented
        try:
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
            sem_tok_weights = preprocess_semantic_tokenizer_weights(
                sem_tok_state, mesh_device, config.semantic_tokenizer
            )
            acoustic_tokenizer = TTAcousticTokenizer(ac_tok_weights, mesh_device)
            semantic_tokenizer = TTSemanticTokenizer(sem_tok_weights, mesh_device)
        except ImportError:
            # Tokenizer implementations pending — use None placeholders
            acoustic_tokenizer = None
            semantic_tokenizer = None

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
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        voice_audio_tt: Optional[ttnn.Tensor] = None,
        neg_input_ids: Optional[torch.Tensor] = None,
        neg_voice_audio_tt: Optional[ttnn.Tensor] = None,
        cfg_scale: float = 1.3,
        num_diffusion_steps: int = 10,
        max_new_tokens: int = 512,
    ) -> TTVibeVoiceOutput:
        """Run VibeVoice TTS generation.

        Args:
            input_ids:       [1, S] host torch.LongTensor (text token ids)
            voice_audio_tt:  [1, 1, 1, T] device ttnn.Tensor (reference voice)
            neg_input_ids:   [1, S] negative prompt (optional, for CFG)
            neg_voice_audio_tt: negative voice (optional)
            cfg_scale:       guidance scale
            num_diffusion_steps: DPM steps
            max_new_tokens:  max AR generation steps

        Returns:
            TTVibeVoiceOutput with sequences and speech_outputs
        """
        generator = TTVibeVoiceGenerator(
            lm_tt=self._lm,
            acoustic_connector=self._ac_conn,
            semantic_connector=self._sem_conn,
            diffusion_head=self._diff_head,
            acoustic_tokenizer=self._ac_tok,
            semantic_tokenizer=self._sem_tok,
            scheduler=self._scheduler,
            device=self._device,
            cfg_scale=cfg_scale,
            num_diffusion_steps=num_diffusion_steps,
            max_new_tokens=max_new_tokens,
        )
        return generator.generate(
            input_ids=input_ids,
            voice_audio_tt=voice_audio_tt,
            neg_input_ids=neg_input_ids,
            neg_voice_audio_tt=neg_voice_audio_tt,
        )
