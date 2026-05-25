# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice Generator — TTNN port of generate() from modeling_vibevoice_inference.py.

Implements the full TTS generation pipeline:
  1. Prefill: voice audio → acoustic/semantic encode → connectors → merged embeds → LM prefill
  2. AR loop: greedy token decode with speech token constraint mask
  3. Speech diffusion: DPM loop → acoustic decode → semantic re-encode → connectors
  4. CFG dual KV cache for negative prompt

No torch tensors on device in the generation loop.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import ttnn

from models.experimental.vibevoice.tt.ttnn_vibevoice_lm import (
    TTVibeVoiceLM,
    KVCache,
    create_kv_cache,
)
from models.experimental.vibevoice.tt.ttnn_speech_connector import TTSpeechConnector
from models.experimental.vibevoice.tt.ttnn_diffusion_head import TTDiffusionHead
from models.experimental.vibevoice.tt.ttnn_dpm_scheduler import (
    TTDPMSolverMultistepScheduler,
    sample_speech_latents,
)


@dataclass
class TTVibeVoiceOutput:
    sequences: torch.Tensor  # [B, S] token ids (host)
    speech_outputs: List[torch.Tensor]  # list of [T] waveforms or latents (host)


def _ttnn_argmax(logits: ttnn.Tensor, device) -> int:
    """Greedy argmax on the last token logits. Returns host Python int."""
    # logits: [B, 1, 1, vocab] or [B, 1, S, vocab]
    logits_torch = ttnn.to_torch(logits).to(torch.float32)
    # Take last position
    last = logits_torch[0, 0, -1, :]  # [vocab]
    return int(last.argmax().item())


def _apply_token_constraint(
    logits: ttnn.Tensor,
    valid_token_ids: torch.Tensor,
    device,
) -> ttnn.Tensor:
    """Zero out (set to -inf) all logits except valid_token_ids.

    logits: [B, 1, 1, vocab]
    """
    # Build mask on host, then move to device
    vocab_size = logits.shape[-1]
    mask = torch.full((1, 1, 1, vocab_size), float("-inf"), dtype=torch.bfloat16)
    mask[:, :, :, valid_token_ids] = 0.0
    mask_tt = ttnn.as_tensor(
        mask,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ttnn.add(logits, mask_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class TTVibeVoiceGenerator:
    """Full VibeVoice generation pipeline using only TT modules.

    Components provided via constructor:
      lm_tt:              TTVibeVoiceLM
      acoustic_connector: TTSpeechConnector
      semantic_connector: TTSpeechConnector
      diffusion_head:     TTDiffusionHead
      acoustic_tokenizer: TTAcousticTokenizer (encode/decode)
      semantic_tokenizer: TTSemanticTokenizer (forward/encode)
      scheduler:          TTDPMSolverMultistepScheduler
    """

    def __init__(
        self,
        lm_tt: TTVibeVoiceLM,
        acoustic_connector: TTSpeechConnector,
        semantic_connector: TTSpeechConnector,
        diffusion_head: TTDiffusionHead,
        acoustic_tokenizer,  # TTAcousticTokenizer
        semantic_tokenizer,  # TTSemanticTokenizer
        scheduler: TTDPMSolverMultistepScheduler,
        device,
        # Generation configuration
        acoustic_speech_token_ids: Optional[List[int]] = None,
        speech_start_token_id: Optional[int] = None,
        speech_end_token_id: Optional[int] = None,
        cfg_scale: float = 1.3,
        num_diffusion_steps: int = 10,
        max_new_tokens: int = 512,
        speech_scaling_factor: float = 1.0,
        speech_bias_factor: float = 0.0,
    ):
        self.lm = lm_tt
        self.acoustic_conn = acoustic_connector
        self.semantic_conn = semantic_connector
        self.diffusion_head = diffusion_head
        self.acoustic_tok = acoustic_tokenizer
        self.semantic_tok = semantic_tokenizer
        self.scheduler = scheduler
        self.device = device

        self.acoustic_speech_token_ids = acoustic_speech_token_ids or []
        self.speech_start_token_id = speech_start_token_id
        self.speech_end_token_id = speech_end_token_id
        self.cfg_scale = cfg_scale
        self.num_diffusion_steps = num_diffusion_steps
        self.max_new_tokens = max_new_tokens
        self.speech_scaling_factor = speech_scaling_factor
        self.speech_bias_factor = speech_bias_factor

    def _encode_voice(self, voice_audio_tt: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Encode voice audio → acoustic_embeds, semantic_embeds via connectors.

        voice_audio_tt: [1, 1, 1, T] float audio on device

        Returns:
            acoustic_emb: [1, T_ac, 1, hidden]
            semantic_emb: [1, T_sem, 1, hidden]
        """
        # Acoustic encode → [1, 1, T_ac, vae_dim=64]
        acoustic_latent = self.acoustic_tok.encode(voice_audio_tt)
        # Semantic encode → [1, 1, T_sem, vae_dim=128]
        semantic_latent = self.semantic_tok.forward(voice_audio_tt)

        # Apply connectors
        acoustic_emb = self.acoustic_conn(acoustic_latent)  # [1, 1, T_ac, hidden]
        semantic_emb = self.semantic_conn(semantic_latent)  # [1, 1, T_sem, hidden]
        return acoustic_emb, semantic_emb

    def _build_initial_embeds(
        self,
        input_ids: torch.Tensor,  # [1, S_text] host
        acoustic_emb: ttnn.Tensor,  # [1, 1, T_ac, hidden]
        semantic_emb: ttnn.Tensor,  # [1, 1, T_sem, hidden]
    ) -> ttnn.Tensor:
        """Build merged inputs_embeds for LM prefill.

        Pattern: [text_embeds | acoustic_embeds | semantic_embeds]
        """
        # Text embeddings (via LM embedding lookup on host)
        text_emb = self.lm._embed(input_ids)  # [1, 1, S_text, hidden]

        # Concatenate along sequence dimension (dim=2 in 4D layout)
        merged = ttnn.concat(
            [text_emb, acoustic_emb, semantic_emb],
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return merged

    def _run_speech_diffusion(
        self,
        condition: ttnn.Tensor,  # [1, 1, 1, hidden] positive cond
        neg_condition: ttnn.Tensor,  # [1, 1, 1, hidden] negative cond (uncond)
        latent_size: int = 64,
    ) -> ttnn.Tensor:
        """Run DPM diffusion loop → speech latent [1, 1, 1, latent_size]."""
        # Initial noise
        noise = torch.randn(1, 1, 1, latent_size, dtype=torch.bfloat16)
        initial_latent = ttnn.as_tensor(
            noise,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return sample_speech_latents(
            self.diffusion_head,
            condition,
            neg_condition,
            self.scheduler,
            initial_latent,
            cfg_scale=self.cfg_scale,
            num_steps=self.num_diffusion_steps,
        )

    def generate(
        self,
        input_ids: torch.Tensor,  # [1, S_text] on host
        voice_audio_tt: ttnn.Tensor,  # [1, 1, 1, T_audio] on device
        neg_input_ids: Optional[torch.Tensor] = None,  # [1, S_neg] negative prompt
        neg_voice_audio_tt: Optional[ttnn.Tensor] = None,
    ) -> TTVibeVoiceOutput:
        """Full VibeVoice TTS generation.

        Returns TTVibeVoiceOutput with generated token ids and speech waveforms.
        """
        device = self.device
        cfg = self.lm.cfg

        # 1. Encode voice audio
        acoustic_emb, semantic_emb = self._encode_voice(voice_audio_tt)

        # 2. Build merged inputs_embeds for positive prompt
        inputs_embeds = self._build_initial_embeds(input_ids, acoustic_emb, semantic_emb)

        # 3. Build negative prompt (for CFG)
        if neg_input_ids is not None and neg_voice_audio_tt is not None:
            neg_ac_emb, neg_sem_emb = self._encode_voice(neg_voice_audio_tt)
            neg_inputs_embeds = self._build_initial_embeds(neg_input_ids, neg_ac_emb, neg_sem_emb)
        else:
            neg_inputs_embeds = None

        # 4. LM prefill for positive prompt
        kv_cache_pos = create_kv_cache(cfg.num_hidden_layers)
        logits_pos, _ = self.lm.forward(
            inputs_embeds,
            start_pos=0,
            kv_cache=kv_cache_pos,
            return_last_hidden=False,
        )

        # 4b. LM prefill for negative prompt (for CFG in diffusion)
        kv_cache_neg = create_kv_cache(cfg.num_hidden_layers)
        if neg_inputs_embeds is not None:
            _, _ = self.lm.forward(
                neg_inputs_embeds,
                start_pos=0,
                kv_cache=kv_cache_neg,
                return_last_hidden=False,
            )

        # 5. AR decode loop
        generated_tokens: List[int] = []
        speech_outputs: List[torch.Tensor] = []
        current_speech_tokens: List[int] = []
        in_speech = False

        prefill_len = inputs_embeds.shape[2]
        valid_tokens_tt = (
            torch.tensor(self.acoustic_speech_token_ids, dtype=torch.long) if self.acoustic_speech_token_ids else None
        )

        prev_token = int(ttnn.to_torch(logits_pos)[0, 0, -1, :].argmax().item())

        for step in range(self.max_new_tokens):
            # Get last generated token embedding
            token_tensor = torch.tensor([[prev_token]], dtype=torch.long)
            token_emb = self.lm._embed(token_tensor)  # [1, 1, 1, hidden]

            start_pos = prefill_len + step
            logits = self.lm.decode_step(token_tensor, start_pos, kv_cache_pos)

            # Check if we're in speech generation mode and apply constraint
            if in_speech and valid_tokens_tt is not None:
                logits = _apply_token_constraint(logits, valid_tokens_tt, device)

            # Greedy decode
            next_token = _ttnn_argmax(logits, device)
            generated_tokens.append(next_token)

            # Track speech segment boundaries
            if self.speech_start_token_id is not None and next_token == self.speech_start_token_id:
                in_speech = True
                current_speech_tokens = []
            elif self.speech_end_token_id is not None and next_token == self.speech_end_token_id and in_speech:
                in_speech = False
                # Run speech diffusion for this segment
                if current_speech_tokens:
                    # Compute condition from last hidden state
                    cond_pos = self._get_last_hidden_as_condition(kv_cache_pos, step, prefill_len)
                    cond_neg = (
                        self._get_last_hidden_as_condition(kv_cache_neg, step, prefill_len)
                        if neg_inputs_embeds is not None
                        else cond_pos
                    )

                    speech_latent = self._run_speech_diffusion(cond_pos, cond_neg)
                    # Decode latent to waveform
                    waveform_tt = self.acoustic_tok.decode(speech_latent)
                    waveform = ttnn.to_torch(waveform_tt).squeeze().to(torch.float32)
                    speech_outputs.append(waveform)
                current_speech_tokens = []
            elif in_speech:
                current_speech_tokens.append(next_token)

            # EOS check
            if self.speech_end_token_id is not None and next_token == cfg.vocab_size - 1:
                break

            prev_token = next_token

        return TTVibeVoiceOutput(
            sequences=torch.tensor(generated_tokens, dtype=torch.long).unsqueeze(0),
            speech_outputs=speech_outputs,
        )

    def _get_last_hidden_as_condition(
        self,
        kv_cache: KVCache,
        step: int,
        prefill_len: int,
    ) -> ttnn.Tensor:
        """Extract a conditioning vector from the KV cache.

        Uses the last value vector from the last layer as a proxy for the hidden state.
        Returns [1, 1, 1, hidden_size].
        """
        cfg = self.lm.cfg
        last_v = kv_cache.values[-1]  # ttnn tensor [1, n_kv, S, head_dim]
        if last_v is not None:
            # Bring to host for condition computation (small tensor, one-time transfer)
            last_v_torch = ttnn.to_torch(last_v).to(torch.float32)  # [1, n_kv, S, head_dim]
            last_v_mean = last_v_torch[:, :, -1:, :].mean(dim=1)  # [1, 1, head_dim]
            hidden = cfg.hidden_size
            cond_torch = torch.zeros(1, 1, 1, hidden, dtype=torch.bfloat16)
            kv_dim = last_v_mean.shape[-1] * cfg.num_key_value_heads
            cond_torch[:, :, :, : min(kv_dim, hidden)] = last_v_mean.view(1, 1, 1, -1)[:, :, :, : min(kv_dim, hidden)]
            return ttnn.as_tensor(
                cond_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            return ttnn.zeros(
                (1, 1, 1, cfg.hidden_size),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
