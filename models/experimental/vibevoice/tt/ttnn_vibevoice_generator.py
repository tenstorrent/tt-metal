# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
VibeVoice Generator — TTNN port of generate() from modeling_vibevoice_inference.py.

Pipeline (aligned with reference):
  1. Prefill: processor speech_tensors/masks → acoustic encode → scatter into inputs_embeds
  2. AR loop: greedy decode with valid-token constraint
  3. On speech_diffusion_id: CFG diffusion → decode → semantic encode → connector sum → next embed
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
    sequences: torch.Tensor  # [B, S] full token ids (prefill + generated)
    speech_outputs: List[torch.Tensor]  # concatenated waveforms per batch row


def _ttnn_argmax(logits: ttnn.Tensor) -> int:
    """Greedy argmax on last position logits. Returns host Python int."""
    logits_torch = ttnn.to_torch(logits).to(torch.float32)
    last = logits_torch[0, 0, -1, :]
    return int(last.argmax().item())


def _apply_token_constraint(
    logits: ttnn.Tensor,
    valid_token_ids: List[int],
    device,
) -> ttnn.Tensor:
    """Mask logits so only valid_token_ids are selectable."""
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


def _embeds_to_host_2d(inputs_embeds: ttnn.Tensor) -> torch.Tensor:
    """[1, 1, S, H] device tensor → [S, H] float32 on host."""
    return ttnn.to_torch(inputs_embeds).to(torch.float32).squeeze(0).squeeze(0)


def _host_2d_to_embeds(embeds_2d: torch.Tensor, device) -> ttnn.Tensor:
    """[S, H] or [1, H] bfloat16 host → [1, 1, S, H] on device."""
    if embeds_2d.dim() == 1:
        embeds_2d = embeds_2d.unsqueeze(0)
    return ttnn.as_tensor(
        embeds_2d.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _condition_from_hidden(last_hidden: ttnn.Tensor) -> ttnn.Tensor:
    """last_hidden [1,1,S,H] → condition [1,1,1,H] at last position."""
    h = last_hidden.shape[2] - 1
    return ttnn.slice(
        last_hidden,
        [0, 0, h, 0],
        [1, 1, h + 1, last_hidden.shape[-1]],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


class TTVibeVoiceGenerator:
    """Full VibeVoice generation pipeline using TT modules."""

    def __init__(
        self,
        lm_tt: TTVibeVoiceLM,
        acoustic_connector: TTSpeechConnector,
        semantic_connector: TTSpeechConnector,
        diffusion_head: TTDiffusionHead,
        acoustic_tokenizer,
        semantic_tokenizer,
        scheduler: TTDPMSolverMultistepScheduler,
        device,
        speech_start_id: int,
        speech_end_id: int,
        speech_diffusion_id: int,
        eos_token_id: int,
        bos_token_id: Optional[int] = None,
        cfg_scale: float = 1.3,
        num_diffusion_steps: int = 10,
        max_new_tokens: Optional[int] = None,
        max_length_times: float = 20.0,
        speech_scaling_factor: Optional[float] = None,
        speech_bias_factor: Optional[float] = None,
        acoustic_fix_std: float = 0.5,
        cpu_acoustic_decoder=None,
    ):
        self.lm = lm_tt
        self.acoustic_conn = acoustic_connector
        self.semantic_conn = semantic_connector
        self.diffusion_head = diffusion_head
        self.acoustic_tok = acoustic_tokenizer
        self.semantic_tok = semantic_tokenizer
        self.scheduler = scheduler
        self.device = device
        # Optional CPU acoustic decoder for streaming-correct final audio decode.
        # If set, accumulated latent frames are decoded on CPU in a single pass
        # (full causal context), matching the reference streaming acoustic cache.
        self.cpu_acoustic_decoder = cpu_acoustic_decoder

        self.speech_start_id = speech_start_id
        self.speech_end_id = speech_end_id
        self.speech_diffusion_id = speech_diffusion_id
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.cfg_scale = cfg_scale
        self.num_diffusion_steps = num_diffusion_steps
        self.max_new_tokens = max_new_tokens
        self.max_length_times = max_length_times
        self.speech_scaling_factor = speech_scaling_factor
        self.speech_bias_factor = speech_bias_factor
        self.acoustic_fix_std = acoustic_fix_std

        self.valid_token_ids = [
            speech_start_id,
            speech_end_id,
            speech_diffusion_id,
            eos_token_id,
        ]
        if bos_token_id is not None:
            self.valid_token_ids.append(bos_token_id)

    def _audio_row_to_tt(self, wav_1d: torch.Tensor) -> ttnn.Tensor:
        """1D waveform [T] → [1, 1, 1, T] on device."""
        audio = wav_1d.to(torch.bfloat16).view(1, 1, 1, -1)
        return ttnn.as_tensor(
            audio,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _encode_acoustic_latents(self, wav_1d: torch.Tensor) -> torch.Tensor:
        """Encode audio → [T_enc, vae_dim] float32 on host (with fix-std sampling)."""
        audio_tt = self._audio_row_to_tt(wav_1d)
        lat_tt = self.acoustic_tok.encode(audio_tt)
        lat = ttnn.to_torch(lat_tt).to(torch.float32).squeeze(0).squeeze(0)  # [T_enc, D]
        if self.acoustic_fix_std:
            lat = lat + self.acoustic_fix_std * torch.randn_like(lat)
        return lat

    def _compute_scale_bias(self, latents_list: List[torch.Tensor], speech_masks: torch.Tensor):
        """Match reference: scale=1/std(masked), bias=-mean(masked) on stacked latents."""
        parts = []
        for i in range(speech_masks.shape[0]):
            n = int(speech_masks[i].sum().item())
            if n > 0:
                parts.append(latents_list[i][:n].reshape(-1, latents_list[i].shape[-1]))
        if not parts:
            return 1.0, 0.0
        flat = torch.cat(parts, dim=0).flatten()
        return (1.0 / flat.std()).item(), (-flat.mean()).item()

    def _process_speech_prefill(
        self,
        speech_tensors: torch.Tensor,
        speech_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return speech_embeds [N_slots, hidden] for scatter into prefill (host float32)."""
        scale = self.speech_scaling_factor
        bias = self.speech_bias_factor
        latents_per_row = []
        for i in range(speech_tensors.shape[0]):
            latents_per_row.append(self._encode_acoustic_latents(speech_tensors[i]))

        if scale is None or bias is None:
            scale, bias = self._compute_scale_bias(latents_per_row, speech_masks)
            self.speech_scaling_factor = scale
            self.speech_bias_factor = bias

        speech_embeds_parts = []
        for i in range(speech_tensors.shape[0]):
            n = int(speech_masks[i].sum().item())
            feats = (latents_per_row[i][:n] + bias) * scale
            feats_tt = ttnn.as_tensor(
                feats.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                device=self.device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            conn_out = self.acoustic_conn(feats_tt)
            conn_torch = ttnn.to_torch(conn_out).to(torch.float32)
            if conn_torch.dim() == 4:
                conn_torch = conn_torch.squeeze(0).squeeze(0)
            else:
                conn_torch = conn_torch.squeeze(0)
            conn_torch = conn_torch[:, :n, :]
            speech_embeds_parts.append(conn_torch)

        return torch.cat(speech_embeds_parts, dim=0)

    def _build_prefill_embeds(
        self,
        input_ids: torch.Tensor,
        speech_tensors: Optional[torch.Tensor],
        speech_masks: Optional[torch.Tensor],
        speech_input_mask: Optional[torch.Tensor],
        prefill_speech_embeds: Optional[torch.Tensor] = None,
    ) -> ttnn.Tensor:
        """Text embeds with speech slots scattered (reference forward prefill)."""
        inputs_embeds = self.lm._embed(input_ids)
        if speech_input_mask is None:
            return inputs_embeds

        embed_2d = _embeds_to_host_2d(inputs_embeds)
        if prefill_speech_embeds is not None:
            speech_embeds = prefill_speech_embeds.to(torch.float32)
        elif speech_tensors is not None and speech_masks is not None:
            speech_embeds = self._process_speech_prefill(speech_tensors, speech_masks)
        else:
            return inputs_embeds
        mask = speech_input_mask[0].cpu().bool()
        n_slots = int(mask.sum().item())
        embed_2d[mask[: embed_2d.shape[0]]] = speech_embeds[:n_slots].to(embed_2d.dtype)
        return _host_2d_to_embeds(embed_2d, self.device)

    def _run_speech_diffusion(
        self,
        condition: ttnn.Tensor,
        neg_condition: ttnn.Tensor,
        latent_size: int = 64,
        rng: Optional[torch.Generator] = None,
    ) -> ttnn.Tensor:
        # Draw 2×latent_size values to match reference's torch.randn(2, vae_dim)
        # (reference cats pos+neg into batch=2, draws one noise per batch entry,
        # then uses speech[:1]).  This keeps our global RNG state aligned.
        noise_2x = torch.randn(2, 1, 1, latent_size, dtype=torch.bfloat16, generator=rng)
        noise = noise_2x[:1]
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

    def _post_diffusion_embeds(self, speech_latent: ttnn.Tensor) -> ttnn.Tensor:
        """Diffusion latent → acoustic+semantic connector sum for next LM step."""
        scale = self.speech_scaling_factor or 1.0
        bias = self.speech_bias_factor or 0.0

        lat_torch = ttnn.to_torch(speech_latent).to(torch.float32)
        scaled = lat_torch / scale - bias
        scaled_tt = ttnn.as_tensor(
            scaled.to(torch.bfloat16),
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        audio_tt = self.acoustic_tok.decode(scaled_tt)
        semantic_tt = self.semantic_tok.forward(audio_tt)

        acoustic_embed = self.acoustic_conn(speech_latent)
        semantic_embed = self.semantic_conn(semantic_tt)
        return ttnn.add(acoustic_embed, semantic_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _reset_neg_cache(self, kv_cache_neg: KVCache):
        """Negative prefill: single speech_start token.

        Returns (neg_pos=1, neg_hidden) where neg_hidden is the hidden state of
        speech_start_id with no prior context — matches the reference's first
        negative forward which processes [speech_start_id] alone.
        """
        neg_ids = torch.tensor([[self.speech_start_id]], dtype=torch.long)
        neg_embeds = self.lm._embed(neg_ids)
        _, neg_hidden = self.lm.forward(neg_embeds, start_pos=0, kv_cache=kv_cache_neg, return_last_hidden=True)
        return 1, neg_hidden

    def _lm_step(
        self,
        inputs_embeds: ttnn.Tensor,
        start_pos: int,
        kv_cache: KVCache,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        logits, last_hidden = self.lm.forward(
            inputs_embeds,
            start_pos=start_pos,
            kv_cache=kv_cache,
            return_last_hidden=True,
        )
        return logits, last_hidden

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.Tensor] = None,
        speech_masks: Optional[torch.Tensor] = None,
        speech_input_mask: Optional[torch.Tensor] = None,
        prefill_speech_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        rng: Optional[torch.Generator] = None,
    ) -> TTVibeVoiceOutput:
        """Run VibeVoice TTS generation aligned with reference generate()."""
        device = self.device
        cfg = self.lm.cfg

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        inputs_embeds = self._build_prefill_embeds(
            input_ids,
            speech_tensors,
            speech_masks,
            speech_input_mask,
            prefill_speech_embeds=prefill_speech_embeds,
        )
        prefill_len = inputs_embeds.shape[2]

        kv_cache_pos = create_kv_cache(cfg.num_hidden_layers)
        logits_pos, prefill_hidden = self.lm.prefill_embeds(
            inputs_embeds,
            kv_cache=kv_cache_pos,
            return_last_hidden=True,
        )

        kv_cache_neg = create_kv_cache(cfg.num_hidden_layers)
        neg_pos, neg_start_hidden = self._reset_neg_cache(kv_cache_neg)
        neg_prev_diffusion_token: Optional[int] = None  # delayed token for negative CFG

        initial_length = input_ids.shape[-1]
        initial_len = int(attention_mask.sum(dim=-1)[0].item())
        if max_new_tokens is not None:
            max_steps = max_new_tokens
        else:
            max_steps = min(
                cfg.max_position_embeddings - initial_length,
                int(self.max_length_times * initial_len),
            )

        sequences = input_ids.clone()
        # Accumulate unscaled latent frames [1,1,1,vae_dim] on host; batch-decode
        # at the end so the causal-conv decoder sees full left context for every
        # frame (identical to the reference's streaming acoustic_cache).
        speech_latent_frames: List[torch.Tensor] = []
        pending_embeds: Optional[ttnn.Tensor] = None

        next_token = _ttnn_argmax(logits_pos)
        step_hidden = prefill_hidden

        scale = self.speech_scaling_factor or 1.0
        bias = self.speech_bias_factor or 0.0

        for step in range(max_steps):
            current_token = next_token
            sequences = torch.cat(
                [sequences, torch.tensor([[current_token]], dtype=torch.long)],
                dim=-1,
            )

            if current_token == self.speech_diffusion_id:
                cond_pos = _condition_from_hidden(step_hidden)
                # Negative CFG: reference processes the PREVIOUS speech_diffusion_id
                # at each step (the current one is appended to negative_input_ids
                # AFTER the negative forward).  For the first diffusion step in a
                # segment, the reference runs the negative model on speech_start_id
                # alone — we captured that hidden in neg_start_hidden.
                if neg_prev_diffusion_token is None:
                    neg_hidden = neg_start_hidden
                else:
                    neg_ids = torch.tensor([[neg_prev_diffusion_token]], dtype=torch.long)
                    _, neg_hidden = self.lm.decode_step(neg_ids, neg_pos, kv_cache_neg, return_last_hidden=True)
                    neg_pos += 1
                neg_prev_diffusion_token = current_token
                cond_neg = _condition_from_hidden(neg_hidden)

                speech_latent = self._run_speech_diffusion(cond_pos, cond_neg, latent_size=64, rng=rng)

                # Accumulate unscaled latent frame for batch decode later.
                speech_latent_frames.append(ttnn.to_torch(speech_latent).to(torch.float32))

                pending_embeds = self._post_diffusion_embeds(speech_latent)

            if current_token == self.eos_token_id:
                break

            start_pos = prefill_len + step
            if pending_embeds is not None:
                logits, step_hidden = self._lm_step(pending_embeds, start_pos, kv_cache_pos)
                pending_embeds = None
            else:
                token_ids = torch.tensor([[current_token]], dtype=torch.long)
                logits, step_hidden = self.lm.decode_step(token_ids, start_pos, kv_cache_pos, return_last_hidden=True)

            if current_token == self.speech_start_id:
                neg_pos, neg_start_hidden = self._reset_neg_cache(kv_cache_neg)
                neg_prev_diffusion_token = None

            logits = _apply_token_constraint(logits, self.valid_token_ids, device)
            next_token = _ttnn_argmax(logits)

        # Batch-decode all accumulated latent frames in a single pass so the
        # causal-conv decoder sees full left context for every frame, matching
        # the reference streaming acoustic cache exactly.
        if speech_latent_frames:
            # Stack: [n, 1, 1, D] → [1, n, D] for the acoustic decoder
            latents_stacked = torch.cat(speech_latent_frames, dim=0)  # [n, 1, 1, D]
            latents_cpu = latents_stacked.squeeze(1).squeeze(1)  # [n, D]
            latents_cpu = latents_cpu.unsqueeze(0)  # [1, n, D]
            # Inverse-normalise: diffusion latent → acoustic VAE latent space
            latents_unscaled = (latents_cpu / scale - bias).to(torch.float32)

            if self.cpu_acoustic_decoder is not None:
                # CPU reference decoder: correct streaming via full causal context.
                with torch.no_grad():
                    audio_cpu = self.cpu_acoustic_decoder.decode(latents_unscaled)
                speech_waveform = audio_cpu.to(torch.float32).reshape(-1)
            else:
                # Fallback: TT device decode (may fail for large n_frames).
                lat_tt = ttnn.as_tensor(
                    latents_unscaled.unsqueeze(1).to(torch.bfloat16),
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                audio_tt = self.acoustic_tok.decode(lat_tt)
                speech_waveform = ttnn.to_torch(audio_tt).to(torch.float32).reshape(-1)
        else:
            speech_waveform = torch.zeros(0)

        return TTVibeVoiceOutput(
            sequences=sequences,
            speech_outputs=[speech_waveform],
        )
