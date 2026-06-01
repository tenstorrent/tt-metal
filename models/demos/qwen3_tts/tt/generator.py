# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-TTS inference pipeline on TT hardware.

Full pipeline:
    Text -> Tokenize (CPU)
    -> (Optional) Ref audio -> Mel (CPU) -> Speaker Encoder (TT) -> speaker_emb
    -> Text tokens + speaker_emb -> Talker (TT, autoregressive) -> CB0 tokens
    -> CB0 hidden states -> Code Predictor (TT) -> CB1-CB15 tokens
    -> All 16 codebooks -> Vocoder (TT) -> 24kHz waveform

Input embedding construction (from HF modeling_qwen3_tts.py):
    Each position in the prefill sequence has TWO embeddings added together:
      text_side = text_embedding(text_token) -> text_projection (Linear->SiLU->Linear)
      codec_side = codec_embedding(codec_token)
      input = text_side + codec_side

    Non-streaming layout (batch=1, language specified):
      Pos  | text_side                          | codec_side
      -----|------------------------------------|-----------------------------
      0    | text_proj(<|im_start|>)            | -  (role prefix, no codec)
      1    | text_proj(assistant)               | -
      2    | text_proj(\n)                      | -
      3    | tts_pad                            | codec(think_id)
      4    | tts_pad                            | codec(think_bos_id)
      5    | tts_pad                            | codec(language_id)
      6    | tts_pad                            | codec(think_eos_id)
     (7)   | tts_pad                            | speaker_embed  (voice cloning only)
      N    | tts_bos                            | codec(pad_id)
      N+1..| text_proj(text_tokens) + tts_eos   | codec_pad * (N_text+1)
      last | tts_pad                            | codec_bos

    During decode, the input at each step is:
      sum(codec_embedding[i](cb_i_token) for all 16 codebooks) + trailing_text_hidden
"""

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

import ttnn
from models.tt_transformers.tt.common import Mode


class TTSGenerator:
    """
    End-to-end TTS inference pipeline on TT device.
    Orchestrates all four model components.
    """

    def __init__(
        self,
        talker,
        code_predictor,
        speaker_encoder,
        vocoder,
        talker_args,
        mesh_device,
        tokenizer=None,
    ):
        self.talker = talker
        self.code_predictor = code_predictor
        self.speaker_encoder = speaker_encoder
        self.vocoder = vocoder
        self.talker_args = talker_args
        self.mesh_device = mesh_device
        self.tokenizer = tokenizer

    @classmethod
    def build(cls, model_path, mesh_device, dtype=ttnn.bfloat16, max_batch_size=1, max_seq_len=4096):
        """
        Build the full TTS pipeline from a HuggingFace checkpoint.

        Args:
            model_path: Path to Qwen3-TTS checkpoint
            mesh_device: TT mesh device
            dtype: Weight dtype (default bfloat16)
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length for Talker
        """
        import os

        from models.demos.qwen3_tts.tt.model_config import TalkerModelArgs

        os.environ["HF_MODEL"] = model_path

        talker_args = TalkerModelArgs(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            use_hf_rope=True,
        )

        state_dict = talker_args.load_state_dict()
        weight_cache_path = talker_args.weight_cache_path(dtype)

        from models.demos.qwen3_tts.tt.talker import TalkerTransformer

        logger.info("Building Talker...")
        talker = TalkerTransformer(
            args=talker_args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
        )

        # Build Code Predictor (TT device, KV cache)
        from models.demos.qwen3_tts.tt.code_predictor import CodePredictorTransformer
        from models.demos.qwen3_tts.tt.model_config import CodePredictorModelArgs

        logger.info("Building Code Predictor (TT)...")
        cp_args = CodePredictorModelArgs(
            mesh_device=mesh_device,
            max_batch_size=max_batch_size,
            max_seq_len=128,
            use_hf_rope=True,
        )
        cp_state_dict = cp_args.load_state_dict()
        cp_weight_cache = cp_args.weight_cache_path(dtype)
        code_predictor = CodePredictorTransformer(
            args=cp_args,
            dtype=dtype,
            mesh_device=mesh_device,
            state_dict=cp_state_dict,
            weight_cache_path=cp_weight_cache,
        )

        # Build Speaker Encoder (runs on host CPU, small model)
        from models.demos.qwen3_tts.tt.speaker_encoder import SpeakerEncoder

        logger.info("Building Speaker Encoder...")
        speaker_encoder = SpeakerEncoder.from_pretrained(model_path, mesh_device)

        # Build Vocoder (Code2Wav, runs on host CPU, ~114M params)
        from models.demos.qwen3_tts.tt.vocoder import Vocoder

        logger.info("Building Vocoder...")
        try:
            vocoder = Vocoder.from_pretrained(model_path)
        except Exception as e:
            logger.warning(f"Vocoder weights not available: {e}")
            vocoder = None

        # Load tokenizer
        from transformers import AutoTokenizer

        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        return cls(talker, code_predictor, speaker_encoder, vocoder, talker_args, mesh_device, tokenizer)

    def generate(
        self,
        text: str,
        language: str = "japanese",
        ref_audio: Optional[np.ndarray] = None,
        ref_sr: int = 24000,
        speaker_emb_tt=None,
        max_new_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.

        Args:
            text: Input text to synthesize
            language: Language code (e.g., "japanese")
            ref_audio: Optional reference audio for voice cloning [num_samples]
            ref_sr: Sample rate of reference audio
            speaker_emb_tt: Optional pre-loaded speaker embedding (ttnn.Tensor).
                           Takes priority over ref_audio if both are provided.
            max_new_tokens: Maximum codec tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Tuple of (waveform_numpy, sample_rate)
        """
        t0 = time.time()

        # Step 1+2: Get speaker embedding as CPU tensor (if voice cloning)
        speaker_emb_torch = None
        if speaker_emb_tt is not None:
            speaker_emb_torch = ttnn.to_torch(
                speaker_emb_tt, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1),
            ).squeeze()[:self.talker_args.dim].unsqueeze(0)  # [1, dim]
            logger.info(f"Using pre-loaded speaker embedding (norm={speaker_emb_torch.norm():.4f})")
        elif ref_audio is not None:
            speaker_emb_torch = self.speaker_encoder.extract_embedding(ref_audio, ref_sr)  # [1, dim]
            logger.info(f"Speaker embedding extracted (norm={speaker_emb_torch.norm():.4f})")

        # Build input embeddings with speaker position inserted (CPU + device)
        input_embeds, trailing_text_hidden, tts_pad_embed = self._build_input_embeds(
            text, language, speaker_emb_torch=speaker_emb_torch,
        )
        seq_len = input_embeds.shape[1]
        logger.info(f"Built input embeddings: seq_len={seq_len}")

        # Step 3+4: Talker (CB0) + Code Predictor (CB1-15) per decode step
        all_codebooks = self._generate_and_predict(
            input_embeds, trailing_text_hidden, tts_pad_embed,
            None, max_new_tokens, temperature, top_k, top_p,
            repetition_penalty=repetition_penalty,
        )
        num_frames = all_codebooks.shape[1]
        logger.info(f"Generated {num_frames} frames (16 codebooks each)")

        # Step 5: Vocoder -> waveform (TT)
        waveform = self._decode_waveform(all_codebooks)

        elapsed = time.time() - t0
        duration = len(waveform) / 24000
        logger.info(f"Generated {duration:.2f}s audio in {elapsed:.2f}s (RTF={elapsed/duration:.3f})")

        return waveform, 24000

    def _build_input_embeds(
        self, text: str, language: str, speaker_emb_torch: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the full prefill input embeddings (text_side + codec_side).

        Follows the HF Qwen3-TTS generate() logic (modeling_qwen3_tts.py:2082-2226).
        Non-streaming mode: all text tokens are included in prefill.

        Args:
            text: Input text string
            language: Language code (e.g., "japanese")
            speaker_emb_torch: Optional [1, dim] CPU torch tensor from speaker encoder.
                When provided, inserted as a codec-side position between think_eos
                and codec_pad (paired with tts_pad on the text side).

        Returns:
            input_embeds: [1, seq_len, dim] combined text+codec embeddings
            trailing_text_hidden: [1, N, dim] text hidden states for decode steps
            tts_pad_embed: [1, 1, dim] tts_pad embedding (for decode steps beyond text)
        """
        args = self.talker_args

        # --- Tokenize ---
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        token_ids = self.tokenizer.encode(formatted)
        input_ids = torch.tensor([token_ids], dtype=torch.long)  # [1, total_len]
        # Structure: [im_start, assistant, \n, ...text..., im_end, \n, im_start, assistant, \n]
        # First 3 = role prefix, last 5 = suffix

        # --- Text embeddings (host) ---
        text_embed_all = self.talker.embed_text_tokens(input_ids)  # [1, total_len, text_hidden_size]

        # --- Special TTS token embeddings via text_projection (on device, then back to host) ---
        special_ids = torch.tensor(
            [[args.tts_bos_token_id, args.tts_eos_token_id, args.tts_pad_token_id]],
            dtype=torch.long,
        )
        special_embed = self.talker.embed_text_tokens(special_ids)  # [1, 3, text_hidden_size]
        # Project to Talker dim on device
        tt_special = ttnn.from_torch(
            special_embed.unsqueeze(1),  # [1, 1, 3, text_hidden_size]
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tt_special_proj = self.talker.text_projection(tt_special)
        special_proj = ttnn.to_torch(
            tt_special_proj,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1),
        )[0, 0, :, : args.dim]  # [3, dim]
        tts_bos_embed = special_proj[0:1].unsqueeze(0)  # [1, 1, dim]
        tts_eos_embed = special_proj[1:2].unsqueeze(0)  # [1, 1, dim]
        tts_pad_embed = special_proj[2:3].unsqueeze(0)  # [1, 1, dim]

        # --- Codec embeddings (host) ---
        codec_embed_weight = self.talker.codec_embed_weight  # [vocab, dim]

        def codec_embed(token_ids_list):
            ids = torch.tensor([token_ids_list], dtype=torch.long)
            return torch.nn.functional.embedding(ids, codec_embed_weight)  # [1, N, dim]

        # Language ID
        lang_map = args.codec_language_id
        if language.lower() in lang_map:
            language_id = lang_map[language.lower()]
        else:
            raise ValueError(f"Language '{language}' not in codec_language_id: {list(lang_map.keys())}")

        # Codec tag: [think_id, think_bos_id, language_id, think_eos_id]
        codec_tag = codec_embed([args.codec_think_id, args.codec_think_bos_id, language_id, args.codec_think_eos_id])
        # Codec suffix: [codec_pad_id, codec_bos_id]
        codec_suffix = codec_embed([args.codec_pad_id, args.codec_bos_id])
        # Full codec prefill: [think, think_bos, lang, think_eos, (speaker?), pad, bos]
        if speaker_emb_torch is not None:
            spk = speaker_emb_torch.float().view(1, 1, -1)  # [1, 1, dim]
            codec_prefill = torch.cat([codec_tag, spk, codec_suffix], dim=1)  # [1, 7, dim]
        else:
            codec_prefill = torch.cat([codec_tag, codec_suffix], dim=1)  # [1, 6, dim]

        # --- Role prefix: first 3 tokens through text_projection ---
        role_ids = input_ids[:, :3]
        role_embed = self.talker.embed_text_tokens(role_ids)  # [1, 3, text_hidden_size]
        tt_role = ttnn.from_torch(
            role_embed.unsqueeze(1),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tt_role_proj = self.talker.text_projection(tt_role)
        role_proj = ttnn.to_torch(
            tt_role_proj,
            mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1),
        )[0, 0, :, : args.dim].unsqueeze(0)  # [1, 3, dim]

        # --- Build combined embedding sequence ---
        # Part 1: Role prefix (text_projection only, no codec overlay)
        # HF code: _talker_input_embed_role = text_projection(text_embed(input_id[:, :3]))
        part_role = role_proj  # [1, 3, dim]

        # Part 2: Codec tag with tts_pad/tts_bos overlay (text_side + codec_side)
        # codec_prefill[:-1] = all entries except codec_bos (last)
        # text_side: tts_pad * (len-2) + tts_bos, paired element-wise with codec_prefill[:-1]
        # Without speaker: 5 positions, with speaker: 6 positions (extra tts_pad+speaker_emb)
        num_codec_prefill_m1 = codec_prefill.shape[1] - 1
        text_side_tag = torch.cat([
            tts_pad_embed.expand(-1, num_codec_prefill_m1 - 1, -1),
            tts_bos_embed,
        ], dim=1)
        part_tag = text_side_tag + codec_prefill[:, :-1, :]

        # Part 3: Non-streaming mode — all text content tokens + tts_eos added with codec_pad
        # Text content = input_ids[:, 3:-5] (skip role prefix and suffix)
        text_content_ids = input_ids[:, 3:-5]  # [1, N_text]
        N_text = text_content_ids.shape[1]

        if N_text > 0:
            text_content_embed = self.talker.embed_text_tokens(text_content_ids)  # [1, N_text, text_hidden_size]
            # Project text content on device
            # Pad to multiple of 32 for tile layout
            pad_to = math.ceil((N_text + 1) / 32) * 32  # +1 for tts_eos
            text_with_eos_raw = torch.cat([text_content_embed, torch.zeros(1, 1, text_content_embed.shape[-1])], dim=1)
            if text_with_eos_raw.shape[1] < pad_to:
                text_with_eos_raw = torch.nn.functional.pad(
                    text_with_eos_raw, (0, 0, 0, pad_to - text_with_eos_raw.shape[1])
                )
            tt_text = ttnn.from_torch(
                text_with_eos_raw.unsqueeze(1),
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            tt_text_proj = self.talker.text_projection(tt_text)
            text_proj_all = ttnn.to_torch(
                tt_text_proj,
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=-1),
            )[0, 0, : N_text + 1, : args.dim].unsqueeze(0)  # [1, N_text+1, dim]

            # Replace the last entry (zeros projected) with tts_eos
            text_proj_all[:, -1:, :] = tts_eos_embed

            # Codec side: codec_pad for each text position + tts_eos position
            codec_pad_embed = codec_embed([args.codec_pad_id])  # [1, 1, dim]
            codec_pad_expanded = codec_pad_embed.expand(-1, N_text + 1, -1)  # [1, N_text+1, dim]

            part_text = text_proj_all + codec_pad_expanded  # [1, N_text+1, dim]
        else:
            part_text = torch.zeros(1, 0, args.dim)

        # Part 4: Final position — tts_pad + codec_bos
        part_final = tts_pad_embed + codec_prefill[:, -1:, :]  # [1, 1, dim] — codec_bos

        # Concatenate all parts
        input_embeds = torch.cat([part_role, part_tag, part_text, part_final], dim=1)  # [1, total, dim]

        # Pad to multiple of 128 for ttnn attention alignment
        total_len = input_embeds.shape[1]
        padded_len = math.ceil(total_len / 128) * 128
        if total_len < padded_len:
            input_embeds = torch.nn.functional.pad(input_embeds, (0, 0, 0, padded_len - total_len))

        # --- Trailing text hidden (for decode steps) ---
        # In non-streaming mode: trailing_text_hidden = tts_pad_embed (single token)
        trailing_text_hidden = tts_pad_embed  # [1, 1, dim]

        return input_embeds, trailing_text_hidden, tts_pad_embed

    def _encode_speaker(self, audio: np.ndarray, sr: int):
        """Extract speaker embedding from reference audio."""
        if self.speaker_encoder is None:
            raise RuntimeError("Speaker encoder not initialized — cannot do voice cloning")
        return self.speaker_encoder.encode(audio, sr=sr)

    def _generate_and_predict(
        self,
        input_embeds,
        trailing_text_hidden,
        tts_pad_embed,
        speaker_emb,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty=1.05,
    ):
        """Autoregressive generation: Talker (CB0) + Code Predictor (CB1-15) per step.

        HF reference pattern (modeling_qwen3_tts.py:1664-1692):
            Each decode step:
              1. Talker forward → CB0 logits → sample CB0 token
              2. Code Predictor(past_hidden, CB0_embed) → CB1-15 tokens
              3. Sum all 16 codec embeddings + trailing_text_hidden → next Talker input

        Returns:
            all_codebooks: [B, num_frames, 16] all codebook tokens
        """
        B = input_embeds.shape[0]
        assert B == 1, "Only batch_size=1 supported"
        args = self.talker_args
        codec_embed_weight = self.talker.codec_embed_weight  # [3072, 2048] on host

        # --- Prefill ---
        norms = input_embeds.squeeze(0).norm(dim=-1)
        nonzero_mask = norms > 0
        last_token_idx = nonzero_mask.nonzero()[-1].item() if nonzero_mask.any() else input_embeds.shape[1] - 1

        tokens_embd, rot_mats, rot_mats_local, tt_page_table, tt_chunk_page_table = (
            self.talker.prepare_inputs_prefill(
                input_embeds,
                start_pos=0,
                last_token_idx=last_token_idx,
            )
        )

        get_last_token = (last_token_idx // 32) * 32
        logits_tt, prefill_hidden_tt = self.talker.ttnn_prefill_forward_with_hidden(
            tokens_embd,
            rot_mats_global=rot_mats,
            rot_mats_local=rot_mats_local,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            get_last_token=get_last_token,
            speaker_emb=speaker_emb,
            pre_projected=True,
        )

        logits = self.talker.process_output_prefill(logits_tt.cpu(), last_token_idx=last_token_idx % 32)
        logits = logits.view(1, 1, args.vocab_size)

        # Extract pre-norm hidden state at last token position for Code Predictor
        prefill_hidden_torch = ttnn.to_torch(prefill_hidden_tt)
        last_in_block = last_token_idx % 32
        talker_hidden_torch = prefill_hidden_torch[:, :, last_in_block:last_in_block + 1, :args.dim]
        talker_hidden_torch = talker_hidden_torch.permute(0, 2, 1, 3).reshape(B, 1, args.dim)

        generated_tokens = []
        cb0_token = self._sample_token(logits, temperature, top_k, top_p)
        generated_tokens.append(cb0_token.item())
        logger.info(f"Prefill done (last_tok_idx={last_token_idx}), first CB0: {cb0_token.item()}")

        # --- Decode loop ---
        prefill_len = last_token_idx + 1
        all_frames = []  # list of [B, 16] per frame

        for step in range(max_new_tokens):
            if cb0_token.item() == args.codec_eos_token_id:
                logger.info(f"EOS at step {step}")
                break

            # Run Code Predictor: generate CB1-15 for this frame
            frame_all_cb = self.code_predictor.predict_codebooks(
                talker_hidden_torch, cb0_token, codec_embed_weight
            )  # [B, 16] — CB0 + CB1..CB15

            all_frames.append(frame_all_cb.unsqueeze(1))  # [B, 1, 16]

            # Build next Talker input: sum of all 16 codec embeddings + trailing_text_hidden
            cb0_emb = torch.nn.functional.embedding(
                cb0_token.unsqueeze(-1), codec_embed_weight
            )  # [B, 1, 2048]

            cb_emb_sum = cb0_emb
            for cb_idx in range(args.num_code_groups - 1):
                cb_token = frame_all_cb[:, cb_idx + 1]  # [B]
                cb_emb = torch.nn.functional.embedding(
                    cb_token.unsqueeze(-1),
                    self.code_predictor.codec_embeddings[cb_idx],
                )  # [B, 1, 2048]
                cb_emb_sum = cb_emb_sum + cb_emb

            decode_input = cb_emb_sum + trailing_text_hidden  # [B, 1, 2048]

            # Prepare decode inputs (position, rotation)
            current_pos = torch.tensor([prefill_len + step], dtype=torch.int64)
            padded_pos = torch.nn.functional.pad(
                current_pos, (0, args.max_batch_size - 1), value=0
            )

            dummy_tokens = torch.zeros(1, args.max_batch_size, dtype=torch.long)
            _, tt_pos, tt_rot_idxs, tt_page_table = self.talker.prepare_inputs_decode(
                dummy_tokens, padded_pos
            )

            # Send pre-embedded decode input to device [1, 1, 32, dim]
            # Pad batch dimension to 32 for tile alignment (matching normal decode path)
            decode_padded = torch.zeros(1, 1, 32, args.dim)
            decode_padded[0, 0, 0, :] = decode_input[0, 0, :]
            tt_decode_embed = ttnn.from_torch(
                decode_padded,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

            decode_residual_mem_cfg = self.talker.args.get_residual_mem_config(
                Mode.DECODE, self.talker.prefetcher
            )
            tt_decode_embed = ttnn.to_memory_config(tt_decode_embed, decode_residual_mem_cfg)

            # Decode forward — returns both logits and post-norm hidden state
            tt_logits, tt_hidden = self.talker.ttnn_decode_forward_preembedded(
                tt_decode_embed, tt_pos, rot_mat_idxs=tt_rot_idxs, page_table=tt_page_table
            )

            # Extract Talker hidden state for next Code Predictor call
            hidden_torch = ttnn.to_torch(tt_hidden)
            talker_hidden_torch = hidden_torch[:, :, :B, :args.dim].permute(0, 2, 1, 3).reshape(B, 1, args.dim)

            logits = self.talker.process_output_decode(tt_logits.cpu(), B=1)
            logits = logits[:, :, : args.vocab_size]

            cb0_token = self._sample_token(
                logits.view(1, 1, -1), temperature, top_k, top_p,
                generated_tokens=generated_tokens, repetition_penalty=repetition_penalty,
            )
            generated_tokens.append(cb0_token.item())

        if all_frames:
            all_codebooks = torch.cat(all_frames, dim=1)  # [B, num_frames, 16]
        else:
            all_codebooks = torch.zeros(B, 0, 16, dtype=torch.long)

        logger.info(
            f"Generated {all_codebooks.shape[1]} frames, "
            f"CB0 range: [{all_codebooks[:,:,0].min().item()}, {all_codebooks[:,:,0].max().item()}]"
        )

        return all_codebooks

    @staticmethod
    def _sample_token(logits, temperature, top_k, top_p, generated_tokens=None, repetition_penalty=1.0):
        """Sample a single token from logits [B, 1, vocab_size]."""
        logits = logits[:, -1, :].clone()  # [B, vocab_size]

        if repetition_penalty != 1.0 and generated_tokens is not None and len(generated_tokens) > 0:
            prev = torch.tensor(generated_tokens, dtype=torch.long)
            score = logits[0, prev]
            score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
            logits[0, prev] = score

        if temperature <= 0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            next_token = sorted_indices.gather(-1, torch.multinomial(sorted_probs, num_samples=1))
            return next_token.squeeze(-1)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def _predict_codebooks(self, cb0_tokens, talker_hidden):
        """Generate CB1-CB15 via Code Predictor for each frame.

        Args:
            cb0_tokens: [B, num_frames] CB0 token IDs
            talker_hidden: [B, num_frames, 2048] Talker hidden states per frame
                          (None if hidden states not captured)

        Returns:
            all_codebooks: [B, num_frames, 16] all codebook tokens
        """
        if self.code_predictor is None:
            raise NotImplementedError("Code Predictor not built")

        B, num_frames = cb0_tokens.shape
        all_codebooks = []

        for frame_idx in range(num_frames):
            if talker_hidden is not None:
                frame_hidden = talker_hidden[:, frame_idx : frame_idx + 1, :]  # [B, 1, 2048]
            else:
                frame_hidden = torch.zeros(B, 1, self.talker_args.dim)

            frame_cb0 = cb0_tokens[:, frame_idx]  # [B]

            frame_all_cb = self.code_predictor.predict_codebooks(
                frame_hidden,
                frame_cb0,
                self.talker.codec_embed_weight,
            )
            all_codebooks.append(frame_all_cb.unsqueeze(1))  # [B, 1, 16]

        return torch.cat(all_codebooks, dim=1)  # [B, num_frames, 16]

    def _decode_waveform(self, all_codebooks):
        """Convert codebook tokens to waveform via Vocoder.

        Args:
            all_codebooks: [B, num_frames, 16] all codebook token IDs

        Returns:
            waveform: numpy array [num_samples] at 24kHz
        """
        if self.vocoder is None:
            raise RuntimeError("Vocoder not initialized — download speech_tokenizer weights")
        # Clamp token values to Vocoder's codebook range [0, 2047]
        max_cb_val = all_codebooks.max().item()
        if max_cb_val >= 2048:
            logger.warning(f"Clamping {(all_codebooks >= 2048).sum().item()} tokens from max={max_cb_val} to 2047")
            all_codebooks = all_codebooks.clamp(max=2047)
        return self.vocoder.decode(all_codebooks)
