# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CosyVoice 3 LLM (Qwen2 backbone) for speech token generation using TTNN APIs.
"""

from typing import Generator

import torch
from loguru import logger

import ttnn
from models.demos.wormhole.cosy_voice.tt.cosyvoice_transformer import CosyVoiceTransformer
from models.demos.wormhole.cosy_voice.tt.model_config import CosyVoiceModelConfig
from models.tt_transformers.tt.common import Mode


class CosyVoice3LM:
    """
    Top-level wrapper for the CosyVoice LLM module.
    Handles token/embedding composition, prefill, and the autoregressive decoding loop.
    """

    def __init__(self, config: CosyVoiceModelConfig, mesh_device, dtype=ttnn.bfloat8_b):
        self.config = config
        self.mesh_device = mesh_device
        self.dtype = dtype

        # Special tokens
        self.sos = config.sos_token
        self.eos = config.eos_token
        self.task_id = config.task_id_token
        self.fill = config.fill_token

        self.stop_token_ids = [config.speech_token_size + i for i in range(200)]

        # Initialize the underlying transformer
        logger.info("Initializing CosyVoiceTransformer...")
        self.model = CosyVoiceTransformer.from_pretrained(config, mesh_device, dtype)

        # Keep a reference to the text embedding (on host for easy prompt composition)
        self.llm_embedding_torch = self.model.text_embedding_torch.float()
        self.speech_embedding_torch = self.model.speech_embedding.weight_torch.float()

    def sampling_ids(self, logp, out_tokens, sampling_kwargs, ignore_eos=False):
        """
        Sample the next token from logits.
        """
        top_k = sampling_kwargs.get("top_k", 25)

        # Apply mask
        mask = torch.zeros_like(logp, dtype=torch.bool)
        if ignore_eos:
            mask[self.eos] = True

        logp = logp.clone()
        logp[mask] = -float("inf")

        # Top-K sampling
        val, idx = logp.topk(top_k, dim=-1)

        # For simplicity in Stage 1, we can just do greedy if top_k <= 1, else multinomial
        if top_k <= 1:
            return idx[0].item()

        probs = torch.softmax(val, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)
        return idx[next_idx.item()].item()

    def format_prompt_embeddings(self, prompt_text: torch.Tensor, prompt_speech_token: torch.Tensor) -> torch.Tensor:
        """
        Compose the initial prompt embeddings on the host.
        """
        # Embed <sos> and <task_id>
        sos_emb = self.speech_embedding_torch[self.sos].reshape(1, 1, -1)
        task_id_emb = self.speech_embedding_torch[self.task_id].reshape(1, 1, -1)

        # Embed speech prompt
        if prompt_speech_token.shape[1] > 0:
            speech_emb = self.speech_embedding_torch[prompt_speech_token.flatten()].unsqueeze(0)
        else:
            speech_emb = torch.zeros((1, 0, self.config.dim), dtype=torch.float32)

        # Embed text prompt
        if prompt_text.shape[1] > 0:
            text_emb = self.llm_embedding_torch[prompt_text.flatten()].unsqueeze(0)
        else:
            text_emb = torch.zeros((1, 0, self.config.dim), dtype=torch.float32)

        # Concatenate: <sos> + text_emb + task_id_emb + speech_emb
        lm_input = torch.cat([sos_emb, text_emb, task_id_emb, speech_emb], dim=1)
        return lm_input

    def inference(
        self,
        text: torch.Tensor,
        text_len: torch.Tensor,
        prompt_text: torch.Tensor,
        prompt_text_len: torch.Tensor,
        prompt_speech_token: torch.Tensor,
        prompt_speech_token_len: torch.Tensor,
        embedding: torch.Tensor,
        sampling: int = 25,
        max_token_text_ratio: float = 20,
        min_token_text_ratio: float = 2,
    ) -> Generator[int, None, None]:
        """
        Run the full generation pipeline:
        1. Format prompt
        2. Prefill
        3. Autoregressive decode
        """
        # 1. Format prompt embeddings
        # For CosyVoice3, the full prompt_text contains both the condition text and target text
        # separated by <|endofprompt|> (151646)

        lm_input = self.format_prompt_embeddings(prompt_text, prompt_speech_token)
        seq_len = lm_input.shape[1]

        # Length constraints
        # For simplicity, using simple min/max length estimation
        max_len = int(text_len.max().item() * max_token_text_ratio)
        min_len = int(text_len.min().item() * min_token_text_ratio)

        logger.info(f"Starting inference with prompt length {seq_len}, max_len {max_len}")

        # 2. Prefill
        device = self.mesh_device
        batch_size = 1

        # Initialize KV Cache
        self.model.args.max_batch_size = batch_size
        kv_cache = self.model.setup_kv_cache(batch_size, self.config.max_seq_len)

        # Prepare prefill inputs (returns ttnn tensors)
        (
            tt_tokens_embd,
            tt_rot_mats_prefill_global,
            tt_rot_mats_prefill_local,
            tt_page_table,
            tt_chunk_page_table,
        ) = self.model.prepare_inputs_prefill_embeddings(
            lm_input,
            start_pos=0,
            batch_size=batch_size,
        )

        logger.info(f"lm_input shape: {lm_input.shape}")
        logger.info(f"tt_tokens_embd shape: {tt_tokens_embd.shape}, padded: {tt_tokens_embd.padded_shape}")

        # Run prefill forward
        tt_hidden_states = self.model.ttnn_prefill_forward(
            tt_tokens_embd,
            rot_mats_global=tt_rot_mats_prefill_global,
            rot_mats_local=tt_rot_mats_prefill_local,
            user_id=0,
            page_table=tt_page_table,
            chunk_page_table=tt_chunk_page_table,
            get_last_token=seq_len - 1,  # We want the logit for the last token
            kv_cache=kv_cache,
            batch_size=batch_size,
        )

        # Read prefill output logit — output is replicated across devices
        # since the speech decoder head weight is replicated (not sharded).
        # Extract the first device's tensor to avoid the multi-device compose error.
        if self.config.num_devices > 1:
            hidden_states_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_hidden_states)[0]).float()
        else:
            hidden_states_torch = ttnn.to_torch(tt_hidden_states).float()
        # Apply output norm & head since process_logits_after_prefill_trace handles batched case
        # For simplicity, our CosyVoiceTransformer already applied norm and head in forward!
        # Because we didn't use `process_hidden_states_after_prefill_trace`.
        # Wait, tt_transformers forward applies norm and lm_head!

        # hidden_states_torch shape is [1, 1, seq_len+padding, vocab_size]
        logits = hidden_states_torch[0, 0, 0, :]
        logp = torch.log_softmax(logits, dim=-1)

        sampling_kwargs = {"top_k": sampling}

        out_tokens = []
        next_token = self.sampling_ids(logp, out_tokens, sampling_kwargs, ignore_eos=True)
        yield next_token
        out_tokens.append(next_token)

        current_pos = seq_len

        # 3. Decode loop
        for i in range(1, max_len):
            # Prepare decode inputs: returns (tokens, current_pos_tt, rope_idxs, page_table)
            token_tensor = torch.tensor([next_token], dtype=torch.int32)
            tt_inputs = self.model.prepare_inputs_decode(token_tensor, torch.tensor([current_pos]))
            (tt_tokens, tt_current_pos, tt_rope_idxs, tt_page_table) = tt_inputs

            # Compute rotation matrices from rope indices (same as ttnn_decode_forward)
            tt_rot_mats_global = self.model.rope_setup.get_rot_mats(tt_rope_idxs)
            tt_rot_mats_local = (
                self.model.rope_local_setup.get_rot_mats(tt_rope_idxs)
                if hasattr(self.model, "rope_local_setup")
                else None
            )

            # Embed the speech token on host and prepare for decode.
            # Decode mode requires height=32 (tile-padded batch) and WIDTH sharding
            # matching the decode residual memory config used by the transformer layers.
            token_emb = self.speech_embedding_torch[next_token].reshape(1, -1)  # [1, dim]
            # Pad to batch=32 as required by decode attention kernels
            token_emb = torch.nn.functional.pad(token_emb, (0, 0, 0, 31))  # [32, dim]
            token_emb = token_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, 32, dim]

            if self.config.num_devices > 1:
                mesh_mapper = ttnn.ShardTensor2dMesh(
                    mesh_device=device, dims=(None, 3), mesh_shape=self.config.cluster_shape
                )
            else:
                mesh_mapper = ttnn.ReplicateTensorToMesh(device)

            tt_tokens_embd = ttnn.from_torch(
                token_emb,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=mesh_mapper,
            )
            # Move to the decode residual memory config (WIDTH sharded, height=32)
            decode_residual_mem_cfg = self.config.get_residual_mem_config(Mode.DECODE)
            tt_tokens_embd = ttnn.to_memory_config(tt_tokens_embd, decode_residual_mem_cfg)

            tt_logits = self.model.forward(
                tt_tokens_embd,
                current_pos=tt_current_pos,
                rot_mats_global=tt_rot_mats_global,
                rot_mats_local=tt_rot_mats_local,
                mode=Mode.DECODE,
                page_table=tt_page_table,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )

            if self.config.num_devices > 1:
                logits_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()
            else:
                logits_torch = ttnn.to_torch(tt_logits).float()
            logits = logits_torch[0, 0, 0, :]
            logp = torch.log_softmax(logits, dim=-1)

            next_token = self.sampling_ids(logp, out_tokens, sampling_kwargs, ignore_eos=(i < min_len))

            if next_token in self.stop_token_ids:
                break

            yield next_token
            out_tokens.append(next_token)
            current_pos += 1

        # Teardown KV Cache
        for cache in kv_cache:
            for tensor in cache:
                if tensor is not None:
                    ttnn.deallocate(tensor)
