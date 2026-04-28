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
        self.llm_embedding_torch = self.model.qwen2_state_dict["model.embed_tokens.weight"].float()
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

        # Read prefill output logit
        hidden_states_torch = ttnn.to_torch(tt_hidden_states)
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
            # Prepare decode inputs
            # Embed the generated token
            token_tensor = torch.tensor([next_token], dtype=torch.int32)
            tt_inputs = self.model.prepare_inputs_decode(token_tensor, torch.tensor([current_pos]))
            (tt_tokens, tt_rot_mats_global, tt_rot_mats_local, tt_page_table) = tt_inputs

            # Since tt_tokens are token IDs, but we want to embed with speech embeddings:
            # We can use speech_embedding directly on host, or let tt_transformers do it
            # if we override self.embd. We will embed on host!
            token_emb = self.speech_embedding_torch[next_token].reshape(1, 1, 1, -1)
            tt_tokens_embd = ttnn.from_torch(
                token_emb,
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
            tt_tokens_embd = ttnn.unsqueeze_to_4D(tt_tokens_embd)

            tt_logits = self.model.forward(
                tt_tokens_embd,
                current_pos=torch.tensor([current_pos]),
                rot_mats_global=tt_rot_mats_global,
                rot_mats_local=tt_rot_mats_local,
                mode=Mode.DECODE,
                page_table=tt_page_table,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )

            logits_torch = ttnn.to_torch(tt_logits)
            logits = logits_torch[0, 0, 0, :]
            logp = torch.log_softmax(logits, dim=-1)

            next_token = self.sampling_ids(logp, out_tokens, sampling_kwargs, ignore_eos=(i < min_len))

            if next_token in self.stop_token_ids:
                break

            yield next_token
            out_tokens.append(next_token)
            current_pos += 1

        # Teardown KV Cache
        self.model.teardown_kv_cache(kv_cache)
