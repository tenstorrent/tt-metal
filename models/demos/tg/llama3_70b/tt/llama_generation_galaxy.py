# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import copy

import torch
from loguru import logger

import ttnn
from models.demos.t3000.llama2_70b.tt.llama_common import BASE_URL, ConcatMesh2DToTensor
from models.demos.tg.llama3_70b.tt.llama_common import upper_pad_sequence_length
from models.demos.tg.llama3_70b.tt.llama_model_galaxy import TtLlamaModel_galaxy as TtLlamaModel
from models.demos.tg.llama3_70b.tt.model_config import get_model_config


class TtLlamaModelForGeneration:
    def __init__(self, configuration, state_dict, model_args, tt_args):
        # Cache Weights setup
        n_layers = model_args.num_layers or 80

        self.params = copy.deepcopy(configuration)

        self.llama_version = model_args.llama_version
        self.max_batch_size = model_args.max_batch_size
        self.max_kv_context_len = model_args.max_kv_context_len

        self.mesh_device = tt_args.mesh_device
        self.cluster_shape = tt_args.cluster_shape

        # Initial model_config is set in decode mode
        model_config = get_model_config(
            llama_version=self.llama_version,
            max_batch_size=self.max_batch_size,
            max_context_len=self.max_kv_context_len,
        )

        # TT model -------------------------------------------------------------
        self.tt_model = TtLlamaModel(
            self.mesh_device,
            self.cluster_shape,
            state_dict,
            BASE_URL,
            n_layers,
            model_config,
            self.params,
            cache_path=tt_args.cache_path,
            read_cache=False,
        )

        del state_dict

    def forward(self, tokens: torch.Tensor, start_pos: int):
        _, seq_len = tokens.shape
        if seq_len == 1:
            return self.decode_forward(tokens, start_pos)
        else:
            return self.prefill_forward(tokens, start_pos)

    def decode_forward(self, tokens: torch.Tensor, start_pos: int):
        batch = tokens.shape[0]
        tt_inp_emb, start_pos, rot_mat, attn_mask = self.tt_model.prepare_inputs(tokens, start_pos, mode="decode")

        tt_logits = self.tt_model(tt_inp_emb, rot_mat, start_pos, attn_mask, mode="decode")

        del tt_inp_emb
        del rot_mat
        del attn_mask

        logits = self._process_logits(tt_logits)

        logits = logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        logits = logits[:batch]  # Remove padded users
        del tt_logits

        return logits

    def prefill_forward_single_user(
        self, tokens: torch.Tensor, start_pos: int, user_id: int, last_token_idx=None, page_table=None
    ):
        batch, seq_len = tokens.shape
        assert batch == 1
        assert start_pos == 0, "start_pos must be 0 for prefill_forward_single_user"
        assert seq_len % 32 == 0, f"seq_len must be divisible by 32, got {seq_len}"
        tt_inp_emb, start_pos, rot_mat, attn_mask = self.tt_model.prepare_inputs(
            tokens,
            start_pos=start_pos,
            valid_seq_len=seq_len,
            mode="prefill",
        )

        tt_logits = self.tt_model(tt_inp_emb, rot_mat, start_pos, attn_mask, user_id=user_id, mode="prefill")

        del tt_inp_emb
        del rot_mat
        del attn_mask

        logits = self._process_logits(tt_logits)
        logits = logits.squeeze(1)
        del tt_logits
        return logits

    def prefill_forward(self, tokens: torch.Tensor, start_pos: int):
        batch, seq_len = tokens.shape
        assert seq_len <= 8 * 1024, f"Only prefill up to 2048 tokens is supported, got {seq_len}"
        prefill_seq_len = upper_pad_sequence_length(
            seq_len, self.tt_model.model_config["PADDING_LENGTH"]
        )  # Pad seq_len to nearest_32 multiple

        batch, seq_len = tokens.shape
        last_token_idx = seq_len - 1
        output_logits = torch.zeros(batch, seq_len, self.params.vocab_size)
        # pad tokens to nearest 32 multiple
        prefill_ids = torch.cat([tokens, torch.zeros(batch, prefill_seq_len - seq_len).long()], dim=-1)

        for user_id in range(batch):
            logger.info(f"Filling kv cache for user {user_id + 1}")

            logits = self.prefill_forward_single_user(prefill_ids[user_id : user_id + 1], start_pos, user_id)

            # Since we give padded_seq_len, we get only the last token
            output_logits[user_id] = logits[:, last_token_idx % 32 : last_token_idx % 32 + 1, :]
        logger.info(f"Finished prefill for all users up to {seq_len} tokens, Starting decode...")

        return output_logits

    def _process_logits(self, tt_logits):
        logits = ttnn.to_torch(
            tt_logits,
            mesh_composer=ConcatMesh2DToTensor(self.mesh_device, dims=(1, 3), cluster_shape=self.cluster_shape),
        )
        return logits[:, 0:1, :, : self.params.vocab_size].float()
