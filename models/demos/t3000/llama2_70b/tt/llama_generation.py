# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn import ConcatMeshToTensor

from loguru import logger

import copy
from models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized as TtLlamaModel
from models.demos.t3000.llama2_70b.tt.llama_common import BASE_URL
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)


class TtLlamaModelForGeneration:
    def __init__(self, configuration, state_dict, model_args, tt_args):
        # Cache Weights setup
        n_layers = model_args.num_layers or 80

        self.params = copy.deepcopy(configuration)

        self.llama_version = model_args.llama_version
        self.max_batch_size = model_args.max_batch_size
        self.max_kv_context_len = model_args.max_kv_context_len

        self.mesh_device = tt_args.mesh_device

        # Initial model_config is set in decode mode
        model_config = get_model_config(
            llama_version=self.llama_version,
            max_batch_size=self.max_batch_size,
            max_context_len=self.max_kv_context_len,
            batch=self.max_batch_size,
            seq_len=1,
        )

        # TT model -------------------------------------------------------------
        self.tt_model = TtLlamaModel(
            self.mesh_device,
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

    def decode_forward(self, tokens: torch.Tensor, start_pos: int, trace_capture=False):
        self._update_model_config("decode", tokens.shape[0], 1)
        batch = tokens.shape[0]
        tt_inp_emb, start_pos, rot_mat, attn_mask, cache_idxs_tt = self.tt_model.prepare_inputs(tokens, start_pos)

        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
            cache_idxs=cache_idxs_tt,
        )

        # del tt_inp_emb
        # del rot_mat
        # del attn_mask

        logits = self._process_logits(tt_logits)

        logits = logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        logits = logits[:batch]  # Remove padded users
        # del tt_logits

        return logits

    def prefill_forward_single_user(self, tokens: torch.Tensor, start_pos: int, user_id: int, unpadded_seq_len=None):
        batch, seq_len = tokens.shape
        assert batch == 1
        assert start_pos == 0, "start_pos must be 0 for prefill_forward_single_user"
        assert seq_len in [128, 2048, 8 * 1024], f"Only prefill up to 128 or 2048 tokens is supported, got {seq_len}"

        self._update_model_config("prefill", batch, seq_len)

        tt_inp_emb, start_pos, rot_mat, attn_mask = self.tt_model.prepare_inputs(
            tokens, start_pos=start_pos, valid_seq_len=seq_len
        )

        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
            user_id=user_id,
            unpadded_seq_len=unpadded_seq_len,
        )

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

        prefill_seq_len = 128 if seq_len <= 128 else 2048 if seq_len <= 2048 else 8 * 1024
        self._update_model_config("prefill", batch, prefill_seq_len)

        batch, seq_len = tokens.shape
        output_logits = torch.zeros(batch, seq_len, self.params.vocab_size)
        # pad tokens to 128 or 2048
        prefill_ids = torch.cat([tokens, torch.zeros(batch, prefill_seq_len - seq_len).long()], dim=-1)

        for user_id in range(batch):
            logger.info(f"Filling kv cache for user {user_id + 1}")

            logits = self.prefill_forward_single_user(
                prefill_ids[user_id : user_id + 1], start_pos, user_id, unpadded_seq_len=seq_len
            )

            # output_logits[user_id] = logits[:, :seq_len, :]
            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = logits[:, seq_len % 32 : seq_len % 32 + 1, :]

        logger.info(f"Finished prefill for all users up to {seq_len} tokens, Starting decode...")

        return output_logits

    def _process_logits(self, tt_logits):
        logits = ttnn.to_torch(
            tt_logits, device=self.mesh_device, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=3)
        )
        return logits[..., : self.params.vocab_size].float()

    def _update_model_config(self, mode, batch, seq_len):
        if self.tt_model.model_config["LLM_MODE"] != mode:
            logger.info(f"Changing mode to {mode}")
            model_config = get_model_config(
                llama_version=self.llama_version,
                max_batch_size=self.max_batch_size,
                max_context_len=self.max_kv_context_len,
                batch=batch,
                seq_len=seq_len,
            )
            self.tt_model.set_model_config(model_config)
