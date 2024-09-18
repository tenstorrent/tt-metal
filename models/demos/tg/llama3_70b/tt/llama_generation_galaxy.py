# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger

import copy
from models.demos.tg.llama3_70b.tt.llama_model_galaxy import TtLlamaModel_galaxy as TtLlamaModel
from models.demos.t3000.llama2_70b.tt.llama_common import BASE_URL, ConcatMesh2DToTensor
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
        self.cluster_shape = tt_args.cluster_shape

        # Initial model_config is set in decode mode
        model_config = get_model_config(
            llama_version=self.llama_version,
            max_batch_size=self.max_batch_size,
            max_context_len=self.max_kv_context_len,
            seq_len=1,
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

    def _process_logits(self, tt_logits):
        logits = ttnn.to_torch(
            tt_logits,
            mesh_composer=ConcatMesh2DToTensor(self.mesh_device, dims=(1, 3), cluster_shape=self.cluster_shape),
        )
        return logits[:, 0:1, :, : self.params.vocab_size].float()
