# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from loguru import logger

import copy
from models.utility_functions import torch2tt_tensor, pad_by_zero, tt2torch_tensor, nearest_32
from models.demos.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized as TtLlamaModel
from models.demos.llama2_70b.tt.llama_common import BASE_URL
from models.demos.llama2_70b.tt.model_config import (
    get_model_config,
)


class TtLlamaModelForGeneration:
    def __init__(self, reference_model, devices, n_devices, n_layers, batch, emulated=False, cache_path=None):
        ## Get state dict
        state_dict = reference_model.state_dict()
        configuration = copy.deepcopy(reference_model.params)

        # Cache Weights setup
        if n_layers == None:
            n_layers = 80

        if n_layers >= 40 and n_devices == 4:
            n_layers_per_group = 20
            assert n_layers % n_layers_per_group == 0
        else:
            n_layers_per_group = None

        model_config = get_model_config(model_config_str="BFLOAT16-DRAM", num_devices=n_devices)

        # TT model -------------------------------------------------------------
        self.tt_model = TtLlamaModel(
            devices,
            state_dict,
            BASE_URL,
            n_layers,
            model_config,
            configuration,
            batch,
            n_layers_per_group=n_layers_per_group,
            emulated=emulated,
            cache_path=cache_path,
        )
        self.params = configuration
        self.devices = devices
        self.n_devices = n_devices

        for device in devices:
            tt_lib.device.Synchronize(device)

        del reference_model
        del state_dict

    def forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        # First, determine whether this is decode or prefill based on shape of the input
        assert len(tokens.shape) == 2
        dim1, dim2 = tokens.shape
        if dim2 == 1:
            # Decode
            # if current model config is not for decode, change it to decode
            if self.tt_model.model_config["LLM_MODE"] != "decode":
                logger.info("Changing mode to decode")
                model_config = get_model_config(
                    model_config_str="BFLOAT16-DRAM", num_devices=self.n_devices, seq_len=dim2
                )
                self.tt_model.set_model_config(model_config)
            return self.decode_forward(tokens, start_pos, *args, **kwargs)
        else:
            # Prefill
            # if current model config is not for prefill, change it to prefill
            if self.tt_model.model_config["LLM_MODE"] != "prefill":
                logger.info("Changing mode to prefill")
                prefill_seq_len = 128 if dim2 <= 128 else 2048
                assert dim2 <= 2048, f"Only prefill up to 2048 tokens is supported, got {dim2}"
                model_config = get_model_config(
                    model_config_str="BFLOAT16-DRAM", num_devices=self.n_devices, seq_len=prefill_seq_len
                )
                self.tt_model.set_model_config(model_config)
            return self.prefill_forward(tokens, start_pos, *args, **kwargs)

    def decode_forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        tt_inp_emb, start_pos, rot_mat, attn_mask = self.tt_model.prepare_inputs(tokens, start_pos)

        tt_out = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            attn_mask,
        )

        del tt_inp_emb
        del rot_mat
        del attn_mask

        for device in self.devices:
            tt_lib.device.Synchronize(device)

        assert isinstance(tt_out, list)  # tt_out should be fractured on N devices
        assert len(tt_out) == len(self.devices)

        tt_outs = [tt2torch_tensor(o) for o in tt_out]
        tt_out = torch.cat(tt_outs, dim=-1)
        tt_out = tt_out[..., : self.params.vocab_size]
        tt_out = tt_out.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        tt_out = tt_out.float()
        return tt_out

    def prefill_forward(self, tokens: torch.Tensor, start_pos: int, *args, **kwargs):
        batch_size, seq_len = tokens.shape
        output_logits = torch.zeros(batch_size, seq_len, self.params.vocab_size)
        padded_seq_len = 128 if seq_len <= 128 else 2048
        # pad tokens to 128 or 2048
        prefill_ids = torch.cat([tokens, torch.zeros(batch_size, padded_seq_len - seq_len).long()], dim=-1)

        for user_id in range(batch_size):
            logger.info(f"Filling kv cache for user {user_id + 1}")

            tt_inp_emb, start_pos, rot_mat, attn_mask = self.tt_model.prepare_inputs(
                prefill_ids[user_id : user_id + 1], start_pos=0
            )

            # TODO for @KevinMi: Attention mask should mask out the padding tokens

            tt_logits = self.tt_model(
                tt_inp_emb,
                rot_mat,
                start_pos,
                attn_mask,
                user_id=user_id,
            )

            del tt_inp_emb
            del rot_mat
            del attn_mask

            logits = torch.cat([tt2torch_tensor(tt_o).squeeze(1) for tt_o in tt_logits], -1)
            logits = logits[..., : self.params.vocab_size].float()
            del tt_logits

            output_logits[user_id] = logits[:, :seq_len, :]

        return output_logits
