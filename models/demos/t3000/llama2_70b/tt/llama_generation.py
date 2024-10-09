# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
import ttnn
from ttnn import ConcatMeshToTensor, ReplicateTensorToMesh

from dataclasses import dataclass
from loguru import logger

import copy
from models.demos.t3000.llama2_70b.tt.llama_model_optimized import TtLlamaModel_optimized as TtLlamaModel
from models.demos.t3000.llama2_70b.tt.llama_common import (
    BASE_URL,
    load_llama_state_dict,
    setup_llama_env,
    check_mesh_device,
)
from models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)


class TtLlamaModelForGeneration:
    def __init__(self, configuration, state_dict, model_args, tt_args, paged_attention_config=None, vllm=False):
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
            vllm=vllm,
        )
        self.model_config = model_config

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
            paged_attention_config=paged_attention_config,
            vllm=vllm,
        )

        del state_dict

    @classmethod
    def initialize_vllm_model(cls, hf_config, t3k_mesh_device):
        # TODO: pass in model args and tt args as parameters from vllm
        @dataclass
        class ModelArgs:
            llama_version: str = None
            ckpt_dir: str = None
            max_batch_size: int = 32
            num_layers: int = 80
            max_kv_context_len: int = 4096

        @dataclass
        class TTArgs:
            mesh_device: object = None
            cache_path: str = None

        # setup configs
        llama_version = "llama3"
        model_config, ckpt_dir, _, cache_path = setup_llama_env(
            llama_version=llama_version,
        )
        check_mesh_device(t3k_mesh_device, model_config)

        # initialize arg classes
        model_args = ModelArgs(llama_version=llama_version, ckpt_dir=ckpt_dir)
        tt_args = TTArgs(mesh_device=t3k_mesh_device, cache_path=cache_path)

        # load state dict
        state_dict = load_llama_state_dict(model_args.ckpt_dir, n_layers=model_args.num_layers)

        # TODO: delete this configuration setup once llama can directly accept hf_config
        from models.demos.t3000.llama2_70b.reference.llama.llama.model import ModelArgs as ReferenceModelArgs
        from pathlib import Path
        import json

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        configuration = ReferenceModelArgs(
            max_seq_len=model_args.max_kv_context_len,
            max_batch_size=model_args.max_batch_size,
            **params,
        )

        return cls(
            configuration=configuration, state_dict=state_dict, model_args=model_args, tt_args=tt_args, vllm=True
        )

    def forward(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None, prompt_lens=None):
        _, seq_len = tokens.shape
        if seq_len == 1:
            return self.decode_forward(tokens, start_pos, page_table=page_table, kv_cache=kv_cache)
        else:
            return self.prefill_forward(
                tokens, start_pos, page_table=page_table, kv_cache=kv_cache, prompt_lens=prompt_lens
            )

    def capture_trace(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None):
        # Get inputs on device
        tt_inp, tt_inp_emb, start_pos, rot_mat, cache_idxs_tt, tt_page_table = self.tt_model.prepare_device_inputs(
            tokens, start_pos, mode="decode", page_table=page_table, return_tokens=True
        )

        # Compile model
        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            cache_idxs=cache_idxs_tt,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            mode="decode",
        )

        # Capture trace
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)

        # Run TT model
        tt_inp_emb = self.tt_model.tt_embd(tt_inp)
        tt_inp_emb = ttnn.interleaved_to_sharded(tt_inp_emb, self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"])
        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            cache_idxs=cache_idxs_tt,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            mode="decode",
        )

        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        logger.info("Done Capturing Decode Trace")

        return trace_id, tt_inp, rot_mat, cache_idxs_tt, tt_logits, tt_page_table

    def delete_trace(self, trace_id):
        ttnn.release_trace(self.mesh_device, trace_id)

    def decode_forward_trace(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        trace_id,
        tt_inp,
        rot_mat,
        cache_idxs_tt,
        tt_logits,
        page_table=None,
        tt_page_table=None,
    ):
        batch = tokens.shape[0]

        # Update preallocated tensors
        (
            updated_tt_inp,
            start_pos,
            updated_rot_mat,
            updated_cache_idxs_tt,
            updated_tt_page_table,
        ) = self.tt_model.prepare_inputs(tokens, start_pos, mode="decode", page_table=page_table)
        ttnn.copy_host_to_device_tensor(updated_tt_inp, tt_inp)
        ttnn.copy_host_to_device_tensor(updated_rot_mat, rot_mat)
        ttnn.copy_host_to_device_tensor(updated_cache_idxs_tt, cache_idxs_tt)
        if page_table is not None:
            ttnn.copy_host_to_device_tensor(updated_tt_page_table, tt_page_table)

        # Run TT model
        ttnn.execute_trace(self.mesh_device, trace_id, cq_id=0, blocking=False)
        updated_tt_logits = ttnn.from_device(tt_logits)

        logits = self._process_logits(updated_tt_logits)

        logits = logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        logits = logits[:batch]  # Remove padded users

        return logits

    def decode_forward(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None):
        batch = tokens.shape[0]

        # Get inputs on device
        tt_inp_emb, start_pos, rot_mat, cache_idxs_tt, tt_page_table = self.tt_model.prepare_device_inputs(
            tokens, start_pos, mode="decode", page_table=page_table
        )

        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            cache_idxs=cache_idxs_tt,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            mode="decode",
        )

        # del tt_inp_emb
        # del rot_mat

        logits = self._process_logits(tt_logits)

        logits = logits.permute(2, 1, 0, 3).squeeze().unsqueeze(1)  # [batch, 1, vocab_size]
        logits = logits[:batch]  # Remove padded users
        # del tt_logits

        return logits

    def prefill_forward_single_user(
        self, tokens: torch.Tensor, start_pos: int, user_id: int, last_token_idx=None, page_table=None, kv_cache=None
    ):
        batch, seq_len = tokens.shape
        assert batch == 1
        assert start_pos == 0, "start_pos must be 0 for prefill_forward_single_user"

        tt_inp_emb, start_pos, rot_mat, _, tt_page_table = self.tt_model.prepare_device_inputs(
            tokens, start_pos=start_pos, valid_seq_len=seq_len, mode="prefill", page_table=page_table
        )

        tt_logits = self.tt_model(
            tt_inp_emb,
            rot_mat,
            start_pos,
            user_id=user_id,
            last_token_idx=last_token_idx,
            page_table=tt_page_table,
            kv_cache=kv_cache,
            mode="prefill",
        )

        del tt_inp_emb
        del rot_mat
        del tt_page_table

        logits = self._process_logits(tt_logits)
        logits = logits.squeeze(1)
        del tt_logits
        return logits

    def prefill_forward(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None, prompt_lens=None):
        batch, batch_seq_len = tokens.shape
        output_logits = torch.zeros(batch, 1, self.params.vocab_size)
        prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([batch_seq_len] * batch)

        if page_table is not None:
            assert isinstance(
                page_table, torch.Tensor
            ), "page_table must be a torch.Tensor when passing into prefill_forward"

        for user_id in range(batch):
            seq_len = prompt_lens[user_id]
            last_token_idx = seq_len - 1

            prefill_seq_len = get_padded_prefill_len(seq_len)
            prefill_ids = torch.cat(
                [tokens[user_id : user_id + 1, :seq_len], torch.zeros(1, prefill_seq_len - seq_len).long()], dim=-1
            )
            if page_table is not None:
                block_size = get_block_size(kv_cache)
                num_padding_blocks = num_blocks_in_seq(prefill_seq_len, block_size) - num_blocks_in_seq(
                    seq_len, block_size
                )
                page_table_user = torch.cat(
                    [page_table, torch.zeros(batch, num_padding_blocks, dtype=torch.int32)], dim=-1
                )

            logger.info(f"Filling kv cache for user {user_id + 1}")

            logits = self.prefill_forward_single_user(
                prefill_ids,
                start_pos,
                user_id,
                last_token_idx=last_token_idx,
                page_table=page_table_user if page_table is not None else None,
                kv_cache=kv_cache,
            )

            # Since we give unpadded_seq_len, only the tile containing the last token is returned
            output_logits[user_id] = logits[:, last_token_idx % 32 : last_token_idx % 32 + 1, :]

        logger.info(f"Finished prefill for all users up to {batch_seq_len} tokens, Starting decode...")

        return output_logits

    def _process_logits(self, tt_logits):
        logits = ttnn.to_torch(
            tt_logits, device=self.mesh_device, mesh_composer=ConcatMeshToTensor(self.mesh_device, dim=3)
        )
        return logits[..., : self.params.vocab_size].float()


def get_padded_prefill_len(seq_len):
    """
    If seq_len is less than 32, pad to 32
    If seq_len is more than 32, pad to whichever is smaller: a power of 2 or a multiple of 1024
    TODO: Generalize for max_mm_seq_len different from 1024
    """
    if seq_len <= 32:
        return 32
    pow_2_pad = nearest_pow_2(seq_len)
    mult_1024_pad = 1024 * math.ceil(seq_len / 1024)
    min_extended_pad = min(pow_2_pad, mult_1024_pad)
    return min_extended_pad


def get_block_size(kv_cache):
    return kv_cache[0][0].shape[2]


def num_blocks_in_seq(seq_len, block_size):
    return math.ceil(seq_len / block_size)


def nearest_pow_2(x):
    return 2 ** math.ceil(math.log2(x))
