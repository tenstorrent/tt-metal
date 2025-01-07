# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import math
import ttnn
import torch
import torch.nn as nn
from tqdm import tqdm
from models.demos.llama3.tt.llama_decoder import TtTransformerBlock
from models.common.rmsnorm import RMSNorm
import ttnn
from typing import Optional
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3.tt.distributed_norm import DistributedNorm
from models.demos.llama3.tt.lm_head import LMHead
from models.demos.llama3.tt.llama_common import copy_host_to_device, get_prefill_rot_mat, HostEmbedding
from models.demos.llama3.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3.tt.llama_embedding import TtLlamaEmbedding


class TtTransformer(LightweightModule):
    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict,
        weight_cache_path,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0
        self.n_layers = args.n_layers
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.model_config = args.get_model_config()
        self.grid_size = self.args.max_grid_size
        state_dict_prefix = args.get_state_dict_prefix("", None)

        self.embd = TtLlamaEmbedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=args.weight_cache_path(dtype),
            state_dict=state_dict,
            dtype=ttnn.bfloat16,  # Row major layout requires bfloat16
        )

        self.rope_setup = TtLlamaRotarySetup(
            mesh_device,
            args.max_batch_size,
            args.head_dim,
            args.max_seq_len,
            args.rope_theta,
            args.use_scaled_rope,
            args.rope_scaling_factor,
        )
        self.trans_mats_dict = self.rope_setup.get_both_trans_mats()

        self.layers = [
            TtTransformerBlock(
                args=args,
                mesh_device=mesh_device,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                transformation_mats=self.trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
            )
            for i in tqdm(range(self.n_layers))
        ]
        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", None),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="norm",
                is_distributed=self.args.is_distributed_norm,
                sharded_program_config=self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"],
                sharded_output_config=self.model_config["LM_HEAD_INPUT_MEMCFG"],
            ),
            args,
            args.is_galaxy,
        )

        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
        )

    def prepare_inputs_prefill(self, tokens, start_pos=0, page_table=None, chunk_page_table=None):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        TODO: Debate whether this function is responsible for padding
        """

        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens_embd = self.embd(tokens)
        tokens_embd = ttnn.unsqueeze_to_4D(tokens_embd)

        tt_rot_mats_prefill = get_prefill_rot_mat(
            self.args.head_dim,
            self.args.max_seq_len,
            self.mesh_device,
            seq_len=S,
            scale_factor=self.args.rope_scaling_factor,
            start_pos=start_pos,
        )

        if page_table is not None:
            tt_page_table = ttnn.from_torch(
                page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=self.mesh_device,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        return tokens_embd, tt_rot_mats_prefill, tt_page_table, tt_chunk_page_table

    def prepare_inputs_decode(self, *inputs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        """
        host_inputs = self.prepare_decode_inputs_host(*inputs)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)  # Helper function
        transformed_device_inputs = self.transform_decode_inputs_device(*device_inputs)
        return transformed_device_inputs

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """
        Inputs are torch tensors or python types. Outputs are ttnn tensors on host.
        NOTE: Tokens and current_pos are padded to batch
        """
        B = tokens.shape[0]
        assert current_pos.shape[0] == B, "Batch size mismatch"
        assert B == self.args.max_batch_size, "Batch size must be equal to max_batch_size"

        tokens = ttnn.from_torch(
            tokens.view(-1),
            device=None,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens = ttnn.unsqueeze_to_4D(tokens)

        rot_current_pos = torch.maximum(
            current_pos, torch.tensor(0, dtype=torch.int64)
        )  # Ensure position indices are non-negative
        rope_idxs = self.rope_setup.get_rot_idxs(rot_current_pos, on_host=True)
        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, 0) if (self.args.is_galaxy and B > 1) else (None, None),
                mesh_shape=self.args.cluster_shape,
            ),
        )

        if page_table is not None:
            page_table = ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.int32,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(None, -2) if (self.args.is_galaxy and B > 1) else (None, None),
                    mesh_shape=self.args.cluster_shape,
                ),
            )
        return tokens, current_pos_tt, rope_idxs, page_table

    def transform_decode_inputs_device(self, tokens, current_pos, rope_idxs, page_table=None):
        """
        Inputs are ttnn tensors on device. This function applies any on-device
        transformations which should happen before forward decode.
        For example: tilize, reshape, shard.
        Return transformed device tensors

        Get rope sin/cos
        Embed tokens
        """
        tt_rot_mats = self.rope_setup.get_rot_mats(rope_idxs)
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        tt_tokens = ttnn.to_memory_config(
            tt_tokens,
            self.args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )
        return tt_tokens, current_pos, tt_rot_mats, page_table

    def process_output_prefill(self, tt_out, last_token_idx):
        """
        Input is ttnn device tensor of logits. Output is torch logits tensor.
        NOTE: In this model, prefill always uses get_last_token
        """
        logits = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, dims=(3, 1) if self.args.is_galaxy else (1, -1), mesh_shape=self.args.cluster_shape
            ),
        )[0, 0, last_token_idx, : self.vocab_size]
        return logits

    def process_output_decode(self, tt_out, B, S=1):
        """
        Input is ttnn device tensor of logits. Output is torch logits tensor
        """
        if self.args.num_devices > 1:
            if self.args.is_galaxy:
                tt_out = ttnn.all_gather(
                    tt_out,
                    dim=3,
                    num_links=2,
                    cluster_axis=0,
                    mesh_device=self.mesh_device,
                    topology=ttnn.Topology.Linear,
                )
            else:
                tt_out = ttnn.all_gather(tt_out, dim=3, num_links=1, topology=ttnn.Topology.Linear)
        tt_out = ttnn.untilize(tt_out, use_multicore=True)
        if self.args.num_devices > 1:
            tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        else:
            tt_out = ttnn.to_torch(tt_out).float()
        tt_out = tt_out[:, :, :B, : self.vocab_size].view(B, S, -1)
        return tt_out

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats,
        user_id,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        return self.forward(
            x,
            current_pos=None,
            rot_mats=rot_mats,
            user_id=user_id,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
        )

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mats,
        page_table=None,
        kv_cache=None,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        return self.forward(
            x,
            current_pos,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table,
            kv_cache=kv_cache,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
    ):
        # No-op if callers already provide the right memory config
        if mode == "decode" and not self.args.is_galaxy:
            x = ttnn.to_memory_config(x, self.model_config["DECODE_RESIDUAL_MEMCFG"])

        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                current_pos,
                rot_mats,
                user_id,
                mode,
                page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )

        if mode == "prefill" and get_last_token == -1:
            return x

        # Slicing the tensor to the nearest ceiling/floor multiples of 32 for the prefill_len, to get the last token
        if get_last_token != -1:
            x = ttnn.slice(x, (0, 0, get_last_token, 0), (1, 1, get_last_token + 32, x.shape[-1]))

        # Output norm
        x = self.norm(x, mode=mode)

        if mode == "prefill" and self.model_config["LM_HEAD_INPUT_MEMCFG"].is_sharded():
            x = ttnn.interleaved_to_sharded(x, self.model_config["LM_HEAD_INPUT_MEMCFG"])

        return self.lm_head(x)
