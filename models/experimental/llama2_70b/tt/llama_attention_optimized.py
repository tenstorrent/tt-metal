# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
import math
import torch
from torch import nn
import ttnn.experimental as tt_lib
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, nearest_32
from models.experimental.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    generate_rot_emb,
    get_weight_cache_path,
    get_rotation_mat,
    gather_cos_sin,
    precompute_freqs,
)


class TtLlamaAttention_optimized:
    def __init__(
        self,
        device_mesh,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        transformation_mats,
        emulated=False,
        cache_path=None,
        batch_size=None,
        read_cache=False,
    ):
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.num_devices = device_mesh.get_num_devices()
        self.model_config = model_config
        self.emulated = emulated
        self.read_cache = read_cache

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_kv_heads = configuration.n_kv_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size if batch_size is None else batch_size
        self.scale = 1 / math.sqrt(self.head_dim)

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices
        self.padded_local_heads = 32

        self.layer_name = f"{base_url}.{layer_num}"
        self.cache_path = cache_path
        self.transformation_mats = transformation_mats

        self.load_weights()
        self.init_kv_cache()

    def set_model_config(self, model_config):
        self.model_config = model_config

    def init_kv_cache(self):
        """
        Generates empty KV cache and pushed to device memory
        """

        cache_k = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                # self.max_seq_len,
                2048 + 128,  # Meets benchmarking spec needs
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                # self.max_seq_len,
                2048 + 128,  # Meets benchmarking spec needs
                self.head_dim,
            )
        )
        layer_past = [cache_k, cache_v]
        # self.layer_past = [
        #     ttnn.from_torch(
        #         lp,
        #         device=self.device_mesh,
        #         mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
        #         layout=ttnn.TILE_LAYOUT,
        #         dtype=ttnn.bfloat8_b,
        #     )
        #     for lp in layer_past
        # ]

        # # add to the list
        # self.layer_past = [ttnn.to_device(lp, self.device_mesh) for lp in self.layer_past]

        self.layer_past = [
            ttnn.to_device(
                ttnn.as_tensor(
                    lp,
                    device=self.device_mesh,
                    mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=self.model_config["DRAM_MEMCFG"],
                    dtype=ttnn.bfloat8_b,
                    cache_file_name=self.cache_path / f"empty_attn_cache{cache_k.shape}",
                ),
                self.device_mesh,
            )
            for lp in layer_past
        ]

    def load_weights(self):
        assert not hasattr(self, "qkv_list"), "qkv_list is already an attribute of this object"
        assert not hasattr(self, "wo_list"), "wo_list is already an attribute of this object"
        # Load weights
        wqkv_cache_str = f"{self.layer_name}.attention.wqkv_fused.weight"
        wq_str = f"{self.layer_name}.attention.wq.weight"
        wk_str = f"{self.layer_name}.attention.wk.weight"
        wv_str = f"{self.layer_name}.attention.wv.weight"
        wo_str = f"{self.layer_name}.attention.wo.weight"

        qkv_cat = None
        pt_wo = None
        if not self.read_cache:
            qkv_list = []
            for i in range(self.num_devices):
                ### Fused QKV Weights
                # Chunk weights
                wq_chunks = torch.chunk(self.state_dict[wq_str], self.n_heads, dim=0)
                wk_chunks = torch.chunk(self.state_dict[wk_str], self.n_kv_heads, dim=0)
                wv_chunks = torch.chunk(self.state_dict[wv_str], self.n_kv_heads, dim=0)

                # Select chunks for the current device
                wq_selected = torch.cat(wq_chunks[i * self.n_local_heads : (i + 1) * self.n_local_heads], dim=0)
                wk_selected = torch.cat(wk_chunks[i * self.n_local_kv_heads : (i + 1) * self.n_local_kv_heads], dim=0)
                wv_selected = torch.cat(wv_chunks[i * self.n_local_kv_heads : (i + 1) * self.n_local_kv_heads], dim=0)

                # Transpose the selected chunks
                wq = torch.transpose(wq_selected, -2, -1)
                wk = torch.transpose(wk_selected, -2, -1)
                wv = torch.transpose(wv_selected, -2, -1)

                # Create interleaved qkv list
                n_repeat = self.n_heads // self.n_kv_heads
                qkv_interleaved = [
                    [
                        wq[..., i * n_repeat * self.head_dim : (i + 1) * n_repeat * self.head_dim],
                        wk[..., i * self.head_dim : (i + 1) * self.head_dim],
                        wv[..., i * self.head_dim : (i + 1) * self.head_dim],
                    ]
                    for i in range(self.n_local_kv_heads)
                ]
                qkv_interleaved = [item for sublist in qkv_interleaved for item in sublist]

                # Concatenate Q, K, V for the current device
                qkv = torch.cat(qkv_interleaved, dim=-1)
                qkv_list.append(qkv)

            qkv_cat = torch.cat(qkv_list, dim=-1)
            qkv_cat = qkv_cat.unsqueeze(0).unsqueeze(0)

            pt_wo = self.state_dict[wo_str].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        qkv_ttnn = ttnn.as_tensor(
            qkv_cat,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            cache_file_name=self.cache_path / wqkv_cache_str,
        )
        self.qkv = ttnn.to_device(qkv_ttnn, self.device_mesh)

        wo_ttnn = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            cache_file_name=self.cache_path / wo_str,
        )

        self.wo = ttnn.to_device(wo_ttnn, self.device_mesh)

    def prepare_inputs(self, x, start_pos):
        assert len(x.size()) == 3
        batch, seq_len, hidden_size = x.shape

        cache_name = lambda name: self.cache_path / (f"{name}")

        as_tensor = lambda tensor, dtype, layout, name, mesh_mapper, device_mesh: ttnn.as_tensor(
            tensor,
            dtype=dtype,
            layout=layout,
            device=device_mesh,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name) if name is not None else None,
        )

        if self.model_config["LLM_MODE"] == "prefill":
            assert (
                seq_len % 128 == 0 and seq_len > 0 and seq_len <= 2048
            ), "Prefill mode only supports seqlen as a multiple of 128 up to 2k"
            assert batch == 1, "prefill mode only supports batch size 1"
            x = x.unsqueeze(0)
            assert x.shape == (1, batch, seq_len, self.hidden_size)
            xs = as_tensor(
                x, ttnn.bfloat16, ttnn.TILE_LAYOUT, None, ReplicateTensorToMesh(self.device_mesh), self.device_mesh
            )

            cos, sin = precompute_freqs(self.head_dim, self.max_seq_len * 2)
            cos_gathered, sin_gathered = gather_cos_sin(torch.arange(start_pos, start_pos + seq_len), cos, sin)
            assert cos_gathered.size() == (1, 1, seq_len, self.head_dim)
            assert sin_gathered.size() == (1, 1, seq_len, self.head_dim)

            cos_gathereds = as_tensor(
                cos_gathered,
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                f"cos_gathered_prefill_{seq_len}",
                ReplicateTensorToMesh(self.device_mesh),
                self.device_mesh,
            )
            sin_gathereds = as_tensor(
                sin_gathered,
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                f"sin_gathered_prefill_{seq_len}",
                ReplicateTensorToMesh(self.device_mesh),
                self.device_mesh,
            )

            rot_mats = [cos_gathereds, sin_gathereds]

            attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            attn_mask = attn_mask.expand(1, batch, -1, -1)

            attn_masks = as_tensor(
                attn_mask,
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                "attn_mask_prefill_{seq_len}",
                ReplicateTensorToMesh(self.device_mesh),
                self.device_mesh,
            )

            attn_masks = ttnn.to_device(attn_masks, self.device_mesh)

            repeat_shape = (self.n_local_heads, 1, 1, 1)
            attn_masks = tt_lib.tensor.repeat(
                attn_masks, repeat_shape, output_mem_config=self.model_config["DRAM_MEMCFG"]
            )

        elif self.model_config["LLM_MODE"] == "decode":
            assert seq_len == 1, "Only supporting decode mode"
            x = x.transpose(0, 1).unsqueeze(1)
            assert x.shape == (seq_len, 1, batch, self.hidden_size)
            xs = as_tensor(
                x, ttnn.bfloat16, ttnn.TILE_LAYOUT, None, ReplicateTensorToMesh(self.device_mesh), self.device_mesh
            )
            xs = ttnn.to_device(xs, self.device_mesh)
            xs = tt_lib.tensor.interleaved_to_sharded(xs, sharded_mem_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"])

            rot_emb = generate_rot_emb(self.head_dim, self.max_seq_len * 2)
            rot_mat = get_rotation_mat(rot_emb, start_pos, seq_len, batch=batch)
            assert rot_mat.size() == (1, batch, self.head_dim, self.head_dim)
            rot_mats = as_tensor(
                rot_mat,
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                None,
                ReplicateTensorToMesh(self.device_mesh),
                self.device_mesh,
            )
            rot_mats = ttnn.to_device(rot_mats, self.device_mesh)

            rot_mats = tt_lib.tensor.interleaved_to_sharded(
                rot_mats, sharded_mem_config=self.model_config["ROT_MAT_MM_IN1_MEMCFG"]
            )

            padded_layer_past_len = nearest_32(start_pos + 1)
            attn_mask_shape = (seq_len, 1, self.padded_local_heads, padded_layer_past_len)
            attn_mask = torch.zeros(*attn_mask_shape)
            attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min

            attn_masks = as_tensor(
                attn_mask,
                ttnn.bfloat16,
                ttnn.TILE_LAYOUT,
                None,
                ReplicateTensorToMesh(self.device_mesh),
                self.device_mesh,
            )
            attn_masks = ttnn.to_device(attn_masks, self.device_mesh)

            repeat_shape = (1, batch, 1, 1)
            attn_masks = tt_lib.tensor.repeat(
                attn_masks, repeat_shape, output_mem_config=self.model_config["DRAM_MEMCFG"]
            )
            # Put attn_mask on the device with the sharded config
            attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
            if attention_mask_memconfig.is_sharded():
                attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
                attn_mask_shard_shape[-1] = padded_layer_past_len
                attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape

                attn_masks = tt_lib.tensor.interleaved_to_sharded(
                    attn_masks, sharded_mem_config=attention_mask_memconfig
                )

        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def __call__(
        self,
        xs,
        rot_mats,
        start_pos: int,
        attn_masks,
        user_id: int = 0,
    ):
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        if self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        # Prefill should have input tensor of shape (1, batch=1, seqlen, hidden_size)
        elif self.model_config["LLM_MODE"] == "prefill":
            return self.prefill_forward(xs, rot_mats, attn_masks, user_id)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def decode_forward(
        self,
        xs,
        rot_mats,
        start_pos: int,
        attn_masks,
    ):
        query_layer, key_layer, value_layer = self.attn_qkv(xs, rot_mats)
        attn_outputs = self.attn_mqa(query_layer, key_layer, value_layer, start_pos, attn_masks)
        return self.attn_selfout(attn_outputs)

    def attn_qkv(
        self,
        xs,
        rot_mats,
    ):
        # Assume input is already padded to 32, even if the batch size is not 32. Batch size is in self.max_batch_size.

        # Reshard
        if self.model_config["LN_ATTN_OUTPUT_MEMCFG"] != self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"]:
            # xs = tt_lib.tensor.reshard(xs, self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"])
            xs = tt_lib.tensor.sharded_to_interleaved(xs, self.model_config["L1_MEMCFG"])
            xs = tt_lib.tensor.interleaved_to_sharded(xs, self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"])

        # Fused QKV
        fused_query_key_value = tt_lib.operations.primary.matmul_1d(
            xs,
            self.qkv,
            program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
            output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        xs.deallocate(True)

        # TMs
        (
            query_layer,  # [seqlen, n_local_heads, bsz, head_dim]
            key_layer,  # [seqlen, n_local_kv_heads, bsz, head_dim]
            value_layer,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        ) = tt_lib.tensor.nlp_create_qkv_heads_decode(
            fused_query_key_value,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )

        fused_query_key_value.deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        query_layer = tt_lib.operations.primary.matmul(
            query_layer,
            rot_mats,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"]
            # [seqlen, n_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_heads, bsz, head_dim]
        )

        key_layer = tt_lib.operations.primary.matmul(
            key_layer,
            rot_mats,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )

        return query_layer, key_layer, value_layer

    def attn_mqa(
        self,
        query_layer,
        key_layer,
        value_layer,
        start_pos: int,
        attn_masks,
        batch_offset: int = 0,
    ):
        padded_layer_past_len = nearest_32(start_pos + 1)

        # K Cache Update
        kv_cache_memcfg = self.model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"]
        if kv_cache_memcfg.is_sharded():
            kv_cache_shard_shape = kv_cache_memcfg.shard_spec.shape
            kv_cache_shard_shape[0] = self.layer_past[0].shape[0] * padded_layer_past_len
            kv_cache_memcfg.shard_spec.shape = kv_cache_shard_shape

        keys = self.layer_past[0]
        tt_lib.tensor.update_cache(keys, key_layer, start_pos, batch_offset=batch_offset)
        key_layer.deallocate(True)
        # key and value layers will have kv_seq_len padded to nearest 32

        keys = self.layer_past[0]
        key_layer = tt_lib.tensor.nlp_kv_cache_load_slice(keys, 0, padded_layer_past_len)

        # PRE-SOFTMAX MM

        key_layer_transposed = tt_lib.tensor.transpose(
            key_layer,
            -2,
            -1,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )

        key_layer.deallocate(True)

        attn_prog_config = self.model_config["ATTN_BATCHED_MM_PROGCFG_LAMBDA"](padded_layer_past_len // 32)
        attn_output_memcfg = self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"]
        attn_output_memcfg.shard_spec.shape[1] = padded_layer_past_len

        attn_weights = tt_lib.operations.primary.matmul(
            query_layer,
            key_layer_transposed,
            program_config=attn_prog_config,
            output_mem_config=attn_output_memcfg,
            output_dtype=self.model_config["ATTN_BATCHED_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        query_layer.deallocate(True)
        key_layer_transposed.deallocate(True)

        # SOFTMAX
        softmax_progcfg = self.model_config["BATCHED_SOFTMAX_PROGCFG"]
        softmax_progcfg.block_w = padded_layer_past_len // 32

        attn_weights = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
            attn_weights,
            self.scale,
            attn_masks,
            program_config=self.model_config["BATCHED_SOFTMAX_PROGCFG"],
            is_causal_mask=True,
        )

        # V CACHE UPDATE

        values = self.layer_past[1]
        tt_lib.tensor.update_cache(values, value_layer, start_pos, batch_offset=batch_offset)
        value_layer.deallocate(True)

        values = self.layer_past[1]
        value_layer = tt_lib.tensor.nlp_kv_cache_load_slice(values, 0, padded_layer_past_len)

        # POST-SOFTMAX MM
        scores_prog_config = self.model_config["SCORES_BATCHED_MM_PROGCFG_LAMBDA"](padded_layer_past_len // 32)

        attn_output = tt_lib.operations.primary.matmul(
            attn_weights,
            value_layer,
            program_config=scores_prog_config,
            output_mem_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
            output_dtype=self.model_config["BFLOAT16_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        attn_weights.deallocate(True)
        value_layer.deallocate(True)

        return attn_output

    def attn_selfout(
        self,
        attn_output,
    ):
        # ATTENTION SELFOUT

        attn_output = tt_lib.tensor.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.n_local_heads,
        )  # seqlen, 1, batch, hidden_size

        # attn_output = tt_lib.tensor.sharded_to_interleaved(
        #     attn_output, output_mem_config=self.model_config["L1_MEMCFG"]
        # )
        attn_output = ttnn.all_gather(
            attn_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            # memory_config=self.model_config["L1_MEMCFG"],
            memory_config=self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        # attn_output = tt_lib.tensor.interleaved_to_sharded(
        #     attn_output, sharded_mem_config=self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"]
        # )
        attn_output = tt_lib.tensor.reshard(attn_output, self.model_config["SELFOUT_MM_INPUT_MEMCFG"])

        attn_output = tt_lib.operations.primary.matmul_1d(
            attn_output,
            self.wo,
            program_config=self.model_config["SELFOUT_MM_PROGCFG"],
            output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            output_dtype=self.model_config["BFP8_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )  # seqlen, 1, batch, hidden_size

        return attn_output

    def prefill_forward(
        self,
        xs,
        rot_mats,
        attn_masks,
        user_id: int = 0,
    ):
        query_layer, key_layer, value_layer = self.prefill_attn_qkv(xs, rot_mats)
        attn_outputs = self.prefill_attn_mqa(query_layer, key_layer, value_layer, attn_masks, user_id)
        return self.prefill_attn_selfout(attn_outputs)

    def prefill_attn_qkv(
        self,
        xs,
        rot_mats,
    ):
        assert xs.shape[1] == 1, "batch must be 1"
        assert xs.shape[2] % 128 == 0 and xs.shape[2] > 0, "Seqlen must be divisible by 128"

        # Fused QKV
        fused_query_key_value = tt_lib.operations.primary.matmul(
            xs,
            self.qkv,
            program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
            output_mem_config=self.model_config["DRAM_MEMCFG"],
            output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        xs.deallocate(True)

        (
            query_layer,  # [bsz, n_local_heads, seq_len, head_dim]
            key_layer,  # [bsz, n_local_kv_heads, seq_len, head_dim]
            value_layer,  # [bsz, n_local_kv_heads, seq_len, head_dim]
        ) = tt_lib.tensor.nlp_create_qkv_heads(
            fused_query_key_value,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            output_mem_config=self.model_config["DRAM_MEMCFG"],
        )

        fused_query_key_value.deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        # query_layer: ttnn.Shape([1, 8, seq_len, 128]) -> [bsz, n_local_heads, seq_len, head_dim]

        query_layer_ret = self.apply_rotary_prefill(query_layer, rot_mats[0], rot_mats[1], self.transformation_mats)
        query_layer.deallocate(True)

        # K Rotary Embeddings

        # key_layer: ttnn.Shape([1, 1, seq_len, 128])
        key_layer_ret = self.apply_rotary_prefill(key_layer, rot_mats[0], rot_mats[1], self.transformation_mats)
        key_layer.deallocate(True)

        return query_layer_ret, key_layer_ret, value_layer

    def apply_rotary_prefill(self, x, cos, sin, transform_mat):
        batch, n_heads, _, _ = x.shape

        cos = ttnn.repeat(cos, ttnn.Shape([batch, n_heads, 1, 1]))
        sin = ttnn.repeat(sin, ttnn.Shape([batch, n_heads, 1, 1]))

        x_transformed = ttnn.matmul(x, transform_mat)

        x_cos = ttnn.mul(cos, x)
        x_sin = ttnn.mul(sin, x_transformed)
        return ttnn.add(x_cos, x_sin)

    def prefill_attn_mqa(
        self,
        query_layer,
        key_layer,
        value_layer,
        attn_masks,
        user_id: int = 0,
    ):
        seq_len = query_layer.shape[2]
        slice_size = 256 if seq_len == 2048 else 128
        cores_y = 4 if slice_size == 128 else 8
        num_slices = seq_len // slice_size  # we do q_lens of 128 per iteration (slice), then we concat the result.

        # FILL K CACHE
        keys = self.layer_past[0]
        # Fill cache expects batch in dim0
        keys_reshaped = ttnn.reshape(keys, [self.max_batch_size, self.n_local_kv_heads, -1, self.head_dim])
        tt_lib.tensor.fill_cache(
            keys_reshaped, tt_lib.tensor.typecast(key_layer, self.model_config["BFP8_DTYPE"]), user_id
        )
        # tt_lib.tensor.fill_cache(
        #     keys_reshaped, key_layer, user_id
        # )

        # FILL V CACHE
        values = self.layer_past[1]
        # Fill cache expects batch in dim0
        values_reshaped = ttnn.reshape(values, [self.max_batch_size, self.n_local_kv_heads, -1, self.head_dim])
        tt_lib.tensor.fill_cache(
            values_reshaped, tt_lib.tensor.typecast(value_layer, self.model_config["BFP8_DTYPE"]), user_id
        )
        # tt_lib.tensor.fill_cache(
        #     values_reshaped, value_layer, user_id
        # )

        # SPDA
        program_config = tt_lib.operations.primary.transformers.SDPAMultiCoreProgramConfig(
            compute_with_storage_grid_size=[8, 8],
            q_chunk_size=128,
            k_chunk_size=128,
        )
        attn_output = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_masks,
            is_causal=True,
            scale=self.scale,
            program_config=program_config,
        )

        # deallocate keys and values
        query_layer.deallocate(True)
        key_layer.deallocate(True)
        value_layer.deallocate(True)

        return attn_output

    def prefill_attn_selfout(self, attn_output):
        # ATTENTION SELFOUT
        attn_output = tt_lib.tensor.nlp_concat_heads(
            attn_output,
            output_mem_config=self.model_config["L1_MEMCFG"],
        )  # seqlen, 1, batch, hidden_size

        attn_output = ttnn.all_gather(
            attn_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DRAM_MEMCFG"],
        )

        seq_tiles = attn_output.shape[2] // 32
        cores_y = 8 if seq_tiles % 8 == 0 else 4
        dense_out_prog_cfg = self.model_config["SELFOUT_MM_PROGCFG_LAMBDA"](seq_tiles, cores_y)
        # print('wo matmul')
        attn_output = tt_lib.operations.primary.matmul(
            attn_output,
            self.wo,
            program_config=dense_out_prog_cfg,
            output_mem_config=self.model_config["DRAM_MEMCFG"],
            output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )  # seqlen, 1, batch, hidden_size

        return attn_output
