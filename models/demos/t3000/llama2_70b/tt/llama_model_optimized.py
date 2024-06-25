# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
from tqdm import tqdm
import torch
import ttnn.experimental as tt_lib
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh


from models.utility_functions import nearest_32, profiler
from models.demos.t3000.llama2_70b.tt.llama_decoder_optimized import TtLlamaDecoder_optimized
from models.demos.t3000.llama2_70b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.tt.llama_common import (
    freqs_to_rotation_matrix,
    get_rotation_mat,
    precompute_freqs,
    gather_cos_sin,
    get_rot_transformation_mat,
)


class TtLlamaModel_optimized:
    def __init__(
        self,
        device_mesh,
        state_dict,
        base_url,
        n_layers,
        model_config,
        configuration,
        cache_path=None,
        read_cache=False,
    ):
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.num_devices = device_mesh.get_num_devices()
        self.model_config = model_config
        self.read_cache = read_cache

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.padded_local_heads = 32
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.vocab_size = configuration.vocab_size
        self.norm_eps = configuration.norm_eps
        self.llama3 = self.vocab_size == 128256
        self.rope_theta = configuration.rope_theta if self.llama3 else 10000.0

        self.cache_path = cache_path
        # Transformation matrix for rotary embeddings
        transformation_mat_torch = get_rot_transformation_mat(32)  # 32 for tile size
        transformation_mats = ttnn.as_tensor(
            transformation_mat_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device_mesh,
            memory_config=model_config["DRAM_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(device_mesh),
        )
        transformation_mats = ttnn.to_device(transformation_mats, device_mesh)

        logger.info("Creating Layers")
        self.layers = [
            TtLlamaDecoder_optimized(
                device_mesh,
                state_dict,
                base_url,
                layer_num,
                model_config,
                configuration,
                transformation_mats,
                cache_path=cache_path,
                read_cache=read_cache,
            )
            for layer_num in tqdm(range(n_layers))
        ]
        logger.info("Done creating layers")

        # Rotary Embedding
        self.cos, self.sin = precompute_freqs(self.head_dim, self.max_seq_len * 2, self.rope_theta)  # for prefill
        self.rot_emb = freqs_to_rotation_matrix(self.cos, self.sin)  # for decode
        # Embedding
        self.tt_embd = TtLlamaEmbedding(
            device_mesh,
            state_dict,
            cache_path,
        )
        self.load_weights()

    def set_model_config(self, model_config):
        self.model_config = model_config
        for layer in self.layers:
            layer.set_model_config(model_config)

    def load_weights(self):
        norm_str = "norm.weight"
        lm_head_str = "output.weight"

        if not self.read_cache:
            H = 8 * 1024
            if self.llama3:
                PADDED_VOCAB = 128 * 1024
            else:
                PADDED_VOCAB = 32 * 1024
            padded_lm_head = torch.zeros(1, 1, H, PADDED_VOCAB)
            padded_lm_head[:, :, :, : self.vocab_size] = self.state_dict[lm_head_str].transpose(-2, -1)

            pt_norm_weight = self.state_dict[norm_str].reshape([1, 1, -1, 32])
        else:
            padded_lm_head = None
            pt_norm_weight = None

        padded_lm_head_ttnn = ttnn.as_tensor(
            padded_lm_head,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            cache_file_name=self.cache_path / lm_head_str,
        )
        self.lm_head = ttnn.to_device(padded_lm_head_ttnn, self.device_mesh)

        norm_ttnn = ttnn.as_tensor(
            pt_norm_weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            cache_file_name=self.cache_path / norm_str,
        )
        self.norm = ttnn.to_device(norm_ttnn, self.device_mesh)

    def prepare_inputs(self, inp_ids, start_pos, valid_seq_len=None):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        inp_ids: (batch, seq)
        start_pos: int
        valid_seq_len: int, optional for mask padding

        returns:
        xs: [(seq, batch, hidden_dim)] * num_devices
        start_pos: int
        rot_mats: [(1, 1, head_dim, head_dim)] * num_devices  for decode
                  [(1, 1, seq, head_dim), (1, 1, seq, head_dim)] * num_devices  for prefill
        attn_masks: [(seq, n_local_heads, batch, max_seq_len)] * num_devices  for decode
                    [(1, n_local_heads, seq, seq)] * num_devices  for prefill
        """
        assert inp_ids.dim() == 2
        batch, seq_len = inp_ids.shape

        cache_name = lambda name: self.cache_path / (f"{'llama3_' if self.llama3 else ''}{name}")

        if self.model_config["LLM_MODE"] == "decode":
            inp_ids = inp_ids.reshape(seq_len, 1, 1, batch)
        else:
            inp_ids = inp_ids.reshape(batch, 1, 1, seq_len)

        x = ttnn.as_tensor(
            inp_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
        )
        x = ttnn.to_device(x, self.device_mesh)

        xs = self.tt_embd(x)

        if self.model_config["LLM_MODE"] == "prefill":
            assert (
                seq_len % 128 == 0 and seq_len > 0
            ), "Prefill mode only supports seqlen as a multiple of 128 up to 8k (llama3) and 2k (llama2)"
            assert batch == 1, "prefill mode only supports batch size 1"
            assert xs.shape == (batch, 1, seq_len, self.hidden_size // self.num_devices)

            cos_gathered, sin_gathered = gather_cos_sin(
                torch.arange(start_pos, start_pos + seq_len), self.cos, self.sin
            )
            assert cos_gathered.size() == (1, 1, seq_len, self.head_dim)
            assert sin_gathered.size() == (1, 1, seq_len, self.head_dim)

            cos_gathereds = ttnn.as_tensor(
                cos_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name(f"cos_gathered_prefill_{seq_len}"),
                memory_config=self.model_config["DRAM_MEMCFG"],
                device=self.device_mesh,
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            )
            sin_gathereds = ttnn.as_tensor(
                sin_gathered,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name(f"sin_gathered_prefill_{seq_len}"),
                memory_config=self.model_config["DRAM_MEMCFG"],
                device=self.device_mesh,
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            )
            cos_gathereds = ttnn.to_device(cos_gathereds, self.device_mesh)
            sin_gathereds = ttnn.to_device(sin_gathereds, self.device_mesh)
            rot_mats = [cos_gathereds, sin_gathereds]

            attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            if valid_seq_len:
                attn_mask[:, valid_seq_len:] = torch.finfo(
                    attn_mask.dtype
                ).min  # Mask columns beyond valid_seq_len as padding
                attn_mask[valid_seq_len:, :] = torch.finfo(
                    attn_mask.dtype
                ).min  # Mask rows beyond valid_seq_len as padding
            attn_mask = attn_mask.expand(batch, 1, -1, -1)

            attn_masks = ttnn.as_tensor(
                attn_mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name(f"attn_masks_prefill_{seq_len}"),
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
                memory_config=self.model_config["DRAM_MEMCFG"],
                device=self.device_mesh,
            )
            attn_masks = ttnn.to_device(attn_masks, self.device_mesh)

        elif self.model_config["LLM_MODE"] == "decode":
            assert seq_len == 1, "Decode mode only supports seq_len=1"
            assert xs.shape == (seq_len, 1, batch, self.hidden_size // self.num_devices)

            xs = tt_lib.tensor.interleaved_to_sharded(
                xs, sharded_mem_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
            )

            rot_mat = get_rotation_mat(self.rot_emb, start_pos, seq_len, batch=batch)
            assert rot_mat.size() == (1, batch, self.head_dim, self.head_dim)

            rot_mats = ttnn.as_tensor(
                rot_mat,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device_mesh,
                cache_file_name=cache_name(f"rot_mat_decode_{start_pos}"),
                memory_config=self.model_config["DRAM_MEMCFG"],
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            )
            rot_mats = ttnn.to_device(rot_mats, self.device_mesh)

            rot_mats = tt_lib.tensor.interleaved_to_sharded(
                rot_mats, sharded_mem_config=self.model_config["ROT_MAT_MM_IN1_MEMCFG"]
            )

            padded_layer_past_len = nearest_32(start_pos + 1)

            padded_layer_past_len = nearest_32(start_pos + 1)
            attn_mask_shape = (seq_len, 1, self.padded_local_heads, padded_layer_past_len)
            attn_mask = torch.zeros(*attn_mask_shape)
            attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min

            attn_masks = ttnn.as_tensor(
                attn_mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name(f"attn_masks_decode_{start_pos}"),
                memory_config=self.model_config["DRAM_MEMCFG"],
                mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
                device=self.device_mesh,
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
        xs: List[tt_lib.tensor.Tensor],
        rot_mats: List[tt_lib.tensor.Tensor],
        start_pos: int,
        attn_masks: List[tt_lib.tensor.Tensor],
        user_id: int = 0,
    ) -> tt_lib.tensor.Tensor:
        if self.model_config["LLM_MODE"] == "prefill":
            return self.prefill_forward(xs, rot_mats, start_pos, attn_masks, user_id)
        elif self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def decode_forward(
        self,
        xs: List[tt_lib.tensor.Tensor],
        rot_mats: List[tt_lib.tensor.Tensor],
        start_pos: int,
        attn_masks: List[tt_lib.tensor.Tensor],
    ) -> tt_lib.tensor.Tensor:
        ### Run all layers
        for layer in self.layers:
            xs = layer(xs, rot_mats, start_pos, attn_masks)  # xs is sharded

        xs = ttnn.all_gather(
            xs,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        # In-place RMSNorm
        norm_out_replicated = tt_lib.operations.primary.rmsnorm(
            xs,
            self.norm_eps,
            self.norm,
            program_config=self.model_config["LN_F_PROGCFG"],
            output_mem_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
            compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
        )

        ### Each device does an LM head fracture
        lm_head_out = tt_lib.operations.primary.matmul_1d(
            norm_out_replicated,
            self.lm_head,
            program_config=self.model_config["LLAMA3_LM_HEAD_MM_PROGCFG"]
            if self.llama3
            else self.model_config["LM_HEAD_MM_PROGCFG"],
            output_mem_config=self.model_config["DRAM_MEMCFG"],
            output_dtype=ttnn.bfloat16,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        norm_out_replicated.deallocate(True)

        return lm_head_out

    def sharded_rmsnorm(self, xs, eps, norm_list):
        # Do sharded RMS by partial sequence length of 128
        # Input xs[0] is [1, 1, seq_len, 8192]
        seq_len = xs.shape[2]
        slice_size = 128
        num_slices = seq_len // slice_size  # we do 128 per iteration (slice), then we concat the result.

        xs_output_cat = ttnn.as_tensor(
            torch.zeros([1, 1, seq_len, self.hidden_size]),
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
        )

        layernorm_num_cores_x, layernorm_num_cores_y = (
            self.model_config["layernorm_params"]["layernorm_num_cores_x"],
            self.model_config["layernorm_params"]["layernorm_num_cores_y"],
        )
        layernorm_shard_height_hidden_dim, layernorm_shard_width_hidden_dim = (
            self.model_config["layernorm_params"]["layernorm_shard_height_hidden_dim"],
            self.model_config["layernorm_params"]["layernorm_shard_width_hidden_dim"],
        )

        for slice_i in range(num_slices):
            xs_slice = tt_lib.tensor.interleaved_to_sharded_partial(
                xs,
                (layernorm_num_cores_x, layernorm_num_cores_y),
                [layernorm_shard_height_hidden_dim, layernorm_shard_width_hidden_dim],
                num_slices,  # num_slices
                slice_i,  # slice_index
                tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                tt_lib.tensor.ShardOrientation.ROW_MAJOR,
            )

            xs_slice = tt_lib.operations.primary.rmsnorm(
                xs_slice,
                eps,
                norm_list,
                program_config=self.model_config["LN_F_PROGCFG"],
                output_mem_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
                compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
            )

            tt_lib.tensor.sharded_to_interleaved_partial(
                xs_slice,
                xs_output_cat,
                num_slices,
                slice_i,
                self.model_config["DRAM_MEMCFG"],
            )
            xs_slice.deallocate(True)
        return xs_output_cat

    def prefill_forward(
        self,
        xs: List[tt_lib.tensor.Tensor],
        rot_mats: List[tt_lib.tensor.Tensor],
        start_pos: int,
        attn_masks: List[tt_lib.tensor.Tensor],
        user_id: int = 0,
    ) -> tt_lib.tensor.Tensor:
        ### Run all layers
        for layer in self.layers:
            xs = layer(xs, rot_mats, start_pos, attn_masks, user_id)  # xs is sharded

        ## Gather fractured layers output
        xs = ttnn.all_gather(
            xs,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DRAM_MEMCFG"],
        )
        ## Duplicate layernorm
        norm_out_replicated = self.sharded_rmsnorm(xs, self.norm_eps, self.norm)

        # Deallocate original input to rmsnorm
        xs.deallocate(True)

        ### Each device does an LM head fracture
        if self.llama3:
            self.model_config["LM_HEAD_MM_PROGCFG"] = self.model_config["LLAMA3_LM_HEAD_MM_PROGCFG"]

        _, _, seq_len, _ = norm_out_replicated.shape

        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"]
        batch_dim = 1 if seq_len < max_mm_seq_len else seq_len // max_mm_seq_len  # Find the division factor
        norm_out_replicated = ttnn.reshape(norm_out_replicated, (1, batch_dim, seq_len // batch_dim, -1))

        lm_head_out = tt_lib.operations.primary.matmul(
            norm_out_replicated,
            self.lm_head,
            program_config=self.model_config["LM_HEAD_MM_PROGCFG"],
            output_mem_config=self.model_config["DRAM_MEMCFG"],
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
        )
        norm_out_replicated.deallocate(True)

        lm_head_out = ttnn.reshape(lm_head_out, (1, 1, seq_len, -1))

        return lm_head_out
