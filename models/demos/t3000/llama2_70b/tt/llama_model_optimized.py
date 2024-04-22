# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from typing import List
from tqdm import tqdm
import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, nearest_32, profiler
from models.demos.t3000.llama2_70b.tt.llama_decoder_optimized import TtLlamaDecoder_optimized
from models.demos.t3000.llama2_70b.tt.llama_embedding import TtLlamaEmbedding
from models.demos.t3000.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    freqs_to_rotation_matrix,
    get_weight_cache_path,
    get_rotation_mat,
    precompute_freqs,
    gather_cos_sin,
    get_rot_transformation_mat,
)


class TtLlamaModel_optimized(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        n_layers,
        model_config,
        configuration,
        batch,
        emulated=False,
        cache_path=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.model_config = model_config
        self.emulated = emulated

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
        transformation_mat_torch = get_rot_transformation_mat(self.head_dim)
        transformation_mats = [torch2tt_tensor(transformation_mat_torch.clone(), device) for device in devices]

        logger.info("Creating Layers")
        self.layers = [
            TtLlamaDecoder_optimized(
                devices,
                state_dict,
                base_url,
                layer_num,
                model_config,
                configuration,
                batch,
                transformation_mats,
                emulated=emulated,
                cache_path=cache_path,
            )
            for layer_num in tqdm(range(n_layers))
        ]
        logger.info("Done creating layers")

        # Rotary Embedding
        self.cos, self.sin = precompute_freqs(self.head_dim, self.max_seq_len * 2, self.rope_theta)  # for prefill
        self.rot_emb = freqs_to_rotation_matrix(self.cos, self.sin)  # for decode
        # Embedding
        self.tt_embd = TtLlamaEmbedding(
            devices,
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

        self.norm_list = []
        self.lm_head_list = []

        test_cache_path = get_weight_cache_path(self.cache_path, lm_head_str, self.num_devices - 1, self.num_devices)
        if test_cache_path.exists():
            for i in range(self.num_devices):
                tensor_cache_path = get_weight_cache_path(self.cache_path, norm_str, i, self.num_devices)
                self.norm_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )

                tensor_cache_path = get_weight_cache_path(self.cache_path, lm_head_str, i, self.num_devices)
                self.lm_head_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )
        else:
            H = 8 * 1024
            if self.llama3:
                PADDED_VOCAB = 128 * 1024
            else:
                PADDED_VOCAB = 32 * 1024
            padded_lm_head = torch.zeros(H, PADDED_VOCAB)
            padded_lm_head[:, : self.vocab_size] = self.state_dict[lm_head_str].transpose(-2, -1)
            padded_lm_head_chunks = torch.chunk(padded_lm_head, self.num_devices, -1)

            for i in range(self.num_devices):
                output_norm_host = tt_lib.tensor.Tensor(
                    # Expand to size of input since we decomped norm
                    self.state_dict[norm_str].reshape([1, 1, -1, 32]),
                    self.model_config["LN_F_WEIGHTS_DTYPE"],
                )
                self.norm_list.append(output_norm_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, norm_str, i, self.num_devices)),
                    output_norm_host,
                )

                lm_head_host = torch2tt_tensor(
                    padded_lm_head_chunks[i],
                    None,
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=self.model_config["BFP8_DTYPE"],
                )
                self.lm_head_list.append(lm_head_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, lm_head_str, i, self.num_devices)),
                    lm_head_host,
                )

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

        cache_name = lambda name: self.cache_path / (f"{name}")

        cache_file_name = lambda name, dtype, layout: f"{cache_name(name)}_dtype_{dtype}_layout_{layout}.bin"

        as_tensor = lambda tensor, dtype, layout, name, device_id: ttnn.as_tensor(
            tensor,
            dtype=dtype,
            layout=layout,
            device=self.devices[device_id],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name) if name is not None else None,
        )

        if self.model_config["LLM_MODE"] == "decode":
            inp_ids = inp_ids.reshape(seq_len, 1, 1, batch)
        else:
            inp_ids = inp_ids.reshape(batch, 1, 1, seq_len)

        x = [
            ttnn.from_torch(
                inp_ids.clone(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.devices[device_id],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for device_id in range(self.num_devices)
        ]

        xs = self.tt_embd(x)

        if self.model_config["LLM_MODE"] == "prefill":
            assert (
                seq_len % 128 == 0 and seq_len > 0 and seq_len <= 2048
            ), "Prefill mode only supports seqlen as a multiple of 128 up to 2k"
            assert batch == 1, "prefill mode only supports batch size 1"
            assert xs[0].shape == (batch, 1, seq_len, self.hidden_size // self.num_devices)

            cos_gathered, sin_gathered = gather_cos_sin(
                torch.arange(start_pos, start_pos + seq_len), self.cos, self.sin
            )
            assert cos_gathered.size() == (1, 1, seq_len, self.head_dim)
            assert sin_gathered.size() == (1, 1, seq_len, self.head_dim)

            cos_gathereds, sin_gathereds = [], []
            for device_id in range(self.num_devices):
                cos_gathereds.append(
                    as_tensor(
                        cos_gathered.clone(),
                        ttnn.bfloat16,
                        ttnn.TILE_LAYOUT,
                        f"cos_gathered_prefill_{seq_len}",
                        device_id,
                    )
                )
                sin_gathereds.append(
                    as_tensor(
                        sin_gathered.clone(),
                        ttnn.bfloat16,
                        ttnn.TILE_LAYOUT,
                        f"sin_gathered_prefill_{seq_len}",
                        device_id,
                    )
                )

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

            attn_masks = []
            for device_id in range(self.num_devices):
                attn_masks.append(
                    as_tensor(
                        attn_mask.clone(), ttnn.bfloat16, ttnn.TILE_LAYOUT, f"attn_mask_prefill_{seq_len}", device_id
                    )
                )

            repeat_shape = (1, self.n_local_heads, 1, 1)
            for i in range(self.num_devices):
                attn_masks[i] = tt_lib.tensor.repeat(
                    attn_masks[i], repeat_shape, output_mem_config=self.model_config["DRAM_MEMCFG"]
                )

        elif self.model_config["LLM_MODE"] == "decode":
            assert seq_len == 1, "Decode mode only supports seq_len=1"
            assert xs[0].shape == (seq_len, 1, batch, self.hidden_size // self.num_devices)

            for device_id in range(self.num_devices):
                xs[device_id] = tt_lib.tensor.interleaved_to_sharded(
                    xs[device_id], sharded_mem_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
                )

            try:
                rot_mat_tt = ttnn.load_tensor(cache_file_name(f"rot_mat_decode_{start_pos}", "BFLOAT16", "TILE"))
                rot_mats = [
                    ttnn.to_device(rot_mat_tt, self.devices[device_id]) for device_id in range(self.num_devices)
                ]
            except (FileNotFoundError, RuntimeError):
                # Use batch=1 because we assume all users use same rot_mat
                rot_mat = get_rotation_mat(self.rot_emb, start_pos, seq_len, batch=1)
                assert rot_mat.size() == (1, 1, self.head_dim, self.head_dim)
                rot_mats = []
                for device_id in range(self.num_devices):
                    rot_mats.append(
                        as_tensor(
                            rot_mat.clone(), ttnn.bfloat16, ttnn.TILE_LAYOUT, f"rot_mat_decode_{start_pos}", device_id
                        )
                    )

            padded_layer_past_len = nearest_32(start_pos + 1)
            try:
                attn_mask_tt = ttnn.load_tensor(cache_file_name(f"attn_mask_decode_{start_pos}", "BFLOAT16", "TILE"))
                attn_masks = [
                    ttnn.to_device(attn_mask_tt, self.devices[device_id]) for device_id in range(self.num_devices)
                ]
            except (FileNotFoundError, RuntimeError):
                attn_mask_shape = (1, seq_len, self.padded_local_heads, padded_layer_past_len)
                attn_mask = torch.zeros(*attn_mask_shape)
                attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min
                attn_masks = []
                for device_id in range(self.num_devices):
                    # BFLOAT16_DTYPE currently pushes faster
                    attn_masks.append(
                        as_tensor(
                            attn_mask.clone(),
                            ttnn.bfloat16,
                            ttnn.TILE_LAYOUT,
                            f"attn_mask_decode_{start_pos}",
                            device_id,
                        )
                    )

            repeat_shape = (batch, 1, 1, 1)
            for i in range(self.num_devices):
                attn_masks[i] = tt_lib.tensor.repeat(
                    attn_masks[i], repeat_shape, output_mem_config=self.model_config["DRAM_MEMCFG"]
                )
            # Put attn_mask on the device with the sharded config
            attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
            if attention_mask_memconfig.is_sharded():
                attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
                attn_mask_shard_shape[-1] = padded_layer_past_len
                attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
            for i in range(self.num_devices):
                attn_masks[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_masks[i], sharded_mem_config=attention_mask_memconfig
                )

        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def forward(
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

        # Convert decoder_output to interleaved
        for i in range(self.num_devices):
            xs[i] = tt_lib.tensor.sharded_to_interleaved(xs[i], output_mem_config=self.model_config["L1_MEMCFG"])

        ## Gather fractured layers output
        if self.emulated:
            xs = tt_all_gather_torch(xs, dim=-1)
        else:
            xs = tt_lib.tensor.all_gather(
                xs,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["L1_MEMCFG"],
            )

        ## Duplicate layernorm
        norm_out_replicated = []
        for i in range(self.num_devices):
            # RMSNorm must execute on sharded input
            xs[i] = tt_lib.tensor.interleaved_to_sharded(
                xs[i], sharded_mem_config=self.model_config["FINAL_ALL_GATHER_OUTPUT_MEMCFG"]
            )
        for i in range(self.num_devices):
            norm_out_replicated.append(
                # In-pace RMSNorm
                tt_lib.operations.primary.rmsnorm(
                    xs[i],
                    self.norm_eps,
                    self.norm_list[i],
                    program_config=self.model_config["LN_F_PROGCFG"],
                    output_mem_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
                    compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
                )
            )

        ### Each device does an LM head fracture
        lm_head_out = []
        for i in range(self.num_devices):
            lm_head_out.append(
                tt_lib.operations.primary.matmul_1d(
                    norm_out_replicated[i],
                    self.lm_head_list[i],
                    program_config=self.model_config["LLAMA3_LM_HEAD_MM_PROGCFG"]
                    if self.llama3
                    else self.model_config["LM_HEAD_MM_PROGCFG"],
                    output_mem_config=self.model_config["DRAM_MEMCFG"],
                    output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
            )
            norm_out_replicated[i].deallocate(True)

        return lm_head_out

    def sharded_rmsnorm(self, xs, eps, norm_list):
        # Do sharded RMS by partial sequence length of 128
        # Input xs[0] is [1, 1, seq_len, 8192]
        seq_len = xs[0].shape[2]
        slice_size = 128
        num_slices = seq_len // slice_size  # we do 128 per iteration (slice), then we concat the result.

        xs_output_cat = []  # this is the output we write to. Initiate as empty tensors
        for i in range(len(xs)):
            xs_output_cat.append(
                torch2tt_tensor(
                    torch.zeros([1, 1, seq_len, self.hidden_size]),
                    self.devices[i],
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
                )
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
            xs_slice = []
            for i in range(self.num_devices):
                xs_slice.append(
                    tt_lib.tensor.interleaved_to_sharded_partial(
                        xs[i],
                        (layernorm_num_cores_x, layernorm_num_cores_y),
                        [layernorm_shard_height_hidden_dim, layernorm_shard_width_hidden_dim],
                        num_slices,  # num_slices
                        slice_i,  # slice_index
                        tt_lib.tensor.TensorMemoryLayout.BLOCK_SHARDED,
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    )
                )

            for i in range(self.num_devices):
                xs_slice[i] = tt_lib.operations.primary.rmsnorm(
                    xs_slice[i],
                    eps,
                    norm_list[i],
                    program_config=self.model_config["LN_F_PROGCFG"],
                    output_mem_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
                    compute_kernel_config=self.model_config["LN_COMPUTE_KERNEL_CONFIG"],
                )

                tt_lib.tensor.sharded_to_interleaved_partial(
                    xs_slice[i],
                    xs_output_cat[i],
                    num_slices,
                    slice_i,
                    self.model_config["DRAM_MEMCFG"],
                )
                xs_slice[i].deallocate(True)
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
        if self.emulated:
            xs = tt_all_gather_torch(xs, dim=-1)
        else:
            xs = tt_lib.tensor.all_gather(
                xs,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["DRAM_MEMCFG"],
            )

        ## Duplicate layernorm
        norm_out_replicated = self.sharded_rmsnorm(xs, self.norm_eps, self.norm_list)

        # Deallocate original input to rmsnorm
        for i in range(self.num_devices):
            xs[i].deallocate(True)

        ### Each device does an LM head fracture
        seq_tiles = norm_out_replicated[0].shape[2] // 32
        self.model_config["LM_HEAD_MM_PROGCFG"] = self.model_config["LM_HEAD_MM_PROGCFG_LAMBDA"](seq_tiles)
        lm_head_out = []
        for i in range(self.num_devices):
            lm_head_out.append(
                tt_lib.operations.primary.matmul(
                    norm_out_replicated[i],
                    self.lm_head_list[i],
                    program_config=self.model_config["LM_HEAD_MM_PROGCFG"],
                    output_mem_config=self.model_config["DRAM_MEMCFG"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
                )
            )
            norm_out_replicated[i].deallocate(True)

        return lm_head_out
