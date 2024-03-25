# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
from tqdm import tqdm
import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, nearest_32, profiler
from models.demos.llama2_70b.tt.llama_decoder_optimized import TtLlamaDecoder_optimized
from models.demos.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    generate_rot_emb,
    get_weight_cache_path,
    get_rotation_mat,
    get_rotation_mat_prefill,
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
        n_layers_per_group=None,
        cache_path=None,
        start_layer_idx=0,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.model_config = model_config

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.n_kv_heads = configuration.n_kv_heads
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.emulated = emulated
        self.batched_attn = self.num_devices == 8
        self.padded_local_heads = 32
        self.llm_mode = model_config["LLM_MODE"]

        self.norm_eps = configuration.norm_eps
        self.vocab_size = configuration.vocab_size

        self.n_layers = n_layers
        self.n_layers_per_group = n_layers_per_group if n_layers_per_group else n_layers
        assert self.n_layers % self.n_layers_per_group == 0, "n_layers must be divisible by n_layers_per_group"
        self.num_layer_groups = self.n_layers // self.n_layers_per_group

        # If n_layers_per_group is not equal to n_layers, we need to reload weights
        self.do_reload = self.n_layers_per_group != self.n_layers

        # Need unique directory for each run's KV cache... unix time
        self.cache_path = cache_path
        kv_unique_dir = str(int(time.time()))
        kv_cache_path = cache_path / kv_unique_dir
        # Ensure kv_cache_path exists
        print("Creating Layers", flush=True)
        kv_cache_path.mkdir(parents=True, exist_ok=True)
        self.layers = [
            TtLlamaDecoder_optimized(
                devices,
                state_dict,
                base_url,
                start_layer_idx + i,
                model_config,
                configuration,
                batch,
                emulated=emulated,
                cache_path=cache_path,
                kv_cache_dir=kv_unique_dir,
            )
            for i in tqdm(range(n_layers))
        ]

        print("Done creating layers", flush=True)

        self.rot_emb = generate_rot_emb(self.head_dim, self.max_seq_len * 2)

        emb_str = "tok_embeddings.weight"
        self.tok_embeddings = torch.nn.Embedding(configuration.vocab_size, self.hidden_size)
        self.tok_embeddings.weight = torch.nn.Parameter(self.state_dict[emb_str])
        self.load_weights()

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

    def free_layers(self, start_layer, end_layer):
        # Save layer for each layer in layer_group
        if self.do_reload:
            for layer in self.layers[start_layer:end_layer]:
                layer.free_layer()

    def load_layers(self, start_layer, end_layer):
        # Load layer for each layer in layer_group
        if self.do_reload:
            for layer in self.layers[start_layer:end_layer]:
                layer.load_layer()

    def prepare_inputs(self, inp_ids, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        x: (batch, seq)
        start_pos: int

        returns
        xs: [(seq, batch, hidden_dim)] * num_devices
        start_pos: int
        rot_mats: [(1, batch, head_dim, head_dim)] * num_devices
        attn_masks: [(seq, n_local_heads, batch, max_seq_len)] * num_devices
        """
        x = self.tok_embeddings(inp_ids)  # [batch, seq, hidden]
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)

        if self.llm_mode == "prefill":
            assert (
                seq_len % 128 == 0 and seq_len > 0 and seq_len <= 2048
            ), "Prefill mode only supports seqlen as a multiple of 128 up to 2k"
            assert batch == 1, "prefill mode only supports batch size 1"
            x = x.unsqueeze(1)  # [batch, 1, seq_len, hidden_dim]
            rot_mat = get_rotation_mat_prefill(rot_mat=self.rot_emb, start_pos=start_pos, seqlen=seq_len, batch=batch)
            attn_mask = torch.full((seq_len, seq_len), torch.finfo(torch.float32).min)
            attn_mask = torch.triu(attn_mask, diagonal=1)
            attn_mask = attn_mask.expand(batch, self.n_local_heads, -1, -1)

            as_tensor = lambda tensor, name, device_id: ttnn.as_tensor(
                tensor,
                dtype=ttnn.bfloat16,
                device=self.devices[device_id],
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=get_weight_cache_path(self.cache_path, name, device_id, self.num_devices),
            )
            # expected shapes:
            # x: [batch, 1, seq_len, hidden_dim]
            # start_pos: int
            # rot_mat: [batch, seq_len, head_dim, head_dim]
            # attn_mask: [batch, self.n_heads, seq_len, seq_len]
            assert x.size() == (batch, 1, seq_len, self.hidden_size)
            assert rot_mat.size() == (batch, seq_len, self.head_dim, self.head_dim)
            assert attn_mask.size() == (batch, self.n_local_heads, seq_len, seq_len)

            x_fractured = torch.chunk(x, self.num_devices, dim=-1)
            xs, rot_mats, attn_masks = [], [], []
            for i in range(self.num_devices):
                xs.append(
                    ttnn.as_tensor(x_fractured[i], dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT)
                )
                rot_mats.append(as_tensor(rot_mat.clone(), f"rot_mat_prefill_{seq_len}", i))
                attn_masks.append(as_tensor(attn_mask.clone(), f"attn_mask_prefill_{seq_len}", i))
        elif self.llm_mode == "decode":
            assert seq_len == 1, "Decode mode only supports seq_len=1"
            x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]
            assert x.size() == (seq_len, 1, batch, self.hidden_size)
            x_fractured = torch.chunk(x, self.num_devices, dim=-1)
            xs = []
            for i in range(self.num_devices):
                xs.append(
                    torch2tt_tensor(
                        x_fractured[i],
                        self.devices[i],
                        tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                    )
                )
            for i in range(self.num_devices):
                xs[i] = tt_lib.tensor.interleaved_to_sharded(
                    xs[i], sharded_mem_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
                )

            rot_mat = get_rotation_mat(
                rot_mat=self.rot_emb,
                start_pos=start_pos,
                seqlen=seq_len,
                batch=1,  # use batch=1 because we assume all users use same rot_mat
            )
            assert rot_mat.size() == (1, 1, self.head_dim, self.head_dim)
            rot_mats = []
            for i in range(self.num_devices):
                rot_mats.append(
                    torch2tt_tensor(
                        rot_mat.clone(),
                        self.devices[i],
                        tt_memory_config=self.model_config["ROT_MAT_MEMCFG"],  # TODO: Put on L1 instead of DRAM
                        tt_dtype=self.model_config["ROT_MAT_DTYPE"],
                    )
                )

            padded_layer_past_len = nearest_32(start_pos + 1)
            if self.batched_attn:
                attn_mask_shape = (1, seq_len, self.padded_local_heads, padded_layer_past_len)
            else:
                attn_mask_shape = (seq_len, 1, batch, padded_layer_past_len)
            attn_mask = torch.zeros(*attn_mask_shape)
            attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min
            assert attn_mask.size() == attn_mask_shape
            attn_masks = []
            for i in range(self.num_devices):
                attn_masks.append(
                    torch2tt_tensor(
                        attn_mask.clone(),
                        self.devices[i],
                        tt_dtype=self.model_config["ATTN_MASK_DTYPE"],  # BFLOAT16_DTYPE currently pushes faster
                    )
                )
            if self.batched_attn:
                repeat_shape = (batch, 1, 1, 1)
            else:
                repeat_shape = (1, self.n_local_heads, 1, 1)

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
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> tt_lib.tensor.Tensor:
        if self.llm_mode == "prefill":
            return self.prefill_forward(xs, rot_mats, start_pos, attn_masks)
        elif self.llm_mode == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        else:
            raise ValueError(f"Unknown llm_mode: {self.llm_mode}")

    def decode_forward(
        self,
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> tt_lib.tensor.Tensor:
        ### Run all layers
        for i in range(self.num_layer_groups):
            start_layer = i * self.n_layers_per_group
            end_layer = start_layer + self.n_layers_per_group

            # Prologue: Load weights and KV cache
            self.load_layers(start_layer, end_layer)

            for layer in self.layers[start_layer:end_layer]:
                xs = layer(xs, rot_mats, start_pos, attn_masks)  # xs is sharded

            # Epilogue: Save KV cache to disk and free weights
            self.free_layers(start_layer, end_layer)

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
                tt_lib.operations.primary.rmsnorm(
                    xs[i],
                    self.norm_eps,
                    self.norm_list[i],
                    program_config=self.model_config["LN_F_PROGCFG"],
                    output_mem_config=self.model_config["LN_F_OUTPUT_MEMCFG"],
                )
            )
            # xs[i].deallocate(True)

        ### Each device does an LM head fracture
        lm_head_out = []
        for i in range(self.num_devices):
            lm_head_out.append(
                tt_lib.operations.primary.matmul_1d(
                    norm_out_replicated[i],
                    self.lm_head_list[i],
                    program_config=self.model_config["LM_HEAD_MM_PROGCFG"],
                    # output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                    # output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
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
        xs: list,
        rot_mats: list,
        start_pos: int,
        attn_masks: list,
    ) -> tt_lib.tensor.Tensor:
        ### Run all layers
        for i in range(self.num_layer_groups):
            start_layer = i * self.n_layers_per_group
            end_layer = start_layer + self.n_layers_per_group

            # Prologue: Load weights and KV cache
            self.load_layers(start_layer, end_layer)

            for layer in self.layers[start_layer:end_layer]:
                xs = layer(xs, rot_mats, start_pos, attn_masks)  # xs is sharded

            # Epilogue: Save KV cache to disk and free weights
            self.free_layers(start_layer, end_layer)

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
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_FP16_ACC_CONFIG"],
                )
            )
            norm_out_replicated[i].deallocate(True)

        return lm_head_out
