# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, nearest_32
from models.demos.llama2_70b.tt.llama_decoder_optimized import TtLlamaDecoder_optimized
from models.demos.llama2_70b.tt.llama_common import generate_rot_emb, gather_rotary_emb, tt_all_gather_torch


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

        emb_str = "tok_embeddings.weight"
        norm_str = "norm.weight"
        lm_head_str = "output.weight"

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
            )
            for i in range(n_layers)
        ]

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
            logger.info(f"Loading MODEL weights from cache")
            for i in range(self.num_devices):
                tensor_cache_path = get_weight_cache_path(self.cache_path, norm_str, i, self.num_devices)
                self.norm_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["LN_F_WEIGHTS_MEMCFG"]
                    )
                )

                tensor_cache_path = get_weight_cache_path(self.cache_path, lm_head_str, i, self.num_devices)
                self.lm_head_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"]
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
                self.norm_list.append(output_norm_host.to(self.devices[i], self.model_config["LN_F_WEIGHTS_MEMCFG"]))
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, norm_str, i, self.num_devices)),
                    output_norm_host,
                )

                lm_head_host = torch2tt_tensor(
                    padded_lm_head_chunks[i],
                    None,
                    tt_memory_config=self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["LM_HEAD_MM_WEIGHTS_DTYPE"],
                )
                self.lm_head_list.append(
                    lm_head_host.to(self.devices[i], self.model_config["LM_HEAD_MM_WEIGHTS_MEMCFG"])
                )
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
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]

        position_ids = torch.ones(seq_len, batch, dtype=torch.long) * start_pos
        rot_mat = gather_rotary_emb(self.rot_emb, position_ids)[:, :1]

        padded_layer_past_len = nearest_32(start_pos + 1)
        attn_mask = torch.zeros(seq_len, 1, batch, padded_layer_past_len)
        attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min
        attn_mask = attn_mask.expand(-1, self.n_local_heads, -1, -1)

        # expected shapes:
        # x: (seq_len, 1, batch, hidden_dim)
        # start_pos: int
        # rot_mat: [1, 1, head_dim, head_dim]
        # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len]
        assert x.size() == (seq_len, 1, batch, self.hidden_size)
        assert rot_mat.size() == (1, 1, self.head_dim, self.head_dim)
        assert attn_mask.size() == (seq_len, self.n_local_heads, batch, padded_layer_past_len)

        x_fractured = torch.chunk(x, self.num_devices, dim=-1)
        xs, rot_mats, attn_masks = [], [], []
        # Put attn_mask on the device with the sharded config
        attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
        if attention_mask_memconfig.is_sharded():
            attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
            attn_mask_shard_shape[-1] = padded_layer_past_len
            attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
        for i in range(self.num_devices):
            device = self.devices[i]
            xs.append(
                torch2tt_tensor(
                    x_fractured[i],
                    device,
                    tt_dtype=self.model_config["WORD_EMBEDDING_OUTPUT_DTYPE"],
                )
            )
            rot_mats.append(
                torch2tt_tensor(
                    rot_mat.clone(),
                    device,
                    tt_memory_config=self.model_config["ROT_MAT_MEMCFG"],  # TODO: Put on L1 instead of DRAM
                    tt_dtype=self.model_config["ROT_MAT_DTYPE"],
                )
            )
            attn_masks.append(
                torch2tt_tensor(
                    attn_mask.clone(),
                    device,
                    tt_dtype=self.model_config["ATTN_MASK_DTYPE"],
                )
            )
        for i in range(self.num_devices):
            xs[i] = tt_lib.tensor.interleaved_to_sharded(
                xs[i], sharded_mem_config=self.model_config["WORD_EMBEDDING_OUTPUT_MEMCFG"]
            )
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
                xs[i], sharded_mem_config=self.model_config["DECODER_ALL_GATHER_OUTPUT_MEMCFG"]
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
                    output_mem_config=self.model_config["LM_HEAD_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["LM_HEAD_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
            )
            norm_out_replicated[i].deallocate(True)

        return lm_head_out
