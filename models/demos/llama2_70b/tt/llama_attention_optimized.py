# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import math
import torch
from torch import nn
import tt_lib
import ttnn
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, nearest_32
from models.demos.llama2_70b.tt.llama_common import (
    tt_all_gather_torch,
    generate_rot_emb,
    get_weight_cache_path,
    get_rotation_mat,
    get_rot_transformation_mat,
    gather_cos_sin,
    precompute_freqs,
)


class TtLlamaAttention_optimized(torch.nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        emulated=False,
        load_weights=True,
        cache_path=None,
        kv_cache_dir=None,
        batch_size=None,
        cache_id=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.model_config = model_config
        self.emulated = emulated
        self.batched_attn = self.num_devices == 8

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size if batch_size is None else batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        assert self.num_devices == 4 or self.num_devices == 8
        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices
        self.padded_local_heads = 32

        self.layer_name = f"{base_url}.{layer_num}"

        self.cache_path = cache_path
        self.cache_id = cache_id
        self.kv_cache_dir = kv_cache_dir
        self.k_cache_stem = f"{self.layer_name}.attention.k_cache"
        self.v_cache_stem = f"{self.layer_name}.attention.v_cache"

        self.transformation_mats = [
            torch2tt_tensor(get_rot_transformation_mat(self.head_dim), device) for device in devices
        ]

        if load_weights:
            self.load_weights()
            self.init_kv_cache()

    def set_model_config(self, model_config):
        self.model_config = model_config

    def init_kv_cache(self):
        """
        Generates empty KV cache and pushed to device memory
        """
        self.layer_past_list = []
        for i in range(self.num_devices):
            cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    # self.max_seq_len,
                    2048 + 128,  # Meets benchmarking spec needs
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    # self.max_seq_len,
                    2048 + 128,  # Meets benchmarking spec needs
                    self.head_dim,
                )
            )
            layer_past = [cache_k, cache_v]
            layer_past = [
                torch2tt_tensor(
                    lp,
                    self.devices[i],
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=self.model_config["KV_CACHE_DTYPE"],
                )
                for lp in layer_past
            ]

            # add to the list
            self.layer_past_list.append(layer_past)

    def save_state(self):
        # Save KV cache
        assert self.cache_path is not None, "Cache path is not defined"

        # Best way to do this?
        # torch2tt_tensor, then tt2torch_tensor with Device None, then dump_tensor?

        for i in range(self.num_devices):
            k_cache_path = self.cache_path / self.kv_cache_dir / f"{self.k_cache_stem}_{i}_{self.num_devices}.bin"
            v_cache_path = self.cache_path / self.kv_cache_dir / f"{self.v_cache_stem}_{i}_{self.num_devices}.bin"

            # Does this deallocate from device?
            # TODO: Replace with tensor.cpu()?
            k_cache = tt2torch_tensor(self.layer_past_list[i][0])
            v_cache = tt2torch_tensor(self.layer_past_list[i][1])

            k_cache_tt = torch2tt_tensor(
                k_cache,
                None,
                tt_memory_config=self.model_config["DRAM_MEMCFG"],
                tt_dtype=self.model_config["KV_CACHE_DTYPE"],
            )

            v_cache_tt = torch2tt_tensor(
                v_cache,
                None,
                tt_memory_config=self.model_config["DRAM_MEMCFG"],
                tt_dtype=self.model_config["KV_CACHE_DTYPE"],
            )

            tt_lib.tensor.dump_tensor(str(k_cache_path), k_cache_tt)
            tt_lib.tensor.dump_tensor(str(v_cache_path), v_cache_tt)

            # TODO: Will this throw if already dellocated?
            self.layer_past_list[i][0].deallocate()
            self.layer_past_list[i][1].deallocate()

        del self.layer_past_list

    def load_state(self):
        # Load KV cache
        assert self.cache_path is not None, "Cache path is not defined"
        assert not hasattr(self, "layer_past_list"), "layer_past_list is already an attribute of this object"

        self.layer_past_list = []

        for i in range(self.num_devices):
            k_cache_path = self.cache_path / self.kv_cache_dir / f"{self.k_cache_stem}_{i}_{self.num_devices}.bin"
            v_cache_path = self.cache_path / self.kv_cache_dir / f"{self.v_cache_stem}_{i}_{self.num_devices}.bin"

            if not k_cache_path.exists():
                assert i == 0, "If KV cache doesn't exist, lookup must fail on device 0"
                logger.warning(f"KV cache not found at {k_cache_path}. Initializing empty KV cache")
                self.init_kv_cache()
                return

            k_cache = tt_lib.tensor.load_tensor(str(k_cache_path)).to(self.devices[i], self.model_config["DRAM_MEMCFG"])
            v_cache = tt_lib.tensor.load_tensor(str(v_cache_path)).to(self.devices[i], self.model_config["DRAM_MEMCFG"])

            layer_past = [k_cache, v_cache]

            # add to the list
            self.layer_past_list.append(layer_past)

    def free_weights(self):
        # Free weights
        for i in range(self.num_devices):
            self.qkv_list[i].deallocate(True)
            self.wo_list[i].deallocate(True)
        del self.qkv_list
        del self.wo_list

    def load_weights(self):
        assert not hasattr(self, "qkv_list"), "qkv_list is already an attribute of this object"
        assert not hasattr(self, "wo_list"), "wo_list is already an attribute of this object"
        # Load weights
        wqkv_cache_str = f"{self.layer_name}.attention.wqkv_fused.weight"
        wq_str = f"{self.layer_name}.attention.wq.weight"
        wk_str = f"{self.layer_name}.attention.wk.weight"
        wv_str = f"{self.layer_name}.attention.wv.weight"
        wo_str = f"{self.layer_name}.attention.wo.weight"

        self.qkv_list = []
        self.wo_list = []

        test_cache_path = get_weight_cache_path(
            self.cache_path, wo_str, self.num_devices - 1, self.num_devices, self.cache_id
        )
        if test_cache_path.exists():
            for i in range(self.num_devices):
                tensor_cache_path = get_weight_cache_path(
                    self.cache_path, wqkv_cache_str, i, self.num_devices, self.cache_id
                )
                self.qkv_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )

                tensor_cache_path = get_weight_cache_path(self.cache_path, wo_str, i, self.num_devices, self.cache_id)
                self.wo_list.append(
                    tt_lib.tensor.load_tensor(str(tensor_cache_path)).to(
                        self.devices[i], self.model_config["DRAM_MEMCFG"]
                    )
                )
        else:
            w0_chunks = torch.chunk(torch.transpose(self.state_dict[wo_str], -1, -2), self.num_devices, dim=-1)

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

                qkv_host = torch2tt_tensor(
                    qkv,
                    None,
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=self.model_config["BFP8_DTYPE"],
                )
                self.qkv_list.append(qkv_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, wqkv_cache_str, i, self.num_devices, self.cache_id)),
                    qkv_host,
                )

                ### WO Weights
                wo_host = torch2tt_tensor(
                    w0_chunks[i],
                    None,
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=self.model_config["BFP8_DTYPE"],
                )
                self.wo_list.append(wo_host.to(self.devices[i], self.model_config["DRAM_MEMCFG"]))
                tt_lib.tensor.dump_tensor(
                    str(get_weight_cache_path(self.cache_path, wo_str, i, self.num_devices, self.cache_id)), wo_host
                )

    def prepare_inputs(self, x, start_pos):
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3
        batch, seq_len, hidden_size = x.shape

        if self.model_config["LLM_MODE"] == "prefill":
            assert (
                seq_len % 128 == 0 and seq_len > 0 and seq_len <= 2048
            ), "Prefill mode only supports seqlen as a multiple of 128 up to 2k"
            assert batch == 1, "prefill mode only supports batch size 1"
            x = x.unsqueeze(1)  # [batch, 1, seq_len, hidden_dim]
            cos, sin = precompute_freqs(self.head_dim, self.max_seq_len * 2)
            cos_gathered, sin_gathered = gather_cos_sin(torch.arange(start_pos, start_pos + seq_len), cos, sin)
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
            # cos_gathered: [1, 1, seq_len, head_dim]
            # sin_gathered: [1, 1, seq_len, head_dim]
            # attn_mask: [batch, self.n_heads, seq_len, seq_len]
            assert x.size() == (batch, 1, seq_len, self.hidden_size)
            assert cos_gathered.size() == (1, 1, seq_len, self.head_dim)
            assert sin_gathered.size() == (1, 1, seq_len, self.head_dim)
            assert attn_mask.size() == (batch, self.n_local_heads, seq_len, seq_len)
            xs, cos_gathereds, sin_gathereds, attn_masks = [], [], [], []
            for i in range(self.num_devices):
                xs.append(
                    ttnn.as_tensor(x.clone(), dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT)
                )
                cos_gathereds.append(as_tensor(cos_gathered.clone(), f"cos_gathered_prefill_{seq_len}", i))
                sin_gathereds.append(as_tensor(sin_gathered.clone(), f"sin_gathered_prefill_{seq_len}", i))
                attn_masks.append(as_tensor(attn_mask.clone(), f"attn_mask_prefill_{seq_len}", i))
            rot_mats = [cos_gathereds, sin_gathereds]

            if self.emulated:
                # save l1 space by sharing the same input across "devices"
                for i in range(1, self.num_devices):
                    # x_multichip[i].deallocate(True)
                    rot_mats[0][i].deallocate(True)
                    rot_mats[1][i].deallocate(True)
                    attn_masks[i].deallocate(True)
                    # x_multichip[i] = x_multichip[0]
                    rot_mats[0][i] = rot_mats[0][0]
                    rot_mats[1][i] = rot_mats[1][0]
                    attn_masks[i] = attn_masks[0]

        elif self.model_config["LLM_MODE"] == "decode":
            assert seq_len == 1, "Only supporting decode mode"
            x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]
            rot_mat = generate_rot_emb(self.head_dim, self.max_seq_len * 2)
            rot_mat = get_rotation_mat(rot_mat, start_pos=start_pos, seqlen=seq_len, batch=batch)
            rot_mat = rot_mat[:, :1]

            padded_layer_past_len = nearest_32(start_pos + 1)
            if self.batched_attn:
                attn_mask_shape = (batch, seq_len, self.padded_local_heads, padded_layer_past_len)
            else:
                attn_mask_shape = (seq_len, self.n_local_heads, batch, padded_layer_past_len)
            attn_mask = torch.zeros(*attn_mask_shape)
            # attn_mask[:, :, :, : start_pos + 1] = -1e9
            attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min
            # attn_mask = attn_mask.expand(-1, self.n_local_heads, -1, -1)

            # expected shapes:
            # x: (seq_len, 1, batch, hidden_dim)
            # start_pos: int
            # rot_mat: [1, 1, head_dim, head_dim]
            # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len] or [batch, seq_len, n_heads, padded_layer_past_len]
            assert x.size() == (seq_len, 1, batch, self.hidden_size)
            assert rot_mat.size() == (1, 1, self.head_dim, self.head_dim)
            # assert attn_mask.size() == (seq_len, self.n_local_heads, batch, padded_layer_past_len)
            assert attn_mask.size() == attn_mask_shape
            xs, rot_mats, attn_masks = [], [], []
            for i in range(self.num_devices):
                device = self.devices[i]
                xs.append(
                    torch2tt_tensor(
                        x.clone(),
                        device,
                        tt_dtype=self.model_config["LN_ATTN_OUTPUT_DTYPE"],
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
            # Put attn_mask on the device with the sharded config
            attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
            if attention_mask_memconfig.is_sharded():
                attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
                attn_mask_shard_shape[-1] = padded_layer_past_len
                attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
            for i in range(self.num_devices):
                xs[i] = tt_lib.tensor.interleaved_to_sharded(
                    xs[i], sharded_mem_config=self.model_config["LN_ATTN_OUTPUT_MEMCFG"]
                )
                attn_masks[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_masks[i], sharded_mem_config=attention_mask_memconfig
                )

            if self.emulated:
                # save l1 space by sharing the same input across "devices"
                for i in range(1, self.num_devices):
                    # x_multichip[i].deallocate(True)
                    rot_mats[i].deallocate(True)
                    attn_masks[i].deallocate(True)
                    # x_multichip[i] = x_multichip[0]
                    rot_mats[i] = rot_mats[0]
                    attn_masks[i] = attn_masks[0]

        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def forward(
        self,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_masks: tt_lib.tensor.Tensor,
        user_id: int = 0,
    ) -> tt_lib.tensor.Tensor:
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        # Prefill should have input tensor of shape (1, batch, seqlen, hidden_size)
        if self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        elif self.model_config["LLM_MODE"] == "prefill":
            return self.prefill_forward(xs, rot_mats, attn_masks, user_id)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def decode_forward(
        self,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_masks: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        query_layer, key_layer, value_layer = self.attn_qkv(xs, rot_mats)
        attn_outputs = self.attn_mqa(query_layer, key_layer, value_layer, start_pos, attn_masks)
        return self.attn_selfout(attn_outputs)

    def attn_qkv(
        self,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        # Assume input is already padded to 32, even if the batch size is not 32. Batch size is in self.max_batch_size.
        assert xs[0].shape[2] == 32, "Input tensor must be padded to 32"

        # Reshard
        if self.model_config["LN_ATTN_OUTPUT_MEMCFG"] != self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"]:
            for i in range(len(xs)):
                xs[i] = tt_lib.tensor.reshard(xs[i], self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"])

        # Fused QKV
        fused_query_key_value = []
        for i in range(len(xs)):
            fused_query_key_value.append(
                tt_lib.operations.primary.matmul_1d(
                    xs[i],
                    self.qkv_list[i],
                    program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
                    output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
            )
            xs[i].deallocate(True)
        # TMs
        if self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] != self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]:
            for i in range(len(fused_query_key_value)):
                fused_query_key_value[i] = tt_lib.tensor.reshard(
                    fused_query_key_value[i], self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]
                )

        query_layer = []
        key_layer = []
        value_layer = []
        for i in range(len(fused_query_key_value)):
            (
                q_heads,  # [seqlen, n_local_heads, bsz, head_dim]
                k_heads,  # [seqlen, n_local_kv_heads, bsz, head_dim]
                v_heads,  # [seqlen, n_local_kv_heads, bsz, head_dim]
            ) = tt_lib.tensor.nlp_create_qkv_heads(
                fused_query_key_value[i],
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
            )
            query_layer.append(q_heads)
            key_layer.append(k_heads)
            value_layer.append(v_heads)
            fused_query_key_value[i].deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        for i in range(len(query_layer)):
            query_layer[i] = tt_lib.operations.primary.matmul(
                query_layer[i],
                rot_mats[i],
                program_config=self.model_config["ROT_MAT_Q_MM_PROGCFG"],
                output_mem_config=self.model_config["ROT_MAT_Q_MM_OUTPUT_MEMCFG"],
                compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"]
                # [seqlen, n_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_heads, bsz, head_dim]
            )
        # Pad and transpose Q for batched matmul
        if self.batched_attn:
            for i in range(len(query_layer)):
                # Pad and transpose Q for batched matmul
                query_layer[i] = tt_lib.tensor.sharded_to_interleaved(
                    query_layer[i], output_mem_config=self.model_config["L1_MEMCFG"]
                )
                query_layer[i] = tt_lib.tensor.pad(
                    query_layer[i], [1, self.padded_local_heads, 32, self.head_dim], [0, 0, 0, 0], 0.0
                )
                query_layer[i] = tt_lib.tensor.transpose(
                    query_layer[i],
                    -2,
                    -3,
                )
                query_layer[i] = tt_lib.tensor.reshape(
                    query_layer[i],
                    32,
                    1,
                    self.padded_local_heads,
                    self.head_dim,  # Batch must be in dim 0 to match K cache
                )

        # K Rotary Embeddings
        for i in range(len(key_layer)):
            key_layer[i] = tt_lib.tensor.sharded_to_interleaved(
                key_layer[i], output_mem_config=self.model_config["L1_MEMCFG"]
            )
        for i in range(len(key_layer)):
            key_layer[i] = tt_lib.operations.primary.matmul(
                key_layer[i],
                rot_mats[i],
                output_mem_config=self.model_config["L1_MEMCFG"],
                compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
                # [seqlen, n_kv_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_kv_heads, bsz, head_dim]
            )
            # Shard key_layer for K cache update
            # key_layer[i] = tt_lib.tensor.interleaved_to_sharded(
            #     key_layer[i],
            #     sharded_mem_config=self.model_config["ROT_MAT_K_MM_OUTPUT_MEMCFG"],)

        return query_layer, key_layer, value_layer

    def attn_mqa(
        self,
        query_layer: tt_lib.tensor.Tensor,
        key_layer: tt_lib.tensor.Tensor,
        value_layer: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_masks: tt_lib.tensor.Tensor,
        batch_offset: int = 0,
    ) -> tt_lib.tensor.Tensor:
        padded_layer_past_len = nearest_32(start_pos + 1)

        if self.batched_attn:
            for i in range(len(query_layer)):
                query_layer[i] = tt_lib.tensor.interleaved_to_sharded(
                    query_layer[i], sharded_mem_config=self.model_config["Q_TRANSPOSE_MEMCFG"]
                )

        # K Cache Update
        kv_cache_memcfg = self.model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"]
        if kv_cache_memcfg.is_sharded():
            kv_cache_shard_shape = kv_cache_memcfg.shard_spec.shape
            kv_cache_shard_shape[0] = self.layer_past_list[0][0].shape[1] * padded_layer_past_len
            kv_cache_memcfg.shard_spec.shape = kv_cache_shard_shape
        for i in range(len(key_layer)):
            keys = self.layer_past_list[i][0]
            tt_lib.tensor.update_cache(keys, key_layer[i], start_pos, batch_offset=batch_offset)
            key_layer[i].deallocate(True)
        # key and value layers will have kv_seq_len padded to nearest 32
        for i in range(len(key_layer)):
            keys = self.layer_past_list[i][0]
            key_layer[i] = tt_lib.tensor.unpad(
                keys,
                [0, 0, 0, 0],
                [
                    self.max_batch_size - 1,
                    self.n_local_kv_heads - 1,
                    padded_layer_past_len - 1,
                    self.head_dim - 1,
                ],
                output_mem_config=self.model_config["DRAM_MEMCFG"],
            )
        for i in range(len(key_layer)):
            key_layer[i] = tt_lib.tensor.interleaved_to_sharded(key_layer[i], sharded_mem_config=kv_cache_memcfg)

        # PRE-SOFTMAX MM
        key_layer_transposed = []
        for i in range(len(key_layer)):
            key_layer_transposed.append(
                tt_lib.tensor.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
                )
            )
            key_layer[i].deallocate(True)

        if self.batched_attn:
            attn_prog_config = self.model_config["ATTN_BATCHED_MM_PROGCFG_LAMBDA"](padded_layer_past_len // 32)
            attn_output_memcfg = self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"]
            attn_output_memcfg.shard_spec.shape[1] = padded_layer_past_len
        attn_weights = []
        for i in range(len(query_layer)):
            if self.batched_attn:
                attn_weights.append(
                    tt_lib.operations.primary.matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        program_config=attn_prog_config,
                        output_mem_config=attn_output_memcfg,
                        output_dtype=self.model_config["ATTN_BATCHED_MM_OUTPUT_DTYPE"],
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
            else:
                attn_weights.append(
                    tt_lib.operations.primary.transformers.group_attn_matmul(
                        query_layer[i],
                        key_layer_transposed[i],
                        compute_with_storage_grid_size=self.devices[i].compute_with_storage_grid_size(),
                        output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
                        output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
            query_layer[i].deallocate(True)
            key_layer_transposed[i].deallocate(True)

        # SOFTMAX
        if self.batched_attn:
            softmax_progcfg = self.model_config["BATCHED_SOFTMAX_PROGCFG"]
            softmax_progcfg.block_w = padded_layer_past_len // 32
        else:
            softmax_progcfg = self.model_config["SOFTMAX_PROGCFG"]
            softmax_progcfg.block_w = padded_layer_past_len // 32
        for i in range(len(attn_weights)):
            if self.batched_attn:
                attn_weights[i] = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
                    attn_weights[i],
                    self.scale,
                    attn_masks[i],
                    program_config=self.model_config["BATCHED_SOFTMAX_PROGCFG"],
                    is_causal_mask=True,
                )
            else:
                attn_weights[i] = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
                    attn_weights[i],
                    self.scale,
                    attn_masks[i],
                    program_config=self.model_config["SOFTMAX_PROGCFG"],
                    is_causal_mask=True,
                )

        # V CACHE UPDATE
        for i in range(len(value_layer)):
            values = self.layer_past_list[i][1]
            tt_lib.tensor.update_cache(values, value_layer[i], start_pos, batch_offset=batch_offset)
            value_layer[i].deallocate(True)
        for i in range(len(value_layer)):
            values = self.layer_past_list[i][1]
            value_layer[i] = tt_lib.tensor.unpad(
                values,
                [0, 0, 0, 0],
                [
                    self.max_batch_size - 1,
                    self.n_local_kv_heads - 1,
                    padded_layer_past_len - 1,
                    self.head_dim - 1,
                ],
                output_mem_config=self.model_config["DRAM_MEMCFG"],
            )
        for i in range(len(value_layer)):
            value_layer[i] = tt_lib.tensor.interleaved_to_sharded(value_layer[i], sharded_mem_config=kv_cache_memcfg)

        # POST-SOFTMAX MM
        if self.batched_attn:
            scores_prog_config = self.model_config["SCORES_BATCHED_MM_PROGCFG_LAMBDA"](padded_layer_past_len // 32)
        attn_output = []
        for i in range(len(attn_weights)):
            if self.batched_attn:
                attn_output.append(
                    tt_lib.operations.primary.matmul(
                        attn_weights[i],
                        value_layer[i],
                        program_config=scores_prog_config,
                        output_mem_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
                        output_dtype=self.model_config["SCORES_BATCHED_MM_OUTPUT_DTYPE"],
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
            else:
                attn_output.append(
                    tt_lib.operations.primary.transformers.group_attn_matmul(
                        attn_weights[i],
                        value_layer[i],
                        compute_with_storage_grid_size=self.devices[i].compute_with_storage_grid_size(),
                        output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
                        output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
            attn_weights[i].deallocate(True)
            value_layer[i].deallocate(True)

        # Move sharded attn_output to interleaved here because all_gather only takes in interleaved tensors for now
        if self.batched_attn:
            for i in range(len(attn_output)):
                if self.emulated:
                    attn_output[i] = tt_lib.tensor.sharded_to_interleaved(
                        attn_output[i],
                        output_mem_config=self.model_config["DRAM_MEMCFG"],
                    )
                else:
                    attn_output[i] = tt_lib.tensor.sharded_to_interleaved(
                        attn_output[i],
                        output_mem_config=self.model_config["L1_MEMCFG"],
                    )
        return attn_output

    def attn_selfout(
        self,
        attn_output: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        # ATTENTION SELFOUT
        if self.batched_attn:
            for i in range(len(attn_output)):
                # TRANSPOSE
                # Get batch in dim 1
                attn_output[i] = tt_lib.tensor.reshape(attn_output[i], 1, 32, 32, 128)
                # Get batch in dim 2
                attn_output[i] = tt_lib.tensor.transpose(
                    attn_output[i],
                    -2,
                    -3,
                )
                # UNPAD
                attn_output_shape = attn_output[i].shape
                attn_output[i] = tt_lib.tensor.unpad(
                    attn_output[i],
                    [0, 0, 0, 0],
                    [
                        attn_output_shape[0] - 1,
                        self.n_local_heads - 1,
                        attn_output_shape[2] - 1,
                        attn_output_shape[3] - 1,
                    ],
                    output_mem_config=self.model_config["L1_MEMCFG"],
                )
                # SHARD TO SCORES_TRANSPOSED_OUTPUT_MEMCFG
                attn_output[i] = tt_lib.tensor.interleaved_to_sharded(
                    attn_output[i], sharded_mem_config=self.model_config["SCORES_TRANSPOSED_OUTPUT_MEMCFG"]
                )

        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.tensor.nlp_concat_heads(
                attn_output[i],
                output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            )  # seqlen, 1, batch, hidden_size
        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.tensor.sharded_to_interleaved(
                attn_output[i], output_mem_config=self.model_config["L1_MEMCFG"]
            )
        # All gather input to dense
        if self.emulated:
            attn_output = tt_all_gather_torch(attn_output, dim=-1)
        else:
            attn_output = tt_lib.tensor.all_gather(
                attn_output,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["L1_MEMCFG"],
            )

        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.tensor.interleaved_to_sharded(
                attn_output[i], sharded_mem_config=self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"]
            )
        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.operations.primary.matmul_1d(
                attn_output[i],
                self.wo_list[i],
                program_config=self.model_config["SELFOUT_MM_PROGCFG"],
                output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
                compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
            )  # seqlen, 1, batch, hidden_size

        return attn_output

    def prefill_forward(
        self,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
        attn_masks: tt_lib.tensor.Tensor,
        user_id: int,
    ) -> tt_lib.tensor.Tensor:
        query_layer, key_layer, value_layer = self.prefill_attn_qkv(xs, rot_mats)
        attn_outputs = self.prefill_attn_mqa(query_layer, key_layer, value_layer, attn_masks, user_id)
        return self.prefill_attn_selfout(attn_outputs)

    def prefill_attn_qkv(
        self,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        assert xs[0].shape[0] == 1, "batch must be 1"
        assert xs[0].shape[2] % 128 == 0 and xs[0].shape[2] > 0, "Seqlen must be divisible by 128"
        seq_len = xs[0].shape[2]

        # Fused QKV
        fused_query_key_value = []
        # x: [1, 1, 128, 8192]
        # QKV: [8192,1280]
        for i in range(len(xs)):
            fused_query_key_value.append(
                tt_lib.operations.primary.matmul(
                    xs[i],
                    self.qkv_list[i],
                    program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
                    # output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"], # TODO: Spill to DRAM for now
                    output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                    compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                )
            )
            xs[i].deallocate(True)

        query_layer = []
        key_layer = []
        value_layer = []
        for i in range(len(fused_query_key_value)):
            (
                q_heads,  # [bsz, n_local_heads, seq_len, head_dim]
                k_heads,  # [bsz, n_local_kv_heads, seq_len, head_dim]
                v_heads,  # [bsz, n_local_kv_heads, seq_len, head_dim]
            ) = tt_lib.tensor.nlp_create_qkv_heads(
                fused_query_key_value[i],
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                # output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
            )
            query_layer.append(q_heads)
            key_layer.append(k_heads)
            value_layer.append(v_heads)
            fused_query_key_value[i].deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        query_layer_ret = []
        for i in range(len(query_layer)):
            # query_layer: ttnn.Shape([1, 8, seq_len, 128]) -> [bsz, n_local_heads, seq_len, head_dim]

            query_layer_ret.append(
                self.apply_rotary_prefill(query_layer[i], rot_mats[0][i], rot_mats[1][i], self.transformation_mats[i])
            )
            query_layer[i].deallocate(True)

        # K Rotary Embeddings
        key_layer_ret = []
        for i in range(len(key_layer)):
            # key_layer: ttnn.Shape([1, 1, seq_len, 128])
            key_layer_ret.append(
                self.apply_rotary_prefill(key_layer[i], rot_mats[0][i], rot_mats[1][i], self.transformation_mats[i])
            )
            key_layer[i].deallocate(True)

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
        query_layer: tt_lib.tensor.Tensor,
        key_layer: tt_lib.tensor.Tensor,
        value_layer: tt_lib.tensor.Tensor,
        attn_masks: tt_lib.tensor.Tensor,
        user_id: int,
    ) -> tt_lib.tensor.Tensor:
        seq_len = query_layer[0].shape[2]
        slice_size = 128
        num_slices = seq_len // slice_size  # we do q_lens of 128 per iteration (slice), then we concat the result.
        attn_output_cat = []  # this is the output we write to. Initiate as empty tensors
        for i in range(len(query_layer)):
            attn_output_cat.append(
                torch2tt_tensor(
                    torch.zeros([1, self.n_local_heads, seq_len, self.head_dim]),
                    self.devices[i],
                    tt_memory_config=self.model_config["DRAM_MEMCFG"],
                    tt_dtype=tt_lib.tensor.DataType.BFLOAT16,
                )
            )

        # FILL K CACHE
        for i in range(len(key_layer)):
            keys = self.layer_past_list[i][0]
            tt_lib.tensor.fill_cache(
                keys, tt_lib.tensor.typecast(key_layer[i], self.model_config["BFP8_DTYPE"]), user_id
            )

        # FILL V CACHE
        for i in range(len(value_layer)):
            values = self.layer_past_list[i][1]
            tt_lib.tensor.fill_cache(
                values, tt_lib.tensor.typecast(value_layer[i], self.model_config["BFP8_DTYPE"]), user_id
            )

        """
        (Pdb) query_layer[0].shape: ttnn.Shape([1, 8, 2048, 128])  -> each slice is [1,8,128,128]
        (Pdb) value_layer[0].shape: ttnn.Shape([1, 1, 2048, 128])
        (Pdb) key_layer[0].shape: ttnn.Shape([1, 1, 2048, 128])
        (Pdb) attn_masks[0].shape: ttnn.Shape([1, 8, 2048, 2048])  -> each slice is [1,8,128,2048]
        """

        # PRE-SOFTMAX MM
        key_layer_transposed = []
        for i in range(len(key_layer)):
            key_layer_transposed.append(
                tt_lib.tensor.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    # output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
                )
            )
            key_layer[i].deallocate(True)

        for slice_i in range(num_slices):
            q_slices = []
            attn_mask_slices = []
            for i in range(len(query_layer)):
                q_slices.append(
                    tt_lib.tensor.interleaved_to_sharded_partial(
                        query_layer[i],
                        (8, 4),
                        [32, self.head_dim],  # each slice is [1,8,128,128], we use 32 cores
                        num_slices,  # num_slices
                        slice_i,  # slice_index
                        tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    )
                )
                attn_mask_slices.append(
                    tt_lib.tensor.interleaved_to_sharded_partial(
                        attn_masks[i],
                        (8, 4),
                        [32, seq_len],  # each slice is [1,8,128,128], we use 32 cores
                        num_slices,  # num_slices
                        slice_i,  # slice_index
                        tt_lib.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
                        tt_lib.tensor.ShardOrientation.ROW_MAJOR,
                    )
                )

            attn_weights = []
            for i in range(len(query_layer)):
                attn_weights.append(
                    tt_lib.operations.primary.matmul(
                        q_slices[i],
                        key_layer_transposed[i],
                        program_config=self.model_config["ATTN_BATCHED_MM_PROGCFG"],
                        output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
                        output_dtype=self.model_config["ATTN_BATCHED_MM_OUTPUT_DTYPE"],
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
                q_slices[i].deallocate(True)

            # SOFTMAX
            softmax_progcfg = self.model_config["BATCHED_SOFTMAX_PROGCFG"]
            softmax_progcfg.block_w = seq_len // 32
            for i in range(len(attn_weights)):
                attn_weights[i] = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
                    attn_weights[i],
                    self.scale,
                    attn_mask_slices[i],
                    program_config=self.model_config["BATCHED_SOFTMAX_PROGCFG"],
                    is_causal_mask=True,
                )
                attn_mask_slices[i].deallocate(True)

            # POST-SOFTMAX MM
            attn_output = []
            for i in range(len(attn_weights)):
                attn_output.append(
                    tt_lib.operations.primary.matmul(
                        attn_weights[i],
                        value_layer[i],
                        program_config=self.model_config["SCORES_BATCHED_MM_PROGCFG"],
                        output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
                        output_dtype=self.model_config["SCORES_BATCHED_MM_OUTPUT_DTYPE"],
                        compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
                    )
                )
                attn_weights[i].deallocate(True)

            # write output to attn_output_cat
            for i in range(len(attn_output)):
                tt_lib.tensor.sharded_to_interleaved_partial(
                    attn_output[i],
                    attn_output_cat[i],
                    num_slices,
                    slice_i,
                    self.model_config["DRAM_MEMCFG"],
                )
                attn_output[i].deallocate(True)

        # deallocate keys and values
        for i in range(len(query_layer)):
            query_layer[i].deallocate(True)
            key_layer_transposed[i].deallocate(True)
            value_layer[i].deallocate(True)

        return attn_output_cat

    def prefill_attn_selfout(
        self,
        attn_output: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        # ATTENTION SELFOUT
        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.tensor.nlp_concat_heads(
                attn_output[i],
                output_mem_config=self.model_config["L1_MEMCFG"],
            )  # seqlen, 1, batch, hidden_size

        # for i in range(len(attn_output)):
        #     attn_output[i] = tt_lib.tensor.sharded_to_interleaved(
        #         attn_output[i], output_mem_config=self.model_config["L1_MEMCFG"]
        #     )

        # All gather input to dense
        if self.emulated:
            attn_output = tt_all_gather_torch(attn_output, dim=-1)
        else:
            attn_output = tt_lib.tensor.all_gather(
                attn_output,
                dim=3,
                num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
                output_mem_config=self.model_config["DRAM_MEMCFG"],
            )

        # for i in range(len(attn_output)):
        #     attn_output[i] = tt_lib.tensor.interleaved_to_sharded(
        #         attn_output[i], sharded_mem_config=self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"]
        #     )

        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.operations.primary.matmul(
                attn_output[i],
                self.wo_list[i],
                program_config=self.model_config["SELFOUT_MM_PROGCFG_LAMBDA"](attn_output[i].shape[2] // 32),
                # output_mem_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
                output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
                compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
            )  # seqlen, 1, batch, hidden_size

        return attn_output
