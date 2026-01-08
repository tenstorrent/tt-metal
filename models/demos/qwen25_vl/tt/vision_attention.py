# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen25_vl.tt.vision_rmsnorm import RMSNorm
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup


class VisionAttention(LightweightModule):
    def __init__(self, *args, **kwargs):
        kwargs["causal_mask"] = False
        self.__init(*args, **kwargs)

    def forward(
        self,
        x,
        rot_mats,
        cu_seqlens,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
    ):
        return self.forward_prefill(
            x,
            cu_seqlens=cu_seqlens,
            rot_mats=rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=None,
        )

    def __init(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        # use_paged_kv_cache=False,
        causal_mask=True,
        # use_kv_cache=True,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        self.causal_mask = causal_mask
        # self.use_kv_cache = use_kv_cache
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.ccl_dtype = configuration.ccl_dtype
        self.num_reduce_scatter_links = configuration.num_reduce_scatter_links
        self.num_all_gather_links = configuration.num_all_gather_links
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size

        self.num_devices_per_group = 1  # [INFO] each device runs a copy of the vision model

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group
        self.padded_head_dim = math.ceil(self.head_dim / self.tile_size) * self.tile_size

        self.dtype = dtype

        self.max_seq_len = configuration.max_seq_len
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats

        self.model_config = configuration.get_model_config()
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip
        self.activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        self.wqkv_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WQKV
        )
        self.wo_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WO
        )
        self.kv_cache_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.KV_CACHE
        )
        self.li_qkv_decode_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_DECODE, configuration=configuration
        )
        self.sdpa_decode_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_DECODE, configuration=configuration
        )
        self.li_o_decode_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_DECODE, configuration=configuration
        )
        self.sdpa_prefill_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_PREFILL, configuration=configuration
        )
        self.li_qkv_prefill_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_PREFILL, configuration=configuration
        )
        self.li_o_prefill_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_PREFILL, configuration=configuration
        )

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Initialize bias tensors as None
        self.wqkv_bias_decode = None
        self.wqkv_bias_prefill = None
        self.wo_bias_decode = None
        self.wo_bias_prefill = None

        # Create combined QKV bias if present in state dict
        if f"{wq_str}.bias" in self.state_dict:
            # Helper function to reshape and pad bias chunk if needed
            def pad_bias_chunk(b):
                if self.head_dim != self.padded_head_dim:
                    # Reshape to separate head dimensions
                    b = b.reshape(self.n_local_heads, self.head_dim)
                    # Pad the head_dim dimension
                    b = torch.nn.functional.pad(b, (0, self.padded_head_dim - self.head_dim))
                    # Reshape back to 1D
                    result = b.reshape(-1)
                    return result
                else:
                    return b

            qkv_bias = torch.concat(
                [
                    torch.concat(
                        [
                            pad_bias_chunk(
                                torch.chunk(self.state_dict[f"{wq_str}.bias"], self.num_devices_per_group)[i]
                            ),
                            pad_bias_chunk(
                                torch.chunk(self.state_dict[f"{wk_str}.bias"], self.num_devices_per_group)[i]
                            ),
                            pad_bias_chunk(
                                torch.chunk(self.state_dict[f"{wv_str}.bias"], self.num_devices_per_group)[i]
                            ),
                        ],
                        dim=-1,
                    )
                    for i in range(self.num_devices_per_group)
                ],
                dim=-1,
            )
            # Prefill can use broadcasting on the bias add so wants a 1d tensor
            self.wqkv_bias_prefill = ttnn.as_tensor(
                qkv_bias,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("wqkv_bias_prefill_sharded"),
            )

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices_per_group == 0
        assert self.n_kv_heads % self.num_devices_per_group == 0
        assert configuration.qkv_size % self.num_devices_per_group == 0
        assert configuration.dim % self.num_devices_per_group == 0

        # wqkv: 4096 x 3072 (2 devices): width-sharded on 12 banks, 3072 over 12 banks.
        # wqkv_mem_config = configuration.create_dram_sharded_mem_config(
        #     configuration.dim, configuration.qkv_size // configuration.num_devices
        # )

        qkv_list = []
        for i in range(self.num_devices_per_group):
            # Chunk weights
            wq_selected = torch.chunk(self.state_dict[f"{wq_str}.weight"], self.num_devices_per_group, dim=0)[i]
            wk_selected = torch.chunk(self.state_dict[f"{wk_str}.weight"], self.num_devices_per_group, dim=0)[i]
            wv_selected = torch.chunk(self.state_dict[f"{wv_str}.weight"], self.num_devices_per_group, dim=0)[i]

            # If head_dim needs padding
            if self.head_dim != self.padded_head_dim:
                # Helper function to reshape and pad weights
                def pad_weight(w):
                    # Reshape to separate head dimensions
                    w = w.reshape(self.n_local_heads, self.head_dim, -1)
                    # Pad the head_dim dimension
                    w = torch.nn.functional.pad(w, (0, 0, 0, self.padded_head_dim - self.head_dim))
                    # Reshape back to 2D
                    result = w.reshape(self.n_local_heads * self.padded_head_dim, -1)
                    return result

                wq_selected = pad_weight(wq_selected)
                wk_selected = pad_weight(wk_selected)
                wv_selected = pad_weight(wv_selected)

            # Transpose the selected chunks
            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)
        # qkv_cat.shape = [1, 1, 1280, 4608])

        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=cache_name("wqkv"),
        )

        def norm_reshard(x, norm, mode):
            """Hack until RMSNorm supports height-sharded output config"""
            if mode == "decode":
                mem_cfg = x.memory_config()
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=x.dtype)
            x = norm(x, mode)
            if mode == "decode":
                x = ttnn.to_memory_config(x, mem_cfg, dtype=x.dtype)
            return x

        if f"{q_norm_str}.weight" in self.state_dict:
            fn_q_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=self.state_dict,
                state_dict_prefix=None,  # we already prefix q_norm_str
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=q_norm_str,
                is_distributed=False,
                sharded_program_config=None,  # FIXME: add height-sharded support. self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=None,  # FIXME: add height-sharded support. self.model_config["CREATE_QKV_DECODE_SHARD"]
            )
            self.q_norm = lambda x, mode: norm_reshard(x, fn_q_norm, mode)
        else:
            self.q_norm = lambda x, mode: x

        if f"{k_norm_str}.weight" in self.state_dict:
            fn_k_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=self.state_dict,
                state_dict_prefix=None,  # we already prefix k_norm_str
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=k_norm_str,
                is_distributed=False,
                sharded_program_config=None,  # FIXME: add height-sharded support. self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=None,  # FIXME: add height-sharded support. self.model_config["CREATE_QKV_DECODE_SHARD"],
            )
            self.k_norm = lambda x, mode: norm_reshard(x, fn_k_norm, mode)
        else:
            self.k_norm = lambda x, mode: x

        # For ring topology we can use all gather matmul for wo
        # self.use_fused_all_gather_matmul = self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]

        # FIXME: workaround until nlp_concat_heads correctly supports sub-tile head dims
        # We are going to pad the input dim of the output weights with zeros in the places
        # that nlp_concat_heads inserts garbage values
        if self.head_dim != self.padded_head_dim:
            # note that torch weights are already transposed to have input in last dim
            pt_wo_t = self.state_dict[f"{wo_str}.weight"]
            # pt_wo_t.shape = [1280, 1280]
            heads = pt_wo_t.reshape(-1, self.n_local_heads, self.head_dim)
            # heads.shape = [-1, 8, 80]
            heads = torch.nn.functional.pad(
                heads, (0, self.padded_head_dim - self.head_dim)
            )  # tail-pad last dim with 0
            pt_wo = heads.reshape(1, 1, -1, self.n_local_heads * self.padded_head_dim).transpose(-1, -2)
            # pt_wo.shape = [1, 1, 768, 2560]

        else:
            pt_wo = self.state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        # wo_mem_config = configuration.create_dram_sharded_mem_config(
        #     pt_wo.shape[-2] // configuration.num_devices, pt_wo.shape[-1]
        # )

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            cache_file_name=cache_name("wo"),
        )
        # self.wo.shape = [1, 1, 384, 2560] each device of N300 sharded on dim=2

        if f"{wo_str}.bias" in self.state_dict:
            # Prefill can use broadcasting on the bias add so wants a 1d tensor
            self.wo_bias_prefill = ttnn.as_tensor(
                self.state_dict[f"{wo_str}.bias"],
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                dtype=self.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("wo_bias_prefill_sharded"),
            )

        self.scale = self.head_dim**-0.5

        dram_shard_grid_width = 8
        target_device_shape = (1, 1)  # each 1x1 device runs a vision model
        self.xqkv_prefill_progcfg = lambda seq_len: ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
            out_subblock_h=1,  # Must be divisible by per_core_M
            out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
            per_core_M=max(
                1, 8 if seq_len >= self.MAX_QKV_MM_SEQ_LEN else math.ceil(seq_len / self.tile_size / 8)  # 8 rows
            ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
            per_core_N=math.ceil(
                configuration.qkv_size / target_device_shape[1] / 32 / dram_shard_grid_width
            ),  # N / TILE_WIDTH / grid width
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=seq_len <= self.MAX_QKV_MM_SEQ_LEN,
        )

    def forward_prefill(
        self,
        x_11SH,
        cu_seqlens,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            if seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {self.MAX_QKV_MM_SEQ_LEN}")
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=self.activation_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
            program_config=self.xqkv_prefill_progcfg(seq_len),
        )

        # FIXME: surely ttnn.linear bias should work?
        if self.wqkv_bias_prefill is not None:
            xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

        # split qkv into heads
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode="prefill")
        k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode="prefill")

        # last_five_unpadded = lambda x, mesh_device: first_five(x, mesh_device, start=-(96 - 80) - 5, end=-(96 - 80))

        ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###

        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)

        # workaround until rotary embeddings support sub-tile head dims
        if self.head_dim != self.padded_head_dim:
            pad_head_dim = lambda x, v: ttnn.pad(
                x, (x.shape[0], x.shape[1], x.shape[2], self.padded_head_dim), (0, 0, 0, 0), v
            )
            # pad with cos = 1, sin = 0
            rot_mats = [pad_head_dim(rot_mats[0], 1.0), pad_head_dim(rot_mats[1], 0.0)]
            # print(f'{rot_mats[0].shape=}')

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=self.kv_cache_dtype)
        ttnn.deallocate(k_heads_1KSD)

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(v_heads_1VSD)

        # SDPA
        if chunk_start_idx is not None:
            # todo)) the use of chunked SDPA seem to imply the use of kv_cache (as keys_BKSD and values_BKSD are associated with kv_cache); Is this correct?
            attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                q_heads_1QSD_8b,
                keys_BKSD,
                values_BKSD,
                page_table,
                chunk_start_idx,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len, chunk_start_idx),
            )
        else:
            attn_output_84SD = ttnn.transformer.windowed_scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                cu_seqlens,
                scale=self.scale,
                compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len),
            )

        # deallocate keys and values
        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.padded_head_dim])

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(attn_output_1QSD)
        # reshaping long sequence to matmul fit on device
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
            dtype=self.activation_dtype or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["VISION_WO_PREFILL_PROGCFG"](seq_len),
        )
        # FIXME: surely ttnn.linear bias should work?
        if self.wo_bias_prefill is not None:
            output_11SH = output_11SH + self.wo_bias_prefill

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        return output_11SH
