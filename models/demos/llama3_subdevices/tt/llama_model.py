# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import gc
from tqdm import tqdm
from models.demos.llama3_subdevices.tt.llama_decoder import TtTransformerBlock
from models.common.rmsnorm import RMSNorm
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_subdevices.tt.distributed_norm import DistributedNorm
from models.demos.llama3_subdevices.tt.lm_head import LMHead
from models.demos.llama3_subdevices.tt.llama_common import copy_host_to_device, get_prefill_rot_mat
from models.demos.llama3_subdevices.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3_subdevices.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL
from models.demos.llama3_subdevices.tt.sampling import TTSampling


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
        enable_prefetcher_performance_mode=False,
        mode="decode",
        allocate_prefill_buffers=True,
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
        self.enable_prefetcher_performance_mode = enable_prefetcher_performance_mode
        state_dict_prefix = args.get_state_dict_prefix("", None)
        self.allocate_prefill_buffers = allocate_prefill_buffers

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

        self.is_prefill_setup = False
        self.is_decode_setup = False
        self.prefetcher_setup = None
        self.mesh_sub_device_manager_id_decode = None
        self.mesh_sub_device_manager_id_prefill = None

        if mode == "decode":
            self.setup_decode()
            self.is_decode_setup = True
        else:
            self.setup_prefill()
            self.is_prefill_setup = True

        self.layers = [
            TtTransformerBlock(
                args=args,
                mesh_device=mesh_device,
                dtype=dtype,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=i,
                n_layers=self.n_layers,
                transformation_mats=self.trans_mats_dict,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                prefetcher_setup=self.prefetcher_setup,
                tt_ccl=self.tt_ccl,
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
            tt_ccl=self.tt_ccl,
        )

        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            tt_ccl=self.tt_ccl,
            prefetcher_setup=self.prefetcher_setup,
        )
        if mode == "decode":
            self.tt_tensors = self.prefetcher_setup.get_input_tensors()
        self.tt_rot_mats_prefill = None

    def setup_prefill(self, mesh_sub_device_manager_id_prefill=None):
        self.prefetcher_setup = TtLlamaPrefetcherSetup(
            self.mesh_device,
            n_tensors=0,
            n_layers=self.n_layers,
            mode="prefill",
            mesh_sub_device_manager_id_prefill=mesh_sub_device_manager_id_prefill,
            save_tensor_addresses=True,
        )
        self.mesh_sub_device_manager_id_prefill = self.prefetcher_setup.mesh_sub_device_manager_id_prefill
        self.mesh_device.set_sub_device_stall_group([self.prefetcher_setup.worker_sub_device_id])
        if mesh_sub_device_manager_id_prefill is None:
            self.tt_ccl = TT_CCL(
                self.mesh_device,
                self.args,
                self.prefetcher_setup.worker_sub_device_id,
                mode="prefill",
                allocate_prefill_buffers=self.allocate_prefill_buffers,
            )
        else:
            self.tt_ccl = self.tt_ccl_prefill

    def setup_decode(self, mesh_sub_device_manager_id_decode=None):
        self.prefetcher_setup = TtLlamaPrefetcherSetup(
            self.mesh_device,
            n_tensors=5,
            n_layers=self.n_layers,
            mesh_sub_device_manager_id_decode=mesh_sub_device_manager_id_decode,
            save_tensor_addresses=True,
        )
        self.mesh_sub_device_manager_id_decode = self.prefetcher_setup.mesh_sub_device_manager_id_decode
        self.mesh_device.set_sub_device_stall_group(
            [self.prefetcher_setup.prefetcher_sub_device_id, self.prefetcher_setup.worker_sub_device_id]
        )
        if mesh_sub_device_manager_id_decode is None:
            self.tt_ccl = TT_CCL(self.mesh_device, self.args, self.prefetcher_setup.worker_sub_device_id)
            self.tt_sampling = TTSampling(
                args=self.args,
                mesh_device=self.mesh_device,
                sampling_params={"top_k": 1, "top_p": 0.00, "seed": 42},
                tt_ccl=self.tt_ccl,
            )
        else:
            self.tt_ccl = self.tt_ccl_decode

    def prepare_prefill_inputs_host(
        self, tokens, user_id=0, page_table=None, chunk_page_table=None, tt_rot_mats_prefill=None
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        """

        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Slice the rot mats to the prefill seqlen
        if tt_rot_mats_prefill is None and self.tt_rot_mats_prefill is None:
            tt_rot_mats_prefill = get_prefill_rot_mat(
                self.args.head_dim,
                self.args.max_seq_len,
                self.mesh_device,
                seq_len=self.args.max_seq_len,
                scale_factor=self.args.rope_scaling_factor,
            )
            self.tt_rot_mats_prefill = tt_rot_mats_prefill
        else:
            tt_rot_mats_prefill = self.tt_rot_mats_prefill

        if page_table is not None:
            # we only want to update the kv cache on the 8 devices (every fourth device starting at user_id//8 ) for a given user_id
            # we are setting the page table to -1 for all other devices to skip the update
            page_table_padded = torch.ones((128, page_table.shape[1]), dtype=torch.int32) * -1
            page_table_padded[user_id // 8 * 32 : (user_id // 8 + 1) * 32, :] = page_table
            tt_page_table = ttnn.from_torch(
                page_table_padded,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, 0), mesh_shape=self.args.cluster_shape
                ),
            )

        else:
            tt_page_table = None

        if chunk_page_table is not None:
            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_chunk_page_table = None

        user_id = ttnn.from_torch(
            torch.tensor([user_id], dtype=torch.int32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return tokens, user_id, tt_page_table, tt_chunk_page_table

    def transform_prefill_inputs_device(
        self,
        tokens,
        user_id,
        page_table=None,
        chunk_page_table=None,
    ):
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        return tt_tokens, user_id, page_table, chunk_page_table

    def prepare_inputs_prefill(
        self, tokens, user_id=0, page_table=None, chunk_page_table=None, tt_rot_mats_prefill=None
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        """
        host_inputs = self.prepare_prefill_inputs_host(
            tokens, user_id, page_table, chunk_page_table, tt_rot_mats_prefill
        )
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)  # Helper function
        transformed_device_inputs = self.transform_prefill_inputs_device(*device_inputs)
        return transformed_device_inputs

    def prepare_inputs_decode(self, *inputs):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        """
        host_inputs = self.prepare_decode_inputs_host(*inputs)
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)  # Helper function
        return device_inputs

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """
        Inputs are torch tensors or python types. Outputs are ttnn tensors on host.
        NOTE: Tokens and current_pos are padded to batch
        """
        B = tokens.shape[0]
        # assert current_pos.shape[0] == B, "Batch size mismatch"
        assert B == self.args.max_batch_size, f"Batch size must be equal to max_batch_size {self.args.max_batch_size}"

        # Necessary padding to be full tile sized when on device
        tokens = torch.nn.functional.pad(tokens.view(-1), (0, 32 - len(tokens)), "constant", 0)
        tokens = ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tokens = ttnn.unsqueeze_to_4D(tokens)
        rot_current_pos = torch.maximum(
            current_pos, torch.tensor(0, dtype=torch.int64)
        )  # Ensure position indices are non-negative
        rope_idxs = self.rope_setup.get_rm_rot_idxs(rot_current_pos, on_host=True)
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
        # print("tokens", tokens.shape, tokens.memory_config)
        tt_rot_mats = self.rope_setup.get_rm_rot_mats(rope_idxs)
        tt_tokens = self.embd(tokens)
        return tt_tokens, current_pos, tt_rot_mats, page_table

    def process_output_prefill(self, tt_out, last_token_idx):
        """
        Input is ttnn device tensor of logits. Output is torch logits tensor.
        NOTE: In this model, prefill always uses get_last_token
        """
        x, _ = self.norm(tt_out, res=None, mode="prefill")

        x = x[:, :, last_token_idx : last_token_idx + 1, :]

        tt_logits = self.lm_head(x, None, mode="prefill")

        # Gather the output across all devices and untilize the tensor (for argmax)
        tt_logits = self.tt_ccl.line_all_gather(
            tt_logits[0],
            dim=3,
            num_links=3,
            cluster_axis=0,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            # buffer_key="SAMPLING",
        )

        tt_logits = ttnn.untilize(tt_logits, use_multicore=True)
        tt_logits = ttnn.reshape(
            tt_logits,
            ttnn.Shape([1, 1, 1, tt_logits.shape[-1]]),
            ttnn.Shape([1, 1, tt_logits.shape[-2], tt_logits.shape[-1]]),
        )

        tt_out = ttnn.argmax(tt_logits, dim=3, keepdim=True, use_multicore=True)
        if isinstance(tt_out, list):
            tt_out = tt_out[0]

        logits = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()[0, 0, 0, :1]
        return logits

    def process_output_decode(self, tt_out, B, S=1):
        """
        Input is ttnn device tensor of tokens. Output is the corresponding torch tensor.
        """
        if isinstance(tt_out, list):
            tt_out = tt_out[0]
        tt_out_cpu = tt_out.cpu(blocking=True, cq_id=0)

        tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt_out_cpu)[0])[0, 0, 0, :]

        return tt_out

    def ttnn_prefill_forward(
        self,
        x,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        rot_mats=None,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        tt_logits = self.forward(
            x,
            current_pos=None,
            rot_mats=rot_mats if rot_mats is not None else self.tt_rot_mats_prefill,
            user_id=user_id,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            get_last_token=get_last_token,
            kv_cache=kv_cache,
        )
        return tt_logits

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs,
        page_table=None,
        kv_cache=None,
    ):
        """
        This method will take device tensors and any other args to run forward.
        It returns ttnn device tensors.
        """
        rot_mats = self.rope_setup.get_rm_rot_mats(rot_mat_idxs)
        x_embd = self.embd(x)
        tt_logits = self.forward(
            x_embd,
            current_pos,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table,
            kv_cache=kv_cache,
        )

        # sampling
        tt_logits = self.tt_sampling(tt_logits[0], x)

        ttnn.plus_one(
            current_pos,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )
        ttnn.plus_one(
            rot_mat_idxs,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )
        return tt_logits

    def switch_mode(self, mode):
        if mode == "decode":
            if self.is_prefill_setup:
                self.tt_ccl.close()
                self.tt_ccl_prefill = self.tt_ccl
                self.is_prefill_setup = False

            if self.is_decode_setup is False:
                self.setup_decode(self.mesh_sub_device_manager_id_decode)
                self.is_decode_setup = True
                # prefetch
                for layer in self.layers:
                    layer.prefetch(self.prefetcher_setup, self.tt_ccl)
                self.norm.tt_ccl = self.tt_ccl
                self.lm_head.tt_ccl = self.tt_ccl
                self.tt_tensors = self.prefetcher_setup.get_input_tensors()
                # Re-create global CB for decode (if it was not already created)
                self.prefetcher_setup.create_global_cb()

        else:
            if self.is_decode_setup:
                del self.prefetcher_setup.global_circular_buffer
                gc.collect()  # This will also release the traces (inside generator.py)
                self.tt_ccl.close()
                self.tt_ccl_decode = self.tt_ccl
                self.is_decode_setup = False

            if self.is_prefill_setup is False:
                self.setup_prefill(self.mesh_sub_device_manager_id_prefill)
                self.is_prefill_setup = True
                for layer in self.layers:
                    layer.prefetch(self.prefetcher_setup, self.tt_ccl)
                self.norm.tt_ccl = self.tt_ccl
                self.lm_head.tt_ccl = self.tt_ccl

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
        if mode == "decode":
            self.prefetcher_setup.create_global_cb()
            garbage_tensor = ttnn.dram_prefetcher(
                self.tt_tensors,
                num_layers=self.n_layers,
                global_cb=self.prefetcher_setup.global_circular_buffer,
                enable_performance_mode=self.enable_prefetcher_performance_mode,
            )
            self.mesh_device.set_sub_device_stall_group([self.prefetcher_setup.worker_sub_device_id])

        if mode == "decode" and not self.args.is_galaxy:
            x = ttnn.to_memory_config(x, self.model_config["DECODE_RESIDUAL_MEMCFG"])

        h = None
        # x needs to be in bfloat16_b as it gets reused as the residual tensor
        for i, layer in enumerate(self.layers):
            x, h = layer(
                x,
                h,
                current_pos,
                rot_mats,
                user_id,
                mode,
                page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache[i] if kv_cache is not None else None,
            )
        # ttnn.deallocate(h)
        if mode == "decode":
            ttnn.deallocate(garbage_tensor)

            # Pre-allocated output of AllReduce in LM Head to avoid memory cloberring
            self.tt_ccl.tt_lm_head_buffer_l1 = ttnn.to_memory_config(
                self.tt_ccl.tt_lm_head_buffer, self.tt_ccl.lm_head_buffer_mem_cfg
            )

        if mode == "prefill":
            return x
        # Output norm
        x, res = self.norm(x, res=None, mode=mode)

        if get_last_token != -1:
            x = x[:, :, get_last_token:, :]

        return self.lm_head(x, None if mode == "prefill" else self.prefetcher_setup.worker_sub_device_id, mode=mode)

    def __del__(self):
        self.tt_ccl.close()

        # clear global saved addresses
        global global_tt_tensor_address
        global_tt_tensor_address = None
