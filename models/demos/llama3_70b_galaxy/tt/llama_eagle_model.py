# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.demos.llama3_70b_galaxy.tt.llama_eagle_decoder import TtEagleDecoder
from models.common.rmsnorm import RMSNorm
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_70b_galaxy.tt.distributed_norm import DistributedNorm
from models.demos.llama3_70b_galaxy.tt.lm_head import LMHead
from models.demos.llama3_70b_galaxy.tt.llama_common import copy_host_to_device
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3_70b_galaxy.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.common.sampling.generator import SamplingGenerator


class TtEagleTransformer(LightweightModule):
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
        decode_mode_only=False,
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
        self.paged_attention_config = paged_attention_config
        self.decode_mode_only = decode_mode_only

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

        # First initialization of decode CCLs and prefetcher
        self.setup_decode()
        self.is_decode_setup = True

        self.mid_layer = TtEagleDecoder(
            args=args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=0,
            n_layers=1,
            transformation_mats=self.trans_mats_dict,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            prefetcher_setup=self.prefetcher_setup,
            tt_ccl=self.tt_ccl,
        )

        self.norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                state_dict=state_dict,
                state_dict_prefix="",
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
            ccl_topology=self.model_config["CCL_TOPOLOGY"],
        )

        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix="lm_head",
            weight_cache_path=weight_cache_path,
            tt_ccl=self.tt_ccl,
            prefetcher_setup=self.prefetcher_setup,
        )

        fc_mem_config = self.model_config["FC_WEIGHTS_MEMCFG"]
        self.fc_weight = ttnn.as_tensor(
            torch.transpose(state_dict[f"fc.weight"], -2, -1).unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat8_b,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-2, -1), mesh_shape=args.cluster_shape),
            layout=ttnn.TILE_LAYOUT,
            memory_config=fc_mem_config,
            cache_file_name=weight_cache_path / f"fc_weight_sharded",
        )

        self.tt_tensors = self.prefetcher_setup.get_input_tensors()

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
            self.sampling = SamplingGenerator(
                args=self.args,
                mesh_device=self.mesh_device,
                tt_ccl=self.tt_ccl,
            )
        else:
            self.tt_ccl = self.tt_ccl_decode

    def prepare_inputs_decode(self, tokens, current_pos, page_table, is_cur_pos_sharded, is_page_table_sharded):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        """
        host_tensors = self.prepare_decode_inputs_host(
            tokens, current_pos, page_table, is_cur_pos_sharded, is_page_table_sharded
        )
        shard_specs = self.prepare_decode_shard_configs(is_cur_pos_sharded, is_page_table_sharded)
        device_inputs = copy_host_to_device(
            host_tensors, mesh_device=self.mesh_device, shard_specs=shard_specs
        )  # Helper function
        return device_inputs

    def prepare_decode_shard_configs(self, is_cur_pos_sharded=False, is_page_table_sharded=False):
        """
        Prepares the sharding configuration for cur_pos and page_table tensors
        """
        cur_pos_memory_config = None
        page_table_memory_config = None
        if is_cur_pos_sharded:
            cur_pos_shard_spec = ttnn.ShardSpec(
                self.args.sub_core_grids,
                (1, self.args.max_batch_size // self.mesh_device.shape[1]),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            cur_pos_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, cur_pos_shard_spec
            )
        if is_page_table_sharded:
            page_table_shard_spec = ttnn.ShardSpec(
                self.args.sub_core_grids,
                (
                    self.args.batch_size_per_device_group,
                    self.paged_attention_config.max_num_blocks // self.args.max_batch_size,
                ),
                ttnn.ShardOrientation.ROW_MAJOR,
            )
            page_table_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, page_table_shard_spec
            )
        return [None, cur_pos_memory_config, None, page_table_memory_config]

    def prepare_decode_inputs_host(
        self, tokens, current_pos, page_table=None, is_cur_pos_sharded=False, is_page_table_sharded=False
    ):
        """
        Inputs are torch tensors or python types. Outputs are ttnn tensors on host.
        NOTE: Tokens and current_pos are padded to batch
        NOTE: if is_cur_pos_sharded is True, current_pos_tt is returned as a device tensor
        NOTE: if is_page_table_sharded is True, page_table is returned as a device tensor
        """
        B = tokens.shape[0]
        # assert current_pos.shape[0] == B, "Batch size mismatch"
        assert (
            B == self.args.max_batch_size
        ), f"Batch size {B} must be equal to max_batch_size {self.args.max_batch_size}"

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
        cur_pos_shard_dim = 0
        if is_cur_pos_sharded:
            cur_pos_shard_dim = 1
            current_pos = current_pos.repeat(self.args.sub_core_grids.num_cores(), 1)
        current_pos_tt = ttnn.from_torch(
            current_pos,
            device=None,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, cur_pos_shard_dim) if (self.args.is_galaxy and B > 1) else (None, None),
                mesh_shape=self.args.cluster_shape,
            ),
        )
        if page_table is not None:
            if is_page_table_sharded:
                page_table_chunks = page_table.split(B // self.args.cluster_shape[1], dim=0)
                repeated_page_table_chunks = [
                    chunk.repeat(self.args.sub_core_grids.num_cores(), 1) for chunk in page_table_chunks
                ]
                page_table = torch.cat(repeated_page_table_chunks, dim=0)

            page_table = ttnn.from_torch(
                page_table,
                device=None,
                dtype=ttnn.uint16 if is_page_table_sharded else ttnn.int32,
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

    def process_output_decode(self, tt_out):
        """
        Input is ttnn device tensor of tokens. Output is the corresponding torch tensor.
        """
        if isinstance(tt_out, list):
            tt_out = tt_out[0]

        tt_out_cpu = tt_out.cpu(blocking=False, cq_id=0)
        return tt_out_cpu, ttnn.record_event(self.mesh_device, 0)

    def _increment_decode_positions_device(self, current_pos, rot_mat_idxs, is_cur_pos_sharded=False):
        ttnn.plus_one(
            current_pos,
            sub_core_grids=self.args.sub_core_grids
            if is_cur_pos_sharded
            else ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
            skip_negative_entries=True,
        )
        ttnn.plus_one(
            rot_mat_idxs,
            sub_core_grids=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        )

    def ttnn_decode_forward(
        self,
        x,
        current_pos,
        rot_mat_idxs,
        page_table=None,
        kv_cache=None,
        tt_out_logits_saved=None,
        is_cur_pos_sharded=False,
        return_logits=False,
        capture_sampling_trace=False,
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
        self._increment_decode_positions_device(current_pos, rot_mat_idxs, is_cur_pos_sharded)

        if return_logits:
            tt_logits = self.tt_ccl.line_all_gather(
                tt_logits[0],
                dim=3,
                num_links=3,
                cluster_axis=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="SAMPLING",
            )

            tt_logits = ttnn.untilize(tt_logits, use_multicore=True, sub_core_grids=self.args.sub_core_grids)

            return tt_logits

        # Save output logits to global python object
        if tt_out_logits_saved is not None:
            tt_out_logits = ttnn.to_torch(
                tt_logits[0],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(3, 1), mesh_shape=self.args.cluster_shape
                ),
            )
            tt_out_logits = tt_out_logits[0, 0, 0, :128256]
            tt_out_logits_saved.copy_(tt_out_logits)

        if capture_sampling_trace:
            return tt_logits

        tt_toks = self.sampling.sample(
            tt_logits[0],
            tt_out_tok=x,
            enable_trace=False,
        )
        return tt_toks

    def switch_mode(self):
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
        batch_size=1,
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
                batch_size=batch_size,
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
