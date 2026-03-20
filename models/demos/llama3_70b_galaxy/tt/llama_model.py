# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import os
from pathlib import Path
from tqdm import tqdm
from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
from models.common.rmsnorm import RMSNorm
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_70b_galaxy.tt.distributed_norm import DistributedNorm
from models.demos.llama3_70b_galaxy.tt.lm_head import LMHead
from models.demos.llama3_70b_galaxy.tt.llama_common import copy_host_to_device, get_prefill_rot_mat
from models.tt_transformers.tt.rope import get_rot_mats
from models.demos.llama3_70b_galaxy.tt.llama_rope import TtLlamaRotarySetup
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_embedding import TtLlamaEmbedding
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.common.sampling.generator import SamplingGenerator


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
        self._bitmask_packed_width = self.args.padded_vocab_size // 32
        if self.args.padded_vocab_size % 32 != 0:
            raise ValueError("Bitmask application requires padded vocab size to be a multiple of 32")
        self._bitmask_shape = (self.args.max_batch_size, self._bitmask_packed_width)
        self._bitmask = ttnn.from_torch(
            torch.zeros(self._bitmask_shape, dtype=torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-1, None), mesh_shape=self.args.cluster_shape),
        )
        self._active_bitmask = None
        self._last_host_bitmask = None
        self._debug_bitmask = os.getenv("TT_DEBUG_BITMASK", "0") == "1"
        self._debug_bitmask_step = 0
        self._debug_bitmask_max_steps = int(os.getenv("TT_DEBUG_BITMASK_MAX_STEPS", "32"))
        self._debug_bitmask_dump = os.getenv("TT_DEBUG_BITMASK_DUMP", "0") == "1"
        self._debug_bitmask_dump_dir = Path(os.getenv("TT_DEBUG_BITMASK_DUMP_DIR", "/tmp/tt_debug_bitmask"))
        self._debug_bitmask_dump_dir.mkdir(parents=True, exist_ok=True)
        # Keep bit shifts identical on every mesh device.
        self.bitmask_arange = ttnn.from_torch(
            torch.arange(32, dtype=torch.int32),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

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
            tt_ccl=self.tt_ccl,
            ccl_topology=self.model_config["CCL_TOPOLOGY"],
        )

        state_dict_prefix = args.get_state_dict_prefix("", None)

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
        if not self.decode_mode_only:  # demo_decode.py uses decode mode only. In this case avoid initializing prefill
            # First initialization of prefill CCLs and prefetcher. It needs to be after initialization of layers, norm and lm_head since those switch modes as well
            # This initialization is required to avoid race condition due to all buffers and semaphores not being allocated at initialization
            self.switch_mode("prefill")
            if not self.args.is_qwen:
                self.setup_prefill()
            self.is_prefill_setup = True

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
            is_qwen=self.args.is_qwen,
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
                is_qwen=True if self.args.is_qwen else False,
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
            is_qwen=self.args.is_qwen,
        )
        self.mesh_sub_device_manager_id_decode = self.prefetcher_setup.mesh_sub_device_manager_id_decode
        self.mesh_device.set_sub_device_stall_group(
            [self.prefetcher_setup.prefetcher_sub_device_id, self.prefetcher_setup.worker_sub_device_id]
        )
        if mesh_sub_device_manager_id_decode is None:
            self.tt_ccl = TT_CCL(
                self.mesh_device,
                self.args,
                self.prefetcher_setup.worker_sub_device_id,
                is_qwen=True if self.args.is_qwen else False,
            )
            self.sampling = SamplingGenerator(
                args=self.args,
                mesh_device=self.mesh_device,
                tt_ccl=self.tt_ccl,
            )
        else:
            self.tt_ccl = self.tt_ccl_decode

    def prepare_prefill_inputs_host(
        self, tokens, user_id=0, page_table=None, chunk_page_table=None, tt_rot_mats_prefill=None, batch_size=1
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
            if self.args.is_qwen:
                tt_rot_mats_prefill = get_rot_mats(
                    head_dim=self.args.head_dim,
                    device=self.mesh_device,
                    seq_len=self.args.max_seq_len,
                    theta=self.args.rope_theta,
                    rope_scaling=self.args.rope_scaling_factor,
                )
            else:
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
            if batch_size > 1:
                assert batch_size == 32, "batch_size must be 32 for batched prefill"
                # we only want to update the kv cache for 8 users per 4 devices
                # pad with -1 for the seqlen of all other users
                devices = 4
                batch_size_per_device = batch_size // devices
                page_table_padded = torch.ones((devices, page_table.shape[1] * batch_size), dtype=torch.int32) * -1
                for i in range(devices):
                    page_table_padded[
                        i,
                        (i * batch_size_per_device)
                        * page_table.shape[1] : (i + 1)
                        * batch_size_per_device
                        * page_table.shape[1],
                    ] = page_table[i * batch_size_per_device : (i + 1) * batch_size_per_device, :].reshape(1, -1)

            else:
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
        self, tokens, user_id=0, page_table=None, chunk_page_table=None, tt_rot_mats_prefill=None, batch_size=1
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        """
        host_inputs = self.prepare_prefill_inputs_host(
            tokens, user_id, page_table, chunk_page_table, tt_rot_mats_prefill, batch_size
        )
        device_inputs = copy_host_to_device(host_inputs, mesh_device=self.mesh_device)  # Helper function
        transformed_device_inputs = self.transform_prefill_inputs_device(*device_inputs)
        return transformed_device_inputs

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
                dims=(None, cur_pos_shard_dim) if (B > 1) else (None, None),
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
                    dims=(None, -2) if (B > 1) else (None, None),
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
        tt_rot_mats = self.rope_setup.get_rm_rot_mats(rope_idxs)
        tt_tokens = self.embd(tokens)
        return tt_tokens, current_pos, tt_rot_mats, page_table

    def process_output_prefill_logits(self, tt_out, last_token_idx):
        """
        Process prefill output to get logits tensor for on-device sampling.
        Returns logits in the same format as decode (before all-gather), suitable for sampling module.
        For non-batched prefill, returns single user logits. For batched prefill, returns list of logits.
        """
        x, _ = self.norm(tt_out, res=None, mode="prefill")
        if isinstance(last_token_idx, list):
            # batched prefill: split the output tensor by the batch size and do the processing for each batch in a loop
            batch_size = len(last_token_idx)
            x_split = ttnn.split(x, x.shape[-2] // batch_size, dim=2)
        else:
            x_split = [x]

        logits_list = []
        for i, x in enumerate(x_split):
            if isinstance(last_token_idx, list):
                last_token_idx_i = last_token_idx[i]
            else:
                last_token_idx_i = last_token_idx
            x = x[:, :, last_token_idx_i : last_token_idx_i + 1, :]
            # lm_head returns logits in sharded format (same as decode before all-gather)
            tt_logits = self.lm_head(x, None, mode="prefill")
            tt_logits = tt_logits[0]
            tt_logits = ttnn.reshape(
                tt_logits,
                ttnn.Shape([1, 1, 1, tt_logits.shape[-1]]),
                ttnn.Shape([1, 1, tt_logits.shape[-2], tt_logits.shape[-1]]),
            )
            logits_list.append(tt_logits)

        return logits_list

    def process_output_prefill(self, tt_out, last_token_idx, tt_out_logits_saved=None):
        """
        Input is ttnn device tensor of logits. Output is torch logits or tokens tensor.
        NOTE: In this model, prefill always uses get_last_token
        """
        x, _ = self.norm(tt_out, res=None, mode="prefill")
        if isinstance(last_token_idx, list):
            # batched prefill: split the output tensor by the batch size and do the processing for each batch in a loop
            batch_size = len(last_token_idx)
            x_split = ttnn.split(x, x.shape[-2] // batch_size, dim=2)
        else:
            x_split = [x]

        toks_list = []
        for i, x in enumerate(x_split):
            if isinstance(last_token_idx, list):
                last_token_idx_i = last_token_idx[i]
            else:
                last_token_idx_i = last_token_idx
            x = x[:, :, last_token_idx_i : last_token_idx_i + 1, :]
            tt_logits = self.lm_head(x, None, mode="prefill")
            # Gather the output across all devices and untilize the tensor (for argmax)
            tt_logits = self.tt_ccl.line_all_gather(
                tt_logits[0],
                dim=3,
                num_links=3,
                cluster_axis=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="SAMPLING",
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

            toks = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()[0, 0, 0, :1]
            toks_list.append(toks)

        if tt_out_logits_saved is not None:
            # make sure tt_out_logits_saved is mutable
            logits_saved = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[0]).float()[0, 0, :, :]
            tt_out_logits_saved.copy_(logits_saved)

        return toks_list if isinstance(last_token_idx, list) else toks

    def process_output_decode(self, tt_out):
        """
        Input is ttnn device tensor of tokens. Output is the corresponding torch tensor.
        """
        if isinstance(tt_out, list):
            tt_out = tt_out[0]

        if isinstance(tt_out, tuple):
            tt_log_probs = tt_out[1]
            tt_out = tt_out[0]
            tt_out_cpu = tt_out.cpu(blocking=False, cq_id=0)

            if tt_log_probs is not None:
                tt_log_probs_cpu = tt_log_probs.cpu(blocking=False, cq_id=0)
            else:
                tt_log_probs_cpu = None
        else:
            tt_out_cpu = tt_out.cpu(blocking=False, cq_id=0)
            tt_log_probs_cpu = None
        return tt_out_cpu, tt_log_probs_cpu, ttnn.record_event(self.mesh_device, 0)

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
        batch_size=1,
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
            get_last_token=get_last_token,  # ignored with mode=="prefill"
            kv_cache=kv_cache,
            batch_size=batch_size,
        )
        return tt_logits

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

    def unpack_bitmask(self, bitmask):
        self.mesh_device.reset_sub_device_stall_group()
        op_kwargs = {"sub_core_grids": self.args.sub_core_grids} if self.args.sub_core_grids is not None else {}
        batch_dim, vocab_dim = bitmask.shape
        mesh_rows, mesh_cols = self.args.cluster_shape
        arange32 = torch.arange(32, dtype=torch.int32)

        # Read back packed input from device to build per-shard torch references
        packed_shards_host = [ttnn.to_torch(s).to(torch.int32) for s in ttnn.get_device_tensors(bitmask)]
        B = packed_shards_host[0].shape[0]
        shard_w = packed_shards_host[0].shape[-1]
        ref_packed = [packed_shards_host[r * mesh_cols] for r in range(mesh_rows)]
        ref_reshaped = [s.reshape(B, shard_w, 1) for s in ref_packed]
        ref_shifted = [torch.bitwise_right_shift(s, arange32[None, None, :]) for s in ref_reshaped]
        ref_anded = [(s & 1) for s in ref_shifted]
        ref_flat = [s.reshape(B, shard_w * 32) for s in ref_anded]
        ref_float = [s.to(torch.float32) for s in ref_flat]
        ref_sub1 = [(s - 1.0) for s in ref_float]
        ref_penalty = [(s * 1e9) for s in ref_sub1]
        _stage_refs = {
            "0_packed_input": (ref_packed, torch.int32),
            "1_reshape_3d": (ref_reshaped, torch.int32),
            "2_right_shift": (ref_shifted, torch.int32),
            "3_bitwise_and": (ref_anded, torch.int32),
            "4_reshape_2d": (ref_flat, torch.int32),
            "5_to_tile": (ref_flat, torch.int32),
            "6_add_neg1": (ref_sub1, torch.float32),
            "7_mul_1e9": (ref_penalty, torch.float32),
        }

        def _check(name, tt_tensor):
            ref_per_row, dtype = _stage_refs[name]
            try:
                shards = ttnn.get_device_tensors(tt_tensor)
                failures = []
                for idx, shard in enumerate(shards):
                    row = idx // mesh_cols
                    actual = ttnn.to_torch(shard).to(dtype)
                    expected = ref_per_row[row].to(dtype)
                    common = tuple(min(a, e) for a, e in zip(actual.shape, expected.shape))
                    a = actual[tuple(slice(0, s) for s in common)]
                    e = expected[tuple(slice(0, s) for s in common)]
                    if not torch.equal(a, e):
                        mm = a != e
                        count = int(mm.sum().item())
                        first = torch.nonzero(mm, as_tuple=False)[0].tolist()
                        failures.append(
                            f"  dev={idx} row={row} col={idx % mesh_cols} "
                            f"first_mismatch={first} got={a[tuple(first)].item()} "
                            f"expected={e[tuple(first)].item()} "
                            f"mismatches={count}/{a.numel()} "
                            f"actual_shape={tuple(actual.shape)} ref_shape={tuple(expected.shape)}"
                        )
                if failures:
                    print(f"[STAGE CHECK] {name} FAILED ({len(failures)}/{len(shards)} shards):")
                    for f in failures:
                        print(f)
                else:
                    print(f"[STAGE CHECK] {name} PASSED ({len(shards)} shards)")
            except Exception as ex:
                import traceback

                print(f"[STAGE CHECK] {name} ERROR: {ex}\n{traceback.format_exc()}")

        # _check("0_packed_input", bitmask)

        bitmask_to_broadcast = ttnn.reshape(bitmask, (batch_dim, vocab_dim, 1), **op_kwargs)
        _check("1_reshape_3d", bitmask_to_broadcast)

        broadcast_unpacked = ttnn.bitwise_right_shift(bitmask_to_broadcast, self.bitmask_arange)
        _check("2_right_shift", broadcast_unpacked)

        broadcast_unpacked = ttnn.bitwise_and(broadcast_unpacked, 1)
        _check("3_bitwise_and", broadcast_unpacked)

        unpacked_bitmask = ttnn.reshape(broadcast_unpacked, (batch_dim, -1), **op_kwargs)
        # _check("4_reshape_2d", unpacked_bitmask)

        converted_bitmask = ttnn.to_layout(unpacked_bitmask, ttnn.TILE_LAYOUT, **op_kwargs)
        # _check("5_to_tile", converted_bitmask)

        result = ttnn.add(converted_bitmask, -1.0, dtype=ttnn.float32, **op_kwargs)
        # _check("6_add_neg1", result)

        result = ttnn.multiply(result, 1e9, **op_kwargs)
        # _check("7_mul_1e9", result)

        if hasattr(self.prefetcher_setup, "prefetcher_sub_device_id"):
            self.mesh_device.set_sub_device_stall_group([self.prefetcher_setup.worker_sub_device_id])
        return result

    def start_bitmask_to_device(self, bitmask):
        target = self._bitmask
        packed_target = (self.args.padded_vocab_size + 31) // 32
        packed_in = bitmask.shape[-1]
        if packed_in < packed_target:
            pad = torch.full(
                (bitmask.shape[0], packed_target - packed_in),
                fill_value=0,
                dtype=bitmask.dtype,
                device=bitmask.device,
            )
            bitmask = torch.cat([bitmask, pad], dim=-1)
        elif packed_in > packed_target:
            raise ValueError(f"Bitmask has too many packed bits: {packed_in} > {packed_target}")
        bitmask_tt = ttnn.from_torch(
            bitmask,
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(-1, None), mesh_shape=self.args.cluster_shape),
        )
        # Keep transfer path simple and deterministic: copy on default queue and synchronize.
        ttnn.copy_host_to_device_tensor(bitmask_tt, target)
        ttnn.synchronize_device(self.mesh_device)
        self._active_bitmask = target

    def complete_bitmask_to_device(self):
        # Transfer is synchronous in start_bitmask_to_device.
        return

    def apply_bitmask_to_logits(self, tt_logits):
        if self._active_bitmask is None:
            return tt_logits
        with ttnn.trace_allocation_safe_scope(self.mesh_device):
            bitmask_unpacked = self.unpack_bitmask(self._active_bitmask)
            ttnn.add_(
                tt_logits,
                bitmask_unpacked,
                **({"sub_core_grids": self.args.sub_core_grids} if self.args.sub_core_grids is not None else {}),
            )
            bitmask_unpacked.deallocate(True)
        return tt_logits

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
        capture_sampling_trace=False,  # If true, return logits so sampling can be traced elsewhere
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

            return tt_logits, None

        tt_logits = tt_logits[0]

        # TODO because we don't apply bitmask here, it means running with capture_sampling_trace=False is not supported
        # This is not a problem as that option doens't work already due to trace invalidation.
        assert capture_sampling_trace, "capture_sampling_trace=False is deprecated"
        # Save output logits to global python object
        if tt_out_logits_saved is not None:
            tt_out_logits = ttnn.to_torch(
                tt_logits,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(3, 1), mesh_shape=self.args.cluster_shape
                ),
            )
            tt_out_logits = tt_out_logits[0, 0, 0, : self.args.vocab_size]

            tt_out_logits_saved.copy_(tt_out_logits)

        if capture_sampling_trace:
            return tt_logits

        tt_toks, tt_log_probs = self.sampling.sample(
            tt_logits,
            tt_out_tok=x,
            enable_trace=False,
        )
        return tt_toks, tt_log_probs

    def _debug_should_log_bitmask(self):
        return self._debug_bitmask and self._debug_bitmask_step < self._debug_bitmask_max_steps

    def _torch_reference_unpack_local(self, packed_local: torch.Tensor) -> torch.Tensor:
        structured_output_arange = torch.arange(32, dtype=torch.int32, device=packed_local.device)
        unpacked = torch.bitwise_right_shift(packed_local[:, :, None], structured_output_arange[None, None, :]) & 1
        unpacked = unpacked.reshape(packed_local.shape[0], -1).to(torch.float32)
        return torch.where(unpacked != 0, torch.tensor(0.0), torch.tensor(-1e9))

    def _sanity_check_unpacked_bitmask(self, packed_device_tensor, unpacked01_device_tensor, penalty_device_tensor):
        packed_shards = ttnn.get_device_tensors(packed_device_tensor)
        unpacked01_shards = ttnn.get_device_tensors(unpacked01_device_tensor)
        penalty_shards = ttnn.get_device_tensors(penalty_device_tensor)
        if len(packed_shards) != len(unpacked01_shards) or len(packed_shards) != len(penalty_shards):
            raise AssertionError(
                f"Bitmask sanity check failed: shard count mismatch packed={len(packed_shards)} "
                f"unpacked01={len(unpacked01_shards)} penalty={len(penalty_shards)}"
            )

        mesh_rows, mesh_cols = self.args.cluster_shape
        num_shards = len(packed_shards)
        if mesh_rows * mesh_cols != num_shards:
            raise AssertionError(
                f"Bitmask sanity check failed: cluster_shape={self.args.cluster_shape} "
                f"does not match shard count={num_shards}"
            )

        packed_locals = [ttnn.to_torch(x).to(torch.int32) for x in packed_shards]
        unpacked01_locals = [ttnn.to_torch(x).to(torch.int32) for x in unpacked01_shards]
        penalty_locals = [ttnn.to_torch(x).to(torch.float32) for x in penalty_shards]

        def _dump_sanity_tensors(tag: str, tensors: dict[str, torch.Tensor]):
            try:
                out_dir = self._debug_bitmask_dump_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                base = f"step{self._debug_bitmask_step:04d}_sanity_{tag}"
                for name, tensor in tensors.items():
                    torch.save(tensor.detach().cpu(), out_dir / f"{base}_{name}.pt")
                print(
                    f"[TT_DEBUG_BITMASK] step={self._debug_bitmask_step} dumped sanity tensors "
                    f"for '{tag}' to {out_dir}"
                )
            except Exception as e:
                print(f"[TT_DEBUG_BITMASK] step={self._debug_bitmask_step} sanity dump failed for '{tag}': {e}")

        def _first_mismatch(a: torch.Tensor, b: torch.Tensor):
            if a.shape != b.shape:
                return None, tuple(a.shape), tuple(b.shape)
            mismatch = a != b
            if not torch.any(mismatch):
                return None
            idx = torch.nonzero(mismatch, as_tuple=False)[0].tolist()
            mismatch_count = int(mismatch.sum().item())
            total_count = int(mismatch.numel())
            return idx, a[tuple(idx)].item(), b[tuple(idx)].item(), mismatch_count, total_count

        # For ShardTensor2dMesh(dims=(-1, None), mesh_shape=(rows, cols)):
        # - row axis shards last tensor dim
        # - col axis replicates. Validate replication and pick col=0 as canonical shard per row.
        def _reassemble_with_replica_check(locals_list: list[torch.Tensor], name: str) -> torch.Tensor:
            canonical = []
            for row in range(mesh_rows):
                base_idx = row * mesh_cols
                ref = locals_list[base_idx]
                for col in range(1, mesh_cols):
                    cur_idx = base_idx + col
                    cur = locals_list[cur_idx]
                    if cur.shape != ref.shape:
                        raise AssertionError(
                            f"Bitmask sanity check failed ({name}): replica shape mismatch "
                            f"row={row} col={col} got={tuple(cur.shape)} expected={tuple(ref.shape)}"
                        )
                    mm = _first_mismatch(cur, ref)
                    if mm is not None:
                        idx, got, exp, mismatch_count, total_count = mm
                        if idx is None:
                            _dump_sanity_tensors(
                                f"{name}_replica_shape_row{row}_col{col}",
                                {"ref": ref, "cur": cur},
                            )
                            raise AssertionError(
                                f"Bitmask sanity check failed ({name}): replica shape mismatch "
                                f"row={row} col={col} got={got} expected={exp}"
                            )
                        _dump_sanity_tensors(
                            f"{name}_replica_row{row}_col{col}",
                            {"ref": ref, "cur": cur},
                        )
                        raise AssertionError(
                            f"Bitmask sanity check failed ({name}): replica mismatch row={row} col={col} idx={idx} "
                            f"got={got} expected={exp} mismatches={mismatch_count}/{total_count}"
                        )
                canonical.append(ref)
            return torch.cat(canonical, dim=1)

        packed_global = _reassemble_with_replica_check(packed_locals, "packed")
        unpacked01_global = _reassemble_with_replica_check(unpacked01_locals, "unpacked01")
        penalty_global = _reassemble_with_replica_check(penalty_locals, "penalty")

        if self._last_host_bitmask is not None:
            host = self._last_host_bitmask.to(torch.int32)
            mm = _first_mismatch(packed_global, host)
            if mm is not None:
                idx, got, exp, mismatch_count, total_count = mm
                if idx is None:
                    _dump_sanity_tensors(
                        "packed_global_shape",
                        {"packed_global": packed_global, "host": host},
                    )
                    raise AssertionError(
                        f"Bitmask sanity check failed (packed global) shape mismatch got={got} expected={exp}"
                    )
                _dump_sanity_tensors(
                    "packed_global",
                    {"packed_global": packed_global, "host": host},
                )
                raise AssertionError(
                    f"Bitmask sanity check failed (packed global) idx={idx} got={got} expected={exp} "
                    f"mismatches={mismatch_count}/{total_count}"
                )

        structured_output_arange = torch.arange(32, dtype=torch.int32, device=packed_global.device)
        expected01_global = (
            torch.bitwise_right_shift(packed_global[:, :, None], structured_output_arange[None, None, :]) & 1
        ).reshape(packed_global.shape[0], -1)
        mm01 = _first_mismatch(unpacked01_global.to(torch.int32), expected01_global.to(torch.int32))
        if mm01 is not None:
            idx, got, exp, mismatch_count, total_count = mm01
            if idx is None:
                _dump_sanity_tensors(
                    "unpacked01_global_shape",
                    {
                        "unpacked01_global": unpacked01_global.to(torch.int32),
                        "expected01_global": expected01_global.to(torch.int32),
                    },
                )
                raise AssertionError(
                    f"Bitmask sanity check failed (unpacked01 global) shape mismatch got={got} expected={exp}"
                )
            _dump_sanity_tensors(
                "unpacked01_global",
                {
                    "packed_global": packed_global.to(torch.int32),
                    "unpacked01_global": unpacked01_global.to(torch.int32),
                    "expected01_global": expected01_global.to(torch.int32),
                },
            )
            raise AssertionError(
                f"Bitmask sanity check failed (unpacked01 global) idx={idx} got={got} expected={exp} "
                f"mismatches={mismatch_count}/{total_count}"
            )

        expected_penalty_global = torch.where(
            expected01_global != 0,
            torch.tensor(0.0, dtype=torch.float32, device=expected01_global.device),
            torch.tensor(-1e9, dtype=torch.float32, device=expected01_global.device),
        )
        mmp = _first_mismatch(penalty_global.to(torch.float32), expected_penalty_global.to(torch.float32))
        if mmp is not None:
            idx, got, exp, mismatch_count, total_count = mmp
            if idx is None:
                _dump_sanity_tensors(
                    "penalty_global_shape",
                    {
                        "penalty_global": penalty_global.to(torch.float32),
                        "expected_penalty_global": expected_penalty_global.to(torch.float32),
                    },
                )
                raise AssertionError(
                    f"Bitmask sanity check failed (penalty global) shape mismatch got={got} expected={exp}"
                )
            _dump_sanity_tensors(
                "penalty_global",
                {
                    "penalty_global": penalty_global.to(torch.float32),
                    "expected_penalty_global": expected_penalty_global.to(torch.float32),
                    "unpacked01_global": unpacked01_global.to(torch.int32),
                },
            )
            raise AssertionError(
                f"Bitmask sanity check failed (penalty global) idx={idx} got={got} expected={exp} "
                f"mismatches={mismatch_count}/{total_count}"
            )

    def _debug_log_device_tensor_slice(self, name, tensor, width=16):
        if not self._debug_should_log_bitmask():
            return
        try:
            t = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
            flat = t.reshape(-1)
            sample = flat[: min(width, flat.numel())].tolist()
            print(
                f"[TT_DEBUG_BITMASK] step={self._debug_bitmask_step} {name}: "
                f"shape={tuple(t.shape)} dtype={t.dtype} sample={sample}"
            )
        except Exception as e:
            print(f"[TT_DEBUG_BITMASK] step={self._debug_bitmask_step} {name}: debug read failed: {e}")

    def _debug_dump_all_shards(self, name, tensor):
        if not self._debug_bitmask_dump or not self._debug_should_log_bitmask():
            return
        try:
            shards = ttnn.get_device_tensors(tensor)
            for shard_idx, shard in enumerate(shards):
                shard_torch = ttnn.to_torch(shard).detach().cpu()
                out_path = self._debug_bitmask_dump_dir / (
                    f"step{self._debug_bitmask_step:04d}_{name}_shard{shard_idx:02d}.pt"
                )
                torch.save(shard_torch, out_path)
            print(
                f"[TT_DEBUG_BITMASK] step={self._debug_bitmask_step} dumped "
                f"{len(shards)} shard(s) for {name} to {self._debug_bitmask_dump_dir}"
            )
        except Exception as e:
            print(f"[TT_DEBUG_BITMASK] step={self._debug_bitmask_step} {name}: dump failed: {e}")

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

        lm_head_output = self.lm_head(
            x, None if mode == "prefill" else self.prefetcher_setup.worker_sub_device_id, mode=mode
        )
        # if mode is decode and Qwen model
        if mode == "decode" and self.args.is_qwen:
            ttnn.to_memory_config(self.tt_ccl.tt_lm_head_buffer, ttnn.DRAM_MEMORY_CONFIG)
        return lm_head_output

    def __del__(self):
        self.tt_ccl.close()

        # clear global saved addresses
        global global_tt_tensor_address
        global_tt_tensor_address = None
