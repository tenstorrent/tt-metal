# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from tqdm import tqdm
from loguru import logger
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
        self,
        tokens,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        batch_size=1,
        start_pos=0,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on host (to be copied to device later).
        start_pos: Absolute position in sequence (for prefix caching, indicates where cached tokens end)

        Returns position_ids instead of rot_mats - rot_mats will be computed
        from position_ids in transform_prefill_inputs_device using ttnn.embedding.
        """
        logger.info("prepare_prefill_inputs_host")
        logger.info(f"user_id: {user_id}")
        logger.info(f"page_table: {page_table}")
        logger.info(f"page_table shape: {page_table.shape if page_table is not None else None}")
        logger.info(f"chunk_page_table: {chunk_page_table}")
        logger.info(f"batch_size: {batch_size}")
        logger.info(f"start_pos: {start_pos}")

        tokens = tokens.reshape(1, 1, 1, -1)
        S = tokens.shape[-1]
        tokens = ttnn.from_torch(
            tokens,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # Create position indices for this request (replaces rot_mats computation)
        # Position indices will be used with ttnn.embedding to compute rot_mats on device
        position_ids = torch.arange(start_pos, start_pos + S, dtype=torch.int32)

        # Validation: Check position_ids are within valid range
        max_pos = position_ids.max().item()
        if hasattr(self, 'rope_setup') and hasattr(self.rope_setup, 'max_seq_len'):
            if max_pos >= self.rope_setup.max_seq_len:
                logger.error(
                    f"[POSITION_IDS] ERROR: position_ids exceed max_seq_len! "
                    f"max_pos={max_pos}, max_seq_len={self.rope_setup.max_seq_len}, "
                    f"start_pos={start_pos}, seq_len={S}"
                )
                raise ValueError(
                    f"Position IDs exceed max_seq_len: max_pos={max_pos}, "
                    f"max_seq_len={self.rope_setup.max_seq_len}"
                )

        logger.info(
            f"[POSITION_IDS] Creating position_ids: start_pos={start_pos}, seq_len={S}, "
            f"position_ids_range=[{position_ids.min().item()}, {position_ids.max().item()}], "
            f"position_ids.shape={position_ids.shape}, position_ids[:5]={position_ids[:5].tolist()}"
        )
        tt_position_ids = ttnn.from_torch(
            position_ids.unsqueeze(0),  # [1, seq_len]
            device=None,  # Host tensor - will be copied to device later
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        logger.info(f"[POSITION_IDS] tt_position_ids created: shape={tt_position_ids.shape}")

        columns = 4
        rows = 8
        user_id_column = user_id // rows  # 0 for user_id 0-7, 1 for user_id 8-15, etc.
        if page_table is not None:
            if batch_size > 1:
                assert batch_size == 32, "batch_size must be 32 for batched prefill"
                # we only want to update the kv cache for 8 users per 4 devices
                # pad with -1 for the seqlen of all other users
                batch_size_per_column = batch_size // columns
                page_table_padded = torch.ones((columns, page_table.shape[1] * batch_size), dtype=torch.int32) * -1
                for i in range(columns):
                    page_table_padded[
                        i,
                        (i * batch_size_per_column)
                        * page_table.shape[1] : (i + 1)
                        * batch_size_per_column
                        * page_table.shape[1],
                    ] = page_table[i * batch_size_per_column : (i + 1) * batch_size_per_column, :].reshape(1, -1)
                chunk_page_table_padded = None  # batch_size>1 => no prefix caching => no chunk_page_table
            else:
                # we only want to update the kv cache on the 8 devices (every fourth device starting at user_id//8 ) for a given user_id
                # we are setting the page table to -1 for all other devices to skip the update
                page_table_padded = torch.ones((columns, page_table.shape[1]), dtype=torch.int32) * -1
                # Note: For prefix caching, page_table is already extracted to a single row (shape: 1, num_blocks),
                # so we always access row 0. The original user_id is used only to compute user_id_column.
                page_table_padded[user_id_column, :] = page_table[0, :]

            logger.info(f"page_table_padded shape: {page_table_padded.shape}")
            logger.info(f"page_table_padded: {page_table_padded}")

            tt_page_table = ttnn.from_torch(
                page_table_padded,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(  # Each mesh column gets one row of the page table
                    self.mesh_device, dims=(None, 0), mesh_shape=self.args.cluster_shape
                ),
            )
            logger.info(f"tt_page_table shape: {tt_page_table.shape}")
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            assert batch_size == 1, "chunk_page_table is only supported for batch_size=1"
            chunk_page_table_padded = torch.ones((columns, chunk_page_table.shape[1]), dtype=torch.int32) * -1
            chunk_page_table_padded[user_id_column, :] = chunk_page_table[0, :]

            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table_padded,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(  # Each mesh column gets one row of the chunk page table
                    self.mesh_device, dims=(None, 0), mesh_shape=self.args.cluster_shape),
            )
        else:
            tt_chunk_page_table = None

        # For paged_fill_cache, the batch_idx must be the local row index in the sharded page_table,
        # not the global user_id. When batch_size=1, page_table is sharded so each device has only 1 row,
        # so the local batch index is always 0. The global user_id is only used to determine which
        # mesh column gets valid data (via user_id_column = user_id // 8).
        local_batch_idx = 0 if batch_size == 1 else user_id
        user_id = ttnn.from_torch(
            torch.tensor([local_batch_idx], dtype=torch.int32),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        logger.info("Returning from prepare_prefill_inputs_host")

        # Return position_ids instead of rot_mats (5th element)
        # rot_mats will be computed from position_ids in transform_prefill_inputs_device
        return tokens, user_id, tt_page_table, tt_chunk_page_table, tt_position_ids

    def transform_prefill_inputs_device(
        self,
        tokens,
        user_id,
        page_table=None,
        chunk_page_table=None,
        position_ids=None,
    ):
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)

        # Compute rot_mats from position_ids using embedding lookup
        rot_mats = None
        if position_ids is not None:
            seq_len = position_ids.shape[-1]
            logger.info(
                f"[ROT_MATS] Computing rot_mats from position_ids: "
                f"position_ids.shape={position_ids.shape}, seq_len={seq_len}"
            )
            rot_mats = self.rope_setup.get_prefill_rot_mats(position_ids, seq_len)

        return tt_tokens, user_id, page_table, chunk_page_table, rot_mats

    def prepare_inputs_prefill(
        self,
        tokens,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        batch_size=1,
        start_pos=0,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Its implementation can take advantage of a few other functions which the
        model must implement.
        start_pos: Absolute position in sequence (for prefix caching, indicates where cached tokens end)
        """
        host_inputs = self.prepare_prefill_inputs_host(
            tokens, user_id, page_table, chunk_page_table, batch_size, start_pos
        )
        # host_inputs is: (tokens, user_id, tt_page_table, tt_chunk_page_table, tt_position_ids)
        # All 5 are host tensors - position_ids will be used to compute rot_mats on device
        (
            tokens_host,
            user_id_host,
            tt_page_table_host,
            tt_chunk_page_table_host,
            tt_position_ids_host,
        ) = host_inputs

        # Copy all 5 host tensors to device
        device_inputs = copy_host_to_device(
            (tokens_host, user_id_host, tt_page_table_host, tt_chunk_page_table_host, tt_position_ids_host),
            mesh_device=self.mesh_device
        )
        logger.info("Returned from copy_host_to_device")
        # transform_prefill_inputs_device computes rot_mats from position_ids
        transformed_device_inputs = self.transform_prefill_inputs_device(*device_inputs)
        logger.info("Returned from transform_prefill_inputs_device")
        # Returns: (tt_tokens, user_id, page_table, chunk_page_table, rot_mats)
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
        # DEBUG: Decode inputs
        logger.info(f"[DECODE] prepare_decode_inputs_host:")
        logger.info(f"[DECODE]   tokens.shape={tokens.shape}, tokens[:8]={tokens[:8].tolist()}")
        logger.info(f"[DECODE]   current_pos.shape={current_pos.shape}, current_pos[:8]={current_pos[:8].tolist()}")
        if page_table is not None:
            logger.info(f"[DECODE]   page_table.shape={page_table.shape}")
            # Log first few rows of page table (first 8 users, first 8 blocks)
            logger.info(f"[DECODE]   page_table[:8, :8]=\n{page_table[:8, :8]}")
        else:
            logger.info(f"[DECODE]   page_table=None")
        logger.info(f"[DECODE]   is_cur_pos_sharded={is_cur_pos_sharded}, is_page_table_sharded={is_page_table_sharded}")
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

        user_id is used to select output from the correct mesh column for paged prefill.
        For batch_size=1, only the user's column has valid KV cache data.
        Mesh is 8x4 (8 rows for heads, 4 columns for data parallel).
        user_id 0-7 → column 0, user_id 8-15 → column 1, etc.
        """
        # Determine which device to read output from based on user_id
        # Device index for row 0 of each column: col 0 → dev 0, col 1 → dev 1, etc.
        output_device_idx = user_id // 8  # 0, 1, 2, or 3

        # DEBUG: Check input tensor shape and sample values before norm
        logger.info(f"[PREFILL_OUTPUT] Input tt_out shape: {tt_out.shape}, user_id={user_id}, output_device_idx={output_device_idx}")

        x, _ = self.norm(tt_out, res=None, mode="prefill")

        # DEBUG: Check tensor shape after norm
        logger.info(f"[PREFILL_OUTPUT] After norm x shape: {x.shape}")
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
            # DEBUG: Check slice bounds
            x_seq_len = x_split[i].shape[2] if hasattr(x_split[i], 'shape') else 'unknown'
            logger.info(
                f"[PREFILL_OUTPUT] Slicing: last_token_idx_i={last_token_idx_i}, "
                f"x_seq_len={x_seq_len}, i={i}"
            )
            if isinstance(x_seq_len, int) and last_token_idx_i >= x_seq_len:
                logger.error(
                    f"[PREFILL_OUTPUT] SLICE OUT OF BOUNDS! "
                    f"last_token_idx_i={last_token_idx_i} >= x_seq_len={x_seq_len}"
                )

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

            # DEBUG: Check if logits contain garbage (NaN, inf, or extreme values)
            logits_sample = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[output_device_idx])
            if torch.isnan(logits_sample).any() or torch.isinf(logits_sample).any():
                logger.error(
                    f"[PREFILL_OUTPUT] Logits contain NaN or Inf! "
                    f"last_token_idx_i={last_token_idx_i}, "
                    f"has_nan={torch.isnan(logits_sample).any()}, "
                    f"has_inf={torch.isinf(logits_sample).any()}"
                )

            tt_out = ttnn.argmax(tt_logits, dim=3, keepdim=True, use_multicore=True)
            if isinstance(tt_out, list):
                tt_out = tt_out[0]
            logger.info(f"Running to_torch(ttnn.get_device_tensors(tt_out)[{output_device_idx}])")
            toks = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[output_device_idx]).float()[0, 0, 0, :1]
            logger.info(f"Returned from to_torch(ttnn.get_device_tensors(tt_out)[{output_device_idx}])")
            # DEBUG: Print the actual token produced by prefill
            tok_val = int(toks[0].item())
            logger.info(f"[PREFILL_OUTPUT] Produced token: {tok_val} (valid range: 0-{self.vocab_size-1})")

            # Validate token ID is within expected range
            tok_val = int(toks[0].item())
            if tok_val < 0 or tok_val >= self.vocab_size:
                logger.error(
                    f"[PREFILL_OUTPUT] Invalid token ID detected: {tok_val}, "
                    f"last_token_idx_i={last_token_idx_i}, "
                    f"x_shape_before_slice={x_split[i].shape}, "  # x_split[i] is original, x is sliced
                    f"tt_logits_shape={tt_logits.shape}, "
                    f"vocab_size={self.vocab_size}"
                )
            toks_list.append(toks)

        if tt_out_logits_saved is not None:
            # make sure tt_out_logits_saved is mutable
            logger.info(f"Running to_torch(ttnn.get_device_tensors(tt_logits)[{output_device_idx}])")
            logits_saved = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[output_device_idx]).float()[0, 0, :, :]
            logger.info(f"Returned from to_torch(ttnn.get_device_tensors(tt_logits)[{output_device_idx}])")
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

        # Save output logits to global python object
        if tt_out_logits_saved is not None:
            tt_out_logits = ttnn.to_torch(
                tt_logits[0],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, dims=(3, 1), mesh_shape=self.args.cluster_shape
                ),
            )
            tt_out_logits = tt_out_logits[0, 0, 0, : self.args.vocab_size]

            tt_out_logits_saved.copy_(tt_out_logits)

        if capture_sampling_trace:
            return tt_logits

        tt_toks, tt_log_probs = self.sampling.sample(
            tt_logits[0],
            tt_out_tok=x,
            enable_trace=False,
        )
        return tt_toks, tt_log_probs

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
