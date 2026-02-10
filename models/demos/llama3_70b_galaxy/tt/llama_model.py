# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from loguru import logger
from tqdm import tqdm
from models.demos.llama3_70b_galaxy.tt.llama_decoder import TtTransformerBlock
from models.common.rmsnorm import RMSNorm
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama3_70b_galaxy.tt.distributed_norm import DistributedNorm
from models.demos.llama3_70b_galaxy.tt.lm_head import LMHead
from models.demos.llama3_70b_galaxy.tt.llama_common import copy_host_to_device, get_prefill_rot_mat
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

        # Device tensor holding max_seq_len; used directly as slice end in prefill (never updated).
        self._tt_seq_len_buffer = ttnn.from_torch(
            torch.tensor([1, 1, self.args.max_seq_len, self.args.head_dim], dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        # [0, 0, 0, 0] - slice to get [0,0] and [0] for building slice start [0, 0, N, 0] from chunk_start_idx [N].
        self._tt_slice_start_zeros_4 = ttnn.from_torch(
            torch.tensor([0, 0, 0, 0], dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def get_or_create_prefill_rot_mats(self):
        """
        Return device-side rot mats for prefill, cached once for max_seq_len.
        """
        if self.tt_rot_mats_prefill is None:
            self.tt_rot_mats_prefill = get_prefill_rot_mat(
                head_dim=self.args.head_dim,
                max_seq_len=self.args.max_seq_len,
                mesh_device=self.mesh_device,
                seq_len=int(self.args.max_seq_len),
                scale_factor=self.args.rope_scaling_factor,
                start_pos=0,
            )
        return self.tt_rot_mats_prefill

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
        chunk_start_idx=0,
        batch_size=1,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on host (to be copied to device later).
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

        columns = 4
        rows = 8
        user_id_column = user_id // rows  # 0 for user_id 0-7, 1 for user_id 8-15, etc.

        if page_table is not None:
            # NOTE ON SENTINELS / SAFETY:
            # - For chunked SDPA (prefix caching, chunk_start_idx > 0), the SDPA reads the page table.
            #   Reading -1 can cause address overflow, so use 0 for padding (vLLM reserves block 0 as read-safe).
            # - For non-chunked SDPA (no prefix caching), SDPA uses fresh K/V tensors, not the cache.
            #   We only use page_table for paged_fill_cache (write), where -1 means "skip write".
            #   Using -1 for inactive columns avoids unnecessary writes and potential race conditions.
            use_chunked_sdpa_path = chunk_start_idx is not None and chunk_start_idx > 0
            inactive_fill_value = 0 if use_chunked_sdpa_path else -1

            # Chunked SDPA requires page_table "stick size" (row width) to be a multiple of 32 Bytes.
            # Page table entries are int32, so this means the number of columns must be a multiple of 8
            # (8 * sizeof(int32) = 32 bytes). Pad with read-safe zeros.
            def _pad_table_cols_to_multiple_of_8_int32(table_2d: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
                assert table_2d.ndim == 2, f"expected 2D table, got shape={tuple(table_2d.shape)}"
                cols = table_2d.shape[1]
                cols_padded = ((cols + 7) // 8) * 8
                if cols_padded == cols:
                    return table_2d
                padded = torch.full((table_2d.shape[0], cols_padded), pad_value, dtype=table_2d.dtype)
                padded[:, :cols] = table_2d
                return padded

            if batch_size > 1:
                assert batch_size == 32, "batch_size must be 32 for batched prefill"
                # Mesh layout padding: (32, num_blocks) -> (4, 32 * num_blocks).
                # For non-chunked SDPA, use -1 for unused regions so paged_fill_cache skips writes.
                # For chunked SDPA (prefix caching), use 0 so SDPA doesn't read -1.
                batch_size_per_column = batch_size // columns
                page_table_padded = (
                    torch.ones((columns, page_table.shape[1] * batch_size), dtype=torch.int32) * inactive_fill_value
                )
                for i in range(columns):
                    row_block = page_table[i * batch_size_per_column : (i + 1) * batch_size_per_column, :].reshape(
                        1, -1
                    )
                    page_table_padded[
                        i,
                        (i * batch_size_per_column)
                        * page_table.shape[1] : (i + 1)
                        * batch_size_per_column
                        * page_table.shape[1],
                    ] = row_block
                chunk_page_table_padded = None  # batch_size>1 => no prefix caching => no chunk_page_table
            else:
                # Mesh layout padding: only the active column is used.
                # For non-chunked SDPA, use -1 for inactive columns so paged_fill_cache skips writes.
                # For chunked SDPA (prefix caching), use 0 so SDPA doesn't read -1.
                num_blocks = page_table.shape[1]
                page_table_padded = torch.ones((columns, num_blocks), dtype=torch.int32) * inactive_fill_value
                # Note: For prefix caching, page_table is already extracted to a single row (shape: 1, num_blocks),
                # so we always access row 0. The original user_id is used only to compute user_id_column.
                page_table_padded[user_id_column, :num_blocks] = page_table[0, :]

                # Ensure row width (in bytes) divisible by 32 for chunked SDPA.
                page_table_padded = _pad_table_cols_to_multiple_of_8_int32(
                    page_table_padded, pad_value=inactive_fill_value
                )

            tt_page_table = ttnn.from_torch(
                page_table_padded,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(  # Each mesh column gets one row of the page table
                    self.mesh_device, dims=(None, 0), mesh_shape=self.args.cluster_shape
                ),
            )
        else:
            tt_page_table = None

        if chunk_page_table is not None:
            assert batch_size == 1, "chunk_page_table is only supported for batch_size=1"
            # Use 0 for inactive columns so no reader ever sees -1.
            chunk_page_table_padded = torch.zeros((columns, chunk_page_table.shape[1]), dtype=torch.int32)
            chunk_page_table_padded[user_id_column, :] = chunk_page_table[0, :]

            tt_chunk_page_table = ttnn.from_torch(
                chunk_page_table_padded,
                device=None,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(  # Each mesh column gets one row of the chunk page table
                    self.mesh_device, dims=(None, 0), mesh_shape=self.args.cluster_shape
                ),
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

        tt_chunk_start_idx = ttnn.from_torch(
            torch.tensor([chunk_start_idx], dtype=torch.int32),
            device=None,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        return tokens, user_id, tt_page_table, tt_chunk_page_table, tt_chunk_start_idx

    def transform_prefill_inputs_device(
        self,
        tokens,
        user_id,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
    ):
        tt_tokens = self.embd(tokens)
        tt_tokens = ttnn.unsqueeze_to_4D(tt_tokens)
        return tt_tokens, user_id, page_table, chunk_page_table, chunk_start_idx

    def prepare_inputs_prefill(
        self,
        tokens,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=0,
        batch_size=1,
    ):
        """
        Inputs are torch tensors or python types. This function returns ttnn
        tensors on device.
        Returns 5 outputs: prefill_input, tt_user_id, page_table_tt, tt_chunk_page_table, tt_chunk_start_idx.
        """
        host_inputs = self.prepare_prefill_inputs_host(
            tokens, user_id, page_table, chunk_page_table, chunk_start_idx, batch_size
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

    def process_output_prefill_logits(self, tt_out, last_token_idx, tt_out_logits_saved=None, user_id=0):
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

    def process_output_prefill(self, tt_out, last_token_idx, tt_out_logits_saved=None, user_id=0):
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
            toks = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[output_device_idx]).float()[0, 0, 0, :1]
            toks_list.append(toks)

        if tt_out_logits_saved is not None:
            # make sure tt_out_logits_saved is mutable
            logits_saved = ttnn.to_torch(ttnn.get_device_tensors(tt_logits)[output_device_idx]).float()[0, 0, :, :]

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
        x,  # ttnn.Tensor, shape [1, 1, seq_len, hidden_dim]; token embeddings input replicated across all devices
        user_id=0,  # ttnn.Tensor, shape [1]; user id replicated across all devices
        page_table=None,  # ttnn.Tensor, shape [4, num_blocks]; main paged-attention page table, replicated across rows, sharded across columns. 32*num_blocks for batched prefill.
        chunk_page_table=None,  # ttnn.Tensor or None, shape [4, num_blocks]; only needed for prefix caching (single user), replicated across rows, sharded across columns
        chunk_start_idx=None,  # ttnn.Tensor, shape [1]; index of cached-token split for prefix caching, replicated across all devices
        start_pos=0,  # int, starting position in sequence for attention (used in SDPA path decision)
        get_last_token=-1,  # int or list[int], output mode: which token to return (last idx or indices)
        kv_cache=None,  # ttnn.Tensor, data parallel across cols, head parallel across rows
        rot_mats=None,  # Tuple[ttnn.Tensor, ttnn.Tensor], each of shape [1, 1, max_seq_len, head_dim]; RoPE matrices for full (0..max_seq_len) replicated across all devices
        batch_size=1,  # int, number of users or batch size for prefill; controls input slicing and paging
    ):
        """
        Prefill forward. Expects rot_mats to be full (0..max_seq_len) from get_or_create_prefill_rot_mats(),
        and chunk_start_idx to be a device tensor (e.g. shape (1,) int32) used for rot mats slicing and chunked SDPA.
        start_pos is a Python int used for attention decisions (SDPA path, program config); must match
        the value in chunk_start_idx.
        """
        assert rot_mats is not None, "prefill requires rot_mats (full from get_or_create_prefill_rot_mats)"
        assert chunk_start_idx is not None and hasattr(
            chunk_start_idx, "shape"
        ), "prefill requires chunk_start_idx as device tensor"
        tt_logits = self.forward(
            x,
            current_pos=None,
            rot_mats=rot_mats,
            user_id=user_id,
            mode="prefill",
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            start_pos=start_pos,
            get_last_token=get_last_token,
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
        chunk_start_idx=None,  # On-device
        start_pos=0,  # Python int
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

        # Prefill: for prefix caching (start_pos > 0), slice rot_mats to [chunk_start_idx, max_seq_len).
        # When start_pos == 0, use full rot_mats as-is (no slice) to avoid ttnn.concat/ttnn.slice device
        # ops that can hang on some builds; trace capture for prefix-cached runs uses start_pos > 0.
        if mode == "prefill" and start_pos > 0 and False:
            full_rot_cos, full_rot_sin = rot_mats[0], rot_mats[1]
            num_devices = self.args.cluster_shape[0] * self.args.cluster_shape[1]
            z = self._tt_slice_start_zeros_4
            tt_slice_starts = ttnn.concat(
                [z[0:2], chunk_start_idx, z[3:4]],
                dim=0,
            )
            rot_cos_slice = ttnn.slice(
                input_tensor=full_rot_cos,
                starts=tt_slice_starts,
                ends=self._tt_seq_len_buffer,
                slice_dim=2,
                num_devices=num_devices,
            )
            rot_sin_slice = ttnn.slice(
                input_tensor=full_rot_sin,
                starts=tt_slice_starts,
                ends=self._tt_seq_len_buffer,
                slice_dim=2,
                num_devices=num_devices,
            )
            rot_mats = (rot_cos_slice, rot_sin_slice)

        h = None
        # x needs to be in bfloat16_b as it gets reused as the residual tensor
        for i, layer in enumerate(self.layers):
            if mode == "prefill":
                logger.info(f"forward prefill: layer {i}/{len(self.layers)}")
            x, h = layer(
                x,
                h,
                current_pos,
                rot_mats,
                user_id,
                mode,
                page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=start_pos,
                chunk_start_idx_tensor=chunk_start_idx if mode == "prefill" else None,
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
