# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from tqdm import tqdm

import ttnn
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: DeepseekGenerator, tt_cache_path):
    """
    Allocate KV cache tensors for vLLM.

    Args:
        kv_cache_shape: Shape tuple (num_blocks, num_kv_heads, block_size, head_size)
        dtype: Data type for the cache
        num_layers: Number of layers to allocate cache for
        model: The DeepseekGenerator model instance
        tt_cache_path: Path for cache files

    Returns:
        List of [k_cache, v_cache] pairs, one per layer
    """
    kv_cache = []
    cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)

    for layer_num in tqdm(range(num_layers), desc="Allocating TT kv caches for each layer"):
        kv_tt_i = [
            ttnn.as_tensor(
                cache_kv,
                device=model.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                # Separate cache files for K and V to avoid collision.
                cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}_layer_{layer_num}",
            )
            for kv in ["k", "v"]
        ]
        kv_cache.append(kv_tt_i)

    return kv_cache


class DeepseekV3ForCausalLM(DeepseekGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations: str = None
    ):
        model_path = "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"
        cache_dir = "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache"
        tokenizer = load_tokenizer(model_path)

        model = cls(
            hf_config=hf_config,
            mesh_device=mesh_device,
            model_path=Path(model_path),
            cache_dir=Path(cache_dir),
            tokenizer=tokenizer,
            override_num_layers=5,
        )
        model._prepare_run_configs("prefill")
        model._prepare_run_configs("decode")
        return model

    @property
    def cache_path(self):
        return self.cache_dir

    def _convert_page_table_for_user(
        self, page_table: torch.Tensor, user_id: int, kv_cache, seq_len: int
    ) -> tuple[ttnn.Tensor, ...]:
        """
        Convert vLLM's block_tables (page_table) to TTNN tensor format for a specific user.
        Creates one page table per layer as expected by the model.

        Args:
            page_table: torch.Tensor of shape [batch_size, max_num_blocks_per_req] from vLLM
            user_id: The user index to extract the page table for
            kv_cache: vLLM's KV cache to determine block_size
            seq_len: Sequence length for this user to calculate actual number of blocks needed

        Returns:
            Tuple of TTNN tensors, one per layer
        """
        if page_table is None:
            # Fall back to internal page tables if vLLM doesn't provide one
            return self.page_tables_tt

        from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
        from models.demos.deepseek_v3.utils.test_utils import even_int_div

        # Calculate expected shape: [batch_per_shard, blocks_per_user]
        _, dp_factor = self.mesh_device.shape
        batch_per_shard = even_int_div(self.batch_size_per_row, self.mesh_device.shape[0])
        blocks_per_user = even_int_div(self.paged_config.max_num_blocks, batch_per_shard)

        # Extract the user's block table row
        user_blocks = page_table[user_id, :blocks_per_user].clone()  # [max_num_blocks_per_req] or less

        # Create a full page table with shape [batch_per_shard, blocks_per_user]
        full_page_table = torch.full((batch_per_shard, blocks_per_user), -1, dtype=torch.int32)
        local_user_id = user_id % batch_per_shard
        num_user_blocks = min(user_blocks.shape[0], blocks_per_user)
        full_page_table[local_user_id, :num_user_blocks] = user_blocks[:num_user_blocks]

        # Convert to TTNN format using the model's helper
        page_table_tt = MLA2D.create_page_table(
            paged_config=self.paged_config,
            mesh_device=self.mesh_device,
            page_table=full_page_table,
            batch_size_per_row=self.batch_size_per_row // self.mesh_device.shape[0],
        )

        # Create one page table per layer (all identical)
        num_layers = self.hf_config.num_hidden_layers
        return tuple(page_table_tt for _ in range(num_layers))

    def prefill_forward(self, *args, **kwargs):
        """Prefill forward pass, following vLLM interface."""
        tokens = kwargs["tokens"]
        prompt_lens = kwargs.get("prompt_lens", None)
        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)

        # Process each user in the batch
        num_of_users = tokens.shape[0]
        last_logits = []

        for user_id in range(num_of_users):
            if prompt_lens is not None and prompt_lens[user_id] == 0:
                last_logits.append(
                    torch.zeros(tokens.shape[1], self.hf_config.vocab_size, device=tokens.device, dtype=tokens.dtype)
                )
                continue

            # Convert vLLM page_table to model format for this specific user
            page_tables = self._convert_page_table_for_user(page_table, user_id)

            original_page_tables = self.page_tables_tt
            self.page_tables_tt = page_tables
            try:
                user_out = self._prefill(tokens[user_id], user_id)
                last_logits.append(user_out.squeeze(0).squeeze(0))  # [1, 1, S, V] -> [S, V]
            finally:
                self.page_tables_tt = original_page_tables

        last_logits = torch.stack(last_logits)  # [num_of_users, S, V]
        return last_logits

    def decode_forward(self, *args, **kwargs):
        """Decode forward pass, following vLLM interface."""
        tokens = kwargs["tokens"]
        start_pos = kwargs["start_pos"]
        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)

        tokens_step = tokens.squeeze(1) if tokens.dim() == 3 else tokens

        # For decode, process the entire batch at once
        # The page_table from vLLM has shape [batch_size, max_num_blocks_per_req]
        # need to create a combined page table for all users in the batch

        if page_table is not None:
            #  need to handle the full batch page table
            # The model expects page tables that can handle all users in the batch
            from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
            from models.demos.deepseek_v3.utils.test_utils import even_int_div

            # Calculate expected shape: [batch_per_shard, blocks_per_user]
            _, dp_factor = self.mesh_device.shape
            batch_per_shard = even_int_div(self.batch_size_per_row, self.mesh_device.shape[0])
            blocks_per_user = even_int_div(self.paged_config.max_num_blocks, batch_per_shard)

            # Create a full page table structure for all users
            full_page_table = torch.full((batch_per_shard, blocks_per_user), -1, dtype=torch.int32)

            batch_size = page_table.shape[0]
            for user_id in range(batch_size):
                local_user_id = user_id % batch_per_shard
                user_blocks = page_table[user_id, :blocks_per_user].clone()
                num_user_blocks = min(user_blocks.shape[0], blocks_per_user)
                full_page_table[local_user_id, :num_user_blocks] = user_blocks[:num_user_blocks]

            # Convert to TTNN format
            page_table_tt = MLA2D.create_page_table(
                paged_config=self.paged_config,
                mesh_device=self.mesh_device,
                page_table=full_page_table,
                batch_size_per_row=self.batch_size_per_row // self.mesh_device.shape[0],
            )

            # Create one page table per layer
            num_layers = self.hf_config.num_hidden_layers
            page_tables = tuple(page_table_tt for _ in range(num_layers))
        else:
            # Fall back to internal page tables
            page_tables = self.page_tables_tt

        original_page_tables = self.page_tables_tt
        self.page_tables_tt = page_tables
        try:
            return_value = (
                self._decode_step(
                    tokens_step=tokens_step, positions=start_pos, batch_size_per_row=self.batch_size_per_row
                )
                .squeeze(0)
                .squeeze(0)
                .unsqueeze(1)  # [1,1,B,V] -> [B, 1, V]
            )
        finally:
            self.page_tables_tt = original_page_tables

        return return_value

    def allocate_kv_cache(self, *args, **kwargs):
        """Allocate KV cache"""
        return allocate_vllm_kv_cache(*args, **kwargs, model=self, tt_cache_path=self.cache_path)
