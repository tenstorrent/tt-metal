# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer


def _pad_tokens(tokens: torch.Tensor, pad_value: int = 0, block_size: int = USERS_PER_ROW) -> torch.Tensor:
    """
    Pad tokens to the nearest multiple of block_size.

    Args:
        tokens: Input tensor of shape [batch_size, seq_len]
        pad_value: Value to use for padding (default: 0)

    Returns:
        Padded tensor of shape [batch_size, padded_seq_len] where padded_seq_len is multiple of block_size
    """
    batch_size, seq_len = tokens.shape
    # Calculate the nearest multiple of block_size
    padded_len = ((seq_len + block_size - 1) // block_size) * block_size
    if padded_len == seq_len:
        return tokens
    padded_tokens = torch.full((batch_size, padded_len), pad_value, dtype=tokens.dtype, device=tokens.device)
    padded_tokens[:, :seq_len] = tokens
    return padded_tokens


class DeepseekV3ForCausalLM(DeepseekGenerator):
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations: str = None
    ):
        model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL")
        cache_dir = os.environ.get("DEEPSEEK_V3_CACHE")
        if not model_path:
            raise ValueError(
                "DEEPSEEK_V3_HF_MODEL is not set. Set the environment variable or initialize via the demo "
                "entrypoint with an explicit --model-path."
            )
        if not cache_dir:
            raise ValueError(
                "DEEPSEEK_V3_CACHE is not set. Set the environment variable or initialize via the demo "
                "entrypoint with an explicit --cache-dir."
            )
        tokenizer = load_tokenizer(model_path)

        model = cls(
            hf_config=hf_config,
            mesh_device=mesh_device,
            model_path=Path(model_path),
            cache_dir=Path(cache_dir),
            tokenizer=tokenizer,
        )

        return model

    @property
    def cache_path(self):
        return self.cache_dir

    def prefill_forward(self, *args, **kwargs):
        start_pos = kwargs.get("start_pos", None)
        assert (start_pos is None) or all(
            x == 0 for x in start_pos
        ), f"Prefix caching is not supported for DeepseekV3ForCausalLM, got start_pos: {start_pos}"
        assert self.model_run_config_prefill is not None, "Model run config prefill is not initialized"

        kwargs.pop("enable_trace", None)
        logger.warning("Prefill tracing not supported for DeepseekGenerator")
        tokens = kwargs["tokens"]
        lengths = kwargs["prompt_lens"]
        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)
        empty_slots = kwargs.get("empty_slots", None)

        if page_table is not None and not hasattr(self, "_validated_vllm_prefill"):
            self._validate_vllm_prefill_inputs(tokens, lengths, page_table)
            self._validated_vllm_prefill = True

        if all(length == 0 for length in lengths):
            return torch.zeros(tokens.shape[0], self.hf_config.vocab_size, device=tokens.device, dtype=tokens.dtype)

        # Set kv_cache if provided and all entries are valid
        if kv_cache is not None and not any(entry is None for entry in kv_cache):
            self.set_kv_cache(kv_cache)

        pad_value = self.tokenizer.pad_token_id
        pad_block_size = self.paged_config.block_size if self.paged_config is not None else USERS_PER_ROW
        max_prompt_len = int(max(lengths)) if len(lengths) else 0
        max_padded_len = (
            ((max_prompt_len + pad_block_size - 1) // pad_block_size) * pad_block_size if max_prompt_len > 0 else 0
        )
        num_of_users = tokens.shape[0]
        if getattr(self, "debug_investigation", False) and not hasattr(self, "_debug_prefill_logged"):
            self._debug_prefill_logged = True
            lens_t = lengths if isinstance(lengths, torch.Tensor) else torch.as_tensor(lengths)
            lens_min = int(lens_t.min().item()) if lens_t.numel() else 0
            lens_max = int(lens_t.max().item()) if lens_t.numel() else 0
            empty_summary = None
            stride_guess = None
            if empty_slots is not None and len(empty_slots) > 0:
                empty_summary = {
                    "len": len(empty_slots),
                    "min": int(min(empty_slots)),
                    "max": int(max(empty_slots)),
                    "head": [int(x) for x in empty_slots[: min(16, len(empty_slots))]],
                }
                # Guess stride by detecting first discontinuity
                for i in range(1, len(empty_slots)):
                    if empty_slots[i] != empty_slots[i - 1] + 1:
                        stride_guess = int(empty_slots[i])
                        break
            logger.info(
                "[INV] prefill_forward: tokens_shape={} num_users={} prompt_lens[min,max]=({},{}) empty_slots={} stride_guess={} page_table_shape={}",
                tuple(tokens.shape),
                num_of_users,
                lens_min,
                lens_max,
                empty_summary,
                stride_guess,
                None if page_table is None else tuple(page_table.shape),
            )
        last_logits = []
        for i in range(num_of_users):
            user_id = empty_slots[i] if empty_slots is not None else i
            if getattr(self, "debug_investigation", False) and i < 8:
                logger.info("[INV] prefill map: local_user_id={} user_id={}", i, int(user_id))
            prompt_len = int(lengths[i])
            if prompt_len == 0:
                last_logits.append(
                    torch.zeros(max_padded_len, self.hf_config.vocab_size, device=tokens.device, dtype=tokens.dtype)
                )
                continue
            user_tokens = tokens[i, :prompt_len].unsqueeze(0)
            user_tokens = _pad_tokens(user_tokens, pad_value, block_size=pad_block_size).squeeze(0)
            user_out = self._prefill(user_tokens, user_id, page_table, local_user_id=i)
            user_logits = user_out.squeeze(0).squeeze(0)  # [1, 1, S, V] -> [S, V]
            if user_logits.shape[0] > prompt_len:
                user_logits = user_logits[:prompt_len]
            if user_logits.shape[0] < max_padded_len:
                pad_len = max_padded_len - user_logits.shape[0]
                pad_logits = user_logits[-1:].expand(pad_len, -1)
                user_logits = torch.cat([user_logits, pad_logits], dim=0)
            last_logits.append(user_logits)
        last_logits = torch.stack(last_logits)  # [num_of_users, S, V]

        return last_logits

    def decode_forward(self, *args, **kwargs):
        assert self.model_run_config_decode is not None, "Model run config decode is not initialized"

        page_table = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)
        if getattr(self, "debug_investigation", False) and not hasattr(self, "_debug_decode_forward_logged"):
            self._debug_decode_forward_logged = True
            positions = kwargs.get("start_pos", None)
            pos_t = positions if isinstance(positions, torch.Tensor) else torch.as_tensor(positions)
            pos_min = int(pos_t.min().item()) if pos_t.numel() else 0
            pos_max = int(pos_t.max().item()) if pos_t.numel() else 0
            logger.info(
                "[INV] decode_forward: tokens_shape={} positions[min,max]=({},{}) page_table_shape={}",
                tuple(kwargs["tokens"].shape),
                pos_min,
                pos_max,
                None if page_table is None else tuple(page_table.shape),
            )
        if page_table is not None and not hasattr(self, "_validated_vllm_decode"):
            self._validate_vllm_decode_inputs(kwargs["tokens"], kwargs["start_pos"], page_table)
            self._validated_vllm_decode = True
        # Set kv_cache if provided and all entries are valid
        if kv_cache is not None and not any(entry is None for entry in kv_cache):
            self.set_kv_cache(kv_cache)

        tokens_step = kwargs["tokens"].squeeze(1)
        return_value = (
            self._decode_step(
                tokens_step=tokens_step,
                positions=kwargs["start_pos"],
                batch_size_per_row=USERS_PER_ROW,
                page_table=page_table,
            )
            .squeeze(0)
            .squeeze(0)
            .unsqueeze(1)
        )  # [1,1,B,V] -> [B, 1, V]
        return return_value

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        assert (
            num_layers == self.hf_config.num_hidden_layers
        ), f"Number of layers {num_layers} does not match the number of layers in the model {self.hf_config.num_hidden_layers}"
        if getattr(self, "debug_investigation", False):
            logger.info(
                "[INV] allocate_kv_cache: kv_cache_shape={} dtype={} num_layers={} paged_block_size={} max_num_blocks={}",
                kv_cache_shape,
                dtype,
                num_layers,
                None if self.paged_config is None else self.paged_config.block_size,
                None if self.paged_config is None else self.paged_config.max_num_blocks,
            )
        if not hasattr(self, "_validated_vllm_kv_cache"):
            self._validate_vllm_kv_cache(kv_cache_shape, dtype, num_layers)
            self._validated_vllm_kv_cache = True

        kv_cache_config = KvCacheConfig(kv_cache_shape=kv_cache_shape, dtype=dtype)
        self._prepare_run_configs("prefill", kv_cache_override=kv_cache_config)
        self._prepare_run_configs("decode", kv_cache_override=kv_cache_config)
        kv_cache = self.get_kv_cache()

        return kv_cache

    def _validate_vllm_kv_cache(self, kv_cache_shape, dtype, num_layers) -> None:
        expected_kvpe_dim = int(self.hf_config.kv_lora_rank + self.hf_config.qk_rope_head_dim)
        expected_block_size = int(self.paged_config.block_size)
        expected_blocks_per_seq = int(self.hf_config.max_seq_len // expected_block_size)
        assert kv_cache_shape[2] == expected_block_size, (
            f"vLLM kv_cache_shape[2] (block_size) mismatch: "
            f"kv_cache_shape[2]={kv_cache_shape[2]} vs "
            f"paged_config.block_size={expected_block_size}"
        )
        assert kv_cache_shape[3] == expected_kvpe_dim, (
            f"vLLM kv_cache_shape[3] (kvpe_dim) mismatch: "
            f"kv_cache_shape[3]={kv_cache_shape[3]} vs "
            f"kv_lora_rank+qk_rope_head_dim={expected_kvpe_dim}"
        )
        assert kv_cache_shape[1] == 1, (
            f"vLLM kv_cache_shape[1] (num_kv_heads) mismatch: " f"kv_cache_shape[1]={kv_cache_shape[1]} vs expected=1"
        )
        assert kv_cache_shape[0] >= expected_blocks_per_seq, (
            f"vLLM kv_cache_shape[0] (max_num_blocks) too small: "
            f"kv_cache_shape[0]={kv_cache_shape[0]} vs "
            f"min_required_blocks={expected_blocks_per_seq}"
        )
        assert num_layers == self.hf_config.num_hidden_layers, (
            f"vLLM num_layers mismatch: num_layers={num_layers} vs "
            f"hf_config.num_hidden_layers={self.hf_config.num_hidden_layers}"
        )

    def _validate_vllm_prefill_inputs(
        self, tokens: torch.Tensor, prompt_lens: torch.Tensor, page_table: torch.Tensor
    ) -> None:
        batch_size = int(tokens.shape[0])
        prompt_lens_t = prompt_lens if isinstance(prompt_lens, torch.Tensor) else torch.as_tensor(prompt_lens)
        max_prompt_len = int(prompt_lens_t.max().item()) if prompt_lens_t.numel() > 0 else 0
        expected_blocks_per_seq = int(self.hf_config.max_seq_len // self.paged_config.block_size)
        expected_max_block_id = int(self.paged_config.max_num_blocks) - 1
        assert page_table.shape[0] == batch_size, (
            f"vLLM page_table.shape[0] (batch) mismatch: "
            f"page_table.shape[0]={page_table.shape[0]} vs tokens.shape[0]={batch_size}"
        )
        assert page_table.shape[1] <= expected_blocks_per_seq, (
            f"vLLM page_table.shape[1] (blocks_per_seq) too large: "
            f"page_table.shape[1]={page_table.shape[1]} vs "
            f"max_blocks_per_seq={expected_blocks_per_seq}"
        )
        page_table_t = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        if page_table_t.numel() > 0:
            min_block_id = int(page_table_t.min().item())
            max_block_id = int(page_table_t.max().item())
            assert min_block_id >= 0, (
                f"vLLM page_table min block id is negative: " f"page_table.min()={min_block_id} vs expected>=0"
            )
            assert max_block_id <= expected_max_block_id, (
                f"vLLM page_table max block id out of range: "
                f"page_table.max()={max_block_id} vs max_allowed={expected_max_block_id}"
            )
        assert max_prompt_len <= self.hf_config.max_seq_len, (
            f"vLLM max prompt length exceeds model max_seq_len: "
            f"max_prompt_len={max_prompt_len} vs hf_config.max_seq_len={self.hf_config.max_seq_len}"
        )

    def _validate_vllm_decode_inputs(
        self, tokens: torch.Tensor, start_pos: torch.Tensor, page_table: torch.Tensor
    ) -> None:
        batch_size = int(tokens.shape[0])
        start_pos_t = start_pos if isinstance(start_pos, torch.Tensor) else torch.as_tensor(start_pos)
        max_start_pos = int(start_pos_t.max().item()) if start_pos_t.numel() > 0 else 0
        expected_blocks_per_seq = int(self.hf_config.max_seq_len // self.paged_config.block_size)
        expected_max_block_id = int(self.paged_config.max_num_blocks) - 1
        assert page_table.shape[0] == batch_size, (
            f"vLLM page_table.shape[0] (batch) mismatch: "
            f"page_table.shape[0]={page_table.shape[0]} vs tokens.shape[0]={batch_size}"
        )
        assert page_table.shape[1] <= expected_blocks_per_seq, (
            f"vLLM page_table.shape[1] (blocks_per_seq) too large: "
            f"page_table.shape[1]={page_table.shape[1]} vs "
            f"max_blocks_per_seq={expected_blocks_per_seq}"
        )
        page_table_t = page_table if isinstance(page_table, torch.Tensor) else torch.as_tensor(page_table)
        if page_table_t.numel() > 0:
            min_block_id = int(page_table_t.min().item())
            max_block_id = int(page_table_t.max().item())
            assert min_block_id >= 0, (
                f"vLLM page_table min block id is negative: " f"page_table.min()={min_block_id} vs expected>=0"
            )
            assert max_block_id <= expected_max_block_id, (
                f"vLLM page_table max block id out of range: "
                f"page_table.max()={max_block_id} vs max_allowed={expected_max_block_id}"
            )
        assert max_start_pos < self.hf_config.max_seq_len, (
            f"vLLM start_pos out of range: "
            f"max_start_pos={max_start_pos} vs hf_config.max_seq_len={self.hf_config.max_seq_len}"
        )
