# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import torch
from loguru import logger

from models.common.sampling.generator import SamplingParams
from models.demos.deepseek_v3.tt.generator import DeepseekGenerator
from models.demos.deepseek_v3.utils.config_dataclass import KvCacheConfig
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer
from models.tt_transformers.tt.common import PagedAttentionConfig


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

    @staticmethod
    def _shape_or_type(value):
        if value is None:
            return "None"
        if hasattr(value, "shape"):
            try:
                return tuple(value.shape)
            except Exception:
                return f"{type(value).__name__}(shape-unavailable)"
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}(len={len(value)})"
        return type(value).__name__

    def _log_sampling_params(
        self, callsite: str, sampling_params: SamplingParams | None, sample_on_device: bool | None = None
    ):
        if sample_on_device is not None:
            logger.info("{} sample_on_device={}", callsite, sample_on_device)
        if sampling_params is None:
            logger.info("{} sampling_params=None", callsite)
            return
        for field_name in (
            "temperature",
            "top_k",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "repetition_penalty",
            "seed",
            "enable_log_probs",
        ):
            field_val = getattr(sampling_params, field_name, None)
            logger.info("{} sampling_params.{}={}", callsite, field_name, self._shape_or_type(field_val))

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
        logger.info(f"prefill_forward, kwargs keys: {kwargs.keys()}")
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
        sampling_params = kwargs.get("sampling_params", None)
        sample_on_device_mode = kwargs.get("sample_on_device_mode", "unknown")
        # sample_on_device_mode is not passed by vLLM, TODO, check if its missing in the vLLM code
        # sample_on_device = sample_on_device_mode in ["all"]
        # sample_on_device = sampling_params is not None
        sample_on_device = True
        logger.info(
            "prefill_forward shapes: tokens={} lengths={} page_table={} kv_cache={} empty_slots={} sampling_params={} sample_on_device_mode={} sample_on_device={}",
            tuple(tokens.shape),
            len(lengths) if lengths is not None else 0,
            self._shape_or_type(page_table),
            self._shape_or_type(kv_cache),
            self._shape_or_type(empty_slots),
            self._shape_or_type(sampling_params),
            sample_on_device_mode,
            sample_on_device,
        )
        self._log_sampling_params("prefill_forward", sampling_params)
        # sampling_params can be None with server mode???
        # sample_on_device = sampling_params is not None

        sample_on_device = True

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
        prefill_tokens = []
        for i in range(num_of_users):
            logger.info(f"Prefill step {i}")
            user_id = empty_slots[i] if empty_slots is not None else i
            prompt_len = int(lengths[i])
            if prompt_len == 0:
                prefill_tokens.append(
                    torch.zeros(max_padded_len, self.hf_config.vocab_size, device=tokens.device, dtype=tokens.dtype)
                )
                continue
            user_tokens = tokens[i, :prompt_len].unsqueeze(0)
            user_tokens = _pad_tokens(user_tokens, pad_value, block_size=pad_block_size).squeeze(0)
            # prefill does not support tracing but sampling can be traced.
            # call it in every pefill??
            self._validate_and_initialize_sampling(
                sampling_params, sample_on_device, enable_trace=True, enable_mtp=False
            )
            prefill_logits = self._prefill(
                user_tokens,
                user_id,
                page_table,
                local_user_id=i,
                sampling_params=sampling_params,
                sample_on_device=sample_on_device,
            )

            if sample_on_device:
                prefill_logits = self._slice_last_token_logits(prefill_logits, prompt_len, expand_to_batch=True)
                prefill_logits_sampled_device = self._sample_tokens_device(prefill_logits, user_slots=[user_id])
                prefill_logits_sampled_host = self._tokens_from_device(
                    prefill_logits_sampled_device, self.mesh_device, batch_size_per_row=1
                )
                pred_token = prefill_logits_sampled_host[0]
                # ttnn.deallocate(prefill_logits)
                # ttnn.deallocate(prefill_logits_sampled_device)
            # else:
            #     # not tested. TODO, test this path.
            #     assert False, "Not tested"
            #     assert isinstance(
            #         prefill_logits, torch.Tensor
            #     ), "prefill_logits should be a torch.Tensor on host"
            #     last_token_logits = prefill_logits[0, 0, max(prompt_len - 1, 0), :]
            #     pred_token = int(
            #         self._sample_on_host(last_token_logits.unsqueeze(0), start_user_idx=user_id).item()
            #     )
            prefill_tokens.append(torch.tensor(pred_token, dtype=torch.int64))
        prefill_tokens = torch.stack(prefill_tokens)  # [num_of_users, S, V]

        logger.info(f"Prefill tokens shape: {prefill_tokens.shape}")

        return prefill_tokens

    def decode_forward(self, *args, **kwargs):
        logger.info(f"decode_forward, kwargs keys: {kwargs.keys()}")
        assert self.model_run_config_decode is not None, "Model run config decode is not initialized"

        page_tables = kwargs.get("page_table", None)
        kv_cache = kwargs.get("kv_cache", None)
        enable_trace = kwargs.get("enable_trace", False)
        read_from_device = kwargs.get("read_from_device", True)
        sampling_params = kwargs.get("sampling_params", None)
        sample_on_device_mode = kwargs.get("sample_on_device_mode", "unknown")

        # sample_on_device_mode is not passed by vLLM, TODO, check if its missing in the vLLM code
        # sample_on_device = sample_on_device_mode in ["all", "decode_only"]

        # sampling_params can be None for warmup. How to handle this?
        # sample_on_device = sampling_params is not None

        sample_on_device = True
        self._log_sampling_params("decode_forward", sampling_params, sample_on_device=sample_on_device)

        # Set kv_cache if provided and all entries are valid
        if kv_cache is not None and not any(entry is None for entry in kv_cache):
            self.set_kv_cache(kv_cache)

        tokens_step = kwargs["tokens"].squeeze(1)
        logger.info(
            "decode_forward shapes: tokens_step={} start_pos={} page_tables={} kv_cache={} sampling_params={} enable_trace={} read_from_device={} sample_on_device_mode={} sample_on_device={}",
            tuple(tokens_step.shape),
            self._shape_or_type(kwargs.get("start_pos")),
            self._shape_or_type(page_tables),
            self._shape_or_type(kv_cache),
            self._shape_or_type(sampling_params),
            enable_trace,
            read_from_device,
            sample_on_device_mode,
            sample_on_device,
        )
        # call it in every decode??
        self._validate_and_initialize_sampling(
            sampling_params, sample_on_device, enable_trace=enable_trace, enable_mtp=False
        )
        decode_logits = super().decode_forward(
            tokens=tokens_step,
            start_pos=kwargs["start_pos"],
            batch_size_per_row=USERS_PER_ROW,
            enable_trace=enable_trace,
            page_table=page_tables,
            sampling_params=sampling_params,
            sample_on_device=sample_on_device,
            read_from_device=read_from_device,
        )

        if sample_on_device:
            pred_tokens_device = self._sample_tokens_device(decode_logits, enable_trace=enable_trace)
            pred_tokens = self._tokens_from_device(
                pred_tokens_device, self.mesh_device, batch_size_per_row=self.batch_size_per_row
            )
            # if not self.enable_trace:
            #     ttnn.deallocate(decode_logits)
            #     ttnn.deallocate(pred_tokens_device)

        # else:
        #     assert False, "Not tested"
        #     # Normalize legacy decode outputs to [B, V], then expose vLLM shape [B, 1, V].
        #     # DeepseekGenerator may return either [1, 1, B, V] (non-trace path) or [B, V]
        #     # (trace path). vLLM sampling expects time-major logits [B, T, V].
        #     if logits.dim() == 4:
        #         logits = logits.squeeze(0).squeeze(0)
        #     elif logits.dim() == 3 and logits.shape[1] == 1:
        #         logits = logits.squeeze(1)
        #     elif logits.dim() != 2:
        #         raise RuntimeError(f"Unexpected decode logits rank for vLLM: shape={tuple(logits.shape)}")
        #     logits = logits.unsqueeze(1)

        return pred_tokens

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        assert (
            num_layers == self.hf_config.num_hidden_layers
        ), f"Number of layers {num_layers} does not match the number of layers in the model {self.hf_config.num_hidden_layers}"

        if kv_cache_shape is not None:
            block_size = int(kv_cache_shape[2])
            max_num_blocks = int(kv_cache_shape[0])
            if (
                self.paged_config is None
                or self.paged_config.block_size != block_size
                or self.paged_config.max_num_blocks != max_num_blocks
            ):
                logger.info(
                    "Aligning paged_config to vLLM kv_cache: block_size={} max_num_blocks={}",
                    block_size,
                    max_num_blocks,
                )
                self.paged_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
                self.page_tables_tt = None
        self.kv_cache_shape = kv_cache_shape

        kv_cache_config = KvCacheConfig(kv_cache_shape=kv_cache_shape, dtype=dtype)
        self._prepare_run_configs("prefill", kv_cache_override=kv_cache_config)
        self._prepare_run_configs("decode", kv_cache_override=kv_cache_config)
        kv_cache = self.get_kv_cache()

        return kv_cache
