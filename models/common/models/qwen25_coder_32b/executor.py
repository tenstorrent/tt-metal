# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-Coder-32B construction and compatibility entry points."""

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.models.qwen2_executor import (
    Qwen25Coder32BExecutor,
    Qwen25Coder32BExecutorConfig,
    build_qwen25_coder_32b_executor as _build_qwen25_coder_32b_executor,
)
from models.common.models.qwen25_coder_32b.hf_adaptor import Qwen25Coder32BForCausalLM
from models.common.models.qwen25_coder_32b.model import _slice_last_token_tile


def build_qwen25_coder_32b_executor(
    llm: Qwen25Coder32BForCausalLM,
    config: Qwen25Coder32BExecutorConfig,
) -> Qwen25Coder32BExecutor:
    return _build_qwen25_coder_32b_executor(llm, config)


def _compat_executor_config(model, *, trace_mode: str, device_sampling_enabled: bool) -> Qwen25Coder32BExecutorConfig:
    runtime_config = getattr(model, "model_args", None)
    if runtime_config is None:
        raise ValueError("Qwen25Coder32B compatibility executor requires model.model_args")
    block_size = 32
    max_seq_len = int(getattr(runtime_config, "max_seq_len", model.config.max_seq_len))
    max_batch_size = int(getattr(runtime_config, "max_batch_size", model.config.max_batch_size))
    max_num_blocks = ((max_seq_len + block_size - 1) // block_size) * max_batch_size
    return Qwen25Coder32BExecutorConfig(
        trace=TraceConfig(mode=trace_mode),
        warmup=WarmupConfig(
            prefill_seq_lens=tuple(getattr(runtime_config, "trace_prefill_supported_seq_lens", (128, 1024))),
            prefill_batch_sizes=(1,),
            include_decode_top_k=device_sampling_enabled,
        ),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            dtype=getattr(runtime_config, "kv_cache_dtype", ttnn.bfloat8_b),
        ),
        device_sampling_enabled=device_sampling_enabled,
    )


class EagerQwen25Coder32BExecutor(Qwen25Coder32BExecutor):
    """Compatibility wrapper over the shared eager runtime."""

    def __init__(self, model, mesh_device):
        del mesh_device
        super().__init__(
            model,
            model.model_args,
            _compat_executor_config(model, trace_mode="none", device_sampling_enabled=False),
        )


class TracedQwen25Coder32BExecutor(Qwen25Coder32BExecutor):
    """Compatibility wrapper over the shared traced runtime."""

    def __init__(
        self,
        model,
        mesh_device,
        ondevice_decode_loop: bool = False,
        fast_prefill_last_token: bool = False,
    ):
        del mesh_device, fast_prefill_last_token
        super().__init__(
            model,
            model.model_args,
            _compat_executor_config(
                model,
                trace_mode="all",
                device_sampling_enabled=bool(ondevice_decode_loop),
            ),
        )


def run_prefill(model, token_ids_tt, *, start_pos: int = 0):
    return model.prefill_from_token_ids(token_ids_tt, start_pos=start_pos)


def run_decode(model, token_id_tt, *, current_pos: int):
    return model.decode_from_token_ids(token_id_tt, current_pos=current_pos)


def run_lm_head(model, hidden_tt):
    if len(hidden_tt.shape) == 4 and hidden_tt.shape[2] > 32:
        old = hidden_tt
        hidden_tt = _slice_last_token_tile(old, hidden_tt.shape[2] - 1)
        ttnn.deallocate(old)
    return model.lm_logits(hidden_tt)
