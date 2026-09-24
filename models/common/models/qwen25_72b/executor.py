# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen2.5-72B construction and concrete compatibility entry points."""

from models.common.models.qwen2_executor import Qwen25_72BExecutor, Qwen25_72BExecutorConfig
from models.common.models.qwen2_executor import build_qwen25_72b_executor as _build_qwen25_72b_executor
from models.common.models.qwen25_72b.hf_adaptor import Qwen25_72BForCausalLM
from models.common.models.qwen25_72b.model import _slice_last_token_tile


def build_qwen25_72b_executor(
    llm: Qwen25_72BForCausalLM,
    config: Qwen25_72BExecutorConfig,
) -> Qwen25_72BExecutor:
    return _build_qwen25_72b_executor(llm, config)


def run_prefill(model, token_ids_tt, *, start_pos: int = 0):
    return model.prefill_from_token_ids(token_ids_tt, start_pos=start_pos)


def run_decode(model, token_id_tt, *, current_pos: int):
    return model.decode_from_token_ids(token_id_tt, current_pos=current_pos)


def run_lm_head(model, hidden_tt):
    if len(hidden_tt.shape) == 4 and hidden_tt.shape[2] > 32:
        old = hidden_tt
        hidden_tt = _slice_last_token_tile(old, hidden_tt.shape[2] - 1)
        old.deallocate(True)
    return model.lm_logits(hidden_tt)
