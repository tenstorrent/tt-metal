# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Temporary harness: profile the LOCAL decode BASELINE (BFP8/BFP8, LoFi).

This reproduces the "first correct traced candidate" precision policy
(BFLOAT8_B attention + MLP weights, LoFi) so its device perf can be captured
under Tracy and compared with the optimized (BFP4/BFP4) final report.
"""

import os

import pytest
import torch

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (
    LARGE_SEQ_LEN,
    SMALL_SEQ_LEN,
    _synthetic_state_dict,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_optimized_decoder import (  # noqa: F401
    _tt_tensor,
    hf_config,
    mesh_device,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizedDecoder, PagedKVConfig


def test_baseline_decode_perf(hf_config, mesh_device):
    if os.environ.get("LLAMA31_8B_BASELINE_PERF") != "1":
        pytest.skip("set LLAMA31_8B_BASELINE_PERF=1 to run baseline decode perf")

    prefix_len = SMALL_SEQ_LEN
    total_len = prefix_len + 1
    state_dict = _synthetic_state_dict(hf_config)
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_seq_len=LARGE_SEQ_LEN,
        paged_kv_config=PagedKVConfig(max_num_blocks=2, block_size=32),
        attention_weight_dtype=ttnn.bfloat8_b,
        mlp_weight_dtype=ttnn.bfloat8_b,
        attention_math_fidelity=ttnn.MathFidelity.LoFi,
        mlp_math_fidelity=ttnn.MathFidelity.LoFi,
    )
    hidden_states = torch.randn(1, total_len, hf_config.hidden_size, dtype=torch.bfloat16)

    tt_prefill = _tt_tensor(
        hidden_states[:, :prefix_len, :].reshape(1, 1, prefix_len, hf_config.hidden_size), mesh_device
    )
    kv_cache = decoder.init_paged_kv_cache()
    page_table = decoder.make_identity_page_table()
    decoder.prefill_forward(tt_prefill, kv_cache=kv_cache, page_table=page_table)
    position_cos, position_sin = decoder.position_tables_for_decode(prefix_len)
    current_pos = decoder.make_current_pos([prefix_len])
    tt_decode_input = _tt_tensor(hidden_states[:, prefix_len:, :].reshape(1, 1, 1, hf_config.hidden_size), mesh_device)
    trace_id, _ = decoder.trace_decode_once(
        tt_decode_input,
        current_pos=current_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        position_cos=position_cos,
        position_sin=position_sin,
    )
    assert trace_id is not None
    print(f"BASELINE_TRACED_DECODE_MS={decoder.timings.traced_decode_ms:.6f}")
