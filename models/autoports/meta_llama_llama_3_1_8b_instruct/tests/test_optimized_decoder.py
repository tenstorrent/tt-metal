# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
from contextlib import contextmanager

import pytest
import torch
from transformers import DynamicCache

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_functional_decoder import (
    EMITTED_BATCH,
    LAYER_IDX,
    _pcc,
    _real_layer_state_dict,
    _reference_layer,
    _rope,
    _synthetic_state_dict,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizedDecoder


def _signpost(name: str) -> None:
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(name)


@contextmanager
def _temporary_env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


@pytest.fixture(scope="module")
def mesh_device():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=100000000)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", local_files_only=True)


def _make_decoder(config, state_dict, mesh_device, max_seq_len=64):
    return OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        max_seq_len=max_seq_len,
        page_block_size=32,
    )


def _prepare_prefill_case(hf_config, mesh_device, seq_len=4, *, decoder_cls=OptimizedDecoder):
    state_dict = _real_layer_state_dict()
    torch.manual_seed(4321)
    hidden = torch.randn(EMITTED_BATCH, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, seq_len)
    decoder = decoder_cls.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
        max_seq_len=64,
        page_block_size=32,
    )
    tt_hidden = decoder_cls.prepare_inputs(hidden, mesh_device)
    tt_cos, tt_sin = decoder_cls.prepare_rope(cos, sin, mesh_device)
    return decoder, tt_hidden, tt_cos, tt_sin


def _run_prefill_warmed_perf_only(decoder, tt_hidden, tt_cos, tt_sin, mesh_device, signpost_prefix: str):
    _ = decoder.prefill_forward(tt_hidden, position_cos=tt_cos, position_sin=tt_sin)
    ttnn.synchronize_device(mesh_device)
    _signpost(signpost_prefix)
    _ = decoder.prefill_forward(tt_hidden, position_cos=tt_cos, position_sin=tt_sin)
    ttnn.synchronize_device(mesh_device)
    _signpost(f"{signpost_prefix}_END")


def test_baseline_prefill_warmed_perf_only(hf_config, mesh_device):
    decoder, tt_hidden, tt_cos, tt_sin = _prepare_prefill_case(
        hf_config, mesh_device, seq_len=4, decoder_cls=FunctionalDecoder
    )
    _run_prefill_warmed_perf_only(decoder, tt_hidden, tt_cos, tt_sin, mesh_device, "PERF_PREFILL_BASELINE")


def test_optimized_prefill_warmed_perf_only(hf_config, mesh_device):
    decoder, tt_hidden, tt_cos, tt_sin = _prepare_prefill_case(hf_config, mesh_device, seq_len=4)
    _run_prefill_warmed_perf_only(decoder, tt_hidden, tt_cos, tt_sin, mesh_device, "PERF_PREFILL")


def test_optimized_prefill_dram_candidate_warmed_perf_only(hf_config, mesh_device):
    with _temporary_env("TT_AUTOOPT_LLAMA_PREFILL_L1_ACTIVATIONS", "0"):
        decoder, tt_hidden, tt_cos, tt_sin = _prepare_prefill_case(hf_config, mesh_device, seq_len=4)
        _run_prefill_warmed_perf_only(decoder, tt_hidden, tt_cos, tt_sin, mesh_device, "PERF_PREFILL_DRAM")


@pytest.mark.parametrize("seq_len", [4, 8])
def test_optimized_prefill_real_weight_pcc(hf_config, mesh_device, seq_len):
    state_dict = _real_layer_state_dict()
    torch.manual_seed(4321)
    hidden = torch.randn(EMITTED_BATCH, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, seq_len)
    ref = _reference_layer(hf_config, state_dict)
    attention_mask = torch.full((EMITTED_BATCH, 1, seq_len, seq_len), torch.finfo(torch.float32).min)
    attention_mask = torch.triu(attention_mask, diagonal=1).to(torch.bfloat16)
    with torch.no_grad():
        expected = ref(hidden, attention_mask=attention_mask, position_embeddings=(cos, sin))

    decoder = _make_decoder(hf_config, state_dict, mesh_device)
    tt_hidden = OptimizedDecoder.prepare_inputs(hidden, mesh_device)
    tt_cos, tt_sin = OptimizedDecoder.prepare_rope(cos, sin, mesh_device)
    _signpost("PERF_PREFILL")
    actual = ttnn.to_torch(decoder.prefill_forward(tt_hidden, position_cos=tt_cos, position_sin=tt_sin)).squeeze(0)
    _signpost("PERF_PREFILL_END")
    measured = _pcc(expected, actual)
    print(f"OPT_REAL_PREFILL_SEQ_{seq_len}_PCC={measured:.6f}")
    assert measured >= 0.99


@pytest.mark.parametrize("weight_kind", ["synthetic", "real"])
def test_optimized_decode_paged_kv_pcc(hf_config, mesh_device, weight_kind):
    state_dict = _synthetic_state_dict(hf_config) if weight_kind == "synthetic" else _real_layer_state_dict()
    decoder = _make_decoder(hf_config, state_dict, mesh_device)
    torch.manual_seed(9876)
    hidden = torch.randn(EMITTED_BATCH, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, 1)

    ref = _reference_layer(hf_config, state_dict)
    cache = DynamicCache()
    position_ids = torch.zeros((EMITTED_BATCH, 1), dtype=torch.long)
    with torch.no_grad():
        expected = ref(
            hidden,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_value=cache,
            use_cache=True,
        )

    page_table = OptimizedDecoder.build_contiguous_page_table(EMITTED_BATCH, max_seq_len=64, block_size=32)
    tt_hidden = OptimizedDecoder.prepare_decode_inputs(hidden, mesh_device, decoder.decode_input_memcfg)
    tt_pos = OptimizedDecoder.prepare_decode_positions(position_ids[:, 0], mesh_device)
    tt_page_table = OptimizedDecoder.prepare_page_table(page_table, mesh_device)
    tt_cos, tt_sin = OptimizedDecoder.prepare_decode_rope(cos[:, 0:1, :], sin[:, 0:1, :], mesh_device)

    _signpost("PERF_DECODE")
    actual = decoder.decode_forward(tt_hidden, current_pos=tt_pos, rot_mats=(tt_cos, tt_sin), page_table=tt_page_table)
    _signpost("PERF_DECODE_END")
    actual_torch = ttnn.to_torch(actual)[0, 0, :EMITTED_BATCH, : hf_config.hidden_size].reshape(
        EMITTED_BATCH, 1, hf_config.hidden_size
    )
    measured = _pcc(expected, actual_torch)
    print(f"OPT_{weight_kind.upper()}_DECODE_PAGED_SEQ_1_PCC={measured:.6f}")
    print(f"OPT_DECODE_KV_CACHE_DTYPE={decoder.evidence.kv_cache_dtype}")
    assert decoder.evidence.decode_uses_common_optimized_block
    assert "bfloat8" in decoder.evidence.kv_cache_dtype.lower()
    assert measured >= (0.95 if weight_kind == "synthetic" else 0.99)


def test_optimized_decode_paged_kv_nonzero_position_pcc(hf_config, mesh_device):
    state_dict = _real_layer_state_dict()
    decoder = _make_decoder(hf_config, state_dict, mesh_device)
    torch.manual_seed(9753)
    hidden = torch.randn(EMITTED_BATCH, 2, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, 2)

    ref = _reference_layer(hf_config, state_dict)
    cache = DynamicCache()
    with torch.no_grad():
        _ = ref(
            hidden[:, 0:1, :],
            position_ids=torch.zeros((EMITTED_BATCH, 1), dtype=torch.long),
            position_embeddings=(cos[:, 0:1, :], sin[:, 0:1, :]),
            past_key_value=cache,
            use_cache=True,
        )
        expected = ref(
            hidden[:, 1:2, :],
            position_ids=torch.ones((EMITTED_BATCH, 1), dtype=torch.long),
            position_embeddings=(cos[:, 1:2, :], sin[:, 1:2, :]),
            past_key_value=cache,
            use_cache=True,
        )

    page_table = OptimizedDecoder.build_contiguous_page_table(EMITTED_BATCH, max_seq_len=64, block_size=32)
    tt_page_table = OptimizedDecoder.prepare_page_table(page_table, mesh_device)

    tt_hidden0 = OptimizedDecoder.prepare_decode_inputs(hidden[:, 0:1, :], mesh_device, decoder.decode_input_memcfg)
    tt_pos0 = OptimizedDecoder.prepare_decode_positions(torch.zeros((EMITTED_BATCH,), dtype=torch.long), mesh_device)
    tt_cos0, tt_sin0 = OptimizedDecoder.prepare_decode_rope(cos[:, 0:1, :], sin[:, 0:1, :], mesh_device)
    _ = decoder.decode_forward(tt_hidden0, current_pos=tt_pos0, rot_mats=(tt_cos0, tt_sin0), page_table=tt_page_table)

    tt_hidden1 = OptimizedDecoder.prepare_decode_inputs(hidden[:, 1:2, :], mesh_device, decoder.decode_input_memcfg)
    tt_pos1 = OptimizedDecoder.prepare_decode_positions(torch.ones((EMITTED_BATCH,), dtype=torch.long), mesh_device)
    tt_cos1, tt_sin1 = OptimizedDecoder.prepare_decode_rope(cos[:, 1:2, :], sin[:, 1:2, :], mesh_device)
    actual = decoder.decode_forward(
        tt_hidden1, current_pos=tt_pos1, rot_mats=(tt_cos1, tt_sin1), page_table=tt_page_table
    )
    actual_torch = ttnn.to_torch(actual)[0, 0, :EMITTED_BATCH, : hf_config.hidden_size].reshape(
        EMITTED_BATCH, 1, hf_config.hidden_size
    )
    measured = _pcc(expected, actual_torch)
    print(f"OPT_REAL_DECODE_PAGED_SEQ_2_POS_1_PCC={measured:.6f}")
    assert measured >= 0.99


def test_optimized_decode_repeated_deterministic(hf_config, mesh_device):
    state_dict = _synthetic_state_dict(hf_config)
    page_table = OptimizedDecoder.build_contiguous_page_table(EMITTED_BATCH, max_seq_len=64, block_size=32)
    outputs = []
    for _ in range(2):
        decoder = _make_decoder(hf_config, state_dict, mesh_device)
        torch.manual_seed(2468)
        hidden = torch.randn(EMITTED_BATCH, 1, hf_config.hidden_size, dtype=torch.bfloat16)
        cos, sin = _rope(hf_config, hidden, 1)
        tt_hidden = OptimizedDecoder.prepare_decode_inputs(hidden, mesh_device, decoder.decode_input_memcfg)
        positions = torch.zeros((EMITTED_BATCH,), dtype=torch.long)
        tt_pos = OptimizedDecoder.prepare_decode_positions(positions, mesh_device)
        tt_page_table = OptimizedDecoder.prepare_page_table(page_table, mesh_device)
        tt_cos, tt_sin = OptimizedDecoder.prepare_decode_rope(cos[:, 0:1, :], sin[:, 0:1, :], mesh_device)
        actual = decoder.decode_forward(
            tt_hidden, current_pos=tt_pos, rot_mats=(tt_cos, tt_sin), page_table=tt_page_table
        )
        outputs.append(
            ttnn.to_torch(actual)[0, 0, :EMITTED_BATCH, : hf_config.hidden_size].reshape(
                EMITTED_BATCH, 1, hf_config.hidden_size
            )
        )
    measured = _pcc(outputs[0], outputs[1])
    print(f"OPT_DECODE_REPEAT_DETERMINISM_PCC={measured:.6f}")
    assert measured >= 0.9999


def test_optimized_decode_trace_replay_real_weight_pcc(hf_config, mesh_device):
    state_dict = _real_layer_state_dict()
    decoder = _make_decoder(hf_config, state_dict, mesh_device)
    torch.manual_seed(1357)
    hidden = torch.randn(EMITTED_BATCH, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(hf_config, hidden, 1)

    ref = _reference_layer(hf_config, state_dict)
    cache = DynamicCache()
    position_ids = torch.zeros((EMITTED_BATCH, 1), dtype=torch.long)
    with torch.no_grad():
        expected = ref(
            hidden,
            position_ids=position_ids,
            position_embeddings=(cos, sin),
            past_key_value=cache,
            use_cache=True,
        )

    page_table = OptimizedDecoder.build_contiguous_page_table(EMITTED_BATCH, max_seq_len=64, block_size=32)
    tt_hidden = OptimizedDecoder.prepare_decode_inputs(hidden, mesh_device, decoder.decode_input_memcfg)
    tt_pos = OptimizedDecoder.prepare_decode_positions(position_ids[:, 0], mesh_device)
    tt_page_table = OptimizedDecoder.prepare_page_table(page_table, mesh_device)
    tt_cos, tt_sin = OptimizedDecoder.prepare_decode_rope(cos[:, 0:1, :], sin[:, 0:1, :], mesh_device)

    _ = decoder.decode_forward(tt_hidden, current_pos=tt_pos, rot_mats=(tt_cos, tt_sin), page_table=tt_page_table)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(tt_hidden, current_pos=tt_pos, rot_mats=(tt_cos, tt_sin), page_table=tt_page_table)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    _signpost("PERF_DECODE_TRACE")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    _signpost("PERF_DECODE_TRACE_END")
    actual_torch = ttnn.to_torch(actual)[0, 0, :EMITTED_BATCH, : hf_config.hidden_size].reshape(
        EMITTED_BATCH, 1, hf_config.hidden_size
    )
    ttnn.release_trace(mesh_device, trace_id)
    measured = _pcc(expected, actual_torch)
    print(f"OPT_REAL_DECODE_TRACE_REPLAY_PCC={measured:.6f}")
    assert measured >= 0.99


def test_optimized_decode_runtime_source_has_no_functional_fallbacks():
    source = inspect.getsource(OptimizedDecoder.decode_forward)
    forbidden = ("FunctionalDecoder.decode_forward", "super().decode_forward", "torch", "from_torch", "to_torch")
    assert not any(token in source for token in forbidden)


def test_optimized_prefill_runtime_source_has_no_functional_fallbacks():
    source = inspect.getsource(OptimizedDecoder.prefill_forward)
    forbidden = ("FunctionalDecoder.prefill_forward", "super().prefill_forward", "torch", "from_torch", "to_torch")
    assert not any(token in source for token in forbidden)
