# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.config import MeshConfig, ModeConfig


MODEL_ID = "google/gemma-4-12B"
ROOT = Path(__file__).resolve().parents[1]
MULTICHIP_DECODER_PATH = ROOT / "tt" / "multichip_decoder.py"
OPTIMIZED_DECODER_PATH = ROOT / "tt" / "optimized_decoder.py"
FUNCTIONAL_TEST_PATH = ROOT / "tests" / "test_functional_decoder.py"
EVIDENCE_DIR = ROOT / "doc" / "multichip_decoder"
PCC_RESULTS = EVIDENCE_DIR / "pcc_results.jsonl"

TARGET_MESH_SHAPE = (1, 8)
PREFILL_SEQ = int(os.getenv("GEMMA4_12B_MULTICHIP_PREFILL_SEQ", os.getenv("GEMMA4_12B_PREFILL_SEQ", "128")))
LONG_SEQ = int(os.getenv("GEMMA4_12B_MULTICHIP_LONG_SEQ", os.getenv("GEMMA4_12B_LONG_SEQ", "1024")))
PCC_THRESHOLD = float(os.getenv("GEMMA4_12B_MULTICHIP_PCC", "0.995"))
SLIDING_DECODE_PCC_THRESHOLD = float(os.getenv("GEMMA4_12B_MULTICHIP_SLIDING_DECODE_PCC", "0.993"))
LONG_CONTEXT_PCC_THRESHOLD = float(os.getenv("GEMMA4_12B_MULTICHIP_LONG_CONTEXT_PCC", "0.992"))
DETERMINISM_THRESHOLD = float(os.getenv("GEMMA4_12B_MULTICHIP_DETERMINISM_PCC", "0.9999"))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_functional = _load_module(FUNCTIONAL_TEST_PATH, "gemma4_12b_functional_decoder_tests_for_multichip")
_functional.EVIDENCE_DIR = EVIDENCE_DIR
_functional.PCC_RESULTS = PCC_RESULTS
_functional.PCC_THRESHOLD = PCC_THRESHOLD
_functional.SLIDING_DECODE_PCC_THRESHOLD = SLIDING_DECODE_PCC_THRESHOLD
_functional.LONG_CONTEXT_PCC_THRESHOLD = LONG_CONTEXT_PCC_THRESHOLD
_functional.PREFILL_SEQ = PREFILL_SEQ
_functional.LONG_SEQ = LONG_SEQ


def _load_optimized_decoder_class():
    return _load_module(OPTIMIZED_DECODER_PATH, "gemma4_12b_optimized_decoder_for_multichip_tests").OptimizedDecoder


def _load_multichip_decoder_class():
    return _load_module(MULTICHIP_DECODER_PATH, "gemma4_12b_multichip_decoder").MultichipDecoder


def _decode_threshold(layer_type: str, *, long_context: bool = False):
    if long_context:
        return LONG_CONTEXT_PCC_THRESHOLD
    if layer_type == "sliding_attention":
        return SLIDING_DECODE_PCC_THRESHOLD
    return PCC_THRESHOLD


def _require_t3k():
    if ttnn.get_num_devices() < 8:
        pytest.skip("google/gemma-4-12B multichip decoder target requires an 8-device T3K mesh")


@contextmanager
def _open_mesh(shape, fabric_config):
    ttnn.set_fabric_config(fabric_config)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*shape), trace_region_size=100000000)
    try:
        yield mesh_device
    finally:
        ttnn.close_mesh_device(mesh_device)
        gc.collect()


def _record_multichip(record):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with PCC_RESULTS.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _make_decoder(decoder_cls, layer_type, mesh_device, hf_layer, *, tp):
    text_config = _functional._hf_text_config()
    layer_idx = _functional._find_layer_idx(text_config, layer_type)
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp), prefill=ModeConfig(tp=tp))
    return decoder_cls.from_state_dict(
        hf_layer.state_dict(),
        hf_config=text_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        mesh_config=mesh_config,
    )


def _to_torch_first_device(tensor, mesh_device):
    return _functional._from_tt(tensor, mesh_device).squeeze(0).float().clone()


def _replica_pccs(tensor):
    device_tensors = ttnn.get_device_tensors(tensor)
    first = ttnn.to_torch(device_tensors[0]).float()
    pccs = []
    for idx, device_tensor in enumerate(device_tensors):
        current = ttnn.to_torch(device_tensor).float()
        passing, pcc = comp_pcc(first, current, DETERMINISM_THRESHOLD)
        assert passing, f"replica {idx} PCC {pcc} below {DETERMINISM_THRESHOLD}"
        pccs.append(pcc)
    return pccs


def _run_decoder_once(decoder_cls, layer_type, seq_len, mesh_shape, fabric_config, *, tp, hf_layer):
    with _open_mesh(mesh_shape, fabric_config) as mesh_device:
        decoder = _make_decoder(decoder_cls, layer_type, mesh_device, hf_layer, tp=tp)
        runtime = _functional._runtime_inputs(layer_type, seq_len, mesh_device, hf_layer=hf_layer, decoder=decoder)
        tt_prefill = decoder.prefill_forward(
            runtime["x_prefill_tt"],
            rope_mats=runtime["rope4"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
        )
        ttnn.synchronize_device(mesh_device)
        tt_decode = decoder.decode_forward(
            runtime["x_decode_tt"],
            rope_mats=runtime["rope2"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
            position_idx=runtime["pos_embed_tt"],
            position_idx_cache=runtime["pos_cache_tt"],
        )
        ttnn.synchronize_device(mesh_device)
        replica_pccs = _replica_pccs(tt_decode) if tp > 1 else [1.0]
        return {
            "prefill": _to_torch_first_device(tt_prefill, mesh_device),
            "decode": _to_torch_first_device(tt_decode, mesh_device),
            "summary": getattr(decoder, "multichip_summary", getattr(decoder, "optimization_summary", {})),
            "kv_cache_shape": list(runtime["kv_cache"][0].shape),
            "kv_cache_padded_shape": list(runtime["kv_cache"][0].padded_shape),
            "page_table_shape": list(runtime["page_table_tt"].shape),
            "position_idx_shape": list(runtime["pos_embed_tt"].shape),
            "position_idx_cache_shape": list(runtime["pos_cache_tt"].shape),
            "decode_replica_pccs": replica_pccs,
        }


def _compare_to_optimized(layer_type, seq_len, *, long_context=False):
    _require_t3k()
    text_config = _functional._hf_text_config()
    layer_idx = _functional._find_layer_idx(text_config, layer_type)
    hf_layer = _functional._synthetic_hf_layer(text_config, layer_idx)

    optimized = _run_decoder_once(
        _load_optimized_decoder_class(),
        layer_type,
        seq_len,
        (1, 1),
        ttnn.FabricConfig.DISABLED,
        tp=1,
        hf_layer=hf_layer,
    )
    multichip = _run_decoder_once(
        _load_multichip_decoder_class(),
        layer_type,
        seq_len,
        TARGET_MESH_SHAPE,
        ttnn.FabricConfig.FABRIC_1D_RING,
        tp=8,
        hf_layer=hf_layer,
    )

    prefill_passing, prefill_pcc = comp_pcc(optimized["prefill"], multichip["prefill"], PCC_THRESHOLD)
    decode_passing, decode_pcc = comp_pcc(
        optimized["decode"], multichip["decode"], _decode_threshold(layer_type, long_context=long_context)
    )
    assert prefill_passing, f"{layer_type} seq={seq_len} multichip prefill PCC {prefill_pcc}"
    assert decode_passing, f"{layer_type} seq={seq_len} multichip decode PCC {decode_pcc}"

    _record_multichip(
        {
            "layer_type": layer_type,
            "layer_idx": layer_idx,
            "seq_len": seq_len,
            "prefill_pcc_vs_optimized": prefill_pcc,
            "prefill_threshold": PCC_THRESHOLD,
            "decode_pcc_vs_optimized": decode_pcc,
            "decode_threshold": _decode_threshold(layer_type, long_context=long_context),
            "optimized_path_class": "OptimizedDecoder",
            "multichip_path_class": "MultichipDecoder",
            "multichip_summary": multichip["summary"],
            "kv_cache_shape": multichip["kv_cache_shape"],
            "kv_cache_padded_shape": multichip["kv_cache_padded_shape"],
            "page_table_shape": multichip["page_table_shape"],
            "position_idx_shape": multichip["position_idx_shape"],
            "position_idx_cache_shape": multichip["position_idx_cache_shape"],
            "decode_replica_pccs": multichip["decode_replica_pccs"],
        }
    )
    return optimized, multichip, prefill_pcc, decode_pcc


def test_multichip_runtime_fallback_audit_source_clean():
    source = MULTICHIP_DECODER_PATH.read_text()
    forbidden = ("ttnn.from_torch", "ttnn.to_torch", "FunctionalDecoder")
    found = [item for item in forbidden if item in source]
    assert not found, f"runtime fallback or functional fallback tokens found in multichip decoder source: {found}"


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_multichip_paged_prefill_then_decode_pcc_vs_optimized(layer_type):
    _compare_to_optimized(layer_type, PREFILL_SEQ)


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_multichip_long_context_paged_prefill_decode_vs_optimized(layer_type):
    if LONG_SEQ < 1024:
        pytest.skip(f"GEMMA4_12B_MULTICHIP_LONG_SEQ={LONG_SEQ} does not cover the sliding-window boundary")
    _compare_to_optimized(layer_type, LONG_SEQ, long_context=True)


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_multichip_decode_trace_replay_pcc_and_determinism_vs_optimized(layer_type):
    _require_t3k()
    text_config = _functional._hf_text_config()
    layer_idx = _functional._find_layer_idx(text_config, layer_type)
    hf_layer = _functional._synthetic_hf_layer(text_config, layer_idx)
    optimized = _run_decoder_once(
        _load_optimized_decoder_class(),
        layer_type,
        PREFILL_SEQ,
        (1, 1),
        ttnn.FabricConfig.DISABLED,
        tp=1,
        hf_layer=hf_layer,
    )

    with _open_mesh(TARGET_MESH_SHAPE, ttnn.FabricConfig.FABRIC_1D_RING) as mesh_device:
        decoder = _make_decoder(_load_multichip_decoder_class(), layer_type, mesh_device, hf_layer, tp=8)
        runtime = _functional._runtime_inputs(layer_type, PREFILL_SEQ, mesh_device, hf_layer=hf_layer, decoder=decoder)
        decoder.prefill_forward(
            runtime["x_prefill_tt"],
            rope_mats=runtime["rope4"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
        )
        ttnn.synchronize_device(mesh_device)

        decoder.decode_forward(
            runtime["x_decode_tt"],
            rope_mats=runtime["rope2"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
            position_idx=runtime["pos_embed_tt"],
            position_idx_cache=runtime["pos_cache_tt"],
        )
        ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = decoder.decode_forward(
            runtime["x_decode_tt"],
            rope_mats=runtime["rope2"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
            position_idx=runtime["pos_embed_tt"],
            position_idx_cache=runtime["pos_cache_tt"],
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        replay_0 = _to_torch_first_device(traced_output, mesh_device)
        replay_passing, replay_pcc = comp_pcc(
            optimized["decode"], replay_0, _decode_threshold(layer_type)
        )
        assert replay_passing, f"{layer_type} traced decode replay PCC {replay_pcc}"

        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        replay_1 = _to_torch_first_device(traced_output, mesh_device)
        ttnn.release_trace(mesh_device, trace_id)
        deterministic, determinism_pcc = comp_pcc(replay_0, replay_1, DETERMINISM_THRESHOLD)
        assert deterministic, f"{layer_type} repeated trace replay PCC {determinism_pcc}"
        replica_pccs = _replica_pccs(traced_output)

    _record_multichip(
        {
            "layer_type": layer_type,
            "layer_idx": layer_idx,
            "seq_len": PREFILL_SEQ,
            "trace_replay_pcc_vs_optimized": replay_pcc,
            "trace_replay_threshold": _decode_threshold(layer_type),
            "trace_determinism_pcc": determinism_pcc,
            "determinism_threshold": DETERMINISM_THRESHOLD,
            "trace_decode_replica_pccs": replica_pccs,
        }
    )


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_multichip_cache_and_stacked_layout_contract(layer_type):
    _require_t3k()
    text_config = _functional._hf_text_config()
    layer_idx = _functional._find_layer_idx(text_config, layer_type)
    hf_layer = _functional._synthetic_hf_layer(text_config, layer_idx)
    with _open_mesh(TARGET_MESH_SHAPE, ttnn.FabricConfig.FABRIC_1D_RING) as mesh_device:
        decoder = _make_decoder(_load_multichip_decoder_class(), layer_type, mesh_device, hf_layer, tp=8)
        runtime = _functional._runtime_inputs(layer_type, PREFILL_SEQ, mesh_device, hf_layer=hf_layer, decoder=decoder)
        expected_head_dim = 512 if layer_type == "full_attention" else 256
        expected_cache_shape = [runtime["kv_cache"][0].shape[0], 1, _functional.BLOCK_SIZE, expected_head_dim]
        assert list(runtime["kv_cache"][0].shape) == expected_cache_shape
        assert list(runtime["kv_cache"][1].shape) == expected_cache_shape
        assert list(runtime["page_table_tt"].shape)[0] == 1
        assert list(runtime["pos_embed_tt"].shape) == [1, 1]
        assert list(runtime["pos_cache_tt"].shape) == [1, 1]
        assert decoder.self_attn.local_kv_heads == 1
        assert decoder.self_attn.kv_replicated == (layer_type == "full_attention")

        tt_prefill = decoder.prefill_forward(
            runtime["x_prefill_tt"],
            rope_mats=runtime["rope4"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
        )
        ttnn.synchronize_device(mesh_device)
        tt_decode = decoder.decode_forward(
            runtime["x_decode_tt"],
            rope_mats=runtime["rope2"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
            position_idx=runtime["pos_embed_tt"],
            position_idx_cache=runtime["pos_cache_tt"],
        )
        ttnn.synchronize_device(mesh_device)
        assert list(tt_decode.shape) == list(runtime["x_decode_tt"].shape)
        assert tt_decode.memory_config() == decoder.decode_residual_memcfg
        replica_pccs = _replica_pccs(tt_decode)
        _record_multichip(
            {
                "layer_type": layer_type,
                "layer_idx": layer_idx,
                "seq_len": PREFILL_SEQ,
                "cache_contract_shape": expected_cache_shape,
                "decode_input_shape": list(runtime["x_decode_tt"].shape),
                "decode_output_shape": list(tt_decode.shape),
                "decode_output_memory_config": str(tt_decode.memory_config()),
                "prefill_output_shape": list(tt_prefill.shape),
                "stacked_layout_contract": "replicated residual input/output with per-device L1 width sharding in decode",
                "decode_replica_pccs": replica_pccs,
            }
        )


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_multichip_perf_warmed_prefill_and_traced_decode(layer_type):
    _require_t3k()
    text_config = _functional._hf_text_config()
    layer_idx = _functional._find_layer_idx(text_config, layer_type)
    hf_layer = _functional._synthetic_hf_layer(text_config, layer_idx)
    with _open_mesh(TARGET_MESH_SHAPE, ttnn.FabricConfig.FABRIC_1D_RING) as mesh_device:
        decoder = _make_decoder(_load_multichip_decoder_class(), layer_type, mesh_device, hf_layer, tp=8)
        runtime = _functional._runtime_inputs(layer_type, PREFILL_SEQ, mesh_device, hf_layer=hf_layer, decoder=decoder)

        decoder.prefill_forward(
            runtime["x_prefill_tt"],
            rope_mats=runtime["rope4"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
        )
        ttnn.synchronize_device(mesh_device)

        _functional._signpost("PERF_PREFILL")
        decoder.prefill_forward(
            runtime["x_prefill_tt"],
            rope_mats=runtime["rope4"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
        )
        ttnn.synchronize_device(mesh_device)
        _functional._signpost("PERF_PREFILL_END")

        decoder.decode_forward(
            runtime["x_decode_tt"],
            rope_mats=runtime["rope2"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
            position_idx=runtime["pos_embed_tt"],
            position_idx_cache=runtime["pos_cache_tt"],
        )
        ttnn.synchronize_device(mesh_device)

        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        decoder.decode_forward(
            runtime["x_decode_tt"],
            rope_mats=runtime["rope2"],
            page_table=runtime["page_table_tt"],
            kv_cache=runtime["kv_cache"],
            position_idx=runtime["pos_embed_tt"],
            position_idx_cache=runtime["pos_cache_tt"],
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)

        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        _functional._signpost("PERF_DECODE")
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh_device)
        _functional._signpost("PERF_DECODE_END")
        ttnn.release_trace(mesh_device, trace_id)
