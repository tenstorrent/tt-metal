# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


MODEL_ID = "google/gemma-4-12B"
ROOT = Path(__file__).resolve().parents[1]
OPTIMIZED_DECODER_PATH = ROOT / "tt" / "optimized_decoder.py"
FUNCTIONAL_TEST_PATH = ROOT / "tests" / "test_functional_decoder.py"
EVIDENCE_DIR = ROOT / "doc" / "optimized_decoder"
PCC_RESULTS = EVIDENCE_DIR / "pcc_results.jsonl"

PCC_THRESHOLD = float(os.getenv("GEMMA4_12B_OPTIMIZED_PCC", os.getenv("GEMMA4_12B_FUNCTIONAL_PCC", "0.995")))
SLIDING_DECODE_PCC_THRESHOLD = float(
    os.getenv("GEMMA4_12B_OPTIMIZED_SLIDING_DECODE_PCC", os.getenv("GEMMA4_12B_SLIDING_DECODE_PCC", "0.993"))
)
LONG_CONTEXT_PCC_THRESHOLD = float(
    os.getenv("GEMMA4_12B_OPTIMIZED_LONG_CONTEXT_PCC", os.getenv("GEMMA4_12B_LONG_CONTEXT_PCC", "0.992"))
)
PREFILL_SEQ = int(os.getenv("GEMMA4_12B_PREFILL_SEQ", "128"))
LONG_SEQ = int(os.getenv("GEMMA4_12B_LONG_SEQ", "1024"))
STRESS_ITERS = int(os.getenv("GEMMA4_12B_OPTIMIZED_STRESS_ITERS", "3"))


pytestmark = pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((1, 1), {"trace_region_size": 0}, id="1x1")],
    indirect=True,
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_functional = _load_module(FUNCTIONAL_TEST_PATH, "gemma4_12b_functional_decoder_tests_for_optimized")
_functional.EVIDENCE_DIR = EVIDENCE_DIR
_functional.PCC_RESULTS = PCC_RESULTS
_functional.PCC_THRESHOLD = PCC_THRESHOLD
_functional.SLIDING_DECODE_PCC_THRESHOLD = SLIDING_DECODE_PCC_THRESHOLD
_functional.LONG_CONTEXT_PCC_THRESHOLD = LONG_CONTEXT_PCC_THRESHOLD
_functional.PREFILL_SEQ = PREFILL_SEQ
_functional.LONG_SEQ = LONG_SEQ


def _load_optimized_decoder_class():
    module = _load_module(OPTIMIZED_DECODER_PATH, "gemma4_12b_optimized_decoder")
    return module.OptimizedDecoder


def _dtype_from_env(name: str, default):
    value = os.getenv(name, "").lower()
    if value in ("", "default"):
        return default
    if value in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if value in ("bfp8", "bfloat8_b"):
        return ttnn.bfloat8_b
    if value in ("bfp4", "bfloat4_b") and hasattr(ttnn, "bfloat4_b"):
        return ttnn.bfloat4_b
    raise ValueError(f"unsupported {name}={value!r}")


def _decode_threshold(layer_type):
    if layer_type == "sliding_attention":
        return SLIDING_DECODE_PCC_THRESHOLD
    return PCC_THRESHOLD


def _make_optimized_decoder_and_hf(layer_type, mesh_device, hf_layer=None):
    text_config = _functional._hf_text_config()
    layer_idx = _functional._find_layer_idx(text_config, layer_type)
    hf_layer = hf_layer or _functional._synthetic_hf_layer(text_config, layer_idx)
    model_args = Gemma4ModelArgs.from_hf_config(text_config)
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
    OptimizedDecoder = _load_optimized_decoder_class()
    decoder = OptimizedDecoder.from_state_dict(
        hf_layer.state_dict(),
        hf_config=text_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        mesh_config=mesh_config,
        dtype=_dtype_from_env("GEMMA4_12B_OPT_ACTIVATION_DTYPE", ttnn.bfloat16),
        attention_dtype=_dtype_from_env("GEMMA4_12B_OPT_ATTENTION_DTYPE", None),
        attention_qkv_dtype=_dtype_from_env("GEMMA4_12B_OPT_ATTENTION_QKV_DTYPE", None),
        attention_o_dtype=_dtype_from_env("GEMMA4_12B_OPT_ATTENTION_O_DTYPE", None),
        shared_mlp_dtype=_dtype_from_env("GEMMA4_12B_OPT_MLP_DTYPE", ttnn.bfloat8_b),
        shared_mlp_down_dtype=_dtype_from_env(
            "GEMMA4_12B_OPT_MLP_DOWN_DTYPE", _dtype_from_env("GEMMA4_12B_OPT_MLP_DTYPE", ttnn.bfloat8_b)
        ),
        shared_mlp_decode_dtype=_dtype_from_env("GEMMA4_12B_OPT_MLP_DECODE_DTYPE", None),
        shared_mlp_decode_down_dtype=_dtype_from_env("GEMMA4_12B_OPT_MLP_DECODE_DOWN_DTYPE", None),
        kv_cache_dtype=_dtype_from_env("GEMMA4_12B_OPT_KV_CACHE_DTYPE", ttnn.bfloat16),
        fuse_mlp_gelu=os.getenv("GEMMA4_12B_OPT_FUSE_MLP_GELU", "1") != "0",
        decode_norm_sharded=os.getenv("GEMMA4_12B_OPT_DECODE_NORM_SHARDED", "1") != "0",
        attention_decode_o_interleaved=os.getenv("GEMMA4_12B_OPT_ATTENTION_DECODE_O_INTERLEAVED", "0") == "1",
    )
    assert decoder.__class__.__name__ == "OptimizedDecoder"
    return text_config, model_args, layer_idx, hf_layer, decoder


def _run_optimized_prefill_then_decode(
    layer_type,
    seq_len,
    mesh_device,
    hf_layer=None,
    decoder=None,
    *,
    prefill_threshold=PCC_THRESHOLD,
    decode_threshold=None,
    weights_label="optimized_synthetic",
):
    if decoder is None:
        _, _, _, hf_layer, decoder = _make_optimized_decoder_and_hf(layer_type, mesh_device, hf_layer)
    elif hf_layer is None:
        raise ValueError("hf_layer must be supplied when decoder is supplied")

    return _functional._run_prefill_then_decode(
        layer_type,
        seq_len,
        mesh_device,
        hf_layer=hf_layer,
        decoder=decoder,
        prefill_threshold=prefill_threshold,
        decode_threshold=_decode_threshold(layer_type) if decode_threshold is None else decode_threshold,
        weights_label=weights_label,
    )


def _record_optimized(record):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with PCC_RESULTS.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def test_optimized_runtime_fallback_audit_source_clean(mesh_device):
    source = OPTIMIZED_DECODER_PATH.read_text()
    forbidden = ("import torch", "ttnn.from_torch", "ttnn.to_torch", "FunctionalDecoder")
    found = [item for item in forbidden if item in source]
    assert not found, f"runtime fallback or functional fallback tokens found in optimized decoder source: {found}"


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_optimized_paged_prefill_then_decode_pcc(layer_type, mesh_device):
    result = _run_optimized_prefill_then_decode(layer_type, PREFILL_SEQ, mesh_device)
    assert result["prefill_pcc"] >= PCC_THRESHOLD
    assert result["decode_pcc"] >= _decode_threshold(layer_type)
    _record_optimized(
        {
            "layer_type": layer_type,
            "layer_idx": result["layer_idx"],
            "seq_len": PREFILL_SEQ,
            "optimized_path_class": result["decoder"].__class__.__name__,
            "optimization_summary": result["decoder"].optimization_summary,
        }
    )


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_optimized_decode_trace_replay_pcc_and_determinism(layer_type, mesh_device):
    result = _run_optimized_prefill_then_decode(layer_type, PREFILL_SEQ, mesh_device)
    decoder = result["decoder"]
    seq_len = result["seq_len"]

    x_decode_tt = _functional._to_tt(result["x_decode"].unsqueeze(0).to(torch.bfloat16), mesh_device)
    pos_embed_tt = _functional._to_tt(
        torch.tensor([[seq_len]], dtype=torch.uint32), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )
    pos_cache_tt = _functional._to_tt(
        torch.tensor([[seq_len]], dtype=torch.int32), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32
    )

    decoder.decode_forward(
        x_decode_tt,
        rope_mats=result["rope2"],
        page_table=result["page_table_tt"],
        kv_cache=result["kv_cache"],
        position_idx=pos_embed_tt,
        position_idx_cache=pos_cache_tt,
    )
    ttnn.synchronize_device(mesh_device)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        x_decode_tt,
        rope_mats=result["rope2"],
        page_table=result["page_table_tt"],
        kv_cache=result["kv_cache"],
        position_idx=pos_embed_tt,
        position_idx_cache=pos_cache_tt,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    replay_0 = _functional._from_tt(traced_output, mesh_device).squeeze(0)
    replay_pcc = _functional._assert_pcc(
        replay_0,
        result["hf_decode"].float(),
        f"{layer_type} optimized traced decode replay",
        threshold=_decode_threshold(layer_type),
    )

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    replay_1 = _functional._from_tt(traced_output, mesh_device).squeeze(0)
    ttnn.release_trace(mesh_device, trace_id)
    determinism_pcc = _functional._assert_pcc(
        replay_1, replay_0, f"{layer_type} optimized repeated traced decode", threshold=0.9999
    )
    _record_optimized(
        {
            "layer_type": layer_type,
            "layer_idx": result["layer_idx"],
            "seq_len": seq_len,
            "optimized_trace_replay_pcc": replay_pcc,
            "trace_replay_threshold": _decode_threshold(layer_type),
            "optimized_determinism_pcc": determinism_pcc,
            "determinism_threshold": 0.9999,
        }
    )


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_optimized_long_context_paged_prefill_decode(layer_type, mesh_device):
    if LONG_SEQ < 1024:
        pytest.skip(f"GEMMA4_12B_LONG_SEQ={LONG_SEQ} does not cover the sliding-window boundary")
    result = _run_optimized_prefill_then_decode(
        layer_type,
        LONG_SEQ,
        mesh_device,
        prefill_threshold=LONG_CONTEXT_PCC_THRESHOLD,
        decode_threshold=LONG_CONTEXT_PCC_THRESHOLD,
        weights_label="optimized_synthetic_long",
    )
    assert result["prefill_pcc"] >= LONG_CONTEXT_PCC_THRESHOLD
    assert result["decode_pcc"] >= LONG_CONTEXT_PCC_THRESHOLD


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_optimized_repeated_prefill_decode_stress(layer_type, mesh_device):
    pccs = []
    for idx in range(STRESS_ITERS):
        result = _run_optimized_prefill_then_decode(
            layer_type,
            PREFILL_SEQ,
            mesh_device,
            weights_label=f"optimized_stress_{idx}",
        )
        pccs.append((result["prefill_pcc"], result["decode_pcc"]))
    _record_optimized({"layer_type": layer_type, "stress_iters": STRESS_ITERS, "pcc_pairs": pccs})


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_optimized_perf_warmed_prefill_and_traced_decode(layer_type, mesh_device):
    _, _, _, hf_layer, decoder = _make_optimized_decoder_and_hf(layer_type, mesh_device)
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


def test_optimized_real_weight_layer0_prefill_decode(mesh_device):
    weight_file = _functional._find_local_real_weight_file()
    if weight_file is None:
        pytest.skip("real google/gemma-4-12B model.safetensors is not present locally")

    from safetensors import safe_open
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    text_config = _functional._hf_text_config()
    layer_idx = _functional._find_layer_idx(text_config, "sliding_attention")
    hf_layer = Gemma4TextDecoderLayer(text_config, layer_idx=layer_idx)

    prefixes = (f"model.language_model.layers.{layer_idx}.", f"language_model.layers.{layer_idx}.", f"layers.{layer_idx}.")
    layer_state = {}
    with safe_open(weight_file, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            for prefix in prefixes:
                if key.startswith(prefix):
                    layer_state[key[len(prefix) :]] = handle.get_tensor(key)
                    break
    if not layer_state:
        pytest.skip(f"no layer {layer_idx} tensors found in {weight_file}")

    hf_layer.load_state_dict(layer_state, strict=True)
    hf_layer.to(torch.bfloat16)
    hf_layer.eval()
    OptimizedDecoder = _load_optimized_decoder_class()
    decoder = OptimizedDecoder.from_state_dict(
        layer_state,
        hf_config=text_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        mesh_config=MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1)),
    )
    assert decoder.layer_type == "sliding_attention"

    result = _run_optimized_prefill_then_decode(
        "sliding_attention",
        32,
        mesh_device,
        hf_layer=hf_layer,
        decoder=decoder,
        weights_label="optimized_real",
    )
    assert result["prefill_pcc"] >= PCC_THRESHOLD
    assert result["decode_pcc"] >= _decode_threshold("sliding_attention")
