# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs


MODEL_ID = "google/gemma-4-12B"
ROOT = Path(__file__).resolve().parents[1]
FUNCTIONAL_DECODER_PATH = ROOT / "tt" / "functional_decoder.py"
PCC_THRESHOLD = float(os.getenv("GEMMA4_12B_FUNCTIONAL_PCC", "0.995"))
SLIDING_DECODE_PCC_THRESHOLD = float(os.getenv("GEMMA4_12B_SLIDING_DECODE_PCC", "0.993"))
LONG_CONTEXT_PCC_THRESHOLD = float(os.getenv("GEMMA4_12B_LONG_CONTEXT_PCC", "0.992"))
PREFILL_SEQ = int(os.getenv("GEMMA4_12B_PREFILL_SEQ", "128"))
LONG_SEQ = int(os.getenv("GEMMA4_12B_LONG_SEQ", "1024"))
USE_4D_DECODE_ROPE = os.getenv("GEMMA4_12B_DECODE_ROPE", "2d").lower() == "4d"
USE_IDENTITY_PAGE_TABLE = os.getenv("GEMMA4_12B_PAGE_TABLE", "permuted").lower() == "identity"
USE_RANK1_CACHE_POS = os.getenv("GEMMA4_12B_CACHE_POS", "rank2").lower() == "rank1"
USE_HF_CACHE_FOR_DECODE = os.getenv("GEMMA4_12B_CACHE_SOURCE", "tt_prefill").lower() == "hf"
DISABLE_TT_SLIDING_WINDOW = os.getenv("GEMMA4_12B_DISABLE_TT_SLIDING_WINDOW", "0") == "1"
BLOCK_SIZE = 64
EVIDENCE_DIR = ROOT / "doc" / "functional_decoder"
PCC_RESULTS = EVIDENCE_DIR / "pcc_results.jsonl"


pytestmark = pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((1, 1), {"trace_region_size": 0}, id="1x1")],
    indirect=True,
)


def _load_functional_decoder_class():
    spec = importlib.util.spec_from_file_location("gemma4_12b_functional_decoder", FUNCTIONAL_DECODER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.FunctionalDecoder


def _hf_text_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    text_config = config.text_config
    text_config._attn_implementation = "eager"
    return text_config


def _find_layer_idx(text_config, layer_type):
    for idx, candidate in enumerate(text_config.layer_types):
        if candidate == layer_type:
            return idx
    raise AssertionError(f"{MODEL_ID} has no {layer_type} layer")


def _build_hf_prefill_mask(seq_len, sliding_window):
    mask = torch.triu(torch.full((1, 1, seq_len, seq_len), float("-inf")), diagonal=1)
    if sliding_window is not None and seq_len > sliding_window:
        idx = torch.arange(seq_len)
        outside_window = idx.unsqueeze(0) < (idx.unsqueeze(1) - sliding_window + 1)
        mask = mask.masked_fill(outside_window.unsqueeze(0).unsqueeze(0), float("-inf"))
    return mask


def _build_hf_decode_mask(cache_len, sliding_window):
    total_len = cache_len + 1
    mask = torch.zeros(1, 1, 1, total_len)
    if sliding_window is not None:
        current_pos = cache_len
        old = torch.arange(total_len) < (current_pos - sliding_window + 1)
        mask = mask.masked_fill(old.reshape(1, 1, 1, total_len), float("-inf"))
    return mask


def _synthetic_hf_layer(text_config, layer_idx):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    torch.manual_seed(20260608 + layer_idx)
    layer = Gemma4TextDecoderLayer(text_config, layer_idx=layer_idx)
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if name.endswith("norm.weight"):
                param.fill_(1.0)
            elif param.ndim >= 2:
                param.normal_(mean=0.0, std=0.0125)
                param.copy_(param.to(torch.bfloat16).float())
            else:
                param.zero_()
        layer.layer_scalar.fill_(1.0)
    layer.to(torch.bfloat16)
    layer.eval()
    return layer


def _to_tt(tensor, mesh_device, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None
    return ttnn.from_torch(tensor, device=mesh_device, layout=layout, dtype=dtype, mesh_mapper=mapper)


def _from_tt(tensor, mesh_device):
    if hasattr(mesh_device, "shape"):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()
    return ttnn.to_torch(tensor).float()


def _rope_tables(text_config, mesh_device, max_seq_len, layer_idx):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    rope = Gemma4TextRotaryEmbedding(text_config)
    layer_type = text_config.layer_types[layer_idx]
    x_dummy = torch.zeros(1, max_seq_len, text_config.hidden_size)
    position_ids = torch.arange(max_seq_len).unsqueeze(0)
    cos, sin = rope(x_dummy, position_ids, layer_type=layer_type)
    cos4 = _to_tt(cos.unsqueeze(0).to(torch.bfloat16), mesh_device)
    sin4 = _to_tt(sin.unsqueeze(0).to(torch.bfloat16), mesh_device)
    cos2 = _to_tt(cos[0].to(torch.bfloat16), mesh_device)
    sin2 = _to_tt(sin[0].to(torch.bfloat16), mesh_device)
    return (cos4, sin4), (cos2, sin2)


def _page_table(max_num_blocks):
    perm = list(range(max_num_blocks))
    if max_num_blocks >= 4 and not USE_IDENTITY_PAGE_TABLE:
        perm[:4] = [2, 0, 3, 1]
    return torch.tensor([perm], dtype=torch.int32)


def _make_decoder_and_hf(layer_type, mesh_device, hf_layer=None):
    text_config = _hf_text_config()
    layer_idx = _find_layer_idx(text_config, layer_type)
    hf_layer = hf_layer or _synthetic_hf_layer(text_config, layer_idx)
    model_args = Gemma4ModelArgs.from_hf_config(text_config)
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1))
    FunctionalDecoder = _load_functional_decoder_class()
    decoder = FunctionalDecoder.from_state_dict(
        hf_layer.state_dict(),
        hf_config=text_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        mesh_config=mesh_config,
        dtype=ttnn.bfloat16,
        attention_dtype=ttnn.bfloat16,
        shared_mlp_dtype=ttnn.bfloat16,
    )
    return text_config, model_args, layer_idx, hf_layer, decoder


def _assert_pcc(actual, expected, label, threshold=PCC_THRESHOLD):
    passing, pcc = comp_pcc(expected, actual, threshold)
    assert passing, f"{label} PCC {pcc} below threshold {threshold}"
    return pcc


def _record_pcc(record):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    with PCC_RESULTS.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _decode_threshold(layer_type):
    if layer_type == "sliding_attention":
        return SLIDING_DECODE_PCC_THRESHOLD
    return PCC_THRESHOLD


def _signpost(name):
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(name)


def _runtime_inputs(layer_type, seq_len, mesh_device, *, hf_layer=None, decoder=None):
    if decoder is None:
        text_config, model_args, layer_idx, hf_layer, decoder = _make_decoder_and_hf(layer_type, mesh_device, hf_layer)
    else:
        text_config = _hf_text_config()
        layer_idx = _find_layer_idx(text_config, layer_type)
        model_args = Gemma4ModelArgs.from_hf_config(text_config)
        if hf_layer is None:
            raise ValueError("hf_layer must be supplied when decoder is supplied")

    if DISABLE_TT_SLIDING_WINDOW and layer_type == "sliding_attention":
        decoder.attention_config.sliding_window = None

    max_num_blocks = math.ceil((seq_len + 1) / BLOCK_SIZE) + 4
    max_seq_len = max_num_blocks * BLOCK_SIZE
    page_table_cpu = _page_table(max_num_blocks)
    page_table_tt = _to_tt(page_table_cpu, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)
    kv_cache = decoder.create_paged_kv_cache(block_size=BLOCK_SIZE, max_num_blocks=max_num_blocks)
    rope4, rope2 = _rope_tables(text_config, mesh_device, max_seq_len, layer_idx)

    torch.manual_seed(9000 + layer_idx + seq_len)
    x_prefill = torch.randn(1, seq_len, model_args.hidden_size, dtype=torch.float32)
    x_decode = torch.randn(1, 1, model_args.hidden_size, dtype=torch.float32)
    x_prefill_tt = _to_tt(x_prefill.unsqueeze(0).to(torch.bfloat16), mesh_device)
    x_decode_tt = _to_tt(x_decode.unsqueeze(0).to(torch.bfloat16), mesh_device)
    pos_embed_tt = _to_tt(
        torch.tensor([[seq_len]], dtype=torch.uint32), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )
    pos_cache_cpu = (
        torch.tensor([seq_len], dtype=torch.int32)
        if USE_RANK1_CACHE_POS
        else torch.tensor([[seq_len]], dtype=torch.int32)
    )
    pos_cache_tt = _to_tt(pos_cache_cpu, mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32)

    return {
        "text_config": text_config,
        "model_args": model_args,
        "layer_idx": layer_idx,
        "hf_layer": hf_layer,
        "decoder": decoder,
        "kv_cache": kv_cache,
        "page_table_tt": page_table_tt,
        "rope4": rope4,
        "rope2": rope2,
        "x_prefill": x_prefill,
        "x_decode": x_decode,
        "x_prefill_tt": x_prefill_tt,
        "x_decode_tt": x_decode_tt,
        "pos_embed_tt": pos_embed_tt,
        "pos_cache_tt": pos_cache_tt,
    }


def _run_prefill_then_decode(
    layer_type,
    seq_len,
    mesh_device,
    hf_layer=None,
    decoder=None,
    *,
    prefill_threshold=PCC_THRESHOLD,
    decode_threshold=None,
    weights_label="synthetic",
):
    runtime = _runtime_inputs(layer_type, seq_len, mesh_device, hf_layer=hf_layer, decoder=decoder)
    text_config = runtime["text_config"]
    model_args = runtime["model_args"]
    layer_idx = runtime["layer_idx"]
    hf_layer = runtime["hf_layer"]
    decoder = runtime["decoder"]
    kv_cache = runtime["kv_cache"]
    page_table_tt = runtime["page_table_tt"]
    rope4 = runtime["rope4"]
    rope2 = runtime["rope2"]
    x_prefill = runtime["x_prefill"]
    x_decode = runtime["x_decode"]
    x_prefill_tt = runtime["x_prefill_tt"]
    x_decode_tt = runtime["x_decode_tt"]
    pos_embed_tt = runtime["pos_embed_tt"]
    pos_cache_tt = runtime["pos_cache_tt"]
    decode_threshold = _decode_threshold(layer_type) if decode_threshold is None else decode_threshold
    sliding_window = model_args.sliding_window if layer_type == "sliding_attention" else None
    x_prefill_ref = x_prefill.to(torch.bfloat16)
    x_decode_ref = x_decode.to(torch.bfloat16)

    from transformers.cache_utils import DynamicCache
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    hf_cache = DynamicCache(config=text_config)
    rope = Gemma4TextRotaryEmbedding(text_config)
    prefill_cos_sin = rope(
        torch.zeros_like(x_prefill_ref), torch.arange(seq_len).unsqueeze(0), layer_type=layer_type
    )
    prefill_cos_sin = tuple(t.to(torch.bfloat16) for t in prefill_cos_sin)
    with torch.no_grad():
        hf_prefill = hf_layer(
            x_prefill_ref,
            position_embeddings=prefill_cos_sin,
            attention_mask=_build_hf_prefill_mask(seq_len, sliding_window),
            past_key_values=hf_cache,
            shared_kv_states={},
        )
    hf_cache_layer = hf_cache.layers[layer_idx]
    hf_cache_layer.keys = hf_cache_layer.keys.to(torch.bfloat16)
    hf_cache_layer.values = hf_cache_layer.values.to(torch.bfloat16)

    tt_prefill = decoder.prefill_forward(
        x_prefill_tt,
        rope_mats=rope4,
        page_table=page_table_tt,
        kv_cache=kv_cache,
    )
    tt_prefill_torch = _from_tt(tt_prefill, mesh_device).squeeze(0)
    prefill_pcc = _assert_pcc(
        tt_prefill_torch,
        hf_prefill.float(),
        f"{layer_type} prefill seq={seq_len}",
        threshold=prefill_threshold,
    )

    if USE_HF_CACHE_FOR_DECODE:
        hf_k_tt = _to_tt(hf_cache_layer.keys, mesh_device)
        hf_v_tt = _to_tt(hf_cache_layer.values, mesh_device)
        ttnn.experimental.paged_fill_cache(kv_cache[0], hf_k_tt, page_table_tt, batch_idx=0, block_size=BLOCK_SIZE)
        ttnn.experimental.paged_fill_cache(kv_cache[1], hf_v_tt, page_table_tt, batch_idx=0, block_size=BLOCK_SIZE)

    decode_cos_sin = rope(torch.zeros_like(x_decode_ref), torch.tensor([[seq_len]]), layer_type=layer_type)
    decode_cos_sin = tuple(t.to(torch.bfloat16) for t in decode_cos_sin)
    with torch.no_grad():
        hf_decode = hf_layer(
            x_decode_ref,
            position_embeddings=decode_cos_sin,
            attention_mask=_build_hf_decode_mask(seq_len, sliding_window),
            past_key_values=hf_cache,
            shared_kv_states={},
        )

    if USE_4D_DECODE_ROPE:
        tt_decode = decoder.decode_forward(
            x_decode_tt,
            rope_mats=rope4,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            position_idx=pos_cache_tt,
            token_index=seq_len,
        )
    else:
        tt_decode = decoder.decode_forward(
            x_decode_tt,
            rope_mats=rope2,
            page_table=page_table_tt,
            kv_cache=kv_cache,
            position_idx=pos_embed_tt,
            position_idx_cache=pos_cache_tt,
        )
    tt_decode_torch = _from_tt(tt_decode, mesh_device).squeeze(0)
    decode_pcc = _assert_pcc(
        tt_decode_torch,
        hf_decode.float(),
        f"{layer_type} decode pos={seq_len}",
        threshold=decode_threshold,
    )
    _record_pcc(
        {
            "layer_type": layer_type,
            "layer_idx": layer_idx,
            "seq_len": seq_len,
            "prefill_pcc": prefill_pcc,
            "prefill_threshold": prefill_threshold,
            "decode_pcc": decode_pcc,
            "decode_threshold": decode_threshold,
            "page_table": "identity" if USE_IDENTITY_PAGE_TABLE else "permuted",
            "cache_pos_rank": 1 if USE_RANK1_CACHE_POS else 2,
            "decode_rope": "4d" if USE_4D_DECODE_ROPE else "2d",
            "weights": weights_label,
        }
    )

    return {
        "text_config": text_config,
        "model_args": model_args,
        "layer_idx": layer_idx,
        "decoder": decoder,
        "kv_cache": kv_cache,
        "page_table_tt": page_table_tt,
        "rope2": rope2,
        "rope4": rope4,
        "x_decode": x_decode,
        "x_decode_tt": x_decode_tt,
        "pos_embed_tt": pos_embed_tt,
        "pos_cache_tt": pos_cache_tt,
        "hf_decode": hf_decode,
        "seq_len": seq_len,
        "prefill_pcc": prefill_pcc,
        "decode_pcc": decode_pcc,
    }


def test_runtime_fallback_audit_source_clean(mesh_device):
    source = FUNCTIONAL_DECODER_PATH.read_text()
    forbidden = ("import torch", "ttnn.from_torch", "ttnn.to_torch")
    found = [item for item in forbidden if item in source]
    assert not found, f"runtime fallback tokens found in functional decoder source: {found}"


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_paged_prefill_then_decode_pcc(layer_type, mesh_device):
    result = _run_prefill_then_decode(layer_type, PREFILL_SEQ, mesh_device)
    assert result["prefill_pcc"] >= PCC_THRESHOLD
    assert result["decode_pcc"] >= _decode_threshold(layer_type)


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_decode_trace_replay_pcc_and_determinism(layer_type, mesh_device):
    result = _run_prefill_then_decode(layer_type, PREFILL_SEQ, mesh_device)
    decoder = result["decoder"]
    seq_len = result["seq_len"]

    x_decode_tt = _to_tt(result["x_decode"].unsqueeze(0).to(torch.bfloat16), mesh_device)
    pos_embed_tt = _to_tt(
        torch.tensor([[seq_len]], dtype=torch.uint32), mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
    )
    pos_cache_tt = _to_tt(
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
    replay_0 = _from_tt(traced_output, mesh_device).squeeze(0)
    replay_pcc = _assert_pcc(
        replay_0,
        result["hf_decode"].float(),
        f"{layer_type} traced decode replay",
        threshold=_decode_threshold(layer_type),
    )

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    replay_1 = _from_tt(traced_output, mesh_device).squeeze(0)
    ttnn.release_trace(mesh_device, trace_id)
    determinism_pcc = _assert_pcc(replay_1, replay_0, f"{layer_type} repeated traced decode", threshold=0.9999)
    _record_pcc(
        {
            "layer_type": layer_type,
            "layer_idx": result["layer_idx"],
            "seq_len": seq_len,
            "trace_replay_pcc": replay_pcc,
            "trace_replay_threshold": _decode_threshold(layer_type),
            "determinism_pcc": determinism_pcc,
            "determinism_threshold": 0.9999,
        }
    )


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_long_context_paged_prefill_decode(layer_type, mesh_device):
    if LONG_SEQ < 1024:
        pytest.skip(f"GEMMA4_12B_LONG_SEQ={LONG_SEQ} does not cover the sliding-window boundary")
    result = _run_prefill_then_decode(
        layer_type,
        LONG_SEQ,
        mesh_device,
        prefill_threshold=LONG_CONTEXT_PCC_THRESHOLD,
        decode_threshold=LONG_CONTEXT_PCC_THRESHOLD,
    )
    assert result["prefill_pcc"] >= LONG_CONTEXT_PCC_THRESHOLD
    assert result["decode_pcc"] >= LONG_CONTEXT_PCC_THRESHOLD


@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"], ids=["sliding", "full"])
def test_perf_warmed_prefill_and_traced_decode(layer_type, mesh_device):
    runtime = _runtime_inputs(layer_type, PREFILL_SEQ, mesh_device)
    decoder = runtime["decoder"]

    decoder.prefill_forward(
        runtime["x_prefill_tt"],
        rope_mats=runtime["rope4"],
        page_table=runtime["page_table_tt"],
        kv_cache=runtime["kv_cache"],
    )
    ttnn.synchronize_device(mesh_device)

    _signpost("PERF_PREFILL")
    decoder.prefill_forward(
        runtime["x_prefill_tt"],
        rope_mats=runtime["rope4"],
        page_table=runtime["page_table_tt"],
        kv_cache=runtime["kv_cache"],
    )
    ttnn.synchronize_device(mesh_device)
    _signpost("PERF_PREFILL_END")

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
    _signpost("PERF_DECODE")
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    _signpost("PERF_DECODE_END")
    ttnn.release_trace(mesh_device, trace_id)


def _find_local_real_weight_file():
    candidates = []
    env_path = os.getenv("GEMMA4_12B_WEIGHTS") or os.getenv("HF_MODEL")
    if env_path:
        p = Path(env_path)
        candidates.append(p / "model.safetensors" if p.is_dir() else p)
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--google--gemma-4-12B"
    candidates.extend(cache_root.glob("snapshots/*/model.safetensors"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def test_real_weight_layer0_prefill_decode(mesh_device):
    weight_file = _find_local_real_weight_file()
    if weight_file is None:
        pytest.skip("real google/gemma-4-12B model.safetensors is not present locally")

    from safetensors import safe_open
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    text_config = _hf_text_config()
    layer_idx = _find_layer_idx(text_config, "sliding_attention")
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
    FunctionalDecoder = _load_functional_decoder_class()
    decoder = FunctionalDecoder.from_state_dict(
        layer_state,
        hf_config=text_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        mesh_config=MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1)),
        dtype=ttnn.bfloat16,
        attention_dtype=ttnn.bfloat16,
        shared_mlp_dtype=ttnn.bfloat16,
    )
    assert decoder.layer_type == "sliding_attention"

    result = _run_prefill_then_decode(
        "sliding_attention",
        32,
        mesh_device,
        hf_layer=hf_layer,
        decoder=decoder,
        weights_label="real",
    )
    assert result["prefill_pcc"] >= PCC_THRESHOLD
    assert result["decode_pcc"] >= _decode_threshold("sliding_attention")
