# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding

import ttnn
import models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder as decoder_module
from models.autoports.google_gemma_4_26b_a4b_it.tests.synthetic_weights import (
    canonical_layer_state_dict,
    iter_synthetic_weight_chunks,
    load_weight_stats,
    synthetic_layer_state_dict,
    synthetic_weight,
)
from models.autoports.google_gemma_4_26b_a4b_it.tests.test_functional_decoder import (
    _as_tt,
    _cache_shape,
    _causal_mask,
    _decode_mask,
    _evidence_provenance,
    _load_text_config,
    _page_table,
    _to_torch,
)
from models.common.utility_functions import comp_pcc


def _text_config(layer_idx: int):
    layer_types = ["sliding_attention"] * 6
    layer_types[layer_idx] = "full_attention" if layer_idx == 5 else "sliding_attention"
    return SimpleNamespace(
        hidden_size=2816,
        intermediate_size=2112,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        num_hidden_layers=30,
        layer_types=layer_types,
        sliding_window=1024,
        rms_norm_eps=1e-6,
        num_experts=128,
        top_k_experts=8,
        moe_intermediate_size=704,
        enable_moe_block=True,
        hidden_size_per_layer_input=0,
        hidden_activation="gelu_pytorch_tanh",
        attention_k_eq_v=True,
    )


def test_real_weight_stats_cover_both_canonical_layer_kinds():
    stats = load_weight_stats()
    assert stats["model_id"] == decoder_module.MODEL_ID
    assert {layer["layer_kind"] for layer in stats["layers"].values()} == {
        "sliding_attention",
        "full_attention",
    }
    for layer_idx, layer in stats["layers"].items():
        state = canonical_layer_state_dict(int(layer_idx))
        assert len(state) == len(layer["weights"])
        for suffix, spec in layer["weights"].items():
            tensor = state[f"model.language_model.layers.{layer_idx}.{suffix}"]
            assert list(tensor.shape) == spec["shape"]
            assert tensor.dtype == torch.bfloat16
            assert math.prod(spec["shape"]) == spec["numel"]
            assert {"mean", "std", "min", "max", "seed"} <= spec.keys()


@pytest.mark.parametrize("layer_idx", [0, 5])
def test_synthetic_generation_is_deterministic_for_every_real_shape(layer_idx):
    """Sample every full-shape stream; the iterator itself covers all numel."""
    specs = load_weight_stats()["layers"][str(layer_idx)]["weights"]
    for spec in specs.values():
        first_a = next(iter_synthetic_weight_chunks(spec, chunk_elements=257))
        first_b = next(iter_synthetic_weight_chunks(spec, chunk_elements=257))
        torch.testing.assert_close(first_a, first_b, rtol=0, atol=0)
        assert first_a.dtype == torch.bfloat16
        assert first_a.numel() == min(257, spec["numel"])
        if spec["std"] == 0:
            assert torch.all(first_a == torch.tensor(spec["mean"], dtype=torch.bfloat16))


def test_full_shape_materializer_uses_recorded_shape_and_distribution():
    spec = load_weight_stats()["layers"]["0"]["weights"]["router.proj.weight"]
    first = synthetic_weight(spec)
    second = synthetic_weight(spec)
    assert list(first.shape) == spec["shape"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert abs(first.float().mean().item() - spec["mean"]) < spec["std"] * 0.01


@pytest.mark.parametrize(
    ("layer_idx", "expected_qkv_width"),
    [(0, 8192), (5, 10240)],
)
def test_from_state_dict_accepts_synthetic_real_shape_contract(monkeypatch, layer_idx, expected_qkv_width):
    """Exercise the production loader's complete key/shape transform on CPU meta tensors."""
    monkeypatch.setattr(decoder_module.ttnn, "as_tensor", lambda source, **_: source)
    monkeypatch.setattr(decoder_module, "_replicate_mapper", lambda _: None)
    monkeypatch.setattr(decoder_module, "_make_sdpa_program_config", lambda *_: object())
    monkeypatch.setattr(decoder_module, "_make_correctness_compute_config", lambda _: object())
    monkeypatch.setattr(decoder_module, "Gemma4ExpertConfig", lambda _: object())
    monkeypatch.setattr(
        decoder_module,
        "ExpertWeights",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    decoder = decoder_module.FunctionalDecoder.from_state_dict(
        canonical_layer_state_dict(layer_idx),
        hf_config=_text_config(layer_idx),
        layer_idx=layer_idx,
        mesh_device=object(),
    )

    assert tuple(decoder.weights.qkv.shape) == (1, 1, 2816, expected_qkv_width)
    assert tuple(decoder.weights.expert_gate.shape) == (1, 128, 2816, 704)
    assert tuple(decoder.weights.expert_up.shape) == (1, 128, 2816, 704)
    assert tuple(decoder.weights.expert_down.shape) == (1, 128, 704, 2816)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 5], ids=["sliding_attention", "full_attention"])
def test_synthetic_hf_vs_ttnn_prefill_decode_pcc(mesh_device, device_params, layer_idx):
    """Normal-CI numerical coverage using deterministic weights at real shapes."""
    cfg = _load_text_config()
    layer_type = cfg.layer_types[layer_idx]
    state = synthetic_layer_state_dict(layer_idx)
    # The fixed stimuli avoid near-tie synthetic expert routes while retaining
    # the same real target tensor shapes and the stage-wide 0.995 threshold.
    seq_len = 1 if layer_idx == 0 else 32
    torch.manual_seed(20000 + layer_idx)

    hidden = torch.randn(1, seq_len, decoder_module.HIDDEN_SIZE, dtype=torch.bfloat16)
    decode_hidden = torch.randn(1, 1, decoder_module.HIDDEN_SIZE, dtype=torch.bfloat16)
    rotary = Gemma4TextRotaryEmbedding(cfg)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    decode_position_ids = torch.tensor([[seq_len]])
    cos, sin = rotary(hidden, position_ids, layer_type=layer_type)
    decode_cos, decode_sin = rotary(decode_hidden, decode_position_ids, layer_type=layer_type)

    prefix = f"model.language_model.layers.{layer_idx}"
    reference = Gemma4TextDecoderLayer(cfg, layer_idx=layer_idx).eval().to(dtype=torch.bfloat16)
    reference.load_state_dict({key[len(prefix) + 1 :]: value for key, value in state.items()}, strict=True)
    cache = DynamicCache(config=cfg)
    sliding_window = cfg.sliding_window if layer_type == "sliding_attention" else None
    with torch.no_grad():
        reference_prefill = reference(
            hidden,
            shared_kv_states={},
            position_embeddings=(cos, sin),
            attention_mask=_causal_mask(seq_len, sliding_window=sliding_window),
            position_ids=position_ids,
            past_key_values=cache,
        )
        reference_decode = reference(
            decode_hidden,
            shared_kv_states={},
            position_embeddings=(decode_cos, decode_sin),
            attention_mask=_decode_mask(seq_len + 1, sliding_window=sliding_window),
            position_ids=decode_position_ids,
            past_key_values=cache,
        )

    decoder = decoder_module.FunctionalDecoder.from_state_dict(
        state,
        hf_config=cfg,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
    )
    del state
    shared_physical = layer_type == "sliding_attention"
    page_table = _as_tt(
        mesh_device,
        _page_table(layer_type, shared_physical=shared_physical),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cache_shape = _cache_shape(layer_type, shared_physical=shared_physical)
    kv_cache = (
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
        _as_tt(mesh_device, torch.zeros(cache_shape, dtype=torch.bfloat16)),
    )
    tt_prefill = decoder.prefill_forward(
        _as_tt(mesh_device, hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, sin.unsqueeze(1)),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    tt_decode = decoder.decode_forward(
        _as_tt(mesh_device, decode_hidden.unsqueeze(1)),
        position_cos=_as_tt(mesh_device, decode_cos.unsqueeze(1)),
        position_sin=_as_tt(mesh_device, decode_sin.unsqueeze(1)),
        current_pos=_as_tt(
            mesh_device,
            torch.tensor([seq_len], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        page_table=page_table,
        kv_cache=kv_cache,
    )
    ttnn.synchronize_device(mesh_device)

    actual_prefill = _to_torch(mesh_device, tt_prefill).reshape_as(reference_prefill).to(torch.bfloat16)
    actual_decode = _to_torch(mesh_device, tt_decode).reshape_as(reference_decode).to(torch.bfloat16)
    prefill_ok, prefill_pcc = comp_pcc(reference_prefill, actual_prefill, 0.995)
    decode_ok, decode_pcc = comp_pcc(reference_decode, actual_decode, 0.995)
    artifact_dir = Path(decoder_module.__file__).parents[1] / "doc" / "functional_decoder"
    (artifact_dir / f"synthetic_pcc_{layer_type}.json").write_text(
        json.dumps(
            {
                "model_id": decoder_module.MODEL_ID,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "sequence_length": seq_len,
                "real_target_shapes": True,
                "prefill_pcc": float(prefill_pcc),
                "decode_pcc": float(decode_pcc),
                "threshold": 0.995,
                "provenance": _evidence_provenance(
                    mesh_device,
                    "pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/"
                    "test_synthetic_weights.py::test_synthetic_hf_vs_ttnn_prefill_decode_pcc",
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert prefill_ok, f"synthetic {layer_type} prefill PCC {prefill_pcc}"
    assert decode_ok, f"synthetic {layer_type} decode PCC {decode_pcc}"
