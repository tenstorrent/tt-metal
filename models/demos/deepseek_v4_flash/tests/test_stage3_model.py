# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v4_flash.converter import convert_hf_checkpoint
from models.demos.deepseek_v4_flash.demo import require_t3k_available
from models.demos.deepseek_v4_flash.synthetic import generate_tiny_hf_checkpoint
from models.demos.deepseek_v4_flash.ttnn_decode_cache import Batch1DecodeCache, Batch1DecodeLayerCache
from models.demos.deepseek_v4_flash.ttnn_model import (
    ModelEmbeddingHeadWeights,
    TtDeepSeekV4FlashTinyModel,
    embed_input_ids_host,
    load_model_embedding_head_weights,
    normalize_tiny_model_layer_ids,
    validate_model_decode_step_input_ids,
    validate_model_embedding_head_weights,
    validate_model_input_ids,
    validate_tiny_model_decode_cache,
)


def test_tiny_model_embedding_head_loading_and_host_embedding_lookup(tmp_path) -> None:
    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=3)
    output = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")

    weights = load_model_embedding_head_weights(output)

    assert weights.embed_weight.shape == (64, 32)
    assert weights.head_weight.shape == (64, 32)
    validate_model_embedding_head_weights(weights, hidden_size=32, vocab_size=64)

    input_ids = torch.tensor([[0, 1, 63, 2]], dtype=torch.int64)
    hidden_states = embed_input_ids_host(input_ids, weights.embed_weight)

    assert hidden_states.shape == (1, 4, 32)
    assert hidden_states.dtype == weights.embed_weight.dtype
    torch.testing.assert_close(hidden_states[0, 2], weights.embed_weight[63])


def test_tiny_model_api_validation_errors() -> None:
    weights = ModelEmbeddingHeadWeights(embed_weight=torch.zeros(64, 32), head_weight=torch.zeros(64, 32))
    validate_model_embedding_head_weights(weights, hidden_size=32, vocab_size=64)
    validate_model_input_ids(torch.zeros(1, 4, dtype=torch.int64), vocab_size=64)

    with pytest.raises(ValueError, match="Expected embed_weight shape"):
        validate_model_embedding_head_weights(
            ModelEmbeddingHeadWeights(embed_weight=torch.zeros(63, 32), head_weight=weights.head_weight),
            hidden_size=32,
            vocab_size=64,
        )
    with pytest.raises(ValueError, match="Expected head_weight shape"):
        validate_model_embedding_head_weights(
            ModelEmbeddingHeadWeights(embed_weight=weights.embed_weight, head_weight=torch.zeros(64, 16)),
            hidden_size=32,
            vocab_size=64,
        )
    with pytest.raises(ValueError, match="input_ids must have shape"):
        validate_model_input_ids(torch.zeros(2, 4, dtype=torch.int64), vocab_size=64)
    with pytest.raises(ValueError, match="at least one token"):
        validate_model_input_ids(torch.zeros(1, 0, dtype=torch.int64), vocab_size=64)
    with pytest.raises(ValueError, match="dtype must be int32 or int64"):
        validate_model_input_ids(torch.zeros(1, 4, dtype=torch.float32), vocab_size=64)
    with pytest.raises(ValueError, match=r"input_ids values must be in \[0, 64\)"):
        validate_model_input_ids(torch.tensor([[0, 64]], dtype=torch.int64), vocab_size=64)
    with pytest.raises(ValueError, match=r"input_ids values must be in \[0, 64\)"):
        embed_input_ids_host(torch.tensor([[-1]], dtype=torch.int64), weights.embed_weight)
    validate_model_decode_step_input_ids(torch.zeros(1, 1, dtype=torch.int64), vocab_size=64)
    with pytest.raises(ValueError, match="decode input_ids must have shape"):
        validate_model_decode_step_input_ids(torch.zeros(1, 2, dtype=torch.int64), vocab_size=64)


def test_tiny_model_layer_id_normalization() -> None:
    assert normalize_tiny_model_layer_ids(layer=2, num_hidden_layers=3) == (2,)
    assert normalize_tiny_model_layer_ids(layer_ids=(0, 1), num_hidden_layers=3) == (0, 1)
    assert normalize_tiny_model_layer_ids(layer_ids=[0, 1, 2], num_hidden_layers=3) == (0, 1, 2)

    with pytest.raises(ValueError, match="layer_ids must be non-empty"):
        normalize_tiny_model_layer_ids(layer_ids=())
    with pytest.raises(ValueError, match="duplicates"):
        normalize_tiny_model_layer_ids(layer_ids=(0, 0))
    with pytest.raises(ValueError, match=r"layer id must be in \[0, 2\)"):
        normalize_tiny_model_layer_ids(layer_ids=(0, 2), num_hidden_layers=2)
    with pytest.raises(ValueError, match="Pass either layer or layer_ids"):
        normalize_tiny_model_layer_ids(layer=1, layer_ids=(0, 1), num_hidden_layers=3)
    with pytest.raises(TypeError, match="layer ids must be integers"):
        normalize_tiny_model_layer_ids(layer_ids=(True,))


def test_tiny_model_decode_cache_layer_ids_are_explicit() -> None:
    layer_cache = Batch1DecodeLayerCache(
        layer_id=2,
        current_position=8,
        batch_size=1,
        hidden_size=32,
        compress_ratio=4,
        head_dim=8,
        index_n_heads=4,
        index_head_dim=8,
        index_topk=8,
        attention_input_history=torch.zeros(1, 8, 32),
        compressed_kv=torch.zeros(1, 2, 8),
        index_compressed_kv=torch.zeros(1, 2, 8),
    )
    cache = Batch1DecodeCache(layer_caches=(layer_cache,))

    validate_tiny_model_decode_cache(cache, layer_ids=(2,))
    with pytest.raises(ValueError, match="decode cache layer_ids"):
        validate_tiny_model_decode_cache(cache, layer_ids=(0,))


@pytest.mark.t3k_compat
def test_t3k_tiny_model_decode_step_matches_full_forward(tmp_path) -> None:
    ttnn = pytest.importorskip("ttnn")
    try:
        require_t3k_available(ttnn)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    source = generate_tiny_hf_checkpoint(tmp_path / "source", num_hidden_layers=3)
    preprocessed = convert_hf_checkpoint(source, tmp_path / "tt_preprocessed")
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4))
    model = None
    try:
        model = TtDeepSeekV4FlashTinyModel.from_preprocessed(preprocessed, mesh_device=mesh_device)
        prefill_ids = torch.zeros(1, 16, dtype=torch.int64)
        decode_ids = torch.zeros(1, 2, dtype=torch.int64)

        prefill_logits, cache = model.prefill_with_decode_cache(prefill_ids)
        assert prefill_logits.shape == (1, 16, 64)
        step_logits = []
        for token_index in range(decode_ids.shape[1]):
            logits, cache = model.decode_step(decode_ids[:, token_index : token_index + 1], cache=cache)
            step_logits.append(logits)
        decode_logits = torch.cat(step_logits, dim=1)
        full_logits = model(torch.cat([prefill_ids, decode_ids], dim=1))[:, -decode_ids.shape[1] :]

        assert decode_logits.shape == (1, 2, 64)
        assert cache.current_position == 18
        assert cache.layer_caches[0].compressed_kv.shape == (1, 4, 8)
        assert int(full_logits[0, -1].float().argmax().item()) == int(decode_logits[0, -1].float().argmax().item())
        passing, pcc_message = comp_pcc(full_logits.float(), decode_logits.float(), pcc=0.99)
        assert passing, f"Decode-step logits diverged from full-token path: {pcc_message}"
    finally:
        model = None
        for submesh in list(mesh_device.get_submeshes()):
            ttnn.synchronize_device(submesh)
            ttnn.close_mesh_device(submesh)
        ttnn.close_mesh_device(mesh_device)
        if hasattr(ttnn, "FabricConfig") and hasattr(ttnn, "set_fabric_config"):
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
