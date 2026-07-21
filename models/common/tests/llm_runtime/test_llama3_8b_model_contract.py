# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest

from models.common.models.llama3_8b import model as llama_model
from models.common.models.llama3_8b.model import Llama3Transformer1D, Llama3Transformer1DConfig, TransformerBlock1D


def _layer():
    return SimpleNamespace(
        attention_norm=object(),
        attention=SimpleNamespace(config=SimpleNamespace(kv_cache=None), kv_cache=None),
        ff_norm=object(),
        feed_forward=object(),
    )


def test_transformer_config_exposes_hidden_width_for_runtime_slices():
    config = Llama3Transformer1DConfig(
        n_layers=1,
        vocab_size=128256,
        max_batch_size=1,
        max_seq_len=1024,
        dim=4096,
        num_devices=1,
        mesh_device=object(),
        embedding_config=object(),
        rope_config=object(),
        block_configs=[object()],
        norm_config=object(),
        lm_head_config=object(),
    )

    assert config.dim == 4096


def test_iter_executor_named_modules_preserves_names_and_order():
    layers = [_layer(), _layer()]
    model = SimpleNamespace(layers=layers, norm=object(), lm_head=object())

    named_modules = list(Llama3Transformer1D.iter_executor_named_modules(model))

    assert named_modules == [
        ("layer[0].attn_norm", layers[0].attention_norm),
        ("layer[0].attention", layers[0].attention),
        ("layer[0].ff_norm", layers[0].ff_norm),
        ("layer[0].mlp", layers[0].feed_forward),
        ("layer[1].attn_norm", layers[1].attention_norm),
        ("layer[1].attention", layers[1].attention),
        ("layer[1].ff_norm", layers[1].ff_norm),
        ("layer[1].mlp", layers[1].feed_forward),
        ("final_norm", model.norm),
        ("lm_head", model.lm_head),
    ]


def test_iter_executor_named_modules_without_layers_yields_nothing():
    model = SimpleNamespace(norm=object(), lm_head=object())

    assert list(Llama3Transformer1D.iter_executor_named_modules(model)) == []


def test_set_kv_cache_binds_and_unbinds_config_and_runtime_references():
    layers = [_layer(), _layer()]
    model = SimpleNamespace(layers=layers)
    kv_cache = [[object(), object()], [object(), object()]]

    Llama3Transformer1D.set_kv_cache(model, kv_cache)

    for layer, expected in zip(layers, kv_cache):
        bound = layer.attention.config.kv_cache
        assert bound == tuple(expected)
        assert bound[0] is expected[0]
        assert bound[1] is expected[1]
        assert layer.attention.kv_cache is bound

    Llama3Transformer1D.set_kv_cache(model, None)
    Llama3Transformer1D.set_kv_cache(model, None)

    for layer in layers:
        assert layer.attention.config.kv_cache is None
        assert layer.attention.kv_cache is None


def test_set_kv_cache_rejects_wrong_layer_count_before_binding(expect_error):
    layers = [_layer(), _layer()]
    model = SimpleNamespace(layers=layers)

    with expect_error(ValueError, "model has 2 layers"):
        Llama3Transformer1D.set_kv_cache(model, [[object(), object()]])

    assert all(layer.attention.config.kv_cache is None for layer in layers)
    assert all(layer.attention.kv_cache is None for layer in layers)


@pytest.mark.parametrize("bad_pair", [[object()], object()])
def test_set_kv_cache_validates_every_pair_before_binding(bad_pair, expect_error):
    layers = [_layer(), _layer()]
    model = SimpleNamespace(layers=layers)

    with expect_error((TypeError, ValueError), "layer 1.*K/V tensor"):
        Llama3Transformer1D.set_kv_cache(
            model,
            [[object(), object()], bad_pair],
        )

    assert all(layer.attention.config.kv_cache is None for layer in layers)
    assert all(layer.attention.kv_cache is None for layer in layers)


class _FakeTensor:
    def __init__(self, shape=(1, 1, 96, 4096), dtype=None):
        self.shape = shape
        self.dtype = dtype
        self.deallocate = MagicMock()


def _identity_all_gather(_norm, tensor, **_kwargs):
    return tensor


def _model_ttnn(
    *, slice_result=None, split_result=None, typecast_result=None, embedding_result=None, unsqueeze_result=None
):
    slice_mock = MagicMock(return_value=slice_result)
    return SimpleNamespace(
        DRAM_MEMORY_CONFIG=object(),
        TILE_LAYOUT=object(),
        bfloat16=object(),
        add=MagicMock(side_effect=lambda lhs, _rhs, **_kwargs: lhs),
        concat=MagicMock(return_value=slice_result),
        deallocate=MagicMock(),
        embedding=MagicMock(return_value=embedding_result),
        interleaved_to_sharded=MagicMock(side_effect=lambda tensor, _memcfg: tensor),
        reshape=MagicMock(side_effect=lambda tensor, *_args, **_kwargs: tensor),
        slice=slice_mock,
        split=MagicMock(return_value=split_result),
        to_memory_config=MagicMock(side_effect=lambda tensor, *_args, **_kwargs: tensor),
        typecast=MagicMock(return_value=typecast_result),
        unsqueeze_to_4D=MagicMock(return_value=unsqueeze_result),
    )


@pytest.mark.parametrize("chunk_start_idx_tensor", [None, object()])
def test_transformer_block_prefill_forwards_scalar_and_tensor_chunk_start(monkeypatch, chunk_start_idx_tensor):
    x = _FakeTensor()
    attn_output = _FakeTensor()
    attention = SimpleNamespace(prefill_forward=MagicMock(return_value=attn_output))
    block = SimpleNamespace(
        activation_dtype=None,
        attention=attention,
        attention_norm=SimpleNamespace(prefill_forward=MagicMock(return_value=x)),
        feed_forward=SimpleNamespace(prefill_forward=MagicMock(side_effect=lambda tensor: tensor)),
        ff_norm=SimpleNamespace(prefill_forward=MagicMock(side_effect=lambda tensor: tensor)),
        prefill_residual_memcfg=object(),
    )
    fake_ttnn = _model_ttnn()
    monkeypatch.setattr(llama_model, "ttnn", fake_ttnn)
    monkeypatch.setattr(llama_model, "_all_gather_rmsnorm_tensor", _identity_all_gather)

    TransformerBlock1D.prefill_forward(
        block,
        x,
        ("cos", "sin"),
        user_id=3,
        page_table="page-table",
        chunk_page_table="chunk-page-table",
        chunk_start_idx=96,
        chunk_start_idx_tensor=chunk_start_idx_tensor,
    )

    attention.prefill_forward.assert_called_once_with(
        x,
        ("cos", "sin"),
        user_id=3,
        page_table="page-table",
        chunk_page_table="chunk-page-table",
        chunk_start_idx=96,
        chunk_start_idx_tensor=chunk_start_idx_tensor,
    )


def test_prefill_uses_runtime_block_slice_then_embeds_exact_row(monkeypatch):
    hidden_states = _FakeTensor(shape=(1, 1, 96, 4096))
    sliced = _FakeTensor(shape=(1, 1, 32, 4096))
    converted = _FakeTensor(shape=(1, 1, 32, 4096))
    embedded = _FakeTensor(shape=(1, 1, 4096))
    selected = _FakeTensor(shape=(1, 1, 1, 4096))
    start_tensor = object()
    end_tensor = object()
    row_index = object()
    fake_ttnn = _model_ttnn(
        slice_result=sliced,
        typecast_result=converted,
        embedding_result=embedded,
        unsqueeze_result=selected,
    )
    model = SimpleNamespace(
        activation_dtypes=[],
        layers=[],
        lm_head=SimpleNamespace(
            config=SimpleNamespace(input_memcfg=None),
            forward=MagicMock(side_effect=lambda tensor: tensor),
        ),
        norm=SimpleNamespace(prefill_forward=MagicMock(side_effect=lambda tensor: tensor)),
    )
    monkeypatch.setattr(llama_model, "ttnn", fake_ttnn)
    monkeypatch.setattr(llama_model, "_all_gather_rmsnorm_tensor", _identity_all_gather)

    Llama3Transformer1D.prefill_forward(
        model,
        hidden_states,
        ("cos", "sin"),
        get_last_token=95,
        last_token_slice=(start_tensor, end_tensor),
        last_token_index=row_index,
    )

    fake_ttnn.slice.assert_called_once_with(
        hidden_states,
        start_tensor,
        end_tensor,
        slice_dim=2,
        num_devices=3,
    )
    fake_ttnn.typecast.assert_called_once_with(sliced, fake_ttnn.bfloat16)
    fake_ttnn.embedding.assert_called_once_with(row_index, converted, layout=fake_ttnn.TILE_LAYOUT)
    fake_ttnn.unsqueeze_to_4D.assert_called_once_with(embedded)
    assert fake_ttnn.deallocate.call_args_list == [call(hidden_states), call(sliced), call(converted)]
    model.norm.prefill_forward.assert_called_once_with(selected)


def test_post_process_prefill_uses_runtime_bounds_for_aligned_block_slice(monkeypatch):
    hidden_states = _FakeTensor(shape=(1, 1, 96, 4096))
    sliced = _FakeTensor(shape=(1, 1, 32, 4096))
    start_tensor = object()
    end_tensor = object()
    fake_ttnn = _model_ttnn(slice_result=sliced)
    model = SimpleNamespace(
        lm_head=SimpleNamespace(
            config=SimpleNamespace(input_memcfg=None),
            forward=MagicMock(side_effect=lambda tensor: tensor),
        ),
        norm=SimpleNamespace(prefill_forward=MagicMock(side_effect=lambda tensor: tensor)),
    )
    monkeypatch.setattr(llama_model, "ttnn", fake_ttnn)
    monkeypatch.setattr(llama_model, "_all_gather_rmsnorm_tensor", _identity_all_gather)

    Llama3Transformer1D.post_process_prefill_output(
        model,
        hidden_states,
        last_token_idx=95,
        last_token_slice=(start_tensor, end_tensor),
    )

    fake_ttnn.slice.assert_called_once_with(
        hidden_states,
        start_tensor,
        end_tensor,
        slice_dim=2,
        num_devices=3,
    )


def test_batched_post_process_uses_runtime_bounds_for_each_active_slot(monkeypatch):
    hidden_states = _FakeTensor(shape=(1, 1, 256, 4096))
    user_states = [_FakeTensor(shape=(1, 1, 128, 4096)) for _ in range(2)]
    block = _FakeTensor(shape=(1, 1, 32, 4096))
    embedded = _FakeTensor(shape=(1, 1, 4096))
    selected = _FakeTensor(shape=(1, 1, 1, 4096))
    start_tensor = object()
    end_tensor = object()
    row_index = object()
    fake_ttnn = _model_ttnn(
        slice_result=block,
        split_result=user_states,
        embedding_result=embedded,
        unsqueeze_result=selected,
    )
    model = SimpleNamespace(
        lm_head=SimpleNamespace(
            config=SimpleNamespace(input_memcfg=None),
            forward=MagicMock(side_effect=lambda tensor: tensor),
        ),
        norm=SimpleNamespace(prefill_forward=MagicMock(side_effect=lambda tensor: tensor)),
    )
    monkeypatch.setattr(llama_model, "ttnn", fake_ttnn)
    monkeypatch.setattr(llama_model, "_all_gather_rmsnorm_tensor", _identity_all_gather)

    Llama3Transformer1D.post_process_batched_prefill_output(
        model,
        hidden_states,
        last_token_idx_list=[4, 17],
        padded_batch=2,
        prefill_seq_len=128,
        last_token_slice=(start_tensor, end_tensor),
        last_token_index=row_index,
    )

    assert fake_ttnn.slice.call_args_list == [
        call(user_states[0], start_tensor, end_tensor, slice_dim=2, num_devices=4),
        call(user_states[1], start_tensor, end_tensor, slice_dim=2, num_devices=4),
    ]
    assert fake_ttnn.embedding.call_args_list == [
        call(row_index, block, layout=fake_ttnn.TILE_LAYOUT),
        call(row_index, block, layout=fake_ttnn.TILE_LAYOUT),
    ]
    assert fake_ttnn.unsqueeze_to_4D.call_args_list == [call(embedded), call(embedded)]
    assert fake_ttnn.deallocate.call_args_list == [call(block), call(block)]


def test_prepare_prefill_rot_mats_gathers_runtime_positions_without_slicing(monkeypatch):
    position_indices = object()
    cos_matrix = object()
    sin_matrix = object()
    cos_rows = object()
    sin_rows = object()
    cos_4d = object()
    sin_4d = object()
    embedding = MagicMock(side_effect=[cos_rows, sin_rows])
    unsqueeze_to_4d = MagicMock(side_effect=[cos_4d, sin_4d])
    fake_ttnn = SimpleNamespace(
        TILE_LAYOUT=object(),
        embedding=embedding,
        unsqueeze_to_4D=unsqueeze_to_4d,
    )
    rope_setup = SimpleNamespace(
        cos_matrix=cos_matrix,
        sin_matrix=sin_matrix,
        load_device_weights=MagicMock(),
    )
    monkeypatch.setattr(llama_model, "ttnn", fake_ttnn)

    result = Llama3Transformer1D.prepare_prefill_rot_mats(
        SimpleNamespace(rope_setup=rope_setup),
        position_indices,
    )

    rope_setup.load_device_weights.assert_called_once_with()
    assert embedding.call_args_list == [
        call(position_indices, cos_matrix, layout=fake_ttnn.TILE_LAYOUT),
        call(position_indices, sin_matrix, layout=fake_ttnn.TILE_LAYOUT),
    ]
    assert unsqueeze_to_4d.call_args_list == [call(cos_rows), call(sin_rows)]
    assert result == (cos_4d, sin_4d)
