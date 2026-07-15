# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect

import ttnn
from models.autoports.google_gemma_4_31b.tt.multichip_decoder import MultichipDecoder
from models.autoports.google_gemma_4_31b.tt.optimized_decoder import OptimizedDecoder


class _FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype
        self.deallocated = False

    def deallocate(self, force):
        assert force is True
        self.deallocated = True


def test_cache_update_input_preserves_bf16_without_a_copy(monkeypatch):
    tensor = _FakeTensor(ttnn.bfloat16)
    monkeypatch.setattr(ttnn, "typecast", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))

    assert OptimizedDecoder._prepare_cache_update_input(tensor) is tensor
    assert not tensor.deallocated


def test_cache_update_input_converts_packed_activation_only_at_update_boundary(monkeypatch):
    tensor = _FakeTensor(ttnn.bfloat8_b)
    converted = _FakeTensor(ttnn.bfloat16)
    calls = []

    def fake_typecast(source, dtype):
        calls.append((source, dtype))
        return converted

    monkeypatch.setattr(ttnn, "typecast", fake_typecast)

    assert OptimizedDecoder._prepare_cache_update_input(tensor) is converted
    assert calls == [(tensor, ttnn.bfloat16)]
    assert tensor.deallocated


def test_selected_multichip_paths_normalize_every_paged_update_operand():
    decode_source = inspect.getsource(MultichipDecoder._decode_attention_tp)
    tail_source = inspect.getsource(OptimizedDecoder._fill_bounded_sliding_cache_exact)

    decode_normalize = decode_source.index("k = self._prepare_cache_update_input(k)")
    decode_branch = decode_source.index("if config.cache_position_modulo is None:")
    assert decode_normalize < decode_branch
    assert "v = self._prepare_cache_update_input(v)" in decode_source

    tail_normalize = tail_source.index("k_tail_users = self._prepare_cache_update_input(")
    tail_loop = tail_source.index("for tail_idx in range(tail_len):")
    assert tail_normalize < tail_loop
    assert "v_tail_users = self._prepare_cache_update_input(" in tail_source


def test_decode_qkv_matmul_produces_the_bf16_format_required_by_head_split():
    for decoder_type, method_name in (
        (OptimizedDecoder, "_decode_attention"),
        (MultichipDecoder, "_decode_attention_tp"),
    ):
        assert decoder_type.qkv_split_input_dtype == ttnn.bfloat16
        source = inspect.getsource(getattr(decoder_type, method_name))
        qkv_matmul = source.index("qkv_sharded = ttnn.linear(")
        forced_dtype = source.index("dtype=self.qkv_split_input_dtype", qkv_matmul)
        head_split = source.index("split_qkv_heads_decode(qkv")
        assert qkv_matmul < forced_dtype < head_split


def test_ordinary_prefill_bulk_fill_keeps_configured_cache_dtype():
    source = inspect.getsource(MultichipDecoder._prefill_attention_tp)
    assert "ttnn.typecast(k, k_cache.dtype)" in source
    assert "ttnn.typecast(v, v_cache.dtype)" in source
