# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import pytest
import torch

from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _assert_pcc,
    _config,
    _page_table,
    _positions,
    _reference_decode_zero_prefix,
    _reference_prefill,
    _synthetic_state,
    _to_torch_decode,
    _to_torch_prefill,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.functional_decoder import FunctionalDecoder
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.fused_decoder import FusedDecoder


def test_fused_path_is_not_functional_fallback():
    assert FusedDecoder._mlp is not FunctionalDecoder._mlp
    assert FusedDecoder.prefill_forward is not FunctionalDecoder.prefill_forward
    assert FusedDecoder.decode_forward is not FunctionalDecoder.decode_forward
    source = inspect.getsource(FusedDecoder._mlp)
    assert "input_tensor_a_activations=[ttnn.UnaryOpType.SILU]" in source
    assert "ttnn.silu(" not in source
    prefill_source = inspect.getsource(FusedDecoder.prefill_forward)
    assert "experimental.nlp_concat_heads" in prefill_source
    assert "if self.batch > 1" in prefill_source
    decode_source = inspect.getsource(FusedDecoder.decode_forward)
    assert "paged_fused_update_cache" in decode_source
    assert "if self.batch == 1" in decode_source


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [31, 33, 65])
def test_fused_non_aligned_prefill_matches_reference(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state(config)
    max_context = ((seq_len + 31) // 32) * 32
    decoder = FusedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=max_context,
    )
    hidden = torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(seq_len)).to(
        torch.bfloat16
    )
    reference, _ = _reference_prefill(config, state, hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, max_context, mesh_device, permute=True),
    )
    _assert_pcc(f"fused-prefill-{seq_len}", reference, _to_torch_prefill(actual))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_fused_decode_matches_reference(mesh_device, batch):
    config = _config()
    state = _synthetic_state(config)
    decoder = FusedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=64,
    )
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(100 + batch)).to(
        torch.bfloat16
    )
    positions = list(range(1, batch + 1))
    reference = _reference_decode_zero_prefix(config, state, hidden, positions, use_long=False)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.decode_forward(
        _to_tt_decode(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(batch, 64, mesh_device, permute=True),
        current_positions=_positions(positions, mesh_device),
        use_long_rope=False,
    )
    _assert_pcc(f"fused-decode-b{batch}", reference, _to_torch_decode(actual))
