# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch

from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _assert_pcc,
    _config,
    _page_table,
    _positions,
    _real_state,
    _reference_decode_zero_prefix,
    _reference_prefill,
    _to_torch_decode,
    _to_torch_prefill,
    _to_tt_decode,
    _to_tt_prefill,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.fused_decoder import FusedDecoder
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder


def _decoder(state, config, mesh_device, batch=1, max_context=96):
    return OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=max_context,
    )


def test_optimized_runtime_dispatch_contract():
    assert OptimizedDecoder._mlp is not FusedDecoder._mlp
    source = inspect.getsource(OptimizedDecoder)
    assert "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig" in source
    assert 'self.weights["down_decode"]' in source
    assert "FunctionalDecoder" not in source
    assert "to_torch" not in source
    assert "from_torch" not in source


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [31, 33, 65])
def test_optimized_prefill_real_reference_non_aligned(mesh_device, seq_len):
    config = _config()
    state = _real_state()
    decoder = _decoder(state, config, mesh_device)
    hidden = (
        torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(9300 + seq_len)) * 0.2
    ).to(torch.bfloat16)
    page_table = _page_table(1, 96, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    reference, _ = _reference_prefill(config, state, hidden)
    _assert_pcc(f"optimized-real-prefill-{seq_len}", reference, _to_torch_prefill(actual))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_decode_real_reference_and_determinism(mesh_device, batch):
    config = _config()
    state = _real_state()
    decoder = _decoder(state, config, mesh_device, batch=batch, max_context=64)
    hidden = (
        torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(9400 + batch)) * 0.2
    ).to(torch.bfloat16)
    positions_list = [33] if batch == 1 else list(range(1, batch + 1))
    positions = _positions(positions_list, mesh_device)
    page_table = _page_table(batch, 64, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()

    def decode():
        return decoder.decode_forward(
            _to_tt_decode(hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=positions,
            use_long_rope=False,
        )

    actual = decode()
    first = _to_torch_decode(actual)
    second = _to_torch_decode(decode())
    assert torch.equal(first, second)
    reference = _reference_decode_zero_prefix(config, state, hidden, positions_list, use_long=False)
    _assert_pcc(f"optimized-real-decode-b{batch}", reference, first)
