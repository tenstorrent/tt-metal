# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Model-specific drafter K ordering in the shared producer's PCC validation."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import ttnn
from models.demos.common.prefill.adapter import get_adapter
from models.demos.common.prefill.runners import prefill_producer


@pytest.fixture
def dflash_table():
    # Five draft layers follow two target rows; only the draft rows are populated.
    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = 7
    config.num_slots = 2
    config.max_sequence_length = 32
    config.chunk_n_tokens = 32
    config.chunk_size_bytes = 4352
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(
        {f"dflash_{kind}_h{head:02d}": config for kind in ("k", "v") for head in range(8)}
    )
    location = ttnn.experimental.disaggregation.KvCacheLocation()
    location.device_group_index = table.add_device_group([ttnn.FabricNodeId(ttnn.MeshId(0), 0)])
    location.noc_addr = 0x1000
    location.size_bytes = 4352
    for config_id in range(table.num_configs()):
        for layer in range(2, 7):
            table.set(layer, 0, 1, location, config_id)
    return table


@pytest.mark.parametrize("model_name", ["gemma4_d_p", "deepseek_v3_d_p"])
def test_producer_draft_pcc_uses_model_rope_layout_and_rejects_corruption(
    dflash_table, monkeypatch, tmp_path, model_name
):
    table = dflash_table
    generator = torch.Generator().manual_seed(17)
    golden_k = torch.randn(5, 8, 32, 128, generator=generator)
    golden_v = torch.randn(5, 8, 32, 128, generator=generator)
    save_file({"k_cache": golden_k}, tmp_path / "k_cache.safetensors")
    save_file({"v_cache": golden_v}, tmp_path / "v_cache.safetensors")

    # HF goldens use half-split K. DeepSeek stores adjacent RoPE pairs; Gemma preserves the halves.
    actual_k = golden_k.clone()
    if model_name == "deepseek_v3_d_p":
        actual_k = torch.stack((golden_k[..., :64], golden_k[..., 64:]), dim=-1).flatten(-2)
    actual = {
        table.config_id_of(f"dflash_{kind}_h{head:02d}"): values[:, head]
        for kind, values in (("k", actual_k), ("v", golden_v))
        for head in range(8)
    }

    def read_slice(kv_table, device_map, config_id, layer, slot, read_len, head_dim, decoder):
        assert kv_table is table
        assert (slot, read_len, head_dim) == (1, 32, 128)
        return actual[config_id][layer - 2, :read_len]

    monkeypatch.setattr(prefill_producer, "ADAPTER", get_adapter(model_name))
    monkeypatch.setattr(prefill_producer, "_read_device_map", lambda *args, **kwargs: {0: 0})
    monkeypatch.setattr(prefill_producer, "_resolve_unique_id", lambda *args: 0)
    monkeypatch.setattr(prefill_producer, "_read_kv_slice", read_slice)
    monkeypatch.setattr(prefill_producer, "_read_slot_kv_and_check_pcc", lambda *args, **kwargs: {"target": 1.0})
    monkeypatch.setenv("PREFILL_DFLASH_GOLDEN_KV_DIR", str(tmp_path))
    monkeypatch.setenv("PREFILL_DFLASH_PCC", "0.99")
    monkeypatch.setenv("PREFILL_PCC_SUMMARY_DIR", str(tmp_path))
    stats = SimpleNamespace(resident={1: SimpleNamespace(real_len=32)})
    assert prefill_producer._verify_resident_slots(table, stats, 0.99, {1: str(tmp_path)})

    actual_k[-1, -1].neg_()
    assert not prefill_producer._verify_resident_slots(table, stats, 0.99, {1: str(tmp_path)})
