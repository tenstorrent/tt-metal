# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only pins for the DeepSeek-V4-Flash prefill adapter's M1: the hand-built config, the layer-kind
schedule, the per-kind KV geometry + the prefill <-> decode KV contract, and the registry. No device, no weights."""

import json
import os
from pathlib import Path

import pytest

from models.demos.common.prefill.adapter import ADAPTER_PATHS, PrefillRunParams, get_adapter
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import (
    DeepSeekV4FlashConfig,
    deepseek_v4_flash_hf_config,
    flash_compress_ratios,
)
from models.demos.deepseek_v3_d_p.tt.v4 import kv_contract as kc
from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import (
    CSA,
    HCA,
    KV_GROUPS,
    MIGRATED_GROUPS,
    PENDING_GROUPS,
    SLIDING,
    V4FlashKvGeometry,
    kind_counts,
    layer_kinds,
    layers_of_kind,
)


def _checkpoint_config():
    for p in (os.environ.get("DEEPSEEK_V4_FLASH_HF_MODEL"), "/mnt/tt-data/sdawle/models/DeepSeek-V4-Flash-0731"):
        if p and Path(p, "config.json").is_file():
            return json.loads(Path(p, "config.json").read_text())
    return None


def test_flash_schedule_is_two_swa_then_csa_hca_interleaved():
    ratios = flash_compress_ratios(43)
    assert ratios[:6] == [0, 0, 4, 128, 4, 128]
    assert {r: ratios.count(r) for r in (0, 4, 128)} == {0: 2, 4: 21, 128: 20}
    ck = _checkpoint_config()
    if ck is None:
        pytest.skip("no DeepSeek-V4-Flash checkpoint on this host to compare against")
    # the checkpoint lists the 43 decoder layers followed by the DSpark MTP layers
    assert ck["compress_ratios"][:43] == ratios
    assert ck["num_hidden_layers"] == 43 and ck["hc_mult"] == 4 and ck["sliding_window"] == 128


def test_hand_built_config_carries_flash_everything():
    cfg = deepseek_v4_flash_hf_config(max_seq=16384)
    assert cfg.num_hidden_layers == 43 and cfg.max_seq_len == 16384
    assert kind_counts(cfg) == {SLIDING: 2, CSA: 21, HCA: 20}
    assert layer_kinds(cfg)[:4] == [SLIDING, SLIDING, CSA, HCA]
    assert cfg.mlp_layer_types == ["hash_moe"] * 3 + ["moe"] * 40
    assert cfg.compress_rates == {CSA: 4, HCA: 128}
    assert cfg.qk_rope_head_dim == 64 and cfg.head_dim == 512 and cfg.num_attention_heads == 64
    assert cfg.q_lora_rank == 1024 and cfg.o_lora_rank == 1024 and cfg.o_groups == 8
    assert cfg.hc_mult == 4 and cfg.hc_sinkhorn_iters == 20 and cfg.hc_eps == 1e-6
    assert cfg.index_n_heads == 64 and cfg.index_head_dim == 128 and cfg.index_topk == 512
    assert cfg.n_routed_experts == 256 and cfg.num_experts_per_tok == 6 and cfg.scoring_func == "sqrtsoftplus"
    assert cfg.vocab_size == 129280 and cfg.sliding_window == 128 and cfg.swiglu_limit == 10.0
    assert set(cfg.rope_parameters) == {"main", "compress"}
    assert cfg.rope_parameters["compress"]["rope_theta"] == 160000.0
    assert cfg.rope_parameters["compress"]["rope_type"] == "yarn" and cfg.rope_parameters["compress"]["factor"] == 16
    assert cfg.rope_parameters["main"]["rope_theta"] == 10000.0
    assert cfg._attn_implementation == "eager"


def test_config_class_default_schedule_is_not_flash():
    """Why the builder passes compress_ratios: DeepseekV4Config's own default is an HCA-first interleave."""
    default = DeepseekV4Config(num_hidden_layers=43)
    assert default.layer_types[:2] == [HCA, HCA]
    assert default.layer_types != deepseek_v4_flash_hf_config().layer_types


def test_adapter_is_registered_lazy_and_describes_m1(expect_error):
    assert "deepseek_v4_flash" in ADAPTER_PATHS
    adapter = get_adapter("deepseek_v4_flash")
    assert adapter.name == "deepseek_v4_flash" and adapter.model_config is DeepSeekV4FlashConfig
    cfg = adapter.load_hf_config()
    assert cfg.num_hidden_layers == 43 and layer_kinds(cfg)[2] == CSA
    assert cfg.max_seq_len % 5120 == 0  # the engine's chunk period; "128k" is 26 chunks = 133,120
    assert adapter.config_builder is deepseek_v4_flash_hf_config
    assert adapter.layer_split_boundaries(43) is None
    assert adapter.hf_model_default.startswith("/mnt/tt-data/")  # the pod's NFS, not the absent /mnt/models
    params = PrefillRunParams(
        mesh_shape=(4, 2),
        num_layers=43,
        first_layer_idx=0,
        is_first_rank=True,
        is_last_rank=True,
        max_seq_len=10240,
        chunk_size=5120,
        num_users=1,
        capacity_factor=1,
        num_links=1,
        gate_mode_name="DEVICE_FP32",
        kv_only_last_layer=False,
        weight_cache_path=None,
    )
    with expect_error(NotImplementedError, "M2..M6"):
        adapter.build_runtime(mesh_device=None, hf_config=cfg, params=params)


# ---------------------------------------------------------------------------------------------------------------
# The KV contract (kv_contract.py) -- plain data both repos import.
# ---------------------------------------------------------------------------------------------------------------


def test_contract_order_names_and_chunk_bytes():
    kc.validate_contract()
    assert KV_GROUPS == ("swa_window", "hca_unified", "csa_unified", "csa_index_k", "csa_pending", "hca_pending")
    assert MIGRATED_GROUPS == KV_GROUPS[:4] and PENDING_GROUPS == KV_GROUPS[4:]
    # decode's dtypes: CSA unified cache is bf16 ROW_MAJOR, HCA/SWA ride the ring's _CACHE_DTYPE (bfp8 tiles),
    # the indexer key cache is bfp8 tiles 128 wide
    by = {g.name: g for g in kc.CONTRACT}
    assert by["csa_unified"].dtype_tag == "bf16_rm" and by["csa_unified"].chunk_size_bytes == 32 * 512 * 2 == 32768
    assert by["swa_window"].dtype_tag == by["hca_unified"].dtype_tag == "bfp8_tile"
    assert by["hca_unified"].chunk_size_bytes == 16 * 1088 == 17408
    assert by["csa_index_k"].dtype_tag == "bfp8_tile" and by["csa_index_k"].chunk_size_bytes == 4 * 1088 == 4352
    assert by["csa_pending"].width == by["hca_pending"].width == 1024  # [kv | gate]
    assert kc.chunk_size_bytes("bfp4_tile", 512) == 16 * 576


def test_contract_extents_follow_the_unified_row_axis():
    """Rows [0,128) = window ring, row 128 + w = entry w; the index cache is one row per entry."""
    by = {g.name: g for g in kc.CONTRACT}
    for s in (5120, 16384, 65536, 133120):
        assert by["swa_window"].extent(s) == 128
        assert by["hca_unified"].extent(s) == 128 + kc.tiles_up(s // 128)
        assert by["csa_unified"].extent(s) == 128 + s // 4
        assert by["csa_index_k"].extent(s) == s // 4
        for g in kc.CONTRACT:
            assert g.extent(s) % kc.CHUNK_N_TOKENS == 0
    assert by["hca_unified"].extent(133120) == 128 + 1056  # 1040 entries -> 33 tiles
    assert by["csa_unified"].extent(133120) == 128 + 33280


def test_contract_window_dtype_follows_the_ring_switch(monkeypatch):
    monkeypatch.setenv("DSV4_FLASH_CACHE_BF4", "1")
    assert kc.window_cache_dtype_tag() == "bfp4_tile"
    assert kc.build_contract()[1].chunk_size_bytes == 16 * 576
    monkeypatch.delenv("DSV4_FLASH_CACHE_BF4")
    assert kc.window_cache_dtype_tag() == "bfp8_tile"


# ---------------------------------------------------------------------------------------------------------------
# Geometry: replicated rows, per kind, in contract order.
# ---------------------------------------------------------------------------------------------------------------


def test_kv_geometry_rows_and_group_order(expect_error):
    cfg = deepseek_v4_flash_hf_config()
    g = V4FlashKvGeometry.from_config(cfg, max_seq_len=65536, sp_factor=4)
    assert (g.hca_entries, g.csa_entries) == (512, 16384)
    shapes = g.group_shapes(num_users=1)
    assert tuple(shapes) == KV_GROUPS  # the prefill <-> decode contract order
    # rows are the same on every chip (replicated), NOT divided by the SP factor
    assert shapes["swa_window"] == (2, 1, 128, 512)
    # allocated rows = migrated extent + the writers' whole-tile headroom (96 for HCA's tail-tile write, 32 for CSA)
    assert shapes["hca_unified"] == (20, 1, 128 + 512 + 96, 512) and g.extent("hca_unified") == 128 + 512
    assert shapes["csa_unified"] == (21, 1, 128 + 16384 + 1280, 512) and g.extent("csa_unified") == 128 + 16384
    assert shapes["csa_index_k"] == (21, 1, 16384 + 1280, 128) and g.extent("csa_index_k") == 16384
    assert shapes["csa_pending"] == (21, 1, 32, 1024)
    assert shapes["hca_pending"] == (20, 1, 128, 1024)
    assert tuple(g.group_shapes(include_pending=False)) == MIGRATED_GROUPS
    # two users double the batch axis only
    assert g.group_shapes(num_users=2)["csa_index_k"] == (42, 1, 16384 + 1280, 128)
    # a ragged entry count rounds up to whole tiles: 5120 tokens -> 40 HCA entries -> 64 rows above the window
    g2 = V4FlashKvGeometry.from_config(cfg, max_seq_len=5120, sp_factor=1)
    assert (g2.extent("hca_unified"), g2.extent("csa_unified"), g2.extent("csa_index_k")) == (
        128 + 64,
        128 + 1280,
        1280,
    )
    assert g2.rows("hca_unified") == 128 + 64 + 96
    # per-chip bytes at 128k (26 chunks): CSA unified ~0.7 GB, index keys ~94 MB, HCA ~12 MB
    b = V4FlashKvGeometry.from_config(cfg, max_seq_len=133120, sp_factor=8).group_bytes()
    assert 0.70e9 < b["csa_unified"] < 0.76e9 and 0.09e9 < b["csa_index_k"] < 0.10e9 and b["hca_unified"] < 0.02e9
    with expect_error(ValueError, "multiple of the HCA rate"):
        V4FlashKvGeometry.from_config(cfg, max_seq_len=1000, sp_factor=3)


def test_rank_slices_keep_only_their_kinds():
    cfg = deepseek_v4_flash_hf_config()
    assert layers_of_kind(cfg, HCA, first_layer_idx=10, num_layers=5) == [11, 13]
    assert layers_of_kind(cfg, CSA, first_layer_idx=10, num_layers=5) == [10, 12, 14]
    assert layers_of_kind(cfg, SLIDING) == [0, 1]
    # a rank holding only layer 3 (HCA) allocates the HCA groups only
    g = V4FlashKvGeometry.from_config(cfg, max_seq_len=10240, sp_factor=1, first_layer_idx=3, num_layers=1)
    assert set(g.group_shapes()) == {"hca_unified", "hca_pending"}
    assert g.layers("hca_unified") == (3,) and g.layers("swa_window") == ()
    # the 4-rank split 11/11/11/10: rank 0 owns both SWA layers and the hash-MoE layers 0..2
    g0 = V4FlashKvGeometry.from_config(cfg, max_seq_len=10240, sp_factor=8, first_layer_idx=0, num_layers=11)
    assert g0.swa_layers == (0, 1) and g0.csa_layers == (2, 4, 6, 8, 10) and g0.hca_layers == (3, 5, 7, 9)
    g3 = V4FlashKvGeometry.from_config(cfg, max_seq_len=10240, sp_factor=8, first_layer_idx=33, num_layers=10)
    assert len(g3.csa_layers) == 5 and len(g3.hca_layers) == 5
