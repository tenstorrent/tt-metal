# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host tests of the GLM-5.4 capacity calculation: per-chip bytes against hand values and the GLM-5.3 check."""

from models.demos.deepseek_v3_d_p.utils import glm54_capacity_model as m


def test_per_user_bytes_8x4():
    # 1,064,960 / 32 = 33,280 rows per chip; 704 B pages; index-K 1040 x 4 bfp8 tiles of 1088 B.
    assert m.KVPE_PAGE_BYTES == 704
    assert m.user_bytes_per_layer(m.MESH_8X4) == {"kvpe": 23_429_120, "index_k": 4_526_080}


def test_weight_components_match_cache_shards():
    # Per-chip shard bytes read from the GLM-5.3 8x4 TTNN cache (q171.3.7 tensorbin log).
    mla = 3_342_336 + 8_912_896 + 940_032 + 6_684_672 + 2_228_224 + 26_738_688 + 11_264  # + norms
    indexer = 8_912_896 + 208_896 + 98_304 + 512
    moe = 24 * 7_077_888 + 3 * 3_342_336 + 786_432 + 512
    c = m.component_bytes(m.MESH_8X4)
    assert (c["mla"], c["indexer"], c["moe"]) == (mla, indexer, moe)
    assert c["dense_ffn"] == 3 * 20_054_016
    assert c["embedding"] == 475_791_360
    # 4x4: twice the routed experts per chip, everything else equal.
    assert m.component_bytes(m.MESH_4X4)["moe"] == moe + 24 * 7_077_888


def test_full_layer_counts():
    assert m.full_layers(0, 25) == 8  # 0, 1, 2, 6, 10, 14, 18, 22
    assert m.full_layers(26, 53) == 7  # 26, 30, ..., 50
    assert m.full_layers(54, 77) == 6  # 54, 58, ..., 74


def test_calibration_bounds():
    # 4x4 rank 0: layers 58-67 (full 58, 62, 66) + embedding; 55 fit, 56 fail (traced).
    w = 10 * (48_858_112 + 350_552_576) + 3 * 9_220_608 + 475_791_360
    u = 10 * 46_858_240 + 3 * 9_052_160
    assert m.CALIBRATION["traced"].budget_bounds() == (w + 55 * u, w + 56 * u)


def test_glm53_configured_counts_fit():
    """The counts SC4 CI runs (54 trunk, 52 MTP) must fit at the bisection-derived budget (mid), which the 4x4
    calibration and the mesh delta set; "low" is the CI floor itself and fits by construction."""
    for split, configured in (("trunk", 54), ("mtp_kv_share", 52)):
        stages = m._glm53_stages()[split]
        assert m._pipeline(stages, "traced")["pipeline_users"]["mid"] >= configured


def test_g2_has_no_index_cache():
    g2 = m._glm54_stages()[2]
    assert g2.index_layers == 0 and "index_k" not in g2.per_user()
    assert g2.kvpe_layers == 28


def test_mesh_budget_delta():
    # Dispatch tokens per chip: sp x 640 x 8 + 32 x (experts per chip - 1) = 41,184 (8x4) and 20,960 (4x4).
    t8, t4 = 8 * 640 * 8 + 32 * 7, 4 * 640 * 8 + 32 * 15
    l8 = m.mesh_layer_independent_bytes(m.MESH_8X4, 5120)
    l4 = m.mesh_layer_independent_bytes(m.MESH_4X4, 2560)
    assert l8["moe_dispatched_bf16"] == t8 * 6144 * 2 == 506_068_992
    assert l4["moe_expert_outputs_bfp8"] == t4 * 6144 * 1088 / 1024
    assert l8["rope_cos_sin"] == 133_120 * 64 * 4 and l4["rope_cos_sin"] == 266_240 * 64 * 4
    delta = (t8 - t4) * 6144 * (2 + 1088 / 1024) - 133_120 * 64 * 4
    assert m.mesh_budget_delta() == delta  # 8x4 has less room than the 4x4 calibration
    assert delta > 0


def test_trunk_last_rank_builds_last_layer_kv_only():
    with_mtp = m.Stage("S", (54, 77), mtp=True)
    trunk = m.Stage("S", (54, 77))
    assert trunk.kv_only_last_layer and not with_mtp.kv_only_last_layer
    c = m.component_bytes(m.MESH_8X4)
    w = trunk.weights()
    assert w["mla"] == 23 * c["mla"] and w["moe"] == 23 * c["moe"]
    assert w["mla_kv_only"] == 940_032 + 1024 + 3072  # kv_a_proj_with_mqa + kv_a_layernorm + attn_norm
    assert trunk.kvpe_layers == 24  # the kv-only layer still writes its KVPE slot
