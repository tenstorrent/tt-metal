# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5._stubs import hunyuan_video15_transformer_block as block


def test_sdpa_chunks_keep_hunyuan_default():
    assert block._select_sdpa_chunks(
        sequence_parallel=True,
        sp=8,
        tp=4,
        blackhole=True,
        environ={},
    ) == (128, 512)


def test_wan_sdpa_preset_is_explicit_and_shape_gated():
    env = {"HY_DIT_SDPA_PRESET": "wan_bh_sp8tp4"}
    assert block._select_sdpa_chunks(
        sequence_parallel=True,
        sp=8,
        tp=4,
        blackhole=True,
        environ=env,
    ) == (288, 512)

    with pytest.raises(ValueError, match="requires Blackhole"):
        block._select_sdpa_chunks(
            sequence_parallel=True,
            sp=4,
            tp=4,
            blackhole=True,
            environ=env,
        )


@pytest.mark.parametrize(
    "env",
    [
        {"HY_DIT_SDPA_Q_CHUNK": "0"},
        {"HY_DIT_SDPA_K_CHUNK": "129"},
        {"HY_DIT_SDPA_PRESET": "unknown"},
    ],
)
def test_sdpa_chunk_configuration_rejects_unsafe_values(env):
    with pytest.raises(ValueError):
        block._select_sdpa_chunks(
            sequence_parallel=True,
            sp=8,
            tp=4,
            blackhole=True,
            environ=env,
        )


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_boolean_flag_accepts_enabled_values(value):
    assert block._enabled(value)


def test_boolean_flag_rejects_typos():
    with pytest.raises(ValueError, match="boolean"):
        block._enabled("maybe")


def test_sharded_layer_norm_l1_guard_accounts_for_circular_buffers():
    assert block._layer_norm_shard_fits(64, 256, 2)
    assert not block._layer_norm_shard_fits(800, 256, 2)


def test_mmrs_overlap_is_explicit_and_tp4_bf16_blackhole_gated():
    env = {"HY_DIT_MMRS_OVERLAP": "1"}
    assert block._select_collective_overlap(
        sharded=True,
        tp=4,
        blackhole=True,
        bf16=True,
        topology=ttnn.Topology.Ring,
        environ=env,
    )

    invalid = (
        dict(sharded=False, tp=4, blackhole=True, bf16=True),
        dict(sharded=True, tp=8, blackhole=True, bf16=True),
        dict(sharded=True, tp=4, blackhole=False, bf16=True),
        dict(sharded=True, tp=4, blackhole=True, bf16=False),
    )
    for kwargs in invalid:
        with pytest.raises(ValueError, match="Blackhole TP=4"):
            block._select_collective_overlap(environ=env, topology=ttnn.Topology.Ring, **kwargs)


def test_mmrs_overlap_rejects_the_galaxy_linear_topology():
    # minimal_matmul_strided_reduce_scatter_async TT_FATALs on non-Ring topology
    # inside the device op; the Hunyuan Galaxy CCLManager is built Linear.
    with pytest.raises(ValueError, match="Ring CCL topology"):
        block._select_collective_overlap(
            sharded=True,
            tp=4,
            blackhole=True,
            bf16=True,
            topology=ttnn.Topology.Linear,
            environ={"HY_DIT_MMRS_OVERLAP": "1"},
        )


def test_mmrs_overlap_defaults_to_legacy_without_restricting_other_topologies():
    assert not block._select_collective_overlap(
        sharded=False,
        tp=1,
        blackhole=False,
        bf16=False,
        topology=None,
        environ={},
    )


def test_rs_domain_bias_defaults_off_and_is_tp_gated():
    assert not block._select_rs_domain_bias(sharded=True, tp=4, environ={})
    assert block._select_rs_domain_bias(sharded=True, tp=4, environ={"HY_DIT_RS_DOMAIN_BIAS": "1"})
    with pytest.raises(ValueError, match="tensor-parallel"):
        block._select_rs_domain_bias(sharded=False, tp=1, environ={"HY_DIT_RS_DOMAIN_BIAS": "1"})


def test_rs_domain_bias_requires_a_tile_aligned_shard():
    assert block._row_bias_shard_width(3072, 4) == 768
    with pytest.raises(ValueError, match="tile-aligned"):
        block._row_bias_shard_width(3072, 5)  # not divisible by tp
    with pytest.raises(ValueError, match="tile-aligned"):
        block._row_bias_shard_width(128, 8)  # 16 columns/device is sub-tile
    with pytest.raises(ValueError, match="tp>1"):
        block._row_bias_shard_width(3072, 1)


@pytest.mark.parametrize("flag", ["HY_DIT_FUSED_HEADS", "HY_DIT_FUSED_QKV_HEADS"])
def test_head_layout_fusions_default_off_and_parse_strictly(flag, monkeypatch):
    # Both default off: they are opt-in until the 720p A/B and a clean
    # re-baseline land, even though 480p output is bit-identical.
    monkeypatch.delenv(flag, raising=False)
    assert not block._enabled(os.environ.get(flag, "0"))
    for value in ("1", "true", "YES", "on"):
        assert block._enabled(value)
    # A typo must fail loudly rather than silently selecting the legacy path --
    # otherwise a mis-spelled flag reads as "the optimization did nothing".
    with pytest.raises(ValueError, match="boolean"):
        block._enabled("fused")


def test_dual_stream_overlap_execution_order_keeps_dependencies_exact():
    events = []

    def hidden_start():
        events.append("hidden.mm_rs")
        return "hidden.rs"

    def context_complete():
        events.append("context.mm_rs_ag")
        return "context.full"

    def hidden_finish(value):
        assert value == "hidden.rs"
        events.append("hidden.ag_bias")
        return "hidden.full"

    outputs = block._run_dual_stream_projection_schedule(
        True,
        hidden_start=hidden_start,
        context_complete=context_complete,
        hidden_finish=hidden_finish,
    )
    assert events == ["hidden.mm_rs", "context.mm_rs_ag", "hidden.ag_bias"]
    assert outputs == ("hidden.full", "context.full")


def test_legacy_dual_stream_schedule_does_not_finish_twice():
    events = []
    outputs = block._run_dual_stream_projection_schedule(
        False,
        hidden_start=lambda: events.append("hidden.legacy") or "hidden.full",
        context_complete=lambda: events.append("context.legacy") or "context.full",
        hidden_finish=lambda value: events.append("unexpected.finish"),
    )
    assert events == ["hidden.legacy", "context.legacy"]
    assert outputs == ("hidden.full", "context.full")
