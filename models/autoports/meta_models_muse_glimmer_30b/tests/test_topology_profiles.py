# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only guards for the P150/P150x2/P150x4 topology profiles."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt import generator_vllm
from models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm import MuseGlimmerForConditionalGeneration
from models.autoports.meta_models_muse_glimmer_30b.tt.model import SUPPORTED_TP
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    MULTICHIP_BOUNDARY_CORES,
    P150X2_DECODE_MATMUL,
    P150X2_PREFILL_MCAST2D,
    P150X2_PREFILL_MINIMAL_BLOCKS,
    mesh_plan,
    multichip_decode_matmul,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    OptimizedDecoder,
    resolve_decode_swiglu_mul_cores,
)

TEXT_CONFIG = SimpleNamespace(
    num_attention_heads=32,
    num_key_value_heads=2,
    head_dim=128,
    hidden_size=6656,
    intermediate_size=19968,
)


@pytest.mark.parametrize(
    ("tp", "local_heads", "local_kv_heads", "kv_replicated", "local_intermediate"),
    [
        (1, 32, 2, False, 19968),
        (2, 16, 1, False, 9984),
        (4, 8, 1, True, 5120),
    ],
)
def test_qualified_topologies_have_exact_mesh_plans(
    tp: int,
    local_heads: int,
    local_kv_heads: int,
    kv_replicated: bool,
    local_intermediate: int,
):
    assert SUPPORTED_TP == (1, 2, 4)
    plan = mesh_plan(TEXT_CONFIG, tp, dram_banks=8)
    assert plan.local_heads == local_heads
    assert plan.local_kv_heads == local_kv_heads
    assert plan.kv_replicated is kv_replicated
    assert plan.local_intermediate == local_intermediate


def test_single_chip_decoder_accepts_a_shared_rope_cache_argument():
    assert "rope_cache" in OptimizedDecoder.from_state_dict.__func__.__annotations__


@pytest.mark.parametrize(
    ("local_intermediate", "expected_cores"),
    [(19968, None), (9984, None), (5120, 80)],
)
def test_swiglu_reshard_is_enabled_only_for_an_exact_topology_width(
    local_intermediate: int,
    expected_cores: int | None,
):
    assert resolve_decode_swiglu_mul_cores(local_intermediate) == expected_cores


def test_p150x2_decode_geometry_is_legal_for_every_supported_weight_dtype():
    k_by_role = {
        "wqkv": 6656,
        "attn_gate": 6656,
        "o_proj": 2048,
        "mlp_gate": 6656,
        "mlp_up": 6656,
        "mlp_down": 9984,
    }
    assert multichip_decode_matmul(2) is P150X2_DECODE_MATMUL
    for (role, _dtype), (cores, in0_block_w) in P150X2_DECODE_MATMUL.items():
        k_tiles = k_by_role[role] // ttnn.TILE_SIZE
        assert k_tiles % cores == 0, f"{role}: {cores} cores pad {k_tiles} K tiles"
        assert (k_tiles // cores) % in0_block_w == 0
        if role in ("wqkv", "attn_gate"):
            assert cores == MULTICHIP_BOUNDARY_CORES
    assert len({P150X2_DECODE_MATMUL[(role, ttnn.bfloat4_b)][0] for role in ("mlp_gate", "mlp_up", "mlp_down")}) == 1


def test_p150x2_prefill_geometry_never_uses_an_illegal_k_block():
    k_tiles_by_role = {
        "wqkv": 208,
        "attn_gate": 208,
        "o_proj": 64,
        "mlp_gate": 208,
        "mlp_up": 208,
        "mlp_down": 312,
    }
    for (role, _dtype), entries in P150X2_PREFILL_MCAST2D.items():
        for _max_rows, (_grid_y, in0_block_w) in entries:
            assert k_tiles_by_role[role] % in0_block_w == 0
    for (role, _dtype), entries in P150X2_PREFILL_MINIMAL_BLOCKS.items():
        for _min_rows, blocks in entries:
            if blocks is not None:
                assert k_tiles_by_role[role] % blocks[1] == 0


@pytest.mark.parametrize("tp", [0, 1, 3, 8])
def test_multichip_geometry_rejects_unqualified_widths(tp: int):
    with pytest.raises(  # allow-pytest.raises: pure selector rejects unpublished topologies
        ValueError, match="tp=2 or tp=4"
    ):
        multichip_decode_matmul(tp)


@pytest.mark.parametrize("num_devices", SUPPORTED_TP)
def test_vllm_loader_admits_every_published_topology(monkeypatch, num_devices):
    calls = []
    monkeypatch.setattr(generator_vllm, "_tt_config_block_size", lambda: 64)
    monkeypatch.setattr(
        generator_vllm,
        "build_generator",
        lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace(),
    )
    hf_config = SimpleNamespace(
        _name_or_path="meta-models/Muse-Glimmer-30B",
        text_config=SimpleNamespace(max_position_embeddings=131072),
    )
    mesh = SimpleNamespace(get_num_devices=lambda: num_devices)
    model = MuseGlimmerForConditionalGeneration.initialize_vllm_model(
        hf_config,
        mesh,
        max_batch_size=1,
        max_seq_len=131072,
    )
    assert model.max_num_seqs == 1
    assert len(calls) == 1
    assert calls[0][1]["max_num_blocks"] == 2048


def test_vllm_loader_rejects_an_unpublished_topology(monkeypatch):
    monkeypatch.setattr(
        generator_vllm,
        "build_generator",
        lambda *args, **kwargs: pytest.fail("rejected topology must not build the model"),
    )
    hf_config = SimpleNamespace(
        _name_or_path="meta-models/Muse-Glimmer-30B",
        text_config=SimpleNamespace(max_position_embeddings=131072),
    )
    mesh = SimpleNamespace(get_num_devices=lambda: 3)
    with pytest.raises(  # allow-pytest.raises: loader must fail before opening/building a model
        ValueError, match="1, 2, or 4 devices"
    ):
        MuseGlimmerForConditionalGeneration.initialize_vllm_model(
            hf_config,
            mesh,
            max_batch_size=1,
        )
