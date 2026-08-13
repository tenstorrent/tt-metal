# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import (
    TARGET_FABRIC,
    TARGET_MESH_SHAPE,
    TARGET_TOPOLOGY,
    TARGET_TP,
    MultichipDecoder,
)
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder


def test_multichip_derives_from_completed_optimized_baseline():
    assert issubclass(MultichipDecoder, OptimizedDecoder)
    assert MultichipDecoder is not OptimizedDecoder
    assert MultichipDecoder.optimization_profile["single_chip_baseline"] == "OptimizedDecoder"


def test_target_is_the_discovered_four_chip_blackhole_ring():
    assert TARGET_MESH_SHAPE == (1, 4)
    assert TARGET_TP == 4
    assert TARGET_FABRIC == ttnn.FabricConfig.FABRIC_1D_RING
    assert TARGET_TOPOLOGY == ttnn.Topology.Ring


def test_loader_is_tp_aware_and_does_not_call_single_chip_loader():
    source = inspect.getsource(MultichipDecoder.from_state_dict.__func__)
    assert "OptimizedDecoder.from_state_dict" not in source
    assert "FunctionalDecoder.from_state_dict" not in source
    assert "_validate_contract" in source
    assert "_configure_multichip_compute" in source


def test_all_semantic_weight_families_are_mesh_parallelized():
    source = inspect.getsource(MultichipDecoder.from_state_dict.__func__)
    for name in ("mlp_gate", "mlp_up", "mlp_down", "qkv_gate", "o_proj"):
        assert name in source
    linear_source = inspect.getsource(MultichipDecoder._load_linear_tensors)
    for name in ("linear_packed_decode", "linear_out_decode", "in_qkv", "in_z", "in_b", "in_a", "conv", "recurrent"):
        assert name in linear_source


def test_tp4_head_and_channel_shapes_are_exact():
    assert 24 // TARGET_TP == 6
    assert 4 // TARGET_TP == 1
    assert 17408 // TARGET_TP == 4352
    assert 16 // TARGET_TP == 4
    assert 48 // TARGET_TP == 12
    assert 5120 // TARGET_TP == 1280
    assert 6 * 256 == 1536
    assert 4 * 128 == 512
    assert 12 * 128 == 1536


def test_row_parallel_boundaries_reduce_on_ring():
    source = inspect.getsource(MultichipDecoder._tp_linear)
    collective = inspect.getsource(MultichipDecoder._all_reduce)
    assert 'self._multichip_candidate == "multichip_preallocated_ccl"' in source
    assert 'output_tensor=buffers["reduce_scatter"]' in source
    assert 'output_tensor=buffers["all_gather"]' in source
    assert "self._all_reduce(output)" in source
    assert "ttnn.all_reduce" in collective
    assert "TARGET_TOPOLOGY" in collective


def test_full_attention_uses_local_heads_and_local_paged_cache():
    decode = inspect.getsource(MultichipDecoder._full_attention_decode)
    prefill = inspect.getsource(MultichipDecoder._full_attention_prefill)
    for source in (decode, prefill):
        assert "num_heads=6" in source or "(self.batch, sequence, 6, 256)" in source
        assert 'self.caches["key"]' in source
        assert 'self.caches["value"]' in source
        assert "page_table" in source
        assert "current_positions" in source


def test_long_prefill_is_tp_local_and_keeps_logical_tail_internal():
    source = inspect.getsource(MultichipDecoder._full_attention_prefill_long)
    assert "n=1536" in source
    assert "n=256" in source
    assert "(self.batch, length, 6, 256)" in source
    assert "(self.batch, length, 1, 256)" in source
    assert "padding = (-length) % ttnn.TILE_SIZE" in source
    assert "row=True" in source


def test_public_layer_boundary_is_explicit_and_stack_compatible():
    assert MultichipDecoder.residual_layout == "replicated_hidden_5120"
    assert MultichipDecoder.optimization_profile["residual_layout"] == "replicated_hidden_5120"
