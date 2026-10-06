# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests of the GDN program-configuration table, the GDN case registry and its CPU-reference cache keys."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn
from models.demos.deepseek_v3_d_p.reference.gdn import GDNReferenceState
from models.demos.deepseek_v3_d_p.tests.gdn.cases import (
    GDN_CASES,
    GDN_MODELS,
    LAYOUTS,
    TOY_GDN_CONFIG,
    build_gdn_case,
    case_program_config,
    registered_gdn_case,
    synthetic_gdn_weights,
)
from models.demos.deepseek_v3_d_p.tests.gdn.reference_cache import cpu_reference_cache_path
from models.demos.deepseek_v3_d_p.tt.gdn.config import gdn_program_config


def test_program_config_table() -> None:
    """Design §3.5: SP > 1 at 640 rows is grouped with G = 2 (10-chunk groups); one SP rank is direct."""
    sp = gdn_program_config(active_seq_len_local=640, sequence_parallel_size=8, tp_ccl_topology=ttnn.Topology.Linear)
    assert (sp.recurrence.local_scan_strategy, sp.recurrence.summary_group_chunks) == ("grouped", 10)
    for rows in (640, 1280):
        direct = gdn_program_config(
            active_seq_len_local=rows, sequence_parallel_size=1, tp_ccl_topology=ttnn.Topology.Ring
        )
        assert direct.recurrence.local_scan_strategy == "direct"
        assert direct.tp_ccl_topology == ttnn.Topology.Ring
    assert (sp.qkv_channel_chunk_size, sp.gated_rms_output_dtype, sp.output_projection_math_fidelity) == (
        512,
        ttnn.bfloat16,
        ttnn.MathFidelity.HiFi2,
    )
    assert not sp.tuned_projection_matmuls


def test_program_config_rejects_unknown_geometries(expect_error) -> None:
    for rows, sp_size in ((1280, 2), (320, 8), (2560, 1), (672, 1)):
        with expect_error(ValueError, "no GDN recurrence configuration"):
            gdn_program_config(
                active_seq_len_local=rows, sequence_parallel_size=sp_size, tp_ccl_topology=ttnn.Topology.Linear
            )


def test_registered_layouts_resolve_to_the_table() -> None:
    for layout, (mesh_shape, tensor_parallel_axis, _, _) in LAYOUTS.items():
        recurrence = case_program_config(registered_gdn_case("toy", layout, "single")).recurrence
        expected = "grouped" if mesh_shape[1 - tensor_parallel_axis] > 1 else "direct"
        assert recurrence.local_scan_strategy == expected, layout


def test_registry_covers_every_model_layout_schedule_and_rank_slices() -> None:
    for model in GDN_MODELS:
        config = GDN_MODELS[model].config()
        for layout in LAYOUTS:
            for schedule in ("single", "chained3", "ragged"):
                spec = registered_gdn_case(model, layout, schedule)
                assert spec.name in GDN_CASES
                if LAYOUTS[layout][3]:
                    # One Galaxy TP4 rank: whole K-head groups.
                    assert spec.key_head_slice == (0, config.num_key_heads // 4)
                    sliced = spec.weight_source().config
                    assert sliced.num_value_heads == config.num_value_heads // 4
                else:
                    assert spec.key_head_slice is None
    # Real-weight cells of the accuracy beads are expressible: random and real-text inputs at LB-A and LB-B.
    assert registered_gdn_case("qwen38_2_4t", "LB-B", "ragged", "real", "text").name in GDN_CASES
    ragged = registered_gdn_case("toy", "LB-A", "ragged")
    assert ragged.chunk_valid_tokens == (1280, 992)  # ends inside SP rank 1


def test_synthetic_weights_are_deterministic_with_unsaturated_gates() -> None:
    first, second = synthetic_gdn_weights(TOY_GDN_CONFIG), synthetic_gdn_weights(TOY_GDN_CONFIG)
    assert first.keys() == second.keys() and all(torch.equal(first[name], second[name]) for name in first)
    hidden = torch.randn(512, TOY_GDN_CONFIG.hidden_size, generator=torch.Generator().manual_seed(0))
    a = hidden @ first["in_proj_a.weight"].float().T
    g = -first["A_log"].exp() * F.softplus(a + first["dt_bias"])
    assert -20.0 < float(g.min()) < -5.0 and -1e-3 < float(g.max()) < -1e-5  # fast-forgetting and long-memory heads
    beta = torch.sigmoid(hidden @ first["in_proj_b.weight"].float().T)
    assert 0.01 < float(beta.min()) and float(beta.max()) < 0.99


def test_reference_cache_key_follows_input_and_initial_state() -> None:
    case = build_gdn_case(registered_gdn_case("toy", "1x1", "chained3"))
    hidden = case.chunk_valid_hidden(0)
    base = cpu_reference_cache_path(case.weights, hidden, None)
    assert base == cpu_reference_cache_path(case.weights, hidden.clone(), None)
    assert base != cpu_reference_cache_path(case.weights, case.chunk_valid_hidden(1), None)
    state = GDNReferenceState.zeros(case.config)
    assert base != cpu_reference_cache_path(case.weights, hidden, state)
    state.recurrent[0, 0, 0] = 1.0
    assert cpu_reference_cache_path(case.weights, hidden, GDNReferenceState.zeros(case.config)) != (
        cpu_reference_cache_path(case.weights, hidden, state)
    )
    sliced = build_gdn_case(registered_gdn_case("toy", "LB-A", "single"))
    assert cpu_reference_cache_path(sliced.weights, sliced.chunk_valid_hidden(0)[:640], None) != base
