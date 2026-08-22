# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in D2 qualification probe for whole-chunk Laguna MoE token dispatch.

This is deliberately a test-only Stage-2 prototype.  It does not switch the
production decoder, alter weight loading, or enable a serving flag.  The probe
compares the current 32 x 256-token ``MultichipDecoder._mlp`` implementation
against an exact 8192-token device-side dispatch/compact-FFN/combine path.

The hardware test is fail-closed and opens only an explicitly selected P150x2:

    TT_RUN_LAGUNA_TOKEN_DISPATCH_PROBE=1
    TT_VISIBLE_DEVICES=2,3
    LAGUNA_PROFILE=p150x2

Expert weights are duplicated into DeepSeek's per-local-expert representation
only for the lifetime of this test.  The production packed expert tensors remain
untouched and are used by the slice-loop reference.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time

import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    close_mesh,
    memory_snapshot,
    open_mesh,
    resolve_profile,
)
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import MultichipDecoder, _sparse_pc
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    get_dispatch_input_mesh_mapper,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_combine import TtCombineModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_dispatch import TtDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_routing_setup import TtMoERoutingSetup
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert

RUN_DEVICE_PROBE = os.environ.get("TT_RUN_LAGUNA_TOKEN_DISPATCH_PROBE") == "1"
RUN_STACKED_PRODUCTION_PROBE = os.environ.get("TT_RUN_LAGUNA_STACKED_TOKEN_DISPATCH_PROBE") == "1"

LAYER = 1
TOKENS = 8192
HIDDEN = 2048
INTERMEDIATE = 512
GLOBAL_EXPERTS = 256
LOCAL_EXPERTS = 128
TOP_K = 8
DISPATCH_GROUP_SIZE = 1
NUM_DISPATCH_GROUPS = 2
METADATA_LEN = 5
MAX_TOKENS_PER_EXPERT = TOKENS
CHUNK_M_TILES = int(os.environ.get("TT_LAGUNA_TOKEN_DISPATCH_CHUNK_M_TILES", "16"))
TILE = 32
ACTIVATION_DTYPE_NAME = os.environ.get("TT_LAGUNA_TOKEN_DISPATCH_ACTIVATIONS", "bf16").strip().lower()
FIDELITY_NAME = os.environ.get("TT_LAGUNA_TOKEN_DISPATCH_FIDELITY", "lofi").strip().lower()
PRESERVE_SLICE_SEMANTICS = os.environ.get("TT_LAGUNA_TOKEN_DISPATCH_PRESERVE_SLICE_SEMANTICS", "0") == "1"

# If every route lands on one ASIC, tile padding adds at most 31 rows for each
# of that ASIC's 128 experts.  This is the exact fail-safe capacity bound.
MAX_DISPATCH_ROWS = TOKENS * TOP_K + (TILE - 1) * LOCAL_EXPERTS


def _dispatch_table() -> torch.Tensor:
    return ExpertMapping.create_dispatch_table(
        num_routed_experts=GLOBAL_EXPERTS,
        dispatch_group_size=DISPATCH_GROUP_SIZE,
        num_dispatch_groups=NUM_DISPATCH_GROUPS,
    )


def test_d2_whole_chunk_capacity_and_mapping_contract():
    """The production D2 expert partition fits the conservative flat buffer."""
    table = _dispatch_table()
    assert table.shape == (2, 256)
    assert torch.equal(table[0, :LOCAL_EXPERTS], torch.zeros(LOCAL_EXPERTS, dtype=torch.int32))
    assert torch.count_nonzero(table[0, LOCAL_EXPERTS:] + 1) == 0
    assert torch.count_nonzero(table[1, :LOCAL_EXPERTS] + 1) == 0
    assert torch.equal(table[1, LOCAL_EXPERTS:], torch.zeros(LOCAL_EXPERTS, dtype=torch.int32))
    assert MAX_DISPATCH_ROWS == 69_504
    assert MAX_DISPATCH_ROWS % TILE == 0
    assert CHUNK_M_TILES in {16, 24, 32, 40, 48, 56, 64}
    assert ACTIVATION_DTYPE_NAME in {"bf8", "bf16"}
    assert FIDELITY_NAME in {"lofi", "hifi2"}
    assert isinstance(PRESERVE_SLICE_SEMANTICS, bool)


def test_local_only_weighted_slot_reduction_matches_dense_ep_semantics():
    """Combine restores route slots; routing weights are applied in the following reduction."""
    indices = torch.tensor([[[0, 128], [129, 1], [2, 130]]], dtype=torch.int64)
    weights = torch.tensor([[[0.25, 0.75], [0.60, 0.40], [0.125, 0.875]]], dtype=torch.float32)
    expert_rows = torch.tensor(
        [
            [[1.0, 2.0], [10.0, 20.0]],
            [[3.0, 6.0], [30.0, 60.0]],
            [[5.0, 10.0], [50.0, 100.0]],
        ],
        dtype=torch.float32,
    )

    # Size-one dispatch groups leave only locally owned routes populated on
    # each ASIC.  This is the exact shape/ownership contract returned by
    # DeepSeek combine before post_combine_reduce.
    combined_slots = torch.zeros((NUM_DISPATCH_GROUPS, 1, 3, 2, 2), dtype=torch.float32)
    for token in range(3):
        for slot in range(2):
            owner = int(indices[0, token, slot]) // LOCAL_EXPERTS
            combined_slots[owner, 0, token, slot] = expert_rows[token, slot]

    weighted_local = (combined_slots * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=3)
    reduced = weighted_local.sum(dim=0)

    dense = torch.zeros((3, GLOBAL_EXPERTS), dtype=torch.float32)
    dense.scatter_(1, indices[0], weights[0])
    expected = torch.zeros((1, 3, 2), dtype=torch.float32)
    for token in range(3):
        for slot in range(2):
            expected[0, token] += dense[token, indices[0, token, slot]] * expert_rows[token, slot]

    torch.testing.assert_close(reduced, expected, atol=0, rtol=0)
    assert not torch.allclose(combined_slots.sum(dim=(0, 3)), expected)


def _safe_probe_environment() -> None:
    visible = [part.strip() for part in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if part.strip()]
    if visible != ["2", "3"]:
        pytest.fail(
            "the opt-in probe may open only physical chips 2,3; "
            f"got TT_VISIBLE_DEVICES={os.environ.get('TT_VISIBLE_DEVICES')!r}"
        )
    if os.environ.get("LAGUNA_PROFILE") != "p150x2":
        pytest.fail("the opt-in probe requires LAGUNA_PROFILE=p150x2")
    if os.environ.get("TT_LAGUNA_MOE_PREFILL_TILE_SPARSE", "0").strip().lower() not in {"0", "false"}:
        pytest.fail("disable the unqualified tile-sparse Stage-1 path for the slice-loop reference")
    if ACTIVATION_DTYPE_NAME not in {"bf8", "bf16"}:
        pytest.fail("TT_LAGUNA_TOKEN_DISPATCH_ACTIVATIONS must be 'bf8' or 'bf16'")
    if FIDELITY_NAME not in {"lofi", "hifi2"}:
        pytest.fail("TT_LAGUNA_TOKEN_DISPATCH_FIDELITY must be 'lofi' or 'hifi2'")
    if CHUNK_M_TILES not in {16, 24, 32, 40, 48, 56, 64}:
        pytest.fail("TT_LAGUNA_TOKEN_DISPATCH_CHUNK_M_TILES must be one of 16,24,32,40,48,56,64")


def _prototype_dtype():
    return {"bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[ACTIVATION_DTYPE_NAME]


def _prototype_compute_config(decoder: MultichipDecoder):
    return {"lofi": decoder._ck_lofi, "hifi2": decoder._ck_hifi2}[FIDELITY_NAME]


def _weighted_route_reduce(combined_slots, weights, indices, dispatch_table):
    """Apply Laguna's top-k weights and sum local route slots on device.

    DeepSeek combine is intentionally an unweighted permutation.  Its paired
    post_combine_reduce primitive performs the fused BF16 multiply + top-k sum
    and uses the sharded dispatch table to skip routes owned by the other ASIC.
    """
    weights_5d = ttnn.unsqueeze(ttnn.unsqueeze(weights, dim=-1), dim=0)
    return ttnn.experimental.deepseek_prefill.post_combine_reduce(
        combined_slots,
        weights_5d,
        indices,
        dispatch_table,
        expert_dim=3,
        output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _print_memory(mesh, label: str) -> dict[str, int | float | str]:
    snapshot = memory_snapshot(ttnn, mesh, label)
    print("LAGUNA_TOKEN_DISPATCH_MEMORY", json.dumps(snapshot, sort_keys=True), flush=True)
    return snapshot


def _timed(mesh, fn):
    ttnn.synchronize_device(mesh)
    start = time.perf_counter()
    result = fn()
    ttnn.synchronize_device(mesh)
    return result, time.perf_counter() - start


def _compose_replicated0(tensor, mesh) -> torch.Tensor:
    composed = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0), dtype=torch.bfloat16)
    return composed[0:1]


def _streaming_pcc(reference: torch.Tensor, candidate: torch.Tensor, rows_per_chunk: int = 256) -> tuple[float, float]:
    """Full-tensor PCC without materialising two float64 flattened copies."""
    assert reference.shape == candidate.shape
    ref = reference.reshape(-1, reference.shape[-1])
    got = candidate.reshape(-1, candidate.shape[-1])
    n = 0
    sx = sy = sxx = syy = sxy = 0.0
    max_abs = 0.0
    for start in range(0, ref.shape[0], rows_per_chunk):
        x = ref[start : start + rows_per_chunk].float()
        y = got[start : start + rows_per_chunk].float()
        n += x.numel()
        sx += x.sum(dtype=torch.float64).item()
        sy += y.sum(dtype=torch.float64).item()
        sxx += (x * x).sum(dtype=torch.float64).item()
        syy += (y * y).sum(dtype=torch.float64).item()
        sxy += (x * y).sum(dtype=torch.float64).item()
        max_abs = max(max_abs, (x - y).abs().max().item())
    covariance = sxy - sx * sy / n
    variance_x = sxx - sx * sx / n
    variance_y = syy - sy * sy / n
    pcc = covariance / math.sqrt(max(variance_x * variance_y, 1e-30))
    return float(pcc), max_abs


def _actual_router(decoder: MultichipDecoder, x, tokens: int = TOKENS):
    """Run Laguna's learned router once over the full outer prefill chunk."""
    cfg = decoder.cfg
    x_flat = ttnn.reshape(x, (1, 1, tokens, HIDDEN))
    logits = ttnn.linear(x_flat, decoder.w["gate_w"], compute_kernel_config=decoder._ck_router)
    scores = ttnn.sigmoid(logits)
    selected_scores = ttnn.add(scores, decoder.w["e_bias"])
    _, indices = ttnn.topk(ttnn.typecast(selected_scores, ttnn.bfloat16), k=TOP_K, dim=-1, sorted=True)
    weights = ttnn.gather(scores, dim=3, index=indices)
    if cfg.norm_topk_prob:
        weights = ttnn.div(weights, ttnn.sum(weights, dim=3, keepdim=True))
    if cfg.routed_scaling != 1.0:
        weights = ttnn.multiply(weights, cfg.routed_scaling)
    return (
        ttnn.reshape(ttnn.to_layout(weights, ttnn.ROW_MAJOR_LAYOUT), (1, tokens, TOP_K)),
        ttnn.reshape(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), (1, tokens, TOP_K)),
    )


def _actual_router_sliced(decoder: MultichipDecoder, x):
    """Preserve the current 32 x 256-row router program/numerics."""
    weights = []
    indices = []
    for start in range(0, TOKENS, decoder.MOE_PREFILL_CHUNK):
        end = min(start + decoder.MOE_PREFILL_CHUNK, TOKENS)
        x_slice = ttnn.slice(x, [0, start, 0], [1, end, HIDDEN])
        weights_slice, indices_slice = _actual_router(decoder, x_slice, end - start)
        weights.append(weights_slice)
        indices.append(indices_slice)
    return ttnn.concat(weights, dim=1), ttnn.concat(indices, dim=1)


def _shared_partial(decoder: MultichipDecoder, x, *, sliced: bool):
    """Run the shared expert either whole-M or with established 256-row semantics."""
    if not sliced:
        return decoder._glu_mlp(
            ttnn.reshape(x, (1, 1, TOKENS, HIDDEN)),
            "sh",
            HIDDEN,
            decoder.cfg.shared_intermediate,
            decoder._ck_shared,
            False,
        )
    partials = []
    for start in range(0, TOKENS, decoder.MOE_PREFILL_CHUNK):
        end = min(start + decoder.MOE_PREFILL_CHUNK, TOKENS)
        x_slice = ttnn.slice(x, [0, start, 0], [1, end, HIDDEN])
        partials.append(
            decoder._glu_mlp(
                ttnn.reshape(x_slice, (1, 1, end - start, HIDDEN)),
                "sh",
                HIDDEN,
                decoder.cfg.shared_intermediate,
                decoder._ck_shared,
                False,
            )
        )
    return ttnn.concat(partials, dim=2)


def _established_slice_loop_mlp(decoder: MultichipDecoder, x):
    """Call the old path directly even when the production opt-in is set."""
    x_flat = ttnn.reshape(x, (1, 1, TOKENS, HIDDEN))
    outputs = []
    for start in range(0, TOKENS, decoder.MOE_PREFILL_CHUNK):
        end = min(start + decoder.MOE_PREFILL_CHUNK, TOKENS)
        chunk = ttnn.slice(x_flat, [0, 0, start, 0], [1, 1, end, HIDDEN])
        outputs.append(decoder._moe(chunk, end - start, False))
    return ttnn.concat(outputs, dim=2)


def _build_test_local_expert_weights(raw: dict) -> list[dict[str, torch.Tensor]]:
    """Reference the real layer weights without copying their host storage."""
    return [
        {
            "gate_proj": raw[f"mlp.experts.{expert}.gate_proj.weight"],
            "up_proj": raw[f"mlp.experts.{expert}.up_proj.weight"],
            "down_proj": raw[f"mlp.experts.{expert}.down_proj.weight"],
        }
        for expert in range(GLOBAL_EXPERTS)
    ]


def _validate_selected_rows_exact(
    dispatched_buffer,
    offsets,
    counts,
    indices_host: torch.Tensor,
    x_host: torch.Tensor,
    mesh,
) -> tuple[int, int]:
    """Check every valid packed row and its exact expert-region plan."""
    composer = get_ep_mesh_composer(mesh)
    packed = ttnn.to_torch(dispatched_buffer, mesh_composer=composer, dtype=torch.bfloat16)
    offsets_host = ttnn.to_torch(ttnn.unsqueeze_to_4D(offsets), mesh_composer=composer).squeeze(2).int()
    counts_host = ttnn.to_torch(ttnn.unsqueeze_to_4D(counts), mesh_composer=composer).squeeze(2).int()

    checked_rows = 0
    padded_rows = 0
    for group in range(NUM_DISPATCH_GROUPS):
        expert_start = group * LOCAL_EXPERTS
        expert_end = expert_start + LOCAL_EXPERTS
        cursor = 0
        for expert in range(expert_start, expert_end):
            routes = torch.nonzero(indices_host[0] == expert, as_tuple=False)
            expected_count = routes.shape[0]
            count = int(counts_host[group, 0, expert])
            offset = int(offsets_host[group, 0, expert])
            assert count == expected_count, f"expert {expert}: count {count} != {expected_count}"
            assert offset == cursor, f"expert {expert}: region offset {offset} != {cursor}"
            if count:
                expected = x_host[0, routes[:, 0]]
                actual = packed[group, 0, offset : offset + count]
                assert torch.equal(actual, expected), f"expert {expert}: dispatched token rows changed"
            checked_rows += count
            cursor += math.ceil(count / TILE) * TILE
        assert cursor <= MAX_DISPATCH_ROWS
        padded_rows += cursor

    assert checked_rows == TOKENS * TOP_K
    print(
        f"LAGUNA_TOKEN_DISPATCH_SELECTED_ROWS exact=true pcc=1.00000000 "
        f"rows={checked_rows} padded_rows={padded_rows}",
        flush=True,
    )
    return checked_rows, padded_rows


def _compose_asics_dim1(tensor, mesh) -> torch.Tensor:
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1),
        dtype=torch.bfloat16,
    )


def _packed_vs_separate_stage_diagnostics(
    decoder: MultichipDecoder,
    routed_expert: TtRoutedExpert,
    x,
    mesh,
) -> tuple[dict[str, float], torch.Tensor, int]:
    """Compare one expert/ASIC against the established packed sparse path.

    This deliberately uses only the first established 256-row MoE slice.  The
    packed path computes every row for each active expert, so local expert 0 on
    both ASICs can be compared row-for-row with the same expert weights loaded
    into the prototype's separate gate/up/down representation.  It isolates
    weight packing/layout and each unfused FFN boundary without changing the
    production decoder.
    """
    tokens = decoder.MOE_PREFILL_CHUNK
    x_slice = ttnn.slice(x, [0, 0, 0], [1, tokens, HIDDEN])
    x_flat = ttnn.reshape(x_slice, (1, 1, tokens, HIDDEN))

    logits = ttnn.linear(x_flat, decoder.w["gate_w"], compute_kernel_config=decoder._ck_router)
    scores = ttnn.sigmoid(logits)
    selected_scores = ttnn.add(scores, decoder.w["e_bias"])
    _, indices = ttnn.topk(ttnn.typecast(selected_scores, ttnn.bfloat16), k=TOP_K, dim=-1, sorted=True)
    selected_weights = ttnn.gather(scores, dim=3, index=indices)
    if decoder.cfg.norm_topk_prob:
        selected_weights = ttnn.div(selected_weights, ttnn.sum(selected_weights, dim=3, keepdim=True))
    if decoder.cfg.routed_scaling != 1.0:
        selected_weights = ttnn.multiply(selected_weights, decoder.cfg.routed_scaling)
    dense = ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=selected_weights)
    dense_local = ttnn.matmul(dense, decoder.w["ep_sel"], compute_kernel_config=decoder._ck_router)
    union = ttnn.sum(dense_local, dim=2, keepdim=True)
    sparsity = ttnn.to_layout(union, ttnn.ROW_MAJOR_LAYOUT)

    packed_gu = ttnn.sparse_matmul(
        x_flat,
        decoder.w["exp_gate_up"],
        sparsity=sparsity,
        program_config=_sparse_pc(2 * INTERMEDIATE, tokens, HIDDEN),
        compute_kernel_config=decoder._ck_moe,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=ttnn.Tile([TILE, TILE]),
    )
    packed_gu = ttnn.reshape(packed_gu, (1, LOCAL_EXPERTS, tokens, 2 * INTERMEDIATE))
    packed_gate = ttnn.slice(packed_gu, [0, 0, 0, 0], [1, LOCAL_EXPERTS, tokens, INTERMEDIATE])
    packed_up = ttnn.slice(
        packed_gu,
        [0, 0, 0, INTERMEDIATE],
        [1, LOCAL_EXPERTS, tokens, 2 * INTERMEDIATE],
    )
    packed_activated = ttnn.mul(ttnn.silu(packed_gate), packed_up)
    packed_down = ttnn.sparse_matmul(
        packed_activated,
        decoder.w["exp_down"],
        sparsity=sparsity,
        is_input_a_sparse=True,
        program_config=_sparse_pc(HIDDEN, tokens, INTERMEDIATE),
        compute_kernel_config=decoder._ck_moe,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=ttnn.Tile([TILE, TILE]),
    )

    indices_host = _compose_replicated0(indices, mesh).reshape(1, tokens, TOP_K).int()
    expert = next(
        local
        for local in range(LOCAL_EXPERTS)
        if torch.any(indices_host == local) and torch.any(indices_host == local + LOCAL_EXPERTS)
    )
    packed_gate_one = ttnn.slice(packed_gate, [0, expert, 0, 0], [1, expert + 1, tokens, INTERMEDIATE])
    packed_up_one = ttnn.slice(packed_up, [0, expert, 0, 0], [1, expert + 1, tokens, INTERMEDIATE])
    packed_activated_one = ttnn.slice(
        packed_activated,
        [0, expert, 0, 0],
        [1, expert + 1, tokens, INTERMEDIATE],
    )
    packed_down_one = ttnn.slice(packed_down, [0, expert, 0, 0], [1, expert + 1, tokens, HIDDEN])

    compute_config = _prototype_compute_config(decoder)
    separate_gate = ttnn.linear(
        x_flat,
        routed_expert.gate_projs[expert],
        compute_kernel_config=compute_config,
    )
    separate_up = ttnn.linear(
        x_flat,
        routed_expert.up_projs[expert],
        compute_kernel_config=compute_config,
    )
    separate_activated = ttnn.mul(ttnn.silu(separate_gate), separate_up)
    separate_down = ttnn.linear(
        separate_activated,
        routed_expert.down_projs[expert],
        compute_kernel_config=compute_config,
    )

    pairs = {
        "ff1_gate_pcc": (_compose_asics_dim1(packed_gate_one, mesh), _compose_asics_dim1(separate_gate, mesh)),
        "ff1_up_pcc": (_compose_asics_dim1(packed_up_one, mesh), _compose_asics_dim1(separate_up, mesh)),
        "activation_pcc": (
            _compose_asics_dim1(packed_activated_one, mesh),
            _compose_asics_dim1(separate_activated, mesh),
        ),
        "ff2_pcc": (_compose_asics_dim1(packed_down_one, mesh), _compose_asics_dim1(separate_down, mesh)),
    }
    metrics = {name: _streaming_pcc(reference, candidate)[0] for name, (reference, candidate) in pairs.items()}
    metrics["representative_local_expert"] = expert
    packed_down_host = pairs["ff2_pcc"][0]
    print("LAGUNA_TOKEN_DISPATCH_FFN_STAGES", json.dumps(metrics, sort_keys=True), flush=True)
    return metrics, packed_down_host, expert


def _prototype_stage_diagnostics(
    *,
    expert_outputs,
    region_offsets,
    counts,
    global_expert_idx,
    combined_slots,
    routed_local,
    weights_host: torch.Tensor,
    indices_host: torch.Tensor,
    packed_down_host: torch.Tensor,
    representative_local_expert: int,
    mesh,
) -> dict[str, float]:
    """Isolate fused FF2, raw combine permutation, and weighted reduction PCC."""
    extracted = ttnn.experimental.deepseek_prefill.extract(
        expert_outputs,
        region_offsets,
        counts,
        global_expert_idx,
        local_expert_id=representative_local_expert,
        max_dispatched_tokens_per_expert=MAX_TOKENS_PER_EXPERT,
    )
    extracted_host = ttnn.to_torch(
        extracted,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        dtype=torch.bfloat16,
    ).reshape(NUM_DISPATCH_GROUPS, MAX_TOKENS_PER_EXPERT, HIDDEN)

    packed_rows = []
    fused_rows = []
    selected_route_rows: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for device_idx in range(NUM_DISPATCH_GROUPS):
        global_expert = representative_local_expert + device_idx * LOCAL_EXPERTS
        all_routes = torch.nonzero(indices_host[0] == global_expert, as_tuple=False)
        in_first_slice = all_routes[:, 0] < packed_down_host.shape[-2]
        ordinal = torch.arange(all_routes.shape[0], dtype=torch.long)[in_first_slice]
        first_routes = all_routes[in_first_slice]
        tokens = first_routes[:, 0].long()
        slots = first_routes[:, 1].long()
        assert ordinal.numel() > 0
        packed_rows.append(packed_down_host[0, device_idx, tokens])
        fused_rows.append(extracted_host[device_idx, ordinal])
        selected_route_rows.append((device_idx, tokens, slots, ordinal))

    packed_rows_host = torch.cat(packed_rows, dim=0)
    fused_rows_host = torch.cat(fused_rows, dim=0)
    fused_ff2_pcc, _ = _streaming_pcc(packed_rows_host, fused_rows_host)

    combined_shape = list(combined_slots.shape)
    combined_starts = [0] * len(combined_shape)
    combined_ends = combined_shape.copy()
    combined_ends[-3] = packed_down_host.shape[-2]
    combined_first = ttnn.slice(combined_slots, combined_starts, combined_ends)
    combined_host = ttnn.to_torch(
        combined_first,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        dtype=torch.bfloat16,
    )

    expected_raw = []
    actual_raw = []
    for device_idx, tokens, slots, ordinal in selected_route_rows:
        expected_raw.append(extracted_host[device_idx, ordinal].float())
        actual_raw.append(combined_host[device_idx, 0, tokens, slots].float())
    raw_combine_pcc, _ = _streaming_pcc(torch.cat(expected_raw), torch.cat(actual_raw))

    routed_shape = list(routed_local.shape)
    routed_starts = [0] * len(routed_shape)
    routed_ends = routed_shape.copy()
    routed_ends[-2] = packed_down_host.shape[-2]
    routed_first = ttnn.slice(routed_local, routed_starts, routed_ends)
    routed_host = ttnn.to_torch(
        routed_first,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        dtype=torch.bfloat16,
    )
    first_slice_weights = (
        weights_host[0, : packed_down_host.shape[-2]].float().reshape(1, 1, packed_down_host.shape[-2], TOP_K, 1)
    )
    host_weighted_sum = (combined_host.float() * first_slice_weights).sum(dim=-2)
    weighted_reduce_pcc, _ = _streaming_pcc(host_weighted_sum, routed_host.float())
    unweighted_sum = combined_host.float().sum(dim=-2)
    unweighted_vs_weighted_pcc, _ = _streaming_pcc(host_weighted_sum, unweighted_sum)

    metrics = {
        "fused_ff2_vs_packed_pcc": fused_ff2_pcc,
        "raw_combine_permutation_pcc": raw_combine_pcc,
        "weighted_post_combine_reduce_pcc": weighted_reduce_pcc,
        "unweighted_vs_weighted_pcc": unweighted_vs_weighted_pcc,
    }
    print("LAGUNA_TOKEN_DISPATCH_POST_FFN_STAGES", json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


def _slice_semantics_diagnostics(decoder: MultichipDecoder, x, mesh) -> dict[str, float]:
    """Measure whole-M vs established 256-row router/shared-expert numerics."""
    full_weights, full_indices = _actual_router(decoder, x)
    sliced_weights, sliced_indices = _actual_router_sliced(decoder, x)
    full_indices_host = _compose_replicated0(full_indices, mesh).reshape(1, TOKENS, TOP_K).int()
    sliced_indices_host = _compose_replicated0(sliced_indices, mesh).reshape(1, TOKENS, TOP_K).int()
    full_weights_host = _compose_replicated0(full_weights, mesh).reshape(1, TOKENS, TOP_K)
    sliced_weights_host = _compose_replicated0(sliced_weights, mesh).reshape(1, TOKENS, TOP_K)

    shared_full = _shared_partial(decoder, x, sliced=False)
    shared_sliced = _shared_partial(decoder, x, sliced=True)
    shared_full_host = ttnn.to_torch(
        shared_full,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        dtype=torch.bfloat16,
    )
    shared_sliced_host = ttnn.to_torch(
        shared_sliced,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
        dtype=torch.bfloat16,
    )
    shared_pcc, _ = _streaming_pcc(shared_sliced_host, shared_full_host)
    router_weight_pcc, _ = _streaming_pcc(sliced_weights_host, full_weights_host)
    metrics = {
        "router_slot_exact_fraction": float((full_indices_host == sliced_indices_host).float().mean()),
        "router_token_topk_set_exact_fraction": float(
            (torch.sort(full_indices_host, dim=-1).values == torch.sort(sliced_indices_host, dim=-1).values)
            .all(dim=-1)
            .float()
            .mean()
        ),
        "router_weight_pcc": router_weight_pcc,
        "shared_partial_pcc": shared_pcc,
    }
    for tensor in (full_weights, full_indices, sliced_weights, sliced_indices, shared_full, shared_sliced):
        ttnn.deallocate(tensor)
    print("LAGUNA_TOKEN_DISPATCH_SLICE_SEMANTICS", json.dumps(metrics, sort_keys=True), flush=True)
    return metrics


@pytest.mark.skipif(
    not RUN_DEVICE_PROBE,
    reason="set TT_RUN_LAGUNA_TOKEN_DISPATCH_PROBE=1 to run the bounded chips-2,3 probe",
)
@pytest.mark.timeout(3600)
def test_laguna_d2_whole_8192_token_dispatch_probe():
    """Qualify exact whole-chunk dispatch against the current slice-loop MoE."""
    _safe_probe_environment()
    profile = resolve_profile("p150x2", trace_region_size=200_000_000)
    mesh = open_mesh(ttnn, profile)
    try:
        hf_config = R.build_config()
        raw = W.load_layer_tensors(LAYER)
        decoder = MultichipDecoder.from_state_dict(
            raw,
            hf_config=hf_config,
            layer_idx=LAYER,
            mesh_device=mesh,
            max_seq_len=TOKENS,
        )
        assert decoder.D == 2
        assert decoder.local_experts == LOCAL_EXPERTS
        assert decoder.PACK_GATE_UP
        assert not decoder.MOE_PREFILL_TILE_SPARSE
        assert decoder.MOE_PREFILL_CHUNK == 256
        prototype_dtype = _prototype_dtype()
        prototype_compute_config = _prototype_compute_config(decoder)

        global_expert_idx = ttnn.from_torch(
            ExpertMapping.create_global_expert_idx_table(
                experts_per_chip=LOCAL_EXPERTS,
                dispatch_group_size=DISPATCH_GROUP_SIZE,
                num_dispatch_groups=NUM_DISPATCH_GROUPS,
            ),
            mesh_mapper=get_ep_mesh_mapper(mesh),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            dtype=ttnn.uint32,
        )
        global_expert_idx = ttnn.squeeze(ttnn.squeeze(global_expert_idx, 0), 0)

        # This is the only duplicated weight representation.  The current path
        # continues to use decoder.w["exp_gate_up"] / decoder.w["exp_down"].
        routed_expert = TtRoutedExpert(
            mesh_device=mesh,
            experts_per_chip=LOCAL_EXPERTS,
            global_expert_idx_table=global_expert_idx,
            emb_dim=HIDDEN,
            hidden_dim=INTERMEDIATE,
            max_tokens=MAX_TOKENS_PER_EXPERT,
            torch_weights=_build_test_local_expert_weights(raw),
            activations_dtype=prototype_dtype,
            weights_dtype=ttnn.bfloat4_b,
            compute_kernel_config=decoder._ck_moe,
        )
        del raw
        gc.collect()

        dispatch_table = _dispatch_table()
        routing_setup = TtMoERoutingSetup(
            mesh_device=mesh,
            expert_dispatch_table=dispatch_table,
            num_links=1,
            experts_per_chip=LOCAL_EXPERTS,
        )
        dispatch = TtDispatchModule(
            mesh_device=mesh,
            dispatch_group_size=DISPATCH_GROUP_SIZE,
            experts_per_chip=LOCAL_EXPERTS,
            num_routed_experts=GLOBAL_EXPERTS,
            num_experts_per_tok=TOP_K,
            metadata_len=METADATA_LEN,
            max_dispatch_buffer_token_size=MAX_DISPATCH_ROWS,
            seq_len_per_chip=TOKENS,
            emb_dim=HIDDEN,
            cluster_axis=0,
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        dispatch_table_tt = TtDispatchModule.shard_expert_dispatch_table(mesh, dispatch_table, dispatch_axis=0)
        combine = TtCombineModule(
            mesh_device=mesh,
            dispatch_group_size=DISPATCH_GROUP_SIZE,
            num_dispatch_groups=NUM_DISPATCH_GROUPS,
            experts_per_chip=LOCAL_EXPERTS,
            num_experts_per_tok=TOP_K,
            seq_len_per_chip=TOKENS,
            cluster_axis=0,
            num_links=1,
            topology=ttnn.Topology.Linear,
            init_zeros=True,
        )

        torch.manual_seed(20260822)
        x_host = (torch.randn((1, TOKENS, HIDDEN), dtype=torch.bfloat16) * 0.25).contiguous()
        x = ttnn.from_torch(
            x_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=get_dispatch_input_mesh_mapper(mesh, sp_axis=0),
        )
        _print_memory(mesh, "weights_and_input")

        def run_prototype(*, capture_stages: bool = False, memory_stages: bool = False):
            if PRESERVE_SLICE_SEMANTICS:
                weights, indices = _actual_router_sliced(decoder, x)
            else:
                weights, indices = _actual_router(decoder, x)
            offsets, counts, region_offsets, histogram = routing_setup(
                ttnn_top_k_experts_indices=indices,
                num_routed_experts=GLOBAL_EXPERTS,
                seq_len_per_chip=TOKENS,
                num_experts_per_tok=TOP_K,
            )
            if memory_stages:
                _print_memory(mesh, "prototype_after_routing")
            dispatched, metadata = dispatch(x, weights, indices, offsets, dispatch_table_tt)
            if memory_stages:
                _print_memory(mesh, "prototype_after_dispatch_bf16")
            dispatched_tiled = ttnn.to_layout(
                ttnn.squeeze(ttnn.squeeze(dispatched, 0), 0),
                ttnn.TILE_LAYOUT,
                dtype=prototype_dtype,
            )
            if not capture_stages:
                ttnn.deallocate(dispatched)
            if memory_stages:
                _print_memory(mesh, f"prototype_after_dispatch_{ACTIVATION_DTYPE_NAME}")
            expert_outputs = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
                dispatched_tiled,
                region_offsets,
                counts,
                global_expert_idx,
                routed_expert.gate_projs,
                routed_expert.up_projs,
                routed_expert.down_projs,
                MAX_TOKENS_PER_EXPERT,
                compute_kernel_config=prototype_compute_config,
                chunk_m_tiles_override=CHUNK_M_TILES,
            )
            if memory_stages:
                _print_memory(mesh, "prototype_after_experts")
            expert_outputs_4d = ttnn.unsqueeze(ttnn.unsqueeze(expert_outputs, 0), 0)
            combined_slots = combine(expert_outputs_4d, metadata, counts, region_offsets)
            routed_local = ttnn.reshape(
                _weighted_route_reduce(combined_slots, weights, indices, dispatch_table_tt),
                (1, 1, TOKENS, HIDDEN),
            )
            shared_partial = _shared_partial(decoder, x, sliced=PRESERVE_SLICE_SEMANTICS)
            local_output = ttnn.add(routed_local, ttnn.reshape(shared_partial, (1, 1, TOKENS, HIDDEN)))
            output = decoder._reduce(local_output)
            if memory_stages:
                _print_memory(mesh, "prototype_output_live")
            captured = None
            if capture_stages:
                captured = {
                    "dispatched": dispatched,
                    "offsets": offsets,
                    "counts": counts,
                    "region_offsets": region_offsets,
                    "indices": indices,
                    "weights": weights,
                    "expert_outputs": expert_outputs,
                    "combined_slots": combined_slots,
                    "routed_local": routed_local,
                }
            return output, captured

        # Current production reference: 32 calls of the exact packed 256-row
        # path, including learned routing, shared expert, and one all-reduce per slice.
        baseline_cold, baseline_cold_s = _timed(mesh, lambda: decoder._mlp(x, TOKENS, False))
        ttnn.deallocate(baseline_cold)
        baseline_warm, baseline_warm_s = _timed(mesh, lambda: decoder._mlp(x, TOKENS, False))
        baseline_host = _compose_replicated0(baseline_warm, mesh)
        ttnn.deallocate(baseline_warm)

        (prototype_cold, captured), prototype_cold_s = _timed(mesh, lambda: run_prototype(capture_stages=True))
        dispatched_cold = captured["dispatched"]
        offsets_cold = captured["offsets"]
        counts_cold = captured["counts"]
        indices_cold = captured["indices"]
        indices_host = _compose_replicated0(indices_cold, mesh).reshape(1, TOKENS, TOP_K).int()
        weights_host = _compose_replicated0(captured["weights"], mesh).reshape(1, TOKENS, TOP_K)
        _, padded_rows = _validate_selected_rows_exact(
            dispatched_cold,
            offsets_cold,
            counts_cold,
            indices_host,
            x_host,
            mesh,
        )

        ffn_stage_metrics, packed_down_host, representative_local_expert = _packed_vs_separate_stage_diagnostics(
            decoder,
            routed_expert,
            x,
            mesh,
        )
        post_ffn_stage_metrics = _prototype_stage_diagnostics(
            expert_outputs=captured["expert_outputs"],
            region_offsets=captured["region_offsets"],
            counts=counts_cold,
            global_expert_idx=global_expert_idx,
            combined_slots=captured["combined_slots"],
            routed_local=captured["routed_local"],
            weights_host=weights_host,
            indices_host=indices_host,
            packed_down_host=packed_down_host,
            representative_local_expert=representative_local_expert,
            mesh=mesh,
        )
        ttnn.deallocate(dispatched_cold)
        ttnn.deallocate(prototype_cold)
        for captured_name in ("expert_outputs", "combined_slots", "routed_local"):
            ttnn.deallocate(captured[captured_name])
        del captured
        gc.collect()

        (prototype_warm, _), prototype_warm_s = _timed(mesh, run_prototype)
        prototype_host = _compose_replicated0(prototype_warm, mesh)
        pcc, max_abs = _streaming_pcc(baseline_host, prototype_host)
        ttnn.deallocate(prototype_warm)
        slice_semantics_metrics = _slice_semantics_diagnostics(decoder, x, mesh)

        # A separate untimed warm pass captures synchronized residency stages;
        # these synchronization points are intentionally excluded from latency.
        memory_output, _ = run_prototype(memory_stages=True)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(memory_output)
        _print_memory(mesh, "prototype_after_output_free")

        counts_per_expert = torch.bincount(indices_host.flatten(), minlength=GLOBAL_EXPERTS)
        max_count = int(counts_per_expert.max())
        compact_kernel_rows = sum(
            math.ceil(int(count) / (CHUNK_M_TILES * TILE)) * (CHUNK_M_TILES * TILE) for count in counts_per_expert
        )
        slice_loop_rows_per_card = []
        for group in range(NUM_DISPATCH_GROUPS):
            expert_start = group * LOCAL_EXPERTS
            expert_end = expert_start + LOCAL_EXPERTS
            rows = 0
            for start in range(0, TOKENS, decoder.MOE_PREFILL_CHUNK):
                chunk = indices_host[0, start : start + decoder.MOE_PREFILL_CHUNK]
                active = torch.unique(chunk[(chunk >= expert_start) & (chunk < expert_end)]).numel()
                rows += active * decoder.MOE_PREFILL_CHUNK
            slice_loop_rows_per_card.append(rows)
        speedup = baseline_warm_s / prototype_warm_s
        result = {
            "tokens": TOKENS,
            "top_k": TOP_K,
            "local_experts": LOCAL_EXPERTS,
            "activation_dtype": ACTIVATION_DTYPE_NAME,
            "fidelity": FIDELITY_NAME,
            "chunk_m_tiles": CHUNK_M_TILES,
            "preserve_slice_semantics": PRESERVE_SLICE_SEMANTICS,
            "max_count_per_expert": max_count,
            "dispatch_padded_rows_total": padded_rows,
            "compact_kernel_rows_total": compact_kernel_rows,
            "slice_loop_rows_per_card": slice_loop_rows_per_card,
            "baseline_cold_ms": baseline_cold_s * 1000,
            "baseline_warm_ms": baseline_warm_s * 1000,
            "prototype_cold_ms": prototype_cold_s * 1000,
            "prototype_warm_ms": prototype_warm_s * 1000,
            "warm_speedup": speedup,
            "output_pcc": pcc,
            "output_max_abs": max_abs,
            "ffn_stage_metrics": ffn_stage_metrics,
            "post_ffn_stage_metrics": post_ffn_stage_metrics,
            "slice_semantics_metrics": slice_semantics_metrics,
        }
        print("LAGUNA_TOKEN_DISPATCH_RESULT", json.dumps(result, sort_keys=True), flush=True)

        assert pcc >= 0.995, f"whole-dispatch output PCC {pcc:.6f} is below the production gate 0.995"
        assert speedup > 1.0, f"whole-dispatch warm path regressed: speedup={speedup:.3f}x"
    finally:
        close_mesh(ttnn, mesh)


@pytest.mark.skipif(
    not RUN_STACKED_PRODUCTION_PROBE,
    reason="set TT_RUN_LAGUNA_STACKED_TOKEN_DISPATCH_PROBE=1 to run the production stacked-weight probe",
)
@pytest.mark.timeout(3600)
def test_laguna_d2_production_stacked_8192_token_dispatch_probe():
    """Qualify the default-off production path without duplicate expert weights."""
    _safe_probe_environment()
    if os.environ.get("TT_LAGUNA_MOE_TOKEN_DISPATCH") != "1":
        pytest.fail("the stacked production probe requires TT_LAGUNA_MOE_TOKEN_DISPATCH=1")

    profile = resolve_profile("p150x2", trace_region_size=200_000_000)
    mesh = open_mesh(ttnn, profile)
    try:
        hf_config = R.build_config()
        raw = W.load_layer_tensors(LAYER)
        decoder = MultichipDecoder.from_state_dict(
            raw,
            hf_config=hf_config,
            layer_idx=LAYER,
            mesh_device=mesh,
            max_seq_len=TOKENS,
        )
        del raw
        gc.collect()

        eligible, reason = decoder._token_dispatch_guard(TOKENS, False)
        assert eligible, reason
        assert decoder._token_dispatch_state is not None
        assert "exp_gate_up" in decoder.w and "exp_down" in decoder.w
        assert "exp_gate" not in decoder.w and "exp_up" not in decoder.w

        torch.manual_seed(20260822)
        x_host = (torch.randn((1, TOKENS, HIDDEN), dtype=torch.bfloat16) * 0.25).contiguous()
        x = ttnn.from_torch(
            x_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=get_dispatch_input_mesh_mapper(mesh, sp_axis=0),
        )
        resident = _print_memory(mesh, "stacked_weights_and_input")

        baseline_cold, baseline_cold_s = _timed(mesh, lambda: _established_slice_loop_mlp(decoder, x))
        ttnn.deallocate(baseline_cold)
        baseline_warm, baseline_warm_s = _timed(mesh, lambda: _established_slice_loop_mlp(decoder, x))
        baseline_host = _compose_replicated0(baseline_warm, mesh)
        ttnn.deallocate(baseline_warm)

        candidate_cold, candidate_cold_s = _timed(mesh, lambda: decoder._mlp(x, TOKENS, False))
        ttnn.deallocate(candidate_cold)
        cache_entries_after_cold = int(mesh.num_program_cache_entries())
        candidate_warm, candidate_warm_s = _timed(mesh, lambda: decoder._mlp(x, TOKENS, False))
        cache_entries_after_warm = int(mesh.num_program_cache_entries())
        candidate_host = _compose_replicated0(candidate_warm, mesh)
        pcc, max_abs = _streaming_pcc(baseline_host, candidate_host)
        ttnn.deallocate(candidate_warm)

        # An untimed pass records residency immediately before every explicit
        # production deallocation, capturing the route/dispatch/expert/combine
        # high-water stages without adding synchronization to latency timing.
        memory_stages = []
        original_deallocate = ttnn.deallocate

        def tracking_deallocate(tensor, *args, **kwargs):
            memory_stages.append(memory_snapshot(ttnn, mesh, f"before_free_{len(memory_stages)}"))
            return original_deallocate(tensor, *args, **kwargs)

        ttnn.deallocate = tracking_deallocate
        try:
            memory_output = decoder._mlp(x, TOKENS, False)
        finally:
            ttnn.deallocate = original_deallocate
        memory_stages.append(_print_memory(mesh, "stacked_output_live"))
        cache_entries_after_memory_pass = int(mesh.num_program_cache_entries())
        ttnn.deallocate(memory_output)
        after_free = _print_memory(mesh, "stacked_after_output_free")

        peak_allocated = max(int(snapshot["allocated_bytes"]) for snapshot in memory_stages)
        speedup = baseline_warm_s / candidate_warm_s
        result = {
            "tokens": TOKENS,
            "baseline_cold_ms": baseline_cold_s * 1000,
            "baseline_warm_ms": baseline_warm_s * 1000,
            "candidate_cold_ms": candidate_cold_s * 1000,
            "candidate_warm_ms": candidate_warm_s * 1000,
            "warm_speedup": speedup,
            "output_pcc": pcc,
            "output_max_abs": max_abs,
            "resident_allocated_bytes": int(resident["allocated_bytes"]),
            "tracked_peak_allocated_bytes": peak_allocated,
            "after_free_allocated_bytes": int(after_free["allocated_bytes"]),
            "after_free_largest_contiguous_bytes_per_bank": int(after_free["largest_contiguous_bytes_free_per_bank"]),
            "program_cache_entries_after_cold": cache_entries_after_cold,
            "program_cache_entries_after_warm": cache_entries_after_warm,
            "program_cache_entries_after_memory_pass": cache_entries_after_memory_pass,
        }
        print("LAGUNA_STACKED_TOKEN_DISPATCH_RESULT", json.dumps(result, sort_keys=True), flush=True)

        assert pcc >= 0.995, f"production stacked output PCC {pcc:.6f} is below 0.995"
        assert candidate_warm_s * 1000 < 255.936, "production stacked warm latency misses the no-regression gate"
        assert speedup >= 3.0, f"production stacked warm speedup {speedup:.3f}x is below the 3x target"
        assert cache_entries_after_warm == cache_entries_after_cold, "warm pass compiled new programs"
        assert cache_entries_after_memory_pass == cache_entries_after_cold, "repeated pass compiled new programs"
        assert peak_allocated <= 1_540_210_688, f"tracked DRAM peak regressed to {peak_allocated} bytes"
        assert (
            int(after_free["largest_contiguous_bytes_free_per_bank"]) >= 3_999_000_000
        ), "post-run DRAM contiguity regressed"
    finally:
        close_mesh(ttnn, mesh)
