# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for pool admission, tensor accounting and plugin reservations."""

import json
from pathlib import Path

import pytest

from models.autoports.qwen_qwen3_8_flash_next.tt.cache_budget import (
    BYTES_PER_FULL_STACK_BLOCK,
    HISTORICAL_INTERLEAVED_HEAD_STATIC_BYTES,
    KNOWN_STATIC_BYTES,
    LM_HEAD_DRAM_BANK_PADDING_DELTA_BYTES,
    PAGE_TOKENS,
    PREFILL_CHECKPOINT_BYTES_PER_SLOT,
    PREFILL_OUTPUT_BYTES,
    factored_workspace_bytes,
    plan_shared_kv_cache,
    runtime_forecast,
)


def test_constants_match_independent_tensor_ledger():
    report = json.loads((Path(__file__).parents[1] / "doc/full_model/capacity_plan.json").read_text())
    assert KNOWN_STATIC_BYTES == report["static_summary"]["known_static_total_bytes"]
    for profile in report["profiles"]:
        budget = plan_shared_kv_cache(profile["capacity"], 1)
        kv = next(r for r in profile["state_ledger"] if r["name"] == "Local K and V")
        assert (
            budget.private_non_kv_bytes_per_cache == profile["per_request_bytes"] - kv["eight_bank_reservation_bytes"]
        )
        assert budget.shared_index_bytes == profile["shared_index_bytes"]


def test_selected_head_bank_padding_is_charged_once_without_duplicate_weight():
    report = json.loads((Path(__file__).parents[1] / "doc/full_model/capacity_plan.json").read_text())
    head = next(row for row in report["terminal_ledger"] if row["name"] == "Vocab-sharded untied LM head")
    logical_k_tiles, logical_n_tiles, banks, bf16_tile_bytes = 2560 // 32, 62080 // 32, 8, 2048
    interleaved = ((logical_k_tiles * logical_n_tiles + banks - 1) // banks) * banks * bf16_tile_bytes
    bank_sharded = logical_k_tiles * ((logical_n_tiles + banks - 1) // banks) * banks * bf16_tile_bytes
    assert head["eight_bank_reservation_bytes"] == bank_sharded == 318504960
    assert head["historical_interleaved_reservation_bytes"] == interleaved == 317849600
    assert bank_sharded - interleaved == LM_HEAD_DRAM_BANK_PADDING_DELTA_BYTES == 655360
    assert KNOWN_STATIC_BYTES - HISTORICAL_INTERLEAVED_HEAD_STATIC_BYTES == bank_sharded - interleaved
    assert not report["terminal_layout_revision"]["duplicate_native_head_allocated"]


def test_compact_checkpoint_padding_and_per_request_ownership():
    report = json.loads((Path(__file__).parents[1] / "doc/full_model/capacity_plan.json").read_text())

    # Separate DRAM buffers round independently to all eight banks.
    def banked_tiles(rows, columns, tile_bytes):
        tiles = ((rows + 31) // 32) * ((columns + 31) // 32)
        return ((tiles + 7) // 8) * 8 * tile_bytes

    conv = banked_tiles(3, 2560, 2048)
    recurrence = 12 * banked_tiles(128, 128, 4096)
    ple = banked_tiles(9, 2560, 2048) + 8 * 64
    qsa = banked_tiles(3, 128, 2048) + 8 * 64
    assert qsa == 16896  # 4 useful BF16 tiles occupy8 banks, plus scalar position.
    assert 36 * (conv + recurrence) + ple + 12 * qsa == PREFILL_CHECKPOINT_BYTES_PER_SLOT == 34576896
    assert 2 * (2 * (conv + recurrence) + ple + qsa) == 4163584
    for context in (128, 262144):
        for batch in (1, 8, 32):
            budget = plan_shared_kv_cache(context, batch)
            assert budget.prefill_checkpoint_bytes == batch * PREFILL_CHECKPOINT_BYTES_PER_SLOT
            profile = next(p for p in report["profiles"] if p["capacity"] == context)
            assert profile["batches"][str(batch)]["prefill_checkpoint_bytes"] == budget.prefill_checkpoint_bytes
    assert report["prefill_checkpoint_revision"]["migration_extra_scratch_bytes"] == 0


def test_one_independent_prefill_workspace_has_exact_batch_saving():
    report = json.loads((Path(__file__).parents[1] / "doc/full_model/capacity_plan.json").read_text())
    # Physical32 rows,62080 vocabulary columns per rank, BF16 tile pages,
    # and eight-bank reservation. Sampler canvas is separately in static bytes.
    tiles = 62080 // 32
    expected = ((tiles + 7) // 8) * 8 * 2048
    assert expected == PREFILL_OUTPUT_BYTES == 3981312
    assert len(report["generator_shared_workspace_ledger"]) == 1
    assert report["generator_shared_workspace_ledger"][0]["eight_bank_reservation_bytes"] == expected
    assert all("prefill output" not in row["name"] for row in report["generator_per_slot_ledger"])
    for profile in report["profiles"]:
        for batch in (1, 8, 16, 32):
            budget = plan_shared_kv_cache(profile["capacity"], batch)
            assert budget.prefill_output_bytes == expected
            assert profile["batches"][str(batch)]["persistent_prefill_output_bytes"] == expected
            private_copies = batch + int(batch > 1)
            before_output = (
                KNOWN_STATIC_BYTES
                + budget.shared_index_bytes
                + private_copies * budget.private_non_kv_bytes_per_cache
                + batch * PREFILL_CHECKPOINT_BYTES_PER_SLOT
                + factored_workspace_bytes(batch)
            )
            old_per_slot_total = before_output + batch * expected
            assert old_per_slot_total - budget.known_resident_before_pool_bytes == (batch - 1) * expected
    revision = report["prefill_workspace_revision"]
    assert revision["B1_saving_bytes_per_rank"] == 0
    assert revision["B32_saving_bytes_per_rank"] == 123420672


def test_factored_source_inventory_and_measured_capture_projection():
    assert factored_workspace_bytes(1) == 0
    assert factored_workspace_bytes(2) == 1344512
    assert factored_workspace_bytes(32) == 17236992
    assert factored_workspace_bytes(3) == 3 * (3 * 163840 + 16384) + 3 * 163840 + 1024
    root = Path(__file__).parents[1] / "doc/full_model"
    measured = json.loads((root / "factored_decode_buckets_tracked_v1.json").read_text())
    prefill = json.loads((root / "prefill_signatures_full48_v2.json").read_text())
    estimate = runtime_forecast(512, 32)
    assert (
        estimate["attention_trace_bytes"] + estimate["moe_trace_bytes"] + estimate["terminal_trace_bytes"]
        == measured["full48_capture_projection"]["projected_total_bytes"]
        == 2194677760
    )
    assert (
        estimate["prefill_trace_bytes"]
        == sum(row["allocated_delta_bytes_per_rank"] for row in prefill["first_capture_memory"])
        == 765526016
    )
    assert estimate["nontrace_runtime_allowance_bytes"] > estimate["historical_prepared_minus_listed_tensor_bytes"]
    assert estimate["reserve_bytes"] == estimate["component_total_bytes"] > 2**31
    budget = plan_shared_kv_cache(512, 32)
    assert budget.runtime_reserve_bytes == estimate["reserve_bytes"]
    assert budget.factored_workspace_bytes == 17236992
    assert not estimate["validated_capacity_claim"]


def test_runtime_forecast_preserves_b1_and_marks_long_context_extrapolation():
    for context in (128, 512, 32768, 32769, 262144):
        budget = plan_shared_kv_cache(context, 1)
        assert budget.runtime_reserve_bytes == 2**31
        assert budget.factored_workspace_bytes == 0
    short, long = runtime_forecast(32768, 3), runtime_forecast(32769, 3)
    assert long["qsa_indexer_blocks"] == 2
    assert long["prefill_trace_bytes"] == 2 * short["prefill_trace_bytes"]
    assert long["attention_trace_bytes"] - short["attention_trace_bytes"] == 3 * 12 * 1703936
    assert long["long_context_extrapolation"] and not long["validated_capacity_claim"]
    override = plan_shared_kv_cache(512, 32, runtime_reserve_bytes=123)
    assert override.runtime_reserve_bytes == 123
    assert override.runtime_forecast["explicit_reserve_override"]


def test_two_slot_reserve_bounds_measured_full_context_adapter():
    evidence = Path(__file__).parents[1] / "doc/vllm_integration/capacity_full_v1.json"
    report = json.loads(evidence.read_text())
    snapshots = report["snapshots"]
    growth = snapshots["replayed"]["allocated_bytes_per_rank"] - snapshots["pool_allocated"]["allocated_bytes_per_rank"]
    forecast = runtime_forecast(262144, 2)
    assert report["passed"] and report["reduced_layers"] is None
    assert growth == forecast["measured_runtime_growth_bytes"]
    assert forecast["reserve_bytes"] - growth > 2**30
    assert (
        snapshots["replayed"]["allocated_bytes_per_rank"]
        == snapshots["all_prefill_buckets"]["allocated_bytes_per_rank"]
    )
    assert not forecast["long_context_extrapolation"]


def test_head_padding_changes_real_admission_boundary_without_shrinking_context():
    old = plan_shared_kv_cache(128, 1, known_static_bytes=HISTORICAL_INTERLEAVED_HEAD_STATIC_BYTES)
    limit = old.known_resident_before_pool_bytes + old.runtime_reserve_bytes + 6 * BYTES_PER_FULL_STACK_BLOCK
    old_at_limit = plan_shared_kv_cache(
        128, 1, allocatable_bytes=limit, known_static_bytes=HISTORICAL_INTERLEAVED_HEAD_STATIC_BYTES
    )
    assert old_at_limit.plugin_max_tokens_all_users() == 160
    selected_at_limit = plan_shared_kv_cache(128, 1, allocatable_bytes=limit)
    assert old_at_limit.available_pool_bytes - selected_at_limit.available_pool_bytes == 655360
    with pytest.raises(ValueError, match="advertised context is unchanged"):  # allow-pytest.raises: CPU-only
        selected_at_limit.plugin_max_tokens_all_users()
    assert plan_shared_kv_cache(262144, 1).max_model_len == 262144


@pytest.mark.parametrize("context,batch", [(128, 1), (128, 32), (262144, 1), (8192, 32), (65, 3)])
def test_pinned_plugin_output_addition_stays_inside_budget(context, batch):
    budget = plan_shared_kv_cache(context, batch)
    returned = budget.plugin_max_tokens_all_users()
    actual_blocks = (returned + PAGE_TOKENS * batch + PAGE_TOKENS - 1) // PAGE_TOKENS
    assert actual_blocks <= budget.maximum_pool_blocks
    assert actual_blocks * BYTES_PER_FULL_STACK_BLOCK <= budget.available_pool_bytes
    assert returned >= ((context + 31) // 32 + 1) * 32
    assert actual_blocks >= 2 * batch


def test_full_context_does_not_silently_shrink_for_many_private_slots():
    budget = plan_shared_kv_cache(262144, 8)
    assert budget.maximum_pool_blocks * PAGE_TOKENS < 262144
    with pytest.raises(ValueError, match="advertised context is unchanged"):  # allow-pytest.raises: CPU-only
        budget.plugin_max_tokens_all_users()


def test_exact_one_block_capacity_boundary_includes_output_and_null():
    base = plan_shared_kv_cache(128, 1)
    minimum = 4 + 1 + 1  # request + null + pinned output reservation
    limit = base.known_resident_before_pool_bytes + base.runtime_reserve_bytes + minimum * BYTES_PER_FULL_STACK_BLOCK
    exact = plan_shared_kv_cache(128, 1, allocatable_bytes=limit)
    assert exact.plugin_max_tokens_all_users() == 5 * PAGE_TOKENS
    too_small = plan_shared_kv_cache(128, 1, allocatable_bytes=limit - 1)
    with pytest.raises(ValueError, match="cannot admit"):  # allow-pytest.raises: CPU-only
        too_small.plugin_max_tokens_all_users()


@pytest.mark.parametrize("context,batch", [(0, 1), (262145, 1), (128, 0), (128, 33), (True, 1), (128, 1.5)])
def test_invalid_geometry_is_rejected(context, batch):
    with pytest.raises(ValueError):  # allow-pytest.raises: CPU-only
        plan_shared_kv_cache(context, batch)


@pytest.mark.parametrize("dtype,tile_bytes", [("bfloat16", 2048), ("bfloat8_b", 1088)])
def test_precision_pool_exact_admission_boundary(dtype, tile_bytes):
    base = plan_shared_kv_cache(128, 1, kv_cache_dtype=dtype)
    # Six full-stack blocks: four request pages, one null page, one output page.
    block_bytes = 12 * 2 * 8 * tile_bytes
    limit = base.known_resident_before_pool_bytes + base.runtime_reserve_bytes + 6 * block_bytes
    exact = plan_shared_kv_cache(128, 1, allocatable_bytes=limit, kv_cache_dtype=dtype)
    assert exact.maximum_pool_blocks == 6
    assert exact.plugin_max_tokens_all_users() == 160
    too_small = plan_shared_kv_cache(128, 1, allocatable_bytes=limit - 1, kv_cache_dtype=dtype)
    assert too_small.maximum_pool_blocks == 5
    with pytest.raises(ValueError, match="cannot admit"):  # allow-pytest.raises: CPU-only
        too_small.plugin_max_tokens_all_users()


def test_bfp8_capacity_changes_only_pool_page_cost():
    bf16 = plan_shared_kv_cache(262144, 1)
    bfp8 = plan_shared_kv_cache(262144, 1, kv_cache_dtype="bfloat8_b")
    assert bfp8.available_pool_bytes == bf16.available_pool_bytes
    assert bfp8.runtime_reserve_bytes == bf16.runtime_reserve_bytes
    assert bfp8.private_non_kv_bytes_per_cache == bf16.private_non_kv_bytes_per_cache
    assert bfp8.maximum_pool_blocks == bfp8.available_pool_bytes // 208896
    assert bf16.maximum_pool_blocks == bf16.available_pool_bytes // 393216
    assert bfp8.maximum_pool_blocks > bf16.maximum_pool_blocks
    assert bfp8.max_model_len == bf16.max_model_len == 262144


def test_weight_precision_delta_is_charged_before_pool_admission():
    delta = 2 * 128 * 3 * (2560 // 32) * (640 // 32) * (1088 - 576)
    baseline = plan_shared_kv_cache(262144, 1, kv_cache_dtype="bfloat8_b")
    upgraded = plan_shared_kv_cache(
        262144, 1, kv_cache_dtype="bfloat8_b", known_static_bytes=KNOWN_STATIC_BYTES + delta
    )
    assert upgraded.known_resident_before_pool_bytes - baseline.known_resident_before_pool_bytes == 629145600
    assert baseline.available_pool_bytes - upgraded.available_pool_bytes == delta
    assert upgraded.maximum_pool_blocks == upgraded.available_pool_bytes // 208896


@pytest.mark.parametrize("dtype", ["bfloat4_b", "float32", "BF16", None])
def test_unsupported_kv_dtype_has_no_silent_byte_estimate(dtype):
    with pytest.raises(ValueError, match="KV cache dtype"):  # allow-pytest.raises: CPU-only
        plan_shared_kv_cache(128, 1, kv_cache_dtype=dtype)
