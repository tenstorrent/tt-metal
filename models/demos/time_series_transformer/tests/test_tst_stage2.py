import pytest


def test_kv_cache_fused_update_cache_op(device):
    """
    Change 4 (KV cache fused op) — BLOCKED, same treatment as FlashDecode.

    ttnn.update_cache / ttnn.kv_cache.update_cache_for_token_ are confirmed
    identical wrappers (both call ttnn::prim::update_cache with batch_index=0,
    UpdateCacheOpType::UPDATE). Signature:

        ttnn.update_cache(cache, input, update_idx: int, batch_offset: int,
                           compute_kernel_config=None) -> Tensor

    Source: ttnn/cpp/ttnn/operations/kv_cache/device/update_cache_device_operation.cpp
    validate_on_program_cache_miss() hard-asserts:

        TT_FATAL(input_tensor.layout() == Layout::TILE &&
                  cache_tensor.layout() == Layout::TILE,
                  "Inputs to update_cache must be tilized")

    This model's k_cache/v_cache are ROW_MAJOR_LAYOUT, required for slice_write
    (see tst_model_cached_additions.py comments -- four to_layout(TILE_LAYOUT)
    calls in the read path already exist and are non-removable for that reason).

    Adopting ttnn.update_cache would require converting cache TILE<->ROW_MAJOR
    on every decode step (on top of the four existing conversions), against a
    measured ~5.9ms/step execute_trace floor that is the dominant per-step cost
    (see test_single_sequence_latency_2cq). No measurement exists showing the
    fusion benefit exceeds the added conversion cost, and none is being taken
    on speculatively.

    Decision: do not pursue for Stage 2. Revisit in Stage 3 alongside the
    fused mega-trace / Flash Attention work already identified there, since
    that work may change the cache layout picture entirely (e.g. moving to a
    permanently-TILE cache), rather than solving it in isolation here.

    Falcon7b reference confirms this is not solvable by call-site swap alone:
    falcon7b's layer_past caches are TILE_LAYOUT from allocation -- they never
    carry TST's ROW_MAJOR/slice_write constraint.
    """
    pytest.xfail(
        "BLOCKED: ttnn.update_cache requires TILE_LAYOUT cache+input "
        "(TT_FATAL in update_cache_device_operation.cpp:32), conflicting "
        "with this model's ROW_MAJOR k_cache/v_cache required for slice_write. "
        "No measured evidence the TILE<->ROW_MAJOR conversion tax nets positive "
        "against the ~5.9ms/step execute_trace floor. Deferred to Stage 3 "
        "pending mega-trace/layout rework, not pursued in isolation."
    )
