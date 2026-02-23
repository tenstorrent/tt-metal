# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Targeted regression tests for bugs found in PR #38341 chain construction changes.

Run with debug logging to see chain construction decisions:
    TT_METAL_LOGGER_LEVEL=Debug pytest tests/ttnn/unit_tests/operations/sdpa/test_sdpa_chain_bugs.py -v -s

Bug 1 (Critical): Mcast gap detection only checks current head's segments,
    missing worker cores from OTHER heads in the mcast rectangle.
    On harvested devices, non-contiguous physical X can put other-head cores
    inside a chain's mcast rectangle, corrupting K/V CBs and semaphores.
Bug 2 (Medium):   best_dist=0 initialization makes the first chain always
    pick the last position as injector instead of the natural wrap-order start.
Bug 3 (Medium):   Excluding the last segment from injector candidacy prevents
    building valid chains when only the last segment is a valid start.
Bug 4 (Low):      Defensive TT_FATAL after uniform_q_mcast check is unreachable.
"""

import math
import torch
import pytest
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fa_rand(*shape):
    """Flash-attention-style random tensor with occasional outliers."""
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def run_sdpa_noncausal(
    device,
    b,
    nh,
    nkv,
    sq,
    sk,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    grid_size=None,
    use_mask=True,
    min_pcc=0.994,
):
    """Run non-causal SDPA and return (pass, pcc_value, rmse_value)."""
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size or device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, sq, d)
    K = fa_rand(b, nkv, sk, d)
    V = fa_rand(b, nkv, sk, d)

    tt_mask = None
    mask = None
    if use_mask:
        mask = torch.bernoulli(torch.full((1, 1, sq, sk), 0.25))
        mask = mask * -1e9
        tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=False,
        attn_mask=tt_mask,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_back = ttnn.to_torch(tt_back)
    tt_back = tt_back[:, :, :sq, :]

    if nkv > 1 and nkv != nh:
        assert nh % nkv == 0
        K_ref = K.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)
        V_ref = V.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)
    else:
        K_ref, V_ref = K, V

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_ref, V_ref, is_causal=False, attn_mask=mask)

    out_pass, out_pcc = comp_pcc(gt, tt_back, min_pcc)
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.info(f"PCC: {out_pcc}  |  RMSE: {rmse}")

    return out_pass, out_pcc, rmse


def compute_work_distribution(b, nh, sq, d, q_chunk_size, grid_x, grid_y):
    """
    Replicate the C++ parallelization logic to predict core->head assignments.
    Returns dict with cores, head_segments, and parallelization factors.
    """
    num_cores = grid_x * grid_y
    q_num_chunks = math.ceil(sq / q_chunk_size)

    batch_parallel_factor = min(b, num_cores)
    nh_parallel_factor = min(num_cores // batch_parallel_factor, nh)
    q_parallel_factor = min(num_cores // (batch_parallel_factor * nh_parallel_factor), q_num_chunks)

    batch_per_core = math.ceil(b / batch_parallel_factor)
    nh_per_core = math.ceil(nh / nh_parallel_factor)
    q_per_core = math.ceil(q_num_chunks / q_parallel_factor)

    cores = []
    head_segments = {}

    for i in range(num_cores):
        logical_x = i % grid_x
        logical_y = i // grid_x

        local_batch_start = (i // (nh_parallel_factor * q_parallel_factor)) * batch_per_core
        local_batch_end = min(local_batch_start + batch_per_core, b)
        local_nh_start = ((i // q_parallel_factor) % nh_parallel_factor) * nh_per_core
        local_nh_end = min(local_nh_start + nh_per_core, nh)
        local_q_start = (i % q_parallel_factor) * q_per_core
        local_q_end = min(local_q_start + q_per_core, q_num_chunks)
        q_count = local_q_end - local_q_start

        head_work = []
        for batch in range(local_batch_start, local_batch_end):
            for h in range(local_nh_start, local_nh_end):
                if q_count > 0:
                    head_id = batch * nh + h
                    head_work.append({"head_id": head_id, "batch": batch, "head": h, "q_count": q_count})
                    if head_id not in head_segments:
                        head_segments[head_id] = []
                    head_segments[head_id].append({"core_idx": i, "q_count": q_count, "logical": (logical_x, logical_y)})

        cores.append({"core_idx": i, "logical": (logical_x, logical_y), "head_work": head_work, "num_heads": len(head_work)})

    return {
        "cores": cores,
        "head_segments": head_segments,
        "batch_parallel_factor": batch_parallel_factor,
        "nh_parallel_factor": nh_parallel_factor,
        "q_parallel_factor": q_parallel_factor,
        "q_per_core": q_per_core,
    }


def print_work_distribution(dist, label=""):
    """Pretty-print heads with multi-core chains for debugging."""
    if label:
        logger.info(f"=== {label} ===")
    logger.info(
        f"Parallelization: batch={dist['batch_parallel_factor']}, "
        f"nh={dist['nh_parallel_factor']}, q={dist['q_parallel_factor']}, "
        f"q_per_core={dist['q_per_core']}"
    )
    for head_id, segs in sorted(dist["head_segments"].items()):
        if len(segs) >= 2:
            cores_str = ", ".join(f"core{s['core_idx']}(log={s['logical']}, q={s['q_count']})" for s in segs)
            logger.info(f"  Head {head_id} ({len(segs)} segments): {cores_str}")


def skip_if_grid_too_small(device, grid_x, grid_y):
    compute_grid = device.compute_with_storage_grid_size()
    if grid_x > compute_grid.x or grid_y > compute_grid.y:
        pytest.skip(f"Need ({grid_x},{grid_y}) grid but device has ({compute_grid.x},{compute_grid.y})")


# ===========================================================================
# Bug 1 (Critical): Mcast gap detection misses worker cores from other heads
#
# The gap check iterates only over segments of the current head, but the
# mcast rectangle [min_x, max_x] on a physical row covers ALL cores in
# that range. Cores from other heads in that range get corrupted by mcast.
#
# To trigger: put multiple heads' chains on the same physical row.
# On harvested devices, non-contiguous phys_x mapping may place other-head
# cores inside a chain's rect. Even on non-harvested devices, the test
# exercises the multi-chain-per-row mcast path for correctness.
# ===========================================================================


class TestMcastGapDetection:
    """Bug 1: Mcast gap detection misses cross-head worker cores."""

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
    @pytest.mark.parametrize(
        "b, nh, sq, d, q_chunk, k_chunk, grid_x, grid_y",
        [
            # 2 heads, each with 4-core chain, all on same row
            (1, 2, 512, 128, 128, 128, 8, 1),
            # 4 heads, each with 2-core chain, all on same row
            (1, 4, 256, 128, 128, 128, 8, 1),
            # 2 batches × 1 head → 2 head_ids, each with 4-core chain on same row
            (2, 1, 512, 128, 128, 128, 8, 1),
            # 2 batches × 2 heads → 4 head_ids, 10x1 single row
            (2, 2, 256, 128, 128, 128, 10, 1),
            # 3 heads on 6x1 → 3 adjacent 2-core chains
            (1, 3, 256, 128, 128, 128, 6, 1),
            # 8 heads on full width → many short chains packed tightly
            (1, 8, 512, 128, 128, 128, 10, 1),
        ],
        ids=[
            "2heads_8x1_4core_chains",
            "4heads_8x1_2core_chains",
            "2batch_8x1_4core_chains",
            "2b2h_10x1_dense",
            "3heads_6x1_adjacent_rects",
            "8heads_10x1_mixed",
        ],
    )
    def test_mcast_multi_head_same_row(self, device, b, nh, sq, d, q_chunk, k_chunk, grid_x, grid_y, dtype):
        """
        Multiple heads' chains packed onto a single physical row.
        On harvested devices, physical X gaps can put cross-head cores inside
        a chain's mcast rectangle → PCC failure reveals corruption.
        """
        skip_if_grid_too_small(device, grid_x, grid_y)

        dist = compute_work_distribution(b, nh, sq, d, q_chunk, grid_x, grid_y)
        print_work_distribution(dist, f"b={b} nh={nh} sq={sq} grid={grid_x}x{grid_y}")

        ok, pcc, rmse = run_sdpa_noncausal(
            device, b, nh, 1, sq, sq, d, q_chunk, k_chunk, dtype,
            grid_size=(grid_x, grid_y), use_mask=True,
        )
        assert ok, f"PCC failed: {pcc} (possible mcast corruption from cross-head gap)"
        assert rmse < 0.015, f"RMSE {rmse} too high"

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
    def test_mcast_gap_full_grid_stress(self, device, dtype):
        """
        Stress test on full device grid. Many heads → many chains per row.
        On harvested devices, maximizes probability of cross-head overlap.
        """
        compute_grid = device.compute_with_storage_grid_size()
        grid_x, grid_y = compute_grid.x, compute_grid.y
        num_cores = grid_x * grid_y

        nh = max(2, num_cores // 3)
        sq, d = 256, 128

        dist = compute_work_distribution(1, nh, sq, d, 128, grid_x, grid_y)
        print_work_distribution(dist, f"Full grid stress: nh={nh} grid={grid_x}x{grid_y}")

        ok, pcc, rmse = run_sdpa_noncausal(
            device, 1, nh, 1, sq, sq, d, 128, 128, dtype,
            grid_size=(grid_x, grid_y), use_mask=False,
        )
        assert ok, f"PCC failed: {pcc}"
        assert rmse < 0.015, f"RMSE {rmse} too high"


# ===========================================================================
# Bug 2 (Medium): best_dist=0 init picks wrong injector for first chain
#
# When injector_phys_x is empty, min_dist=UINT32_MAX for all candidates.
# Since UINT32_MAX > 0 (best_dist init), best_pos updates on every iteration
# and ends up as the LAST position. The swap moves the last element to front.
#
# Correctness is maintained but the injector choice is suboptimal.
# Run with TT_METAL_LOGGER_LEVEL=Debug to see which core is picked:
#   "Building chain for head 0 ... injector phys_x=<X>"
# ===========================================================================


class TestBestDistInit:
    """Bug 2: First chain always picks last position as injector."""

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
    @pytest.mark.parametrize(
        "b, nh, sq, d, q_chunk, k_chunk, grid_x, grid_y",
        [
            # Single chain configs → first (and only) chain triggers best_dist=0
            (1, 1, 512, 128, 128, 128, 4, 1),
            (1, 1, 384, 128, 128, 128, 3, 1),
            (1, 1, 1024, 128, 128, 128, 8, 1),
        ],
        ids=["1head_4x1", "1head_3x1", "1head_8x1"],
    )
    def test_first_chain_injector_selection(self, device, b, nh, sq, d, q_chunk, k_chunk, grid_x, grid_y, dtype):
        """
        Single chain → first chain → best_dist=0 bug picks last segment.
        Check debug log: injector phys_x should be the FIRST core's phys_x
        but will be the LAST core's phys_x due to the bug.
        """
        skip_if_grid_too_small(device, grid_x, grid_y)

        dist = compute_work_distribution(b, nh, sq, d, q_chunk, grid_x, grid_y)
        print_work_distribution(dist, f"First chain injector: grid={grid_x}x{grid_y}")

        ok, pcc, rmse = run_sdpa_noncausal(
            device, b, nh, 1, sq, sq, d, q_chunk, k_chunk, dtype,
            grid_size=(grid_x, grid_y), use_mask=True,
        )
        assert ok, f"PCC failed: {pcc}"


# ===========================================================================
# Bug 3 (Medium): Last segment excluded from injector candidacy
#
# Loops bound with `idx + 1 < segments.size()`, unnecessarily excluding the
# last segment. The wrap-around build handles last-as-start via modular
# arithmetic. The bug is latent: hard to trigger with current parallelization
# because shared cores (needed so all-but-last are already in chains) only
# happen when q_parallel=1, which means no multi-core chains.
# ===========================================================================


class TestLastSegmentExclusion:
    """Bug 3: Last segment excluded from injector candidacy (latent)."""

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
    @pytest.mark.parametrize(
        "b, nh, sq, d, q_chunk, k_chunk, grid_x, grid_y",
        [
            # Minimal 2-segment chains where excluding last = only 1 candidate
            (1, 1, 256, 128, 128, 128, 2, 1),
            (1, 1, 384, 128, 128, 128, 3, 1),
            (1, 4, 256, 128, 128, 128, 8, 1),
        ],
        ids=["2seg_2x1", "3seg_3x1", "4heads_8x1"],
    )
    def test_last_segment_edge(self, device, b, nh, sq, d, q_chunk, k_chunk, grid_x, grid_y, dtype):
        """Exercises small chains where last-segment exclusion most constrains candidates."""
        skip_if_grid_too_small(device, grid_x, grid_y)

        dist = compute_work_distribution(b, nh, sq, d, q_chunk, grid_x, grid_y)
        print_work_distribution(dist, f"Last segment: grid={grid_x}x{grid_y}")

        ok, pcc, rmse = run_sdpa_noncausal(
            device, b, nh, 1, sq, sq, d, q_chunk, k_chunk, dtype,
            grid_size=(grid_x, grid_y), use_mask=True,
        )
        assert ok, f"PCC failed: {pcc}"


# ===========================================================================
# Bug 4 context: Non-uniform q_chunk_count → mcast fallback to unicast
#
# The defensive TT_FATAL is unreachable dead code (can't test it firing).
# Instead, we test the path that SHOULD trigger the mcast-ineligibility check:
# chains with mixed q_chunk_counts due to uneven q_num_chunks / q_parallel.
# Verifies the unicast fallback with descending sort produces correct results.
# ===========================================================================


class TestNonUniformQChunks:
    """Bug 4 context: Mixed q_chunk_counts → unicast chains with descending sort."""

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
    @pytest.mark.parametrize(
        "b, nh, sq, d, q_chunk, k_chunk, grid_x, grid_y",
        [
            # q_chunks=3 on 2 cores → q_per_core=2 → counts: [2, 1]
            (1, 1, 384, 128, 128, 128, 2, 1),
            # q_chunks=5 on 3 cores → q_per_core=2 → counts: [2, 2, 1]
            (1, 1, 640, 128, 128, 128, 3, 1),
            # q_chunks=7 on 4 cores → counts: [2, 2, 2, 1]
            (1, 1, 896, 128, 128, 128, 4, 1),
            # Multiple heads with uneven tails: 2 heads, 4 cores
            (1, 2, 384, 128, 128, 128, 4, 1),
            # 2 heads, 6 cores: q_chunks=5 → q_pf=3 → counts: [2, 2, 1] per head
            (1, 2, 640, 128, 128, 128, 6, 1),
        ],
        ids=["1h_2c", "1h_3c", "1h_4c", "2h_4c", "2h_6c"],
    )
    def test_mixed_q_unicast_fallback(self, device, b, nh, sq, d, q_chunk, k_chunk, grid_x, grid_y, dtype):
        """
        Non-uniform q_chunk_counts should disable mcast, fall back to unicast
        with descending sort. Debug log should show:
        "Head X: mcast ineligible - mixed q_chunk_counts"
        """
        skip_if_grid_too_small(device, grid_x, grid_y)

        dist = compute_work_distribution(b, nh, sq, d, q_chunk, grid_x, grid_y)
        print_work_distribution(dist, f"Mixed q: grid={grid_x}x{grid_y}")

        # Verify test setup: at least one head has mixed q_counts
        has_mixed = False
        for head_id, segs in dist["head_segments"].items():
            if len(segs) >= 2:
                q_counts = set(s["q_count"] for s in segs)
                if len(q_counts) > 1:
                    has_mixed = True
                    logger.info(f"Head {head_id} has mixed q_counts: {[s['q_count'] for s in segs]}")
        assert has_mixed, "Test setup error: no heads with mixed q_chunk_counts"

        ok, pcc, rmse = run_sdpa_noncausal(
            device, b, nh, 1, sq, sq, d, q_chunk, k_chunk, dtype,
            grid_size=(grid_x, grid_y), use_mask=True,
        )
        assert ok, f"PCC failed: {pcc} (mixed q_chunks unicast path)"
        assert rmse < 0.015, f"RMSE {rmse} too high"


# ===========================================================================
# Integration: SDXL UNet shapes that originally showed the regression
# ===========================================================================


class TestSDXLRegressionShapes:
    """Regression tests using the SDXL UNet shapes from the PR description."""

    @pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bfp8"])
    @pytest.mark.parametrize(
        "b, nh, sq, d, q_chunk, k_chunk",
        [
            (1, 1, 4096, 64, 128, 128),
            (1, 1, 4096, 64, 256, 128),
            (1, 1, 1024, 64, 128, 128),
            (1, 1, 1024, 64, 256, 256),
        ],
        ids=["large_q128k128", "large_q256k128", "small_q128k128", "small_q256k256"],
    )
    def test_sdxl_unet_shapes(self, device, b, nh, sq, d, q_chunk, k_chunk, dtype):
        """SDXL UNet SDPA shapes on full grid."""
        compute_grid = device.compute_with_storage_grid_size()
        min_pcc = 0.993 if dtype == ttnn.bfloat8_b else 0.994

        ok, pcc, rmse = run_sdpa_noncausal(
            device, b, nh, 1, sq, sq, d, q_chunk, k_chunk, dtype,
            grid_size=(compute_grid.x, compute_grid.y), use_mask=False,
        )
        assert ok, f"PCC failed: {pcc}"
