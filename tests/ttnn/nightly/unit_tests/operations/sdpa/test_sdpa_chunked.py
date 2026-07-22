# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import time
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest
from models.common.utility_functions import is_slow_dispatch


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def is_watcher_enabled():
    return os.environ.get("TT_METAL_WATCHER") is not None


def run_test_chunked_sdpa(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    q_chunk_size,
    k_chunk_size,
    prefill_chunk_size,
    page_block_size,
    q_dtype,
    k_dtype,
    use_high_precision_compute,
    flexible=False,
    grid_size=None,
    trace=False,
):
    """Run chunked SDPA over paged K/V and compare to PyTorch SDPA.

    flexible: If True, pass chunk_start_idx as a device tensor (ttnn.Tensor [1] int32).
              If False, pass chunk_start_idx as a Python int (legacy). Use flexible for
              trace capture/replay or prefix caching so the offset can change without recompile.
    trace: If True, compile once, capture one SDPA call, then replay for each chunk after
           updating Q and chunk_start_idx_tensor on device. Requires flexible=True and a
           device created with trace_region_size (see test parametrization).
    """
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size or device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    if use_high_precision_compute:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)
    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_repeated, V_repeated, is_causal=True)

    # Print shapes of all inputs along with input names
    logger.debug(f"Q: {Q.shape}")
    logger.debug(f"K: {K.shape}")
    logger.debug(f"V: {V.shape}")

    assert s % prefill_chunk_size == 0, "s must be divisible by prefill_chunk_size"
    assert prefill_chunk_size % page_block_size == 0, "prefill_chunk_size must be divisible by page_block_size"
    num_prefill_chunks = s // prefill_chunk_size
    # Prepare K, V paged for TT
    max_num_blocks_per_seq = s // page_block_size
    assert max_num_blocks_per_seq * page_block_size == s
    max_num_blocks = b * max_num_blocks_per_seq
    assert max_num_blocks * page_block_size == b * s

    # Shuffle paged KV cache according to some random page_table
    permutation = torch.randperm(max_num_blocks)
    reverse_permutation = torch.argsort(permutation)
    # page_table is the reverse permutation from shuffled -> unshuffled, and is used to map
    # a virtual block to the physical block id.
    page_table = reverse_permutation.reshape(b, max_num_blocks_per_seq)

    def page_cache(cache):
        paged_cache = (
            cache.reshape(b, nkv, max_num_blocks_per_seq, page_block_size, d)
            .transpose(1, 2)
            .reshape(max_num_blocks, nkv, page_block_size, d)
        )

        shuffled_page_cache = paged_cache[permutation]
        return shuffled_page_cache

    def unpage_cache(cache):
        unshuffled_page_cache = cache[reverse_permutation]
        paged_cache_back = (
            unshuffled_page_cache.reshape(b, nkv, max_num_blocks_per_seq, page_block_size, d)
            .transpose(1, 2)
            .reshape(b, nkv, s, d)
        )
        return paged_cache_back

    # Check that we can convert from normal to paged to normal
    assert torch.allclose(unpage_cache(page_cache(K)), K), "K is not equal to unpage_cache(page_cache(K))"
    assert torch.allclose(unpage_cache(page_cache(V)), V), "V is not equal to unpage_cache(page_cache(V))"

    tt_paged_K = ttnn.Tensor(page_cache(K), k_dtype).to(ttnn.TILE_LAYOUT).to(device)
    tt_paged_V = ttnn.Tensor(page_cache(V), k_dtype).to(ttnn.TILE_LAYOUT).to(device)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    def _params_str(chunk_idx=None, op="sdpa"):
        parts = [
            f"b={b} nh={nh} nkv={nkv} s={s} d={d}",
            f"q_chunk={q_chunk_size} k_chunk={k_chunk_size}",
            f"prefill_chunk={prefill_chunk_size} page_block={page_block_size}",
            f"flexible={flexible} trace={trace}",
        ]
        if chunk_idx is not None:
            parts.append(f"chunk_idx={chunk_idx}/{num_prefill_chunks}")
        parts.append(f"op={op}")
        return " | ".join(parts)

    if trace:
        # trace=True requires flexible (chunk_start_idx_tensor) so replay can update offset on device
        assert flexible, "trace=True requires flexible=True"
        # Persistent device tensors for capture and replay (same buffers used by trace)
        tt_Q_chunk = ttnn.Tensor(Q[:, :, 0:prefill_chunk_size], q_dtype).to(ttnn.TILE_LAYOUT).to(device)
        chunk_start_idx_tensor = ttnn.Tensor(torch.tensor([0], dtype=torch.int32), ttnn.int32).to(device)
        # Compile: run chunk 0 once
        ttnn.transformer.chunked_scaled_dot_product_attention(
            tt_Q_chunk,
            tt_paged_K,
            tt_paged_V,
            page_table_tt,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        # Capture: record one SDPA call
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        tt_back = ttnn.transformer.chunked_scaled_dot_product_attention(
            tt_Q_chunk,
            tt_paged_K,
            tt_paged_V,
            page_table_tt,
            chunk_start_idx_tensor=chunk_start_idx_tensor,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.end_trace_capture(device, tid, cq_id=0)

        # Execute: for each chunk, update inputs on device and replay; use replay output for PCC
        for chunk_idx in range(num_prefill_chunks):
            Q_chunk = Q[:, :, chunk_idx * prefill_chunk_size : (chunk_idx + 1) * prefill_chunk_size]
            chunk_start_idx = chunk_idx * prefill_chunk_size
            ttnn.copy(
                ttnn.Tensor(Q_chunk, q_dtype).to(ttnn.TILE_LAYOUT).to(device),
                tt_Q_chunk,
            )
            ttnn.copy(
                ttnn.Tensor(torch.tensor([chunk_start_idx], dtype=torch.int32), ttnn.int32).to(device),
                chunk_start_idx_tensor,
            )
            ttnn.synchronize_device(device)
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - t0
            logger.info(f"execute_trace time={elapsed:.4f}s | {_params_str(chunk_idx=chunk_idx, op='replay')}")
            tt_back_cpu = tt_back.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            gt_chunk = gt[:, :, chunk_idx * prefill_chunk_size : (chunk_idx + 1) * prefill_chunk_size]
            out_pass, out_pcc = comp_pcc(gt_chunk, tt_back_cpu, 0.998)
            logger.debug(f"trace replay chunk {chunk_idx} vs pytorch: {out_pcc}")
            assert out_pass
        ttnn.release_trace(device, tid)
    else:
        for chunk_idx in range(num_prefill_chunks):
            # Chunk Q
            Q_chunk = Q[:, :, chunk_idx * prefill_chunk_size : (chunk_idx + 1) * prefill_chunk_size]
            tt_Q_chunk = ttnn.Tensor(Q_chunk, q_dtype).to(ttnn.TILE_LAYOUT).to(device)
            chunk_start_idx = chunk_idx * prefill_chunk_size

            if flexible:
                chunk_start_idx_tensor = ttnn.Tensor(torch.tensor([chunk_start_idx], dtype=torch.int32), ttnn.int32).to(
                    device
                )
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                tt_back = ttnn.transformer.chunked_scaled_dot_product_attention(
                    tt_Q_chunk,
                    tt_paged_K,
                    tt_paged_V,
                    page_table_tt,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    program_config=program_config,
                    compute_kernel_config=compute_kernel_config,
                )
            else:
                ttnn.synchronize_device(device)
                t0 = time.perf_counter()
                tt_back = ttnn.transformer.chunked_scaled_dot_product_attention(
                    tt_Q_chunk,
                    tt_paged_K,
                    tt_paged_V,
                    page_table_tt,
                    chunk_start_idx,
                    program_config=program_config,
                    compute_kernel_config=compute_kernel_config,
                )
            ttnn.synchronize_device(device)
            elapsed = time.perf_counter() - t0
            logger.info(f"chunked_sdpa time={elapsed:.4f}s | {_params_str(chunk_idx=chunk_idx, op='sdpa')}")
            tt_back = tt_back.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
            gt_chunk = gt[:, :, chunk_idx * prefill_chunk_size : (chunk_idx + 1) * prefill_chunk_size]
            out_pass, out_pcc = comp_pcc(gt_chunk, tt_back, 0.998)
            logger.debug(f"python vs pytorch: {out_pcc}")
            assert out_pass


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("k_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("q_chunk_size", [128, 256], ids=["q128", "q256"])
@pytest.mark.parametrize("k_chunk_size", [128, 256], ids=["k128", "k256"])
@pytest.mark.parametrize("prefill_chunk_size", [1024, 2048])
@pytest.mark.parametrize("page_block_size", [64, 128])
# @pytest.mark.parametrize("flexible", [False, True], ids=["legacy", "flexible"])
@pytest.mark.parametrize("flexible", [True], ids=["flexible"])
# @pytest.mark.parametrize("flexible", [False], ids=["legacy"])
@pytest.mark.parametrize(
    "trace,device_params",
    [(False, {}), (True, {"trace_region_size": 256 * 1024})],
    indirect=["device_params"],
    ids=["no_trace", "trace"],
)
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    [
        [1, 8, 1, 16 * 1024, 128],
    ],  # Llama2-70B
)
def test_sdpa_chunked(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    q_chunk_size,
    k_chunk_size,
    prefill_chunk_size,
    page_block_size,
    flexible,
    trace,
    q_dtype,
    k_dtype,
    use_high_precision_compute=False,
):
    """Chunked SDPA: legacy (chunk_start_idx int) vs flexible (chunk_start_idx_tensor).
    When trace=True, device is created with trace_region_size; test captures one SDPA then replays per chunk."""
    if trace and not flexible:
        pytest.skip("Trace is not supported for legacy chunked SDPA")
    if trace and is_slow_dispatch():
        pytest.skip("Trace is not supported for slow dispatch")
    for _ in range(2):
        run_test_chunked_sdpa(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            q_chunk_size,
            k_chunk_size,
            prefill_chunk_size,
            page_block_size,
            q_dtype,
            k_dtype,
            use_high_precision_compute,
            flexible=flexible,
            trace=trace,
        )

    # Program cache: non-trace path only runs SDPA (1 entry). Trace path also runs two
    # ttnn.copy calls per chunk (Q chunk and chunk_start_idx_tensor), so 3 entries total.
    expected_entries = 3 if trace else 1
    assert (
        device.num_program_cache_entries() == expected_entries
    ), "Program cache should have {} entry/entries but has {}".format(
        expected_entries, device.num_program_cache_entries()
    )


def test_chunked_sdpa_legacy_scalar_chunk_start_idx_program_cache_key(device):
    """
    Legacy scalar chunk_start_idx changes chunked SDPA's compile-time effective
    K length, so different scalar offsets must not share one program cache entry.
    """
    b, nh, nkv, q_chunk_size, k_chunk_size, page_block_size, d = 1, 1, 1, 128, 128, 128, 128
    num_pages = 2

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    torch.manual_seed(0)
    tt_q = ttnn.from_torch(
        torch.randn(b, nh, q_chunk_size, d),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_k = ttnn.from_torch(
        torch.randn(num_pages, nkv, page_block_size, d),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_v = ttnn.from_torch(
        torch.randn(num_pages, nkv, page_block_size, d),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    page_table = ttnn.Tensor(torch.arange(num_pages, dtype=torch.int32).reshape(b, num_pages), ttnn.int32).to(device)

    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:
        for chunk_start_idx in (0, q_chunk_size):
            ttnn.transformer.chunked_scaled_dot_product_attention(
                tt_q,
                tt_k,
                tt_v,
                page_table,
                chunk_start_idx,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
            ttnn.synchronize_device(device)

        assert device.num_program_cache_entries() == 2, (
            "Legacy scalar chunk_start_idx changes chunked SDPA program geometry and must be part of "
            f"the cache key, got {device.num_program_cache_entries()} entries"
        )
    finally:
        device.disable_and_clear_program_cache()
        device.enable_program_cache()


@pytest.mark.skipif(is_watcher_enabled(), reason="Kernel OOM with watcher enabled")
@pytest.mark.parametrize("q_dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("k_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("q_chunk_size", [128])
@pytest.mark.parametrize("k_chunk_size", [128])
@pytest.mark.parametrize("prefill_chunk_size", [1024])
@pytest.mark.parametrize("page_block_size", [64])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    [
        [2, 1, 1, 4096, 128],
    ],  # Llama2-70B
)
def test_sdpa_chunked_iterate_batch(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    q_chunk_size,
    k_chunk_size,
    prefill_chunk_size,
    page_block_size,
    q_dtype,
    k_dtype,
    use_high_precision_compute=False,
):
    """
    This tests chunked prefill where a single core has more than one user to process.
    """
    for _ in range(2):
        run_test_chunked_sdpa(
            device,
            b,
            nh,
            nkv,
            s,
            d,
            q_chunk_size,
            k_chunk_size,
            prefill_chunk_size,
            page_block_size,
            q_dtype,
            k_dtype,
            use_high_precision_compute,
            grid_size=(1, 1),
        )

    expected_entries = s // prefill_chunk_size
    assert (
        device.num_program_cache_entries() == expected_entries
    ), "Program cache should have {} entry/entries but has {}".format(
        expected_entries, device.num_program_cache_entries()
    )


@pytest.mark.timeout(60)
def _permute_view_general(t_alloc, view_kv, view_block_size, view_head_dim):
    """Reinterpret paged-cache tiles under a different (kv, block, head_dim) view.

    Same helper as the decode / paged-cache flexible-geometry tests: kernels address by
    linear tile index within a block, so a plain torch reshape of untilized data does
    not match TILE_LAYOUT storage when alloc and view geometries differ.
    """
    N, alloc_kv, alloc_block_size, alloc_head_dim = t_alloc.shape
    TILE = 32
    alloc_BR_t = alloc_block_size // TILE
    alloc_Wt = alloc_head_dim // TILE
    view_BR_t = view_block_size // TILE
    view_Wt = view_head_dim // TILE
    alloc_total_tiles = alloc_kv * alloc_BR_t * alloc_Wt
    view_total_tiles = view_kv * view_BR_t * view_Wt
    assert alloc_total_tiles == view_total_tiles

    t = t_alloc.view(N, alloc_kv, alloc_BR_t, TILE, alloc_Wt, TILE)
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    t = t.reshape(N, alloc_total_tiles, TILE, TILE)
    t = t.reshape(N, view_kv, view_BR_t, view_Wt, TILE, TILE)
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    return t.reshape(N, view_kv, view_block_size, view_head_dim)


def test_chunked_sdpa_geometry_override_hma_shared_buffer(device):
    """HMA-shared cache: declared (nkv, block, head_dim) differs from the call view.

    Cache allocated as nkv=2 / block=64 / head_dim=128; call view is nkv=1 /
    block=64 / head_dim=256 (same elems/block). Overrides must address the buffer
    correctly so chunked SDPA matches the torch reference for the view geometry.

    Physical tiles are filled in the *declared* tile-grid (as from_torch on the
    alloc shape does); the reference reinterprets those same tiles through the
    view grid via ``_permute_view_general`` — matching how the kernel reads with
    PagedCacheGeometryOverride.
    """
    torch.manual_seed(7)
    b, nh, nkv_view, s, d_view = 1, 4, 1, 512, 256
    nkv_cache, block_size, d_cache = 2, 64, 128
    assert nkv_view * block_size * d_view == nkv_cache * block_size * d_cache
    q_chunk_size = 128
    prefill_chunk_size = 256
    assert s % prefill_chunk_size == 0

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=128,
        exp_approx_mode=True,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    num_pages = s // block_size
    page_table = torch.arange(num_pages, dtype=torch.int32).reshape(b, num_pages)

    # Fill the shared buffer in the *declared* tile-grid (peer-layer allocation).
    K_alloc = fa_rand(num_pages, nkv_cache, block_size, d_cache)
    V_alloc = fa_rand(num_pages, nkv_cache, block_size, d_cache)
    tt_k = ttnn.from_torch(K_alloc, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(V_alloc, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # Same DRAM bytes as the kernel sees under the view override.
    K_view_pages = _permute_view_general(K_alloc, nkv_view, block_size, d_view)
    V_view_pages = _permute_view_general(V_alloc, nkv_view, block_size, d_view)
    K = K_view_pages.reshape(b, nkv_view, s, d_view)
    V = V_view_pages.reshape(b, nkv_view, s, d_view)

    Q = fa_rand(b, nh, s, d_view)
    K_rep = K.repeat_interleave(nh // nkv_view, dim=1)
    V_rep = V.repeat_interleave(nh // nkv_view, dim=1)
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_rep, V_rep, is_causal=True)

    for chunk_idx in range(s // prefill_chunk_size):
        start = chunk_idx * prefill_chunk_size
        end = start + prefill_chunk_size
        tt_q = ttnn.from_torch(Q[:, :, start:end], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_out = ttnn.transformer.chunked_scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            page_table_tt,
            start,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            block_size=block_size,
            num_kv_heads=nkv_view,
        )
        out = tt_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()
        ok, msg = comp_pcc(gt[:, :, start:end], out, 0.998)
        assert ok, f"HMA geometry override chunk {chunk_idx} PCC fail: {msg}"


@pytest.mark.timeout(30)
def test_chunked_sdpa_geometry_override_rejects_elems_per_block_mismatch(device, expect_error):
    """Overrides that break the elems/block invariant must fail validation."""
    b, nh, nkv_cache, block_size, d_cache = 1, 4, 2, 64, 128
    num_pages = 4
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=True,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_q = ttnn.from_torch(
        torch.randn(b, nh, 128, 256),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_k = ttnn.from_torch(
        torch.randn(num_pages, nkv_cache, block_size, d_cache),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_v = ttnn.from_torch(
        torch.randn(num_pages, nkv_cache, block_size, d_cache),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    page_table = ttnn.Tensor(torch.arange(num_pages, dtype=torch.int32).reshape(b, num_pages), ttnn.int32).to(device)

    with expect_error(RuntimeError, "geometry mismatch|elems/block"):
        ttnn.transformer.chunked_scaled_dot_product_attention(
            tt_q,
            tt_k,
            tt_v,
            page_table,
            0,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            # 1*32*256 != 2*64*128
            block_size=32,
            num_kv_heads=1,
        )
