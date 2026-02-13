# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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

    # Print number of program cache entries
    assert device.num_program_cache_entries() == 1, "Program cache should only have 1 entry but has {}".format(
        device.num_program_cache_entries()
    )
