# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.mla_test_utils import run_flash_mla_prefill_impl


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope",
    # batch, seq_len, num heads q, num heads kv, kv lora rank, dim rope
    [
        # Full-size shapes (seq_len 1k/4k) run in tests/ttnn/nightly/unit_tests/operations/sdpa/test_mla_prefill.py
        (2, 256, 128, 1, 512, 64),  # large head count
        (2, 512, 64, 1, 256, 0),  # zero d_rope
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype, use_paged_attention, block_size",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b, True, 128),
        (ttnn.bfloat8_b, ttnn.bfloat4_b, True, 32),
    ],
)
def test_flash_mla_prefill(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    use_paged_attention,
    block_size,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
        use_paged_attention,
        block_size,
    )


def test_chunked_flash_mla_prefill_partial_single_page_paged_cache(
    device,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_impl(
        device,
        batch=1,
        seq_len=64,
        nh=32,
        nkv=1,
        kv_lora_rank=64,
        d_rope=64,
        q_dtype=ttnn.bfloat16,
        dtype=ttnn.bfloat8_b,
        use_paged_attention=True,
        block_size=64,
    )


@pytest.mark.parametrize("cache_dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.parametrize("chunk", [64, 128])
def test_chunked_flash_mla_prefill_separate_v(device, cache_dtype, chunk, function_level_defaults, reset_seeds):
    """Chunked MLA prefill over separate paged K (192 wide) and V (128 wide) caches, as MiMo's global layers keep
    them: every chunk attends to the earlier chunks and itself through a shuffled page table."""
    import torch

    from models.common.utility_functions import comp_pcc

    nh, dk, dv, block, seq = 16, 192, 128, 64, 512
    num_blocks = seq // block
    scale = dk**-0.5
    q = torch.randn(1, nh, seq, dk)
    k = torch.randn(1, 1, seq, dk)
    v = torch.randn(1, 1, seq, dv)
    perm = torch.randperm(num_blocks)

    def paged(t):
        blocks = t[0, 0].reshape(num_blocks, block, -1)
        cache = torch.zeros(num_blocks, 1, block, t.shape[-1])
        cache[perm, 0] = blocks
        return ttnn.from_torch(cache, device=device, dtype=cache_dtype, layout=ttnn.TILE_LAYOUT)

    k_cache, v_cache = paged(k), paged(v)
    page_table = ttnn.from_torch(perm.reshape(1, -1).to(torch.int32), device=device, dtype=ttnn.int32)
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
    )
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    k_ref = ttnn.to_torch(k_cache).float()[perm, 0].reshape(1, 1, seq, dk)
    v_ref = ttnn.to_torch(v_cache).float()[perm, 0].reshape(1, 1, seq, dv)
    for start in range(0, seq, chunk):
        q_chunk = q[:, :, start : start + chunk]
        out = ttnn.transformer.chunked_flash_mla_prefill(
            ttnn.from_torch(q_chunk, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            k_cache,
            dv,
            page_table,
            start,
            scale=scale,
            program_config=program,
            compute_kernel_config=compute,
            input_tensor_v=v_cache,
        )
        got = ttnn.to_torch(out).float()
        end = start + chunk
        scores = (q_chunk @ k_ref[:, :, :end].transpose(-1, -2)) * scale
        pos_q = torch.arange(start, end)[:, None]
        scores = scores.masked_fill(torch.arange(end)[None, :] > pos_q, float("-inf"))
        expected = torch.softmax(scores, dim=-1) @ v_ref[:, :, :end]
        assert got.shape == expected.shape, (got.shape, expected.shape)
        passing, message = comp_pcc(expected, got, 0.999)
        assert passing, f"chunk at {start}: {message}"


@pytest.mark.parametrize("chunk", [64, 128])
def test_chunked_flash_mla_prefill_start_tensor(device, chunk, function_level_defaults, reset_seeds):
    """chunk_start_idx_tensor: the chunk's start read from a device tensor gives the same output as the scalar start,
    and one program serves every start (the scalar start is part of the program cache key)."""
    import torch

    nh, dk, dv, block, seq = 16, 192, 128, 64, 1024
    num_blocks = seq // block
    q = torch.randn(1, nh, seq, dk)
    perm = torch.randperm(num_blocks)
    k_cache, v_cache = (
        ttnn.from_torch(
            torch.randn(num_blocks, 1, block, d), device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        for d in (dk, dv)
    )
    # The flexible path reads the whole page table row as one stick: a multiple of 8 int32 entries.
    page_table = ttnn.from_torch(perm.reshape(1, -1).to(torch.int32), device=device, dtype=ttnn.int32)
    kwargs = dict(
        scale=dk**-0.5,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        ),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False
        ),
        input_tensor_v=v_cache,
    )
    programs = None
    for start in range(chunk, seq, chunk):
        q_chunk = ttnn.from_torch(
            q[:, :, start : start + chunk], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        scalar = ttnn.to_torch(
            ttnn.transformer.chunked_flash_mla_prefill(q_chunk, k_cache, dv, page_table, start, **kwargs)
        )
        start_tt = ttnn.from_torch(torch.tensor([start], dtype=torch.int32), device=device, dtype=ttnn.int32)
        before = device.num_program_cache_entries()
        runtime = ttnn.to_torch(
            ttnn.transformer.chunked_flash_mla_prefill(
                q_chunk, k_cache, dv, page_table, chunk_start_idx_tensor=start_tt, **kwargs
            )
        )
        if programs is None:
            programs = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == before, f"chunk at {start} compiled a new program"
        assert torch.equal(scalar, runtime), f"chunk at {start}: tensor start differs from the scalar start"


def _mimo_global_chunk_setup(device, seq, chunk, block=64, nh=8, dk=192, dv=128):
    """MiMo global-layer shapes per chip: 8 Q heads over one K/V head, separate BFP8 paged K (192) and V (128)
    caches behind a shuffled page table."""
    import torch

    num_blocks = seq // block
    q = torch.randn(1, nh, seq, dk)
    k = torch.randn(1, 1, seq, dk)
    v = torch.randn(1, 1, seq, dv)
    perm = torch.randperm(num_blocks)

    def paged(t):
        cache = torch.zeros(num_blocks, 1, block, t.shape[-1])
        cache[perm, 0] = t[0, 0].reshape(num_blocks, block, -1)
        return ttnn.from_torch(cache, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    k_cache, v_cache = paged(k), paged(v)
    page_table = ttnn.from_torch(perm.reshape(1, -1).to(torch.int32), device=device, dtype=ttnn.int32)
    k_ref = ttnn.to_torch(k_cache).float()[perm, 0].reshape(1, 1, seq, dk)
    v_ref = ttnn.to_torch(v_cache).float()[perm, 0].reshape(1, 1, seq, dv)
    return q, k_cache, v_cache, page_table, k_ref, v_ref


def _causal_chunk_reference(q_chunk, k_ref, v_ref, start, scale):
    import torch

    end = start + q_chunk.shape[2]
    scores = (q_chunk @ k_ref[:, :, :end].transpose(-1, -2)) * scale
    pos_q = torch.arange(start, end)[:, None]
    scores = scores.masked_fill(torch.arange(end)[None, :] > pos_q, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v_ref[:, :, :end]


def _run_grouped_split(device, q, k_cache, v_cache, page_table, starts, chunk, q_chunk, k_chunk, heads, splits, grid):
    """Every start through one program: returns {start: output} and asserts the program cache does not grow."""
    import torch

    dk, dv = q.shape[-1], 128
    program = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
        k_splits=splits,
        q_heads_per_unit=heads,
    )
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    outs = {}
    entries = None
    for start in starts:
        q_tt = ttnn.from_torch(
            q[:, :, start : start + chunk], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        start_tt = ttnn.from_torch(torch.tensor([start], dtype=torch.int32), device=device, dtype=ttnn.int32)
        out = ttnn.transformer.chunked_flash_mla_prefill(
            q_tt,
            k_cache,
            dv,
            page_table,
            chunk_start_idx_tensor=start_tt,
            scale=dk**-0.5,
            program_config=program,
            compute_kernel_config=compute,
            input_tensor_v=v_cache,
        )
        outs[start] = ttnn.to_torch(out).float()
        if entries is None:
            entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == entries, f"start {start} compiled a new program"
    return outs


@pytest.mark.parametrize("q_chunk, k_chunk", [(128, 128), (256, 256)])
def test_chunked_flash_mla_prefill_grouped_heads(device, q_chunk, k_chunk, function_level_defaults, reset_seeds):
    """q_heads_per_unit=2 (two Q heads share each K/V chunk read) is bit exact against one head per unit, for
    starts 0, one chunk, a long prefix and a multiple of 64 that is not q aligned, all through one program."""
    import torch

    from models.common.utility_functions import comp_pcc

    seq, chunk = 4096, 512
    q, k_cache, v_cache, page_table, k_ref, v_ref = _mimo_global_chunk_setup(device, seq, chunk)
    starts = [0, chunk, seq - chunk, 1088]
    grid = device.compute_with_storage_grid_size()
    run = lambda heads: _run_grouped_split(
        device, q, k_cache, v_cache, page_table, starts, chunk, q_chunk, k_chunk, heads, 1, grid
    )
    one, two = run(1), run(2)
    for start in starts:
        assert torch.equal(one[start], two[start]), f"start {start}: q_heads_per_unit=2 differs from 1"
        if start % q_chunk == 0:
            expected = _causal_chunk_reference(q[:, :, start : start + chunk], k_ref, v_ref, start, 192**-0.5)
            passing, message = comp_pcc(expected, two[start], 0.999)
            assert passing, f"start {start}: {message}"


@pytest.mark.parametrize("heads, splits", [(1, 3), (2, 3), (2, 15), (1, 15)])
def test_chunked_flash_mla_prefill_k_split(device, heads, splits, function_level_defaults, reset_seeds):
    """k_splits: each unit's keys split over several work items, merged after the program. Start 0 leaves most
    slices empty; the other starts are not slice aligned; one program serves every start."""
    import torch

    from models.common.utility_functions import comp_pcc

    seq, chunk, q_chunk, k_chunk = 4096, 512, 256, 256
    q, k_cache, v_cache, page_table, k_ref, v_ref = _mimo_global_chunk_setup(device, seq, chunk)
    starts = [0, chunk, 1536, seq - chunk]
    grid = device.compute_with_storage_grid_size()
    base = _run_grouped_split(device, q, k_cache, v_cache, page_table, starts, chunk, q_chunk, k_chunk, 1, 1, grid)
    split = _run_grouped_split(
        device, q, k_cache, v_cache, page_table, starts, chunk, q_chunk, k_chunk, heads, splits, grid
    )
    for start in starts:
        expected = _causal_chunk_reference(q[:, :, start : start + chunk], k_ref, v_ref, start, 192**-0.5)
        got = split[start]
        assert got.shape == expected.shape, (got.shape, expected.shape)
        passing, message = comp_pcc(expected, got, 0.999)
        base_err = (base[start] - expected).abs().max().item()
        err = (got - expected).abs().max().item()
        vs_base = (got - base[start]).abs().max().item()
        print(
            f"K_SPLIT heads={heads} splits={splits} start={start}: {message} max_abs={err:.4f} "
            f"(unsplit {base_err:.4f}, vs unsplit {vs_base:.4f})"
        )
        assert passing, f"start {start}: {message}"
        assert err <= 2 * base_err + 0.02, f"start {start}: max abs error {err} vs unsplit {base_err}"
