# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ring-distributed SDPA (ttnn.transformer.ring_distributed_scaled_dot_product_attention) with precision recipes,
against an FP64 causal reference.

Each device computes two slabs of the causal attention, sequence chunks ring_id and 2 * ring_size - 1 - ring_id
(the legacy op's split, test_sdpa_ring_distributed.py). One device emulates every ring position through an explicit
ring_id; a mesh test infers ring_id from each device's coordinate. Recipes and their numerics:
tech_reports/FlashAttention/SDPAPrecisionRecipes.md.
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.sdpa_recipe_test_utils import (
    L2_PCT_BOUND,
    VARIANTS,
    recipe_hardware,
    inputs_for,
    key_mask,
    l2_pct,
    randn,
    reference,
    stored,
    to_device,
)

pytestmark = recipe_hardware


def slabs(ring_size, ring_id, s):
    """Rows of the two slabs a ring position computes, in output order."""
    rows = s // (2 * ring_size)
    late = 2 * ring_size - 1 - ring_id
    return torch.cat([torch.arange(ring_id * rows, (ring_id + 1) * rows), torch.arange(late * rows, (late + 1) * rows)])


def causal_reference(q, k, v, scale=None):
    s = q.shape[2]
    return reference(q, k, v, key_mask(s, s, causal=True), scale)


def config(device, q_chunk, k_chunk, exp_approx_mode=None):
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=exp_approx_mode,
    )


def fp32_dest_config(device):
    """The production caller's compute config (Galaxy Llama: HiFi4-class, FP32 dest); the recipe ignores it."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


# b, nh, nkv, s, d, ring_size, q_chunk, k_chunk (0 = op-chosen). Slabs (s / (2 * ring_size) rows) hold whole Q
# chunks; K chunks need not divide anything.
CASES = {
    "ring4_q128_k256": (1, 8, 1, 4096, 128, 4, 128, 256),
    "ring2_q256_k512": (1, 2, 2, 4096, 128, 2, 256, 512),
    "ring4_odd_q96_k160_d64": (1, 2, 2, 3072, 64, 4, 96, 160),
    "ring8_q32_k64": (1, 2, 1, 1024, 128, 8, 32, 64),
    "gqa_batch2": (2, 8, 2, 2048, 128, 4, 128, 256),
    "op_selected": (1, 8, 1, 8192, 128, 4, 0, 0),
}


def run_ring_positions(device, variant, case, *, scale=None, q_dtype=None, kv_dtype=None, **kwargs):
    """Every ring position on one device; returns the output rows each computed, gathered into sequence order,
    and the FP64 reference on the stored inputs."""
    b, nh, nkv, s, d, ring_size, q_chunk, k_chunk = CASES[case]
    precision = VARIANTS[variant][0]
    q, k, v = randn(b, nh, s, d, seed=51), randn(b, nkv, s, d, seed=52), randn(b, nkv, s, d, seed=53)
    if q_dtype is None:
        tq, tk, tv = inputs_for(device, variant, q, k, v)
        if precision == ttnn.SDPAPrecision.FAST:
            q, k, v = (ttnn.to_torch(x) for x in (tq, tk, tv))
    else:
        q, k, v = stored(q, q_dtype), stored(k, kv_dtype), stored(v, kv_dtype)
        tq, tk, tv = to_device(device, q, q_dtype), to_device(device, k, kv_dtype), to_device(device, v, kv_dtype)
    program_config = config(device, q_chunk, k_chunk) if q_chunk else None
    actual = torch.zeros(b, nh, s, d)
    for ring_id in range(ring_size):
        out = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            tq,
            tk,
            tv,
            ring_size=ring_size,
            ring_id=ring_id,
            scale=scale,
            program_config=program_config,
            precision=precision,
            **kwargs,
        )
        assert out.dtype == tq.dtype and list(out.shape) == [b, nh, s // ring_size, d]
        actual[:, :, slabs(ring_size, ring_id, s)] = ttnn.to_torch(out).float()
    return actual, causal_reference(q, k, v, scale)


@pytest.mark.parametrize("case", CASES, ids=CASES.keys())
@pytest.mark.parametrize("variant", VARIANTS)
def test_ring_distributed_sdpa_recipe(device, variant, case):
    actual, expected = run_ring_positions(device, variant, case)
    assert l2_pct(actual, expected) < L2_PCT_BOUND[variant]


@pytest.mark.parametrize("variant", ["standard", "balanced", "accurate"])
def test_ring_distributed_sdpa_recipe_galaxy_arguments(device, variant):
    """The production call (Galaxy Llama prefill): BFP8 Q/K/V, GQA, a custom scale and an FP32-dest compute config.
    The output is BFP8 like Q, so the bound allows for its rounding."""
    scale = 0.0625
    actual, expected = run_ring_positions(
        device,
        variant,
        "ring4_q128_k256",
        scale=scale,
        q_dtype=ttnn.bfloat8_b,
        kv_dtype=ttnn.bfloat8_b,
        compute_kernel_config=fp32_dest_config(device),
    )
    output_rounding = 1.5 * l2_pct(stored(expected.bfloat16(), ttnn.bfloat8_b), expected)
    assert l2_pct(actual, expected) < L2_PCT_BOUND[variant] + output_rounding


@pytest.mark.parametrize("variant", ["standard", "accurate"])
def test_ring_distributed_sdpa_recipe_program_cache(device, variant):
    """Programs differ per ring position only in runtime args; each position's program is cached on its own."""
    device.enable_program_cache()
    for _ in range(2):
        actual, expected = run_ring_positions(device, variant, "ring4_q128_k256")
        assert l2_pct(actual, expected) < L2_PCT_BOUND[variant]


def test_ring_distributed_sdpa_recipe_rejects_partial_chunks(device, expect_error):
    """Slabs must hold whole Q chunks (the legacy rule), here 512-row slabs and Q chunks of 384 rows."""
    q, k = randn(1, 1, 4096, 128, seed=54), randn(1, 1, 4096, 128, seed=55)
    tq, tk = to_device(device, q), to_device(device, k)
    with expect_error(RuntimeError, "must hold whole Q chunks"):
        ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            tq,
            tk,
            tk,
            ring_size=4,
            ring_id=0,
            program_config=config(device, 384, 512),
            precision=ttnn.SDPAPrecision.STANDARD,
        )


@pytest.mark.parametrize("variant", ["standard", "accurate", "fast_bfp8"])
def test_ring_distributed_sdpa_recipe_paged_prefix(device, variant):
    """Prefix caching (the legacy op's chunk_start_idx + page_table): Q holds the sequence's new rows after a cached
    prefix, K/V a shuffled paged cache of the whole sequence."""
    b, nh, nkv, s, d, block, prefix, ring_size = 1, 8, 1, 4096, 128, 64, 1024, 4
    precision, kv_dtype = VARIANTS[variant]
    q = randn(b, nh, s - prefix, d, seed=56)
    k, v = randn(b, nkv, s, d, seed=57), randn(b, nkv, s, d, seed=58)
    tq, tk, tv = inputs_for(device, variant, q, k, v)
    if precision == ttnn.SDPAPrecision.FAST:
        q, k, v = (ttnn.to_torch(x) for x in (tq, tk, tv))
    blocks = s // block
    permutation = torch.randperm(blocks, generator=torch.Generator().manual_seed(59))
    page_table = ttnn.from_torch(
        torch.argsort(permutation).reshape(b, blocks).to(torch.int32), dtype=ttnn.int32, device=device
    )

    def paged(x):
        cache = x.reshape(b, nkv, blocks, block, d).transpose(1, 2).reshape(blocks, nkv, block, d)[permutation]
        return to_device(device, cache, kv_dtype if precision == ttnn.SDPAPrecision.FAST else ttnn.bfloat16)

    pk, pv = paged(k), paged(v)
    actual = torch.zeros(b, nh, s - prefix, d)
    for ring_id in range(ring_size):
        out = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
            tq,
            pk,
            pv,
            ring_size=ring_size,
            ring_id=ring_id,
            program_config=config(device, 128, 256),
            page_table=page_table,
            chunk_start_idx=prefix,
            precision=precision,
        )
        actual[:, :, slabs(ring_size, ring_id, s - prefix)] = ttnn.to_torch(out).float()
    expected = reference(q, k, v, key_mask(s - prefix, s, causal=True, q_offset=prefix))
    assert l2_pct(actual, expected) < L2_PCT_BOUND[variant]


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("variant", ["standard", "accurate", "fast_bfp8"])
def test_ring_distributed_sdpa_recipe_mesh(mesh_device, variant):
    """ring_id inferred from each device's mesh coordinate: one program per device, differing in its slabs."""
    b, nh, nkv, s, d, ring_size, q_chunk, k_chunk = CASES["ring4_q128_k256"]
    if mesh_device.get_num_devices() != ring_size:
        pytest.skip(f"Requires a 1x{ring_size} mesh")
    precision, kv_dtype = VARIANTS[variant]
    q, k, v = randn(b, nh, s, d, seed=58), randn(b, nkv, s, d, seed=59), randn(b, nkv, s, d, seed=60)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    tq, tk, tv = (
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=replicate)
        for x in (q, k, v)
    )
    if precision == ttnn.SDPAPrecision.FAST:
        tq = ttnn.transformer.prepare_sdpa_input(tq, is_query=True)
        tk, tv = (ttnn.transformer.prepare_sdpa_input(x, is_query=False, dtype=kv_dtype) for x in (tk, tv))
        q, k, v = (ttnn.to_torch(ttnn.get_device_tensors(x)[0]) for x in (tq, tk, tv))
    out = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
        tq,
        tk,
        tv,
        ring_size=ring_size,
        program_config=config(mesh_device, q_chunk, k_chunk),
        precision=precision,
    )
    expected = causal_reference(q, k, v)
    for ring_id, chip in enumerate(ttnn.get_device_tensors(out)):
        rows = slabs(ring_size, ring_id, s)
        assert l2_pct(ttnn.to_torch(chip), expected[:, :, rows]) < L2_PCT_BOUND[variant]
