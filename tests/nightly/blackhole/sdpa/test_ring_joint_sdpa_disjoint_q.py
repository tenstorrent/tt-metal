# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe two 256-token prefill chunks in one CP=8 ring-joint SDPA call.

The proposed layout gives rank r Q rows [32*r, 32*r+32) and
[256+32*r, 256+32*r+32), packed into 64 local rows. The current API infers
the ownership stripe from those 64 rows, independently of q_chunk_size.

Controls cover contiguous 64-row ownership and two successive 32-row calls.
The one-call disjoint case is a strict numerical xfail; use --runxfail to
show its mismatch against attention in the original token order. No model
weights are needed. On Galaxy, the existing runtime opens four CP=8 rings.
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    call_sdpa,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
)

CP = 8
TILE = 32
TOKENS = 512
CACHE_CAPACITY = 1536  # Force the chunked path even at prefix=0; include an invalid tail.
HEAD_DIM = 64


@pytest.fixture(scope="module")
def cp8_runtime():
    if MESH_CONFIG.sp_size != CP:
        pytest.skip(f"This probe requires CP=8, detected CP={MESH_CONFIG.sp_size}")
    runtime = open_ring_joint_sdpa_runtime(MESH_CONFIG, topology=ttnn.Topology.Linear)
    try:
        yield runtime
    finally:
        close_ring_joint_sdpa_runtime(runtime)


def _rank_major_positions(start, end, stripe):
    """Natural positions packed by rank, then by successive ownership stripes."""
    positions = torch.arange(start, end).reshape(-1, CP, stripe)
    return positions.transpose(0, 1).reshape(-1)


def _probe(runtime, *, prefix, stripe, call_tokens, q_chunk_size, metadata):
    mesh = runtime.mesh_device
    heads = MESH_CONFIG.tp_size  # One independent head on each TP rank.
    shape = (1, heads, CACHE_CAPACITY, HEAD_DIM)
    # Zero logits make each row the mean of exactly its visible values. Scaled
    # tile IDs and +/-1 features are exactly representable in BFP8. Keep their
    # magnitudes near one so the absolute tolerance covers BF16 accumulation
    # (the metadata path does not support fp32_dest_acc_en).
    q = torch.zeros(shape)
    k = torch.zeros(shape)
    generator = torch.Generator().manual_seed(2026)
    tags = torch.randint(0, 2, (1, heads, CACHE_CAPACITY // TILE, HEAD_DIM), generator=generator).float() * 2 - 1
    tags[..., 0] = torch.arange(CACHE_CAPACITY // TILE) / TILE
    v = tags.repeat_interleave(TILE, dim=2)
    # A causal reference in natural token order, independent of cache packing.
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    allocated = []

    def upload(tensor, dtype, *, gathered=False):
        result = ttnn.from_torch(
            tensor.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(1, None if gathered else 2)),
        )
        allocated.append(result)
        return result

    def scalar(value):
        result = ttnn.from_torch(
            torch.tensor([value], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        allocated.append(result)
        return result

    try:
        cache_order = _rank_major_positions(0, CACHE_CAPACITY, stripe)
        tt_k = upload(k[:, :, cache_order], ttnn.bfloat8_b)
        tt_v = upload(v[:, :, cache_order], ttnn.bfloat8_b)
        buffer_k = upload(torch.zeros(shape), ttnn.bfloat8_b, gathered=True)
        buffer_v = upload(torch.zeros(shape), ttnn.bfloat8_b, gathered=True)
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=runtime.sdpa_compute_grid,
            q_chunk_size=q_chunk_size,
            k_chunk_size=32,
            exp_approx_mode=False,
        )
        composer = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=tuple(mesh.shape), dims=(1, 2))
        outputs = []
        for start in range(prefix, prefix + TOKENS, call_tokens):
            end = start + call_tokens
            positions = _rank_major_positions(start, end, stripe)
            tt_q = upload(q[:, :, positions], ttnn.bfloat16)
            kwargs = (
                {"slot_id": scalar(0), "kv_actual_isl_tensor": scalar(start)} if metadata else {"kv_actual_isl": start}
            )
            tt_out = call_sdpa(
                tt_q,
                tt_k,
                tt_v,
                logical_n=end,
                is_causal=True,
                is_balanced=False,
                p_buf_k=buffer_k,
                p_buf_v=buffer_v,
                program_config=program_config,
                compute_kernel_config=runtime.compute_kernel_config,
                ccl_semaphore_handles=runtime.ccl_semaphore_handles,
                num_links=runtime.num_links,
                sp_axis=runtime.sp_axis,
                mesh_device=mesh,
                topology=runtime.topology,
                worker_sub_device_id=runtime.worker_sub_device_id,
                ccl_column=runtime.ccl_column,
                **kwargs,
            )
            allocated.append(tt_out)
            packed_output = ttnn.to_torch(tt_out, mesh_composer=composer).float()
            outputs.append(packed_output[:, :, torch.argsort(positions)])
        output = torch.cat(outputs, dim=2)
        expected = reference[:, :, prefix : prefix + TOKENS]
        logger.info(
            "CP=8 prefix={} stripe={} call_tokens={} q_chunk={} metadata={}: max_abs_error={:.6f}; "
            "token {} value={:.6f}, expected={:.6f}",
            prefix,
            stripe,
            call_tokens,
            q_chunk_size,
            metadata,
            (output - expected).abs().max().item(),
            prefix + 256,
            output[0, 0, 256, 0].item(),
            expected[0, 0, 256, 0].item(),
        )
        return output, expected
    finally:
        for tensor in reversed(allocated):
            ttnn.deallocate(tensor)


@pytest.mark.parametrize("prefix", [0, 512])
@pytest.mark.parametrize("metadata", [False, True], ids=["host", "device_metadata"])
@pytest.mark.parametrize("stripe,call_tokens", [(64, 512), (32, 256)], ids=["contiguous_one_call", "striped_two_calls"])
def test_ring_joint_sdpa_cp8_layout_controls(cp8_runtime, prefix, metadata, stripe, call_tokens):
    output, expected = _probe(
        cp8_runtime, prefix=prefix, stripe=stripe, call_tokens=call_tokens, q_chunk_size=32, metadata=metadata
    )
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.03)


@pytest.mark.parametrize("prefix", [0, 512])
@pytest.mark.parametrize("metadata", [False, True], ids=["host", "device_metadata"])
@pytest.mark.parametrize("q_chunk_size", [32, 64])
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Ring SDPA infers 64-token ownership from Q; it cannot express two 32-token stripes per rank in one call",
)
def test_ring_joint_sdpa_cp8_disjoint_q_one_call(cp8_runtime, prefix, metadata, q_chunk_size):
    output, expected = _probe(
        cp8_runtime, prefix=prefix, stripe=32, call_tokens=512, q_chunk_size=q_chunk_size, metadata=metadata
    )
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.03)
