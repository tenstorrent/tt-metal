# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.attention.global_kv_cache import (
    GLOBAL_HEAD_DIM,
    GLOBAL_PACKED_DIM,
    GLOBAL_ROTARY_DIM,
    global_kv_indices,
    pack_global_kv_device,
    pack_global_kv_reference,
    pack_global_query_device,
    pack_global_query_reference,
    sliding_kv_indices,
    unpack_global_value_device,
    unpack_global_value_reference,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


def _partial_rope(x, cos, sin):
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


def test_global_kv_indices_partition_the_head():
    rotary, nonrotary, value = global_kv_indices()

    assert rotary.numel() == GLOBAL_ROTARY_DIM
    assert nonrotary.numel() == GLOBAL_HEAD_DIM - GLOBAL_ROTARY_DIM
    assert value.numel() == GLOBAL_HEAD_DIM
    assert torch.equal(torch.sort(torch.cat((rotary, nonrotary))).values, torch.arange(GLOBAL_HEAD_DIM))
    assert torch.equal(torch.sort(value).values, torch.arange(GLOBAL_HEAD_DIM))
    assert torch.equal(rotary[:6], torch.tensor([0, 256, 1, 257, 2, 258]))


@pytest.mark.parametrize("seq_len", [1, 7])
def test_packed_global_cache_matches_canonical_attention(seq_len):
    torch.manual_seed(0)
    shape = (1, 1, seq_len, GLOBAL_HEAD_DIM)
    value = torch.randn(shape)
    query = torch.randn(shape)
    gamma = torch.randn(GLOBAL_HEAD_DIM)

    # Identity outside the active 64 lanes in each NeoX half, matching Gemma's
    # 25% partial-RoPE cache.
    angle = torch.randn(1, 1, seq_len, GLOBAL_ROTARY_DIM // 2)
    cos = torch.ones(shape)
    sin = torch.zeros(shape)
    active_cos = angle.cos()
    active_sin = angle.sin()
    cos[..., :64] = active_cos
    cos[..., 256:320] = active_cos
    sin[..., :64] = active_sin
    sin[..., 256:320] = active_sin

    canonical_k = _partial_rope(value * gamma, cos, sin)
    canonical_q = _partial_rope(query, cos, sin)
    packed = pack_global_kv_reference(canonical_k, value)
    packed_q = pack_global_query_reference(canonical_q, gamma)

    assert packed.shape[-1] == GLOBAL_PACKED_DIM
    packed_k = packed[..., :GLOBAL_HEAD_DIM]
    packed_v = packed[..., GLOBAL_ROTARY_DIM:]
    torch.testing.assert_close(
        packed_q @ packed_k.transpose(-1, -2),
        canonical_q @ canonical_k.transpose(-1, -2),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(unpack_global_value_reference(packed_v), value)


def test_global_kv_indices_reject_invalid_dimensions(expect_error):
    with expect_error(ValueError, "positive even"):
        global_kv_indices(head_dim=511)
    with expect_error(ValueError, "smaller than"):
        global_kv_indices(rotary_dim=512)


def _adjacent_rope(x, cos, sin):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).reshape_as(x)
    return x * cos + rotated * sin


def test_projection_permutations_write_packed_cache_directly():
    torch.manual_seed(11)
    shape = (1, 2, 7, GLOBAL_HEAD_DIM)
    query_raw = torch.randn(shape)
    value_raw = torch.randn(shape)
    q_gamma = torch.randn(GLOBAL_HEAD_DIM)
    k_gamma = torch.randn(GLOBAL_HEAD_DIM)
    rotary, nonrotary, value_order = global_kv_indices()
    rotary_neox = torch.sort(rotary).values
    query_order = torch.cat((rotary, nonrotary))

    angle = torch.randn(1, 1, shape[-2], GLOBAL_ROTARY_DIM // 2)
    cos = torch.ones(1, 1, shape[-2], GLOBAL_HEAD_DIM)
    sin = torch.zeros_like(cos)
    cos[..., :64] = angle.cos()
    cos[..., 256:320] = angle.cos()
    sin[..., :64] = angle.sin()
    sin[..., 256:320] = angle.sin()

    def rms(x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    canonical_v = rms(value_raw)
    canonical_q = _partial_rope(rms(query_raw) * q_gamma, cos, sin)
    canonical_k = _partial_rope(canonical_v * k_gamma, cos, sin)

    # These are exactly the row permutations folded into q_proj, tied k_proj,
    # q_norm, and o_proj by load_attention_weights.
    direct_v = rms(value_raw.index_select(-1, value_order))
    direct_q = rms(query_raw.index_select(-1, query_order))
    direct_q = direct_q * q_gamma.index_select(0, query_order)
    direct_q[..., GLOBAL_ROTARY_DIM:] *= k_gamma.index_select(0, nonrotary)
    direct_q = torch.cat(
        (
            _adjacent_rope(
                direct_q[..., :GLOBAL_ROTARY_DIM],
                cos.index_select(-1, rotary),
                sin.index_select(-1, rotary),
            ),
            direct_q[..., GLOBAL_ROTARY_DIM:],
        ),
        dim=-1,
    )

    active_k = direct_v[..., -GLOBAL_ROTARY_DIM:] * k_gamma.index_select(0, rotary_neox)
    active_k = _partial_rope(active_k, cos.index_select(-1, rotary_neox), sin.index_select(-1, rotary_neox))
    interleave = torch.stack((torch.arange(64), torch.arange(64, 128)), dim=1).reshape(-1)
    direct_cache = torch.cat((active_k.index_select(-1, interleave), direct_v), dim=-1)

    torch.testing.assert_close(direct_q, pack_global_query_reference(canonical_q, k_gamma), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(direct_cache, pack_global_kv_reference(canonical_k, canonical_v), rtol=2e-5, atol=2e-5)

    packed_output = torch.randn(shape)
    output_weight = torch.randn(GLOBAL_HEAD_DIM, 96)
    packed_weight = output_weight.index_select(0, value_order)
    torch.testing.assert_close(
        packed_output @ packed_weight,
        unpack_global_value_reference(packed_output) @ output_weight,
        rtol=2e-5,
        atol=2e-5,
    )


def test_sliding_projection_order_matches_canonical_rope():
    torch.manual_seed(19)
    shape = (1, 4, 9, 256)
    query = torch.randn(shape)
    key = torch.randn(shape)
    q_gamma = torch.randn(256)
    k_gamma = torch.randn(256)
    order = sliding_kv_indices()
    angle = torch.randn(1, 1, shape[-2], 128)
    cos = torch.cat((angle.cos(), angle.cos()), dim=-1)
    sin = torch.cat((angle.sin(), angle.sin()), dim=-1)

    def rms(x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    canonical_q = _partial_rope(rms(query) * q_gamma, cos, sin)
    canonical_k = _partial_rope(rms(key) * k_gamma, cos, sin)
    direct_q = _adjacent_rope(
        rms(query.index_select(-1, order)) * q_gamma.index_select(0, order),
        cos.index_select(-1, order),
        sin.index_select(-1, order),
    )
    direct_k = _adjacent_rope(
        rms(key.index_select(-1, order)) * k_gamma.index_select(0, order),
        cos.index_select(-1, order),
        sin.index_select(-1, order),
    )

    torch.testing.assert_close(direct_q, canonical_q.index_select(-1, order), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(direct_k, canonical_k.index_select(-1, order), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        direct_q @ direct_k.transpose(-1, -2),
        canonical_q @ canonical_k.transpose(-1, -2),
        rtol=2e-5,
        atol=2e-5,
    )


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_device_packed_global_transforms_match_reference(mesh_device):
    torch.manual_seed(1)
    shape = (1, 1, 32, GLOBAL_HEAD_DIM)
    value = torch.randn(shape, dtype=torch.bfloat16)
    query = torch.randn(shape, dtype=torch.bfloat16)
    gamma = torch.randn(GLOBAL_HEAD_DIM, dtype=torch.bfloat16)

    angle = torch.randn(1, 1, 32, GLOBAL_ROTARY_DIM // 2)
    cos = torch.ones(shape)
    sin = torch.zeros(shape)
    cos[..., :64] = angle.cos()
    cos[..., 256:320] = angle.cos()
    sin[..., :64] = angle.sin()
    sin[..., 256:320] = angle.sin()
    canonical_q = _partial_rope(query.float(), cos, sin).bfloat16()

    rotary, nonrotary, _ = global_kv_indices()
    rotary_neox = torch.sort(rotary).values
    rotary_gamma = gamma[rotary_neox].reshape(1, 1, 1, -1)
    packed_q_scale = torch.cat((torch.ones(GLOBAL_ROTARY_DIM, dtype=gamma.dtype), gamma[nonrotary])).reshape(
        1, 1, 1, -1
    )

    def put(x, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            x,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    packed = pack_global_kv_device(
        put(value),
        put(rotary_gamma, ttnn.ROW_MAJOR_LAYOUT),
        put(cos.bfloat16()),
        put(sin.bfloat16()),
    )
    packed_q = pack_global_query_device(
        put(canonical_q),
        put(packed_q_scale, ttnn.ROW_MAJOR_LAYOUT),
    )
    packed_v = ttnn.slice(packed, [0, 0, 0, GLOBAL_ROTARY_DIM], [1, 1, 32, GLOBAL_PACKED_DIM])
    restored_v = unpack_global_value_device(packed_v)

    actual_packed = ttnn.to_torch(ttnn.get_device_tensors(packed)[0]).float()
    actual_q = ttnn.to_torch(ttnn.get_device_tensors(packed_q)[0]).float()
    actual_v = ttnn.to_torch(ttnn.get_device_tensors(restored_v)[0]).float()
    canonical_k = _partial_rope((value.float() * gamma), cos, sin)
    assert_with_pcc(pack_global_kv_reference(canonical_k, value.float()), actual_packed, 0.999)
    assert_with_pcc(pack_global_query_reference(canonical_q.float(), gamma.float()), actual_q, 0.999)
    assert_with_pcc(value.float(), actual_v, 0.9999)
