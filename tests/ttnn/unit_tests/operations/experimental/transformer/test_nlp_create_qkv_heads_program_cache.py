# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Program-cache regression tests for ttnn.experimental.nlp_create_qkv_heads
(NlpCreateHeadsDeviceOperation), covering both program factories.

These pin the override_runtime_arguments cache-hit path (which replaced
get_dynamic_runtime_args): every per-core runtime arg and tensor-backed CB
address must be re-derived from the CURRENT tensors on a hit. The key case is
re-running the SAME config with re-allocated buffers so input/output addresses
differ on the hit -> all three outputs (Q, K, V) must still be correct.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, tt2torch_tensor, skip_for_blackhole


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


def _refs_interleaved(A, batch, seq_len, head_dim, num_q_heads, num_kv_heads, transpose_k_heads):
    ref_q, ref_k, ref_v = torch.split(
        A, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
    )
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [batch, seq_len, num_kv_heads, head_dim]).transpose(-3, -2)
    if transpose_k_heads:
        ref_k = ref_k.transpose(-2, -1)
    return ref_q, ref_k, ref_v


def _make_interleaved_input(device, batch, seq_len, head_dim, num_q_heads, num_kv_heads, dtype, mem_config, seed):
    torch.manual_seed(seed)
    A = torch.randn([batch, 1, seq_len, (num_q_heads + 2 * num_kv_heads) * head_dim])
    return A, ttnn.Tensor(A, dtype).to(ttnn.TILE_LAYOUT).to(device, mem_config)


def _run_interleaved(in0_t, num_q_heads, num_kv_heads, transpose_k_heads, mem_config):
    return ttnn.experimental.nlp_create_qkv_heads(
        in0_t,
        None,
        num_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transpose_k_heads=transpose_k_heads,
        memory_config=mem_config,
    )


def _check(outs, refs, pcc):
    for got, ref, name in zip(outs, refs, ("Q", "K", "V")):
        passing, val = comp_pcc(tt2torch_tensor(got), ref, pcc)
        assert passing, f"{name} output mismatch on cache hit: pcc={val}"


@pytest.mark.parametrize("transpose_k_heads", (False, True), ids=["no_transpose", "transpose_k"])
def test_nlp_cqkv_interleaved_addr_change_on_hit(device, isolate_program_cache, transpose_k_heads):
    """Interleaved factory: same config twice with re-allocated buffers -> 1 entry, all outputs correct.

    Holding the first input alive forces the second input (and all outputs) to different addresses,
    so a stale baked address on the cache hit would corrupt Q/K/V.
    """
    batch, seq_len, head_dim, num_q_heads, num_kv_heads = 1, 128, 64, 8, 2
    dtype, mem_config = ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG

    A1, in1 = _make_interleaved_input(device, batch, seq_len, head_dim, num_q_heads, num_kv_heads, dtype, mem_config, 1)
    outs1 = _run_interleaved(in1, num_q_heads, num_kv_heads, transpose_k_heads, mem_config)
    _check(outs1, _refs_interleaved(A1, batch, seq_len, head_dim, num_q_heads, num_kv_heads, transpose_k_heads), 1.0)

    A2, in2 = _make_interleaved_input(device, batch, seq_len, head_dim, num_q_heads, num_kv_heads, dtype, mem_config, 2)
    assert in1.buffer_address() != in2.buffer_address(), "inputs must land at different addresses to exercise the hit"
    outs2 = _run_interleaved(in2, num_q_heads, num_kv_heads, transpose_k_heads, mem_config)
    _check(outs2, _refs_interleaved(A2, batch, seq_len, head_dim, num_q_heads, num_kv_heads, transpose_k_heads), 1.0)

    assert device.num_program_cache_entries() == 1


def test_nlp_cqkv_interleaved_shape_change(device, isolate_program_cache):
    """Different input shape (seq_len) -> distinct entries (input TensorSpec is hashed)."""
    head_dim, num_q_heads, num_kv_heads = 64, 8, 2
    dtype, mem_config = ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG
    for seq_len in (128, 256):
        A, in_t = _make_interleaved_input(device, 1, seq_len, head_dim, num_q_heads, num_kv_heads, dtype, mem_config, 1)
        outs = _run_interleaved(in_t, num_q_heads, num_kv_heads, False, mem_config)
        _check(outs, _refs_interleaved(A, 1, seq_len, head_dim, num_q_heads, num_kv_heads, False), 1.0)
    assert device.num_program_cache_entries() == 2


def _make_sharded_input(device, batch, seq_len, head_dim, num_q_heads, num_kv_heads, dtype, seed):
    torch.manual_seed(seed)
    compute_grid_size = device.compute_with_storage_grid_size()
    num_cores = num_kv_heads
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid_size, True)
    q_shape = [seq_len, 1, batch, num_cores, num_q_heads // num_cores * head_dim]
    kv_shape = [seq_len, 1, batch, num_cores, num_kv_heads // num_cores * head_dim]
    Q, K, V = torch.randn(q_shape), torch.randn(kv_shape), torch.randn(kv_shape)
    A = torch.concat([Q.flatten(-2, -1), K.flatten(-2, -1), V.flatten(-2, -1)], -1)
    A_interleaved = torch.concat([Q, K, V], -1).flatten(-2, -1)
    in_shard_spec = ttnn.ShardSpec(
        shard_grid, [seq_len * batch, A_interleaved.shape[-1] // num_cores], ttnn.ShardOrientation.ROW_MAJOR
    )
    in_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in_shard_spec)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_shard_spec)
    in_t = ttnn.Tensor(A_interleaved, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)
    return A, in_t, out_mem_config


def _refs_sharded(A, batch, seq_len, head_dim, num_q_heads, num_kv_heads):
    ref_q, ref_k, ref_v = torch.split(
        A, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1
    )
    ref_q = torch.reshape(ref_q, [seq_len, batch, num_q_heads, head_dim]).transpose(-3, -2)
    ref_k = torch.reshape(ref_k, [seq_len, batch, num_kv_heads, head_dim]).transpose(-3, -2)
    ref_v = torch.reshape(ref_v, [seq_len, batch, num_kv_heads, head_dim]).transpose(-3, -2)
    return ref_q, ref_k, ref_v


@skip_for_blackhole("L1 and Circular buffers are crashing on BH, see #12349")
def test_nlp_cqkv_sharded_addr_change_on_hit(device, isolate_program_cache):
    """Sharded factory: same config twice with re-allocated buffers -> 1 entry, all outputs correct.

    The Sharded reader/writer bake q/k/v base + per-core start addresses as raw uint32 args, so this
    is the case the old get_dynamic path guarded; override_runtime_arguments must re-derive them.
    """
    batch, seq_len, head_dim, num_q_heads, num_kv_heads = 32, 1, 64, 16, 8
    dtype = ttnn.bfloat16

    A1, in1, out_cfg1 = _make_sharded_input(device, batch, seq_len, head_dim, num_q_heads, num_kv_heads, dtype, 1)
    q1, k1, v1 = ttnn.experimental.nlp_create_qkv_heads(
        in1, None, num_heads=num_q_heads, num_kv_heads=num_kv_heads, transpose_k_heads=False, memory_config=out_cfg1
    )
    _check((q1, k1, v1), _refs_sharded(A1, batch, seq_len, head_dim, num_q_heads, num_kv_heads), 1.0)

    A2, in2, out_cfg2 = _make_sharded_input(device, batch, seq_len, head_dim, num_q_heads, num_kv_heads, dtype, 2)
    assert in1.buffer_address() != in2.buffer_address(), "inputs must land at different addresses to exercise the hit"
    q2, k2, v2 = ttnn.experimental.nlp_create_qkv_heads(
        in2, None, num_heads=num_q_heads, num_kv_heads=num_kv_heads, transpose_k_heads=False, memory_config=out_cfg2
    )
    _check((q2, k2, v2), _refs_sharded(A2, batch, seq_len, head_dim, num_q_heads, num_kv_heads), 1.0)

    assert device.num_program_cache_entries() == 1
