# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Program-cache regression tests for ttnn.experimental.create_qkv_heads
(CreateQKVHeadsDeviceOperation).

Pins the cache-keying granularity so it can be verified BEFORE and AFTER
removing CreateQKVHeadsDeviceOperation::compute_program_hash. The custom hash
keyed on num_q_heads, num_kv_heads, head_dim, transpose_k_heads,
output_mem_config, the input tensor (TensorSpec = logical shape) and a redundant
compute_with_storage_grid_size() (constant per device). The framework default
hashes all attributes + the input TensorSpec, i.e. the same distinctions minus
the redundant device-grid term, and keys on logical shape (no padded coarsening).

- Same config -> reuse (1 entry).
- transpose_k_heads toggled (a hashed attribute, same input) -> distinct entries.
- Different input shape (seq_len) -> distinct entries.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, skip_for_blackhole, tt2torch_tensor


def run_cqkv(device, batch, seq_len, num_q_heads, num_kv_heads, head_dim, cores_h, cores_w, transpose_k, dtype):
    """Build block-sharded QKV, run create_qkv_heads inside the cache counter; return pcc(q)."""
    torch.manual_seed(1234)
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

    q_shape = [batch, 1, seq_len, num_kv_heads, num_q_heads // num_kv_heads * head_dim]
    k_shape = [batch, 1, seq_len, num_kv_heads, head_dim]
    v_shape = [batch, 1, seq_len, num_kv_heads, head_dim]
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(v_shape, dtype=torch_dtype)
    QKV = torch.concat([Q.flatten(-2, -1), K.flatten(-2, -1), V.flatten(-2, -1)], -1)
    QKV_interleaved = torch.concat([Q, K, V], -1).flatten(-2, -1)

    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores_w - 1, cores_h - 1))})
    in_shard_spec = ttnn.ShardSpec(grid, [seq_len, QKV.shape[-1] // cores_w], ttnn.ShardOrientation.ROW_MAJOR)
    in_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, in_shard_spec)
    # The device op recomputes the real output shard spec from the input and only consumes the output
    # memory_config's layout + buffer_type; the shard_spec shape passed here is functionally ignored.
    # But the full MemoryConfig (incl. shard_spec) is a hashed op attribute, so keep it INDEPENDENT of
    # seq_len -- otherwise varying the input shape would also vary the output attribute and the shape
    # test would no longer isolate input-TensorSpec keying.
    out_shard_spec = ttnn.ShardSpec(grid, [32, QKV.shape[-1] // cores_w], ttnn.ShardOrientation.ROW_MAJOR)
    out_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard_spec)
    in0_t = ttnn.Tensor(QKV_interleaved, dtype).to(ttnn.TILE_LAYOUT).to(device, in_mem_config)

    with device.cache_entries_counter.measure():
        q, k, v = ttnn.experimental.create_qkv_heads(
            in0_t,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            transpose_k_heads=transpose_k,
            memory_config=out_mem_config,
        )

    ref_q = torch.split(QKV, [num_q_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)[0]
    ref_q = torch.reshape(ref_q, [batch, seq_len, num_q_heads, head_dim]).transpose(-3, -2)
    passing, _ = comp_pcc(ref_q, tt2torch_tensor(q), 0.999)
    assert passing, "create_qkv_heads Q output mismatch"


# A known-good block-sharded config from test_create_qkv_heads.py.
CFG = dict(batch=7, seq_len=224, num_q_heads=8, num_kv_heads=8, head_dim=64, cores_h=7, cores_w=8)


@pytest.fixture
def isolate_program_cache(device):
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    yield
    device.disable_and_clear_program_cache()


@skip_for_blackhole("L1 and Circular buffers are crashing on BH, see #12349")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_cqkv_cache_reuse_same_config(device, isolate_program_cache):
    """Same config run twice -> 1 entry."""
    run_cqkv(device, **CFG, transpose_k=True, dtype=ttnn.bfloat16)
    run_cqkv(device, **CFG, transpose_k=True, dtype=ttnn.bfloat16)
    assert device.cache_entries_counter.total == 1


@skip_for_blackhole("L1 and Circular buffers are crashing on BH, see #12349")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_cqkv_cache_miss_transpose_k(device, isolate_program_cache):
    """transpose_k_heads toggled (hashed attribute, same input) -> 2 entries."""
    run_cqkv(device, **CFG, transpose_k=True, dtype=ttnn.bfloat16)
    run_cqkv(device, **CFG, transpose_k=False, dtype=ttnn.bfloat16)
    assert device.cache_entries_counter.total == 2


@skip_for_blackhole("L1 and Circular buffers are crashing on BH, see #12349")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8192}], indirect=True)
def test_cqkv_cache_miss_different_shape(device, isolate_program_cache):
    """Different input shape (seq_len), output attributes held constant -> 2 entries.

    Only the input TensorSpec differs between the two calls (out_mem_config is
    seq_len-independent), so this isolates the default input-shape keying.
    """
    run_cqkv(device, **{**CFG, "seq_len": 224}, transpose_k=True, dtype=ttnn.bfloat16)
    run_cqkv(device, **{**CFG, "seq_len": 384}, transpose_k=True, dtype=ttnn.bfloat16)
    assert device.cache_entries_counter.total == 2
