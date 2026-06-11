# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for the Metal descriptor slow-path-rebuild perf bug (#46506) on the
sharded path of nlp_create_qkv_heads.

The sharded program factory bakes raw q/k/v base + per-core start addresses into the
reader/writer runtime args. Those addresses are derived from the *input* (and optional kv)
buffer address, so they change whenever the buffer is re-allocated. Without a
get_dynamic_runtime_args() declaration, the framework falls back to re-running
create_descriptor() on every cache hit (slow path).

We set TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT=1 BEFORE importing ttnn so the
adapter raises instead of silently rebuilding. We then run the op 3x, each time with a
FRESH input tensor (so the buffer address changes across cache hits), and assert:
  - no slow-path-rebuild exception is raised, and
  - q/k/v match a torch reference split (PCC).

NOTE on nlp_create_qkv_heads_boltz: its sharded path is DEFERRED here. The boltz sharded
output spec gives each head an output volume of seq*seq*head_dim and shards it as
{TILE_HEIGHT, head_dim} per core (num_cores_to_corerangeset over num_q_heads). For any
tile-aligned seq (>= 32) this produces num_q_heads * (seq*seq / TILE_HEIGHT) shards, which
far exceeds the available cores (e.g. seq=32 -> 32 shards/head), so a valid single-device
sharded call cannot be constructed. The boltz get_dynamic_runtime_args fix is identical to
the base op's and exercises the same code, but is left without direct test coverage here.
"""

import os

os.environ["TT_METAL_FORBID_DESCRIPTOR_REBUILD_ON_CACHE_HIT"] = "1"

import pytest
import torch

import ttnn


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _reference_split(t, batch, seq, num_q, num_kv, head_dim):
    """Torch reference matching the sharded input layout.

    Input width is laid out per shard-group g (one group per kv head):
      [ num_q/num_kv q-heads | 1 k-head | 1 v-head ]
    """
    qpg = num_q // num_kv
    shard_w = (qpg + 2) * head_dim
    ref_q, ref_k, ref_v = [], [], []
    for g in range(num_kv):
        base = g * shard_w
        for h in range(qpg):
            ref_q.append(t[:, :, :, base + h * head_dim : base + (h + 1) * head_dim])
        ref_k.append(t[:, :, :, base + qpg * head_dim : base + (qpg + 1) * head_dim])
        ref_v.append(t[:, :, :, base + (qpg + 1) * head_dim : base + (qpg + 2) * head_dim])
    return (
        torch.cat(ref_q, dim=1),  # [batch, num_q, seq, head_dim]
        torch.cat(ref_k, dim=1),  # [batch, num_kv, seq, head_dim]
        torch.cat(ref_v, dim=1),
    )


def test_nlp_create_qkv_heads_sharded_no_descriptor_rebuild(device):
    batch, seq = 1, 32
    num_q, num_kv, head_dim = 8, 2, 32
    qpg = num_q // num_kv
    shard_w = (qpg + 2) * head_dim
    width = num_kv * shard_w

    # Input is WIDTH_SHARDED across num_kv cores; each shard holds one kv-group's
    # q-heads + k + v. Output is HEIGHT_SHARDED.
    in_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_kv - 1, 0))])
    out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)

    seen_addrs = set()
    for _ in range(3):
        t = torch.randn([batch, 1, seq, width], dtype=torch.float32)
        shard_spec = ttnn.ShardSpec(in_grid, [batch * seq, shard_w], ttnn.ShardOrientation.ROW_MAJOR)
        in_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)
        tt_in = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_mem)
        seen_addrs.add(tt_in.buffer_address())

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            tt_in,
            num_heads=num_q,
            num_kv_heads=num_kv,
            transpose_k_heads=False,
            memory_config=out_mem,
        )
        ttnn.synchronize_device(device)

        ref_q, ref_k, ref_v = _reference_split(t, batch, seq, num_q, num_kv, head_dim)
        qt, kt, vt = ttnn.to_torch(q), ttnn.to_torch(k), ttnn.to_torch(v)

        assert list(qt.shape) == [batch, num_q, seq, head_dim]
        assert list(kt.shape) == [batch, num_kv, seq, head_dim]
        assert list(vt.shape) == [batch, num_kv, seq, head_dim]
        assert _pcc(qt, ref_q) > 0.999, "Q output mismatch on sharded cache-hit dispatch"
        assert _pcc(kt, ref_k) > 0.999, "K output mismatch on sharded cache-hit dispatch"
        assert _pcc(vt, ref_v) > 0.999, "V output mismatch on sharded cache-hit dispatch"


@pytest.mark.skip(
    reason="nlp_create_qkv_heads_boltz sharded path deferred: its sharded output spec "
    "(num_q_heads * seq*seq/TILE_HEIGHT shards of {TILE_HEIGHT, head_dim}) cannot fit on the "
    "available cores for any tile-aligned seq, so a valid single-device sharded call cannot be "
    "constructed. The get_dynamic_runtime_args fix is identical to the base op (covered above)."
)
def test_nlp_create_qkv_heads_boltz_sharded_no_descriptor_rebuild(device):
    pass
