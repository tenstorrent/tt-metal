# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Chunked indexer consistency (step 4 slice 2): running _indexer_topk in chunks
with the host K-cache must select the same indices as one single-shot call —
chunking is an implementation detail, not a model change. Chunk size is a
parameter (1k dev default per agreement 15).
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v32.tests.test_mla import build_cpu_reference
from models.demos.deepseek_v32.tt import ops
from models.demos.deepseek_v32.tt.mla import ttMLA


def _shard(t, mesh_device):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, -1)),
    )


@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize("seq_len,chunk", [(2048, 1024)], ids=["2k_c1k"])
def test_indexer_chunked_matches_single_shot(mesh_device, seq_len, chunk, config_only):
    config = config_only
    args, _, weights = build_cpu_reference(seq_len)
    config.max_seq_len = seq_len

    torch.manual_seed(7)
    x = torch.randn(1, 1, seq_len, config.hidden_size, dtype=torch.bfloat16)

    def make_mla():
        return ttMLA(config, dict(weights), mesh_device, layer_idx=0, seq_len=seq_len, sp_axis=0, tp_axis=1)

    single = make_mla()
    idx_single = ops._to_host(single._indexer_topk(_shard(x, mesh_device), seq_len)).long()[0, 0]

    chunked = make_mla()
    rows = []
    for s in range(0, seq_len, chunk):
        idx = chunked._indexer_topk(_shard(x[:, :, s : s + chunk], mesh_device), chunk, start_pos=s)
        rows.append(ops._to_host(idx).long()[0, 0])  # [chunk, k<=topk]

    k = args.index_topk
    for s, r in zip(range(0, seq_len, chunk), rows):
        for i in range(r.shape[0]):
            pos = s + i
            want = set(idx_single[pos][idx_single[pos] <= pos].tolist())
            got = set(r[i][r[i] <= pos].tolist())
            n = min(pos + 1, k)
            assert len(want & got) >= n - max(2, n // 100), f"row {pos}: overlap {len(want & got)}/{n}"
