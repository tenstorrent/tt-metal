# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Chunked indexer consistency (step 4 slice 2): running _indexer_topk in chunks
with the host K-cache must select the same indices as one single-shot call —
chunking is an implementation detail, not a model change. Chunk size is a
parameter (1k dev default per agreement 15).
"""

import pytest
from ttnn.device import is_blackhole

import ttnn
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from models.demos.deepseek_v32.tests.mesh_utils import parametrize_mesh_device_tp_only
from models.demos.deepseek_v32.tests.test_mla import build_cpu_reference, make_hidden
from models.demos.deepseek_v32.tt import ops
from models.demos.deepseek_v32.tt.mla import ttMLA

pytestmark = pytest.mark.dev  # indexer self-consistency, no CPU truth — inner loop


def _shard(t, mesh_device):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(None, -1)),
    )


# TP-only: this checks chunked vs single-shot indexer selection (a chunking property,
# independent of SP). The indexer is replicated under TP; its SP layout is covered e2e
# in test_mla. Sweeping SP here would only re-read replicated shards through the flat
# row layout this test assumes.
@parametrize_mesh_device_tp_only()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,  # device indexer stems do TP all-reduce
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        }
    ],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len,chunk", [(2048, 1024)], ids=["2k_c1k"])
def test_indexer_chunked_matches_single_shot(
    mesh_device, seq_len, chunk, device_params, config_only, ds_layer, ds_checkpoint, ds_repo, ds_input
):
    config = config_only
    args, _, weights, _ = build_cpu_reference(seq_len, layer=ds_layer, checkpoint_path=ds_checkpoint, repo=ds_repo)
    config.max_seq_len = seq_len

    x = make_hidden(seq_len, config.hidden_size, seed=7, input_path=ds_input).unsqueeze(0)  # [1,1,seq,hidden]

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
