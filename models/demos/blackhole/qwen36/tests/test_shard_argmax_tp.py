# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shard-argmax greedy pick must equal the gathered argmax on every device.
Pins first/last shard, cross-shard ties, in-shard ties, and all-negative rows so padding cannot win."""
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

from .test_factory import parametrize_mesh_tp

# Qwen3.6-27B (248320) and Qwen3.5-9B (151936) both divide by 8; the second also divides by 4.
VOCABS = [248320, 151936]


def _scalar_per_device(t):
    out = []
    for d in ttnn.get_device_tensors(t):
        f = d.to_list()
        while isinstance(f, list):
            f = f[0]
        out.append(int(f))
    return out


def _cases(vocab, shard):
    """name -> host [1,1,1,vocab] fp32 row."""
    torch.manual_seed(0)
    cases = {"random": torch.randn(1, 1, 1, vocab, dtype=torch.float32)}

    def flat(scale=0.1):
        return torch.randn(1, 1, 1, vocab, dtype=torch.float32) * scale

    t = flat()
    t[0, 0, 0, 7] = 50.0
    cases["first_shard"] = t
    t = flat()
    t[0, 0, 0, vocab - 5] = 50.0
    cases["last_shard"] = t
    t = flat()
    for p in (2 * shard + 11, 5 * shard + 3, vocab - 100):
        t[0, 0, 0, p] = 50.0
    cases["tie_across_shards"] = t
    t = flat()
    t[0, 0, 0, shard + 100] = 50.0
    t[0, 0, 0, shard + 900] = 50.0
    cases["tie_within_shard"] = t
    cases["all_negative"] = -torch.rand(1, 1, 1, vocab, dtype=torch.float32) - 1.0
    return cases


@torch.no_grad()
@pytest.mark.parametrize("vocab", VOCABS)
@parametrize_mesh_tp()
def test_shard_argmax_matches_gathered(mesh_device, vocab):
    """Shard-combined pick == gathered argmax == torch argmax, on every device."""
    tp = mesh_device.get_num_devices()
    if tp == 1:
        pytest.skip("shard argmax is a mesh path; a single device never gathers")
    if vocab % tp:
        pytest.skip(f"vocab {vocab} does not fracture evenly over {tp} devices")
    shard = vocab // tp
    topo = ttnn.Topology.Linear

    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device)
    offsets = tpc.vocab_shard_offsets(mesh_device, tp, shard)

    for name, host in _cases(vocab, shard).items():
        ref = int(torch.argmax(host[0, 0, 0]))
        sharded = ttnn.from_torch(
            host,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
        )
        replicated = ttnn.from_torch(
            host,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        gathered = _scalar_per_device(tpc.greedy_pick(replicated, mesh_device, tt_ccl, topo))
        combined = _scalar_per_device(
            tpc.greedy_pick(sharded, mesh_device, tt_ccl, topo, shard_offsets=offsets, vocab_size=vocab)
        )
        ttnn.deallocate(sharded)
        ttnn.deallocate(replicated)

        assert len(set(combined)) == 1, f"{name}: devices disagree on the pick: {combined}"
        assert combined[0] == ref, f"{name}: shard pick {combined[0]} != torch argmax {ref}"
        assert gathered[0] == ref, f"{name}: gathered pick {gathered[0]} != torch argmax {ref}"
