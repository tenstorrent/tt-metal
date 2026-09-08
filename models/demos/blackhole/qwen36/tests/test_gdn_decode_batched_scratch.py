# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH: batched GDN decode, original ttnn path vs fused op (QWEN36_GDN_DECODE_FUSED=2) on the real layer-0 GDN
module with a 32-slot state, B = 1..32 active users: output / state agreement + trace-timed per-call time.

  QWEN36_GDN_DECODE_FUSED=2 pytest models/demos/blackhole/qwen36/tests/test_gdn_decode_batched_scratch.py -s
"""
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": 256 * 1024 * 1024,
    }
]
BMAX = 32


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _trace_us(mesh, fn, k=8, reps=3):
    fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = [fn() for _ in range(k)]
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = min(best, time.perf_counter() - t0)
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    return best / k * 1e6


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_gdn_decode_batched(mesh_device):
    mesh = mesh_device
    mesh.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=BMAX, max_seq_len=4096, layer_indices=[0, 1, 2, 3])
    gdn = model.layers[0].attention
    assert not model.layers[0].is_full_attention
    # the batched (Bmax-slot) GDN state is allocated with the KV caches, as in serving
    num_blocks = 64
    kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, 64, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=BMAX)
    assert gdn.rec_state.shape[0] == BMAX, gdn.rec_state.shape
    dim = model.args.dim
    torch.manual_seed(0)
    comp = ttnn.ConcatMeshToTensor(mesh, dim=3)

    def randomize_states():
        rec = 0.05 * torch.randn(BMAX, gdn.Nv * mesh.get_num_devices(), gdn.Dk, gdn.Dv)  # per-device heads along dim 1
        r = ttnn.from_torch(
            rec,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )
        ttnn.copy(r, gdn.rec_state)
        ttnn.deallocate(r)
        for m in range(gdn.K):
            c = 0.5 * torch.randn(1, BMAX, gdn.qkv_dim_tp * mesh.get_num_devices())
            t = ttnn.from_torch(
                c,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
            )
            ttnn.copy(t, gdn.conv_states[m])
            ttnn.deallocate(t)
        gdn._hist_packed_valid = False

    def snapshot():
        return ttnn.clone(gdn.rec_state), [ttnn.clone(c) for c in gdn.conv_states]

    def restore(snap):
        ttnn.copy(snap[0], gdn.rec_state)
        for m in range(gdn.K):
            ttnn.copy(snap[1][m], gdn.conv_states[m])
        gdn._hist_packed_valid = False

    for B in (1, 2, 4, 8, 16, 32):
        x = ttnn.from_torch(
            torch.randn(1, 1, B, dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        randomize_states()
        snap = snapshot()
        # original path
        gdn._decode_fused_conv, gdn._decode_fused = False, False
        out_ref = ttnn.to_torch(gdn.forward_decode(x), mesh_composer=comp).float().reshape(B, dim)
        rec_ref = ttnn.to_torch(gdn.rec_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).float()
        conv_ref = [
            ttnn.to_torch(c, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=2)).float() for c in gdn.conv_states
        ]
        # fused path from the same starting state
        restore(snap)
        gdn._decode_fused_conv, gdn._decode_fused = True, True
        out_f = ttnn.to_torch(gdn.forward_decode(x), mesh_composer=comp).float().reshape(B, dim)
        rec_f = ttnn.to_torch(gdn.rec_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).float()
        # fused path keeps the history packed: compare it with the original path's shifted conv_states, re-packed
        hist_f = ttnn.to_torch(ttnn.get_device_tensors(gdn.conv_hist_packed)[0])  # device 0: [Bmax, Nv, 4, 32, 32]
        C = gdn.qkv_dim_tp
        conv_ref_dev0 = [c[:, :, :C] for c in conv_ref]  # device 0's channel shard
        exp_slots = torch.stack(
            [gdn._pack_head_tiles([conv_ref_dev0[j][0, b] for j in range(4)], parity=b & 1) for b in range(B)]
        )
        hist_ok = torch.allclose(hist_f[:B].float(), exp_slots.float(), atol=0, rtol=0)
        pcc_out = min(_pcc(out_ref[b], out_f[b]) for b in range(B))
        pcc_rec = min(_pcc(rec_ref[b], rec_f[b]) for b in range(B))
        untouched = torch.equal(rec_ref[B:], rec_f[B:]) if B < BMAX else True
        # timing (trace) of both paths
        gdn._decode_fused_conv, gdn._decode_fused = False, False
        us_ref = _trace_us(mesh, lambda: gdn.forward_decode(x))
        gdn._decode_fused_conv, gdn._decode_fused = True, True
        gdn._ensure_conv_hist_packed()
        us_f = _trace_us(mesh, lambda: gdn.forward_decode(x))
        print(
            f"GDN_BATCHED B={B}: out pcc(min over users)={pcc_out:.6f} rec pcc={pcc_rec:.6f} history exact={hist_ok} "
            f"untouched={untouched} | original {us_ref:.0f} us  fused {us_f:.0f} us  ({us_ref / us_f:.2f}x)",
            flush=True,
        )
        assert pcc_out > 0.999 and pcc_rec > 0.9999 and untouched, (B, pcc_out, pcc_rec, untouched)
        ttnn.deallocate(x)
