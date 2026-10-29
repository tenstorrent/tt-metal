# SPDX-License-Identifier: Apache-2.0
"""Device probe: union state backing (snapshot rows inside attention K/V pages).

Synthetic pool (3 groups x G recurrent layers, G attention layers) on the TP4 mesh:
exact FP32/BF16 save/restore through BFP8 pages, untouched blocks unchanged,
scrubbed blocks finite, warmup leaves the live state and null block clean.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import ttnn

from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import LinearState, PagedState
from models.autoports.qwen_qwen3_8_27b.tt.serving_state import ServingStatePool


def host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=int, default=1536)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = dict(block=a.block, blocks=a.blocks, group=a.group, checks={})
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        g = torch.Generator().manual_seed(7)
        G, B, N = a.group, a.batch, a.blocks
        kinds = []
        for _ in range(G):
            kinds += ["linear_attention"] * 3 + ["full_attention"]
        layers = [SimpleNamespace(layer_type=k) for k in kinds]
        model = SimpleNamespace(mesh=mesh, layers=layers, layer_indices=list(range(len(layers))))
        dram = ttnn.DRAM_MEMORY_CONFIG
        rep = ttnn.ReplicateTensorToMesh(mesh)

        def live_state():
            conv = torch.randn(B, 3, 2560, generator=g).bfloat16()
            rec = torch.randn(B, 12, 128, 128, generator=g) * torch.exp(torch.randn(B, 12, 1, 1, generator=g) * 4)
            return LinearState(
                ttnn.from_torch(conv, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=rep),
                ttnn.from_torch(rec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=rep))

        pages = N * a.block // 32

        def attn():
            # Random finite BFP8 content stands in for live attention data.
            return PagedState(*[ttnn.from_torch(torch.randn(pages, 1, 32, 256, generator=g), dtype=ttnn.bfloat8_b,
                                                layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=dram, mesh_mapper=rep)
                                for _ in range(2)])

        states = [live_state() if k == "linear_attention" else attn() for k in kinds]
        rec_idx = [i for i, k in enumerate(kinds) if k == "linear_attention"]
        attn_idx = [i for i, k in enumerate(kinds) if k == "full_attention"]
        groups = [rec_idx[j::3] for j in range(3)]  # equal partition, G layers each
        pool = ServingStatePool(model, states, num_blocks_by_layer={i: N for i in rec_idx},
                                sharing_groups=groups, union_block_size=a.block)
        T = a.block // 32  # 32-token pages per block
        attn_before = {i: (host(states[i].key), host(states[i].value)) for i in attn_idx}
        live_before = {i: (host(states[i].conv), host(states[i].recurrent)) for i in rec_idx}
        pool.warmup()
        out["checks"]["warmup_live_exact"] = all(
            torch.equal(host(states[i].conv), live_before[i][0]) and torch.equal(host(states[i].recurrent), live_before[i][1])
            for i in rec_idx)
        out["checks"]["warmup_other_blocks_unchanged"] = all(
            torch.equal(host(states[i].key)[T:], attn_before[i][0][T:]) and torch.equal(host(states[i].value)[T:], attn_before[i][1][T:])
            for i in attn_idx)
        out["checks"]["warmup_null_block_finite"] = all(
            torch.isfinite(host(states[i].key)[:T]).all() and torch.isfinite(host(states[i].value)[:T]).all() for i in attn_idx)
        out["checks"]["warmup_null_block_state_region_zero"] = all(
            (host(states[i].key)[:T] == 0).all() for i in attn_idx)

        # Save slot 1 of every group into blocks 2,3,4; restore into slot 3; compare bits.
        ids = {}
        for gi, grp in enumerate(groups):
            for i in grp:
                ids[i] = 2 + gi
        pool.save(1, ids)
        saved = {i: (host(states[i].conv)[1].clone(), host(states[i].recurrent)[1].clone()) for i in rec_idx}
        out["checks"]["save_untouched_blocks_unchanged"] = all(
            torch.equal(host(states[i].key)[5 * T:], attn_before[i][0][5 * T:])
            and torch.equal(host(states[i].key)[T:2 * T], attn_before[i][0][T:2 * T]) for i in attn_idx)
        pool.restore(3, ids)
        out["checks"]["restore_exact"] = all(
            torch.equal(host(states[i].conv)[3].view(torch.int16), saved[i][0].view(torch.int16))
            and torch.equal(host(states[i].recurrent)[3].view(torch.int32), saved[i][1].view(torch.int32)) for i in rec_idx)
        out["checks"]["restore_other_slots_unchanged"] = all(
            torch.equal(host(states[i].recurrent)[[0, 1, 2]], live_before[i][1][[0, 1, 2]]) for i in rec_idx)
        out["checks"]["groups_share_ordinal_targets"] = len({id(pool.union_targets[i]) for i in rec_idx}) == G

        # Attention claims block 3: scrubbed to finite values, restore from it rejected.
        table = torch.tensor([[1, 3, 0]], dtype=torch.int32)
        pool.claim_for_attention(table)
        out["checks"]["claimed_block_finite"] = all(
            torch.isfinite(host(states[i].key)[3 * T:4 * T]).all() and torch.isfinite(host(states[i].value)[3 * T:4 * T]).all()
            for i in attn_idx)
        out["checks"]["claimed_block_key_zero"] = all((host(states[i].key)[3 * T:4 * T] == 0).all() for i in attn_idx)
        out["checks"]["unclaimed_state_blocks_tracked"] = pool.state_blocks == {2, 4}
        # Informational: FP32 state bytes read as BFP8 (why claims must scrub).
        out["state_block_nonfinite_values_as_bfp8"] = int(sum((~torch.isfinite(host(states[i].key)[2 * T:3 * T])).sum() for i in attn_idx))
        try:
            pool.restore(0, ids)
            out["checks"]["stale_restore_rejected"] = False
        except ValueError:
            out["checks"]["stale_restore_rejected"] = True
        out["counters"] = pool.counters
        out["status"] = "pass" if all(out["checks"].values()) else "fail"
    except Exception as e:
        out["status"] = "fail"
        out["error"] = repr(e)
        raise
    finally:
        a.output.write_text(json.dumps(out, indent=1, default=str))
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
