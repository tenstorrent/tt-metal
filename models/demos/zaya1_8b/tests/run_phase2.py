"""Phase 2 PCC validation: ZAYA1 MoE block (router + experts + MoD), layer 1.

Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_phase2.py
"""
import os
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt import standard as S
from models.demos.zaya1_8b.tt.moe import ZayaMoEBlock

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")
results = []


def g(name):
    return torch.load(os.path.join(GOLDEN, f"{name}.pt"), weights_only=False)


def rec(name, ok, pcc):
    results.append((name, ok, pcc))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} pcc={pcc}")


def main():
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    try:
        w = ZayaWeights()
        block = ZayaMoEBlock(device, w, layer=1)

        zb = g("L1_zaya_block")
        hidden_t = zb["in"][0]                       # [1,6,2048] post input_norm
        hidden = S.to_dev(hidden_t, device)
        out_tt, rnext_tt = block.forward(hidden)

        # 1) block output vs golden expert_output
        ok, pcc = comp_pcc(zb["out"][0].float(), ttnn.to_torch(out_tt).float().reshape(zb["out"][0].shape), 0.98)
        rec("moe_block_out", ok, pcc)

        # 2) router_next (pre-norm down output) vs golden
        ok, pcc = comp_pcc(zb["out"][2].float(), ttnn.to_torch(rnext_tt).float().reshape(zb["out"][2].shape), 0.99)
        rec("router_next", ok, pcc)

        # 3) route_prob + choice vs golden router
        r = g("L1_router")
        gate, _ = block.router.forward(hidden)
        gate_t = ttnn.to_torch(gate).float().reshape(1, -1, 17)
        route_prob = gate_t.sum(-1).reshape(-1)      # chosen prob per token
        choice = gate_t.argmax(-1).reshape(-1)
        gp = r["out"][0].float().reshape(-1)
        gc = r["out"][1].reshape(-1)
        ok_p, pcc_p = comp_pcc(gp, route_prob, 0.98)
        rec("route_prob", ok_p, pcc_p)
        ch_match = bool((choice == gc).all())
        rec("expert_choice(exact)", ch_match, f"{choice.tolist()} vs {gc.long().tolist()}")
    finally:
        ttnn.close_mesh_device(device)

    npass = sum(1 for _, ok, _ in results if ok)
    print(f"\n=== Phase 2 summary: {npass}/{len(results)} passed ===")
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
