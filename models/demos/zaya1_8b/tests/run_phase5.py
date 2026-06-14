"""Phase 5 PCC validation: full 80-layer ZAYA1-8B prefill forward vs golden.

Compares per-layer hidden_states at several depths + final logits + top-1 token.
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_phase5.py
"""
import os
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt.model import ZayaModel

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
        inputs = g("inputs")
        input_ids = inputs["input_ids"]
        S = input_ids.shape[1]
        hs_golden = g("hidden_states.pt".replace(".pt", ""))   # list of 81 [1,S,2048]

        print("building model (loads ~17GB to device)...")
        model = ZayaModel(device, S)
        cap = {}
        logits_tt = model.forward(input_ids, capture_hidden=cap)

        # per-layer hidden (INFORMATIONAL vs fp32 golden: top-1 MoE routing can flip individual
        # tokens at intermediate layers under bf16; the meaningful criteria are logits + top-1).
        for k in [1, 2, 4, 10, 40, 79]:
            _, pcc = comp_pcc(hs_golden[k].float(), cap[k].reshape(hs_golden[k].shape), 0.9)
            print(f"  (info) hidden[{k}] pcc={pcc}")

        # final logits + top-1 token
        lg = g("logits")["out"].float()
        got = ttnn.to_torch(logits_tt).float().reshape(lg.shape)
        ok, pcc = comp_pcc(lg, got, 0.95)
        rec("logits", ok, pcc)
        top_g = lg[0, -1].argmax().item()
        top_t = got[0, -1].argmax().item()
        rec("top1_token(exact)", top_g == top_t, f"golden={top_g} got={top_t}")
        # diagnostics on the last-token logits
        lgl, gotl = lg[0, -1], got[0, -1]
        print("  golden top5:", torch.topk(lgl, 5).indices.tolist(), [round(v, 3) for v in torch.topk(lgl, 5).values.tolist()])
        print("  got    top5:", torch.topk(gotl, 5).indices.tolist(), [round(v, 3) for v in torch.topk(gotl, 5).values.tolist()])
        print(f"  golden logit@9079={lgl[9079]:.3f} @528={lgl[528]:.3f} | got logit@9079={gotl[9079]:.3f} @528={gotl[528]:.3f}")
        print(f"  rank of golden-top({top_g}) in got: {(gotl > gotl[top_g]).sum().item()}; rank of got-top({top_t}) in golden: {(lgl > lgl[top_t]).sum().item()}")
        print(f"  per-position argmax golden={lg[0].argmax(-1).tolist()} got={got[0].argmax(-1).tolist()}")
        print(f"  last-token logits pcc={comp_pcc(lgl, gotl, 0.9)[1]}")
    finally:
        ttnn.close_mesh_device(device)

    npass = sum(1 for _, ok, _ in results if ok)
    print(f"\n=== Phase 5 summary: {npass}/{len(results)} passed ===")
    raise SystemExit(0 if npass == len(results) else 1)


if __name__ == "__main__":
    main()
