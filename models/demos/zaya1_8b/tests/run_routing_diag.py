"""Diagnose device-bf16 vs bf16-CPU routing divergence: find first MoE layer/token
where the device picks a different top-1 expert than the bf16-CPU reference.
Run: TT_DEVICE=1 /home/yito/work/run_zaya.sh python models/demos/zaya1_8b/tests/run_routing_diag.py
"""
import os
import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.demos.zaya1_8b.tt.model_args import ZayaWeights
from models.demos.zaya1_8b.tt.model import ZayaModel

GOLDEN = os.path.join(os.path.dirname(__file__), "..", "reference", "golden")


def main():
    ids = torch.load(os.path.join(GOLDEN, "inputs.pt"), weights_only=False)["input_ids"]
    bf16_choices = torch.load(os.path.join(GOLDEN, "bf16_router_choices.pt"), weights_only=False)
    bf16_hs = torch.load(os.path.join(GOLDEN, "bf16_hidden_states.pt"), weights_only=False)
    S = ids.shape[1]

    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape((1, 1)))
    try:
        model = ZayaModel(device, S, w=ZayaWeights(), verbose=False)
        cap_h, cap_c = {}, {}
        model.forward(ids, capture_hidden=cap_h, capture_choices=cap_c)

        print("=== per-MoE-layer expert choice: device vs bf16-CPU (mismatches only) ===")
        first_div = None
        for layer in sorted(cap_c):
            dev = cap_c[layer].long()
            ref = bf16_choices[layer].long()
            mism = (dev != ref)
            if mism.any():
                idxs = mism.nonzero().reshape(-1).tolist()
                print(f"  layer {layer}: dev={dev.tolist()} ref={ref.tolist()}  mismatch@{idxs}")
                if first_div is None:
                    first_div = layer
        print(f"\nfirst divergent MoE layer: {first_div}")

        print("\n=== device-vs-bf16CPU hidden pcc (last token) at key layers ===")
        for k in [1, 2, 10, 20, 30, 39, 40, 50, 79]:
            if k in cap_h:
                gg = bf16_hs[k].float().reshape(1, -1, 2048)
                cc = cap_h[k].reshape(1, -1, 2048)
                _, allp = comp_pcc(gg, cc, 0.9)
                _, lastp = comp_pcc(gg[0, -1], cc[0, -1], 0.9)
                print(f"  hidden[{k}] all={round(allp,4)} lasttok={round(lastp,4)}")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
