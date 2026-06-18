# SPDX-License-Identifier: Apache-2.0
"""Per-layer PCC diff: TT prefill activations vs the HF CPU reference oracle.

TT dump  (QWEN36_DUMP_ACTS): {"acts": [N,1,L,H], "names":[...]}  acts[0]=embed, acts[1..64]=layer outputs.
HF dump  (hf_ref.py):        {"layers": [64,1,L,H], "embed"/"embeddings": [1,L,H], "final_norm", "logits"}.

Finds the FIRST layer whose PCC drops below threshold -> isolates the buggy layer.
Run on host (CPU torch) or in the image:
  python3 models/demos/qwen36_27b/t3k/pcc_diff.py [tt.pt] [hf.pt]
"""
import sys
import torch


def pcc(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    if a.shape != b.shape:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a @ b) / denom)


def main():
    tt_path = sys.argv[1] if len(sys.argv) > 1 else "/home/yito/work/tt_acts.pt"
    hf_path = sys.argv[2] if len(sys.argv) > 2 else "/home/yito/work/hf_ref_acts.pt"
    tt = torch.load(tt_path, map_location="cpu")
    hf = torch.load(hf_path, map_location="cpu")

    tt_acts = tt["acts"]  # [N,1,L,H]; [0]=embed, [1..]=layers
    tt_names = tt["names"]
    hf_layers = hf["layers"]  # [64,1,L,H]
    hf_embed = hf.get("input_embeds", hf.get("embed", hf.get("embeddings")))

    print(f"TT: {len(tt_names)} tensors, shape {tuple(tt_acts.shape)}")
    print(f"HF: layers {tuple(hf_layers.shape)}\n")

    prev = None
    if hf_embed is not None:
        prev = pcc(tt_acts[0], hf_embed)
        print(f"{'embed':>16}  PCC={prev:.6f}")

    # bf8 weights cause gradual PCC decay even when correct; a BUG shows as a sharp
    # single-layer drop. Report each layer's PCC and the drop from the previous.
    n = min(tt_acts.shape[0] - 1, hf_layers.shape[0])
    drops = []
    for i in range(n):
        p = pcc(tt_acts[i + 1], hf_layers[i])
        drop = (prev - p) if prev is not None else 0.0
        drops.append((drop, i, p))
        mark = "  <== BIG DROP" if drop > 0.05 else ""
        print(f"{tt_names[i+1]:>16}  PCC={p:.6f}  d={drop:+.6f}{mark}")
        prev = p

    print()
    drops.sort(reverse=True)
    print("Largest single-layer PCC drops (bug localizes to the layer where the drop happens):")
    for drop, i, p in drops[:5]:
        print(f"  layer {i:2d} ({tt_names[i+1]}):  PCC={p:.6f}  drop={drop:+.6f}")


if __name__ == "__main__":
    main()
