# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""Component check: TT encoder vs CPU FP32 transformers encoder on the same input, same job.

Prints PCC / max-abs / row NRMSE at the subsampling output and selected conformer layers
(valid frames only), plus the final encoder output.
Usage: python tests/check_encoder_pcc.py [--input /input] [--weights /weights] [--case short]
       [--residual fp32|bf16|both]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def metrics(ref, out):
    ref = ref.double().flatten()
    out = out.double().flatten()
    pcc = torch.corrcoef(torch.stack([ref, out]))[0, 1].item()
    return pcc, (ref - out).abs().max().item()


def row_nrmse(ref, out, lens):
    worst = 0.0
    for b, n in enumerate(lens):
        r, o = ref[b, :n].double(), out[b, :n].double()
        worst = max(worst, (torch.sqrt(((r - o) ** 2).mean()) / torch.sqrt((r ** 2).mean())).item())
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/input")
    ap.add_argument("--weights", default="/weights")
    ap.add_argument("--case", default=None)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--residual", default="fp32", choices=["fp32", "bf16", "both"])
    args = ap.parse_args()

    data = np.load(os.path.join(args.input, "inputs.npz"))
    names = [k[:-5] for k in data.keys() if k.endswith("__mel")]
    name = args.case or names[0]
    mel = data[f"{name}__mel"].astype(np.float32)
    lens = data[f"{name}__mel_lengths"].astype(np.int64)
    mask = (np.arange(mel.shape[1])[None] < lens[:, None]).astype(np.int64)
    print(f"[case] {name} mel={mel.shape} lens={lens.tolist()}", flush=True)

    from transformers import ParakeetForTDT
    model = ParakeetForTDT.from_pretrained(args.weights, dtype=torch.float32).eval()
    ref_taps = {}
    hooks = [model.encoder.subsampling.register_forward_hook(
        lambda m, i, o: ref_taps.__setitem__("subsampling", o.detach().float()))]
    for i, layer in enumerate(model.encoder.layers):
        hooks.append(layer.register_forward_hook(
            lambda m, inp, o, i=i: ref_taps.__setitem__(f"layer{i}", (o[0] if isinstance(o, tuple) else o).detach().float())))
    with torch.inference_mode():
        ref = model.encoder(input_features=torch.from_numpy(mel), attention_mask=torch.from_numpy(mask)).last_hidden_state
    for h in hooks:
        h.remove()

    import ttnn
    import backend as be
    device = ttnn.open_device(device_id=0, **be.DEVICE_OPTIONS)
    try:
        with open(os.path.join(args.weights, "config.json")) as f:
            cfg = json.load(f)
        t0 = time.perf_counter()
        bk = be.create_backend(args.weights, cfg, device, precision="bf16")
        print(f"[load] {time.perf_counter() - t0:.1f}s", flush=True)
        L = bk.cfg.layers
        keys = ["subsampling"] + [f"layer{i}" for i in range(L)
                                  if i % args.every == args.every - 1 or i == 0 or i >= L - 4]
        variants = ["bf16", "fp32"] if args.residual == "both" else [args.residual]
        for var in variants:
            bk.residual_dtype = ttnn.float32 if var == "fp32" else ttnn.bfloat16
            print(f"== residual {var}", flush=True)
            taps = {}
            _, sub_lens, t_out = bk.encode_device(mel, lens, taps=taps)
            vl = [int(n) for n in sub_lens]
            for k in keys:
                r, o = ref_taps[k], taps[k]
                rv = torch.cat([r[b, :n] for b, n in enumerate(vl)])
                ov = torch.cat([o[b, :n] for b, n in enumerate(vl)])
                pcc, mae = metrics(rv, ov)
                print(f"[tap] {k:12s} pcc={pcc:.6f} maxabs={mae:.4f} nrmse={row_nrmse(r, o, vl):.5f}", flush=True)
            out = bk.encode(mel, lens)["encoder"]
            out_t = torch.from_numpy(out)
            print(f"[shape] tt={tuple(out.shape)} ref={tuple(ref.shape)}")
            rv = torch.cat([ref[b, :n] for b, n in enumerate(vl)])
            ov = torch.cat([out_t[b, :n] for b, n in enumerate(vl)])
            pcc, mae = metrics(rv, ov)
            print(f"[final] residual={var} pcc={pcc:.6f} maxabs={mae:.4f} "
                  f"max_row_nrmse={row_nrmse(ref, out_t, vl):.5f}", flush=True)
            for _ in range(2):
                t0 = time.perf_counter()
                bk.encode(mel, lens)
                print(f"[time] residual={var} encode {time.perf_counter() - t0:.3f}s", flush=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
