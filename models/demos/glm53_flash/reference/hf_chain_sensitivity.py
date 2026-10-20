# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How fast the GLM-5.3-Flash stack amplifies small per-layer errors on the host (no device).

The full-model chain test (``tests/test_full_model.py``) feeds every layer the TT output of the
previous one; its per-layer PCC against the dump decays along the stack even though each layer on
its exact HF input scores >= 0.9989 (``tests/test_all_layers.py``). This script measures the model's
OWN sensitivity on the same dump prompt, layer by layer (modules streamed one at a time):

* ``--noise r1,r2,...``: Gaussian noise of relative row size ``r`` (per token row, over the 4x4096
  stream vector; a per-layer PCC of about 1 - r^2 / 2) is added to every layer's output before it
  enters the next layer, i.e. "HF + an r-sized error per layer"; the PCC of each stream vs the dump
  ``x_out`` of the layer is printed per layer (the ``0`` stream is the unperturbed sanity check).
* ``--fp32``: the same chain with every layer in fp32 (weights and activations) against the bf16
  dump: the amplification of HF's own bf16 rounding.
* ``--tt-chain DIR``: the TT chain outputs saved by ``GLM53_FULLMODEL_SAVE`` (``tt_layer<i>_prefill.pt``);
  HF layer i runs on the TT output of layer i - 1 (fresh cache, prefill only) and is compared with the
  TT output of layer i: the per-layer TT error INSIDE the chain (same input on both sides), next to the
  TT-vs-dump PCC of that layer. A per-layer error that stays at the standalone level while the
  chain-vs-dump PCC decays means accumulation / amplification, not a growing TT defect.

    python models/demos/glm53_flash/reference/hf_chain_sensitivity.py --noise 0,0.01,0.02,0.04
    python models/demos/glm53_flash/reference/hf_chain_sensitivity.py --fp32
    python models/demos/glm53_flash/reference/hf_chain_sensitivity.py --tt-chain /path/to/saved
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import default_snapshot_dir, shard_headers  # noqa: E402
from hf_layer_reference import build_layer, load_config  # noqa: E402


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def row_pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.detach().float().reshape(a.shape[1], -1)
    b = b.detach().float().reshape(b.shape[1], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)


def add_row_noise(x: torch.Tensor, r: float, gen: torch.Generator) -> torch.Tensor:
    """x [1, S, 4, H] -> x + noise with per-row norm r * ||row|| (rows = tokens)."""
    if r == 0:
        return x
    xf = x.float()
    S = xf.shape[1]
    rows = xf.reshape(S, -1)
    noise = torch.randn(rows.shape, generator=gen)
    noise = noise / noise.norm(dim=1, keepdim=True) * rows.norm(dim=1, keepdim=True) * r
    return (rows + noise).reshape(xf.shape).to(x.dtype)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, default=Path("bringup/references/glm53_flash/L45_S128"))
    ap.add_argument("--noise", type=str, default="0,0.01,0.02,0.04", help="relative per-layer error sizes")
    ap.add_argument("--fp32", action="store_true", help="fp32 chain vs the bf16 dump instead of noise streams")
    ap.add_argument("--layers", type=int, default=None, help="stop after this many layers")
    ap.add_argument("--tt-chain", type=Path, default=None, help="dir of GLM53_FULLMODEL_SAVE TT chain outputs")
    args = ap.parse_args()
    from transformers import DynamicCache

    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    headers = shard_headers(snapshot)
    n_layers = args.layers or text.num_hidden_layers
    x0 = torch.load(args.dump / "prefill_input.pt")["x0"]  # [1, S, 4, H] bf16 (exact embedding)
    S = x0.shape[1]
    mask = torch.ones(1, S, dtype=torch.bool)
    if args.tt_chain is not None:
        tt_chain(args, snapshot, text, headers, n_layers, x0, mask)
        return
    if args.fp32:
        streams = {"fp32": (x0.float(), DynamicCache(config=text), 0.0)}
    else:
        streams = {f"r={r}": (x0.clone(), DynamicCache(config=text), r) for r in map(float, args.noise.split(","))}
    gens = {k: torch.Generator().manual_seed(1234 + i) for i, k in enumerate(streams)}
    print("layer " + " ".join(f"{k:>18s}" for k in streams) + "   (pcc vs dump x_out, worst row)", flush=True)
    topks = {k: None for k in streams}
    for li in range(n_layers):
        t0 = time.time()
        layer, sd = build_layer(snapshot, text, li, headers)
        del sd
        if args.fp32:
            for p in list(layer.parameters()) + list(layer.buffers()):
                p.data = p.data.float()
        ref = torch.load(args.dump / f"layer{li}_prefill.pt")["x_out"]
        cells = []
        for k, (x, cache, r) in streams.items():
            y, topks[k] = layer(
                x, attention_mask=mask, past_key_values=cache, use_cache=True, prev_topk_indices=topks[k]
            )
            rows = row_pcc(ref, y)
            cells.append(f"{pcc(ref, y):.5f}/{float(rows.min()):.4f}")
            streams[k] = (add_row_noise(y, r, gens[k]), cache, r)
        print(f"{li:5d} " + " ".join(f"{c:>18s}" for c in cells) + f"   {time.time() - t0:.0f}s", flush=True)
        del layer
        gc.collect()


@torch.no_grad()
def tt_chain(args, snapshot, text, headers, n_layers, x0, mask):
    from transformers import DynamicCache

    tt = lambda li: torch.load(args.tt_chain / f"tt_layer{li}_prefill.pt")["x_out"]
    print("layer  HF(TT_in) vs TT_out   TT_out vs dump   HF(TT_in) vs dump   worst row HF(TT_in) vs TT_out", flush=True)
    for li in range(n_layers):
        t0 = time.time()
        layer, sd = build_layer(snapshot, text, li, headers)
        del sd
        x_in = x0 if li == 0 else tt(li - 1)
        y, _ = layer(x_in, attention_mask=mask, past_key_values=DynamicCache(config=text), use_cache=True)
        ref = torch.load(args.dump / f"layer{li}_prefill.pt")["x_out"]
        y_tt = tt(li)
        rows = row_pcc(y, y_tt)
        print(
            f"{li:5d}  {pcc(y, y_tt):.6f}            {pcc(ref, y_tt):.6f}         {pcc(ref, y):.6f}          "
            f"{float(rows.min()):.5f} ({int(rows.argmin())})   {time.time() - t0:.0f}s",
            flush=True,
        )
        del layer
        gc.collect()


if __name__ == "__main__":
    main()
