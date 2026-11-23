# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Inkling-2-Mini DSpark drafter (`mtp.safetensors`) -> tt weight cache, §4 style, under ``<cache>/mtp/``.

    python -m models.demos.inkling2_mini.convert.mtp [--ckpt DIR] [--out DIR]

Own ``manifest.json`` (schema 1, every entry ``layer: None``) so ``cache.load_global(<cache>/mtp)`` loads it and
the main v1 manifest is untouched. Files ``global/mtp/<name>.tensorbin``. Matmul weights are ``[1,1,in,out]``.
Derived tensors (host fp64, as vLLM ``Inkling2MarkovHead.fold``): ``alpha`` = softmax(fuse_weight) rows,
``fold_w`` = (unembed @ hidden_out.weight)^T, ``fold_b`` = unembed @ hidden_out.bias (pad cols -1e4).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

import ttnn
from models.demos.inkling2_mini.convert.convert import DTYPES, LAYOUTS, VOCAB_REAL, VOCAB_TT, roundtrip_pcc, rows32

HIDDEN, RANK, N_LAYERS = 2304, 256, 3
BIG = "BFP8"  # drafter matmul weights; acceptance only, never the output tokens


def mm(t):
    """torch [out, in] -> [1, 1, in, out]."""
    return t.float().T.contiguous()[None, None]


def tensors(ckpt: Path):
    """Yield (name, torch tensor, dtype, layout)."""
    with safe_open(ckpt / "mtp.safetensors", "pt") as f:
        g = lambda k: f.get_tensor(k)
        for i in range(N_LAYERS):
            p = f"layers.{i}."
            a = p + "self_attn."
            wqkv = torch.cat([g(a + "q_proj.weight"), g(a + "k_proj.weight"), g(a + "v_proj.weight")])
            yield f"l{i}.wqkv", mm(wqkv), BIG, "TILE"
            yield f"l{i}.wkv_ctx", mm(torch.cat([g(a + "k_proj_ctx.weight"), g(a + "v_proj_ctx.weight")])), BIG, "TILE"
            yield f"l{i}.wo", mm(g(a + "o_proj.weight")), BIG, "TILE"
            yield f"l{i}.w_gate_up", mm(
                torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")])
            ), BIG, "TILE"
            yield f"l{i}.w_down", mm(g(p + "mlp.down_proj.weight")), BIG, "TILE"
            for n, k in (
                ("in_norm", "input_layernorm"),
                ("post_norm", "post_attention_layernorm"),
                ("ctx_norm", "dflare_norm"),
            ):
                yield f"l{i}.{n}", g(p + k + ".weight").reshape(72, 32), "BF16", "RM"
            for n in ("q_norm", "k_norm", "k_norm_ctx"):
                yield f"l{i}.{n}", g(a + n + ".weight").reshape(4, 32), "BF16", "RM"
        yield "norm", g("norm.weight").reshape(72, 32), "BF16", "RM"
        yield "rnn_norm", g("markov_head.dflare_norm.weight").reshape(72, 32), "BF16", "RM"
        yield "mask_embed", rows32(g("mask_embed")), "BF16", "TILE"
        alpha = torch.softmax(g("fuse_weight").double(), -1)  # [layers + 1, taps]
        yield "alpha", F.pad(alpha.float(), (0, 32 - alpha.shape[1], 0, 32 - alpha.shape[0])), "FP32", "TILE"
        yield "markov_w1", g("markov_head.markov_w1.weight"), "BF16", "RM"
        rw = g("markov_head.rnn_proj.weight")  # [3R, 2R + 2H]: cols [state | prev | hidden | f_rnn]
        yield "rnn_w_sp", mm(rw[:, : 2 * RANK]), "BF16", "TILE"
        yield "rnn_w_h", mm(rw[:, 2 * RANK : 2 * RANK + HIDDEN]), "BF16", "TILE"
        yield "rnn_w_f", mm(rw[:, 2 * RANK + HIDDEN :]), "BF16", "TILE"
        yield "rnn_b", rows32(g("markov_head.rnn_proj.bias")), "FP32", "TILE"
        ho_w, ho_b = g("markov_head.hidden_out.weight").double(), g("markov_head.hidden_out.bias").double()
    with safe_open(ckpt / "model-00001-of-00001.safetensors", "pt") as f:
        un = f.get_tensor("model.llm.unembed.weight")[:VOCAB_REAL].double()
    fold_w = F.pad((un @ ho_w).T, (0, VOCAB_TT - VOCAB_REAL)).float()  # [R, V_tt]
    fold_b = F.pad((un @ ho_b), (0, VOCAB_TT - VOCAB_REAL), value=-1.0e4).float()
    del un
    yield "fold_w", fold_w[None, None], BIG, "TILE"
    yield "fold_b", fold_b.view(1, 1, 1, -1), "FP32", "TILE"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", type=Path, default=Path(os.environ.get("INKLING_CKPT", "/work/models/Inkling-2-Mini")))
    ap.add_argument(
        "--out", type=Path, default=Path(os.environ.get("INKLING_CACHE", "/work/inkling-bringup/cache/v1")) / "mtp"
    )
    args = ap.parse_args(argv)
    entries = []
    for name, t, dtype, layout in tensors(args.ckpt):
        if t.dim() == 2:
            t = t[None, None]
        pcc = roundtrip_pcc(t.to(torch.bfloat16), dtype)[0] if dtype in ("BFP8", "BFP4") else 1.0
        if pcc < 0.99:
            raise ValueError(f"mtp.{name}: {dtype} PCC {pcc:.5f} < 0.99")
        rel = f"global/mtp/{name}.tensorbin"
        (args.out / rel).parent.mkdir(parents=True, exist_ok=True)
        host = t.to(torch.float32 if dtype == "FP32" else torch.bfloat16).contiguous()
        ttnn.dump_tensor(args.out / rel, ttnn.from_torch(host, dtype=DTYPES[dtype], layout=LAYOUTS[layout]))
        entries.append(
            {
                "name": name,
                "layer": None,
                "file": rel,
                "dtype": dtype,
                "layout": layout,
                "shape": list(t.shape),
                "src": ["mtp.safetensors"],
                "pcc": round(pcc, 7),
            }
        )
        print(f"{name} {dtype} {list(t.shape)} pcc {pcc:.5f}", flush=True)
    m = {"schema": 1, "complete": True, "source": str(args.ckpt), "kind": "mtp", "tensors": entries}
    (args.out / "manifest.json").write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
