#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Compare per-token cross-entropy of a truncated Qwen3.8 between HF (CPU, fp32) and ttml (TP=4).

Both sides load the real checkpoint truncated to the first ``--layers`` layers
(+ final norm + lm_head) and score the same tokens (the start of a GovReport
validation example). Matching per-token losses pin down the loader (norm +1
folding, conv taps, QKV permutation, TP sharding) and the fused kernels.

    python validate_vs_hf.py hf   --out hf.npz          # CPU only
    python validate_vs_hf.py ttml --out tt.npz          # 4 chips
    python validate_vs_hf.py compare hf.npz tt.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

MODEL = os.environ.get("QWEN38_MODEL_PATH", "/localdev/aleksajovanovic/models/Qwen/Qwen3.8-27B")
DATA = os.path.join(os.environ.get("GOVREPORT_DATA", "/localdev/umales/data/govreport/tok"), "validation.npz")


def tokens(seq: int) -> np.ndarray:
    d = np.load(DATA)
    t = d["tokens"][d["offsets"][0] : d["offsets"][1]]
    if len(t) <= seq:  # longer than one example: run on across example boundaries
        t = d["tokens"]
    return t[: seq + 1].astype(np.int64)


def run_hf(layers: int, seq: int, out: Path) -> None:
    import torch
    from safetensors import safe_open
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    raw = json.load(open(f"{MODEL}/config.json"))["text_config"]
    raw["num_hidden_layers"] = layers
    raw["layer_types"] = raw["layer_types"][:layers]
    cfg = Qwen3_5TextConfig(**raw)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(cfg).to(torch.float32).eval()

    wanted = model.state_dict()
    index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    state = {}
    for ck_name, shard in index.items():
        if ck_name == "lm_head.weight":
            name = "lm_head.weight"
        elif ck_name.startswith("model.language_model."):
            name = "model." + ck_name[len("model.language_model.") :]
        else:
            continue
        if name not in wanted:
            continue
        with safe_open(f"{MODEL}/{shard}", "pt") as f:
            state[name] = f.get_tensor(ck_name).to(torch.float32)
    missing = [k for k in wanted if k not in state]
    assert not missing, f"missing {missing[:5]}"
    model.load_state_dict(state, strict=True)

    t = torch.from_numpy(tokens(seq))
    with torch.no_grad():
        logits = model(t[None, :-1]).logits[0].float()
    loss = torch.nn.functional.cross_entropy(logits, t[1:], reduction="none")
    np.savez(out, loss=loss.numpy())
    print(f"HF {layers} layers: mean loss {loss.mean():.4f}")


def run_ttml(layers: int, seq: int, out: Path, sequence_parallel: bool, causal_check: bool = False) -> None:
    import ttnn
    import ttml
    from ttml.models.qwen38 import Qwen38Config, Qwen38Transformer
    from ttml.models.qwen38.loading import load_from_safetensors

    tp = 4
    ttml.open_device_mesh(ttml.Mesh((1, tp), ("dp", "tp")))
    ttml.autograd.AutoContext.get_instance().initialize_parallelism_context(
        ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=True)
    )
    ctx = ttml.autograd.AutoContext.get_instance()
    device = ctx.get_device()
    mesh = ttml.mesh()

    config = Qwen38Config.from_hf_json(Path(MODEL) / "config.json")
    config.num_hidden_layers = layers
    config.layer_types = config.layer_types[:layers]
    config.use_tp = True
    config.delta_rule_impl = "fused"
    config.conv_impl = "fused"
    config.sequence_parallel = sequence_parallel
    model = Qwen38Transformer(config)
    load_from_safetensors(model, Path(MODEL), config)
    model.eval()

    composer = ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig([0, 1]))

    def per_token_loss(t):
        inputs = ttml.autograd.Tensor.from_numpy(
            t[:-1].reshape(1, 1, 1, seq), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
        )
        targets = ttml.autograd.Tensor.from_numpy(t[1:].reshape(1, seq), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)
        loss = ttml.ops.distributed.vocab_parallel_cross_entropy_loss(
            model(inputs, None), targets, cluster_axis=mesh.axis_index("tp"), reduce=ttml.ops.ReduceType.NONE
        )
        per_tok = loss.to_numpy(composer=composer)[0, 0].reshape(-1)  # device (0, 0); TP copies are equal
        ctx.reset_graph()
        return per_tok

    t = tokens(seq).astype(np.uint32)
    per_tok = per_token_loss(t)
    np.savez(out, loss=per_tok.astype(np.float32))
    print(f"ttml {layers} layers: mean loss {per_tok.mean():.4f}")
    if causal_check:
        # Change every token from seq/2 on; losses at positions < seq/2 - 1 must not move.
        cut = seq // 2
        t2 = t.copy()
        t2[cut:] = np.random.default_rng(0).integers(0, 1000, len(t2) - cut, dtype=np.uint32)
        per_tok2 = per_token_loss(t2)
        diff = np.abs(per_tok[: cut - 1] - per_tok2[: cut - 1])
        moved = np.nonzero(diff > 0)[0]
        print(
            f"causality: max |loss change| before position {cut - 1} = {diff.max():.4e}; "
            f"{len(moved)} positions changed" + (f", first {moved[:5].tolist()}" if len(moved) else "")
        )
        print("CAUSAL" if diff.max() == 0 else "LEAK")


def compare(a: Path, b: Path) -> int:
    x, y = np.load(a)["loss"], np.load(b)["loss"]
    diff = np.abs(x - y)
    corr = np.corrcoef(x, y)[0, 1]
    print(
        f"mean loss {x.mean():.4f} vs {y.mean():.4f}; per-token |diff| mean {diff.mean():.4f} "
        f"p99 {np.percentile(diff, 99):.4f}; Pearson r {corr:.5f}"
    )
    ok = corr > 0.99 and abs(x.mean() - y.mean()) < 0.05 * x.mean()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["hf", "ttml", "compare"])
    p.add_argument("files", nargs="*")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--out", type=Path)
    p.add_argument("--sequence-parallel", action="store_true")
    p.add_argument("--causal-check", action="store_true", help="ttml: verify outputs ignore future tokens")
    args = p.parse_args()
    if args.mode == "hf":
        run_hf(args.layers, args.seq, args.out)
    elif args.mode == "ttml":
        run_ttml(args.layers, args.seq, args.out, args.sequence_parallel, args.causal_check)
    else:
        return compare(Path(args.files[0]), Path(args.files[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
