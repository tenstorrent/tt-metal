# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPTQ rounding of gpt-oss-20b's MXFP4 routed-expert weights onto the bfloat4_b grid of the fused decode streams.

Not used by the release: the gpt-oss-20b release settings (models/demos/gpt_oss/__init__.py) use the K-GPTQ rounding
instead (16-value groups along the input dimension, read from GPT_OSS_DECODE_SPLIT_ROUND_DIR by the decode streams and,
with GPT_OSS_PREFILL_KGPTQ, by the routed prefill experts). The model no longer has a switch that loads this tool's
output-grouped rounding; the tool is kept for reference and re-calibration studies.

The decode streams hold the experts in bfloat4_b: blocks of 16 consecutive output columns at one input row share an
exponent, each value a sign and a 3-bit magnitude (uniform steps of 2^(e - 2), e the block max's exponent). MXFP4's
values {0, 0.5, 1, 1.5, 2, 3, 4, 6} x scale do not fit that grid: the conversion's round-to-nearest leaves ~10% error on
every expert's output, and the decoded model samples as if at temperature 1.10 (top-1 vs the FP32 reference 90.0%).

GPTQ rounds the input rows one at a time and pushes each row's rounding error onto the rows not rounded yet through the
inverse Hessian of real expert inputs, so the errors cancel in the expert's output; each block also takes the finer of
its two candidate steps when that is closer (the device derives the same step from the clipped block max). The result
is still exactly on the bfloat4_b grid: the device conversion keeps it, and decode runs the same bytes at the same speed.
QuietBox 2, 180K calibration tokens: expert output error 11.2% / 9.5% -> 6.1% / 5.8% (gate|up / down), top-1 vs FP32
94.1%, effective temperature 1.008 (bfloat8_b experts: 96.5%, 1.008, decode 22% slower).

  # 1. calibration: the HF reference (CPU) over text files, MoE inputs of every layer (~70 s per 4096 tokens)
  python models/demos/gpt_oss/tools/gptq_experts.py collect calib.pt texts/*.txt
  # 2. rounding (~2.5 min per layer at 8 threads): one layer_<L>.pt per layer
  python models/demos/gpt_oss/tools/gptq_experts.py round rounded/ calib.pt [calib2.pt ...]
  # 3. (the model no longer has a switch that loads them, see above)

Calibration size matters: ~500 tokens per expert overfits (worse than round-to-nearest on held-out tokens); 82K tokens
in total (~10K per expert) generalizes. Use 8 threads: at 16 the per-row ops spend ~50 ms each synchronizing.
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch

LUT = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
ALPHA, LIMIT = 1.702, 7.0  # gpt-oss SwiGLU
TOKENS_PER_CHUNK = 4096


def _snapshot():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("openai/gpt-oss-20b", allow_patterns=["*.safetensors", "*.json"]))


class Checkpoint:
    """The MXFP4 expert weights and biases straight from the HF checkpoint (exact)."""

    def __init__(self):
        self.path = _snapshot()
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def _get(self, name):
        from safetensors import safe_open

        with safe_open(str(self.path / self.index[name]), "pt") as f:
            return f.get_tensor(name)

    def mxfp4(self, name):
        """[E, out, in] values of an MXFP4 tensor."""
        blocks, scales = self._get(name + "_blocks"), self._get(name + "_scales")
        codes = torch.stack([LUT[(blocks & 15).long()], LUT[(blocks >> 4).long()]], -1)
        w = codes.reshape(*blocks.shape[:-1], 32) * torch.pow(2.0, scales.float() - 127).unsqueeze(-1)
        return w.reshape(blocks.shape[0], blocks.shape[1], -1)

    def layer(self, i):
        p = f"model.layers.{i}.mlp.experts."
        return dict(
            gate_up_proj=self.mxfp4(p + "gate_up_proj")
            .transpose(1, 2)
            .contiguous(),  # [E, H, 2I], gate / up interleaved
            gate_up_proj_bias=self._get(p + "gate_up_proj_bias").float(),
            down_proj=self.mxfp4(p + "down_proj").transpose(1, 2).contiguous(),  # [E, I, H]
            down_proj_bias=self._get(p + "down_proj_bias").float(),
        )


def bfp4_rne(w):
    """[..., N] -> the bfloat4_b grid (blocks of 16 along N), round half to even: ttnn's conversion."""
    b = w.reshape(*w.shape[:-1], -1, 16)
    m = b.abs().amax(-1, keepdim=True)
    step = torch.pow(2.0, torch.floor(torch.log2(m.clamp_min(1e-30))) - 2)
    q = torch.clamp(torch.round(b / step), -7, 7) * step
    return torch.where(m > 0, q, b).reshape(w.shape)


def bfp4_best(w):
    """bfp4_rne, or per block the next finer step (the block max clips to 7 steps) when that is closer."""
    b = w.reshape(*w.shape[:-1], -1, 16)
    q0 = bfp4_rne(w).reshape(b.shape)
    m = b.abs().amax(-1, keepdim=True)
    step = torch.pow(2.0, torch.floor(torch.log2(m.clamp_min(1e-30))) - 3)
    q1 = torch.where(m > 0, torch.clamp(torch.round(b / step), -7, 7) * step, b)
    finer = ((q1 - b) ** 2).sum(-1, keepdim=True) < ((q0 - b) ** 2).sum(-1, keepdim=True)
    return torch.where(finer, q1, q0).reshape(w.shape)


def gptq(W, H, damp=0.01, block=128):
    """W [E, K, N] (K: input rows), H [E, K, K] = X^T X of the calibration inputs -> W rounded onto the bfloat4_b grid.
    Rows in order of input energy (act order); each row's error pushed onto the rows after it."""
    E, K, N = W.shape
    W, H = W.clone(), H.clone()
    dead = torch.diagonal(H, dim1=1, dim2=2) == 0
    H += torch.diag_embed(dead.float())
    W.masked_fill_(dead.unsqueeze(-1), 0)
    perm = torch.argsort(torch.diagonal(H, dim1=1, dim2=2), dim=1, descending=True)
    W = torch.gather(W, 1, perm.unsqueeze(-1).expand(-1, -1, N))
    H = torch.gather(torch.gather(H, 1, perm.unsqueeze(-1).expand(-1, -1, K)), 2, perm.unsqueeze(1).expand(-1, K, -1))
    H += damp * torch.diagonal(H, dim1=1, dim2=2).mean(-1)[:, None, None] * torch.eye(K)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H.double())), upper=True).float()
    Q = torch.zeros_like(W)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1, Err, Hi = W[:, i1:i2].clone(), torch.zeros(E, i2 - i1, N), Hinv[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            q = bfp4_best(W1[:, i])
            Q[:, i1 + i] = q
            err = (W1[:, i] - q) / Hi[:, i, i].unsqueeze(-1)
            W1[:, i:] -= Hi[:, i, i:].unsqueeze(-1) * err.unsqueeze(1)
            Err[:, i] = err
        W[:, i2:] -= torch.bmm(Hinv[:, i1:i2, i2:].transpose(1, 2), Err)
    return torch.zeros_like(Q).scatter_(1, perm.unsqueeze(-1).expand(-1, -1, N), Q)


def swiglu(gate_up):
    g, u = gate_up[..., ::2].clamp(max=LIMIT), gate_up[..., 1::2].clamp(-LIMIT, LIMIT)
    return (u + 1) * (g * torch.sigmoid(g * ALPHA))


def round_layer(weights, x, routed, damp=0.01):
    """One layer's experts: GPTQ of the gate / up halves (the decode streams keep them in separate column ranges, the
    bfloat4_b blocks never mix them) and of down. x [T, H]: the layer's MoE inputs; routed [T, top_k]: their experts."""
    gu, gub, dn = weights["gate_up_proj"], weights["gate_up_proj_bias"], weights["down_proj"]
    E, H, I2 = gu.shape
    Hg, Hd = torch.zeros(E, H, H), torch.zeros(E, I2 // 2, I2 // 2)
    for e in range(E):
        X = x[(routed == e).any(-1)]
        A = swiglu(X @ gu[e] + gub[e])
        Hg[e], Hd[e] = X.t() @ X, A.t() @ A
    q_gate, q_up = gptq(gu[:, :, 0::2].contiguous(), Hg, damp), gptq(gu[:, :, 1::2].contiguous(), Hg, damp)
    out = dict(weights)
    out["gate_up_proj"] = torch.stack([q_gate, q_up], -1).reshape(gu.shape)
    out["down_proj"] = gptq(dn, Hd, damp)
    for name, w in (("gate", q_gate), ("up", q_up), ("down", out["down_proj"])):
        if not torch.equal(bfp4_rne(w), w):  # the device conversion would round it again
            raise RuntimeError(f"layer's {name} weights are not on the bfloat4_b grid after GPTQ")
    return out


def collect(out, texts):
    """The HF reference (CPU, bf16) over the texts in 4096-token chunks; per layer the MoE inputs and routed experts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    cap = {i: ([], []) for i in range(len(model.model.layers))}
    for i, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_pre_hook(
            lambda mod, args, i=i: cap[i][0].append(args[0].reshape(-1, args[0].shape[-1]).half().clone())
        )
        layer.mlp.router.register_forward_hook(
            lambda mod, args, out, i=i: cap[i][1].append(out[2].reshape(-1, out[2].shape[-1]).short().clone())
        )
    for path in texts:
        ids = tok.encode(Path(path).read_text())
        for s in range(0, len(ids), TOKENS_PER_CHUNK):
            t = time.time()
            with torch.no_grad():
                model(torch.tensor([ids[s : s + TOKENS_PER_CHUNK]]))
            print(f"{path} [{s}:{s + TOKENS_PER_CHUNK}] {time.time() - t:.0f} s", flush=True)
    torch.save({i: (torch.cat(x), torch.cat(r)) for i, (x, r) in cap.items()}, out)


def round_all(out_dir, calib_files, layers, damp):
    os.makedirs(out_dir, exist_ok=True)
    parts = [torch.load(f) for f in calib_files]
    ckpt = Checkpoint()
    for i in layers:
        t = time.time()
        x = torch.cat([p[i][0] for p in parts]).float()
        routed = torch.cat([p[i][1] for p in parts])
        out = round_layer(ckpt.layer(i), x, routed, damp)
        torch.save({k: v.bfloat16() if k in ("gate_up_proj", "down_proj") else v for k, v in out.items()},
                   f"{out_dir}/layer_{i}.pt")  # fmt: skip
        print(f"layer {i}: {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("texts", nargs="+")
    r = sub.add_parser("round")
    r.add_argument("out_dir")
    r.add_argument("calib", nargs="+")
    r.add_argument("--layers", default="all")
    r.add_argument("--damp", type=float, default=0.01)
    a = p.parse_args()
    if a.cmd == "collect":
        collect(a.out, [f for t in a.texts for f in sorted(glob.glob(t))])
    else:
        n = json.loads((_snapshot() / "config.json").read_text())["num_hidden_layers"]
        layers = range(n) if a.layers == "all" else [int(v) for v in a.layers.split(",")]
        round_all(a.out_dir, a.calib, layers, a.damp)
