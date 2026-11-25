# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPTQ rounding of gpt-oss-20b's dense weights (attention q / k / v / o, router, LM head) onto the bfloat8_b grid.

This builds GPT_OSS_DENSE_ROUNDED_DECODE (dense/dense_gptq_device.pt), the dense weights the gpt-oss-20b release
settings read (models/demos/gpt_oss/__init__.py) for prefill and decode alike (utils/lazy_state_dict.py).

The grid. The device holds these weights as [in, out] in normal bfloat8_b tiles: blocks of 16 consecutive output
columns of one input row share an exponent, each value a sign and a 7-bit magnitude in steps of 2^(e - 6), e the
exponent of the block's largest value. Converting the bf16 checkpoint rounds every value to the nearest grid point
independently; GPTQ instead rounds the input rows one at a time and pushes each row's error onto the rows not yet
rounded through the inverse Hessian (H = X^T X of the layer's real inputs), so the errors cancel in the output. Each
block also takes the next finer step 2^(e - 7) (the max clips to 127 steps) when that rounds it closer. The result is
still on the grid, so the device conversion keeps it and nothing runs slower. Blocks run along the output dimension,
so rows are rounded in act order (largest input energy first), as in tools/gptq_experts.py.

q / k are rounded in Meta order (the loader's convert_hf_qkv_to_meta_format permutation of the output rows), because
that is the order the blocks see on device; the file holds them in Meta order, as utils/lazy_state_dict.py expects.

  # 1. calibration: the Hugging Face reference (CPU, bf16) over text files; accumulates X^T X for every dense input
  python models/demos/gpt_oss/tools/dense_gptq.py collect dense_calib.pt texts/*.txt
  # 2. rounding: one file of every rounded dense weight (bfloat16, [out, in] as in the checkpoint)
  python models/demos/gpt_oss/tools/dense_gptq.py round dense_gptq_device.pt dense_calib.pt [dense_calib2.pt ...]
  # 3. check a file (yours or the published one): keys, shapes, on the grid
  python models/demos/gpt_oss/tools/dense_gptq.py check dense_gptq_device.pt
  # then
  export GPT_OSS_DENSE_ROUNDED_DECODE=$PWD/dense_gptq_device.pt

The published file (hf://jerrywangTT/gpt-oss-20b-qb2-kgptq dense/) was calibrated on the same kind of text as the
experts (gpt-oss-20b's AIME 2025 reasoning traces; see that repo's README). Run from the tt-metal root (q / k use
models.tt_transformers' Meta-order conversion). Only weights are rounded; biases stay the checkpoint's.
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import torch

GROUP = 16  # values per bfloat8_b block (one exponent)
MAG = 127  # largest magnitude in steps (7 bits)
TOKENS_PER_CHUNK = 4096
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _snapshot():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download("openai/gpt-oss-20b", allow_patterns=["*.safetensors", "*.json"]))


def bfp8_rne(w):
    """[..., N] -> the bfloat8_b grid (blocks of 16 along N, the step from each block's max), round half to even."""
    b = w.reshape(*w.shape[:-1], -1, GROUP)
    m = b.abs().amax(-1, keepdim=True)
    step = torch.pow(2.0, torch.floor(torch.log2(m.clamp_min(1e-30))) - 6)
    q = torch.clamp(torch.round(b / step), -MAG, MAG) * step
    return torch.where(m > 0, q, b).reshape(w.shape)


def bfp8_best(w):
    """bfp8_rne, or per block the next finer step (the block max clips to 127 steps) when that is closer."""
    b = w.reshape(*w.shape[:-1], -1, GROUP)
    q0 = bfp8_rne(w).reshape(b.shape)
    m = b.abs().amax(-1, keepdim=True)
    step = torch.pow(2.0, torch.floor(torch.log2(m.clamp_min(1e-30))) - 7)
    q1 = torch.where(m > 0, torch.clamp(torch.round(b / step), -MAG, MAG) * step, b)
    finer = ((q1 - b) ** 2).sum(-1, keepdim=True) < ((q0 - b) ** 2).sum(-1, keepdim=True)
    return torch.where(finer, q1, q0).reshape(w.shape)


def gptq_n(W, H, damp=0.01, block=128, cols=16384):
    """W [K, N] (K: input rows, N: outputs), H [K, K] = X^T X -> W rounded onto the bfloat8_b grid (blocks along N).
    Act order; each row's error pushed onto the rows after it. Columns are independent given H, so N runs in chunks
    of `cols` (a multiple of 16) to bound memory (the LM head is 2880 x 201088)."""
    K, N = W.shape
    H = H.clone().double()
    dead = torch.diagonal(H) == 0
    H += torch.diag(dead.double())
    perm = torch.argsort(torch.diagonal(H), descending=True)
    H = H[perm][:, perm]
    H += damp * torch.diagonal(H).mean() * torch.eye(K, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True).float()
    out = torch.empty(K, N)
    for c1 in range(0, N, cols):
        c2 = min(c1 + cols, N)
        Wc = W[:, c1:c2].float().clone()
        Wc[dead] = 0
        Wc = Wc[perm]
        Q = torch.zeros_like(Wc)
        for i1 in range(0, K, block):
            i2 = min(i1 + block, K)
            W1, Err, Hi = Wc[i1:i2].clone(), torch.zeros(i2 - i1, c2 - c1), Hinv[i1:i2, i1:i2]
            for i in range(i2 - i1):
                q = bfp8_best(W1[i])
                Q[i1 + i] = q
                err = (W1[i] - q) / Hi[i, i]
                W1[i:] -= Hi[i, i:].unsqueeze(-1) * err.unsqueeze(0)
                Err[i] = err
            Wc[i2:] -= Hinv[i1:i2, i2:].t() @ Err
        out[perm, c1:c2] = Q
    return out


class Dense:
    """gpt-oss-20b's dense weights from the HF checkpoint ([out, in], fp32), q / k in Meta order."""

    def __init__(self):
        self.path = _snapshot()
        self.config = json.loads((self.path / "config.json").read_text())
        self.index = json.loads((self.path / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name):
        from safetensors import safe_open

        with safe_open(str(self.path / self.index[name]), "pt") as f:
            w = f.get_tensor(name)
        if name.endswith(("q_proj.weight", "k_proj.weight")):
            from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

            w = convert_hf_qkv_to_meta_format({name: w}, self.config["head_dim"])[name]
        return w.float()

    def names(self):
        """(weight name, Hessian key) of every rounded weight."""
        out = []
        for i in range(self.config["num_hidden_layers"]):
            p = f"model.layers.{i}."
            out += [(p + f"self_attn.{n}.weight", f"{i}.{'o' if n == 'o_proj' else 'qkv'}") for n in LINEARS]
            out.append((p + "mlp.router.weight", f"{i}.router"))
        out.append(("lm_head.weight", "lm_head"))
        return out


def collect(out, texts):
    """The HF reference (CPU, bf16) over the texts in 4096-token chunks; X^T X (fp32) of the inputs of q / k / v (one
    input), o_proj, the router and the LM head."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    model = AutoModelForCausalLM.from_pretrained("openai/gpt-oss-20b", dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    hess, tokens = {}, [0]

    def acc(key):
        def hook(mod, args):
            x = args[0].reshape(-1, args[0].shape[-1]).float()
            hess[key] = hess.get(key, 0) + x.t() @ x

        return hook

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.register_forward_pre_hook(acc(f"{i}.qkv"))
        layer.self_attn.o_proj.register_forward_pre_hook(acc(f"{i}.o"))
        layer.mlp.router.register_forward_pre_hook(acc(f"{i}.router"))
    model.lm_head.register_forward_pre_hook(acc("lm_head"))
    for path in texts:
        ids = tok.encode(Path(path).read_text())
        for s in range(0, len(ids), TOKENS_PER_CHUNK):
            t = time.time()
            chunk = ids[s : s + TOKENS_PER_CHUNK]
            with torch.no_grad():
                model(torch.tensor([chunk]))
            tokens[0] += len(chunk)
            print(f"{path} [{s}:{s + len(chunk)}] {time.time() - t:.0f} s", flush=True)
    torch.save({"tokens": tokens[0], "hessians": hess}, out)


def _output_error(q, w, H):
    """Relative output error of q against w ([out, in]) over the calibration inputs: sqrt(tr(D H D^T) / tr(W H W^T)),
    each trace as a sum of (M @ H) * M, so no [out, out] matrix (201088^2 for the LM head) is formed."""
    d = q - w
    return (((d @ H) * d).sum() / ((w @ H) * w).sum()).sqrt().item()


def round_all(out, calib_files, damp, only=None):
    parts = [torch.load(f) for f in calib_files]
    print(f"{sum(p['tokens'] for p in parts)} calibration tokens", flush=True)
    dense, result = Dense(), {}
    for name, hkey in dense.names():
        if only and not any(s in name for s in only):
            continue
        t = time.time()
        H = sum(p["hessians"][hkey] for p in parts)
        w = dense.get(name)  # [out, in]
        q = gptq_n(w.t().contiguous(), H, damp).t().contiguous()
        if not torch.equal(bfp8_rne(q), q):
            raise RuntimeError(f"{name} is not on the bfloat8_b grid after GPTQ")
        if not torch.equal(q.bfloat16().float(), q):
            raise RuntimeError(f"{name} is not exact in bfloat16")
        result[name] = q.bfloat16()
        rtn = bfp8_rne(w.t()).t()
        rel, rel0 = _output_error(q, w, H), _output_error(rtn, w, H)
        print(f"{name}: output rel. error {rel:.3%} (round-to-nearest {rel0:.3%}), {time.time() - t:.0f} s", flush=True)
    torch.save(result, out)


def check(path):
    """Keys / shapes against the checkpoint, every weight on the bfloat8_b grid ([in, out], blocks along out)."""
    d, dense = torch.load(path, mmap=True), Dense()
    names = [n for n, _ in dense.names()]
    missing, extra = sorted(set(names) - set(d)), sorted(set(d) - set(names))
    print(f"{len(d)} weights; missing {missing or 'none'}; extra {extra or 'none'}")
    for name in names:
        if name not in d:
            continue
        w, ref = d[name].float(), dense.get(name)
        assert w.shape == ref.shape, f"{name}: {tuple(w.shape)} != {tuple(ref.shape)}"
        on = torch.equal(bfp8_rne(w.t()), w.t())
        moved = (w - ref).abs().gt(0).float().mean().item()
        print(f"{name}: on grid {on}, {moved:.1%} of values differ from the checkpoint", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "8")))
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("texts", nargs="+")
    r = sub.add_parser("round")
    r.add_argument("out")
    r.add_argument("calib", nargs="+")
    r.add_argument("--damp", type=float, default=0.01)
    r.add_argument("--only", default="", help="comma-separated name substrings (a quick partial run)")
    k = sub.add_parser("check")
    k.add_argument("path")
    a = p.parse_args()
    if a.cmd == "collect":
        collect(a.out, [f for t in a.texts for f in sorted(glob.glob(t))])
    elif a.cmd == "round":
        round_all(a.out, a.calib, a.damp, [s for s in a.only.split(",") if s])
    else:
        check(a.path)
