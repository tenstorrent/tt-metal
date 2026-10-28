# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Generate the fp32 CPU goldens for Granite FlowState r1.1 from the pinned upstream reference.

Runs only inside the reference venv (Python 3.12, torch 2.11.0+cpu, granite-tsfm 0.3.11), never from
tt-metal's python_env, with the worktree kept off the import path:

    ttq run --model flowstate --kind cpu --sh 'env -u PYTHONPATH -u VIRTUAL_ENV HF_HUB_OFFLINE=1 \
        /home/andy/bringup/venvs/flowstate/bin/python \
        models/experimental/flowstate_forecast/tests/golden/make_goldens.py --tier test \
        --out /home/andy/bringup/goldens/flowstate \
        --manifest models/experimental/flowstate_forecast/tests/golden/MANIFEST.json'

Tiers (``--tier``), each a set of ``torch.save`` dicts; every file saves its inputs:

  test   single.pt          1 x 2048 -> 64, s = 1: RevIN stats, normalized tokens, embedding, per layer the SSM
                            block / MLP / layer outputs, decoder coefficients and pre-denorm output, finals, and the
                            per-(s, L) constants (Lbar, Bbar per layer, Legendre W)
         model16.pt         16 x 2048 -> 64: layer outputs, pre-denorm decoder output, finals
         single_long.pt     1 x 16360 -> 48, s = 0.25: pre-denorm decoder output, finals, last-layer output
         <path case>.pt     nan_mix, mask0, daily, monthly, fivemin, cpm_edge (CONTRACT §2): RevIN stats,
                            pre-denorm decoder output, finals; layer outputs where B * L <= 2200
         modules.pt         small random-input goldens for each module (embedding, S5 block, MLP, layer, decoder)
  b0     b0_<case>.pt       the B0 SSM goldens (CONTRACT §5 as amended by D0 items 2-3), layers 0, 3, 4, 5:
                            u (the layer input, rounded to bf16, the device's activation dtype; it is the
                            golden's input, so the rounding is not an error term), Lbar, Bbar, and for the
                            reference conv output x, Re(C x) and the block output on u a sha256 plus the first and
                            last 64 steps. The full tensors and the float64 truth are recomputed in the CPU test
                            (`b0_reference.py`, same torch 2.11 CPU, checked against the digests): a full
                            complex64 x is 134 MB per (case, layer) at L = 16384, against a 0.5 GB budget.
  bench  bench_<shape>.pt   paper, long_context, long_horizon, scale_q, single, single_long: inputs, RevIN stats,
                            pre-denorm decoder output, finals, cpu_s (chunked: 16 series at L <= 4096, 4 above)

MANIFEST.json gets repo, commit, pip freeze, HF revision, safetensors sha256, threads, seeds and the golden sizes;
the script fails if the golden directory exceeds 0.5 GB (D0 amendment 2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"  # every tier: never download under the watermark (D2 fix 2)

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inputs as gen  # noqa: E402

THREADS = 8
torch.set_num_threads(THREADS)

REPO = "ibm-granite/granite-timeseries-flowstate-r1"
REVISION = "b80e0f1b1c2d90edebe3d4f243ad046a3f935daa"
SAFETENSORS_SHA256 = "a4a745458cfef1cce84ef060a2e7fe515e40a2e5b0ea54fb079570a75f324862"
CODE = {
    "repo": "https://github.com/ibm-granite/granite-tsfm",
    "tag": "v0.3.11",
    "commit": "dfe846e6c1f1b0086b4a7b0831249ac8aeb7dcd7",
}
# sha256 of the setup scratchpad's modeling_flowstate.py the script was written against; reference_sources()
# must report the same hash for the installed 0.3.11 copy, or the script is re-checked before goldens (D2 fix 3)
SCRATCHPAD_MODELING_SHA256 = "f8ef6534003cfd924289124388ec81d1d45ea875408a9e828601f6f5d2b0ec8d"
GOLDEN_BUDGET = 512 * 2**20
B0_LAYERS = (0, 3, 4, 5)

# name: (batch, context, horizon, s, seed, nan_rows)
TEST_CASES = {
    "single": (1, 2048, 64, 1.0, 1, 0),
    "model16": (16, 2048, 64, 1.0, 2, 2),
    "single_long": (1, 16360, 48, 0.25, 3, 0),
    "mask0": (2, 1000, 6, 1.0, 5, 0),
    "daily": (1, 4000, 30, 24 / 365, 6, 0),
    "monthly": (3, 500, 18, 2.0, 7, 0),
    "fivemin": (1, 15976, 480, 1 / 12, 8, 0),
    "cpm_edge": (1, 1024, 65, 1.0, 9, 0),
}
NAN_MIX = (4, 1024, 64, 1.0, 4)
# B0: (batch, L, s, seed); L is the encoder length, the context is L - mask_n with horizon = fl (mask_n = 0)
B0_CASES = {
    "s1_L4096": (2, 4096, 1.0, 11),
    "s0.25_L16384": (2, 16384, 0.25, 12),
    "s1_12_L16384": (2, 16384, 1 / 12, 13),
    "s6_L682": (4, 682, 6.0, 14),
}
BENCH_CASES = {
    "paper": (64, 4038, 64, 1.0, 101, 4),
    "long_context": (16, 16360, 48, 0.25, 102, 2),
    "long_horizon": (64, 3382, 720, 1.0, 103, 4),
    "scale_q": (256, 675, 8, 6.0, 104, 8),
    "single": (1, 2048, 64, 1.0, 105, 0),
    "single_long": (1, 16360, 48, 0.25, 106, 0),
}


def load_model():
    from tsfm_public import FlowStateForPrediction

    model, info = FlowStateForPrediction.from_pretrained(REPO, revision=REVISION, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    # D2 fix 1: check the loaded dtypes before casting; the checkpoint is stored fp32, so .float() is a no-op
    # kept only as a guard. A non-fp32 parameter fails here instead of being cast silently.
    bad = {n: str(p.dtype) for n, p in model.named_parameters() if p.dtype != torch.float32}
    assert not bad, f"non-fp32 parameters after load: {bad}"
    return model.eval().float()


def stored_dtypes(path):
    """dtype of every safetensors entry as stored on disk (header only), for MANIFEST."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    return {k: v["dtype"] for k, v in header.items() if k != "__metadata__"}


def reference_sources():
    """sha256 of the installed granite-tsfm FlowState sources the goldens come from (D2 fix 3)."""
    import tsfm_public

    root = Path(tsfm_public.__file__).resolve().parent
    files = sorted(p for p in root.rglob("*.py") if "flowstate" in str(p.relative_to(root)).lower())
    assert any(p.name == "modeling_flowstate.py" for p in files), files
    shas = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    modeling = [v for k, v in shas.items() if k.endswith("modeling_flowstate.py")]
    assert (
        SCRATCHPAD_MODELING_SHA256 in modeling
    ), f"installed modeling_flowstate.py differs from the scratchpad: {shas}"
    return shas


def bookkeeping(model, context, horizon, s):
    """The reference's own index arithmetic (FlowStateForPrediction.forward)."""
    fl = int(model.config.decoder_patch_len / s + 1e-6)
    mask_n = max(0, horizon - fl)
    max_context = min(16 * 1024, int(model.config.context_length / s)) - mask_n
    ctx = min(context, max_context)
    return {
        "fl": fl,
        "mask_n": mask_n,
        "max_context": max_context,
        "L": ctx + mask_n,
        "min_context": ctx if mask_n > 0 else 0,
    }


class Capture:
    """Forward hooks; ``full`` adds per-layer SSM / MLP outputs."""

    def __init__(self, model, full):
        self.d, self.h = {}, []
        m = model.model

        def norm_hook(mod, args, out):
            if args[1] == "norm":
                self.d["revin_mean"] = mod.mean.clone()
                self.d["revin_stdev"] = mod.stdev.clone()
                self.d["normalized"] = out.clone()

        self.h.append(m.norm.register_forward_hook(norm_hook))
        self.h.append(
            m.decoder.register_forward_hook(
                lambda mod, a, out: self.d.__setitem__("decoder_pre_denorm", out.last_hidden_state.clone())
            )
        )
        if full:
            self.h.append(m.embed.register_forward_hook(lambda mod, a, out: self.d.__setitem__("embed", out.clone())))
            for i, lay in enumerate(m.encoder.layers):
                self.h.append(
                    lay.ssm.register_forward_hook(lambda mod, a, out, i=i: self.d.__setitem__(f"ssm{i}", out.clone()))
                )
                self.h.append(
                    lay.mlp.register_forward_hook(lambda mod, a, out, i=i: self.d.__setitem__(f"mlp{i}", out.clone()))
                )

    def close(self):
        for h in self.h:
            h.remove()


def run(model, x, horizon, s, full=False, layers=True):
    cap = Capture(model, full)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(past_values=x, scale_factor=s, prediction_length=horizon, batch_first=True)
    cpu_s = time.perf_counter() - t0
    cap.close()
    d = dict(cap.d)
    # in-block ssm / mlp outputs for the first and last layer only: the middle ones (33 MB) put the golden
    # directory over the 0.5 GB budget, and the per-layer outputs below still check every block
    n = len(model.model.encoder.layers)
    for i in range(1, n - 1):
        d.pop(f"ssm{i}", None)
        d.pop(f"mlp{i}", None)
    d.update(
        quantile_outputs=out.quantile_outputs,
        prediction_outputs=out.prediction_outputs,
        cpu_s=cpu_s,
        combined_normalized=model._combine_cpm_predictions(d["decoder_pre_denorm"], horizon),
    )
    hs = out.hidden_states[:-1]
    d.update({f"layer{i}": h for i, h in enumerate(hs)} if layers else {f"layer{len(hs) - 1}": hs[-1]})
    if full:
        d["decoder_coeffs"] = out.hidden_states[-1]
    return d


def constants(model, s, L, fl):
    """Per-(s, L) constants with the reference's own expressions; Lbar = kernel[:, -2] (k = 1), not the kernel."""
    sf = torch.ones(1) * s
    d = {}
    for i, lay in enumerate(model.model.encoder.layers):
        kernel, bbar = lay.ssm.get_discretized(sf, L=L)
        d[f"Lbar{i}"] = kernel[0, -2].clone()
        d[f"Bbar{i}"] = bbar[0].clone()
    d["W"] = model.model.decoder.get_kernel(s, fl, torch.device("cpu"))
    return d


def case(model, name, x, horizon, s, seed, full=False):
    bk = bookkeeping(model, x.shape[1], horizon, s)
    # per-layer outputs only when small (B * L <= 2200 tokens: <= 4.5 MB per layer; budget); the last layer's read rows always
    d = run(model, x, horizon, s, full=full, layers=x.shape[0] * bk["L"] <= 2200)
    d.update(inputs=x, horizon=horizon, s=s, seed=seed, **bk)
    if full:
        d.update(constants(model, s, bk["L"], bk["fl"]))
    return d


def modules_golden(model, seed):
    """Small random-input module goldens (L = 96, B = 2, s = 1, and a non-aligned L = 70 at s = 6)."""
    g = torch.Generator().manual_seed(seed)
    m = model.model
    d = {}
    with torch.no_grad():
        tok = torch.randn(96, 2, 2, generator=g)
        tok[..., 1] = (tok[..., 1] > 1).float()
        d["embed_in"], d["embed_out"] = tok, m.embed(tok)
        for L, s in ((96, 1.0), (70, 6.0)):
            u = torch.randn(L, 2, 512, generator=g)
            sf = torch.ones(2) * s
            d[f"u_L{L}"] = u
            for i in (0, 5):
                lay = m.encoder.layers[i]
                model.model.config.min_context = 0 if i < 5 else L - 7
                d[f"ssm{i}_L{L}"] = lay.ssm(u, sf)
                d[f"layer{i}_L{L}"] = lay(u, scale_factor=sf)
            z = torch.randn(L, 2, 512, generator=g)
            d[f"mlp_in_L{L}"], d[f"mlp0_L{L}"] = z, m.encoder.layers[0].mlp(z)
        o = torch.randn(13, 2, 512, generator=g)
        dec = m.decoder(encoder_output=type("E", (), {"last_hidden_state": o})(), prediction_length=6, scale_factor=1.0)
        d["dec_in"], d["dec_out"] = o, dec.last_hidden_state
    model.model.config.min_context = 0
    return d


def b0_golden(model, B, L, s, seed):
    """The B0 SSM goldens: layer inputs from a real forward, then each captured layer's SSM block on bf16 u."""
    x = gen.realistic(B, L, seed)  # horizon = fl -> mask_n = 0, so the encoder length is exactly L
    fl = int(model.config.decoder_patch_len / s + 1e-6)
    assert bookkeeping(model, L, fl, s)["L"] == L
    layer_in = {}
    hooks = [
        model.model.encoder.layers[i].register_forward_hook(
            lambda mod, a, out, i=i: layer_in.__setitem__(i, a[0].clone())
        )
        for i in B0_LAYERS
    ]
    with torch.no_grad():
        model(past_values=x, scale_factor=s, prediction_length=fl, batch_first=True)
    for h in hooks:
        h.remove()
    d = {"inputs": x, "s": s, "L": L, "seed": seed, "layers": list(B0_LAYERS)}
    sf = torch.ones(B) * s
    model.model.config.min_context = 0  # every captured block returns all L steps (non-last semantics)
    with torch.no_grad():
        for i in B0_LAYERS:
            blk = model.model.encoder.layers[i].ssm
            last = blk.last
            blk.last = False
            u = layer_in[i].to(torch.bfloat16).float()
            kernel, bbar = blk.get_discretized(sf, L=L)
            bu = torch.einsum("bnm,lbm->lbn", bbar.real, u) + torch.einsum("bnm,lbm->lbn", bbar.imag, u) * 1.0j
            xs = blk.apply_ssm_kern_ff(bu, kernel)
            y = torch.einsum("hn,...bn->...bh", blk.C_tilde_r, xs.real) + torch.einsum(
                "hn,...bn->...bh", blk.C_tilde_i, -xs.imag
            )
            d[f"u{i}"] = u.to(torch.bfloat16)
            d[f"Lbar{i}"] = kernel[0, -2].clone()
            d[f"Bbar{i}"] = bbar[0].clone()
            nbytes = sum(
                v.numel() * v.element_size()
                for k, v in d.items()
                if torch.is_tensor(v) and k.endswith(str(i)) and k[:1] in "uLB"
            )
            print(f"b0 L={L} s={s:.4g} layer {i}: {nbytes / 2**20:.1f} MB (u, Lbar, Bbar)", flush=True)
            for key, t in (("x", xs), ("recx", y), ("block", blk(u, sf))):
                d[f"{key}_head{i}"], d[f"{key}_tail{i}"] = t[:64].clone(), t[-64:].clone()
                d[f"{key}_sha256_{i}"] = hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
            blk.last = last
    return d


def bench_golden(model, B, ctx, horizon, s, seed, nan_rows):
    x = gen.realistic(B, ctx, seed, nan_rows=nan_rows)
    chunk = 16 if bookkeeping(model, ctx, horizon, s)["L"] <= 4096 else 4
    parts, cpu_s = [], 0.0
    for i in range(0, B, chunk):
        d = run(model, x[i : i + chunk], horizon, s, layers=False)
        cpu_s += d["cpu_s"]
        parts.append(d)
    keep = ("revin_mean", "revin_stdev", "quantile_outputs", "prediction_outputs", "combined_normalized")
    d = {k: torch.cat([p[k] for p in parts], 0) for k in keep}
    d["decoder_pre_denorm"] = torch.cat([p["decoder_pre_denorm"] for p in parts], 1)
    d.update(inputs=x, horizon=horizon, s=s, seed=seed, cpu_s=cpu_s, chunk=chunk, **bookkeeping(model, ctx, horizon, s))
    return d


def save(out, name, d, sizes):
    path = out / f"{name}.pt"
    torch.save(d, path)
    sizes[name] = path.stat().st_size
    print(f"{name}: {sizes[name] / 2**20:.1f} MB", flush=True)


def safetensors_sha256():
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, "model.safetensors", revision=REVISION, local_files_only=True)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest(), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=("test", "b0", "bench"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sha, st_path = safetensors_sha256()
    assert sha == SAFETENSORS_SHA256, sha
    model = load_model()
    sizes, seeds = {}, {}

    if args.tier == "test":
        for name, (B, ctx, H, s, seed, nan_rows) in TEST_CASES.items():
            seeds[name] = seed
            save(
                args.out,
                name,
                case(model, name, gen.realistic(B, ctx, seed, nan_rows), H, s, seed, full=name == "single"),
                sizes,
            )
        B, ctx, H, s, seed = NAN_MIX
        seeds["nan_mix"] = seed
        save(args.out, "nan_mix", case(model, "nan_mix", gen.nan_mix(ctx, seed), H, s, seed), sizes)
        seeds["modules"] = 21
        save(args.out, "modules", modules_golden(model, 21), sizes)
    elif args.tier == "b0":
        for name, (B, L, s, seed) in B0_CASES.items():
            seeds[f"b0_{name}"] = seed
            save(args.out, f"b0_{name}", b0_golden(model, B, L, s, seed), sizes)
    else:
        for name, (B, ctx, H, s, seed, nan_rows) in BENCH_CASES.items():
            seeds[f"bench_{name}"] = seed
            save(args.out, f"bench_{name}", bench_golden(model, B, ctx, H, s, seed, nan_rows), sizes)

    total = sum(p.stat().st_size for p in args.out.glob("*.pt"))
    manifest = json.loads(args.manifest.read_text()) if args.manifest.exists() else {}
    manifest.update(
        {
            "hf_repo": REPO,
            "hf_revision": REVISION,
            "safetensors_sha256": sha,
            "code": CODE,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "threads": THREADS,
            "pip_freeze": subprocess.run(
                ["uv", "pip", "freeze", "--python", sys.executable], capture_output=True, text=True
            ).stdout.split(),
            "input_generator": "tests/golden/inputs.py",
            "safetensors_stored_dtypes": stored_dtypes(st_path),
            "reference_sources_sha256": reference_sources(),
        }
    )
    manifest.setdefault("seeds", {}).update(seeds)
    manifest.setdefault("golden_bytes", {}).update(sizes)
    manifest["golden_total_bytes"] = total
    args.manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(f"total goldens: {total / 2**20:.1f} MB (budget {GOLDEN_BUDGET / 2**20:.0f} MB)")
    if total > GOLDEN_BUDGET:
        sys.exit("golden directory over the 0.5 GB budget (D0 amendment 2)")


if __name__ == "__main__":
    main()
