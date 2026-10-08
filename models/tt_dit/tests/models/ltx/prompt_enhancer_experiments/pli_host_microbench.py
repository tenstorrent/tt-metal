# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Phase 0: CPU-only microbenchmark of the Gemma-4-E2B host PLI computation.

No device, no model init. Loads the three PLI weights (plus the main embedding
table) straight from the HF snapshot and times a faithful standalone copy of
``Gemma4Model.compute_host_pli`` / ``_compute_per_layer_inputs``
(models/demos/gemma4/tt/model.py), token by token, the way decode calls it.

Arms:
  current        proj_w.float() fresh on every token (what runs today, model.py:748)
  legacy-double  ``compute_host_embeddings`` -> ``compute_host_pli``: the decode call
                 chain at model.py:2222, which does the main-embedding lookup twice
  cached-fp32    quick win: convert the projection to fp32 once at load
  bf16-linear    quick win: run the linear in bf16, norm still fp32

Prints ms/token (median over N tokens after warmup) per arm, a component
breakdown of the current arm, and max |delta| of each quick win vs current.

Results also land in $PLI_MICROBENCH_JSON if set.
"""

import glob
import json
import os
import statistics
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

SNAPSHOT = os.environ.get(
    "E2B_SNAPSHOT",
    "/mnt/models/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7",
)
N_TOKENS = int(os.environ.get("PLI_BENCH_TOKENS", "50"))
WARMUP = int(os.environ.get("PLI_BENCH_WARMUP", "5"))
SEED = int(os.environ.get("PLI_BENCH_SEED", "10"))


def load_weights(snapshot):
    """The four tensors, bf16 on CPU, exactly as Gemma4Model keeps them (model.py:438-457)."""
    want = {
        "embed_tokens_per_layer": None,
        "per_layer_model_projection": None,
        "per_layer_projection_norm": None,
        "embed_tokens": None,
    }
    for shard in sorted(glob.glob(os.path.join(snapshot, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                for name in want:
                    if key.endswith(f"{name}.weight") and not (
                        name == "embed_tokens" and "per_layer" in key
                    ):
                        want[name] = f.get_tensor(key)
    missing = [k for k, v in want.items() if v is None]
    if missing:
        raise SystemExit(f"missing tensors in {snapshot}: {missing}")
    return want


def load_eps(snapshot):
    with open(os.path.join(snapshot, "config.json")) as f:
        cfg = json.load(f)
    for scope in (cfg, cfg.get("text_config", {})):
        if "rms_norm_eps" in scope:
            return scope["rms_norm_eps"]
    raise SystemExit("rms_norm_eps not found in config.json")


def main():
    t0 = time.perf_counter()
    w = load_weights(SNAPSHOT)
    eps = load_eps(SNAPSHOT)
    load_s = time.perf_counter() - t0

    embed_w = w["embed_tokens_per_layer"]  # [vocab, full_n_layers * pli]
    proj_w = w["per_layer_model_projection"]  # [full_n_layers * pli, hidden]
    norm_w = w["per_layer_projection_norm"]  # [pli]
    main_w = w["embed_tokens"]  # [vocab, hidden]

    pli_size = norm_w.shape[0]
    full_n_layers = embed_w.shape[-1] // pli_size
    hidden = main_w.shape[-1]
    vocab = embed_w.shape[0]
    print(
        f"loaded in {load_s:.1f}s: vocab={vocab} hidden={hidden} "
        f"layers={full_n_layers} pli={pli_size} "
        f"embed_tokens_per_layer={tuple(embed_w.shape)} ({embed_w.numel() * 2 / 2**30:.2f} GB bf16)"
    )

    # Scales as the model sets them (model.py: embed_scale, per_layer_*_scale)
    embed_scale = hidden**0.5
    per_layer_embed_scale = pli_size**0.5
    per_layer_model_projection_scale = hidden**-0.5
    per_layer_input_scale = 2**-0.5

    def pli_from(token_tensor, embeds, proj, bf16_linear=False):
        """_compute_per_layer_inputs body, minus the per-call .float() when `proj` is pre-converted."""
        pli_embed = F.embedding(token_tensor.long(), embed_w) * per_layer_embed_scale
        pli_embed = pli_embed.reshape(*token_tensor.shape, full_n_layers, pli_size)
        if bf16_linear:
            pli_proj = F.linear(embeds.to(torch.bfloat16), proj) * per_layer_model_projection_scale
        else:
            pli_proj = F.linear(embeds.float(), proj) * per_layer_model_projection_scale
        pli_proj = pli_proj.reshape(*embeds.shape[:-1], full_n_layers, pli_size)
        pli_proj_f = pli_proj.float()
        var = pli_proj_f.pow(2).mean(-1, keepdim=True)
        pli_proj = (pli_proj_f * torch.rsqrt(var + eps) * norm_w.float()).to(pli_proj.dtype)
        out = (pli_proj + pli_embed.float()) * per_layer_input_scale
        return torch.stack(
            [out[:, :, i, :].to(torch.bfloat16) for i in range(full_n_layers)], dim=2
        )

    def arm_current(tok):
        t = torch.tensor([[tok]], dtype=torch.long)
        embeds = F.embedding(t, main_w).float() * embed_scale  # compute_host_pli main lookup
        return pli_from(t.int(), embeds, proj_w.float())  # fresh fp32 copy per token

    def arm_legacy_double(tok):
        t = torch.tensor([[tok]], dtype=torch.long)
        _ = F.embedding(t, main_w).float() * embed_scale  # compute_host_embeddings, discarded
        return arm_current(tok)

    proj_fp32 = None

    def arm_cached_fp32(tok):
        t = torch.tensor([[tok]], dtype=torch.long)
        embeds = F.embedding(t, main_w).float() * embed_scale
        return pli_from(t.int(), embeds, proj_fp32)

    def arm_bf16(tok):
        t = torch.tensor([[tok]], dtype=torch.long)
        embeds = F.embedding(t, main_w).float() * embed_scale
        return pli_from(t.int(), embeds, proj_w, bf16_linear=True)

    torch.manual_seed(SEED)
    tokens = torch.randint(0, vocab, (WARMUP + N_TOKENS,)).tolist()

    proj_fp32 = proj_w.float()  # the cached-fp32 quick win: one conversion at "load"

    results = {"threads": torch.get_num_threads(), "n_tokens": N_TOKENS}
    arms = [
        ("current", arm_current),
        ("legacy-double", arm_legacy_double),
        ("cached-fp32", arm_cached_fp32),
        ("bf16-linear", arm_bf16),
    ]
    ref = {}
    for name, fn in arms:
        times = []
        outs = {}
        for i, tok in enumerate(tokens):
            s = time.perf_counter()
            out = fn(tok)
            dt = (time.perf_counter() - s) * 1000
            if i >= WARMUP:
                times.append(dt)
                outs[tok] = out
        med, mean = statistics.median(times), statistics.fmean(times)
        if name == "current":
            ref = outs
            delta = 0.0
        else:
            delta = max(
                (outs[t].float() - ref[t].float()).abs().max().item() for t in ref
            )
        results[name] = {"median_ms": round(med, 3), "mean_ms": round(mean, 3), "max_abs_delta": delta}
        print(f"{name:14s} {med:7.2f} ms/token median  {mean:7.2f} mean  max|Δ| vs current {delta:.3e}")

    # Component split of the current arm, medians over the same tokens
    comps = {"main_lookup": [], "pli_lookup": [], "proj_float": [], "linear_fp32": [], "norm_combine": []}
    for tok in tokens[WARMUP:]:
        t = torch.tensor([[tok]], dtype=torch.long)
        s = time.perf_counter(); embeds = F.embedding(t, main_w).float() * embed_scale
        comps["main_lookup"].append(time.perf_counter() - s)
        s = time.perf_counter(); pe = F.embedding(t.long(), embed_w) * per_layer_embed_scale
        comps["pli_lookup"].append(time.perf_counter() - s)
        s = time.perf_counter(); pw = proj_w.float()
        comps["proj_float"].append(time.perf_counter() - s)
        s = time.perf_counter(); pp = F.linear(embeds.float(), pw) * per_layer_model_projection_scale
        comps["linear_fp32"].append(time.perf_counter() - s)
        s = time.perf_counter()
        pp = pp.reshape(1, 1, full_n_layers, pli_size).float()
        var = pp.pow(2).mean(-1, keepdim=True)
        pp = pp * torch.rsqrt(var + eps) * norm_w.float()
        _ = (pp + pe.reshape(1, 1, full_n_layers, pli_size).float()) * per_layer_input_scale
        comps["norm_combine"].append(time.perf_counter() - s)
    print("component medians (current arm):")
    results["components_ms"] = {}
    for k, v in comps.items():
        ms = statistics.median(v) * 1000
        results["components_ms"][k] = round(ms, 3)
        print(f"  {k:14s} {ms:7.2f} ms")

    out_path = os.environ.get("PLI_MICROBENCH_JSON")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
