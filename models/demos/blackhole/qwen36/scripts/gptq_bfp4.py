#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Offline GPTQ for Qwen3.6/3.8-27B weights stored as ttnn bfloat4_b (MLP gate/up/down, GDN in/out projections).

Runs on a CPU or CUDA box with torch + transformers (the version the tt-metal python env pins, which has
transformers.models.qwen3_5) + safetensors; no ttnn, no Tenstorrent device. See scripts/gptq_bfp4_README.md for the
full runbook (commands, resource estimates, emulator gate, eval gate, on-device validation).

Subcommands
  calib-data  tokenize + pack calibration (or held-out eval) text into an [n, seqlen] token tensor
  quantize    sequential GPTQ, layer by layer, written as a full bf16 HF checkpoint in a NEW dir + gptq_bfp4.json
  verify      every GPTQ tensor is a ttnn bfloat4_b fixed point in the device layout; every other tensor is unchanged
  kl          weight-emulated KL / top-1 vs BF16 for: deployed (all BFP8), rtn (preset dtypes, round-to-nearest),
              gptq (preset dtypes, this checkpoint); exits 1 if the gptq gate fails

The quantizer groups 16 OUTPUT rows per shared exponent (ttnn's face-row block along N of the [K, N] weight), unlike
tt-lab's requant.cpp which groups 16 INPUT columns; see utils/gptq_bfp4.py for the adaptation.
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import torch

from models.demos.blackhole.qwen36.utils import gptq_bfp4 as G

TARGETS = ("gdn_in", "gdn_out", "mlp_gateup", "mlp_down")  # also the true-sequential order inside a layer
SCRIPT_VERSION = 1


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Model / checkpoint helpers
# ---------------------------------------------------------------------------------------------------------------------


def load_text_config(model_dir):
    from transformers.models.qwen3_5 import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig.from_pretrained(model_dir)
    if (getattr(cfg, "num_experts", 0) or 0) > 0:
        sys.exit("MoE checkpoints are not supported (dense Qwen3.6/3.8-27B only)")
    return cfg


def load_model(model_dir):
    """Text-only HF model in bf16 on CPU (same classes as Qwen36ModelArgs.load_state_dict). ~54 GB RAM for 27B."""
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM

    cfg = load_text_config(model_dir)
    model = Qwen3_5ForCausalLM.from_pretrained(
        model_dir, config=cfg, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True
    )
    model.eval()
    model.requires_grad_(False)
    return model, cfg


def weight_map(model_dir):
    """{checkpoint key: shard file} for sharded or single-file safetensors checkpoints."""
    idx = Path(model_dir) / "model.safetensors.index.json"
    if idx.is_file():
        return json.loads(idx.read_text())["weight_map"]
    from safetensors import safe_open

    single = Path(model_dir) / "model.safetensors"
    with safe_open(str(single), "pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def layer_key_map(wmap):
    """{(layer, suffix): checkpoint key} for decoder-layer tensors (drops mtp.* and the vision tower). Handles both
    text-only ("model.layers.N.") and VLM ("model.language_model.layers.N.") checkpoint layouts."""
    out = {}
    for k in wmap:
        if k.startswith("mtp.") or "visual" in k:
            continue
        m = re.match(r"^(?:.*\.)?layers\.(\d+)\.(.+)$", k)
        if m:
            key = (int(m[1]), m[2])
            assert key not in out, f"ambiguous checkpoint keys {out[key]} / {k}"
            out[key] = k
    return out


def target_specs(cfg, tp):
    """Per target: layer type it applies to, stacked HF modules (one shared input), device-layout groups."""
    hid, inter = cfg.hidden_size, cfg.intermediate_size
    nk, dk = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    nv, dv = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    key_dim, value_dim = nk * dk, nv * dv
    assert inter % (G.BLOCK * tp) == 0 and hid % (G.BLOCK * tp) == 0, "per-device N slice not 16-aligned"
    return {
        "gdn_in": dict(
            layer_type="linear_attention",
            mods=["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.in_proj_a", "linear_attn.in_proj_b"],
            groups=lambda: G.gdn_in_groups(2 * key_dim + value_dim, value_dim, nv, tp),
        ),
        "gdn_out": dict(
            layer_type="linear_attention", mods=["linear_attn.out_proj"], groups=lambda: G.consecutive_groups(hid)
        ),
        "mlp_gateup": dict(
            layer_type=None, mods=["mlp.gate_proj", "mlp.up_proj"], groups=lambda: G.consecutive_groups(2 * inter)
        ),
        "mlp_down": dict(layer_type=None, mods=["mlp.down_proj"], groups=lambda: G.consecutive_groups(hid)),
    }


def applies(spec, layer_type):
    return spec["layer_type"] is None or spec["layer_type"] == layer_type


def get_mod(layer, dotted):
    m = layer
    for p in dotted.split("."):
        m = getattr(m, p)
    return m


# ---------------------------------------------------------------------------------------------------------------------
# Layer-by-layer execution (inputs of layer 0 + per-layer-type kwargs captured once through the HF text model)
# ---------------------------------------------------------------------------------------------------------------------


class _Catcher(torch.nn.Module):
    """Identity stand-in for a decoder layer: records its inputs; the HF loop runs embed -> catchers -> norm only."""

    def __init__(self, idx, sink):
        super().__init__()
        self.idx, self.sink = idx, sink

    def forward(self, hidden_states, **kwargs):
        if self.idx == 0:
            self.sink["h"].append(hidden_states.detach())
        self.sink["kw"].setdefault(self.idx, kwargs)
        return hidden_states


def capture_layer_inputs(model, tokens, act_device):
    """tokens LongTensor [n, L] -> (list of layer-0 inputs [1, L, H] on act_device, {layer_idx: kwargs})."""
    layers = model.model.layers
    saved = list(layers)
    sink = {"h": [], "kw": {}}
    try:
        for i in range(len(layers)):
            layers[i] = _Catcher(i, sink)
        with torch.no_grad():
            for i in range(tokens.shape[0]):
                model.model(input_ids=tokens[i : i + 1], use_cache=False)
    finally:
        for i, layer in enumerate(saved):
            layers[i] = layer
    hs = [h.to(act_device) for h in sink["h"]]
    return hs, sink["kw"]


def _to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_to(o, device) for o in obj)
    if isinstance(obj, list):
        return [_to(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: _to(v, device) for k, v in obj.items()}
    return obj


def run_layer(layer, hs, kw, device, act_device):
    out = []
    with torch.no_grad():
        for h in hs:
            o = layer(h.to(device), **kw)
            out.append((o[0] if isinstance(o, tuple) else o).to(act_device))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# calib-data
# ---------------------------------------------------------------------------------------------------------------------


def _docs_from_jsonl(path, tok):
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if "messages" in r:
            yield tok.apply_chat_template(r["messages"], tokenize=False)
        elif "prompt" in r and ("response" in r or "completion" in r):
            msgs = [
                {"role": "user", "content": r["prompt"]},
                {"role": "assistant", "content": r.get("response", r.get("completion"))},
            ]
            yield tok.apply_chat_template(msgs, tokenize=False)
        elif "text" in r:
            yield r["text"]


def _docs_from_hf(spec):
    """name[:config]:split:field, streamed."""
    from datasets import load_dataset

    parts = spec.split(":")
    if len(parts) == 3:
        name, split, field = parts
        conf = None
    else:
        name, conf, split, field = parts
    for r in load_dataset(name, conf, split=split, streaming=True):
        yield r[field]


def cmd_calib_data(a):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    eos = tok.eos_token_id
    rng = random.Random(a.seed)
    streams = [("jsonl:" + p, _docs_from_jsonl(p, tok)) for p in a.jsonl] + [
        ("hf:" + s, _docs_from_hf(s)) for s in a.hf_dataset
    ]
    if not streams:
        sys.exit("give at least one --jsonl or --hf-dataset")
    need = a.nsamples * a.seqlen
    per_stream = -(-need // len(streams))  # equal token share per source
    samples, used = [], {}
    for name, docs in streams:
        buf, n_docs = [], 0
        for d in docs:
            if rng.random() < a.skip_frac:  # different --seed/--skip-frac => disjoint-ish held-out sets
                continue
            buf += tok(d, add_special_tokens=False)["input_ids"] + [eos]
            n_docs += 1
            if len(buf) >= per_stream:
                break
        used[name] = dict(docs=n_docs, tokens=len(buf))
        samples += [buf[i : i + a.seqlen] for i in range(0, len(buf) - a.seqlen + 1, a.seqlen)]
    rng.shuffle(samples)
    if len(samples) < a.nsamples:
        sys.exit(f"only {len(samples)} full windows of {a.seqlen}; add data or lower --nsamples")
    ids = torch.tensor(samples[: a.nsamples], dtype=torch.long)
    h = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    torch.save({"input_ids": ids, "sources": used, "sha256": h, "seed": a.seed}, a.out)
    log(f"wrote {a.out}: {tuple(ids.shape)} sha256 {h[:16]} sources {used}")


# ---------------------------------------------------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------------------------------------------------


def _fingerprint(a, calib_sha, targets, layers):
    keys = dict(
        v=SCRIPT_VERSION,
        calib=calib_sha,
        tp=a.tp,
        targets=sorted(targets),
        damp=a.damp,
        block=a.blocksize,
        act_order=a.act_order,
        exp_search=a.exp_search,
        layers=sorted(layers),
    )
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:16]


def cmd_quantize(a):
    from safetensors import safe_open
    from safetensors.torch import save_file

    out = Path(a.out)
    src = Path(a.model).resolve()
    if out.resolve() == src:
        sys.exit("--out must be a new directory, not the source checkpoint")
    if (out / G.MARKER_NAME).exists() and not a.overwrite:
        sys.exit(f"{out} already holds a finished GPTQ checkpoint (use --overwrite)")
    if out.name != src.name:
        log(
            f"WARNING: --out basename {out.name!r} != {src.name!r}. The demo accuracy refs, perf targets and env38.sh key on "
            f"basename(HF_MODEL); use e.g. /data/models/gptq-bfp4-v1/{src.name}"
        )
    parts_dir = out / ".gptq_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    act_device = torch.device(a.act_device or a.device)
    targets = [t for t in TARGETS if t in a.targets.split(",")]
    assert targets, f"--targets must name some of {TARGETS}"

    calib = torch.load(a.calib)
    tokens = calib["input_ids"]
    model, cfg = load_model(str(src))
    specs = target_specs(cfg, a.tp)
    wmap = weight_map(src)
    keymap = layer_key_map(wmap)
    layer_types = cfg.layer_types
    n_layers = cfg.num_hidden_layers
    only = set(range(n_layers)) if not a.layers else {int(x) for x in a.layers.split(",")}
    fp = _fingerprint(a, calib["sha256"], targets, only)
    log(f"calib {tuple(tokens.shape)} sha {calib['sha256'][:16]}, targets {targets}, fingerprint {fp}")

    hs, kws = capture_layer_inputs(model, tokens, act_device)
    log(f"captured layer-0 inputs for {len(hs)} samples")
    stats = []
    for L in range(n_layers):
        t0 = time.time()
        layer = model.model.layers[L].to(device)
        kw = _to(kws[L], device)
        part = parts_dir / f"layer_{L:03d}.safetensors"
        todo = [t for t in targets if applies(specs[t], layer_types[L])] if L in only else []
        done = None
        if part.exists():
            with safe_open(str(part), "pt") as f:
                if (f.metadata() or {}).get("fingerprint") == fp:
                    done = {k: f.get_tensor(k) for k in f.keys()}
        if done is not None:
            for t in todo:  # resume: install the stored result, still propagate activations
                for m in specs[t]["mods"]:
                    get_mod(layer, m).weight.data.copy_(done[keymap[(L, m + ".weight")]].to(device))
            log(f"layer {L}: resumed from {part.name}")
        elif todo:
            result, rows = {}, []
            for t in todo:
                spec = specs[t]
                mods = [get_mod(layer, m) for m in spec["mods"]]
                K = mods[0].in_features
                H = torch.zeros(K, K, dtype=torch.float32, device=device)
                ntok = [0]

                def hook(_m, inp, _o, H=H, ntok=ntok):
                    x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                    H.addmm_(x.t(), x)
                    ntok[0] += x.shape[0]

                handle = mods[0].register_forward_hook(hook)
                run_layer(layer, hs, kw, device, act_device)  # outputs dropped; only the hook matters
                handle.remove()
                H /= max(ntok[0], 1)
                W = torch.cat([m.weight.data.float() for m in mods], dim=0)
                groups = spec["groups"]().to(device)
                Q = G.gptq_quantize(W, H, groups, a.blocksize, a.damp, a.act_order, a.exp_search)
                rtn = G.ttnn_bfp_roundtrip(W, groups)  # == RTN on these bf16 weights (what ttnn does today)
                ref = float(((W @ H) * W).sum())
                lq, lr = G.proxy_loss(W, Q, H) / ref, G.proxy_loss(W, rtn, H) / ref
                rows.append(dict(layer=L, target=t, rel_err_gptq=lq, rel_err_rtn=lr, tokens=ntok[0]))
                log(f"layer {L} {t}: rel output err gptq {lq:.3e} vs rtn {lr:.3e} ({lr / max(lq, 1e-30):.2f}x)")
                Qb = Q.to(torch.bfloat16)
                off = 0
                for name, m in zip(spec["mods"], mods):
                    n = m.out_features
                    m.weight.data.copy_(Qb[off : off + n])  # quantized weights feed the later sub-layers / layers
                    result[keymap[(L, name + ".weight")]] = Qb[off : off + n].cpu().contiguous()
                    off += n
                del H, W, Q, rtn
            save_file(result, str(part), metadata={"fingerprint": fp})
            stats += rows
        hs = run_layer(layer, hs, kw, device, act_device)
        model.model.layers[L] = layer.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        log(f"layer {L} ({layer_types[L]}) done in {time.time() - t0:.1f}s")

    export(
        src,
        out,
        parts_dir,
        wmap,
        dict(
            format=G.MARKER_FORMAT,
            tp=a.tp,
            targets=targets,
            layers=sorted(only),
            source=str(src),
            source_weight_map_sha256=hashlib.sha256(json.dumps(wmap, sort_keys=True).encode()).hexdigest(),
            calib_sha256=calib["sha256"],
            calib_shape=list(tokens.shape),
            calib_sources=calib.get("sources"),
            damp=a.damp,
            blocksize=a.blocksize,
            act_order=a.act_order,
            exp_search=a.exp_search,
            fingerprint=fp,
            created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
        stats,
    )
    if not a.keep_parts:
        shutil.rmtree(parts_dir)
    log(f"done: {out}")


def export(src, out, parts_dir, wmap, marker, stats):
    """Full HF checkpoint in `out`: same shard names/index, GPTQ tensors replaced, everything else copied byte-wise
    (tensors re-serialized unchanged). The marker is written last: a dir without it is NOT a GPTQ checkpoint."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    replaced = {}
    for p in sorted(parts_dir.glob("layer_*.safetensors")):
        with safe_open(str(p), "pt") as f:
            if (f.metadata() or {}).get("fingerprint") != marker["fingerprint"]:
                continue
            for k in f.keys():
                replaced[k] = p
    (out / G.MARKER_NAME).unlink(missing_ok=True)
    for name in sorted(os.listdir(src)):
        s = src / name
        if s.is_file() and not name.endswith(".safetensors"):
            shutil.copy2(s, out / name)
    shards = sorted(set(wmap.values()))
    for shard in shards:
        tensors = {}
        with safe_open(str(src / shard), "pt") as f:
            meta = f.metadata() or {"format": "pt"}
            for k in f.keys():
                t = f.get_tensor(k)
                if k in replaced:
                    with safe_open(str(replaced[k]), "pt") as g:
                        q = g.get_tensor(k)
                    assert q.shape == t.shape and q.dtype == t.dtype, (k, q.shape, t.shape, q.dtype, t.dtype)
                    t = q
                tensors[k] = t
        save_file(tensors, str(out / shard), metadata=meta)
        log(f"wrote {shard} ({sum(k in replaced for k in tensors)} GPTQ tensors)")
    marker["tensors"] = sorted(replaced)
    marker["stats"] = stats
    tmp = out / (G.MARKER_NAME + ".tmp")
    tmp.write_text(json.dumps(marker, indent=1))
    os.replace(tmp, out / G.MARKER_NAME)


# ---------------------------------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------------------------------


def cmd_verify(a):
    from safetensors import safe_open

    out, src = Path(a.out), Path(a.model)
    meta = G.read_marker(out)
    assert meta is not None, f"{out} has no {G.MARKER_NAME}"
    cfg = load_text_config(str(out))
    specs = target_specs(cfg, meta["tp"])
    wmap = weight_map(out)
    keymap = layer_key_map(wmap)
    gptq = set(meta["tensors"])

    def get(d, k):
        with safe_open(str(Path(d) / wmap[k]), "pt") as f:
            return f.get_tensor(k)

    bad = 0
    for L in range(cfg.num_hidden_layers):
        for t, spec in specs.items():
            keys = [keymap.get((L, m + ".weight")) for m in spec["mods"]]
            if not all(k in gptq for k in keys):
                assert not any(k in gptq for k in keys), f"layer {L} {t}: partial target"
                continue
            W = torch.cat([get(out, k) for k in keys], dim=0)
            rt = G.ttnn_bfp_roundtrip(W, spec["groups"]())
            ok = torch.equal(rt, W.float())
            bad += not ok
            if not ok or a.verbose:
                log(f"layer {L} {t}: {'OK' if ok else 'NOT A BFP4 FIXED POINT'}")
    if not a.quick:
        for k in wmap:
            if k not in gptq and not torch.equal(get(out, k), get(src, k)):
                log(f"CHANGED non-GPTQ tensor {k}")
                bad += 1
    log(f"verify: {len(gptq)} GPTQ tensors, {bad} problems")
    sys.exit(1 if bad else 0)


# ---------------------------------------------------------------------------------------------------------------------
# kl: weight-emulated accuracy gate (activations stay bf16 torch; weights exactly as ttnn stores them)
# ---------------------------------------------------------------------------------------------------------------------

ATTN_MODS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]


def _variant_weights(variant, L, layer, layer_type, specs, gptq_targets, gptq_tensor):
    """{module: emulated weight} for one layer. deployed: every decoder linear BFP8 (today's serving defaults);
    rtn / gptq: the QWEN36_GPTQ_BFP4 preset (targets BFP4, attention + rest BFP8); gptq uses the checkpoint values."""
    out = {}
    lin = {m: get_mod(layer, m) for m in (ATTN_MODS if layer_type == "full_attention" else [])}
    for t, spec in specs.items():
        if not applies(spec, layer_type):
            continue
        mods = [get_mod(layer, m) for m in spec["mods"]]
        groups = spec["groups"]().to(mods[0].weight.device)
        if variant != "deployed" and t in gptq_targets:
            if variant == "gptq":
                W = torch.cat([gptq_tensor(L, m).to(mods[0].weight.device) for m in spec["mods"]], 0)
            else:
                W = torch.cat([m.weight.data for m in mods], 0)
            Q = G.ttnn_bfp_roundtrip(W, groups, G.BFP4_MANT_BITS)
        else:
            Q = G.ttnn_bfp_roundtrip(torch.cat([m.weight.data for m in mods], 0), groups, G.BFP8_MANT_BITS)
        off = 0
        for m in mods:
            out[m] = Q[off : off + m.out_features].to(torch.bfloat16)
            off += m.out_features
    for m in lin.values():
        out[m] = G.ttnn_bfp_roundtrip(m.weight.data, G.consecutive_groups(m.out_features), G.BFP8_MANT_BITS).to(
            torch.bfloat16
        )
    return out


def cmd_kl(a):
    from safetensors import safe_open

    device = torch.device(a.device)
    act_device = torch.device(a.act_device or a.device)
    meta = G.read_marker(a.gptq)
    assert meta is not None, f"{a.gptq} has no {G.MARKER_NAME}"
    tokens = torch.load(a.eval)["input_ids"][: a.nsamples]
    model, cfg = load_model(a.model)
    specs = target_specs(cfg, meta["tp"])
    gmap = weight_map(a.gptq)
    keymap = layer_key_map(gmap)
    in_gptq = set(meta["tensors"])

    def gptq_tensor(L, mod):  # lazy: one layer's GPTQ tensors in RAM at a time
        k = keymap[(L, mod + ".weight")]
        assert k in in_gptq, f"{k} is not a GPTQ tensor"
        with safe_open(str(Path(a.gptq) / gmap[k]), "pt") as f:
            return f.get_tensor(k)

    variants = a.variants.split(",")
    hs0, kws = capture_layer_inputs(model, tokens, act_device)
    streams = {"bf16": hs0, **{v: [h.clone() for h in hs0] for v in variants}}
    for L in range(cfg.num_hidden_layers):
        layer = model.model.layers[L].to(device)
        kw = _to(kws[L], device)
        streams["bf16"] = run_layer(layer, streams["bf16"], kw, device, act_device)
        for v in variants:
            new = _variant_weights(v, L, layer, cfg.layer_types[L], specs, set(meta["targets"]), gptq_tensor)
            orig = {m: m.weight.data for m in new}
            for m, w in new.items():
                m.weight.data = w
            streams[v] = run_layer(layer, streams[v], kw, device, act_device)
            for m, w in orig.items():
                m.weight.data = w
        model.model.layers[L] = layer.to("cpu")
        log(f"kl: layer {L} done")
    norm = model.model.norm.to(device)
    head_w = model.lm_head.weight.data.to(device)
    head_q = G.ttnn_bfp_roundtrip(head_w, G.consecutive_groups(head_w.shape[0]), G.BFP8_MANT_BITS).to(torch.bfloat16)
    res = {v: dict(kl=[], agree=0) for v in variants}
    n_pos, ref_next = 0, 0
    with torch.no_grad():
        for i in range(len(hs0)):
            ids = tokens[i, a.skip + 1 :].to(device)
            for s in range(a.skip, tokens.shape[1] - 1, a.chunk):
                e = min(s + a.chunk, tokens.shape[1] - 1)
                lr = torch.log_softmax((norm(streams["bf16"][i][0, s:e].to(device)) @ head_w.t()).float(), -1)
                am = lr.argmax(-1)
                ref_next += int((am == ids[s - a.skip : e - a.skip]).sum())
                for v in variants:
                    lq = torch.log_softmax((norm(streams[v][i][0, s:e].to(device)) @ head_q.t()).float(), -1)
                    res[v]["kl"].append((lr.exp() * (lr - lq)).sum(-1).cpu())
                    res[v]["agree"] += int((lq.argmax(-1) == am).sum())
                n_pos += e - s
    summary = {"positions": n_pos, "bf16_next_token_acc": ref_next / n_pos, "variants": {}}
    for v in variants:
        kl = torch.cat(res[v]["kl"])
        summary["variants"][v] = dict(
            kl_mean=float(kl.mean()),
            kl_p99=float(kl.quantile(0.99)),
            kl_max=float(kl.max()),
            top1_agree=res[v]["agree"] / n_pos,
        )
    gate = None
    if "gptq" in variants and "deployed" in variants:
        g, d = summary["variants"]["gptq"], summary["variants"]["deployed"]
        gate = dict(
            top1_drop_pt=100 * (d["top1_agree"] - g["top1_agree"]),
            kl_vs_rtn=(g["kl_mean"] / summary["variants"]["rtn"]["kl_mean"]) if "rtn" in variants else None,
        )
        gate["pass"] = gate["top1_drop_pt"] <= a.max_top1_drop and (
            gate["kl_vs_rtn"] is None or gate["kl_vs_rtn"] <= a.max_kl_vs_rtn
        )
        summary["gate"] = gate
    print(json.dumps(summary, indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=1))
    if summary["bf16_next_token_acc"] < 0.3:
        log("WARNING: bf16 reference next-token accuracy < 30% -- the layer loop or eval data is suspect")
    sys.exit(0 if gate is None or gate["pass"] else 1)


# ---------------------------------------------------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    dflt_dev = "cuda" if torch.cuda.is_available() else "cpu"

    c = sub.add_parser("calib-data")
    c.add_argument("--model", required=True, help="checkpoint dir (tokenizer + chat template)")
    c.add_argument("--out", required=True)
    c.add_argument("--jsonl", action="append", default=[], help="jsonl with messages | prompt+response | text")
    c.add_argument("--hf-dataset", action="append", default=[], help="name[:config]:split:field (streamed)")
    c.add_argument("--nsamples", type=int, default=128)
    c.add_argument("--seqlen", type=int, default=2048)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--skip-frac", type=float, default=0.0, help="randomly skip this fraction of docs (held-out sets)")

    q = sub.add_parser("quantize")
    q.add_argument("--model", required=True, help="source bf16 HF checkpoint (Qwen3.8-27B)")
    q.add_argument("--calib", required=True, help="calib-data output .pt")
    q.add_argument("--out", required=True, help="NEW checkpoint dir")
    q.add_argument(
        "--tp", type=int, default=8, help="TP degree the device layout is calibrated for (8 on Galaxy DP4xTP8)"
    )
    q.add_argument("--targets", default=",".join(TARGETS))
    q.add_argument("--layers", default=None, help="comma list of layers to quantize (smoke tests); others copied")
    q.add_argument("--damp", type=float, default=0.01)
    q.add_argument("--blocksize", type=int, default=128)
    q.add_argument("--no-act-order", dest="act_order", action="store_false")
    q.add_argument("--exp-search", action="store_true", help="tt-lab style per-block exponent E vs E-1 search")
    q.add_argument("--device", default=dflt_dev)
    q.add_argument("--act-device", default=None, help="where calibration activations live (default --device)")
    q.add_argument("--keep-parts", action="store_true")
    q.add_argument("--overwrite", action="store_true")

    v = sub.add_parser("verify")
    v.add_argument("--model", required=True, help="source checkpoint")
    v.add_argument("--out", required=True, help="GPTQ checkpoint")
    v.add_argument("--quick", action="store_true", help="skip the unchanged-tensor comparison")
    v.add_argument("--verbose", action="store_true")

    k = sub.add_parser("kl")
    k.add_argument("--model", required=True, help="source bf16 checkpoint (reference + deployed/rtn variants)")
    k.add_argument("--gptq", required=True, help="GPTQ checkpoint")
    k.add_argument("--eval", required=True, help="held-out calib-data .pt (NOT the calibration set)")
    k.add_argument("--nsamples", type=int, default=16)
    k.add_argument("--variants", default="deployed,rtn,gptq")
    k.add_argument("--skip", type=int, default=32, help="ignore the first positions of each sample")
    k.add_argument("--chunk", type=int, default=256)
    k.add_argument("--max-top1-drop", type=float, default=1.0, help="gate: pt of top-1 agreement lost vs deployed")
    k.add_argument("--max-kl-vs-rtn", type=float, default=0.6, help="gate: gptq mean KL / rtn mean KL")
    k.add_argument("--json", default=None)
    k.add_argument("--device", default=dflt_dev)
    k.add_argument("--act-device", default=None)

    a = p.parse_args(argv)
    torch.manual_seed(0)
    {"calib-data": cmd_calib_data, "quantize": cmd_quantize, "verify": cmd_verify, "kl": cmd_kl}[a.cmd](a)


if __name__ == "__main__":
    main()
