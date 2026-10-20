# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host (CPU, transformers) reference for single GLM-5.3-Flash decoder layers.

Loads only the tensors of the requested layers straight from the safetensors shards of the HF
checkpoint, dequantizes the fp8 block-scaled groups to bf16 (the same utility the DeepSeek port
uses), applies the checkpoint -> module key conversion transformers performs for ``glm5_next``,
and runs the real ``Glm5NextTextDecoderLayer`` modules with a ``DynamicCache``.

Two entry points:

* ``build_layer(layer_idx)``: the HF module for one layer with real weights, in the dtypes
  ``from_pretrained`` would give it (bf16 except the strict-fp32 groups).
* ``main``: run a real prompt through the embedding and layers ``0..max_layer`` (prefill), then
  ``num_decode`` single-token decode steps, and dump every layer's input/output residual streams
  plus the autoregressive state to ``bringup/references/glm53_flash``. The TTNN tests compare
  against those files, so the device never needs the host modules at test time.
* ``main --generate N``: greedy generation of ``N`` tokens from ``--prompt`` through the whole
  stack (embedding, all 45 layers streamed one module at a time per token, ``hc_head`` mean, final
  norm, ``lm_head``, argmax), the host side of the full-model test (``tests/test_full_model.py``).
  The model's own bf16 path is used for the head; the result is resumable (``generate_state.pt``).

Residual contract (from ``Glm5NextTextModel.forward``): the layer input/output is the mHC stream
tensor ``[B, S, hc_mult=4, hidden=4096]``; layer 0's input is the embedding repeated 4x.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint import (  # noqa: E402
    EMBED_KEY,
    FINAL_NORM_KEY,
    LM_HEAD_KEY,
    default_snapshot_dir,
    load_layer_raw,
    load_tensors,
    shard_headers,
)

# Glm5NextPreTrainedModel._keep_in_fp32_modules_strict
STRICT_FP32 = ("e_score_correction_bias", "conv1d", "dt_bias", "A_log")


def cast_like_from_pretrained(module):
    """bf16 for every parameter/buffer except the strict-fp32 groups, which keep their checkpoint
    precision. (Casting the whole module to bf16 and re-casting those tensors to fp32 afterwards
    would already have rounded them: the router bias at ~5.5 only resolves 0.03 in bf16, which
    flips the top-8 on ~half of real tokens.)"""
    for name, p in list(module.named_parameters()) + list(module.named_buffers()):
        if any(s in name for s in STRICT_FP32):
            p.data = p.data.to(torch.float32)
        else:
            p.data = p.data.to(torch.bfloat16)
    return module.eval()


def convert_to_hf_module_keys(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Checkpoint naming -> Glm5NextTextDecoderLayer state_dict naming (transformers glm5_next mapping)."""
    sd: dict[str, torch.Tensor] = {}
    renames = [
        (r"^self_attn\.f_a_proj\.", "self_attn.forget_gate.f_a_proj."),
        (r"^self_attn\.f_b_proj\.", "self_attn.forget_gate.f_b_proj."),
        (r"^self_attn\.dt_bias$", "self_attn.forget_gate.dt_bias"),
        (r"^self_attn\.A_log$", "self_attn.forget_gate.A_log"),
        (r"^hc_attn_fn$", "attn_hc.fn"),
        (r"^hc_attn_base$", "attn_hc.base"),
        (r"^hc_attn_scale$", "attn_hc.scale"),
        (r"^hc_ffn_fn$", "ffn_hc.fn"),
        (r"^hc_ffn_base$", "ffn_hc.base"),
        (r"^hc_ffn_scale$", "ffn_hc.scale"),
    ]
    experts: dict[int, dict[str, torch.Tensor]] = {}
    conv = {}
    for k, t in raw.items():
        m = re.match(r"^mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$", k)
        if m:
            experts.setdefault(int(m.group(1)), {})[m.group(2)] = t
            continue
        m = re.match(r"^self_attn\.([qkv])_conv1d\.weight$", k)
        if m:
            conv[m.group(1)] = t
            continue
        nk = k
        for pat, rep in renames:
            nk = re.sub(pat, rep, nk)
        sd[nk] = t
    if experts:
        n = max(experts) + 1
        assert sorted(experts) == list(range(n)), "expert index gap"
        sd["mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([experts[e]["gate_proj"], experts[e]["up_proj"]], dim=0) for e in range(n)]
        )
        sd["mlp.experts.down_proj"] = torch.stack([experts[e]["down_proj"] for e in range(n)])
    if conv:
        sd["self_attn.conv1d.weight"] = torch.cat([conv["q"], conv["k"], conv["v"]], dim=0)
    return sd


# --------------------------------------------------------------------------------------------
# HF modules
# --------------------------------------------------------------------------------------------
def load_config(snapshot: Path):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(snapshot))
    text = cfg.text_config
    # The checkpoint's config.json names the DSA layers "deepseek_sparse_attention"; the modeling
    # code's Glm5NextTextModel.forward keys its mask mapping on "indexed_attention" (the config
    # __post_init__ only renames "full_attention"). A layer-level run needs the same key, so
    # normalize here; the decoder layer itself only distinguishes "linear_attention" from the rest.
    text.layer_types = ["indexed_attention" if t != "linear_attention" else t for t in text.layer_types]
    return cfg, text


def build_layer(snapshot: Path, text_config, layer_idx: int, headers=None, raw=None):
    """``raw``: the layer's ``load_layer_raw`` dict when the caller already holds it (a device
    test loads it once for the HF and the TT side); loaded here otherwise."""
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    if raw is None:
        raw = load_layer_raw(snapshot, layer_idx, headers)
    sd = convert_to_hf_module_keys(raw)
    with torch.device("meta"):
        layer = Glm5NextTextDecoderLayer(text_config, layer_idx)
    layer = layer.to_empty(device="cpu")
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)  # no second copy of the experts
    missing = [k for k in missing]
    if missing or unexpected:
        raise RuntimeError(f"layer {layer_idx}: missing={missing} unexpected={unexpected}")
    cast_like_from_pretrained(layer)
    return layer, sd


def layer_stats(sd: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Per-tensor name/shape/dtype/mean/std, the synthetic-weight contract for CI tests."""
    out = {}
    for k, t in sd.items():
        tf = t.float()
        out[k] = {
            "shape": list(t.shape),
            "dtype": str(t.dtype).replace("torch.", ""),
            "mean": float(tf.mean()),
            "std": float(tf.std()) if tf.numel() > 1 else 0.0,
            "absmax": float(tf.abs().max()),
        }
    return out


# --------------------------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def run_layers(
    snapshot: Path,
    layers: list[int],
    prompt_ids: torch.Tensor,
    num_decode: int,
    out_dir: Path,
    text_config,
    *,
    stream: bool = False,
    perturb: float = 0.0,
):
    """Prefill ``prompt_ids`` then ``num_decode`` greedy-free decode steps (next token = a fixed id
    sequence, since no LM head is run) through the given consecutive layers, dumping per-layer
    tensors. Layers must start at 0 so layer 0 can take the real embedding.

    ``stream=True`` runs layer by layer (prefill + all decode steps of layer i, then layer i+1) and
    frees each module afterwards, so all 45 layers fit in host RAM (an MoE layer is ~15 GB bf16).
    The per-layer cache entries are independent, so the dumps are identical to the stacked order."""
    from transformers import DynamicCache

    headers = shard_headers(snapshot)
    embed = load_tensors(snapshot, [EMBED_KEY], headers)[EMBED_KEY]
    hc = text_config.hc_mult
    B, T = prompt_ids.shape
    mask = torch.ones(B, T, dtype=torch.bool)
    cache = DynamicCache(config=text_config)

    x = embed[prompt_ids].unsqueeze(2).expand(-1, -1, hc, -1).contiguous()  # [B, T, 4, H] bf16
    if perturb:
        # control for chain-sensitivity studies: relative Gaussian noise on the layer-0 input (the size of a
        # bf16 / bfp8 device error), everything downstream unchanged
        g = torch.Generator().manual_seed(1234)
        x = (x.float() * (1.0 + perturb * torch.randn(x.shape, generator=g))).to(x.dtype)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt_ids": prompt_ids, "x0": x}, out_dir / "prefill_input.pt")
    if stream:
        return _run_layers_streaming(snapshot, layers, x, T, num_decode, out_dir, text_config, headers, embed, cache)
    modules = {}
    index_ids: dict[int, list] = {}
    topk = None
    for li in layers:
        t0 = time.time()
        layer, sd = build_layer(snapshot, text_config, li, headers)
        modules[li] = layer
        idx_mod = getattr(layer.self_attn, "indexer", None)
        if idx_mod is not None:
            # the layer returns topk_indices only for "shared" successors; record the indexer output directly
            index_ids[li] = []
            idx_mod.register_forward_hook(lambda m, i, o, li=li: index_ids[li].append(o.detach().clone()))
        (out_dir / f"layer{li}_weight_stats.json").write_text(json.dumps(layer_stats(sd), indent=1))
        x_in = x
        x, topk = layer(x, attention_mask=mask, past_key_values=cache, use_cache=True, prev_topk_indices=topk)
        rec = {"x_in": x_in, "x_out": x, "topk_indices": topk}
        if li in index_ids:
            rec["index_ids"] = index_ids[li][-1]  # [B, S, W] int32, -1 = unused slot
        rec.update(_cache_snapshot(cache, li))
        torch.save(rec, out_dir / f"layer{li}_prefill.pt")
        print(f"layer {li} prefill T={T}: {time.time() - t0:.1f}s  out std {x.float().std():.4f}", flush=True)

    # decode steps: feed fixed follow-on token ids (the harness has no LM head)
    gen = torch.Generator().manual_seed(0)
    for step in range(num_decode):
        tok = torch.randint(0, text_config.vocab_size, (B, 1), generator=gen)
        x = embed[tok].unsqueeze(2).expand(-1, -1, hc, -1).contiguous()
        mask_d = torch.ones(B, 1, dtype=torch.bool)
        topk = None
        for li in layers:
            x_in = x
            x, topk = modules[li](
                x, attention_mask=mask_d, past_key_values=cache, use_cache=True, prev_topk_indices=topk
            )
            rec = {"token": tok, "position": T + step, "x_in": x_in, "x_out": x, "topk_indices": topk}
            if li in index_ids:
                rec["index_ids"] = index_ids[li][-1]
            rec.update(_cache_snapshot(cache, li))
            torch.save(rec, out_dir / f"layer{li}_decode{step}.pt")
        print(f"decode step {step} (pos {T + step}) done", flush=True)


@torch.no_grad()
def _run_layers_streaming(snapshot, layers, x_pre, T, num_decode, out_dir, text_config, headers, embed, cache):
    """Layer-by-layer order of ``run_layers`` (same token ids, same records), one module in memory."""
    import gc

    hc = text_config.hc_mult
    B = x_pre.shape[0]
    mask = torch.ones(B, T, dtype=torch.bool)
    mask_d = torch.ones(B, 1, dtype=torch.bool)
    gen = torch.Generator().manual_seed(0)
    toks = [torch.randint(0, text_config.vocab_size, (B, 1), generator=gen) for _ in range(num_decode)]
    x_dec = [embed[t].unsqueeze(2).expand(-1, -1, hc, -1).contiguous() for t in toks]
    topk_pre = None
    topk_dec = [None] * num_decode
    for li in layers:
        t0 = time.time()
        layer, sd = build_layer(snapshot, text_config, li, headers)
        (out_dir / f"layer{li}_weight_stats.json").write_text(json.dumps(layer_stats(sd), indent=1))
        del sd
        index_ids: list = []
        idx_mod = getattr(layer.self_attn, "indexer", None)
        if idx_mod is not None:
            idx_mod.register_forward_hook(lambda m, i, o: index_ids.append(o.detach().clone()))
        x_in = x_pre
        x_pre, topk_pre = layer(
            x_in, attention_mask=mask, past_key_values=cache, use_cache=True, prev_topk_indices=topk_pre
        )
        rec = {"x_in": x_in, "x_out": x_pre, "topk_indices": topk_pre}
        if idx_mod is not None:
            rec["index_ids"] = index_ids[-1]
        rec.update(_cache_snapshot(cache, li))
        torch.save(rec, out_dir / f"layer{li}_prefill.pt")
        t1 = time.time()
        for step in range(num_decode):
            x_in = x_dec[step]
            x_dec[step], topk_dec[step] = layer(
                x_in, attention_mask=mask_d, past_key_values=cache, use_cache=True, prev_topk_indices=topk_dec[step]
            )
            rec = {
                "token": toks[step],
                "position": T + step,
                "x_in": x_in,
                "x_out": x_dec[step],
                "topk_indices": topk_dec[step],
            }
            if idx_mod is not None:
                rec["index_ids"] = index_ids[-1]
            rec.update(_cache_snapshot(cache, li))
            torch.save(rec, out_dir / f"layer{li}_decode{step}.pt")
        print(
            f"layer {li} prefill T={T}: {t1 - t0:.1f}s  out std {x_pre.float().std():.4f}; "
            f"{num_decode} decode steps {time.time() - t1:.1f}s",
            flush=True,
        )
        del layer, idx_mod, index_ids, rec
        gc.collect()


def _cache_snapshot(cache, li: int) -> dict:
    """Clone the per-layer cache state so later steps do not alias it."""
    out = {}
    try:
        layer = cache.layers[li]
    except (AttributeError, IndexError):
        return out
    for attr in ("conv_states", "recurrent_states", "keys", "values", "indexer_keys"):
        v = getattr(layer, attr, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = [t.clone() if torch.is_tensor(t) else t for t in v]
        elif torch.is_tensor(v):
            v = v.clone()
        out[f"cache_{attr}"] = v
    return out


@torch.no_grad()
def generate_streaming(
    snapshot: Path,
    text_config,
    prompt_ids: torch.Tensor,
    n: int,
    out_dir: Path,
    *,
    resume: bool = True,
    layers: list[int] | None = None,
) -> dict:
    """Greedy generation of ``n`` tokens with every layer streamed per token (prefill pass, then
    ``n - 1`` decode passes; a pass builds each of the 45 HF layers in turn, ~10 s per MoE layer on
    a warm page cache, and frees it). The head is the model's bf16 path: ``hc_head`` (stream mean)
    -> ``norm`` -> ``lm_head`` in bf16 -> argmax (first index on ties, as ``torch.argmax``).
    Writes ``generate.json`` (prompt ids, generated ids, per-step top-5 ids/logits, timings) and a
    resumable ``generate_state.pt`` (the ``DynamicCache`` + tokens so far) after every token."""
    import gc

    from transformers import DynamicCache
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead, Glm5NextTextRMSNorm

    headers = shard_headers(snapshot)
    embed = load_tensors(snapshot, [EMBED_KEY], headers)[EMBED_KEY]
    head_w = load_tensors(snapshot, [FINAL_NORM_KEY, LM_HEAD_KEY], headers)
    norm = Glm5NextTextRMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
    norm.weight.data = head_w[FINAL_NORM_KEY].to(torch.bfloat16)
    lm_w = head_w[LM_HEAD_KEY].to(torch.bfloat16)
    hc_head = Glm5NextTextHyperHead()
    hc = text_config.hc_mult
    layers = (
        list(range(text_config.num_hidden_layers)) if layers is None else layers
    )  # a subset only for mechanics checks
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "generate_state.pt"

    def head(x_last: torch.Tensor) -> torch.Tensor:
        """x_last [1, 1, 4, 4096] bf16 -> logits [vocab] fp32 (bf16 values)."""
        h = norm(hc_head(x_last))  # bf16 [1, 1, H]
        return torch.nn.functional.linear(h, lm_w)[0, -1].float()

    def record(step: int, position: int, logits: torch.Tensor, seconds: float) -> dict:
        top = logits.topk(5)
        return {
            "step": step,
            "position": position,
            "token": int(logits.argmax()),
            "top5_ids": top.indices.tolist(),
            "top5_logits": [round(float(v), 4) for v in top.values],
            "pass_s": round(seconds, 1),
        }

    def run_pass(x: torch.Tensor, cache, mask: torch.Tensor) -> torch.Tensor:
        topk = None
        for li in layers:
            layer, sd = build_layer(snapshot, text_config, li, headers)
            del sd
            x, topk = layer(x, attention_mask=mask, past_key_values=cache, use_cache=True, prev_topk_indices=topk)
            del layer
            gc.collect()
        return x

    B, T = prompt_ids.shape
    if resume and state_file.exists():
        st = torch.load(state_file, weights_only=False)
        assert torch.equal(st["prompt_ids"], prompt_ids), "generate_state.pt belongs to another prompt"
        cache, generated, records = st["cache"], st["generated"], st["records"]
        print(f"resuming after {len(generated)} generated tokens", flush=True)
    else:
        cache, generated, records = DynamicCache(config=text_config), [], []

    def save():
        torch.save({"prompt_ids": prompt_ids, "generated": generated, "records": records, "cache": cache}, state_file)
        (out_dir / "generate.json").write_text(
            json.dumps(
                {"prompt_ids": prompt_ids[0].tolist(), "generated": generated, "records": records, "n": n}, indent=1
            )
        )

    if not generated:
        t0 = time.time()
        x = embed[prompt_ids].unsqueeze(2).expand(-1, -1, hc, -1).contiguous()
        x = run_pass(x, cache, torch.ones(B, T, dtype=torch.bool))
        logits = head(x[:, -1:])
        records.append(record(0, T - 1, logits, time.time() - t0))
        generated.append(records[-1]["token"])
        save()
        print(
            f"prefill T={T}: token {generated[-1]} ({time.time() - t0:.0f}s); top5 {records[-1]['top5_ids']}",
            flush=True,
        )
    while len(generated) < n:
        t0 = time.time()
        pos = T + len(generated) - 1  # position of the token being fed (the last generated one)
        x = embed[torch.tensor([[generated[-1]]])].unsqueeze(2).expand(-1, -1, hc, -1).contiguous()
        x = run_pass(x, cache, torch.ones(B, 1, dtype=torch.bool))
        logits = head(x)
        records.append(record(len(generated), pos, logits, time.time() - t0))
        generated.append(records[-1]["token"])
        save()
        print(
            f"decode pos {pos}: token {generated[-1]} ({time.time() - t0:.0f}s); top5 {records[-1]['top5_ids']}",
            flush=True,
        )
    return {"prompt_ids": prompt_ids[0].tolist(), "generated": generated, "records": records}


def parse_layers(spec: str) -> list[int]:
    """``"0,1,2"`` or ``"0-44"`` (inclusive) -> sorted layer indices."""
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument(
        "--layers",
        type=str,
        default=None,
        help="consecutive layers starting at 0 (a,b,c or a-b); default 0,1,2,3,4 for dumps, all layers for --generate",
    )
    ap.add_argument(
        "--stream",
        action="store_true",
        help="layer-by-layer order, one module in memory at a time (needed for all 45 layers); same dumps",
    )
    ap.add_argument("--prompt", type=str, default="The capital of France is Paris. The theory of general relativity")
    ap.add_argument("--seq-len", type=int, default=None, help="repeat/truncate the prompt to this many tokens")
    ap.add_argument("--num-decode", type=int, default=3)
    ap.add_argument(
        "--random-ids",
        type=int,
        default=None,
        metavar="SEED",
        help="use uniformly random token ids of --seq-len (non-repetitive) instead of repeating --prompt",
    )
    ap.add_argument("--out", type=Path, default=Path("bringup/references/glm53_flash"))
    ap.add_argument(
        "--perturb", type=float, default=0.0, help="relative noise on the layer-0 input (chain-sensitivity control)"
    )
    ap.add_argument(
        "--generate",
        type=int,
        default=None,
        metavar="N",
        help="greedy-generate N tokens from --prompt through all layers (streamed per token) instead of dumping",
    )
    ap.add_argument("--no-resume", action="store_true", help="--generate: ignore an existing generate_state.pt")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    snapshot = args.snapshot or default_snapshot_dir()
    cfg, text = load_config(snapshot)
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    if args.random_ids is not None:
        assert args.seq_len is not None, "--random-ids needs --seq-len"
        g = torch.Generator().manual_seed(args.random_ids)
        ids = torch.randint(0, text.vocab_size, (1, args.seq_len), generator=g)
    else:
        ids = tok(args.prompt, return_tensors="pt").input_ids
        if args.seq_len is not None:
            reps = (args.seq_len + ids.shape[1] - 1) // ids.shape[1]
            ids = ids.repeat(1, reps)[:, : args.seq_len]
    print(f"snapshot {snapshot}\nprompt tokens {ids.shape}", flush=True)
    torch.manual_seed(0)
    if args.generate is not None:
        res = generate_streaming(
            snapshot,
            text,
            ids,
            args.generate,
            args.out,
            resume=not args.no_resume,
            layers=parse_layers(args.layers) if args.layers else None,
        )
        print(f"generated ids {res['generated']}\ntext: {tok.decode(res['generated'])!r}", flush=True)
        return
    layers = parse_layers(args.layers or "0,1,2,3,4")
    assert layers[0] == 0 and layers == list(range(layers[0], layers[-1] + 1)), "layers must be 0..N"
    run_layers(snapshot, layers, ids, args.num_decode, args.out, text, stream=args.stream, perturb=args.perturb)


if __name__ == "__main__":
    main()
