# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""BFloat4-weight decode demo: load as many real decoder layers as fit, decode.

This is a *qualitative* smoke test (not a PCC parity check). It builds a
``DeepSeekV4`` prefill front-end whose big matmul weights (attention
projections, shared MLP, routed experts, ``lm_head``) live on device in
``bfloat4_b`` so that as many of the 43 decoder layers as possible fit in the
Blackhole 32 GB DRAM at once. It then:

* tokenizes a real text string,
* embeds + expands to the ``hc_mult`` residual-stream stack,
* runs every decoder layer that fits (stopping at the first OOM, or at
  ``DEEPSEEK_V4_DECODE_LAYERS`` if set),
* collapses the streams with the final :class:`DeepSeekV4HyperHead` + shared
  RMSNorm, projects with ``lm_head``, and
* decodes the argmax tokens with the V4-Flash tokenizer so a human can eyeball
  whether the (partial-stack) next-token predictions look like plausible text.

Because only a fraction of the 43 layers fit, this is **not** the full model;
the decode is a sanity check on the machinery + bf4 precision, not a
correctness guarantee.

Dual-interpreter, like the other V4 tests, but the *system* subprocess here is
lightweight — it only tokenizes and emits the YaRN RoPE tables (no weight
loading, no HF layer forward). Two subcommands:

* ``ref  <out> [text] [model_dir]`` -> tokenize + dump RoPE tables / config.
* ``decode <json> [model_dir]``     -> load the tokenizer and print the decode.

Run it (ttnn venv), printing the decode::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache \\
    pytest -s models/experimental/deepseek_v4_flash/tests/test_bf4_decode_demo.py

Set ``DEEPSEEK_V4_DECODE_LAYERS=N`` to cap the layer count (default: as many as
fit). Set ``DEEPSEEK_V4_CACHE_DIR`` to reuse the converted ttnn weight tiles
(per-layer 256-expert dequant is skipped on a hit).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch


_DEFAULT_MODEL_DIR = "/home/smanoj/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash"
_DEFAULT_TEXT = "The capital of France is"


# --------------------------------------------------------------------------- #
# Reference / helper subprocesses (system interpreter, no ttnn).
# --------------------------------------------------------------------------- #
def _reference_main() -> None:
    """``ref <out_path> [text] [model_dir]``: tokenize + dump RoPE tables.

    Deliberately light — no checkpoint weights are loaded; only the tokenizer
    and the (config-only) YaRN rotary embedding are needed.
    """
    import importlib.metadata as _md

    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)
    # sys.path.insert(0, _CACHED_TRANSFORMERS)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tt"))

    import weight_loader as WL  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M  # noqa: E402
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402

    out_path = sys.argv[2]
    text = sys.argv[3] if len(sys.argv) > 3 else _DEFAULT_TEXT
    model_dir = sys.argv[4] if len(sys.argv) > 4 else _DEFAULT_MODEL_DIR

    loader = WL.DeepseekV4WeightLoader(model_dir)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    ids = tokenizer(text)["input_ids"]
    real_len = len(ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else config.eos_token_id
    seq_len = ((len(ids) + 31) // 32) * 32  # pad to a tile multiple
    ids = ids + [pad_id] * (seq_len - len(ids))
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    position_ids = torch.arange(seq_len).unsqueeze(0)
    dummy = torch.zeros(1, seq_len, 1, dtype=dtype)
    rotary = M.DeepseekV4RotaryEmbedding(config).to(dtype)

    def half(cos_sin):
        cos, sin = cos_sin
        return cos[0].contiguous(), sin[0].contiguous()

    rope = {
        "main": half(rotary(dummy, position_ids=position_ids, layer_type="main")),
        "compress": half(rotary(dummy, position_ids=position_ids, layer_type="compress")),
        "win": {},
    }
    # One windowed-RoPE table per distinct compress-rate (CSA / HCA layers).
    for cr in sorted({int(v) for v in config.compress_rates.values()}):
        n_win = seq_len // cr
        win_pos = (torch.arange(n_win) * cr).unsqueeze(0)
        rope["win"][cr] = half(rotary(dummy, position_ids=win_pos, layer_type="compress"))

    cfg_dict = config.to_dict()
    num_hash_layers = int(cfg_dict.get("num_hash_layers", 3))
    mlp_layer_types = list(getattr(config, "mlp_layer_types", None) or cfg_dict.get("mlp_layer_types") or [])
    if not mlp_layer_types:
        mlp_layer_types = ["hash_moe"] * num_hash_layers + ["moe"] * (config.num_hidden_layers - num_hash_layers)

    torch.save(
        {
            "input_ids": input_ids,
            "last_real_len": real_len,
            "text": text,
            "rope": rope,
            "config": {
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "head_dim": config.head_dim,
                "qk_rope_head_dim": config.qk_rope_head_dim,
                "o_groups": config.o_groups,
                "o_lora_rank": config.o_lora_rank,
                "rms_norm_eps": config.rms_norm_eps,
                "layer_types": list(config.layer_types),
                "mlp_layer_types": mlp_layer_types,
                "compress_rates": {k: int(v) for k, v in config.compress_rates.items()},
                "sliding_window": config.sliding_window,
                "num_hidden_layers": config.num_hidden_layers,
                "num_hash_layers": num_hash_layers,
                "num_local_experts": config.n_routed_experts,
                "num_experts_per_tok": config.num_experts_per_tok,
                "moe_intermediate_size": config.moe_intermediate_size,
                "routed_scaling_factor": config.routed_scaling_factor,
                "swiglu_limit": config.swiglu_limit,
                "hc_mult": config.hc_mult,
                "hc_sinkhorn_iters": config.hc_sinkhorn_iters,
                "hc_eps": config.hc_eps,
            },
        },
        out_path,
    )
    print(f"REFERENCE_OK seq={seq_len} real_len={real_len} -> {out_path}")


def _decode_main() -> None:
    """``decode <json_path> [model_dir]``: load the tokenizer and print the decode."""
    import importlib.metadata as _md

    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)
    # sys.path.insert(0, _CACHED_TRANSFORMERS)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tt"))

    import weight_loader as WL  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    json_path = sys.argv[2]
    model_dir = sys.argv[3] if len(sys.argv) > 3 else _DEFAULT_MODEL_DIR
    with open(json_path) as fh:
        data = json.load(fh)

    loader = WL.DeepseekV4WeightLoader(model_dir)
    tok = AutoTokenizer.from_pretrained(loader.snapshot_dir)

    ids = data["input_ids"]
    predicted = data["predicted"]
    n = data["last_real_len"]
    n_layers = data["n_layers"]

    print("=" * 78)
    print(f"BF4 decode demo  |  {n_layers} of {data['num_hidden_layers']} layers loaded")
    print("=" * 78)
    print(f"PROMPT          : {tok.decode(ids[:n])!r}")
    print(f"NEXT-TOKEN PRED : {tok.decode([predicted[n - 1]])!r}   (token id {predicted[n - 1]})")
    print()
    print("Top-5 next-token candidates (after the prompt):")
    for tid, p in data["topk"]:
        print(f"    {p:6.3f}  id={tid:<7d} {tok.decode([tid])!r}")
    print()

    # Teacher-forced view: prediction at position i should match token i+1.
    correct = sum(int(predicted[i] == ids[i + 1]) for i in range(n - 1))
    print(f"Teacher-forced next-token accuracy over the prompt: {correct}/{max(n - 1, 1)}")
    shifted = tok.decode(predicted[: n - 1]) if n > 1 else ""
    print(f"Argmax-of-each-position decode (model 'echo'): {shifted!r}")
    print("=" * 78)


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if _mode == "ref":
        _reference_main()
    elif _mode == "decode":
        _decode_main()
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# pytest side (ttnn venv).
# --------------------------------------------------------------------------- #
import gc  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.deepseek_v4_flash import (  # noqa: E402
    DeepSeekV4DecoderLayer,
    DeepSeekV4Embedding,
    DeepSeekV4HashRouter,
    DeepSeekV4HyperHead,
    DeepSeekV4PreloadedExperts,
    DeepSeekV4RMSNorm,
    Linear,
    WeightCache,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.weight_loader import (  # noqa: E402
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)


_SYSTEM_PYTHON = "/localdev/smanoj/metal_1/python_env/bin/python"

_THIS_FILE = str(Path(__file__).resolve())
_MASK_NEG = -1.0e9
_WEIGHT_DTYPE = ttnn.bfloat4_b
_CACHE_DIR = os.environ.get("DEEPSEEK_V4_CACHE_DIR", "../cache")


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(Path(_DEFAULT_MODEL_DIR))
    except FileNotFoundError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Mask helpers (mirror modeling_deepseek_v4; built host-side, ttnn venv).
# --------------------------------------------------------------------------- #
def _sliding_causal_mask(seq_len: int, sliding_window: int, dtype: torch.dtype) -> torch.Tensor:
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype).masked_fill(~keep, torch.finfo(dtype).min)
    return mask.view(1, 1, seq_len, seq_len)


def _block_bias(seq_len: int, n_windows: int, compress_rate: int, dtype: torch.dtype) -> torch.Tensor:
    position_ids = torch.arange(seq_len).unsqueeze(0)
    entry = torch.arange(n_windows).view(1, 1, 1, n_windows)
    threshold = ((position_ids + 1) // compress_rate).view(1, 1, seq_len, 1)
    bias = torch.zeros(1, 1, seq_len, n_windows, dtype=dtype)
    return bias.masked_fill(entry >= threshold, torch.finfo(dtype).min)


# --------------------------------------------------------------------------- #
# Weight plumbing (reuse the loader + dequant; bf4 conversion happens on device).
# --------------------------------------------------------------------------- #
def _w(loader: DeepseekV4WeightLoader, name: str):
    """Lazy (dequantized) fetch: returns a thunk so a populated tile cache can
    skip the checkpoint read entirely (the module never calls the thunk on a
    cache hit)."""
    return lambda: dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


def _attn_keys(layer_type: str) -> list[str]:
    keys = [
        "q_a_proj.weight",
        "q_a_norm.weight",
        "q_b_proj.weight",
        "kv_proj.weight",
        "kv_norm.weight",
        "o_a_proj.weight",
        "o_b_proj.weight",
        "sinks",
    ]
    if layer_type != "sliding_attention":
        keys += [
            "compressor.kv_proj.weight",
            "compressor.gate_proj.weight",
            "compressor.kv_norm.weight",
            "compressor.position_bias",
        ]
    return keys


def _build_layer_weights(loader: DeepseekV4WeightLoader, layer_idx: int, layer_type: str, is_hash: bool) -> dict:
    weights: dict[str, torch.Tensor] = {}
    for k in _attn_keys(layer_type):
        weights[f"self_attn.{k}"] = _w(loader, f"layers.{layer_idx}.self_attn.{k}")
    weights["mlp.gate.weight"] = _w(loader, f"layers.{layer_idx}.mlp.gate.weight")
    if not is_hash:
        weights["mlp.gate.e_score_correction_bias"] = _w(loader, f"layers.{layer_idx}.mlp.gate.e_score_correction_bias")
    for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        weights[f"mlp.shared_experts.{k}"] = _w(loader, f"layers.{layer_idx}.mlp.shared_experts.{k}")
    for hc in ("attn_hc", "ffn_hc"):
        for p in ("fn", "base", "scale"):
            weights[f"{hc}.{p}"] = _w(loader, f"layers.{layer_idx}.{hc}.{p}")
    for k in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        weights[k] = _w(loader, f"layers.{layer_idx}.{k}")
    return weights


def _expert_provider(loader: DeepseekV4WeightLoader, layer_idx: int):
    def provider(e: int):
        base = f"layers.{layer_idx}.mlp.experts.{e}"
        gate = _w(loader, f"{base}.gate_proj.weight")()
        up = _w(loader, f"{base}.up_proj.weight")()
        down = _w(loader, f"{base}.down_proj.weight")()
        return torch.cat([gate, up], dim=0).float(), down.float()

    return provider


def _hash_gate(loader: DeepseekV4WeightLoader, layer_idx: int, cfg, device) -> DeepSeekV4HashRouter:
    weights = {
        "gate.weight": _w(loader, f"layers.{layer_idx}.mlp.gate.weight"),
        "gate.tid2eid": loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").long(),
    }
    return DeepSeekV4HashRouter(cfg, weights, device)


def _to_tt(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _layer_cache(layer_idx: int):
    if not _CACHE_DIR:
        return None
    return WeightCache(os.path.join(_CACHE_DIR, "bf4", "ttnn")).sub(f"layers.{layer_idx}")


def _generate_reference(out_path: Path, text: str) -> bool:
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, "ref", str(out_path), text, _DEFAULT_MODEL_DIR],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(f"reference generation failed:\n{proc.stderr[-3000:]}")
        return False
    return out_path.is_file()


def _rope_for_layer(bundle: dict, layer_type: str, compress_rate: int | None, device):
    rope = bundle["rope"]
    cos, sin = rope["main"] if layer_type == "sliding_attention" else rope["compress"]
    cos_full, sin_full = make_rope_table(cos, sin)
    cos_tt = _to_tt(cos_full, device)
    sin_tt = _to_tt(sin_full, device)
    neg_sin_tt = _to_tt(-sin_full, device)

    cos_win_tt = sin_win_tt = None
    if layer_type != "sliding_attention":
        cw, sw = rope["win"][compress_rate]
        cw, sw = make_rope_table(cw, sw)
        cos_win_tt = _to_tt(cw, device)
        sin_win_tt = _to_tt(sw, device)
    return cos_tt, sin_tt, neg_sin_tt, cos_win_tt, sin_win_tt


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)  # heavy: bf4 conversion of every expert on a cold cache
@torch.no_grad()
@pytest.mark.parametrize("text", (_DEFAULT_TEXT,))
def test_bf4_decode_demo(device, reset_seeds, text: str) -> None:
    # --- reference bundle (tokenizer + RoPE tables) ------------------------- #
    if _CACHE_DIR:
        ref_dir = Path(_CACHE_DIR) / "bf4" / "ref"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_path = ref_dir / "decode_demo.pt"
    else:
        ref_path = Path(tempfile.mkdtemp()) / "decode_demo.pt"
    if not ref_path.is_file() and not _generate_reference(ref_path, text):
        pytest.skip("could not generate tokenizer / RoPE reference")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])
    input_ids = bundle["input_ids"]  # [1, S]
    seq_len = input_ids.shape[1]
    last_real_len = bundle["last_real_len"]

    max_layers = int(os.environ.get("DEEPSEEK_V4_DECODE_LAYERS", cfg.num_hidden_layers))
    max_layers = min(max_layers, cfg.num_hidden_layers)

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    top_cache = WeightCache(os.path.join(_CACHE_DIR, "bf4", "ttnn")) if _CACHE_DIR else None

    # --- final head (built first so its DRAM is reserved before the layers) - #
    embed = DeepSeekV4Embedding(loader, device, cache=top_cache)
    hyper_head = DeepSeekV4HyperHead(
        cfg,
        {
            "hc_fn": _w(loader, "hc_head.hc_fn"),
            "hc_base": _w(loader, "hc_head.hc_base"),
            "hc_scale": _w(loader, "hc_head.hc_scale"),
        },
        device,
        cache=top_cache.sub("hc_head") if top_cache else None,
    )
    final_norm = DeepSeekV4RMSNorm(
        _w(loader, "norm.weight"), cfg.rms_norm_eps, device, top_cache.file("norm") if top_cache else None
    )
    lm_head = Linear(
        _w(loader, "lm_head.weight"),
        device,
        top_cache.file("lm_head") if top_cache else None,
        dtype=_WEIGHT_DTYPE,
    )

    # --- embedding -> [B, S, D] -> hc_mult residual-stream stack ------------ #
    ids_tt = ttnn.from_torch(input_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    inputs_embeds = embed(ids_tt)  # [B, S, D]
    b, s, d = inputs_embeds.shape
    streams_tt = ttnn.reshape(inputs_embeds, [b, s, 1, d])
    streams_tt = ttnn.repeat(streams_tt, ttnn.Shape([1, 1, cfg.hc_mult, 1]))  # [B, S, hc_mult, D]

    # --- decoder layers: load as many as fit, run, free per layer ----------- #
    n_loaded = 0
    for li in range(max_layers):
        layer_type = cfg.layer_types[li]
        is_hash = cfg.mlp_layer_types[li] == "hash_moe"
        compress_rate = None if layer_type == "sliding_attention" else cfg.compress_rates[layer_type]

        try:
            cache = _layer_cache(li)
            weights = _build_layer_weights(loader, li, layer_type, is_hash)
            gate = _hash_gate(loader, li, cfg, device) if is_hash else None
            experts = DeepSeekV4PreloadedExperts(
                cfg,
                _expert_provider(loader, li),
                device,
                dtype=_WEIGHT_DTYPE,
                cache=cache.sub("mlp") if cache else None,
            )
            layer = DeepSeekV4DecoderLayer(
                cfg, li, weights, device, experts=experts, gate=gate, cache=cache, weight_dtype=_WEIGHT_DTYPE
            )
        except RuntimeError as exc:
            logger.warning(f"layer {li}: out of device memory ({str(exc)[:160]}); stopping at {n_loaded} layers")
            gc.collect()
            break

        cos_tt, sin_tt, neg_sin_tt, cos_win_tt, sin_win_tt = _rope_for_layer(bundle, layer_type, compress_rate, device)
        n_win = None if compress_rate is None else seq_len // compress_rate
        if layer_type == "sliding_attention":
            mask = _sliding_causal_mask(seq_len, cfg.sliding_window, torch.float32)
        else:
            mask = torch.cat(
                [
                    _sliding_causal_mask(seq_len, cfg.sliding_window, torch.float32),
                    _block_bias(seq_len, n_win, compress_rate, torch.float32),
                ],
                dim=-1,
            )
        mask_tt = _to_tt(mask.clamp_min(_MASK_NEG), device)

        try:
            streams_tt = layer.forward(
                streams_tt,
                cos_tt,
                sin_tt,
                neg_sin_tt,
                mask_tt,
                cos_win=cos_win_tt,
                sin_win=sin_win_tt,
                input_ids=input_ids,
            )
        except RuntimeError as exc:
            logger.warning(f"layer {li}: OOM during forward ({str(exc)[:160]}); stopping at {n_loaded} layers")
            gc.collect()
            break

        n_loaded += 1
        logger.info(f"ran layer {li:2d} ({layer_type}, {'hash' if is_hash else 'topk'} moe)")

        # Free this layer's resident weights before building the next one.
        del layer, experts, gate
        gc.collect()

    assert n_loaded > 0, "no decoder layer fit in device memory"

    # --- head: collapse streams -> norm -> lm_head -> argmax ---------------- #
    collapsed = hyper_head(streams_tt)  # [B, S, D]
    normed = final_norm(collapsed)
    logits = lm_head(normed)  # [B, S, vocab]
    logits_t = ttnn.to_torch(logits).reshape(seq_len, -1).float()

    predicted = logits_t.argmax(dim=-1).tolist()
    top_p, top_i = torch.topk(torch.softmax(logits_t[last_real_len - 1], dim=-1), 5)
    topk = [[int(i), float(p)] for p, i in zip(top_p.tolist(), top_i.tolist())]

    # --- decode (system tokenizer subprocess) ------------------------------- #
    out_json = ref_path.with_suffix(".decode.json")
    with open(out_json, "w") as fh:
        json.dump(
            {
                "input_ids": input_ids[0].tolist(),
                "predicted": predicted,
                "topk": topk,
                "last_real_len": last_real_len,
                "n_layers": n_loaded,
                "num_hidden_layers": cfg.num_hidden_layers,
            },
            fh,
        )
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, "decode", str(out_json), _DEFAULT_MODEL_DIR],
        capture_output=True,
        text=True,
    )
    logger.info(f"\n{proc.stdout}")
    if proc.returncode != 0:
        logger.warning(f"decode subprocess failed:\n{proc.stderr[-2000:]}")

    logger.info(f"ran {n_loaded}/{cfg.num_hidden_layers} decoder layers in bf4 (seq={seq_len})")
