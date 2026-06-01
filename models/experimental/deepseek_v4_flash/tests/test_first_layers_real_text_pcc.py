# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end PCC test: tokenizer + the first 3 ``DeepSeekV4DecoderLayer``s.

Takes a real text string, tokenizes it with the V4-Flash tokenizer, embeds it,
and runs the first three decoder layers on the **real** checkpoint weights —
exercising the full prefill front-end of the model: embedding, the
``hc_mult``-stream residual expansion, hash-routed MoE (layers 0-2 are
``hash_moe``), sliding attention (layers 0-1) and compressed-sparse attention
(layer 2), and both HyperConnections per layer.

Two-process, like the other V4 PCC tests:

* **reference side** (``__main__``, a fresh subprocess): builds the three HF
  ``DeepseekV4DecoderLayer``s (``transformers>=5.10``), loads the real
  (dequantized) checkpoint weights into them, tokenizes the text, runs the
  embedding + 3 layers, and dumps the per-layer RoPE tables / masks, the
  ``input_ids``, and the final residual-stream stack. Run as its own process so
  the heavy HF build never touches the ttnn runtime held by the pytest process.
* **pytest side** (ttnn venv): runs the ttnn embedding + 3 ttnn decoder layers
  from the *same* loader and PCC-compares the final stream stack.

The routed experts live on device in ``bfloat4_b`` (the ``fused_experts`` op's
design dtype -- its per-core circular buffers are sized for the bf4 weight slice
and overflow L1 at wider dtypes), so all three layers' 256 experts fit the
Blackhole DRAM at once.

Set ``DEEPSEEK_V4_CACHE_DIR=<dir>`` to skip the slow weight loading on reruns:
the converted ttnn weight tiles are dumped/reused (the per-layer 256-expert
dequant is skipped on a hit) and the HF reference bundle is cached too, so the
expensive system-python subprocess only runs once per text.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


_DEFAULT_MODEL_DIR = "/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash"
_N_LAYERS = 3
_DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog near the riverbank at dawn."


# --------------------------------------------------------------------------- #
# Shared mask helpers (used by the reference side).
# --------------------------------------------------------------------------- #
def _reference_sliding_causal_mask(seq_len: int, sliding_window: int, dtype: torch.dtype) -> torch.Tensor:
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype).masked_fill(~keep, torch.finfo(dtype).min)
    return mask.view(1, 1, seq_len, seq_len)


def _reference_block_bias(
    position_ids: torch.Tensor, n_windows: int, compress_rate: int, dtype: torch.dtype
) -> torch.Tensor:
    batch, seq_len = position_ids.shape
    entry = torch.arange(n_windows).view(1, 1, 1, n_windows)
    threshold = ((position_ids + 1) // compress_rate).view(batch, 1, seq_len, 1)
    bias = torch.zeros(batch, 1, seq_len, n_windows, dtype=dtype)
    return bias.masked_fill(entry >= threshold, torch.finfo(dtype).min)


# --------------------------------------------------------------------------- #
# Reference side (executed only as ``__main__`` in the reference subprocess).
# --------------------------------------------------------------------------- #
def _ref_load_layer_weights(layer, loader, quant, layer_idx: int) -> None:
    def w(name: str) -> torch.Tensor:
        # transformers >= 5.10 nests the lightning-indexer scoring head under an
        # extra ``indexer.scorer`` module (``DeepseekV4IndexerScorer``); the
        # checkpoint still stores ``weights_proj`` flat on the indexer, so drop the
        # ``scorer`` segment to match the loader's HF -> checkpoint name map.
        name = name.replace(".indexer.scorer.", ".indexer.")
        full = f"layers.{layer_idx}.{name}"
        return quant.dequantize_weight(loader.get_tensor(full), loader.get_scale(full)).to(torch.float32)

    state = layer.state_dict()
    n_experts = state["mlp.experts.gate_up_proj"].shape[0]
    sd: dict[str, torch.Tensor] = {}
    for key in state:
        if key == "mlp.experts.gate_up_proj":
            rows = [
                torch.cat([w(f"mlp.experts.{e}.gate_proj.weight"), w(f"mlp.experts.{e}.up_proj.weight")], dim=0)
                for e in range(n_experts)
            ]
            sd[key] = torch.stack(rows, dim=0)
        elif key == "mlp.experts.down_proj":
            sd[key] = torch.stack([w(f"mlp.experts.{e}.down_proj.weight") for e in range(n_experts)], dim=0)
        elif key == "mlp.gate.tid2eid":
            sd[key] = loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").long()
        else:
            sd[key] = w(key)
    layer.load_state_dict(sd, strict=True, assign=True)


def _reference_main() -> None:
    """Generate the gold-reference bundle. Args: <out_path> [text] [model_dir]."""
    import importlib.metadata as _md

    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tt"))

    import weight_loader as WL  # noqa: E402
    import quant as Q  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M  # noqa: E402
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402

    out_path = sys.argv[1]
    text = sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_TEXT
    model_dir = sys.argv[3] if len(sys.argv) > 3 else _DEFAULT_MODEL_DIR

    loader = WL.DeepseekV4WeightLoader(model_dir)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    ids = tokenizer(text)["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else config.eos_token_id
    seq_len = ((len(ids) + 31) // 32) * 32  # pad to a tile multiple
    ids = ids + [pad_id] * (seq_len - len(ids))
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    embed_w = loader.get_tensor("embed_tokens.weight").to(dtype)
    inputs_embeds = torch.nn.functional.embedding(input_ids, embed_w)
    hc_mult = config.hc_mult
    streams = inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()
    position_ids = torch.arange(seq_len).unsqueeze(0)

    rotary = M.DeepseekV4RotaryEmbedding(config).to(dtype)
    pe = {
        "main": rotary(streams, position_ids=position_ids, layer_type="main"),
        "compress": rotary(streams, position_ids=position_ids, layer_type="compress"),
    }
    sliding_mask = _reference_sliding_causal_mask(seq_len, config.sliding_window, dtype)

    per_layer: list[dict] = []
    with torch.no_grad():
        for li in range(_N_LAYERS):
            layer = M.DeepseekV4DecoderLayer(config, li).to(dtype).eval()
            _ref_load_layer_weights(layer, loader, Q, li)

            layer_type = config.layer_types[li]
            rope_type = "main" if layer_type == "sliding_attention" else "compress"
            cos_q, sin_q = pe[rope_type]

            entry: dict = {"layer_type": layer_type, "cos_q": cos_q[0].contiguous(), "sin_q": sin_q[0].contiguous()}
            mask = sliding_mask
            if layer_type != "sliding_attention":
                cr = config.compress_rates[layer_type]
                n_win = seq_len // cr
                win_pos = (torch.arange(n_win) * cr).unsqueeze(0)
                cos_win, sin_win = rotary(streams, position_ids=win_pos, layer_type="compress")
                mask = torch.cat([sliding_mask, _reference_block_bias(position_ids, n_win, cr, dtype)], dim=-1)
                entry["cos_win"] = cos_win[0].contiguous()
                entry["sin_win"] = sin_win[0].contiguous()
            entry["mask"] = mask

            streams = layer(
                streams,
                input_ids=input_ids,
                position_embeddings=pe,
                position_ids=position_ids,
                attention_mask=mask,
                past_key_values=None,
            )
            entry["stream_out"] = streams.contiguous()  # per-layer reference for localized PCC
            per_layer.append(entry)
            del layer

    torch.save(
        {
            "input_ids": input_ids,
            "per_layer": per_layer,
            "output": streams,
            "text": text,
            "config": {
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "head_dim": config.head_dim,
                "qk_rope_head_dim": config.qk_rope_head_dim,
                "o_groups": config.o_groups,
                "o_lora_rank": config.o_lora_rank,
                "rms_norm_eps": config.rms_norm_eps,
                "layer_types": list(config.layer_types),
                "compress_rates": dict(config.compress_rates),
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
    print(f"REFERENCE_OK {_N_LAYERS} layers, seq={seq_len} -> {out_path}")


if __name__ == "__main__":
    _reference_main()
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# pytest side (ttnn venv).
# --------------------------------------------------------------------------- #
import shutil  # noqa: E402
import subprocess  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.common.utility_functions import comp_allclose, comp_pcc  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.deepseek_v4_flash import (  # noqa: E402
    DeepSeekV4DecoderLayer,
    DeepSeekV4Embedding,
    DeepSeekV4HashRouter,
    DeepSeekV4PreloadedExperts,
    WeightCache,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.weight_loader import (  # noqa: E402
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)


_SYSTEM_PYTHON = shutil.which("python") or sys.executable
_THIS_FILE = str(Path(__file__).resolve())
_MASK_NEG = -1.0e9
PCC_THRESHOLD = 0.97
# Opt-in on-disk cache (ttnn weight tiles + HF reference bundle). ``None`` = off.
_CACHE_DIR = os.environ.get("DEEPSEEK_V4_CACHE_DIR")


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(Path(_DEFAULT_MODEL_DIR))
    except FileNotFoundError:
        return False
    return True


def _reference_path(tmp_path: Path, name: str) -> tuple[Path, bool]:
    """``(path, needs_generation)`` for the reference bundle (cached if enabled)."""
    if _CACHE_DIR:
        d = Path(_CACHE_DIR) / "ref"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{name}.pt"
        return p, not p.is_file()
    return tmp_path / f"{name}.pt", True


def _weight_cache(layer_idx: int):
    """Per-layer :class:`WeightCache` for the ttnn weights, or ``None`` (off)."""
    if not _CACHE_DIR:
        return None
    return WeightCache(os.path.join(_CACHE_DIR, "ttnn")).sub(f"layers.{layer_idx}")


def _generate_reference(out_path: Path, text: str) -> bool:
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, str(out_path), text, _DEFAULT_MODEL_DIR],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(f"reference generation failed:\n{proc.stderr[-3000:]}")
        return False
    return out_path.is_file()


def _w(loader: DeepseekV4WeightLoader, name: str) -> torch.Tensor:
    return dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


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


def _build_layer_weights(loader: DeepseekV4WeightLoader, layer_idx: int, layer_type: str) -> dict:
    weights: dict[str, torch.Tensor] = {}
    for k in _attn_keys(layer_type):
        weights[f"self_attn.{k}"] = _w(loader, f"layers.{layer_idx}.self_attn.{k}")
    weights["mlp.gate.weight"] = _w(loader, f"layers.{layer_idx}.mlp.gate.weight")
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
        gate = _w(loader, f"{base}.gate_proj.weight")
        up = _w(loader, f"{base}.up_proj.weight")
        down = _w(loader, f"{base}.down_proj.weight")
        return torch.cat([gate, up], dim=0).to(torch.bfloat16), down.to(torch.bfloat16)

    return provider


def _hash_gate(loader: DeepseekV4WeightLoader, layer_idx: int, cfg, device) -> DeepSeekV4HashRouter:
    weights = {
        "gate.weight": _w(loader, f"layers.{layer_idx}.mlp.gate.weight"),
        "gate.tid2eid": loader.get_tensor(f"layers.{layer_idx}.mlp.gate.tid2eid").long(),
    }
    return DeepSeekV4HashRouter(cfg, weights, device)


def _to_tt(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(1800)  # heavy: HF 3-layer reference + 3x256 expert load on a cold cache
@torch.no_grad()
@pytest.mark.parametrize("text", (_DEFAULT_TEXT,))
def test_first_layers_real_text_pcc(device, reset_seeds, tmp_path, text: str) -> None:
    ref_path, need_gen = _reference_path(tmp_path, "first_layers")
    if need_gen and not _generate_reference(ref_path, text):
        pytest.skip("could not generate HF reference (cached transformers / checkpoint unavailable)")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])
    input_ids = bundle["input_ids"]  # [1, S]
    seq_len = input_ids.shape[1]

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    top_cache = WeightCache(os.path.join(_CACHE_DIR, "ttnn")) if _CACHE_DIR else None

    # Embedding -> [B, S, D] -> broadcast to the hc_mult residual-stream stack.
    embed = DeepSeekV4Embedding(loader, device, cache=top_cache)
    ids_tt = ttnn.from_torch(input_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    inputs_embeds = embed(ids_tt)  # [B, S, D]
    b, s, d = inputs_embeds.shape
    streams_tt = ttnn.reshape(inputs_embeds, [b, s, 1, d])
    streams_tt = ttnn.repeat(streams_tt, ttnn.Shape([1, 1, cfg.hc_mult, 1]))  # [B, S, hc_mult, D]

    for li in range(_N_LAYERS):
        entry = bundle["per_layer"][li]
        layer_type = entry["layer_type"]

        layer_cache = _weight_cache(li)
        weights = _build_layer_weights(loader, li, layer_type)
        gate = _hash_gate(loader, li, cfg, device)
        experts = DeepSeekV4PreloadedExperts(
            cfg,
            _expert_provider(loader, li),
            device,
            dtype=ttnn.bfloat4_b,
            cache=layer_cache.sub("mlp") if layer_cache else None,
        )
        layer = DeepSeekV4DecoderLayer(cfg, li, weights, device, experts=experts, gate=gate, cache=layer_cache)

        cos_full, sin_full = make_rope_table(entry["cos_q"], entry["sin_q"])
        cos_tt = _to_tt(cos_full, device)
        sin_tt = _to_tt(sin_full, device)
        neg_sin_tt = _to_tt(-sin_full, device)
        mask_tt = _to_tt(entry["mask"].clamp_min(_MASK_NEG), device)

        cos_win_tt = sin_win_tt = None
        if layer_type != "sliding_attention":
            cw, sw = make_rope_table(entry["cos_win"], entry["sin_win"])
            cos_win_tt = _to_tt(cw, device)
            sin_win_tt = _to_tt(sw, device)

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

        if "stream_out" in entry:
            cur = ttnn.to_torch(streams_tt).reshape(entry["stream_out"].shape).to(torch.float32)
            ref_cur = entry["stream_out"].to(torch.float32)
            _, msg = comp_pcc(ref_cur, cur, pcc=PCC_THRESHOLD)
            logger.info(
                f"[layer {li} ({layer_type})] PCC: {msg} | tt nan={torch.isnan(cur).any().item()} "
                f"inf={torch.isinf(cur).any().item()} min={cur.min().item():.3f} max={cur.max().item():.3f}"
            )

    out_torch = ttnn.to_torch(streams_tt).reshape(bundle["output"].shape).to(torch.float32)
    reference = bundle["output"].to(torch.float32)

    passing, pcc_message = comp_pcc(reference, out_torch, pcc=PCC_THRESHOLD)
    logger.info(comp_allclose(reference, out_torch))
    logger.info(f"[first {_N_LAYERS} layers, seq={seq_len}] PCC: {pcc_message}")

    assert passing, f"first-{_N_LAYERS}-layers PCC < {PCC_THRESHOLD}: {pcc_message}"
