# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode-path PCC test for the ttnn ``DeepSeekV4Attention`` on real weights.

A dual-interpreter test running on the **real** V4-Flash checkpoint weights for a
single attention block:

* **reference side** (run as ``__main__`` under the *system* interpreter):
  instantiates the HuggingFace ``DeepseekV4Attention`` for the requested layer,
  loads the real (dequantized) checkpoint weights for that layer into it, runs the
  reference forward over a random ``[B, S, D]`` hidden-state sequence (full
  prefill), and dumps the RoPE tables, the additive mask, the input, and the
  reference output to a ``.pt`` bundle.
* **pytest side** (ttnn venv): builds the ttnn :class:`DeepSeekV4Attention` from
  the *same* loader (identical dequantized weights) and exercises the **decode**
  path (:meth:`DeepSeekV4Attention.decode`, ``T == 1``): it replays every token
  one step at a time through ``decode`` (filling the running KV / compressor
  cache) and PCC-compares the last few decoded rows against the reference's
  full-prefill row at the same position (decode is the per-token-equivalent of a
  full prefill).

The two interpreters are kept apart because the cached ``transformers`` shipping
``deepseek_v4`` imports cleanly only under the system interpreter, while ttnn
lives in the venv. The ``__main__`` guard runs before the ttnn imports so the
subprocess never touches ttnn.

Set ``DEEPSEEK_V4_CACHE_DIR=<dir>`` to skip the slow weight loading on reruns:
the converted ttnn weight tiles are dumped/reused and the HF reference bundle is
cached too, so the expensive system-python subprocess only runs once per
(layer, batch, seq_len).

Run (ttnn venv)::

    pytest -s models/experimental/deepseek_v4_flash/tests/test_attention_real_weights.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


# Cached transformers (the only install with ``deepseek_v4``).
_DEFAULT_MODEL_DIR = "/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash"


# --------------------------------------------------------------------------- #
# Reference side (executed only as ``__main__`` under the system interpreter).
# --------------------------------------------------------------------------- #
def _reference_sliding_causal_mask(seq_len: int, sliding_window: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive [1, 1, S, S] sliding-window causal mask (0 keep / min mask)."""
    i = torch.arange(seq_len).view(seq_len, 1)
    j = torch.arange(seq_len).view(1, seq_len)
    keep = (j <= i) & (i - j < sliding_window)
    mask = torch.zeros(seq_len, seq_len, dtype=dtype).masked_fill(~keep, torch.finfo(dtype).min)
    return mask.view(1, 1, seq_len, seq_len)


def _reference_block_bias(
    position_ids: torch.Tensor, n_windows: int, compress_rate: int, dtype: torch.dtype
) -> torch.Tensor:
    """Additive [B, 1, S, n_windows] causal block bias over compressed windows.

    Query ``t`` may attend compressed entry ``w`` iff ``w < (t + 1) // compress_rate``
    — the degenerate CSA/HCA top-k for ``seq_len <= index_topk * compress_rate``.
    """
    batch, seq_len = position_ids.shape
    entry = torch.arange(n_windows).view(1, 1, 1, n_windows)
    threshold = ((position_ids + 1) // compress_rate).view(batch, 1, seq_len, 1)
    bias = torch.zeros(batch, 1, seq_len, n_windows, dtype=dtype)
    return bias.masked_fill(entry >= threshold, torch.finfo(dtype).min)


def _ref_load_attn_weights(attn, loader, quant, layer_idx: int) -> None:
    """Populate an HF ``DeepseekV4Attention`` state-dict from the checkpoint."""

    def w(name: str) -> torch.Tensor:
        full = f"layers.{layer_idx}.self_attn.{name}"
        return quant.dequantize_weight(loader.get_tensor(full), loader.get_scale(full)).to(torch.float32)

    sd = {key: w(key) for key in attn.state_dict()}
    attn.load_state_dict(sd, strict=True, assign=True)


def _reference_main() -> None:
    """Generate the gold-reference bundle. Args: <out_path> <layer_idx> [batch] [seq_len] [model_dir]."""
    import importlib.metadata as _md

    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)

    # weight_loader / quant are standalone (torch + safetensors only); import by path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tt"))

    import weight_loader as WL  # noqa: E402
    import quant as Q  # noqa: E402
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M  # noqa: E402
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config  # noqa: E402

    out_path = sys.argv[1]
    layer_idx = int(sys.argv[2])
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    model_dir = sys.argv[5] if len(sys.argv) > 5 else _DEFAULT_MODEL_DIR

    loader = WL.DeepseekV4WeightLoader(model_dir)
    config = DeepseekV4Config.from_pretrained(loader.snapshot_dir)
    config._attn_implementation = "eager"
    dtype = torch.float32

    attn = M.DeepseekV4Attention(config, layer_idx).to(dtype).eval()
    _ref_load_attn_weights(attn, loader, Q, layer_idx)

    layer_type = config.layer_types[layer_idx]
    rope_layer_type = "main" if layer_type == "sliding_attention" else "compress"

    torch.manual_seed(1234)
    hidden = torch.randn(batch, seq_len, config.hidden_size, dtype=dtype)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    rotary = M.DeepseekV4RotaryEmbedding(config).to(dtype)
    cos_q, sin_q = rotary(hidden, position_ids=position_ids, layer_type=rope_layer_type)
    position_embeddings = {
        "main": rotary(hidden, position_ids=position_ids, layer_type="main"),
        "compress": rotary(hidden, position_ids=position_ids, layer_type="compress"),
    }

    mask = _reference_sliding_causal_mask(seq_len, config.sliding_window, dtype).expand(batch, 1, seq_len, seq_len)

    bundle: dict = {}
    if layer_type != "sliding_attention":
        cr = config.compress_rates[layer_type]
        n_windows = seq_len // cr
        assert n_windows >= 1, f"seq_len {seq_len} too short for compress_rate {cr}"
        win_positions = (torch.arange(n_windows) * cr).unsqueeze(0).expand(batch, -1)
        cos_win, sin_win = rotary(hidden, position_ids=win_positions, layer_type="compress")
        mask = torch.cat([mask, _reference_block_bias(position_ids, n_windows, cr, dtype)], dim=-1)
        bundle["cos_win"] = cos_win[0].contiguous()
        bundle["sin_win"] = sin_win[0].contiguous()
        bundle["n_windows"] = n_windows
        bundle["compress_rate"] = cr

    with torch.no_grad():
        output, _ = attn(
            hidden,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=mask,
            past_key_values=None,
        )

    bundle.update(
        {
            "hidden": hidden,
            "cos_q": cos_q[0].contiguous(),
            "sin_q": sin_q[0].contiguous(),
            "mask": mask,
            "output": output,
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "config": {
                "hidden_size": config.hidden_size,
                "num_attention_heads": config.num_attention_heads,
                "head_dim": config.head_dim,
                "qk_rope_head_dim": config.qk_rope_head_dim,
                "o_groups": config.o_groups,
                "o_lora_rank": config.o_lora_rank,
                "rms_norm_eps": config.rms_norm_eps,
                "sliding_window": config.sliding_window,
                "layer_types": list(config.layer_types),
                "compress_rates": dict(config.compress_rates),
            },
        }
    )
    torch.save(bundle, out_path)
    print(f"REFERENCE_OK layer={layer_idx} ({layer_type}) -> {out_path}")


if __name__ == "__main__":
    _reference_main()
    raise SystemExit(0)


# --------------------------------------------------------------------------- #
# pytest side (ttnn venv).
# --------------------------------------------------------------------------- #
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402
from models.common.utility_functions import comp_allclose, comp_pcc  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.attention import (  # noqa: E402
    DeepSeekV4Attention,
    _LayerKVCache,
    make_rope_table,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight  # noqa: E402
from models.experimental.deepseek_v4_flash.tt.weight_loader import (  # noqa: E402
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)


_SYSTEM_PYTHON = shutil.which("python") or sys.executable
_THIS_FILE = str(Path(__file__).resolve())
_WEIGHT_DTYPE = ttnn.bfloat16
# Decode reuses the *prefill* HF reference: a single-token decode step is the
# bit-for-bit-equivalent of a full prefill over the same tokens-so-far, so the
# decoded row at position p must match the reference's full-prefill row p. Decode
# reads the incrementally-built KV / compressor cache (vs full-prefill attention),
# which widens the gap a touch, hence the slightly looser threshold.
DECODE_PCC_THRESHOLD = 0.97
# How many tokens to decode (one device step each) past the seeded prefix.
_DECODE_STEPS = 4
# On-disk cache (ttnn weight tiles + HF reference bundles). Defaults to a dir
# under the system temp dir; override with ``DEEPSEEK_V4_CACHE_DIR`` (set it to an
# empty string to disable caching and always reload from the checkpoint).
_CACHE_DIR = (
    os.environ.get("DEEPSEEK_V4_CACHE_DIR", os.path.join(tempfile.gettempdir(), "deepseek_v4_flash_cache")) or None
)


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
    """Per-layer attention :class:`WeightCache` for the ttnn weights, or ``None``."""
    if not _CACHE_DIR:
        return None
    return WeightCache(os.path.join(_CACHE_DIR, "ttnn")).sub(f"layers.{layer_idx}").sub("self_attn")


def _generate_reference(out_path: Path, layer_idx: int, batch: int, seq_len: int) -> bool:
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, str(out_path), str(layer_idx), str(batch), str(seq_len), _DEFAULT_MODEL_DIR],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(f"reference generation failed for layer {layer_idx}:\n{proc.stderr[-3000:]}")
        return False
    return out_path.is_file()


def _w(loader: DeepseekV4WeightLoader, name: str):
    """Lazy fetch + dequantize: returns a thunk producing the fp32 tensor."""
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


def _build_attn_weights(loader: DeepseekV4WeightLoader, layer_idx: int, layer_type: str) -> dict:
    """``DeepSeekV4Attention``-relative-keyed (dequantized) weight dict."""
    return {k: _w(loader, f"layers.{layer_idx}.self_attn.{k}") for k in _attn_keys(layer_type)}


def _to_tt(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _rope_rows(cos_half: torch.Tensor, sin_half: torch.Tensor, device) -> tuple:
    """``(cos, sin, neg_sin)`` ttnn tables for a half-table slice (see ``make_rope_table``)."""
    cos_full, sin_full = make_rope_table(cos_half, sin_half)
    return _to_tt(cos_full, device), _to_tt(sin_full, device), _to_tt(-sin_full, device)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4-Flash checkpoint not found under {_DEFAULT_MODEL_DIR}")
@torch.no_grad()
@pytest.mark.timeout(14400)
@pytest.mark.parametrize("layer_idx", (1, 5))  # 1 = sliding-only, 5 = HCA compressor
@pytest.mark.parametrize("seq_len", (256,))
@pytest.mark.parametrize("batch_size", (1,))
def test_attention_real_weights_decode(
    device, reset_seeds, tmp_path, layer_idx: int, batch_size: int, seq_len: int
) -> None:
    """Decode-path PCC for ``DeepSeekV4Attention.decode`` against an HF full-prefill.

    Replays every token one device step at a time through :meth:`DeepSeekV4Attention.decode`
    (which fills the running sliding-K=V + compressor cache exactly as a prefill
    would), and PCC-compares the last ``_DECODE_STEPS`` decoded rows against the
    reference's full-prefill rows at the same absolute positions. The first
    ``seq_len - 32`` steps only seed the cache.
    """
    ref_path, need_gen = _reference_path(tmp_path, f"attn_pcc_{layer_idx}_{batch_size}_{seq_len}")
    if not need_gen and "sliding_window" not in torch.load(ref_path, weights_only=False)["config"]:
        need_gen = True
    if need_gen and not _generate_reference(ref_path, layer_idx, batch_size, seq_len):
        pytest.skip(f"could not generate HF reference for layer {layer_idx}")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])
    layer_type = bundle["layer_type"]
    is_compressor = layer_type != "sliding_attention"

    loader = DeepseekV4WeightLoader(_DEFAULT_MODEL_DIR)
    cache = _weight_cache(layer_idx)
    weights = _build_attn_weights(loader, layer_idx, layer_type)
    attn = DeepSeekV4Attention(cfg, layer_idx, weights, device, cache=cache, weight_dtype=_WEIGHT_DTYPE)

    hidden = bundle["hidden"]  # [B, S, D]
    reference = bundle["output"].to(torch.float32)  # full-prefill output [B, S, D]

    split = seq_len - 32  # one tile of room so the decode steps have reference rows
    assert split % 32 == 0 and split + _DECODE_STEPS <= seq_len
    cr = cfg.compress_rates[layer_type] if is_compressor else None

    kv_cache = _LayerKVCache(cfg.sliding_window, is_compressor)
    for pos in range(split + _DECODE_STEPS):
        cos_d, sin_d, neg_sin_d = _rope_rows(bundle["cos_q"][pos : pos + 1], bundle["sin_q"][pos : pos + 1], device)
        cos_win_d = sin_win_d = None
        if is_compressor and (pos + 1) // cr > 0:
            n_win = (pos + 1) // cr
            cw, sw = make_rope_table(bundle["cos_win"][:n_win], bundle["sin_win"][:n_win])
            cos_win_d = _to_tt(cw, device)
            sin_win_d = _to_tt(sw, device)

        out_tt = attn.decode(
            _to_tt(hidden[:, pos : pos + 1].reshape(batch_size, 1, 1, cfg.hidden_size), device),
            cos_d,
            sin_d,
            neg_sin_d,
            cos_win_d,
            sin_win_d,
            kv_cache,
        )
        if pos < split:
            continue  # seeding the cache; no reference row to compare yet

        ref_row = reference[:, pos : pos + 1]
        out_torch = ttnn.to_torch(out_tt).reshape(ref_row.shape).to(torch.float32)

        passing, pcc_message = comp_pcc(ref_row, out_torch, pcc=DECODE_PCC_THRESHOLD)
        logger.info(comp_allclose(ref_row, out_torch))
        logger.info(f"[attention layer {layer_idx} ({layer_type}) pos {pos}] PCC: {pcc_message}")
        assert passing, f"layer {layer_idx} attention decode pos {pos} PCC < {DECODE_PCC_THRESHOLD}: {pcc_message}"
