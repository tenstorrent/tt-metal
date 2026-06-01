# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-layer PCC test for the ttnn ``TtDeepseekV4Attention`` (prefill).

This single file plays two roles:

* **pytest side** (imported by the ttnn venv): builds the ttnn attention port
  with the reference's weights/inputs and compares the output with PCC across
  all three attention layer types (sliding / CSA / HCA).
* **reference side** (run as ``__main__`` under the *system* interpreter): the
  gold reference is HuggingFace ``transformers==5.8.1`` ``DeepseekV4Attention``
  (whose ``forward`` is byte-identical to the repo's ``modular_deepseek_v4.py``).
  It dumps deterministic random weights, the RoPE tables, the exact additive
  mask, the input, and the reference output to a ``.pt`` bundle.

The two interpreters are kept apart on purpose: the ttnn venv's ``transformers``
predates ``deepseek_v4``, and the only install on the box that ships it (cached
``transformers==5.8.1``) imports cleanly only under the system interpreter. So
the pytest side re-invokes *this same file* as a subprocess under the system
python to produce the reference. The ``__main__`` guard runs *before* the ttnn
imports, so the subprocess never imports ttnn and the venv never imports the
cached transformers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


LAYER_TYPES = ("sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention")
# Cached transformers 5.8.1 (the only install with ``deepseek_v4`` / ``gpt_oss``).
_CACHED_TRANSFORMERS = "/home/ttuser/.cache/uv/archive-v0/U5SPsIWJupLz-bDcPI13a"


# --------------------------------------------------------------------------- #
# Reference side (executed only as ``__main__`` under the system interpreter).
# --------------------------------------------------------------------------- #
def _reference_build_config(DeepseekV4Config):
    """Reduced-but-tile-friendly config.

    head_dim=256 -> qk_rope_head_dim=32 (a clean multiple of the 32-wide tile),
    H*head_dim=1024 splits into o_groups=2. Small compress_rates keep
    n_windows > 1 at modest seq lengths so the CSA Ca/Cb overlap shift has a
    previous window to read.
    """
    return DeepseekV4Config(
        hidden_size=512,
        q_lora_rank=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        o_groups=2,
        o_lora_rank=128,
        num_hidden_layers=3,
        layer_types=list(LAYER_TYPES),
        mlp_layer_types=["moe", "moe", "moe"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 16},
        index_n_heads=4,
        index_head_dim=128,
        index_topk=512,
        sliding_window=64,
        rms_norm_eps=1.0e-6,
        max_position_embeddings=8192,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        vocab_size=1024,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        attn_implementation="eager",
    )


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

    Matches the reference HCA bias and the CSA indexer's degenerate top-k (which,
    for seq_len <= index_topk * compress_rate, selects every window): query ``t``
    may attend compressed entry ``w`` iff ``w < (t + 1) // compress_rate``.
    """
    batch, seq_len = position_ids.shape
    entry = torch.arange(n_windows).view(1, 1, 1, n_windows)
    threshold = ((position_ids + 1) // compress_rate).view(batch, 1, seq_len, 1)
    bias = torch.zeros(batch, 1, seq_len, n_windows, dtype=dtype)
    return bias.masked_fill(entry >= threshold, torch.finfo(dtype).min)


def _reference_main() -> None:
    """Generate the gold-reference bundle. Args: <out_path> <layer_type> [batch] [seq_len]."""
    import importlib.metadata as _md

    # The cached transformers 5.8.1 wheel pins ``tokenizers>=0.22``; the box has
    # 0.21.4. The version is only enforced by an import-time check, so spoof it.
    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)
    sys.path.insert(0, _CACHED_TRANSFORMERS)

    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    out_path = sys.argv[1]
    layer_type = sys.argv[2]
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    torch.manual_seed(1234)
    config = _reference_build_config(DeepseekV4Config)
    layer_idx = LAYER_TYPES.index(layer_type)
    dtype = torch.float32

    attn = M.DeepseekV4Attention(config, layer_idx).to(dtype).eval()
    # nn.Parameter(torch.empty(...)) members (sinks, position_bias) are
    # uninitialized; reinit *all* params deterministically so the dumped weights
    # are well-defined and reproducible on the ttnn side.
    with torch.no_grad():
        for p in attn.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.02)

    hidden = torch.randn(batch, seq_len, config.hidden_size, dtype=dtype)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)

    rotary = M.DeepseekV4RotaryEmbedding(config).to(dtype)
    rope_layer_type = "main" if layer_type == "sliding_attention" else "compress"
    cos_q, sin_q = rotary(hidden, position_ids=position_ids, layer_type=rope_layer_type)
    position_embeddings = {rope_layer_type: (cos_q, sin_q)}

    # Build the exact additive mask and hand it to the reference so the reference
    # uses *our* mask verbatim (kv width == mask width -> no internal block_bias
    # recompute). The ttnn side loads the identical tensor.
    mask = _reference_sliding_causal_mask(seq_len, config.sliding_window, dtype).expand(batch, 1, seq_len, seq_len)

    bundle: dict = {}
    if layer_type != "sliding_attention":
        cr = config.compress_rates[layer_type]
        n_windows = seq_len // cr
        win_positions = (torch.arange(n_windows) * cr).unsqueeze(0).expand(batch, -1)
        cos_win, sin_win = rotary(hidden, position_ids=win_positions, layer_type="compress")
        mask = torch.cat([mask, _reference_block_bias(position_ids, n_windows, cr, dtype)], dim=-1)
        bundle["cos_win"] = cos_win[0].contiguous()
        bundle["sin_win"] = sin_win[0].contiguous()
        bundle["n_windows"] = n_windows
        bundle["compress_rate"] = cr

    with torch.no_grad():
        output, _ = attn(
            hidden_states=hidden,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=mask,
            past_key_values=None,
        )

    bundle.update(
        {
            "state_dict": {k: v.detach().cpu() for k, v in attn.state_dict().items()},
            "hidden": hidden,
            "position_ids": position_ids,
            "cos_q": cos_q[0].contiguous(),  # [S, Rd/2]
            "sin_q": sin_q[0].contiguous(),
            "mask": mask,  # [B, 1, S, Skv]
            "output": output,  # [B, S, hidden]
            "layer_type": layer_type,
            "config": {
                "hidden_size": config.hidden_size,
                "q_lora_rank": config.q_lora_rank,
                "num_attention_heads": config.num_attention_heads,
                "head_dim": config.head_dim,
                "qk_rope_head_dim": config.qk_rope_head_dim,
                "o_groups": config.o_groups,
                "o_lora_rank": config.o_lora_rank,
                "rms_norm_eps": config.rms_norm_eps,
                "sliding_window": config.sliding_window,
                "compress_rates": config.compress_rates,
                "layer_types": list(config.layer_types),
            },
        }
    )
    torch.save(bundle, out_path)
    print(f"REFERENCE_OK {layer_type} -> {out_path}")


# Reference mode must short-circuit *before* the ttnn imports below: when this
# file is run as a script under the system interpreter it acts purely as the
# reference generator and must never import ttnn / pytest.
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
    DeepSeekV4Attention,
    make_rope_table,
)


# The reference needs the cached transformers 5.8.1, which imports cleanly only
# under the system interpreter (the ttnn venv's huggingface_hub/transformers are
# too old). Fall back to whatever is on PATH if the canonical path is missing.
_SYSTEM_PYTHON = (
    "/usr/bin/python3" if Path("/usr/bin/python3").exists() else (shutil.which("python3") or sys.executable)
)
_THIS_FILE = str(Path(__file__).resolve())
_MASK_NEG = -1.0e9
PCC_THRESHOLD = 0.99


def _generate_reference(out_path: Path, layer_type: str, batch: int, seq_len: int) -> bool:
    """Run *this file* as the reference generator subprocess. False if it can't run."""
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, str(out_path), layer_type, str(batch), str(seq_len)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(f"reference generation failed for {layer_type}:\n{proc.stderr[-2000:]}")
        return False
    return out_path.is_file()


def _to_tt(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("layer_type", LAYER_TYPES)
@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1, 2))
def test_attention_pcc(device, reset_seeds, tmp_path, layer_type: str, batch_size: int, seq_len: int) -> None:
    ref_path = tmp_path / f"ref_{layer_type}.pt"
    if not _generate_reference(ref_path, layer_type, batch_size, seq_len):
        pytest.skip(f"could not generate HF reference for {layer_type} (cached transformers 5.8.1 unavailable)")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])
    layer_idx = cfg.layer_types.index(layer_type)

    attn = DeepSeekV4Attention(cfg, layer_idx, bundle["state_dict"], device)

    hidden_tt = _to_tt(bundle["hidden"], device)
    cos_full, sin_full = make_rope_table(bundle["cos_q"], bundle["sin_q"])
    cos_tt = _to_tt(cos_full, device)
    sin_tt = _to_tt(sin_full, device)
    neg_sin_tt = _to_tt(-sin_full, device)

    mask_tt = _to_tt(bundle["mask"].clamp_min(_MASK_NEG), device)

    cos_win_tt = sin_win_tt = None
    if layer_type != "sliding_attention":
        cw, sw = make_rope_table(bundle["cos_win"], bundle["sin_win"])
        cos_win_tt = _to_tt(cw, device)
        sin_win_tt = _to_tt(sw, device)

    out_tt = attn.forward(
        hidden_tt,
        cos_tt,
        sin_tt,
        neg_sin_tt,
        mask_tt,
        cos_win=cos_win_tt,
        sin_win=sin_win_tt,
    )
    out_torch = ttnn.to_torch(out_tt).reshape(bundle["output"].shape).to(torch.float32)

    reference = bundle["output"].to(torch.float32)
    passing, pcc_message = comp_pcc(reference, out_torch, pcc=PCC_THRESHOLD)
    logger.info(comp_allclose(reference, out_torch))
    logger.info(f"[{layer_type}] PCC: {pcc_message}")

    assert passing, f"{layer_type} attention PCC < {PCC_THRESHOLD}: {pcc_message}"
