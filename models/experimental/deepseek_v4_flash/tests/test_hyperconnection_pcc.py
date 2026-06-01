# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-module PCC test for the ttnn ``DeepSeekV4HyperConnection`` (mHC).

Same dual-interpreter split as ``test_moe_pcc.py`` / ``test_attention_pcc.py``:

* **pytest side** (ttnn venv): builds the ttnn ``DeepSeekV4HyperConnection`` with
  the reference's weights/input and compares the ``(post, comb, collapsed)``
  triple with PCC.
* **reference side** (``__main__`` under the *system* interpreter): the gold
  reference is HuggingFace ``transformers==5.8.1``
  ``DeepseekV4HyperConnection`` (byte-identical to ``modular_deepseek_v4.py``).
  It dumps deterministic random weights, the input stream stack, and the three
  outputs to a ``.pt`` bundle.

The interpreters are kept apart for the same reason as the other tests: the ttnn
venv's ``transformers`` predates ``deepseek_v4``; only the cached
``transformers==5.8.1`` ships it, and it imports cleanly only under the system
interpreter. The ``__main__`` guard runs before the ttnn imports so the
subprocess never imports ttnn.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


_CACHED_TRANSFORMERS = "/home/ttuser/.cache/uv/archive-v0/U5SPsIWJupLz-bDcPI13a"


# --------------------------------------------------------------------------- #
# Reference side (executed only as ``__main__`` under the system interpreter).
# --------------------------------------------------------------------------- #
def _reference_build_config(DeepseekV4Config):
    """Reduced-but-tile-friendly config (mirrors the other module tests).

    Only ``hidden_size`` + the ``hc_*`` knobs matter for the HyperConnection;
    the rest are required for a valid ``DeepseekV4Config``. ``hc_mult`` /
    ``hc_sinkhorn_iters`` / ``hc_eps`` are left at their checkpoint defaults
    (4 / 20 / 1e-6).
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
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
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
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        attn_implementation="eager",
    )


def _reference_main() -> None:
    """Generate the gold-reference bundle. Args: <out_path> [batch] [seq_len]."""
    import importlib.metadata as _md

    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)
    sys.path.insert(0, _CACHED_TRANSFORMERS)

    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    out_path = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    torch.manual_seed(1234)
    config = _reference_build_config(DeepseekV4Config)
    dtype = torch.float32

    hc = M.DeepseekV4HyperConnection(config).to(dtype).eval()
    # Match the model's ``_init_weights`` regime for this module: normal ``fn``,
    # zero ``base``, unit ``scale`` (so the scale/sigmoid path is exercised).
    with torch.no_grad():
        torch.nn.init.normal_(hc.fn, mean=0.0, std=0.02)
        torch.nn.init.zeros_(hc.base)
        torch.nn.init.ones_(hc.scale)

    hidden_streams = torch.randn(batch, seq_len, config.hc_mult, config.hidden_size, dtype=dtype)

    with torch.no_grad():
        post, comb, collapsed = hc(hidden_streams)

    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in hc.state_dict().items()},
        "hidden_streams": hidden_streams,
        "post": post,  # [B, S, H]
        "comb": comb,  # [B, S, H, H]
        "collapsed": collapsed,  # [B, S, D]
        "config": {
            "hidden_size": config.hidden_size,
            "hc_mult": config.hc_mult,
            "hc_sinkhorn_iters": config.hc_sinkhorn_iters,
            "hc_eps": config.hc_eps,
            "rms_norm_eps": config.rms_norm_eps,
        },
    }
    torch.save(bundle, out_path)
    print(f"REFERENCE_OK hyperconnection -> {out_path}")


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
from models.experimental.deepseek_v4_flash.tt.hyperconnection import (  # noqa: E402
    DeepSeekV4HyperConnection,
)


_SYSTEM_PYTHON = (
    "/usr/bin/python3" if Path("/usr/bin/python3").exists() else (shutil.which("python3") or sys.executable)
)
_THIS_FILE = str(Path(__file__).resolve())
PCC_THRESHOLD = 0.99


def _generate_reference(out_path: Path, batch: int, seq_len: int) -> bool:
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, str(out_path), str(batch), str(seq_len)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(f"reference generation failed for hyperconnection:\n{proc.stderr[-2000:]}")
        return False
    return out_path.is_file()


@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1, 2))
def test_hyperconnection_pcc(device, reset_seeds, tmp_path, batch_size: int, seq_len: int) -> None:
    ref_path = tmp_path / "ref_hc.pt"
    if not _generate_reference(ref_path, batch_size, seq_len):
        pytest.skip("could not generate HF reference for hyperconnection (cached transformers 5.8.1 unavailable)")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])

    hc = DeepSeekV4HyperConnection(cfg, bundle["state_dict"], device)

    hidden_tt = ttnn.from_torch(bundle["hidden_streams"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    post_tt, comb_tt, collapsed_tt = hc.forward(hidden_tt)

    outputs = {
        "post": (ttnn.to_torch(post_tt).reshape(bundle["post"].shape).float(), bundle["post"].float()),
        "comb": (ttnn.to_torch(comb_tt).reshape(bundle["comb"].shape).float(), bundle["comb"].float()),
        "collapsed": (
            ttnn.to_torch(collapsed_tt).reshape(bundle["collapsed"].shape).float(),
            bundle["collapsed"].float(),
        ),
    }

    all_pass = True
    msgs = []
    for name, (got, ref) in outputs.items():
        passing, pcc_message = comp_pcc(ref, got, pcc=PCC_THRESHOLD)
        logger.info(f"[hyperconnection:{name}] {comp_allclose(ref, got)}")
        logger.info(f"[hyperconnection:{name}] PCC: {pcc_message}")
        all_pass = all_pass and passing
        if not passing:
            msgs.append(f"{name}: {pcc_message}")

    assert all_pass, f"hyperconnection PCC < {PCC_THRESHOLD} (batch={batch_size}, seq={seq_len}): {'; '.join(msgs)}"
