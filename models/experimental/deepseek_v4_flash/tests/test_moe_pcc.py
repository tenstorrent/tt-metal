# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Per-layer PCC test for the ttnn ``DeepSeekV4SparseMoeBlock`` (prefill).

This single file plays two roles (same split as ``test_attention_pcc.py``):

* **pytest side** (imported by the ttnn venv): builds the ttnn MoE port with the
  reference's weights/inputs and compares the output with PCC.
* **reference side** (run as ``__main__`` under the *system* interpreter): the
  gold reference is HuggingFace ``transformers==5.8.1``
  ``DeepseekV4SparseMoeBlock`` (whose ``forward`` is byte-identical to the
  repo's ``modular_deepseek_v4.py``). It dumps deterministic random weights,
  the input, and the reference output to a ``.pt`` bundle.

The two interpreters are kept apart on purpose: the ttnn venv's ``transformers``
predates ``deepseek_v4``, and the only install on the box that ships it (cached
``transformers==5.8.1``) imports cleanly only under the system interpreter. So
the pytest side re-invokes *this same file* as a subprocess under the system
python to produce the reference. The ``__main__`` guard runs *before* the ttnn
imports, so the subprocess never imports ttnn and the venv never imports the
cached transformers.

Scope: the standard top-k routed MoE layer (``mlp_layer_types == "moe"``). The
static ``hash_moe`` router is not exercised here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


# Cached transformers 5.8.1 (the only install with ``deepseek_v4`` / ``gpt_oss``).
_CACHED_TRANSFORMERS = "/home/ttuser/.cache/uv/archive-v0/U5SPsIWJupLz-bDcPI13a"


# --------------------------------------------------------------------------- #
# Reference side (executed only as ``__main__`` under the system interpreter).
# --------------------------------------------------------------------------- #
def _reference_build_config(DeepseekV4Config):
    """Reduced-but-tile-friendly config (mirrors the attention test).

    Small ``n_routed_experts`` / ``moe_intermediate_size`` keep the dense
    expert compute cheap; all dims are multiples of the 32-wide tile.
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

    # The cached transformers 5.8.1 wheel pins ``tokenizers>=0.22``; the box has
    # 0.21.4. The version is only enforced by an import-time check, so spoof it.
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
    layer_idx = config.mlp_layer_types.index("moe")
    dtype = torch.float32

    moe = M.DeepseekV4SparseMoeBlock(config, layer_idx).to(dtype).eval()
    # Reinit *all* params deterministically (gate_up_proj / down_proj parameters
    # are ``torch.empty``). Also fill the ``e_score_correction_bias`` buffer with
    # small random values so the routing bias path is exercised on both sides.
    with torch.no_grad():
        for p in moe.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
        moe.gate.e_score_correction_bias.normal_(mean=0.0, std=0.02)

    hidden = torch.randn(batch, seq_len, config.hidden_size, dtype=dtype)

    with torch.no_grad():
        output = moe(hidden)

    bundle = {
        "state_dict": {k: v.detach().cpu() for k, v in moe.state_dict().items()},
        "hidden": hidden,
        "output": output,  # [B, S, hidden]
        "config": {
            "hidden_size": config.hidden_size,
            "num_local_experts": config.num_local_experts,
            "num_experts_per_tok": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "routed_scaling_factor": config.routed_scaling_factor,
            "swiglu_limit": config.swiglu_limit,
            "rms_norm_eps": config.rms_norm_eps,
        },
    }
    torch.save(bundle, out_path)
    print(f"REFERENCE_OK moe -> {out_path}")


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
from models.experimental.deepseek_v4_flash.tt.moe import (  # noqa: E402
    DeepSeekV4PreloadedExperts,
    DeepSeekV4SparseMoeBlock,
)


# The reference needs the cached transformers 5.8.1, which imports cleanly only
# under the system interpreter (the ttnn venv's huggingface_hub/transformers are
# too old). Fall back to whatever is on PATH if the canonical path is missing.
_SYSTEM_PYTHON = (
    "/usr/bin/python3" if Path("/usr/bin/python3").exists() else (shutil.which("python3") or sys.executable)
)
_THIS_FILE = str(Path(__file__).resolve())
PCC_THRESHOLD = 0.99


def _generate_reference(out_path: Path, batch: int, seq_len: int) -> bool:
    """Run *this file* as the reference generator subprocess. False if it can't run."""
    proc = subprocess.run(
        [_SYSTEM_PYTHON, _THIS_FILE, str(out_path), str(batch), str(seq_len)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        logger.warning(f"reference generation failed for moe:\n{proc.stderr[-2000:]}")
        return False
    return out_path.is_file()


def _to_tt(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize("seq_len", (128,))
@pytest.mark.parametrize("batch_size", (1, 2))
def test_moe_pcc(device, reset_seeds, tmp_path, batch_size: int, seq_len: int) -> None:
    ref_path = tmp_path / "ref_moe.pt"
    if not _generate_reference(ref_path, batch_size, seq_len):
        pytest.skip("could not generate HF reference for moe (cached transformers 5.8.1 unavailable)")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])

    # Routed experts arrive stacked (``[E, 2I, H]`` / ``[E, H, I]``); feed them to
    # the on-device experts via a per-expert provider. bf16 storage keeps the PCC
    # comparison about compute fidelity rather than the BFloat4 storage choice.
    state_dict = bundle["state_dict"]
    stacked_gate_up = state_dict["experts.gate_up_proj"]  # [E, 2I, H]
    stacked_down = state_dict["experts.down_proj"]  # [E, H, I]

    def _provider(e: int):
        return stacked_gate_up[e], stacked_down[e]  # ([2I, H], [H, I])

    experts = DeepSeekV4PreloadedExperts(cfg, _provider, device, dtype=ttnn.bfloat16)
    moe = DeepSeekV4SparseMoeBlock(cfg, state_dict, device, experts=experts)

    hidden_tt = _to_tt(bundle["hidden"], device)
    out_tt = moe.forward(hidden_tt)
    out_torch = ttnn.to_torch(out_tt).reshape(bundle["output"].shape).to(torch.float32)

    reference = bundle["output"].to(torch.float32)
    passing, pcc_message = comp_pcc(reference, out_torch, pcc=PCC_THRESHOLD)
    logger.info(comp_allclose(reference, out_torch))
    logger.info(f"[moe] PCC: {pcc_message}")

    assert passing, f"moe PCC < {PCC_THRESHOLD} (batch={batch_size}, seq={seq_len}): {pcc_message}"
