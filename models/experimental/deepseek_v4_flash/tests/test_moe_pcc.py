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


def _cached_transformers_candidates() -> list[str]:
    """Cached ``transformers`` wheels that ship ``deepseek_v4``, newest first.

    The venv's own ``transformers`` predates ``deepseek_v4``, so the reference has to
    import one out of the uv wheel cache. Which versions are cached changes as uv
    evicts entries, and not every cached wheel imports cleanly under the system
    interpreter (its other dependencies may be missing), so the candidates are
    discovered and tried in turn rather than pinned to one path.
    """
    roots = Path("/home/ttuser/.cache/uv/archive-v0").glob("*/transformers/models/deepseek_v4")
    ordered = sorted(roots, key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p.parents[2]) for p in ordered]


# --------------------------------------------------------------------------- #
# Reference side (executed only as ``__main__`` under the system interpreter).
# --------------------------------------------------------------------------- #
def _reference_build_config(DeepseekV4Config):
    """The real DeepSeek-V4-Flash MoE config.

    Every field the MoE block reads is the production value from
    ``configuration_deepseek_v4.py`` -- in particular ``hidden_size=4096`` and
    ``moe_intermediate_size=2048``, which is what the ``fused_experts`` device op
    behind :class:`DeepSeekV4PreloadedExperts` is hard-wired to (H must be exactly
    64 cores * 2 tiles * 32). The op has no fallback path, so a reduced hidden size
    would not exercise it at all.

    Two deliberate reductions, neither of which changes the per-token math:
      * ``n_routed_experts`` 256 -> 8. Every expert is a [2I, H] + [H, I] pair, so
        the real 256 would be ~25 GB of fp32 reference weights. 8 is the smallest
        count that still leaves a real choice for ``num_experts_per_tok=6``.
      * ``num_hidden_layers`` 43 -> 3. Only one layer is ever built, and the index
        picks which; the block itself is layer-independent.
    """
    return DeepseekV4Config(
        hidden_size=4096,
        q_lora_rank=1024,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        o_groups=8,
        o_lora_rank=1024,
        num_hidden_layers=3,
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        mlp_layer_types=["moe", "moe", "moe"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        sliding_window=128,
        rms_norm_eps=1.0e-6,
        max_position_embeddings=1048576,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        vocab_size=129280,
        n_routed_experts=8,
        num_experts_per_tok=6,
        n_shared_experts=1,
        moe_intermediate_size=2048,
        scoring_func="sqrtsoftplus",
        norm_topk_prob=True,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        attn_implementation="eager",
    )


def _reference_main() -> None:
    """Generate the gold-reference bundle. Args: <out_path> [batch] [seq_len]."""
    import importlib.metadata as _md
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    # The cached transformers 5.8.1 wheel pins ``tokenizers>=0.22``; the box has
    # 0.21.4. The version is only enforced by an import-time check, so spoof it.
    _orig_version = _md.version
    _md.version = lambda name: "0.22.0" if name.lower() == "tokenizers" else _orig_version(name)

    out_path = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seq_len = int(sys.argv[3]) if len(sys.argv) > 3 else 32

    torch.manual_seed(1234)
    config = _reference_build_config(DeepseekV4Config)
    layer_idx = config.mlp_layer_types.index("moe")
    dtype = torch.float32

    moe = M.DeepseekV4SparseMoeBlock(config, layer_idx).to(dtype).eval()
    # Reinit *all* params deterministically (gate_up_proj / down_proj parameters
    # are ``torch.empty``). Also fill the ``e_score_correction_bias`` buffer with
    # small random values so the routing bias path is exercised on both sides.
    #
    # Weights are round-tripped through bfloat16 before the reference runs. The ttnn
    # side uploads them as bf16 either way, so this keeps the comparison about compute
    # fidelity rather than the weight cast, and halves the (already ~400 MB at these
    # real dims) bundle written to disk.
    with torch.no_grad():
        for p in moe.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.02)
            p.copy_(p.to(torch.bfloat16).to(dtype))
        moe.gate.e_score_correction_bias.normal_(mean=0.0, std=0.02)

    hidden = torch.randn(batch, seq_len, config.hidden_size, dtype=dtype)

    with torch.no_grad():
        output = moe(hidden)

    bundle = {
        # bf16 is lossless here (the weights were rounded to bf16 above) and keeps the
        # bundle to a size worth writing to disk at the real expert dims.
        "state_dict": {k: v.detach().cpu().to(torch.bfloat16) for k, v in moe.state_dict().items()},
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
_REFERENCE_PYTHONS = [
    p for p in (sys.executable, "/usr/bin/python3", shutil.which("python3")) if p and Path(p).exists()
]
_THIS_FILE = str(Path(__file__).resolve())

# Expert-weight storage formats, with the PCC each is expected to reach against the
# fp32 reference. bfloat4_b is what the model actually ships (256 resident experts only
# fit in DRAM at 4 bits), so it is the configuration that matters; bfloat8_b is included
# as the low-quantization-error control that isolates compute fidelity from storage.
# bfloat16 is not testable here: at the real dims its weight slices blow the op's L1
# budget (the gathered activation block already costs ~768 KB per core).
# Observed: bfloat8_b 0.9999, bfloat4_b 0.9943.
WEIGHT_DTYPE_PCC = {
    ttnn.bfloat8_b: 0.999,
    ttnn.bfloat4_b: 0.99,
}


def _generate_reference(out_path: Path, batch: int, seq_len: int) -> bool:
    """Run *this file* as the reference generator subprocess. False if it can't run.

    Which interpreter can import the cached ``deepseek_v4`` transformers depends on
    where that wheel's own dependencies (``tokenizers`` in particular) happen to be
    installed, so each available interpreter is tried in turn. The ``__main__`` guard
    runs before this file's ttnn imports, so the subprocess never pulls in ttnn
    regardless of which one is used.
    """
    failures = []
    for python in _REFERENCE_PYTHONS:
        proc = subprocess.run(
            [python, _THIS_FILE, str(out_path), str(batch), str(seq_len)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and out_path.is_file():
            return True
        failures.append(f"--- {python}\n{proc.stderr[-1000:]}")
    logger.warning("reference generation failed for moe:\n" + "\n".join(failures))
    return False


def _to_tt(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


# At the real dims the reference is a dense [T, E] expert sweep over 4096x4096 weights
# and the ttnn side runs one fused_experts op per token, so the token count is kept
# modest; it does not change what either path computes.
@pytest.mark.parametrize("weight_dtype", tuple(WEIGHT_DTYPE_PCC), ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("seq_len", (32,))
@pytest.mark.parametrize("batch_size", (1, 2))
def test_moe_pcc(device, reset_seeds, tmp_path, batch_size: int, seq_len: int, weight_dtype) -> None:
    ref_path = tmp_path / "ref_moe.pt"
    if not _generate_reference(ref_path, batch_size, seq_len):
        pytest.skip("could not generate HF reference for moe (cached transformers 5.8.1 unavailable)")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])

    pcc_threshold = WEIGHT_DTYPE_PCC[weight_dtype]

    # Routed experts arrive stacked (``[E, 2I, H]`` / ``[E, H, I]``); feed them to
    # the on-device experts via a per-expert provider.
    state_dict = bundle["state_dict"]
    stacked_gate_up = state_dict["experts.gate_up_proj"]  # [E, 2I, H]
    stacked_down = state_dict["experts.down_proj"]  # [E, H, I]

    def _provider(e: int):
        return stacked_gate_up[e], stacked_down[e]  # ([2I, H], [H, I])

    experts = DeepSeekV4PreloadedExperts(cfg, _provider, device, dtype=weight_dtype)
    moe = DeepSeekV4SparseMoeBlock(cfg, state_dict, device, experts=experts)

    # The ttnn block takes [B, S, 1, H] (the reference's [B, S, H] with the tile row axis).
    hidden_tt = _to_tt(bundle["hidden"].unsqueeze(2), device)
    out_tt = moe.forward(hidden_tt)
    out_torch = ttnn.to_torch(out_tt).reshape(bundle["output"].shape).to(torch.float32)

    reference = bundle["output"].to(torch.float32)
    passing, pcc_message = comp_pcc(reference, out_torch, pcc=pcc_threshold)
    logger.info(comp_allclose(reference, out_torch))
    logger.info(f"[moe] PCC: {pcc_message}")

    assert passing, f"moe PCC < {pcc_threshold} (batch={batch_size}, seq={seq_len}, {weight_dtype}): {pcc_message}"
