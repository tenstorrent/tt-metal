# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Real-checkpoint PCC test for the ttnn DeepSeek-V4-Flash MoE block (prefill).

Unlike ``test_moe_pcc.py`` (reduced config + random weights, validated against
the HF reference via a subprocess), this test loads the *actual* V4-Flash
checkpoint weights for one ``moe`` decoder layer through
:class:`DeepseekV4WeightLoader`, dequantizes them on host, and compares:

* **reference**: a faithful pure-torch reimplementation of ``DeepseekV4SparseMoeBlock``
  (router -> routed experts -> ``+`` shared expert), run in fp32 on host, and
* **device**: the ttnn block (:class:`DeepSeekV4SparseMoeBlock` with a
  :class:`DeepSeekV4PreloadedExperts`, which keeps all 256 experts on device in
  BFloat4_b so they fit).

The device keeps all 256 routed experts resident in BFloat4_b (4-bit), so the
reference reads those exact quantised weights back for its routed-expert math;
the PCC gap then isolates the ttnn compute path rather than the deliberate 4-bit
storage choice. Everything runs in the ttnn venv — no cached transformers /
subprocess is needed because the reference is hand-written.

The routed experts are MXFP4 (int8-packed fp4 + e8m0 block scale); the shared
expert and the dense projections are block-FP8 (e4m3 + e8m0); the router gate is
bf16. See ``tt/quant.py`` for the dequantizers.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tt.moe import (
    DeepSeekV4PreloadedExperts,
    DeepSeekV4SparseMoeBlock,
)
from models.experimental.deepseek_v4_flash.tt.quant import dequantize_weight
from models.experimental.deepseek_v4_flash.tt.weight_loader import (
    DeepseekV4WeightLoader,
    resolve_snapshot_dir,
)

DEFAULT_MODEL_DIR = Path("/home/ttuser/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash")

# Layers 0..2 are ``hash_moe`` (frozen tid2eid routing); 3+ are standard ``moe``.
MOE_LAYER = 5
PCC_THRESHOLD = 0.99


def _checkpoint_available() -> bool:
    try:
        resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return False
    return True


def _load_config(loader: DeepseekV4WeightLoader) -> types.SimpleNamespace:
    cfg_path = loader.snapshot_dir / "config.json"
    raw = json.loads(cfg_path.read_text())
    return types.SimpleNamespace(
        hidden_size=raw["hidden_size"],
        num_local_experts=raw["n_routed_experts"],
        num_experts_per_tok=raw["num_experts_per_tok"],
        moe_intermediate_size=raw["moe_intermediate_size"],
        routed_scaling_factor=raw.get("routed_scaling_factor", 1.5),
        swiglu_limit=raw.get("swiglu_limit", 10.0),
        rms_norm_eps=raw.get("rms_norm_eps", 1.0e-6),
    )


def _dq(loader: DeepseekV4WeightLoader, name: str) -> torch.Tensor:
    """Dequantize an HF-named tensor to fp32 via its companion scale."""
    return dequantize_weight(loader.get_tensor(name), loader.get_scale(name))


def _expert_provider(loader: DeepseekV4WeightLoader, layer: int):
    """Return ``provider(e) -> (gate_up [2I, H], down [H, I])`` in bf16.

    ``gate_up`` is the HF packed layout ``cat([gate_proj, up_proj])`` (rows
    ``0:I`` gate, ``I:2I`` up), matching ``DeepseekV4Experts.gate_up_proj``.
    """

    def provider(e: int):
        base = f"layers.{layer}.mlp.experts.{e}"
        gate = _dq(loader, f"{base}.gate_proj.weight")  # [I, H]
        up = _dq(loader, f"{base}.up_proj.weight")  # [I, H]
        down = _dq(loader, f"{base}.down_proj.weight")  # [H, I]
        gate_up = torch.cat([gate, up], dim=0).to(torch.bfloat16)  # [2I, H]
        return gate_up, down.to(torch.bfloat16)

    return provider


def _torch_router_topk(
    flat: torch.Tensor, gate_w: torch.Tensor, gate_bias: torch.Tensor, cfg: types.SimpleNamespace
) -> list[set]:
    """fp32 ``DeepseekV4TopKRouter`` top-k selection (for the routing diagnostic)."""
    logits = flat @ gate_w.float().t()
    scores = torch.sqrt(F.softplus(logits))
    idx = torch.topk(scores + gate_bias.float(), cfg.num_experts_per_tok, dim=-1, sorted=False).indices
    return [set(r.tolist()) for r in idx]


def _torch_experts_and_shared(
    flat: torch.Tensor,
    dense_w: torch.Tensor,
    experts: DeepSeekV4PreloadedExperts,
    shared: dict,
    cfg: types.SimpleNamespace,
) -> torch.Tensor:
    """fp32 routed-experts + shared-expert compute for a *given* routing.

    Takes the dense per-token/expert routing weights ``dense_w [T, E]`` (already
    softmax-normalised and ``routed_scaling_factor``-scaled) so the comparison
    isolates the expert / shared arithmetic from bf16 top-k *selection* noise,
    which is reported separately. Routed-expert weights are read back from the
    device (``experts.expert_matmul_weights``) so the reference consumes the same
    BFloat4-quantised values the on-device matmuls do — the PCC then reflects
    compute fidelity, not the deliberate 4-bit storage choice. Matches the
    per-expert math of ``DeepseekV4Experts`` + ``DeepseekV4MLP``.
    """
    limit = cfg.swiglu_limit
    routed = torch.zeros_like(flat)
    for e in (dense_w.abs().sum(0) > 0).nonzero().flatten().tolist():
        gate_up, down = experts.expert_matmul_weights(e)  # [H, 2I], [I, H] (bf4-rounded fp32)
        x = flat @ gate_up  # [T, 2I]
        gate, up = x.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        act = F.silu(gate) * up
        out_e = act @ down  # [T, H]
        routed += out_e * dense_w[:, e : e + 1]

    sg, su, sd = shared["gate"].float(), shared["up"].float(), shared["down"].float()
    shared_out = (F.silu(flat @ sg.t()) * (flat @ su.t())) @ sd.t()
    return routed + shared_out


@pytest.mark.skipif(
    not _checkpoint_available(),
    reason=f"V4-Flash checkpoint not found under {DEFAULT_MODEL_DIR}",
)
@torch.no_grad()
@pytest.mark.parametrize("seq_len", (32,))
@pytest.mark.parametrize("batch_size", (1,))
def test_moe_real_weights_pcc(device, reset_seeds, batch_size: int, seq_len: int) -> None:
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    if not loader.has(f"layers.{MOE_LAYER}.mlp.gate.e_score_correction_bias"):
        pytest.skip(f"layer {MOE_LAYER} is not a standard `moe` layer")
    cfg = _load_config(loader)

    gate_w = loader.get_tensor(f"layers.{MOE_LAYER}.mlp.gate.weight")  # bf16 [E, H]
    gate_bias = loader.get_tensor(f"layers.{MOE_LAYER}.mlp.gate.e_score_correction_bias")  # f32 [E]
    shared = {
        "gate": _dq(loader, f"layers.{MOE_LAYER}.mlp.shared_experts.gate_proj.weight"),
        "up": _dq(loader, f"layers.{MOE_LAYER}.mlp.shared_experts.up_proj.weight"),
        "down": _dq(loader, f"layers.{MOE_LAYER}.mlp.shared_experts.down_proj.weight"),
    }
    provider = _expert_provider(loader, MOE_LAYER)

    torch.manual_seed(1234)
    hidden = torch.randn(batch_size, seq_len, cfg.hidden_size, dtype=torch.float32)
    flat = hidden.reshape(-1, cfg.hidden_size).float()

    # ttnn block: real router + shared expert weights, streaming routed experts.
    weights = {
        "gate.weight": gate_w,
        "gate.e_score_correction_bias": gate_bias,
        "shared_experts.gate_proj.weight": shared["gate"].to(torch.bfloat16),
        "shared_experts.up_proj.weight": shared["up"].to(torch.bfloat16),
        "shared_experts.down_proj.weight": shared["down"].to(torch.bfloat16),
    }
    experts = DeepSeekV4PreloadedExperts(cfg, provider, device)
    block = DeepSeekV4SparseMoeBlock(cfg, weights, device, experts=experts)

    hidden_tt = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x_flat = ttnn.reshape(hidden_tt, [1, 1, flat.shape[0], cfg.hidden_size])

    # Drive the fp32 reference with the *device* router's routing so the PCC
    # compares expert / shared arithmetic on identical routing. The router still
    # runs on device (here and again inside ``block.forward``, deterministically);
    # bf16 top-k *selection* divergence vs fp32 is reported separately below
    # because it is an inherent dtype effect, not a port bug.
    dense_w_tt = block.gate(x_flat)
    dense_w = ttnn.to_torch(dense_w_tt).reshape(flat.shape[0], cfg.num_local_experts).float()

    reference = _torch_experts_and_shared(flat, dense_w, experts, shared, cfg).reshape(hidden.shape)

    out_tt = block.forward(hidden_tt)
    out_torch = ttnn.to_torch(out_tt).reshape(reference.shape).to(torch.float32)

    # Routing-agreement diagnostic: how often the bf16 device router picks the
    # same expert set as the fp32 reference router (soft guard against a broken
    # router, while tolerating boundary flips among near-tied scores).
    ref_sets = _torch_router_topk(flat, gate_w, gate_bias, cfg)
    tt_sets = [set((dense_w[t] > 0).nonzero().flatten().tolist()) for t in range(flat.shape[0])]
    overlap = sum(len(a & b) for a, b in zip(ref_sets, tt_sets)) / len(ref_sets)
    logger.info(f"[moe real weights] router overlap {overlap:.3f}/{cfg.num_experts_per_tok} experts/token")

    passing, pcc_message = comp_pcc(reference, out_torch, pcc=PCC_THRESHOLD)
    logger.info(comp_allclose(reference, out_torch))
    logger.info(f"[moe real weights] layer={MOE_LAYER} PCC: {pcc_message}")

    assert overlap >= cfg.num_experts_per_tok - 1.0, f"router selection overlap too low: {overlap:.3f}"
    assert passing, f"real-weights moe PCC < {PCC_THRESHOLD} (batch={batch_size}, seq={seq_len}): {pcc_message}"
