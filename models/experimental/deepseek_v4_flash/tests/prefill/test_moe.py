# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for the ttnn prefill MoE block (``DeepSeekV4PrefillMoE``).

Each test builds the reference ``DeepseekV4SparseMoeBlock`` (fp32, CPU; a standalone copy of the HF
modeling code from ``models.demos.deepseek_v3_d_p.reference.deepseek_v4``) with randomised weights at
the real V4-Flash dimensions, runs it over a whole chunk of tokens, and compares the device block:

* ``test_prefill_moe_pcc``          -- the full block (router + routed experts + shared expert) for a
  learned-router layer and a hash-router layer, at several chunk lengths and expert counts, plus the
  intermediate results that localise a failure: the router's score row (and, on hash layers, the
  gathered expert ids, which must match exactly) and the fraction of tokens that agree with the
  reference token by token,
* ``test_prefill_moe_chunked``      -- a chunk longer than the routed op's 512-token limit, which the
  block feeds to the op in slices,
* ``test_prefill_moe_shared_expert``-- the shared expert alone,
* ``test_prefill_moe_tp4``          -- the real ``I = 2048`` on a 1x4 tensor-parallel submesh (needs an 8x4
  system mesh and fabric, like the decode TP test).

Single-device tests run the *TP=4 slice* of the model: the routed op is built for ``I_local == 512``, so
the reference is configured with ``moe_intermediate_size = 512`` and the block is a plain ``tp_size=1``
block over it. That is the per-chip work of the real model; only the all-reduce is missing, and it is
covered by the TP test.

The routed weights are BFloat4_b (what the model ships) and the op gathers its SwiGLU activations in
bf8, so the floors below are set against that quantization noise, not against bf16 numerics. The
learned router ranks on bf16 scores, so a token whose 6th and 7th best experts are within one bf16 ulp
can pick a different expert than the fp32 reference; those tokens cost a little PCC, which is why the
learned-router floor is looser than the hash-router one (where the selection is exact).

Run::

    pytest -s models/experimental/deepseek_v4_flash/tests/prefill/test_moe.py
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4SparseMoeBlock
from models.experimental.deepseek_v4_flash.tt.decode.moe import (
    DeepSeekV4HashRouter,
    DeepSeekV4PreloadedExperts,
)
from models.experimental.deepseek_v4_flash.tt.prefill.moe import MAX_OP_TOKENS, DeepSeekV4PrefillMoE

_SEED = 1234
_HIDDEN = 4096
_I_LOCAL = 512  # per-chip intermediate width the routed op is built for (I = 2048 at TP = 4)
_TOP_K = 6
_VOCAB = 1024  # hash layers: size of the token-id -> expert-id table
_WEIGHT_STD = 0.02
_WEIGHT_DTYPE = ttnn.bfloat4_b

# Output PCC floors against the fp32 reference (see the module docstring).
_PCC_LEARNED = 0.985
_PCC_HASH = 0.99
_SCORES_PCC = 0.999
# Learned routing: tokens that fall below this per-token PCC are counted as routing disagreements,
# and at most this fraction of the tokens may.
_TOKEN_PCC = 0.95
_MAX_BAD_TOKEN_FRACTION = 0.1


def _config(num_experts: int, inter: int) -> DeepseekV4Config:
    """V4-Flash MoE dimensions; layer 0 is a hash layer, layer 1 a learned one."""
    return DeepseekV4Config(
        hidden_size=_HIDDEN,
        moe_intermediate_size=inter,
        n_routed_experts=num_experts,
        num_experts_per_tok=_TOP_K,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "sliding_attention"],
        mlp_layer_types=["hash_moe", "moe"],
        vocab_size=_VOCAB,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
    )


def _reference(num_experts: int, inter: int, layer_idx: int, num_tokens: int):
    """Random reference layer plus its input/output over ``num_tokens`` tokens.

    Returns ``(module, hidden [1, T, D], token_ids [1, T], output [1, T, D])``. Parameters are
    rounded through bf16 first: the device holds them in bf16 / bf4 anyway, so this keeps the
    comparison about compute fidelity rather than the weight cast.
    """
    torch.manual_seed(_SEED + 100 * layer_idx + num_tokens + num_experts)
    cfg = _config(num_experts, inter)
    module = DeepseekV4SparseMoeBlock(cfg, layer_idx).eval()
    with torch.no_grad():
        for p in module.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=_WEIGHT_STD)
            p.copy_(p.to(torch.bfloat16).to(torch.float32))
        if not module.is_hash:
            module.gate.e_score_correction_bias.normal_(mean=0.0, std=_WEIGHT_STD)
        else:
            # k distinct experts per token id
            table = torch.stack([torch.randperm(num_experts)[:_TOP_K] for _ in range(_VOCAB)])
            module.gate.tid2eid.copy_(table)
    hidden = torch.randn(1, num_tokens, _HIDDEN).to(torch.bfloat16).to(torch.float32)
    token_ids = torch.randint(0, _VOCAB, (1, num_tokens))
    with torch.no_grad():
        output = module(hidden, token_ids) if module.is_hash else module(hidden)
    return module, hidden, token_ids, output


def _model_config(module: DeepseekV4SparseMoeBlock) -> types.SimpleNamespace:
    """The handful of config fields the ttnn modules read."""
    cfg = module.gate
    return types.SimpleNamespace(
        hidden_size=_HIDDEN,
        num_local_experts=cfg.num_experts,
        num_experts_per_tok=_TOP_K,
        moe_intermediate_size=module.experts.intermediate_dim,
        routed_scaling_factor=cfg.routed_scaling_factor,
        swiglu_limit=module.experts.limit,
        rms_norm_eps=1.0e-6,
    )


def _build(module: DeepseekV4SparseMoeBlock, device, tp_size: int = 1) -> DeepSeekV4PrefillMoE:
    """The ttnn prefill block over ``module``'s weights (routed weights uploaded in the decode layout)."""
    config = _model_config(module)
    state_dict = {k: v.detach().to(torch.bfloat16) for k, v in module.state_dict().items()}
    stacked_gate_up = state_dict["experts.gate_up_proj"]  # [E, 2I, D]
    stacked_down = state_dict["experts.down_proj"]  # [E, D, I]

    def provider(e: int):
        return stacked_gate_up[e], stacked_down[e]

    experts = DeepSeekV4PreloadedExperts(config, provider, device, dtype=_WEIGHT_DTYPE, tp_size=tp_size)
    gate = DeepSeekV4HashRouter(config, state_dict, device) if module.is_hash else None
    return DeepSeekV4PrefillMoE(config, state_dict, device, experts=experts, gate=gate, tp_size=tp_size)


def _to_tt(t: torch.Tensor, device, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _check_routing(moe: DeepSeekV4PrefillMoE, module: DeepseekV4SparseMoeBlock, x_tt, hidden, token_ids) -> None:
    """The router's outputs against the reference's: the score row, and on hash layers the ids."""
    num_tokens = hidden.shape[1]
    routing = moe.route(x_tt, token_ids if moe.is_hash else None)
    scores = ttnn.to_torch(routing.scores).reshape(num_tokens, -1).float()
    with torch.no_grad():
        ref_scores = torch.sqrt(F.softplus(F.linear(hidden.reshape(-1, _HIDDEN), module.gate.weight)))
    passing, msg = comp_pcc(ref_scores, scores, _SCORES_PCC)
    logger.info(f"[moe] router scores PCC: {msg}")
    assert passing, f"router score PCC < {_SCORES_PCC}: {msg}"
    if moe.is_hash:
        ids = ttnn.to_torch(routing.indices).reshape(num_tokens, _TOP_K).round().long()
        ref_ids = module.gate.tid2eid[token_ids.reshape(-1)].long()
        assert torch.equal(ids, ref_ids), "hash router gathered different expert ids than tid2eid[token_ids]"


def _run_case(device, num_experts: int, layer_idx: int, num_tokens: int) -> None:
    """One full-block comparison; see the module docstring for the thresholds."""
    module, hidden, token_ids, ref_out = _reference(num_experts, _I_LOCAL, layer_idx, num_tokens)
    moe = _build(module, device)
    logger.info(f"[moe] {'hash' if moe.is_hash else 'learned'} layer, E={num_experts}, T={num_tokens}, I={_I_LOCAL}")

    x_tt = _to_tt(hidden.reshape(1, 1, num_tokens, _HIDDEN), device)
    _check_routing(moe, module, x_tt, hidden, token_ids)

    out_tt = moe(x_tt, token_ids if moe.is_hash else None)
    out = ttnn.to_torch(out_tt).reshape(num_tokens, _HIDDEN).float()
    ref = ref_out.reshape(num_tokens, _HIDDEN).float()

    floor = _PCC_HASH if moe.is_hash else _PCC_LEARNED
    passing, msg = comp_pcc(ref, out, floor)
    logger.info(f"[moe] output PCC: {msg}")

    token_pcc = torch.tensor([torch.corrcoef(torch.stack([ref[t], out[t]]))[0, 1].item() for t in range(num_tokens)])
    bad = (token_pcc < _TOKEN_PCC).float().mean().item()
    logger.info(
        f"[moe] per-token PCC: min {token_pcc.min():.4f}, median {token_pcc.median():.4f}, bad fraction {bad:.3f}"
    )
    assert passing, f"moe output PCC < {floor}: {msg}"
    assert bad <= _MAX_BAD_TOKEN_FRACTION, f"{bad:.1%} of the tokens have PCC < {_TOKEN_PCC}"
    if moe.is_hash:
        assert bad == 0.0, "hash routing is exact, so no token may disagree with the reference"


# (layer type, experts, tokens). Covers one tile row and odd tile-row counts, fewer experts than the 15
# core groups, and the full 256-expert scale.
_CASES = {
    "learned_e64_t32": (1, 64, 32),
    "learned_e64_t96": (1, 64, 96),
    "learned_e64_t480": (1, 64, 480),
    "hash_e64_t96": (0, 64, 96),
    "hash_e64_t480": (0, 64, 480),
    "learned_e256_t480": (1, 256, 480),
    "hash_e256_t480": (0, 256, 480),
}


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_prefill_moe_pcc(device, reset_seeds, case):
    layer_idx, num_experts, num_tokens = _CASES[case]
    _run_case(device, num_experts, layer_idx, num_tokens)


@pytest.mark.parametrize("layer_idx", (0, 1), ids=("hash", "learned"))
def test_prefill_moe_chunked(device, reset_seeds, layer_idx):
    """T > 512: the routed op is fed in slices of ``MAX_OP_TOKENS`` (here 512 + 512 + 32 rows)."""
    _run_case(device, 64, layer_idx, 2 * MAX_OP_TOKENS + 32)


def test_prefill_moe_shared_expert(device, reset_seeds):
    """The shared expert alone (dense SwiGLU, no clamp) against the reference MLP."""
    num_tokens = 96
    module, hidden, _, _ = _reference(16, _I_LOCAL, 1, num_tokens)
    moe = _build(module, device)
    x_tt = _to_tt(hidden.reshape(1, 1, num_tokens, _HIDDEN), device)
    out = ttnn.to_torch(moe.shared_experts(x_tt)).reshape(num_tokens, _HIDDEN).float()
    # The decode shared expert applies no clamp, so compare against the unclamped dense MLP (the
    # random activations here are far below the clamp anyway).
    sd = module.shared_experts
    with torch.no_grad():
        x = hidden.reshape(num_tokens, _HIDDEN)
        ref = sd.down_proj(F.silu(sd.gate_proj(x)) * sd.up_proj(x))
    passing, msg = comp_pcc(ref, out, 0.999)
    logger.info(f"[moe] shared expert PCC: {msg}")
    assert passing, f"shared expert PCC < 0.999: {msg}"


# --------------------------------------------------------------------------- #
# Tensor parallel: the real I = 2048 on a 1x4 submesh.
# --------------------------------------------------------------------------- #
_TP_SIZE = 4
_PARENT_MESH = (8, 4)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("layer_idx", (0, 1), ids=("hash", "learned"))
def test_prefill_moe_tp4(mesh_device, reset_seeds, layer_idx):
    """Full-width experts (I = 2048) column/row-sharded over a 1x4 submesh; one all-reduce at the end."""
    if tuple(mesh_device.shape) != _PARENT_MESH:
        pytest.skip(f"need an {_PARENT_MESH[0]}x{_PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(0, 0))
    assert submesh.get_num_devices() == _TP_SIZE

    num_tokens, num_experts = 64, 16
    module, hidden, token_ids, ref_out = _reference(num_experts, 4 * _I_LOCAL, layer_idx, num_tokens)
    moe = _build(module, submesh, tp_size=_TP_SIZE)

    x_tt = ttnn.from_torch(
        hidden.reshape(1, 1, num_tokens, _HIDDEN).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    out_tt = moe(x_tt, token_ids if moe.is_hash else None)
    # The all-reduce replicates the residual; read one chip's copy.
    out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))[0]
    out = out.reshape(num_tokens, _HIDDEN).float()

    floor = _PCC_HASH if moe.is_hash else _PCC_LEARNED
    passing, msg = comp_pcc(ref_out.reshape(num_tokens, _HIDDEN).float(), out, floor)
    logger.info(f"[moe tp{_TP_SIZE}] PCC: {msg}")
    assert passing, f"moe TP{_TP_SIZE} PCC < {floor}: {msg}"
