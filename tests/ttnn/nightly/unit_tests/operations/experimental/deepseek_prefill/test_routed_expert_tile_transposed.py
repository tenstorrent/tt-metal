# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
``weights_tile_transposed`` of ``unified_routed_expert_moe``: every weight tile is stored transposed within itself
(same tile grid) and the op's matmuls transpose it back as they unpack it.

Why: a bfloat4_b tile shares one exponent per 16 values along a tile row. In the normal (in, out) weight layout those
16 values are 16 output columns of one input row. gpt-oss's K-GPTQ expert weights are rounded on the grid whose groups
are 16 consecutive input rows of one output column (inside one MXFP4 block), so the normal layout re-rounds them.
Transposing each tile puts the input dimension along the tile rows, where those values are stored exactly.

The host test checks that packing claim (no device). The device test runs the real op on one chip with gpt-oss-20b
expert shapes: with the flag it must compute the K-grouped weights' FFN (vs an fp32 reference on exactly those
weights), and more accurately than the normal layout fed the same weights.
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.gpt_oss.tt.experts.prefill_routed import tile_transpose
from tests.ttnn.utils_for_testing import comp_pcc

# gpt-oss-20b: hidden_size = intermediate_size = 2880
EMB_DIM = 2880
HIDDEN_DIM = 2880
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0

SINGLE_CHIP_MESH_PARAMS = [
    pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-chip"),
]


def _bf4_round_trip(t):
    """t through a host bfloat4_b TILE pack and unpack (the device stores exactly these values)."""
    return ttnn.to_torch(ttnn.from_torch(t.contiguous(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


def k_grouped_bf4(w_hf):
    """w_hf (out, in) moved onto the bfloat4_b grid whose 16-value groups run along `in` (as K-GPTQ weights are)."""
    stored = _bf4_round_trip(tile_transpose(w_hf.T.contiguous()))  # (in, out), each tile transposed
    return tile_transpose(stored).T.contiguous()


def _swigluoai_ffn(x, w):
    """Clamped SwiGLU-OAI expert FFN (gpt-oss), fp32. Weights in HF (out, in)."""
    gate = F.linear(x, w["gate_proj"]).clamp(max=SWIGLU_LIMIT)
    up = F.linear(x, w["up_proj"]).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return F.linear((up + 1.0) * gate * torch.sigmoid(SWIGLU_ALPHA * gate), w["down_proj"])


def _k_grouped_weights():
    torch.manual_seed(7)
    return {
        "gate_proj": k_grouped_bf4(torch.randn(HIDDEN_DIM, EMB_DIM) * 0.08),
        "up_proj": k_grouped_bf4(torch.randn(HIDDEN_DIM, EMB_DIM) * 0.08),
        "down_proj": k_grouped_bf4(torch.randn(EMB_DIM, HIDDEN_DIM) * 0.05),
    }


def test_tile_transpose_round_trip():
    """Host only: tile_transpose is its own inverse and commutes with a full transpose."""
    w = torch.randn(96, 160)
    assert torch.equal(tile_transpose(tile_transpose(w)), w)
    assert torch.equal(tile_transpose(w.T.contiguous()), tile_transpose(w).T)
    assert torch.equal(tile_transpose(w)[:32, 32:64], w[:32, 32:64].T)


def test_k_grouped_weights_pack_exactly_tile_transposed():
    """Host only: K-grouped bfloat4_b values survive the tile-transposed pack bit-exactly; the normal pack re-rounds."""
    w = _k_grouped_weights()["gate_proj"]  # (out, in) on the K-grouped grid
    stored_t = tile_transpose(w.T.contiguous())  # what the routed loader packs with GPT_OSS_PREFILL_KGPTQ
    assert torch.equal(_bf4_round_trip(stored_t), stored_t), "tile-transposed pack changed K-grouped values"
    stored_n = w.T.contiguous()  # the normal layout
    changed = (_bf4_round_trip(stored_n) != stored_n).float().mean().item()
    logger.info(f"normal-layout pack re-rounds {changed:.1%} of the K-grouped values")
    assert changed > 0.01, "expected the normal layout to re-round K-grouped values"


def _run_op(mesh_device, x, weights, num_tokens, tile_transposed):
    def idx(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.uint32
        )

    global_idx, counts, offsets = idx([0]), idx([num_tokens]), idx([0])
    torch_weights = {k: tile_transpose(v) for k, v in weights.items()} if tile_transposed else weights
    expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=1,
        global_expert_idx_table=global_idx,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        max_tokens=num_tokens,
        torch_weights=[torch_weights],
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
    )
    inp = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat8_b)
    out = ttnn.experimental.deepseek_prefill.unified_routed_expert_moe(
        inp,
        offsets,
        counts,
        global_idx,
        expert.gate_projs,
        expert.up_projs,
        expert.down_projs,
        max_dispatched_tokens_per_expert=num_tokens,
        compute_kernel_config=expert.compute_kernel_config,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
        weights_tile_transposed=tile_transposed,
    )
    return ttnn.to_torch(out).float().reshape(-1, EMB_DIM)[:num_tokens]


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("num_tokens", [128, 1024], ids=["t128", "t1k"])
@pytest.mark.parametrize(
    "mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
def test_routed_expert_tile_transposed_weights(mesh_device, device_params, num_tokens):
    weights = _k_grouped_weights()
    torch.manual_seed(11)
    x = torch.randn(num_tokens, EMB_DIM)
    with torch.no_grad():
        ref = _swigluoai_ffn(x, weights)

    out_t = _run_op(mesh_device, x, weights, num_tokens, tile_transposed=True)
    out_n = _run_op(mesh_device, x, weights, num_tokens, tile_transposed=False)
    assert torch.isfinite(out_t).all(), "tile-transposed output has NaN / Inf"

    _, pcc_t = comp_pcc(ref, out_t, 0.99)
    _, pcc_n = comp_pcc(ref, out_n, 0.0)
    err_t = ((out_t - ref).norm() / ref.norm()).item()
    err_n = ((out_n - ref).norm() / ref.norm()).item()
    logger.info(
        f"t={num_tokens}: tile-transposed PCC {pcc_t:.6f} rel.err {err_t:.4f}; normal PCC {pcc_n:.6f} rel.err {err_n:.4f}"
    )
    assert pcc_t >= 0.99, f"tile-transposed PCC {pcc_t}"
    assert err_t < err_n, f"tile-transposed ({err_t:.4f}) should beat the re-rounded normal layout ({err_n:.4f})"
