# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Dense MLP vs a torch reference at real dims.
Pattern: ``minimax_m3/tests/unit/test_dense_mlp_vs_ref.py``.

Mistral's MLP is ``down(silu(gate(x)) * up(x))`` at hidden 12288 / intermediate 28672, on EVERY one
of the 88 layers (there is no MoE and no per-layer schedule). Built with the production ``MLP``
class and random weights, compared against the HF-derived torch reference on the same weights.

On the target mesh this exercises the whole reason the entry has a donor: gate/up are
column-parallel over TP (each device computes intermediate/8 = 3584 columns) and down is
row-parallel, so every device's ``down`` output is a PARTIAL SUM — the closing TP all-reduce is what
makes the result correct, not an optimization. A missing collective here shows up as a PCC of
roughly 1/8 the reference, which is exactly what this test would catch.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.reference.mistral_config import reduced_text_config
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.mlp import MLP

from ..test_factory import build_mesh_and_ccl, from_device_replicated, parametrize_mesh, to_device


@parametrize_mesh()
@pytest.mark.parametrize(
    "tokens, hidden, inter",
    [
        (128, C.HIDDEN_SIZE, C.INTERMEDIATE_SIZE),  # the real layer geometry
        (512, C.HIDDEN_SIZE, C.INTERMEDIATE_SIZE),  # a longer chunk
    ],
    ids=["t128", "t512"],
)
def test_dense_mlp_vs_ref(mesh_device, device_params, tokens, hidden, inter, reset_seeds):
    """The production MLP vs the torch reference, random weights, real dims."""
    # HF Linear layout [out, in]. Small scale keeps the post-silu distribution non-degenerate and
    # the bf8 weight quantization honest at 28672-wide contractions.
    gate_w = torch.randn(inter, hidden) * 0.02
    up_w = torch.randn(inter, hidden) * 0.02
    down_w = torch.randn(hidden, inter) * 0.02
    x = torch.randn(1, 1, tokens, hidden) * 0.1

    hf_config = reduced_text_config(hidden_size=hidden, intermediate_size=inter)
    ref = reference.mlp_reference(
        x, {"gate_proj.weight": gate_w, "up_proj.weight": up_w, "down_proj.weight": down_w}, hf_config
    )

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    mlp = MLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict={"gate_proj.weight": gate_w, "up_proj.weight": up_w, "down_proj.weight": down_w},
        mesh_config=mesh_config,
        ccl_manager=ccl,
        weight_dtype=ttnn.bfloat16,  # isolate the parallel split; bf8 weights are covered end-to-end
    )

    out_tt = mlp(to_device(x, mesh_device))
    # Post-all-reduce the result is replicated across TP, so device 0 holds the full answer.
    out = from_device_replicated(out_tt, (1, 1, tokens, hidden))

    passing, pcc = comp_pcc(ref.reshape(1, 1, tokens, hidden), out, SPEC.pcc)
    logger.info(f"dense_mlp tokens={tokens} hidden={hidden} inter={inter} tp={mesh_config.tp}: pcc={pcc}")
    assert passing, f"MLP PCC fail (tokens={tokens}): {pcc}"


@parametrize_mesh()
def test_dense_mlp_spec_weight_dtypes(mesh_device, device_params, reset_seeds):
    """The MLP must load at the dtypes the BINDING spec fixes, and still clear the PCC bar.

    ``dataformats.weights.mlp.{up,gate,down}`` inherit ``weights.default`` = bfloat8_b here, so this
    is the configuration the model actually serves at — worth its own row, because a bf8 28672-deep
    contraction is where a dataformat mistake would first show.
    """
    tokens, hidden, inter = 128, C.HIDDEN_SIZE, C.INTERMEDIATE_SIZE
    gate_w = torch.randn(inter, hidden) * 0.02
    up_w = torch.randn(inter, hidden) * 0.02
    down_w = torch.randn(hidden, inter) * 0.02
    x = torch.randn(1, 1, tokens, hidden) * 0.1

    hf_config = reduced_text_config(hidden_size=hidden, intermediate_size=inter)
    ref = reference.mlp_reference(
        x, {"gate_proj.weight": gate_w, "up_proj.weight": up_w, "down_proj.weight": down_w}, hf_config
    )

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    mlp = MLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict={"gate_proj.weight": gate_w, "up_proj.weight": up_w, "down_proj.weight": down_w},
        mesh_config=mesh_config,
        ccl_manager=ccl,
        weight_dtype=None,  # -> the spec's per-projection dataformats
    )
    assert mlp.gate_proj.dtype == SPEC.mlp_gate_dtype
    assert mlp.up_proj.dtype == SPEC.mlp_up_dtype
    assert mlp.down_proj.dtype == SPEC.mlp_down_dtype

    out = from_device_replicated(mlp(to_device(x, mesh_device)), (1, 1, tokens, hidden))
    passing, pcc = comp_pcc(ref.reshape(1, 1, tokens, hidden), out, SPEC.pcc)
    logger.info(f"dense_mlp at spec dtypes ({SPEC.mlp_gate_dtype}): pcc={pcc}")
    assert passing, f"MLP PCC fail at the spec's weight dataformats: {pcc}"
