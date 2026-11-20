# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host arithmetic only (run it under the mock Galaxy): the dumps of tests/test_real.py ($HY4_MOE_DUMP/real_L<n>.pt:
the summed outputs of the 32 shares for the routed experts, the shared expert and the chip program) against exact
arithmetic on the weights the streams hold (BFP4, the shared expert as two BFP4 terms) and against the reference
(the checkpoint's weights), both in float64 on the BF16 x the device has.

One HY4_MK5 line per (layer, output, reference): PCC, relative rms error, gain. The label of the run is
$HY4_MOE_FIDELITY (default custom_mm); the dumps are read from $HY4_MOE_DUMP. Needs $HY4_HF_MODEL."""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe import prepare
from models.experimental.hy4_preview.tests.common import pcc

E, I = 256, L.INTERMEDIATE
LABEL = os.environ.get("HY4_MOE_FIDELITY", "custom_mm")


def round_tiles(w):
    return L.bfp4_round(w.transpose(1, 2)).transpose(1, 2)


def report(layer, name, ref, dev):
    ref, dev = ref.double(), dev.double()
    rel = ((ref - dev).pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    gain = ((ref * dev).sum() / ref.pow(2).sum()).item()
    logger.info(
        f"HY4_MK5 real L{layer} [{LABEL}] {name}: pcc={pcc(ref.float(), dev.float()):.6f} rel_rms={rel:.3e} gain={gain:.5f}"
    )


@pytest.mark.parametrize("layer", [int(v) for v in os.environ.get("HY4_MOE_LAYERS", "1,2,3,4,5").split(",")])
def test_real_exact_host(layer):
    if not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        pytest.skip("host arithmetic: run under the mock Galaxy")
    path = Path(os.environ.get("HY4_MOE_DUMP", "")) / f"real_L{layer}.pt"
    if not path.is_file():
        pytest.skip(f"no dump {path} (tests/test_real.py with HY4_MOE_DUMP)")
    d = torch.load(path)
    x = d["x"].to(torch.bfloat16).double()  # what the device multiplies
    gate_up, down = prepare.layer_experts(layer, shared=False)
    g, u, dn = prepare.checkpoint_tensors(
        [f"model.layers.{layer}.mlp.shared_experts.{n}_proj.weight" for n in ("gate", "up", "down")]
    )
    r_idx, r_w, d_idx, d_w = d["r_idx"], d["r_w"].double(), d["d_idx"], d["d_w"].double()

    out = {
        k: torch.zeros(32, L.HIDDEN, dtype=torch.float64)
        for k in ("routed_ref", "routed_exact", "chip_ref", "chip_exact")
    }

    def add(key, idx, w, gu, dw, e):
        rows = (idx == e).any(dim=-1).nonzero().flatten()
        if len(rows) == 0:
            return
        wr = (w * (idx == e))[rows].sum(dim=-1, keepdim=True)
        s = x[rows] @ gu.T
        out[key][rows] += (L.swiglu(s[:, :I], s[:, I:]) * wr) @ dw.T

    for e in sorted(set(r_idx.flatten().tolist()) | set(d_idx.flatten().tolist())):
        gu, dw = gate_up[e].double(), down[e].double()
        q_gu, q_dw = round_tiles(gate_up[e][None].float())[0].double(), round_tiles(down[e][None].float())[0].double()
        add("routed_ref", r_idx, r_w, gu, dw, e)
        add("routed_exact", r_idx, r_w, q_gu, q_dw, e)
        add("chip_ref", d_idx, d_w, gu, dw, e)
        add("chip_exact", d_idx, d_w, q_gu, q_dw, e)

    sgu, sdw = torch.cat((g, u), dim=0).float(), dn.float()
    shared = {}
    for key, a, b in (
        ("ref", sgu.double(), sdw.double()),
        ("exact", L.two_terms_value(sgu[None])[0].double(), L.two_terms_value(sdw[None])[0].double()),
    ):
        s = x @ a.T
        shared[key] = L.swiglu(s[:, :I], s[:, I:], float("inf")) @ b.T

    for key in ("ref", "exact"):
        what = "the reference (checkpoint weights)" if key == "ref" else "exact arithmetic on the stored weights"
        report(layer, f"routed experts vs {what}", out[f"routed_{key}"], d["routed"])
        report(layer, f"shared expert vs {what}", shared[key], d["shared"])
        report(
            layer,
            f"chip program (the device's picks and weights) vs {what}",
            out[f"chip_{key}"] + shared[key],
            d["chip"],
        )
    report(
        layer,
        "the BFP4 format alone: exact arithmetic on the stored weights vs the reference, routed",
        out["routed_ref"],
        out["routed_exact"],
    )
