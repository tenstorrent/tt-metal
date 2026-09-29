# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 MoE (bead F1) vs the reference ``MoE`` at V4.1 dims.

The input is the reference's own MoE input (``ffn_in``, the normed residual) of V4.1 layer 2 for one
5120-token chunk, from the §6 oracle; synthetic weights (seeded) or the checkpoint's layer 2. The
reference's gate logits, routing, routed sum, shared expert and output come from the vendored
``Gate`` / ``Expert`` modules (reference numerics: fp32 gate GEMM, FP8 activation QDQ into FP4 routed and
FP8 shared expert GEMMs). Device: ``TtV41Moe`` with bfp8 or bfp4 routed experts (dev-spec D-B: both
dtypes, same bars).

G1 bars: gate logits >= 0.997, selected scores >= 0.99, top-6 recall >= 0.95, routed >= 0.96,
final >= 0.982, shared expert >= 0.999; bit-identical repeats.

D-I: ``TtV41Moe`` emulates the reference's FP8 activation QDQ on the expert inputs (the gate reads the
unquantized input, as in the reference); the test also runs the inner ``TtMoe`` without it and reports
both (only the production path is gated).
"""

import json
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tests.v41.moe_reference import LAYER, SEQ, device_weights, reference
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import get_sp_mesh_composer, get_tp_mesh_composer
from models.demos.deepseek_v3_d_p.tt.v41.moe import TtV41Moe
from tests.ttnn.utils_for_testing import comp_pcc

BARS = {"gate_logits": 0.997, "scores": 0.99, "recall": 0.95, "routed": 0.96, "final": 0.982, "shared": 0.999}
EXPERT_DTYPES = {"bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}

MESHES = [
    pytest.param(
        shape,
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=f"mesh-{shape[0]}x{shape[1]}"),
        id=f"fabric2d-mesh-{shape[0]}x{shape[1]}",
    )
    for shape in ((2, 4), (4, 2))
]


def _run(moe: TtV41Moe, mesh_device, x: torch.Tensor, emulate_actq: bool = True) -> dict:
    """One forward. ``emulate_actq=False`` bypasses the wrapper's expert-input FP8 QDQ (D-I comparison only)."""
    shape = tuple(mesh_device.shape)
    tt_x = ttnn.from_torch(
        x[None, None],
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(2, 3)),
    )
    start = time.perf_counter()
    if emulate_actq:
        out, inter = moe(tt_x, return_intermediates=True)
    else:
        out, inter = moe.moe(ttnn.squeeze(tt_x, dim=0), return_intermediates=True, actual_isl=None)
        out = ttnn.unsqueeze(out, dim=0)
    ttnn.synchronize_device(mesh_device)
    elapsed = time.perf_counter() - start
    sp, tp = get_sp_mesh_composer(mesh_device), get_tp_mesh_composer(mesh_device)
    k = C.NUM_EXPERTS_PER_TOKEN
    host = {
        "logits": ttnn.to_torch(inter.gate_logits, mesh_composer=sp).float().reshape(SEQ, -1),
        "weights": ttnn.to_torch(inter.gate_scores, mesh_composer=sp).float().reshape(SEQ, k),
        "indices": ttnn.to_torch(inter.gate_indices, mesh_composer=sp, dtype=torch.int32).reshape(SEQ, k),
        "shared": ttnn.to_torch(inter.shared_output, mesh_composer=tp).float().reshape(SEQ, -1),
        "routed": ttnn.to_torch(inter.routed_output, mesh_composer=tp).float().reshape(SEQ, -1),
        "final": ttnn.to_torch(out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, shape, dims=(2, 3))).reshape(
            SEQ, -1
        ),
        "seconds": elapsed,
    }
    return host


def _pcc(expected, actual) -> float:
    return comp_pcc(expected.float(), actual.float(), 0.0)[1]


def _metrics(ref: dict, dev: dict) -> dict:
    ref_idx, dev_idx = ref["indices"].long(), dev["indices"].long()
    hits = (dev_idx[:, :, None] == ref_idx[:, None, :]).any(-1).sum(-1)
    k = ref_idx.shape[-1]
    return {
        "gate_logits": _pcc(ref["logits"], dev["logits"]),
        # selected-weight distributions (slot order near score ties is not observable after the reduce)
        "scores": _pcc(
            ref["weights"].sort(-1, descending=True).values, dev["weights"].sort(-1, descending=True).values
        ),
        "recall": hits.float().mean().item() / k,
        "top6_set_flip_rate": (hits < k).float().mean().item(),
        "routed": _pcc(ref["routed"], dev["routed"]),
        "shared": _pcc(ref["shared"], dev["shared"]),
        "final": _pcc(ref["final"], dev["final"]),
        "ms": round(dev["seconds"] * 1e3, 2),
    }


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("expert_dtype", list(EXPERT_DTYPES))
@pytest.mark.parametrize("weights_source", ["synthetic", "real"])
@pytest.mark.parametrize("mesh_device, device_params", MESHES, indirect=True)
def test_v41_moe(mesh_device, device_params, weights_source, expert_dtype):
    ref, model = reference(weights_source)
    weights = device_weights(weights_source, model)
    del model
    moe = TtV41Moe(
        mesh_device,
        C,
        LAYER,
        weights,
        SEQ,
        routed_expert_weights_dtype=EXPERT_DTYPES[expert_dtype],
    )
    del weights

    first = _run(moe, mesh_device, ref["x"])
    repeat = _run(moe, mesh_device, ref["x"])
    for name in ("final", "routed", "shared", "logits", "weights", "indices"):
        assert torch.isfinite(first[name].float()).all(), f"non-finite {name}"
        assert torch.equal(first[name], repeat[name]), f"{name} differs between repeated runs"
    first["seconds"] = repeat["seconds"]  # warm (program cache) timing
    unquantized = _run(moe, mesh_device, ref["x"], emulate_actq=False)
    unquantized = _run(moe, mesh_device, ref["x"], emulate_actq=False)  # warm timing

    report = {
        "mesh": list(mesh_device.shape),
        "weights": weights_source,
        "expert_dtype": expert_dtype,
        "device": _metrics(ref, first),
        "device_without_actq_qdq": _metrics(ref, unquantized),
    }
    logger.info(f"V41_MOE_RESULT {json.dumps(report)}")
    failed = {k: v for k, v in report["device"].items() if k in BARS and v < BARS[k]}
    assert not failed, f"below G1 bars {BARS}: {failed}; {report}"
