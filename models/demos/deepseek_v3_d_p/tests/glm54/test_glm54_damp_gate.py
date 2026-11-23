# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 damp gate on device (``tt.glm54.damp_gate.TtDampGate``) against the reference op
``damp_gate_reference`` (FP32) on the device's own bf16 inputs, on the 8x4 mesh (SP 8 rows, TP 4 hidden).

Both gate widths (A1a) through the one code path: scalar per token (the synthetic weights' form) and per channel.
Each case checks beta, x_k and the update x_k - x_{k-1} = beta * (h - x_{k-1}) at PCC >= GATE_PCC (the approved
bar), beta's largest absolute error (a scale check PCC cannot make), that a second eager run is bit-identical,
and that a captured trace replays bit-identical to eager. Negative controls hold the bar against the faults it
exists for: a gate that reads the pass output h instead of the pass input (A1c), one that drops the norm gain
(A1e), and one that drops the bias (A1d) must each score below it.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.glm_5_4 import damp_gate_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_4.synthetic_gates import GLM54_SYNTHETIC_SEED, synthetic_gates
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import glm_5_4_hf_config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep
from models.demos.deepseek_v3_d_p.tt.glm54.damp_gate import TtDampGate
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

# Approved acceptance (epic q171 DESIGN): gate PCC >= 0.999 vs the reference op.
GATE_PCC = 0.999
# beta lies in (0, 1); a dropped bias or a wrong gate moves it by >= 0.1 (pass-1 centre 0.2 vs sigmoid(0) = 0.5).
BETA_MAX_ABS_ERR = 0.01
SEQ = 1024  # 128 rows per SP chip
INPUT_SEED = 7


def _pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.detach().flatten().double(), b.detach().flatten().double()]))[0, 1])


def _inputs(hidden):
    """x_{k-1} and h_k [1, 1, SEQ, hidden] bf16. Hidden states share a dominant direction across tokens, as real
    activations do (see synthetic_gates), so the inputs carry a common component plus per-token noise."""
    g = torch.Generator().manual_seed(INPUT_SEED)
    common = torch.randn(hidden, generator=g)
    x = 0.5 * common + torch.randn(1, 1, SEQ, hidden, generator=g)
    h = x + 0.5 * torch.randn(1, 1, SEQ, hidden, generator=g)
    return x.to(torch.bfloat16), h.to(torch.bfloat16)


def _to_device(mesh_device, t):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(-2, -1)),
    )


def _to_host(mesh_device, t):
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    ).float()


def _beta_to_host(mesh_device, t, width, tp):
    """beta as [1, 1, SEQ, width]: a width-1 beta is replicated over TP, so check the copies agree and keep one."""
    host = _to_host(mesh_device, t)
    if width == 1:
        copies = host.reshape(1, 1, SEQ, tp, -1)[..., 0]
        assert torch.equal(copies, copies[..., :1].expand_as(copies)), "TP copies of the scalar beta differ"
        return copies[..., :1]
    return host


@pytest.mark.parametrize("width", ["scalar", "per_channel"])
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
def test_glm54_damp_gate_matches_reference(mesh_device, device_params, width):
    sp_axis, tp_axis = 0, 1
    tp = mesh_device.shape[tp_axis]
    config = glm_5_4_hf_config()
    hidden = config.hidden_size
    gate = synthetic_gates(hidden, 1, seed=GLM54_SYNTHETIC_SEED, per_channel=width == "per_channel")[0]
    w = gate["proj_weight"].shape[0]

    tt_gate = TtDampGate(
        mesh_device,
        gate,
        eps=config.rms_norm_eps,
        tp_axis=tp_axis,
        num_links=2,
        topology=per_axis_topology(device_params["fabric_config"])[tp_axis],
    )
    assert tt_gate.width == w
    x, h = _inputs(hidden)
    tt_x, tt_h = _to_device(mesh_device, x), _to_device(mesh_device, h)

    beta_dev = _beta_to_host(mesh_device, tt_gate.beta(tt_x), w, tp)
    out_dev = _to_host(mesh_device, tt_gate(tt_x, tt_h))
    out_again = _to_host(mesh_device, tt_gate(tt_x, tt_h))
    assert torch.equal(out_dev, out_again), "two eager gate runs differ"

    def ref(*, x_prev=x, norm_weight=gate["norm_weight"], proj_bias=gate["proj_bias"]):
        return damp_gate_reference(
            x_prev,
            h,
            norm_weight=norm_weight,
            proj_weight=gate["proj_weight"],
            proj_bias=proj_bias,
            eps=config.rms_norm_eps,
        )

    def verdict(out_ref, beta_ref):
        """(scores, beta max abs error, failures) of the device against one reference result."""
        scores = {
            "beta": _pcc(beta_dev, beta_ref),
            "x_k": _pcc(out_dev, out_ref),
            "update": _pcc(out_dev - x.float(), out_ref - x.float()),
        }
        err = float((beta_dev - beta_ref).abs().max())
        failures = [f"{k} PCC {v:.6f} < {GATE_PCC}" for k, v in scores.items() if v < GATE_PCC]
        if err > BETA_MAX_ABS_ERR:
            failures.append(f"beta max abs err {err:.3e} > {BETA_MAX_ABS_ERR}")
        return scores, err, failures

    out_ref, beta_ref = ref()
    scores, beta_err, failures = verdict(out_ref, beta_ref)
    logger.info(
        f"{width} gate (width {w}): PCC {scores}; beta max abs err {beta_err:.2e}; "
        f"beta range {float(beta_ref.min()):.3f}..{float(beta_ref.max()):.3f}"
    )
    # Negative controls: the same checks against a faulty gate must fail (they say nothing otherwise).
    # A1c fault: beta from the pass output, the update still x_{k-1} + beta * (h - x_{k-1}).
    beta_from_h = ref(x_prev=h)[1]
    faults = {
        "beta reads pass output (A1c)": (x.float() + beta_from_h * (h.float() - x.float()), beta_from_h),
        "no norm gain (A1e)": ref(norm_weight=torch.ones_like(gate["norm_weight"])),
        "no bias (A1d)": ref(proj_bias=torch.zeros_like(gate["proj_bias"])),
    }
    for name, fault in faults.items():
        f_scores, f_err, f_failures = verdict(*fault)
        logger.info(f"{width} gate control '{name}': PCC {f_scores}; beta max abs err {f_err:.2e}")
        if not f_failures:
            failures.append(f"control '{name}' passes the checks: they cannot see it")

    # Traced: capture the gate on persistent inputs and replay it twice.
    with TracedStep(mesh_device, lambda: tt_gate(tt_x, tt_h)) as traced:
        traced.capture()
        replays = []
        for _ in range(2):
            traced.replay()
            replays.append(_to_host(mesh_device, traced.output))
    if not all(torch.equal(r, out_dev) for r in replays):
        failures.append("traced replay differs from eager")
    assert not failures, failures
