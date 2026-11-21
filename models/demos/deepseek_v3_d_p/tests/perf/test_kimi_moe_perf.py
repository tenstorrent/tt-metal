# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 / Kimi-K3.1 MoE perf gate: host time of one MoE layer forward on the MoE block, as the models run it.

One MoE layer through the model adapter (TtMoEBlockFfn: MoE head -> sparse dispatch -> flat experts + combine -> MoE
tail; K3's seeded random weights, block_cases.MODELS) at each model's stage and chunk:

* kimi_k3-8x4-5120: K3 on one 8x4 Galaxy (the runners' *_8x4 configs), 5,120-token chunk - the shape the TtMoe-era
  gate measured.
* kimi_k3_1-4x4-sp1-2560: K3.1's production stage, one tray half (SP=4 on mesh axis 1, TP=4 on axis 0), 2,560-token
  chunk, the mesh opened with K3.1's L1_SMALL_SIZE. Needs the Galaxy narrowed to the tray half: run it through
  tests/run_4x4_stage_tests.sh (unnarrowed, the system mesh is 8x4 and the case skips; the 8x4 case skips the other way).

The gate is host time as tests/measure measures it (the Kimi perf tests' gate): untraced fastest of 10 after 10
warm-up, traced one replay, the CPU clock held; each two-sided against PERF_TARGETS_MS ± MARGIN[mode] (a case without
targets only reports), and the traced output must match the untraced one. Per-op device time of the same calls is
test_moe_block_models.py::test_moe_block_perf (BLOCK_US), which the CI leg runs next to this.

    python3 -m pytest models/demos/deepseek_v3_d_p/tests/perf/test_kimi_moe_perf.py -k kimi_k3-8x4
    models/demos/deepseek_v3_d_p/tests/run_4x4_stage_tests.sh --half top -- \\
        models/demos/deepseek_v3_d_p/tests/perf/test_kimi_moe_perf.py -k kimi_k3_1-4x4-sp1
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole, skip_with_llk_assert, skip_with_watcher
from models.demos.deepseek_v3_d_p.tests.measure.gate import MARGIN, band_failure, gate
from models.demos.deepseek_v3_d_p.tests.measure.timing import time_untraced
from models.demos.deepseek_v3_d_p.tests.measure.trace import TracedStep, time_traced
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_perf_metrics
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import require_high_power
from tt_py_test_utils_common.perf.cpu_clock import hold_full_cpu_clock

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "tests/ttnn/unit_tests/operations/experimental/moe_block"))
from block_cases import MESH_SP_AXIS, MESHES, MODELS  # noqa: E402

MODES = ("untraced", "traced")
# Traced output must match the untraced output of the same forward.
TRACE_PCC = 0.9999


@dataclass(frozen=True)
class MoEPerfCase:
    model: str  # block_cases.MODELS key
    mesh: str  # block_cases.MESHES key
    chunk: int  # tokens a forward (the whole mesh's)
    # Host ms of one MoE layer forward through a device sync on a 14 kW CI Galaxy (bh_sc1_high_power), as the CI leg
    # runs it: (untraced fastest of 10, traced one replay). None: measured and reported, not gated.
    targets_ms: Optional[tuple] = None

    @property
    def id(self) -> str:
        return f"{self.model}-{self.mesh}-{self.chunk}"


# Targets: CI runs 44900949382 (bh-glx-120-d09u02) and 44905230066 (d06u02), 14 kW, 2026-11-21; untraced the fastest
# run, traced the mean of the runs (as test_sparse_mla_perf's).
CASES = [
    MoEPerfCase("kimi_k3", "8x4", 5120, targets_ms=(2.38, 2.27)),
    MoEPerfCase("kimi_k3_1", "4x4-sp1", 2560, targets_ms=(3.63, 3.54)),
]
PERF_TARGETS_MS = {(c.id, mode): ms for c in CASES if c.targets_ms is not None for mode, ms in zip(MODES, c.targets_ms)}


def _params():
    out = []
    for c in CASES:
        shape, make_params, topology = MESHES[c.mesh]
        out.append(
            pytest.param(
                c,
                shape,
                make_params(**MODELS[c.model].mesh_params()),
                marks=pytest.mark.requires_mesh_topology(mesh_shape=shape, topology=topology),
                id=c.id,
            )
        )
    return out


def _device_tensors(tensor):
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)]


def _setup(case: MoEPerfCase, mesh_device):
    """One MoE layer on the block (weights made on its first call) and its input on the device, as
    test_moe_block_models sets a random-weight case up."""
    from models.demos.deepseek_v3_d_p.tt.moe_block.moe_block_ffn import MoEBlockFfnOptions, TtMoEBlockFfn

    m = MODELS[case.model]
    R, C = tuple(mesh_device.shape)
    H = m.cfg.EMB_SIZE
    sp_axis = MESH_SP_AXIS.get(case.mesh, 0)
    ffn = TtMoEBlockFfn(
        mesh_device,
        m.cfg,
        layer_idx=m.layer,
        state_dict=m.weights(R * C),
        options=MoEBlockFfnOptions(sp_axis=sp_axis),
    )
    x = (torch.randn(case.chunk, H, generator=torch.Generator().manual_seed(7)) * m.x_scale).to(torch.bfloat16)
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, case.chunk, H),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # tokens (dim 2) over the SP axis, hidden (dim 3) over TP
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, 3) if sp_axis == 0 else (3, 2)),
    )
    return ffn, x_tt


@skip_with_llk_assert("LLK assertions perturb kernel timing")
@skip_with_watcher("Watcher perturbs kernel timing")
@pytest.mark.skipif(not is_blackhole(), reason="Blackhole only")
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("case, mesh_device, device_params", _params(), indirect=["mesh_device", "device_params"])
def test_kimi_moe_perf(request, case, mesh_device, device_params):
    label = f"{case.id} MoE block"
    require_high_power(f"{label} perf")
    ffn, x_tt = _setup(case, mesh_device)

    def step():
        return ffn(x_tt)

    clock = hold_full_cpu_clock()
    request.addfinalizer(clock.restore)
    print(f"{label} host: {clock.status}")

    samples = {}
    samples["untraced"], out = time_untraced(step, lambda: ttnn.synchronize_device(mesh_device))
    eager = _device_tensors(out)
    ttnn.deallocate(out)
    with TracedStep(mesh_device, step) as traced:
        traced.capture()
        samples["traced"] = time_traced(traced)
        replayed = _device_tensors(traced.output)
    trace_pcc = min(comp_pcc(a, b, TRACE_PCC)[1] for a, b in zip(eager, replayed))

    shape = tuple(mesh_device.shape)
    results = {}
    for mode in MODES:
        measured, target = min(samples[mode]), PERF_TARGETS_MS.get((case.id, mode))
        results[mode] = (samples[mode], measured, target, MARGIN[mode])
        emit_perf_metrics(
            f"{case.id.replace('-', '_')}_moe_block_{mode}",
            f"{label} ({mode})",
            {
                "model": case.model,
                "mesh": f"{shape[0]}x{shape[1]}",
                "sp_axis": MESH_SP_AXIS.get(case.mesh, 0),
                "chunk_tokens": case.chunk,
                "fabric": device_params["fabric_config"].name,
                "weights": "random",
                "mode": mode,
                "samples_ms": samples[mode],
                "measured_ms": measured,
                "target_ms": target,
                "margin_pct": MARGIN[mode] * 100.0,
                "within_band": target is not None and band_failure(mode, measured, target, MARGIN[mode]) is None,
            },
        )
    gate(label, results, trace_pcc)
