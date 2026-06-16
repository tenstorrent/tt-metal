# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-side perf test for the common ``TTMoEGate`` on a 4×8 mesh, per model config — TRACE-based.

Captures ``TTMoEGate.forward`` into a trace, then times ``execute_trace`` inside the signpost window, so the
harness (``perf_tt_moe_gate.py``) can isolate the device-kernel time. Two things matter here:

  • TRACE (not an eager loop): eager per-op dispatch lets ops overlap / leaves dispatch gaps, so the profiler
    under-measures per-op device-kernel time vs a recorded trace — the trace flow gives the real latency.
  • 4×8 mesh (mirrors test_tt_moe_gate.py): TTMoEGate is a PER-DEVICE gate (no cross-chip comms), so its
    weight + buffers replicate to every chip and each chip runs the same one-token-per-core routing; the
    input is replicated too. Per-device latency = one chip's time (read back below from one chip).

Parametrized over every ``configs/*.yaml``; the parent drives ONE model per subprocess (``-k <model>``) so
each signpost window measures a single model. The only skips mirror ``TTMoEGate.__init__``'s guards —
n_group ∉ {1, 8}, or n_group=8 not at exactly 256 experts select-8. Measurement only (no PCC).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.modules.moe.tt_moe_gate import TTMoEGate
from models.common.modules.moe.tt_moe_gate_config import TTMoEGateConfig
from models.perf.benchmarking_utils import BenchmarkProfiler

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "modules/moe/configs"
CONFIG_PATHS = sorted(CONFIGS_DIR.glob("*.yaml"))
assert CONFIG_PATHS, f"no YAML configs found in {CONFIGS_DIR}"


def _config_id(path: Path) -> str:
    return path.stem


@pytest.mark.parametrize(
    "device_params",
    # trace_region_size is REQUIRED for trace capture/execute. The gate replicates per chip (no CCL/fabric).
    [{"trace_region_size": 700000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    # 4×8 = 32-chip mesh (TG/Galaxy), same as test_tt_moe_gate.py. Auto-skips if fewer chips are available.
    "mesh_device",
    [pytest.param((1, 4), id="1x4")],
    indirect=True,
)
@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=_config_id)
@pytest.mark.parametrize("warmup_iters, num_iters", [(5, 10)])
def test_tt_moe_gate_perf(mesh_device, config_path: Path, warmup_iters, num_iters):
    """Trace-based device-perf for TTMoEGate on a single model gate config (4×8 mesh, replicated)."""
    gate_config = TTMoEGateConfig.from_yaml(config_path.read_text())

    num_experts = gate_config.num_routed_experts
    k = gate_config.select_experts_k
    hidden = gate_config.hidden_size
    n_group = gate_config.n_group

    # Skip only what TTMoEGate genuinely can't build (mirrors its __init__ guards / test_tt_moe_gate.py):
    #   • n_group ∈ {1, 8} only.
    #   • n_group=8 (deepseek grouped op) is hardwired to 256 experts select-8.
    if n_group not in (1, 8):
        pytest.skip(f"only n_group 1 (generalized/ungrouped) / 8 (deepseek grouped) are wired; got n_group={n_group}")
    if n_group == 8 and (num_experts != 256 or k != 8):
        pytest.skip(f"n_group=8 (deepseek grouped op) is hardwired to 256 experts select-8; got N={num_experts}, k={k}")

    batch = gate_config.batch_per_device or 32  # PER-DEVICE batch (one token per core); replicated to every chip

    torch.manual_seed(42)
    gate_weight = ((2 * torch.rand((hidden, num_experts), dtype=torch.bfloat16)) - 1) * 0.1
    # TWO optional biases (present per config, see TTMoEGate):
    #   score-correction bias (deepseek/noaux_tc, config.score_correction_bias) → torch_gate_bias
    #   router LINEAR bias    (gpt-oss, config.router_bias)                      → torch_gate_proj_bias
    correction_bias = (2 * torch.rand((num_experts,)) - 1) if gate_config.score_correction_bias else None
    proj_bias = (2 * torch.rand((num_experts,)) - 1) if gate_config.router_bias else None

    # config-driven entry point (mirrors TTMoEDecode / test_tt_moe_gate.py). TTMoEGate's from_torch calls have
    # no mesh_mapper → the weight/buffers replicate to every chip.
    gate = TTMoEGate(
        mesh_device,
        gate_config,
        torch_gate_weight=gate_weight,
        torch_gate_bias=correction_bias,
        torch_gate_proj_bias=proj_bias,
    )
    # replicate the hidden states to every chip — each chip routes the same batch.
    tt_x = ttnn.from_torch(
        ((2 * torch.rand((batch, hidden), dtype=torch.bfloat16)) - 1).reshape(1, 1, batch, hidden),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run():
        return gate.forward(tt_x)

    # 1) compile (warm the program cache) — outside any trace.
    run()
    ttnn.synchronize_device(mesh_device)

    # 2) capture the warmup trace (warmup_iters forwards).
    trace_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(warmup_iters):
        run()
    ttnn.end_trace_capture(mesh_device, trace_warmup, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # 3) capture the main trace (num_iters forwards) — what the harness measures.
    trace_main = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(num_iters):
        run()
    ttnn.end_trace_capture(mesh_device, trace_main, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # host-side wall-clock markers (mirrors real_test_moe_gate.py). NOTE: execute_trace is non-blocking, so
    # these time the DISPATCH, not device execution — the real per-op device-kernel time is read by the
    # parent harness from the signpost-bracketed window. Kept only for flow parity.
    profiler = BenchmarkProfiler()

    # 4) execute the warmup trace OUTSIDE the signpost window.
    profiler.start("warmup")
    ttnn.execute_trace(mesh_device, trace_warmup, blocking=False)
    ttnn.release_trace(mesh_device, trace_warmup)
    profiler.end("warmup")

    # 5) execute the main trace INSIDE the signpost window (device-side signposts bracket the trace's ops
    #    even though execute_trace is non-blocking — the harness reads the per-op device-kernel time here).
    logger.info(
        f"[tt_moe_gate perf] {config_path.stem}: N={num_experts} k={k} hidden={hidden} batch={batch} "
        f"n_group={n_group} score_func={gate_config.score_func} mesh={tuple(mesh_device.shape)}, {num_iters} iters (trace)"
    )
    signpost("start")
    profiler.start("main")
    ttnn.execute_trace(mesh_device, trace_main, blocking=False)
    ttnn.release_trace(mesh_device, trace_main)
    profiler.end("main")
    signpost("stop")
    ttnn.synchronize_device(mesh_device)

    # One eager call + light sanity (indices in range); read back one chip (every chip computed the same).
    _, idx = run()
    ttnn.synchronize_device(mesh_device)
    di = ttnn.to_torch(idx, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape(-1, k)[:batch]
    di = di.to(torch.int32)
    assert int(di.min()) >= 0 and int(di.max()) < num_experts, f"indices out of range: {di}"


if __name__ == "__main__":
    pytest.main([__file__])
