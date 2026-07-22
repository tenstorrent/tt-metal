# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# KDA prefill perf: realtime profiler (NOT tracy), before (single device) vs after (TP (2,4)).
# Per-forward device-kernel time grouped into matmul / collective / other, vs ROOFLINE.md targets.
# KDA note: warm == long (fixed-size state -> per-chunk cost is cache-depth-independent); cold = N chunks.

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.kimi_delta_attention.torch_functional import KimiDeltaAttentionRef
from models.experimental.kimi_delta_attention.tt.ttnn_kda import TtKimiDeltaAttention
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

torch.manual_seed(3)

_F2D = {
    "fabric_config": ttnn.FabricConfig.FABRIC_2D,
    "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
    "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
    "l1_small_size": 24576,  # native conv1d (prefill short-conv) needs L1 scratch
}


def _group(sources):
    s = " ".join(sources).lower()
    if any(w in s for w in ("all_gather", "reduce_scatter", "ccl", "fabric", "all_reduce")):
        return "collective"
    if any(w in s for w in ("matmul", "bmm")):
        return "matmul"
    return "other"


def _profile(mesh, run_fn):
    """One measured forward -> {group: device_ns} (max across chips per program, summed by group)."""
    ttnn.synchronize_device(mesh)
    _, records = profile_realtime_program(mesh, run_fn, collect_all=True)
    per_prog = {}  # runtime_id -> (max_ns, sources)
    for r in records:
        rid = r["runtime_id"]
        if not rid:
            continue
        d = float(r["duration_ns"])
        if rid not in per_prog or d > per_prog[rid][0]:
            per_prog[rid] = (d, r.get("kernel_sources", ()))
    groups = {"matmul": 0.0, "collective": 0.0, "other": 0.0}
    for ns, src in per_prog.values():
        groups[_group(src)] += ns
    return groups


def _mk_ref():
    # real KDA head dims (num_heads=32/TP4 -> 8 heads/chip), smaller hidden for a fast dev loop
    return KimiDeltaAttentionRef(
        hidden_size=2304, head_dim=128, num_heads=32, num_v_heads=32, conv_size=4, mode="recurrent"
    ).eval()


def _run(mesh, layer, T, tag):
    x = torch.randn(1, T, 2304)
    layer.forward(x)  # warm up (JIT compile), discarded
    g = _profile(mesh, lambda: layer.forward(x))
    tot = sum(g.values()) / 1e3
    logger.info(
        f"[kda_perf {tag}] T={T} device_us: matmul={g['matmul']/1e3:.1f} "
        f"collective={g['collective']/1e3:.1f} other={g['other']/1e3:.1f}  TOTAL={tot:.1f}us"
    )
    return g


@pytest.mark.parametrize("device_params", [_F2D], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 1)], indirect=True)
@pytest.mark.parametrize("T", [256, 640])
def test_kda_perf_before(mesh_device, T):
    """Before distribution: TP=1 (8,1) — all 32 heads per chip, no head-shard, no TP all-reduce."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("realtime profiler not active")
    _run(mesh_device, TtKimiDeltaAttention(_mk_ref(), mesh_device), T, "before(8,1)TP1")


@pytest.mark.parametrize("device_params", [_F2D], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("T", [256, 640])
def test_kda_perf_after(mesh_device, T):
    """After distribution: TP=4 head-shard (8 heads/chip) on LoudBox (2,4)."""
    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail("realtime profiler not active")
    _run(mesh_device, TtKimiDeltaAttention(_mk_ref(), mesh_device), T, "after(2,4)TP4")
