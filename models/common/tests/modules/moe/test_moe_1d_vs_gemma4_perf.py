# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Side-by-side trace-captured perf: MoE1D vs the Gemma4 reference MoEBlock.

This is the documented follow-up to test_moe_1d_perf.py's self-baseline (see that file's docstring):
a *direct* head-to-head timing of the generic MoE1D module against the module it was extracted from —
`models/demos/gemma4/tt/moe.py::MoEBlock` (router + routed experts).

Fairness (the whole point):
  - BOTH modules are built from the **same** torch weights (one `_make_weights()` draw, re-shaped into
    each module's expected layout) and fed the **same** inputs.
  - **Matched config**: same E/H/I/top_k, expert weights bfloat8_b, router bfloat16, sparse_matmul
    bfloat16, in0_block_w=1 (MoE1D default == reference default), GeGLU, softmax->topk->sum-norm
    routing, router rms-norm + scale + hidden**-0.5 + per-expert-scale. This is the same op sequence
    on both sides, so any delta is module overhead, not workload.
  - **Identical measurement harness**: per-module warm-up -> trace capture -> signposted replay loop,
    median + range over NUM_ITERS. Host dispatch is removed from both numbers.

Single device (1,1): no collectives, so the reported time is pure compute (matmul + eltwise) for both.
The multi-device TP all-reduce split is deferred with the multi-device bringup.

Run via the device queue, e.g.:
    pytest models/common/tests/modules/moe/test_moe_1d_vs_gemma4_perf.py -rA -s
For the per-op compute breakdown, run under the device profiler (TT_METAL_DEVICE_PROFILER=1) and read
the ops log between the "moe1d-*"/"gemma4-*" signposts.
"""

import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.moe.moe_1d import ExpertActivation, MoE1D, MoE1DConfig, RoutingStrategy

try:
    from tracy import signpost
except ImportError:  # tracy not always importable outside the profiler build

    def signpost(_msg):  # noqa: D103
        pass


# Representative dims — match test_moe_1d_perf.py so the MoE1D side reproduces the established
# self-baseline numbers and the reference is measured at the exact same point.
H = 256
I = 128
E = 8
TOP_K = 2
NUM_ITERS = 30
ROUTER_PRENORM_EPS = 1e-6


def _make_weights(tp):
    """One canonical draw of torch weights, shared by both modules (re-shaped per their layouts).

    On TP > 1, pre-pad the intermediate dim so the per-device slice is tile(32)-aligned (MoE1D requires
    the caller to do this; the Gemma4 ref pads internally, so feeding it the same padded weights is a
    no-op pad on its side). E.g. I=128 / tp=8 = 16 -> pad to 32/device -> padded I = 256.
    """
    torch.manual_seed(0)
    gate_w = torch.randn(E, H, I) * 0.1  # [E, H, I]
    up_w = torch.randn(E, H, I) * 0.1  # [E, H, I]
    down_w = torch.randn(E, I, H) * 0.1  # [E, I, H]
    router_he = torch.randn(H, E)  # [H, E] (ttnn linear in x out); peaked (N(0,1)) -> stable top-k
    scale_h = torch.randn(H) * 0.1 + 1.0  # router pre-linear scale, [H]
    per_expert_scale = torch.rand(E) * 0.5 + 0.75  # [E]

    if tp > 1:
        per_device = I // tp
        padded_per_device = ((per_device + 31) // 32) * 32
        pad = padded_per_device * tp - I
        if pad > 0:
            gate_w = torch.nn.functional.pad(gate_w, (0, pad))  # [E, H, I+pad]
            up_w = torch.nn.functional.pad(up_w, (0, pad))
            down_w = torch.nn.functional.pad(down_w, (0, 0, 0, pad))  # pad dim -2 -> [E, I+pad, H]

    return gate_w, up_w, down_w, router_he, scale_h, per_expert_scale


def _build_moe1d(mesh_device, gate_w, up_w, down_w, router_he, scale_h, per_expert_scale):
    """Generic MoE1D, Gemma4-matched config. On >1 device, mesh_device drives col/row-parallel + all-reduce."""

    def lz(src, dt=ttnn.bfloat8_b):
        return LazyWeight(source=src, dtype=dt)

    cfg = MoE1DConfig(
        gate_proj=lz(gate_w.unsqueeze(0)),  # [1, E, H, I]
        up_proj=lz(up_w.unsqueeze(0)),  # [1, E, H, I]
        down_proj=lz(down_w.unsqueeze(0)),  # [1, E, I, H]
        router_weight=lz(router_he.reshape(1, 1, H, E), ttnn.bfloat16),
        top_k=TOP_K,
        routing_strategy=RoutingStrategy.SOFTMAX_TOPK_SUMNORM,
        activation_strategy=ExpertActivation.GEGLU,
        router_prenorm_eps=ROUTER_PRENORM_EPS,
        router_input_scalar=H**-0.5,
        router_scale=lz(scale_h.reshape(1, 1, 1, H), ttnn.bfloat16),
        per_expert_scale=lz(per_expert_scale.reshape(1, 1, 1, E), ttnn.bfloat16),
        mesh_device=mesh_device,
    )
    return MoE1D.from_config(cfg)


def _build_gemma4(mesh_device, tp, gate_w, up_w, down_w, router_he, scale_h, per_expert_scale):
    """Gemma4 reference MoEBlock from the same weights. Lazy-imported so collection never touches demos."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.moe import MoEBlock

    # Feed the ref the SAME (already tile-aligned) intermediate width we feed MoE1D. weights.py then
    # slices the fused gate/up at this width and its own per-device pad is a no-op — so both modules
    # run byte-identical weights (zeros in any padded tail produce zero contributions on both sides).
    intermediate_size = gate_w.shape[-1]
    hf_config = SimpleNamespace(
        hidden_size=H,
        num_experts=E,
        top_k_experts=TOP_K,
        moe_intermediate_size=intermediate_size,
        rms_norm_eps=ROUTER_PRENORM_EPS,
    )

    # HF gate_up_proj is fused [E, 2*I, H] (gate then up, contiguous). weights.py slices + transposes
    # it back to [1,E,H,I], so feed each half as [E,I,H] = our [E,H,I] transposed.
    gate_up = torch.cat([gate_w.transpose(-2, -1), up_w.transpose(-2, -1)], dim=1)  # [E, 2*I, H]
    state_dict = {
        # Gemma4Router transposes proj.weight (-2,-1) -> [H,E]; HF stores Linear weight as [out, in] = [E, H].
        "router.proj.weight": router_he.transpose(0, 1).contiguous(),  # [E, H]
        "router.scale": scale_h.clone(),  # [H]
        "router.per_expert_scale": per_expert_scale.clone(),  # [E]
        "experts.gate_up_proj": gate_up.contiguous(),  # [E, 2*I, H]
        # weights.py transposes down (-2,-1) -> [1,E,I,H]; feed [E,H,I] = our [E,I,H] transposed.
        "experts.down_proj": down_w.transpose(-2, -1).contiguous(),  # [E, H, I]
    }

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    return MoEBlock(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        dtype=ttnn.bfloat8_b,  # matched: bf8 expert weights
        router_dtype=ttnn.bfloat16,
    )


def _to_dev(t, mesh_device):
    # On a multi-device mesh the activation is replicated across the TP group (each device holds the
    # full hidden vector and computes its expert-intermediate shard, recombined by the all-reduce).
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if mesh_device.get_num_devices() > 1 else None
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _trace_measure(mesh_device, run_fn, start_msg, stop_msg):
    """Warm-up (compile + lazy load) -> trace capture -> signposted replay loop. Returns (median, lo, hi) us."""
    # 1. Warm-up / compile (also triggers MoE1D lazy weight load + prefill-sparsity cache).
    _ = run_fn()
    ttnn.synchronize_device(mesh_device)

    # 2. Capture the trace (host dispatch happens once here).
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    _ = run_fn()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # 3. Replay between signposts — measured (device-bound) region. Per-iter sync = per-forward sample.
    samples_us = []
    signpost(start_msg)
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        samples_us.append((time.perf_counter() - t0) * 1e6)
    signpost(stop_msg)

    ttnn.release_trace(mesh_device, trace_id)
    samples_us.sort()
    return samples_us[len(samples_us) // 2], samples_us[0], samples_us[-1]


@pytest.mark.parametrize("ttnn_mesh_device", [(1, 1), (1, 8)], ids=["1x1", "1x8"], indirect=True)
@pytest.mark.parametrize("mode,seq_len", [("decode", 1), ("prefill", 128)], ids=["decode", "prefill-128"])
def test_moe_1d_vs_gemma4_perf(ttnn_mesh_device: ttnn.MeshDevice, mode, seq_len):
    mesh_device = ttnn_mesh_device
    ttnn.SetDefaultDevice(mesh_device)
    tp = mesh_device.get_num_devices()  # 1D mesh (1, N) -> TP = N

    weights = _make_weights(tp)
    moe1d = _build_moe1d(mesh_device, *weights)
    ref = _build_gemma4(mesh_device, tp, *weights)

    # Same inputs to both; separate device tensors so neither module's trace aliases the other's.
    torch.manual_seed(1)
    ri = torch.randn(1, 1, seq_len, H) * 0.5
    ei = torch.randn(1, 1, seq_len, H) * 0.5
    m_ri, m_ei = _to_dev(ri, mesh_device), _to_dev(ei, mesh_device)
    r_ri, r_ei = _to_dev(ri, mesh_device), _to_dev(ei, mesh_device)

    # Which module(s) to measure. Default both; set MOE_PERF_MODULES=moe1d|gemma4 to isolate one
    # (used to bisect a multi-device hang: run the trusted reference alone to tell box from code).
    which = os.getenv("MOE_PERF_MODULES", "both")
    m_res = r_res = None
    if which in ("both", "moe1d"):
        logger.info(f"[{mode} (1,{tp})] measuring MoE1D ...")
        m_res = _trace_measure(mesh_device, lambda: moe1d.forward(m_ri, m_ei, mode), "moe1d-start", "moe1d-stop")
    if which in ("both", "gemma4"):
        logger.info(f"[{mode} (1,{tp})] measuring Gemma4 reference ...")
        r_res = _trace_measure(mesh_device, lambda: ref(r_ri, r_ei), "gemma4-start", "gemma4-stop")

    ttnn.SetDefaultDevice(None)

    # On tp>1 the time includes a TP all-reduce. NOTE: MoE1D auto-selects a Ring all-reduce on >=8
    # devices while the Gemma4 ref hardcodes Topology.Linear — a real implementation difference the
    # collective half of any tp>1 delta reflects (not a workload mismatch).
    collectives = "compute-only; no collectives" if tp == 1 else f"compute + TP={tp} all-reduce"
    lines = [f"MoE1D-vs-Gemma4 PERF [{mode} seq={seq_len} (1,{tp})] ({collectives}):"]
    if m_res:
        lines.append(f"    MoE1D   median={m_res[0]:7.1f}us range=[{m_res[1]:.1f},{m_res[2]:.1f}]us")
    if r_res:
        lines.append(f"    Gemma4  median={r_res[0]:7.1f}us range=[{r_res[1]:.1f},{r_res[2]:.1f}]us")
    if m_res and r_res:
        lines.append(f"    ratio MoE1D/Gemma4 = {m_res[0] / r_res[0]:.3f}x  (over {NUM_ITERS} iters each)")
    logger.info("\n".join(lines))
    assert (m_res is None or m_res[0] > 0) and (r_res is None or r_res[0] > 0)
