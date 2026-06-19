# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Side-by-side trace-captured perf: MoE1D vs the GPT-OSS reference MoE MLP.

The GPT-OSS sibling of test_moe_1d_vs_gemma4_perf.py: a *direct* head-to-head timing of the generic
MoE1D module against the module it was (jointly with Gemma4) extracted from —
`models/demos/gpt_oss/tt/mlp.py::MLP` (TopKRouter + Experts).

Fairness (the whole point):
  - BOTH modules are built from the **same** torch weights (one `_make_weights()` draw, re-shaped into
    each module's expected layout — GPT-OSS interleaves gate/up as even/odd columns of a fused
    `gate_up_proj`, MoE1D keeps them separate) and fed the **same** input.
  - **Matched config**: same E/H/I/top_k, bfloat4_b expert weights (the GPT-OSS reference hardcodes
    bf4 — see mlp.py), bfloat16 router, bfloat8_b sparse_matmul, topk->softmax routing with router
    bias, SwiGLU-with-clamp (limit=7.0, alpha=1.702) experts with gate/up/down bias. Same op
    sequence on both sides, so any delta is module overhead, not workload.
  - **Identical measurement harness**: per-module warm-up -> trace capture -> signposted replay loop,
    median + range over NUM_ITERS. Host dispatch is removed from both numbers.

Parallelization: the GPT-OSS `MLP` defaults to throughput (expert-parallel) experts on a multi-row
mesh; we force `use_throughput_experts=False` so the reference runs the **tensor-parallel** `Experts`
path — the same column-shard-intermediate + all-reduce strategy MoE1D uses — keeping the (1,8)
comparison apples-to-apples (module overhead, not TP-vs-EP). The throughput/EP path is the real
decode path on a 2D (rows>1) mesh and is a separate benchmark.

Single device (1,1): no collectives, so the reported time is pure compute for both.

Run via the device queue, e.g.:
    pytest models/common/tests/modules/moe/test_moe_1d_vs_gptoss_perf.py -rA -s
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


# Shape: GPT-OSS-20B (H=2880, I=2880, E=32). Unlike the Gemma4 benchmark there is no "tiny" point —
# the GPT-OSS reference's GPTOSSProgramConfig hardcodes core grids ((3,4)/(5,6)) tuned for production
# dims, so toy shapes produce a non-rectangular work grid (num_cores_with_work != mcast receivers) and
# the reference sparse_matmul TT_FATALs. The real shape is therefore the only meaningful comparison.
SHAPES = {
    "realish": dict(H=2880, I=2880, E=32),
}
TOP_K = 4  # GPT-OSS num_experts_per_tok
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
EXPERT_DTYPE = ttnn.bfloat4_b  # the GPT-OSS reference hardcodes bf4 experts (mlp.py)
NUM_ITERS = 30


def _make_weights(tp, shape):
    """One canonical draw of torch weights, shared by both modules (re-shaped per their layouts).

    On TP > 1, pre-pad the intermediate dim so the per-device slice is tile(32)-aligned (MoE1D requires
    the caller to do this; the GPT-OSS ref pads internally, so feeding it the same padded weights is a
    no-op pad on its side). E.g. I=2880 / tp=8 = 360 -> pad to 384/device -> padded I = 3072.
    """
    H, I, E = shape["H"], shape["I"], shape["E"]
    torch.manual_seed(0)
    gate_w = torch.randn(E, H, I) * 0.1  # [E, H, I]
    up_w = torch.randn(E, H, I) * 0.1  # [E, H, I]
    down_w = torch.randn(E, I, H) * 0.1  # [E, I, H]
    router_eh = torch.randn(E, H)  # [E, H] (HF Linear weight: out=E, in=H); peaked -> stable top-k
    router_b = torch.randn(E) * 0.1  # [E]
    gate_b = torch.randn(E, I) * 0.05  # [E, I]
    up_b = torch.randn(E, I) * 0.05  # [E, I]
    down_b = torch.randn(E, H) * 0.05  # [E, H]

    if tp > 1:
        per_device = I // tp
        padded_per_device = ((per_device + 31) // 32) * 32
        pad = padded_per_device * tp - I
        if pad > 0:
            gate_w = torch.nn.functional.pad(gate_w, (0, pad))  # [E, H, I+pad]
            up_w = torch.nn.functional.pad(up_w, (0, pad))
            down_w = torch.nn.functional.pad(down_w, (0, 0, 0, pad))  # pad dim -2 -> [E, I+pad, H]
            gate_b = torch.nn.functional.pad(gate_b, (0, pad))  # [E, I+pad]
            up_b = torch.nn.functional.pad(up_b, (0, pad))

    return gate_w, up_w, down_w, router_eh, router_b, gate_b, up_b, down_b


def _build_moe1d(mesh_device, shape, gate_w, up_w, down_w, router_eh, router_b, gate_b, up_b, down_b):
    """Generic MoE1D, GPT-OSS-matched config. On >1 device, mesh_device drives col/row-parallel + all-reduce."""
    H, E = shape["H"], shape["E"]

    def lz(src, dt=ttnn.bfloat16):
        return LazyWeight(source=src, dtype=dt)

    cfg = MoE1DConfig(
        gate_proj=lz(gate_w.unsqueeze(0), EXPERT_DTYPE),  # [1, E, H, I]
        up_proj=lz(up_w.unsqueeze(0), EXPERT_DTYPE),  # [1, E, H, I]
        down_proj=lz(down_w.unsqueeze(0), EXPERT_DTYPE),  # [1, E, I, H]
        router_weight=lz(router_eh.t().contiguous().reshape(1, 1, H, E)),  # [1, 1, H, E] (in x out)
        router_bias=lz(router_b.reshape(1, 1, 1, E)),
        gate_bias=lz(gate_b.unsqueeze(0)),  # [1, E, I]
        up_bias=lz(up_b.unsqueeze(0)),  # [1, E, I]
        down_bias=lz(down_b.unsqueeze(0)),  # [1, E, H]
        top_k=TOP_K,
        routing_strategy=RoutingStrategy.TOPK_SOFTMAX,
        activation_strategy=ExpertActivation.SWIGLU_CLAMP,
        swiglu_limit=SWIGLU_LIMIT,
        swiglu_alpha=SWIGLU_ALPHA,
        expert_weight_dtype=EXPERT_DTYPE,
        sparse_matmul_dtype=ttnn.bfloat8_b,
        mesh_device=mesh_device,
    )
    return MoE1D.from_config(cfg)


def _build_gptoss(mesh_device, tp, shape, gate_w, up_w, down_w, router_eh, router_b, gate_b, up_b, down_b):
    """GPT-OSS reference MLP from the same weights. Lazy-imported so collection never touches demos."""
    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.ccl import CCLManager
    from models.demos.gpt_oss.tt.mlp import MLP

    H, E = shape["H"], shape["E"]
    intermediate_size = gate_w.shape[-1]  # the SAME (already tile-aligned) width we feed MoE1D

    # GPT-OSS fuses gate/up as INTERLEAVED even/odd columns of gate_up_proj (weights.py reads
    # [..., ::2] = gate, [..., 1::2] = up), so interleave our separate gate/up the same way to feed
    # both modules byte-identical weights.
    gate_up = torch.stack([gate_w, up_w], dim=-1).reshape(E, H, 2 * intermediate_size)  # [E, H, 2I]
    gate_up_bias = torch.stack([gate_b, up_b], dim=-1).reshape(E, 2 * intermediate_size)  # [E, 2I]
    state_dict = {
        "router.weight": router_eh.contiguous(),  # [E, H]
        "router.bias": router_b.contiguous(),  # [E]
        "experts.gate_up_proj": gate_up.contiguous(),  # [E, H, 2I]
        "experts.gate_up_proj_bias": gate_up_bias.contiguous(),  # [E, 2I]
        "experts.down_proj": down_w.contiguous(),  # [E, I, H]
        "experts.down_proj_bias": down_b.contiguous(),  # [E, H]
    }

    hf_config = SimpleNamespace(
        hidden_size=H,
        num_local_experts=E,
        num_experts_per_tok=TOP_K,
        intermediate_size=intermediate_size,
        swiglu_limit=SWIGLU_LIMIT,
        rms_norm_eps=1e-5,
    )

    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=tp), prefill=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    return MLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        dtype=EXPERT_DTYPE,
        mesh_config=mesh_config,
        use_throughput_experts=False,  # force the tensor-parallel Experts path (matches MoE1D's TP)
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
@pytest.mark.parametrize("shape_id", ["realish"])
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 1),
        ("prefill", 128),
        ("prefill", 512),
        ("prefill", 1024),
        ("prefill", 2048),
    ],
    ids=["decode", "prefill-128", "prefill-512", "prefill-1024", "prefill-2048"],
)
def test_moe_1d_vs_gptoss_perf(ttnn_mesh_device: ttnn.MeshDevice, shape_id, mode, seq_len):
    mesh_device = ttnn_mesh_device
    ttnn.SetDefaultDevice(mesh_device)
    tp = mesh_device.get_num_devices()  # 1D mesh (1, N) -> TP = N
    shape = SHAPES[shape_id]
    H = shape["H"]

    weights = _make_weights(tp, shape)
    moe1d = _build_moe1d(mesh_device, shape, *weights)
    ref = _build_gptoss(mesh_device, tp, shape, *weights)

    # GPT-OSS uses ONE hidden state for both router and experts (unlike Gemma4's split inputs); feed
    # the same draw to both modules. Separate device tensors so neither trace aliases the other's.
    torch.manual_seed(1)
    x = torch.randn(1, 1, seq_len, H) * 0.5
    m_in = _to_dev(x, mesh_device)
    r_in = _to_dev(x, mesh_device)
    is_decode = mode == "decode"

    # Which module(s) to measure. Default both; set MOE_PERF_MODULES=moe1d|gptoss to isolate one.
    which = os.getenv("MOE_PERF_MODULES", "both")
    m_res = r_res = None
    if which in ("both", "moe1d"):
        logger.info(f"[{mode} (1,{tp})] measuring MoE1D ...")
        m_res = _trace_measure(mesh_device, lambda: moe1d.forward(m_in, m_in, mode), "moe1d-start", "moe1d-stop")
    if which in ("both", "gptoss"):
        logger.info(f"[{mode} (1,{tp})] measuring GPT-OSS reference ...")
        r_res = _trace_measure(mesh_device, lambda: ref(r_in, is_decode), "gptoss-start", "gptoss-stop")

    ttnn.SetDefaultDevice(None)

    collectives = "compute-only; no collectives" if tp == 1 else f"compute + TP={tp} all-reduce"
    tag = f"{mode} seq={seq_len} (1,{tp}) shape={shape_id} experts=bf4"
    lines = [f"MoE1D-vs-GPTOSS PERF [{tag}] ({collectives}):"]
    if m_res:
        lines.append(f"    MoE1D   median={m_res[0]:7.1f}us range=[{m_res[1]:.1f},{m_res[2]:.1f}]us")
    if r_res:
        lines.append(f"    GPT-OSS median={r_res[0]:7.1f}us range=[{r_res[1]:.1f},{r_res[2]:.1f}]us")
    if m_res and r_res:
        lines.append(f"    ratio MoE1D/GPT-OSS = {m_res[0] / r_res[0]:.3f}x  (over {NUM_ITERS} iters each)")
    logger.info("\n".join(lines))
    assert (m_res is None or m_res[0] > 0) and (r_res is None or r_res[0] > 0)
