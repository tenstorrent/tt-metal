# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Device-time perf harness for the TTTv2 ``MoE1D`` module (1D mesh topology: N150 1x1,
N300 1x2, T3K 1x8), tensor-parallel only.

A standalone, trace-captured, tracy-signposted microbenchmark that measures the device
kernel time of the MoE region (router + routed experts) and compares ``MoE1D`` against
the reference unit it was factored from (Gemma4 ``MoEBlock``), at matched
config/shapes/dtypes.

Design (orchestrator + worker, mirroring ``models/tt_dit/utils/sweep_mm_block_sizes.py``):

  - ``test_moe1d_perf`` (Stage 1) and ``test_moe1d_vs_gemma_perf`` (Stage 2) are
    *orchestrators*. They are NOT profiled and do NOT open a device. Each spawns the
    worker in a tracy-profiled subprocess via ``run_device_profiler``, then parses the
    resulting device-ops CSV and prints a device-µs table (median + range over N traced
    replays) plus a Matmul / collective split.
  - ``test_moe1d_perf_worker`` runs inside that profiled subprocess. It opens the mesh
    (with fabric + a trace region) via the shared ``ttnn_mesh_device`` fixture, builds
    the requested target (``moe1d`` or ``gemma``) with deterministic random weights, and
    runs the warmup → trace-capture → signposted-replay pattern. Configuration is passed
    through env vars (``MOE_PERF_*``) so the worker takes no pytest params beyond the
    mesh.

Trace replay collapses op-to-op host gaps, so the signposted window is steady-state
*device* time. The metric per replay is the sum of per-op ``DEVICE KERNEL DURATION``
across the window, averaged over devices.

Run (through the device MCP queue — never Bash; ttnn opens the cluster):

    source activate.sh && \
      pytest models/common/tests/modules/moe/test_moe_1d_perf.py \
      -k "moe1d_perf and 1x2 and decode and small" -v -s --timeout=2000

Pick exactly one (mesh, mode, shape) per run with ``-k`` to stay under the profiler
marker budget (~12k/core); MoE has many ops × num_experts. The ``small`` shape is sized
for Tracy; ``gemma_real`` (E=128) is realistic but heavy — shrink or expect overflow.
Results docs live under ``dev-tools/agents-context/tttv2-module-bringup/moe/profiling/``;
this committed source intentionally hardcodes no ``dev-tools`` path (artifacts land in the
default profiler dir).
"""

import os
import statistics

import pytest
from loguru import logger

import ttnn

# Signpost API with a no-op fallback so the module imports/collects outside tracy.
try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_a, **_k):
        pass


# ============================================================================
# Canonical perf cases
# ============================================================================
#
# Shapes: H=hidden, I=intermediate, E=num_experts, K=top_k.
#  - "small"      : Tracy-safe self-baseline. I divisible by 8 so 1x8 TP shards
#                   tile-aligned (no per-device padding divergence).
#  - "gemma_real" : Gemma4-31B MoE dims (hidden=2816, moe_intermediate=704,
#                   E=128, top_k=8). Realistic but marker-heavy under Tracy.
SHAPES = {
    "small": dict(H=256, I=512, E=8, K=2),
    "gemma_real": dict(H=2816, I=704, E=128, K=8),
    # GPT-OSS (Stage 3): dims chosen so the reference's fixed `GPTOSSProgramConfig`
    # core grids tile the matmuls — gate/up grid is 3×4=12 cores (needs per-device
    # I/32 a multiple of 12), down grid is 5×6=30 cores (needs H/32 a multiple of 30).
    # H=960 → 30 tiles; I=768 → 24 tiles (12/dev on 1x2). 1x8 would need I=3072
    # (per-device I/32=12) — too marker-heavy for Tracy, so Stage 3 runs 1x1/1x2.
    "gptoss": dict(H=960, I=768, E=8, K=2),
}

# Shapes used per stage (Stage 1/2 = Gemma family; Stage 3 = GPT-OSS family).
GEMMA_SHAPE_IDS = ["small", "gemma_real"]
GPTOSS_SHAPE_IDS = ["gptoss"]

# mesh id -> (rows, cols). This is a 1D TP mesh exposed as (1, N).
MESHES = {"1x1": (1, 1), "1x2": (1, 2), "1x8": (1, 8)}
# Stage 3 meshes: GPT-OSS's fixed expert core grids only tile the `gptoss` shape on
# 1x1 / 1x2 (1x8 needs a much larger intermediate — see SHAPES note).
GPTOSS_MESH_IDS = ["1x1", "1x2"]

# mode -> seq_len. Prefill seq_len must be a multiple of 32.
MODE_SEQ = {"decode": 1, "prefill": 128}

DEFAULT_TRIALS = 5
TRACE_REGION_SIZE = 200_000_000
# MoE has many ops × E; give the profiler plenty of program-support headroom.
OP_SUPPORT_COUNT = 20000

THIS_FILE = "models/common/tests/modules/moe/test_moe_1d_perf.py"


# ============================================================================
# Deterministic weights (shared by MoE1D and the Gemma4 reference)
# ============================================================================


def _gemma_weights(H, I, E, seed=1234):
    """Random Gemma-shaped weights in HF layout, deterministic from ``seed``.

    Both the MoE1D and Gemma4-reference workers regenerate these from the same seed in
    their own subprocess, so device *time* (not numerics) is the only variable.
    """
    import torch

    torch.manual_seed(seed)
    # Well-separated router logits (×4) — avoids pathological random near-tie top-k.
    router_hf = torch.randn(E, H) * (H**-0.5) * 4.0  # [E, H]
    gate_hf = torch.randn(E, I, H) * (H**-0.5)  # [E, I, H]
    up_hf = torch.randn(E, I, H) * (H**-0.5)  # [E, I, H]
    down_hf = torch.randn(E, H, I) * (I**-0.5)  # [E, H, I]
    router_input_scale = torch.randn(H) * 0.1 + 1.0  # [H]
    per_expert = torch.rand(E) + 0.5  # [E]
    router_gamma = torch.ones(H)  # Gemma router RMSNorm: pure normalize (no learned gamma)
    return dict(
        router_hf=router_hf,
        gate_hf=gate_hf,
        up_hf=up_hf,
        down_hf=down_hf,
        router_input_scale=router_input_scale,
        per_expert=per_expert,
        router_gamma=router_gamma,
    )


def _gpt_oss_weights(H, I, E, seed=4321):
    """Random GPT-OSS-shaped weights in HF layout, deterministic from ``seed``.

    Biases on every projection, no router/expert pre-norm (caller pre-norms). Used to
    build both the GPT-OSS-config MoE1D and the reference GPT-OSS ``MLP`` from identical
    numbers, so device *time* is the only variable.
    """
    import torch

    torch.manual_seed(seed)
    router_hf = torch.randn(E, H) * (H**-0.5) * 4.0  # [E, H]
    router_bias = torch.randn(E) * 0.1  # [E]
    gate_hf = torch.randn(E, I, H) * (H**-0.5)  # [E, I, H]
    up_hf = torch.randn(E, I, H) * (H**-0.5)  # [E, I, H]
    down_hf = torch.randn(E, H, I) * (I**-0.5)  # [E, H, I]
    gate_bias = torch.randn(E, I) * 0.1  # [E, I]
    up_bias = torch.randn(E, I) * 0.1  # [E, I]
    down_bias = torch.randn(E, H) * 0.1  # [E, H]
    return dict(
        router_hf=router_hf,
        router_bias=router_bias,
        gate_hf=gate_hf,
        up_hf=up_hf,
        down_hf=down_hf,
        gate_bias=gate_bias,
        up_bias=up_bias,
        down_bias=down_bias,
    )


def _build_moe1d_gptoss(mesh_device, shape, w, *, expert_dtype, router_dtype, topology):
    """Build a GPT-OSS-config MoE1D (clamped-SwiGLU, biases, topk→softmax) from shared weights."""
    from models.common.modules.lazy_weight import LazyWeight
    from models.common.modules.moe.moe_1d import ExpertActivation, MoE1D, MoEConfig, RoutingNorm

    H, I, E, K = shape["H"], shape["I"], shape["E"], shape["K"]

    def lw(src, dt=expert_dtype):
        return LazyWeight(source=src, dtype=dt)

    # Match the GPT-OSS reference's expert compute config for a fair comparison: the
    # reference experts (models/demos/gpt_oss/tt/experts/*.py) pass NO compute_kernel_config
    # → ttnn default LoFi, and pack activations as bfloat8_b. MoE1D's resolved default is
    # HiFi4+fp32_dest_acc + bf16 activations (tuned for PCC headroom), which is ~4×/2× heavier
    # and not what a production GPT-OSS port would run. Build the GPT-OSS config the way the
    # reference does so the perf delta reflects op structure, not a fidelity/dtype mismatch.
    lofi_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    config = MoEConfig(
        router_weight=LazyWeight(source=w["router_hf"].t().contiguous().reshape(1, 1, H, E), dtype=router_dtype),
        gate_proj=lw(w["gate_hf"].transpose(-2, -1).unsqueeze(0).contiguous()),  # [1,E,H,I]
        up_proj=lw(w["up_hf"].transpose(-2, -1).unsqueeze(0).contiguous()),
        down_proj=lw(w["down_hf"].transpose(-2, -1).unsqueeze(0).contiguous()),  # [1,E,I,H]
        top_k=K,
        router_bias=LazyWeight(source=w["router_bias"].reshape(1, 1, 1, E), dtype=router_dtype),
        gate_bias=lw(w["gate_bias"].reshape(1, E, I)),
        up_bias=lw(w["up_bias"].reshape(1, E, I)),
        down_bias=lw(w["down_bias"].reshape(1, E, H)),
        expert_activation=ExpertActivation.CLAMPED_SWIGLU,
        routing_norm=RoutingNorm.TOPK_SOFTMAX,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        expert_weight_dtype=expert_dtype,
        router_dtype=router_dtype,
        # Reference-matched expert compute config (see comment above).
        activation_dtype=ttnn.bfloat8_b,
        expert_compute_kernel_cfg=lofi_cfg,
        mesh_device=mesh_device,
        topology=topology,
        num_reduce_scatter_links=1,
    )
    return MoE1D.from_config(config)


def _build_gpt_oss_mlp(mesh_device, shape, w, *, router_dtype):
    """Build the GPT-OSS reference ``MLP`` (sparse ``Experts`` path) from the SAME weights.

    Forced ``use_throughput_experts=False`` so it runs the standard sparse-matmul experts
    MoE1D mirrors (NOT the Galaxy EP all-to-all throughput path, which is out of scope).
    Expert weights are ``bfloat4_b`` (the dtype the GPT-OSS ``MLP`` hardcodes).
    """
    from types import SimpleNamespace

    import torch

    from models.demos.gpt_oss.config import MeshConfig, ModeConfig
    from models.demos.gpt_oss.tt.ccl import CCLManager
    from models.demos.gpt_oss.tt.mlp import MLP

    H, I, E, K = shape["H"], shape["I"], shape["E"], shape["K"]

    hf_config = SimpleNamespace(
        num_local_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        intermediate_size=I,
        swiglu_limit=7.0,
    )

    # GPT-OSS fused layout: gate_up_proj [E, H, 2I] interleaved (gate=even, up=odd cols);
    # gate_up_proj_bias [E, 2I] interleaved; down_proj [E, I, H]; down_proj_bias [E, H].
    gate_t = w["gate_hf"].transpose(-2, -1).contiguous()  # [E, H, I]
    up_t = w["up_hf"].transpose(-2, -1).contiguous()  # [E, H, I]
    gate_up = torch.empty(E, H, 2 * I, dtype=gate_t.dtype)
    gate_up[..., ::2] = gate_t
    gate_up[..., 1::2] = up_t
    gate_up_bias = torch.empty(E, 2 * I, dtype=w["gate_bias"].dtype)
    gate_up_bias[..., ::2] = w["gate_bias"]
    gate_up_bias[..., 1::2] = w["up_bias"]

    state_dict = {
        "router.weight": w["router_hf"],
        "router.bias": w["router_bias"],
        "experts.gate_up_proj": gate_up,
        "experts.gate_up_proj_bias": gate_up_bias,
        "experts.down_proj": w["down_hf"].transpose(-2, -1).contiguous(),  # [E, I, H]
        "experts.down_proj_bias": w["down_bias"],
    }

    cols = mesh_device.shape[1]
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=cols, ep=1))
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Ring)

    return MLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        dtype=router_dtype,
        mesh_config=mesh_config,
        use_throughput_experts=False,
        tensor_cache_path=None,
    )


def _build_moe1d(mesh_device, shape, w, *, expert_dtype, router_dtype, include_expert_norm):
    """Build a MoE1D (Gemma config) from the shared torch weights.

    ``include_expert_norm`` controls the Stage-1/Stage-2 boundary: Stage 1 (self-baseline)
    folds the expert pre-norm inside MoE1D; Stage 2 drops it (``=None``) because the Gemma4
    ``MoEBlock`` does not include ``pre_feedforward_layernorm_2`` — see module docstring.
    """
    from models.common.modules.lazy_weight import LazyWeight
    from models.common.modules.moe.moe_1d import ExpertActivation, MoE1D, MoEConfig, RoutingNorm

    H, I, E, K = shape["H"], shape["I"], shape["E"], shape["K"]

    def lw(src, dt=expert_dtype):
        return LazyWeight(source=src, dtype=dt)

    router_weight = LazyWeight(source=w["router_hf"].t().contiguous().reshape(1, 1, H, E), dtype=router_dtype)
    gate_proj = lw(w["gate_hf"].transpose(-2, -1).unsqueeze(0).contiguous())  # [1,E,H,I]
    up_proj = lw(w["up_hf"].transpose(-2, -1).unsqueeze(0).contiguous())
    down_proj = lw(w["down_hf"].transpose(-2, -1).unsqueeze(0).contiguous())  # [1,E,I,H]
    router_norm_weight = LazyWeight(source=w["router_gamma"].reshape(1, 1, H // 32, 32), dtype=ttnn.bfloat16)
    router_input_scale = LazyWeight(source=w["router_input_scale"].reshape(1, 1, 1, H), dtype=ttnn.bfloat16)
    per_expert_scale = LazyWeight(source=w["per_expert"].reshape(1, 1, 1, E), dtype=ttnn.bfloat16)
    expert_norm_weight = None
    if include_expert_norm:
        # Reuse router_gamma shape for an expert norm gamma (ones) — keeps the op present.
        expert_norm_weight = LazyWeight(source=w["router_gamma"].reshape(1, 1, H // 32, 32), dtype=ttnn.bfloat16)

    config = MoEConfig(
        router_weight=router_weight,
        gate_proj=gate_proj,
        up_proj=up_proj,
        down_proj=down_proj,
        top_k=K,
        router_norm_weight=router_norm_weight,
        router_input_scale=router_input_scale,
        router_logit_scale=H**-0.5,
        per_expert_scale=per_expert_scale,
        expert_norm_weight=expert_norm_weight,
        expert_activation=ExpertActivation.GEGLU,
        routing_norm=RoutingNorm.SOFTMAX_TOPK_RENORM,
        expert_weight_dtype=expert_dtype,
        router_dtype=router_dtype,
        mesh_device=mesh_device,
        # Match the Gemma4 reference's collective placement for a fair region.
        topology=ttnn.Topology.Linear,
        num_reduce_scatter_links=1,
    )
    return MoE1D.from_config(config)


def _build_gemma_moeblock(mesh_device, shape, w, *, expert_dtype, router_dtype):
    """Build the Gemma4 reference ``MoEBlock`` from the SAME torch weights / dtypes."""
    from types import SimpleNamespace

    import torch

    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.moe import MoEBlock

    H, I, E, K = shape["H"], shape["I"], shape["E"], shape["K"]

    hf_config = SimpleNamespace(
        num_experts=E,
        top_k_experts=K,
        hidden_size=H,
        moe_intermediate_size=I,
        rms_norm_eps=1e-6,
    )

    # State dict in the layout the Gemma4 router/experts loaders expect.
    #   router.proj.weight : [E, H]   (loader transposes)
    #   router.scale       : [H]
    #   router.per_expert_scale : [E]
    #   experts.gate_up_proj : [E, 2I, H]  (first half gate, second up)
    #   experts.down_proj    : [E, H, I]
    fused_gate_up = torch.cat([w["gate_hf"], w["up_hf"]], dim=1)  # [E, 2I, H]
    state_dict = {
        "router.proj.weight": w["router_hf"],
        "router.scale": w["router_input_scale"],
        "router.per_expert_scale": w["per_expert"],
        "experts.gate_up_proj": fused_gate_up,
        "experts.down_proj": w["down_hf"],
    }

    cols = mesh_device.shape[1]
    mesh_config = MeshConfig(tuple(mesh_device.shape), decode=ModeConfig(tp=cols))
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    return MoEBlock(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        dtype=expert_dtype,
        router_dtype=router_dtype,
        tensor_cache_path=None,
    )


# ============================================================================
# Trace-capture + signposted replay (the measured region)
# ============================================================================


def _measure(mesh_device, forward_fn, n_trials):
    """Warmup → capture one forward → warm replay → N signposted measured replays.

    Each measured replay is its own ``signpost("start"/"stop")`` window so the parser can
    recover a per-replay distribution (median + range). ``forward_fn`` rebinds the same
    persistent input buffer each call (trace replay requires it).
    """
    warm = forward_fn()
    ttnn.synchronize_device(mesh_device)
    if hasattr(warm, "deallocate"):
        warm.deallocate(True)

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_out = forward_fn()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    try:
        # Warm replay (no signpost) so the first measured window is steady-state.
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        for i in range(n_trials):
            signpost("start")
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            signpost("stop")
            logger.info(f"measured replay {i + 1}/{n_trials} done")
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    return trace_out


def _measure_untraced(mesh_device, make_input, run_forward, n_trials):
    """Warmup → N signposted *untraced* forwards (fresh input each call).

    Used for the GPT-OSS reference, which (a) infers ``sparse_matmul`` nnz at runtime in
    decode (an event sync forbidden inside trace capture) and (b) deallocates its own
    input — both incompatible with the persistent-buffer trace-replay path. The metric is
    Σ ``DEVICE KERNEL DURATION``, which excludes host-dispatch gaps, so an untraced forward
    yields the *same* per-op device time as a traced replay — only the host gaps (absent
    from this sum) differ. A fresh input per call sidesteps the in-place input dealloc.
    """
    warm = run_forward(make_input())
    ttnn.synchronize_device(mesh_device)
    if hasattr(warm, "deallocate"):
        warm.deallocate(True)

    for i in range(n_trials):
        inp = make_input()  # fresh — built outside the signpost window
        signpost("start")
        out = run_forward(inp)
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
        logger.info(f"measured (untraced) forward {i + 1}/{n_trials} done")
        if hasattr(out, "deallocate"):
            out.deallocate(True)

    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)


# ============================================================================
# Worker — runs inside the tracy-profiled subprocess
# ============================================================================


def _mesh_param_from_env():
    """Build the ``ttnn_mesh_device`` fixture param from ``MOE_PERF_MESH`` (e.g. '1,8')."""
    ms = os.environ.get("MOE_PERF_MESH", "1,1")
    r, c = (int(x) for x in ms.split(","))
    param = {"mesh_shape": (r, c), "trace_region_size": TRACE_REGION_SIZE}
    # Stage 3 mirrors GPT-OSS, whose CCL ops require the ring fabric on every
    # multi-device mesh (the conftest default is FABRIC_1D for <8 devices).
    if r * c > 1 and int(os.environ.get("MOE_PERF_STAGE", "1")) == 3:
        param["fabric_config"] = ttnn.FabricConfig.FABRIC_1D_RING
    return param


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [_mesh_param_from_env()],
    ids=[os.environ.get("MOE_PERF_MESH", "1,1").replace(",", "x")],
    indirect=True,
)
def test_moe1d_perf_worker(ttnn_mesh_device):
    """Profiled body. Reads config from ``MOE_PERF_*`` env, runs the signposted replays.

    Skipped unless invoked by an orchestrator (``MOE_PERF_WORKER=1``), so collecting this
    file in a normal pytest run never opens a device here.
    """
    if not os.environ.get("MOE_PERF_WORKER"):
        pytest.skip("worker is driven by the perf orchestrator (set MOE_PERF_WORKER=1)")

    import torch  # noqa: F401  (ensures torch present for weight build)

    mesh_device = ttnn_mesh_device
    target = os.environ.get("MOE_PERF_TARGET", "moe1d")
    shape_name = os.environ.get("MOE_PERF_SHAPE", "small")
    mode = os.environ.get("MOE_PERF_MODE", "decode")
    stage = int(os.environ.get("MOE_PERF_STAGE", "1"))
    n_trials = int(os.environ.get("MOE_PERF_TRIALS", str(DEFAULT_TRIALS)))

    shape = SHAPES[shape_name]
    H, I, E, K = shape["H"], shape["I"], shape["E"], shape["K"]
    seq_len = MODE_SEQ[mode]
    # Stage 3 mirrors GPT-OSS (bfloat4_b experts, ring all-reduce); Stages 1/2 mirror
    # Gemma4 (bfloat8_b experts, linear all-reduce). Dtypes are matched on both sides.
    expert_dtype = ttnn.bfloat4_b if stage == 3 else ttnn.bfloat8_b
    router_dtype = ttnn.bfloat16

    logger.info(
        f"[worker] target={target} stage={stage} mesh={tuple(mesh_device.shape)} mode={mode} "
        f"seq={seq_len} shape={shape_name} (H={H} I={I} E={E} K={K}) trials={n_trials}"
    )

    w = _gpt_oss_weights(H, I, E) if stage == 3 else _gemma_weights(H, I, E)

    torch.manual_seed(7)
    torch_input = torch.randn(1, 1, seq_len, H, dtype=torch.float32).to(torch.bfloat16)

    def make_input():
        return ttnn.from_torch(
            torch_input,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # Stage 3 (GPT-OSS) is measured untraced: the reference infers nnz at runtime (decode)
    # and deallocates its own input, both incompatible with trace replay. Σ device-kernel-ns
    # is host-dispatch-independent, so untraced is directly comparable to the traced sum.
    traced = stage != 3

    if target == "moe1d":
        if stage == 3:
            # GPT-OSS-config MoE1D: clamped-SwiGLU, biases, topk→softmax, ring all-reduce.
            model = _build_moe1d_gptoss(
                mesh_device,
                shape,
                w,
                expert_dtype=expert_dtype,
                router_dtype=router_dtype,
                topology=ttnn.Topology.Ring,
            )
        else:
            # Stage 1 self-baseline folds the expert norm in; Stage 2 drops it to match MoEBlock.
            model = _build_moe1d(
                mesh_device,
                shape,
                w,
                expert_dtype=expert_dtype,
                router_dtype=router_dtype,
                include_expert_norm=(stage == 1),
            )

        def run_forward(inp):
            return model.forward(inp, mode)

    elif target == "gemma":
        block = _build_gemma_moeblock(mesh_device, shape, w, expert_dtype=expert_dtype, router_dtype=router_dtype)

        # Boundary fix (option a): MoEBlock takes (router_input, expert_input) and applies
        # no expert pre-norm; feed the same x to both so the region matches MoE1D's
        # router(norm)+experts (expert_norm dropped on the MoE1D side for Stage 2).
        def run_forward(inp):
            return block(inp, inp)

    elif target == "gptoss":
        mlp = _build_gpt_oss_mlp(mesh_device, shape, w, router_dtype=router_dtype)
        is_decode = mode == "decode"

        # MLP(hidden, is_decode) is router + sparse experts — the equal-work region vs MoE1D.
        def run_forward(inp):
            return mlp(inp, is_decode)

    else:
        pytest.fail(f"unknown MOE_PERF_TARGET={target!r}")

    if traced:
        # Persistent input — trace replay rebinds the same buffer (MoE1D does not free it).
        tt_input = make_input()
        out = _measure(mesh_device, lambda: run_forward(tt_input), n_trials)
        out_t = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
        assert torch.isfinite(out_t).all(), "traced MoE forward produced non-finite output"
        logger.info(f"[worker] done; output shape {tuple(out_t.shape)}")
        # Drain profiler + free device tensors before the fixture closes the mesh (a buffered
        # profiler + live trace tensors corrupt the heap at teardown on some configs).
        out.deallocate(True)
        tt_input.deallocate(True)
        ttnn.synchronize_device(mesh_device)
        ttnn.ReadDeviceProfiler(mesh_device)
    else:
        _measure_untraced(mesh_device, make_input, run_forward, n_trials)
        logger.info("[worker] done (untraced)")


# ============================================================================
# CSV parsing — per-replay region µs + Matmul/collective split
# ============================================================================

_MATMUL_PAT = "matmul"
_COLLECTIVE_PAT = ("allreduce", "reducescatter", "allgather")


def _summarize_windows(subdir, num_devices):
    """Per signpost window: region µs, matmul µs, collective µs.

    Extends ``sweep_mm_block_sizes.parse_ops_log``: instead of one region collapsed by
    op, we recover every start/stop window (one per replay) and split by op category.
    Per-device rows are collapsed by dividing each category's summed device-ns by
    ``num_devices`` (the mean per-device device-time = steady-state concurrent wall time,
    matching parse_ops_log's average-over-devices intent).
    """
    import pandas as pd
    from tracy.process_model_log import get_latest_ops_log_filename

    filename = get_latest_ops_log_filename(subdir)
    df = pd.read_csv(filename)

    sp = df[df["OP TYPE"] == "signpost"]
    starts = sp[sp["OP CODE"] == "start"].index.tolist()
    stops = sp[sp["OP CODE"] == "stop"].index.tolist()
    if not starts or not stops:
        logger.warning(f"no start/stop signposts in {filename}")
        return []

    windows = []
    for s, e in zip(starts, stops):
        if e <= s:
            continue
        w = df.iloc[s + 1 : e]
        w = w[w["OP TYPE"] != "signpost"]
        w = w[w["DEVICE KERNEL DURATION [ns]"] != "-"]
        if w.empty:
            continue
        dur = w["DEVICE KERNEL DURATION [ns]"].astype(float)
        code = w["OP CODE"].astype(str).str.lower()
        mm = dur[code.str.contains(_MATMUL_PAT)].sum()
        coll = dur[code.apply(lambda c: any(p in c for p in _COLLECTIVE_PAT))].sum()
        windows.append(
            dict(
                region_us=dur.sum() / num_devices / 1000.0,
                matmul_us=mm / num_devices / 1000.0,
                collective_us=coll / num_devices / 1000.0,
                n_ops=len(w),
            )
        )
    return windows


def _stats(windows, key):
    vals = [w[key] for w in windows]
    return statistics.median(vals), min(vals), max(vals)


# ============================================================================
# Orchestrators — spawn the profiled worker, parse, print
# ============================================================================


def _run_worker_and_parse(*, stage, target, mesh_id, mode, shape_name, trials=DEFAULT_TRIALS):
    """Set env, profile the worker in a subprocess, return per-replay window summaries."""
    from tracy.process_model_log import run_device_profiler

    rows, cols = MESHES[mesh_id]
    num_devices = rows * cols
    subdir = f"moe_perf_s{stage}_{target}_{mesh_id}_{mode}_{shape_name}"

    env_keys = {
        "MOE_PERF_WORKER": "1",
        "MOE_PERF_MESH": f"{rows},{cols}",
        "MOE_PERF_MODE": mode,
        "MOE_PERF_SHAPE": shape_name,
        "MOE_PERF_TARGET": target,
        "MOE_PERF_STAGE": str(stage),
        "MOE_PERF_TRIALS": str(trials),
    }
    saved = {k: os.environ.get(k) for k in env_keys}
    os.environ.update(env_keys)

    command = f"pytest {THIS_FILE}::test_moe1d_perf_worker -x -s --timeout=2000"

    try:
        run_device_profiler(
            command,
            subdir,
            device_analysis_types=["device_kernel_duration"],
            op_support_count=OP_SUPPORT_COUNT,
        )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    try:
        windows = _summarize_windows(subdir, num_devices)
    except FileNotFoundError:
        windows = []
    return windows, num_devices


def _print_table(label, windows):
    med, lo, hi = _stats(windows, "region_us")
    mm, _, _ = _stats(windows, "matmul_us")
    cc, _, _ = _stats(windows, "collective_us")
    logger.info(
        f"  {label:<22} region {med:8.2f} µs  [{lo:7.2f}, {hi:7.2f}]  "
        f"matmul {mm:7.2f}  collective {cc:6.2f}  (rows/replay={windows[0]['n_ops']}, replays={len(windows)})"
    )
    return med


@pytest.mark.parametrize("shape_name", GEMMA_SHAPE_IDS, ids=GEMMA_SHAPE_IDS)
@pytest.mark.parametrize("mode", list(MODE_SEQ), ids=list(MODE_SEQ))
@pytest.mark.parametrize("mesh_id", list(MESHES), ids=list(MESHES))
def test_moe1d_perf(mesh_id, mode, shape_name):
    """Stage 1 — MoE1D self-baseline. Device-µs (median + range) + matmul/collective split."""
    windows, num_devices = _run_worker_and_parse(
        stage=1, target="moe1d", mesh_id=mesh_id, mode=mode, shape_name=shape_name
    )
    if not windows:
        pytest.skip(f"no profiler windows for {mesh_id}/{mode}/{shape_name} (mesh unavailable on this box?)")

    logger.info(f"=== MoE1D Stage 1 [{mesh_id} {mode} {shape_name}] (devices={num_devices}) ===")
    _print_table("MoE1D", windows)


@pytest.mark.parametrize("shape_name", GEMMA_SHAPE_IDS, ids=GEMMA_SHAPE_IDS)
@pytest.mark.parametrize("mode", list(MODE_SEQ), ids=list(MODE_SEQ))
@pytest.mark.parametrize("mesh_id", list(MESHES), ids=list(MESHES))
def test_moe1d_vs_gemma_perf(mesh_id, mode, shape_name):
    """Stage 2 — MoE1D vs Gemma4 MoEBlock at matched config/shape/dtype.

    Both targets run in separate profiled subprocesses from the same seeded weights.
    Expect near-parity (MoE1D's op sequence is ~1:1 with the Gemma reference). Any gap →
    inspect the matmul/collective split for the divergent op.
    """
    moe1d_w, num_devices = _run_worker_and_parse(
        stage=2, target="moe1d", mesh_id=mesh_id, mode=mode, shape_name=shape_name
    )
    gemma_w, _ = _run_worker_and_parse(stage=2, target="gemma", mesh_id=mesh_id, mode=mode, shape_name=shape_name)
    if not moe1d_w or not gemma_w:
        pytest.skip(f"no profiler windows for {mesh_id}/{mode}/{shape_name} (mesh unavailable on this box?)")

    logger.info(f"=== MoE1D vs Gemma4 MoEBlock [{mesh_id} {mode} {shape_name}] (devices={num_devices}) ===")
    moe1d_med = _print_table("MoE1D", moe1d_w)
    gemma_med = _print_table("Gemma4 MoEBlock", gemma_w)
    delta = (moe1d_med - gemma_med) / gemma_med * 100.0
    logger.info(f"  Δ MoE1D vs MoEBlock: {delta:+.1f}%  (MoE1D {moe1d_med:.2f} µs / MoEBlock {gemma_med:.2f} µs)")


@pytest.mark.parametrize("shape_name", GPTOSS_SHAPE_IDS, ids=GPTOSS_SHAPE_IDS)
@pytest.mark.parametrize("mode", ["prefill"], ids=["prefill"])
@pytest.mark.parametrize("mesh_id", GPTOSS_MESH_IDS, ids=GPTOSS_MESH_IDS)
def test_moe1d_vs_gpt_oss_perf(mesh_id, mode, shape_name):
    """Stage 3 — MoE1D (GPT-OSS config) vs GPT-OSS ``MLP`` at matched config/shape/dtype.

    The reference ``MLP`` is forced to ``use_throughput_experts=False`` so it runs the
    standard sparse-matmul ``Experts`` path that MoE1D mirrors (clamped-SwiGLU + biases),
    NOT the Galaxy EP all-to-all throughput path (out of MoE1D's TP-only scope). Both use
    ``bfloat4_b`` experts and ``ttnn.all_reduce(Ring)`` under FABRIC_1D_RING.

    **Prefill only.** The GPT-OSS reference *decode* path calls ``sparse_matmul(nnz=None)``,
    which infers the non-zero count at runtime via a host read → an event sync that
    ``ttnn`` forbids inside trace capture (``Event Synchronization is not supported during
    trace capture``). So the GPT-OSS decode reference is not trace-measurable with this
    harness. MoE1D decode uses a static ``nnz=top_k`` and traces fine; the prefill path
    (static all-ones sparsity, ``nnz=E``) is trace-safe on both sides — that's the
    apples-to-apples region here.
    """
    moe1d_w, num_devices = _run_worker_and_parse(
        stage=3, target="moe1d", mesh_id=mesh_id, mode=mode, shape_name=shape_name
    )
    gpt_w, _ = _run_worker_and_parse(stage=3, target="gptoss", mesh_id=mesh_id, mode=mode, shape_name=shape_name)
    if not moe1d_w or not gpt_w:
        pytest.skip(f"no profiler windows for {mesh_id}/{mode}/{shape_name} (mesh unavailable on this box?)")

    logger.info(f"=== MoE1D vs GPT-OSS MLP [{mesh_id} {mode} {shape_name}] (devices={num_devices}) ===")
    moe1d_med = _print_table("MoE1D (GPT-OSS cfg)", moe1d_w)
    gpt_med = _print_table("GPT-OSS MLP", gpt_w)
    delta = (moe1d_med - gpt_med) / gpt_med * 100.0
    logger.info(f"  Δ MoE1D vs GPT-OSS MLP: {delta:+.1f}%  (MoE1D {moe1d_med:.2f} µs / MLP {gpt_med:.2f} µs)")
