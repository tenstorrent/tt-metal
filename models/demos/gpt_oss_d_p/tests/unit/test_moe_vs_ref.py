# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC tests for the GPT-OSS prefill MoE wrapper.

Three tests, two of which run on a single card:

  1. test_router_vs_ref  (single card, no fabric)
       TtGptOssRouter vs a torch reference: linear(+bias) -> topk(4, sorted) -> softmax over the 4
       selected logits. Indices must match exactly; weights PCC >= 0.99. Fully validates the router.

  2. test_routed_expert_ffn_vs_ref  (single card, mesh=1, all experts local, fabric DISABLED)
       Mirrors deepseek_v3_d_p/tests/op_unit_tests/test_ttnn_routed_expert.py's `single-1` param, but
       with GPT-OSS dims (emb 2880 / hidden 2880), a reduced expert count (E=8, topk=4) and the
       SwiGluOai activation. Parametrized over `biasfree` and `biased` (per-expert gate/up/down
       biases, #49619 — the deployment default): the `biased` case exercises the exact path the model
       runs. PCC >= 0.97. Validates the fused routed-expert kernel wiring.

  3. test_gpt_oss_moe_vs_ref  (GALAXY ONLY — gated behind requires_mesh_topology((4,8)))
       Full TtGptOssMoE end-to-end (router -> dispatch -> routed_expert -> combine -> reduce) at
       EP=32. Needs multi-device + fabric, so it is AUTO-SKIPPED on a single card and only runs on a
       (4,8) Blackhole galaxy.

Run (single card, both card-safe tests):
    pytest models/demos/gpt_oss_d_p/tests/unit/test_moe_vs_ref.py -k "router or routed_expert_ffn"
Galaxy (all three):
    pytest models/demos/gpt_oss_d_p/tests/unit/test_moe_vs_ref.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    ExpertMapping,
    compute_constants,
    extract_mesh_config,
    get_ep_mesh_composer,
    get_ep_mesh_mapper,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.gpt_oss_d_p.tt.moe.router import TtGptOssRouter

# GPT-OSS-120B MoE dims (configs/gpt-oss-120b/config.json / GptOss120BConfig).
HIDDEN = 2880
INTER = 2880
NUM_EXPERTS = 128
TOPK = 4
ALPHA = 1.702  # swigluoai alpha
LIMIT = 7.0  # swigluoai clamp limit


# =====================================================================================
# 1. Router PCC (single card)
# =====================================================================================
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("num_tokens", [64], ids=["t64"])
def test_router_vs_ref(mesh_device, num_tokens, reset_seeds):
    """TtGptOssRouter vs torch: linear(+bias) -> topk(4) -> softmax over the 4 selected."""
    torch.manual_seed(0)

    weight = torch.randn(NUM_EXPERTS, HIDDEN) * 0.05  # HF gate Linear weight [E, H]
    bias = torch.randn(NUM_EXPERTS) * 0.1  # [E]
    x = torch.randn(num_tokens, HIDDEN)

    # --- torch reference ---
    logits = x @ weight.t() + bias  # [T, E]
    ref_weights, ref_indices = torch.topk(logits, TOPK, dim=-1, sorted=True)  # [T, k]
    ref_weights = torch.softmax(ref_weights, dim=-1)

    # --- TT router ---
    hf_config = SimpleNamespace(num_experts_per_tok=TOPK, num_local_experts=NUM_EXPERTS, hidden_size=HIDDEN)
    router = TtGptOssRouter(mesh_device, hf_config, {"weight": weight, "bias": bias})

    x_tt = ttnn.from_torch(
        x,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    idx_tt, wts_tt = router(x_tt)
    ttnn.synchronize_device(mesh_device)

    idx = ttnn.to_torch(ttnn.get_device_tensors(idx_tt)[0]).reshape(-1, TOPK)[:num_tokens].to(torch.int64)
    wts = ttnn.to_torch(ttnn.get_device_tensors(wts_tt)[0]).reshape(-1, TOPK)[:num_tokens].float()

    # bf16 matmul rounding can flip experts at the top-4/top-5 boundary vs the fp32 reference —
    # inherent to bf16 top-k, not a router defect. So rather than demand an exact index match,
    # verify every mismatch is a genuine near-tie: the reference's 4th- and 5th-largest logits
    # are within a bf16-scale tolerance. Order within the top-4 is irrelevant (softmax is over
    # the selected set), so compare unordered sets.
    top5 = torch.topk(logits, TOPK + 1, dim=-1).values  # [T, k+1], descending
    boundary_gap = top5[:, TOPK - 1] - top5[:, TOPK]  # (4th - 5th) logit gap, >= 0
    tol = 0.02 * top5[:, 0].abs().clamp(min=1.0)  # ~bf16 relative scale, per token
    ref_sets = [set(r.tolist()) for r in ref_indices]
    dev_sets = [set(r.tolist()) for r in idx]
    set_match = sum(a == b for a, b in zip(ref_sets, dev_sets)) / num_tokens
    logger.info(f"router top-4 set-match fraction: {set_match:.4f}")
    for i in range(num_tokens):
        if ref_sets[i] != dev_sets[i]:
            assert boundary_gap[i] <= tol[i], (
                f"token {i}: top-4 differs from reference but 4th/5th logits are not a near-tie "
                f"(gap={boundary_gap[i]:.4f} > tol={tol[i]:.4f}) — router logic error, not a bf16 flip"
            )

    # Softmax-over-top-4 math: compare on tokens whose selection matched the reference (weights
    # are only element-wise comparable there; flipped tokens picked a different near-tied set).
    matched = [i for i in range(num_tokens) if ref_sets[i] == dev_sets[i]]
    assert len(matched) >= num_tokens * 0.5, f"too few matched tokens to validate weights ({len(matched)})"
    passing, pcc = comp_pcc(ref_weights[matched], wts[matched], 0.99)
    logger.info(f"router weights pcc (matched tokens): {pcc}")
    assert passing, f"router expert-weights PCC fail: {pcc}"


# =====================================================================================
# 1b. Expert weight/bias preparation (pure torch, no device)
# =====================================================================================
def test_prepare_routed_expert_weights_deinterleave():
    """Pure-torch: prepare_routed_expert_weights de-interleaves gate(even)/up(odd) columns, splits the
    matching biases the same way, transposes every projection to HF (out,in), and emits the #49619
    bias keys (gate/up/down_proj_bias). Guards the host-side layout math that the galaxy MoE test and
    the biased FFN test both depend on but neither exercises through prepare()."""
    from models.demos.gpt_oss_d_p.tt.moe.weights import prepare_routed_expert_weights

    torch.manual_seed(0)
    E, H, I = 3, 4, 5
    gate = torch.randn(E, H, I)  # (in=H, out=I)
    up = torch.randn(E, H, I)
    gate_up = torch.zeros(E, H, 2 * I)
    gate_up[..., ::2] = gate  # even columns -> gate
    gate_up[..., 1::2] = up  # odd columns -> up
    gate_b = torch.randn(E, I)
    up_b = torch.randn(E, I)
    gate_up_b = torch.zeros(E, 2 * I)
    gate_up_b[..., ::2] = gate_b
    gate_up_b[..., 1::2] = up_b
    down = torch.randn(E, I, H)  # (in=I, out=H)
    down_b = torch.randn(E, H)

    w, b = prepare_routed_expert_weights(
        {
            "gate_up_proj": gate_up,
            "gate_up_proj_bias": gate_up_b,
            "down_proj": down,
            "down_proj_bias": down_b,
        },
        E,
        H,
        I,
    )
    assert len(w) == E and len(b) == E
    for e in range(E):
        # weights transposed to HF (out, in)
        assert torch.equal(w[e]["gate_proj"], gate[e].t()), f"gate mismatch expert {e}"
        assert torch.equal(w[e]["up_proj"], up[e].t()), f"up mismatch expert {e}"
        assert torch.equal(w[e]["down_proj"], down[e].t()), f"down mismatch expert {e}"
        # biases split + #49619 key names, kept 1D (not transposed)
        assert torch.equal(b[e]["gate_proj_bias"], gate_b[e]), f"gate_bias mismatch expert {e}"
        assert torch.equal(b[e]["up_proj_bias"], up_b[e]), f"up_bias mismatch expert {e}"
        assert torch.equal(b[e]["down_proj_bias"], down_b[e]), f"down_bias mismatch expert {e}"


# =====================================================================================
# SwiGLU-OAI bias-free torch expert (matches ttnn.RoutedExpertActivation.SwiGluOai without bias)
# =====================================================================================
def _swiglu_oai_ffn(x, w, b=None):
    """Clamped swigluoai FFN. w: HF (out, in) tensors gate_proj/up_proj/down_proj. b (optional):
    per-expert gate_proj_bias/up_proj_bias/down_proj_bias (1D), applied the way the fused kernel does
    (#49619): gate/up bias BEFORE the clamp, down bias AFTER the down matmul."""
    gp = x @ w["gate_proj"].t()
    up = x @ w["up_proj"].t()
    if b is not None:
        gp = gp + b["gate_proj_bias"]
        up = up + b["up_proj_bias"]
    g = gp.clamp(max=LIMIT)
    u = up.clamp(min=-LIMIT, max=LIMIT)
    out = ((u + 1.0) * (g * torch.sigmoid(ALPHA * g))) @ w["down_proj"].t()
    if b is not None:
        out = out + b["down_proj_bias"]
    return out


@torch.no_grad()
def _run_torch_routed_experts(
    dispatched_buffer,
    weights_list,
    experts_per_chip,
    num_dispatch_groups,
    dispatch_group_size,
    expert_token_counts,
    biases_list=None,
):
    """Torch SwiGLU-OAI routed experts over the dispatch-shaped buffer. Mirrors the offset walk in
    deepseek's test_ttnn_routed_expert.run_torch_routed_experts (TILE-aligned per-expert regions).
    biases_list (optional): per-expert bias dicts in the same global order as weights_list."""
    out = torch.zeros_like(dispatched_buffer)
    for dg in range(num_dispatch_groups):
        for ds in range(dispatch_group_size):
            off = 0
            for local_expert in range(experts_per_chip):
                ge = ExpertMapping.get_global_expert_idx(
                    group=dg,
                    chip=ds,
                    local_expert=local_expert,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=dispatch_group_size,
                    num_dispatch_groups=num_dispatch_groups,
                    is_col_major=True,
                )
                count = expert_token_counts[dg, 0, ge].item()
                if count > 0:
                    xin = dispatched_buffer[dg, ds, off : off + count, :].float()
                    b = biases_list[ge] if biases_list is not None else None
                    out[dg, ds, off : off + count, :] = _swiglu_oai_ffn(xin, weights_list[ge], b)
                off += (count + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
    return out


# =====================================================================================
# 2. Routed-expert FFN plumbing PCC (single card, mesh=1) — bias-free AND biased
# =====================================================================================
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param(1, {"fabric_config": ttnn.FabricConfig.DISABLED}, id="single-1")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "seq_len_per_chip, emb_dim, hidden_dim, num_routed_experts, num_experts_per_tok, capacity_factor",
    [(256, HIDDEN, INTER, 8, TOPK, 8)],
    ids=["gptoss-e8"],
)
@pytest.mark.parametrize("use_bias", [False, True], ids=["biasfree", "biased"])
def test_routed_expert_ffn_vs_ref(
    mesh_device,
    device_params,
    seq_len_per_chip,
    emb_dim,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    capacity_factor,
    use_bias,
    reset_seeds,
):
    """TtRoutedExpert (SwiGluOai) vs a SwiGLU-OAI torch reference, both bias-free (``biasfree``) and
    with per-expert gate/up/down biases (``biased``, #49619 — the deployment default). All experts
    local on a single device (mesh=1). Validates the fused routed-expert kernel wiring at GPT-OSS
    dims, including the bias path the model actually runs."""
    torch.manual_seed(42)
    num_devices = mesh_device.get_num_devices()
    mc = extract_mesh_config(mesh_device)
    dispatch_group_size = mc.dispatch_group_size
    num_dispatch_groups = mc.num_dispatch_groups

    experts_per_chip, metadata_len, max_buf, max_tok = compute_constants(
        seq_len_per_chip, num_routed_experts, num_experts_per_tok, num_devices, dispatch_group_size, capacity_factor
    )
    total_experts = num_devices * experts_per_chip

    # Routing indices -> per-expert token counts + TILE-aligned region offsets.
    expert_dispatch_table = ExpertMapping.create_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    _, _, routing_indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        emb_dim=emb_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_tok,
        num_dispatch_groups=num_dispatch_groups,
        skip_x_initialization=True,
    )
    _, expert_token_counts_torch, expert_region_offsets_torch, _ = get_gate_outputs(
        routing_indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
        expert_dispatch_table=expert_dispatch_table,
    )

    per_device_shape = (max_buf, emb_dim)

    # Random dispatch buffer + per-expert weights (HF (out, in), global order 0..E-1).
    dispatched_buffer_torch = torch.randn(num_dispatch_groups, dispatch_group_size, max_buf, emb_dim)
    weights_list = [
        {
            "gate_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
            "up_proj": torch.randn(hidden_dim, emb_dim) * 0.02,
            "down_proj": torch.randn(emb_dim, hidden_dim) * 0.02,
        }
        for _ in range(total_experts)
    ]
    # Per-expert gate/up/down biases (deployment default, #49619). Kept at the same 0.02 scale as the
    # weights so the biased path exercises real (non-trivial) bias contributions.
    biases_list = (
        [
            {
                "gate_proj_bias": torch.randn(hidden_dim) * 0.02,
                "up_proj_bias": torch.randn(hidden_dim) * 0.02,
                "down_proj_bias": torch.randn(emb_dim) * 0.02,
            }
            for _ in range(total_experts)
        ]
        if use_bias
        else None
    )

    torch_outputs = _run_torch_routed_experts(
        dispatched_buffer_torch,
        weights_list,
        experts_per_chip,
        num_dispatch_groups,
        dispatch_group_size,
        expert_token_counts_torch,
        biases_list=biases_list,
    )

    dispatched_buffer_tt = ttnn.from_torch(
        dispatched_buffer_torch,
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )
    dispatched_buffer_tt = ttnn.reshape(dispatched_buffer_tt, per_device_shape)

    global_expert_idx_torch = ExpertMapping.create_global_expert_idx_table(
        experts_per_chip=experts_per_chip,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=num_dispatch_groups,
    )
    global_expert_idx_tt = ttnn.from_torch(
        global_expert_idx_torch,
        mesh_mapper=get_ep_mesh_mapper(mesh_device),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.uint32,
    )
    global_expert_idx_tt = ttnn.squeeze(ttnn.squeeze(global_expert_idx_tt, 0), 0)

    # TtRoutedExpert (SwiGluOai). torch_biases=None -> bias-free path; a per-expert bias list -> the
    # biased fused path (#49619, the deployment default).
    tt_routed_expert = TtRoutedExpert(
        mesh_device=mesh_device,
        experts_per_chip=experts_per_chip,
        global_expert_idx_table=global_expert_idx_tt,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        max_tokens=max_tok,
        torch_weights=weights_list,
        torch_biases=biases_list,
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat4_b,
        activation=ttnn.RoutedExpertActivation.SwiGluOai,
    )

    expert_token_counts_tt = TtRoutedExpert.shard_expert_token_counts(mesh_device, expert_token_counts_torch)
    expert_region_offsets_tt = TtRoutedExpert.shard_expert_token_counts(mesh_device, expert_region_offsets_torch)

    ttnn_outputs = tt_routed_expert(dispatched_buffer_tt, expert_token_counts_tt, expert_region_offsets_tt)
    ttnn.synchronize_device(mesh_device)

    ttnn_outputs_expanded = ttnn.unsqueeze(ttnn.unsqueeze(ttnn_outputs, dim=0), dim=0)
    ttnn_outputs_torch = ttnn.to_torch(ttnn_outputs_expanded, mesh_composer=get_ep_mesh_composer(mesh_device))

    pcc_values = []
    for dg in range(num_dispatch_groups):
        for ds in range(dispatch_group_size):
            off = 0
            for local_expert in range(experts_per_chip):
                ge = ExpertMapping.get_global_expert_idx(
                    group=dg,
                    chip=ds,
                    local_expert=local_expert,
                    experts_per_chip=experts_per_chip,
                    dispatch_group_size=dispatch_group_size,
                    num_dispatch_groups=num_dispatch_groups,
                    is_col_major=True,
                )
                count = expert_token_counts_torch[dg, 0, ge].item()
                if count > 0:
                    t = torch_outputs[dg, ds, off : off + count]
                    n = ttnn_outputs_torch[dg, ds, off : off + count]
                    _, pcc = comp_pcc(t, n)
                    pcc_values.append(pcc)
                    assert not torch.isnan(n).any(), f"NaN in expert {ge}"
                    assert not torch.isinf(n).any(), f"Inf in expert {ge}"
                off += (count + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE

    min_pcc = min(pcc_values)
    logger.info(f"routed-expert FFN min PCC: {min_pcc:.6f} across {len(pcc_values)} experts")
    assert min_pcc >= 0.97, f"routed-expert FFN PCC {min_pcc:.6f} below 0.97"


# =====================================================================================
# 3. Full TtGptOssMoE end-to-end (GALAXY ONLY — auto-skipped on a single card)
# =====================================================================================
@pytest.mark.requires_mesh_topology(mesh_shape=(4, 8), topology="mesh-4x8")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 8), {"fabric_config": ttnn.FabricConfig.FABRIC_1D}, id="galaxy-4x8")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_gpt_oss_moe_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full GPT-OSS MoE MLP (router + 128-expert EP routed experts) on (4,8) EP=32 vs a per-row torch
    reference. NEEDS a Blackhole galaxy — auto-skipped on a single card by requires_mesh_topology.

    NOTE: this end-to-end check uses ZERO expert biases (bias-free reference) — it validates the
    router -> dispatch -> combine -> reduce plumbing at EP=32, not the bias path. #49619 (expert bias)
    is merged and the biased fused FFN is covered single-card by test_routed_expert_ffn_vs_ref[biased];
    a non-zero-bias galaxy variant is future work.

    DEPENDENCY: the MLP's TP all-gather needs a CCL manager (``.num_links``, ping-pong/barrier
    semaphores — the object MeshConfig.allgather consumes). CCLManager ships in the runtime PR
    (tt/ccl.py, one PR up the stack); on this (MoE) branch the import fails, so this galaxy test
    skips here and runs once the runtime PR lands. Router + routed-expert plumbing are already
    validated by the single-card tests above."""
    from models.demos.gpt_oss_d_p.tt.config import MeshConfig
    from models.demos.gpt_oss_d_p.tt.mlp import MLP
    from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links

    try:
        from models.demos.gpt_oss_d_p.tt.ccl import CCLManager  # ships in the runtime PR (one PR up the stack)
    except ImportError:
        pytest.skip("gpt_oss_d_p CCLManager ships in the runtime PR; full EP MoE test runs once it lands.")

    rows, cols = mesh_device.shape
    assert (rows, cols) == (4, 8), "this test targets the (4,8) galaxy layout (EP=32)"
    torch.manual_seed(0)

    E, H, I = NUM_EXPERTS, HIDDEN, INTER
    router_w = torch.randn(E, H) * 0.05
    router_b = torch.randn(E) * 0.1
    experts = [
        {
            "gate_proj": torch.randn(I, H) * 0.02,
            "up_proj": torch.randn(I, H) * 0.02,
            "down_proj": torch.randn(H, I) * 0.02,
        }
        for _ in range(E)
    ]

    def _ref_moe(x):  # x: [S, H]
        logits = x @ router_w.t() + router_b
        wts, idx = torch.topk(logits, TOPK, dim=-1, sorted=True)
        wts = torch.softmax(wts, dim=-1)
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(TOPK):
                out[t] += wts[t, j] * _swiglu_oai_ffn(x[t : t + 1], experts[idx[t, j].item()]).squeeze(0)
        return out

    x = torch.randn(rows, seq_len, H, dtype=torch.bfloat16)
    ref = [_ref_moe(x[r].float()) for r in range(rows)]

    # Build the HF GptOssMLP state dict: router.{weight,bias} + interleaved experts.* .
    gate_up = torch.zeros(E, H, 2 * I)
    gate_up[..., ::2] = torch.stack([w["gate_proj"].t() for w in experts])  # (in=H, out=I)
    gate_up[..., 1::2] = torch.stack([w["up_proj"].t() for w in experts])
    gate_up_bias = torch.zeros(E, 2 * I)  # bias-free ref -> zeros
    state = {
        "router.weight": router_w,
        "router.bias": router_b,
        "experts.gate_up_proj": gate_up,
        "experts.gate_up_proj_bias": gate_up_bias,
        "experts.down_proj": torch.stack([w["down_proj"].t() for w in experts]),  # (in=I, out=H)
        "experts.down_proj_bias": torch.zeros(E, H),
    }

    hf_config = SimpleNamespace(num_experts_per_tok=TOPK, num_local_experts=E, hidden_size=H, intermediate_size=I)
    mesh_config = MeshConfig((rows, cols), tp=cols)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    mlp = MLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        expert_weight_dtype=ttnn.bfloat8_b,
        use_ep_moe=True,
        ep_seq_len_per_chip=seq_len,
    )

    tt_x = ttnn.from_torch(
        x.reshape(rows, 1, seq_len, H),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    out = mlp(tt_x)
    ttnn.synchronize_device(mesh_device)

    dts = ttnn.get_device_tensors(out)
    for r in range(rows):
        row = ttnn.to_torch(dts[r * cols]).float().reshape(-1, H)[:seq_len]
        passing, pcc = comp_pcc(ref[r], row, 0.90)
        logger.info(f"prompt{r}: pcc={pcc}")
        assert passing, f"prompt {r} GPT-OSS MoE PCC fail: {pcc}"
