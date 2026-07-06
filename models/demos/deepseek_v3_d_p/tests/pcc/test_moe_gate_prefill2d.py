# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0


import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate as ReferenceMoEGate
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    create_fabric_router_config,
    create_gate_weights,
    get_max_payload_size,
    get_sp_mesh_composer,
    load_gate_weights_from_hf,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode, TtMoEGateConfig, TtMoEGatePrefill
from models.demos.deepseek_v3_d_p.tt.moe.validation_helpers import (
    ValidationResult,
    compare_pcc,
    compare_recall,
    grouped_gate_golden_act,
    score_activation,
    validate_composed,
)
from models.demos.deepseek_v3_d_p.tt.moe.visualization_helpers import log_validation_results
from models.demos.deepseek_v3_d_p.utils.test_utils import adjust_shapes_for_testing, get_input_mem_config
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import GOLDEN_LONGBOOK_TRACE, load_trace_gate_input

# Gate models under test, keyed by a stable id. The gate itself picks grouped vs plain top-k
# routing from n_expert_groups (Kimi / V4 have a single group) and sigmoid vs sqrtsoftplus from
# SCORE_FUNC, so each model is fully described by its config class.
GATE_MODELS = {
    "deepseek_v3": DeepSeekV3Config,
    "kimi": KimiK26Config,
    "dsv4_pro": DeepSeekV4ProConfig,
    "dsv4_flash": DeepSeekV4FlashConfig,
}

# First MoE layer in DeepSeek-V3 (metadata moe_layer_offset == 3); the golden
# trace stores its gate input as post_attn_norm_layer_3.
_MOE_LAYER_IDX = 3

# Relative gate-weight tolerance for the tie-aware top-k recall. A device/reference expert swap at a
# crowded grouped-gate boundary is credited when the two experts' gate weights agree within this
# fraction. The boundary swaps observed on the device fp32 gate are near-exact weight ties: an rtol
# sweep on DEVICE_FP32 mesh-4x2 passes for rtol >= 0.002 and fails below (recall dips under the 0.95
# gate), so this is set just at that knee to credit the ties while staying as tight as possible.
# pcc_scores remains the correctness backstop for the selected-weight distribution.
RECALL_WEIGHT_RTOL = 0.002

_DEFAULT_HF_REPO = "deepseek-ai/DeepSeek-V3"
_LOCAL_FALLBACKS = (
    "models/demos/deepseek_v3/reference",
    "/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528",
)


def _resolve_model_id() -> str:
    """Resolve a model identifier (local dir or HF repo ID) for gate weight loading.

    Checks DEEPSEEK_V3_HF_MODEL and standard local paths first; falls back to the
    HF repo ID so that ``load_hf_state_dict_filtered`` can resolve from the HF cache.
    """
    env_path = os.getenv("DEEPSEEK_V3_HF_MODEL")
    if env_path and (Path(env_path) / "model.safetensors.index.json").exists():
        return env_path
    for fallback in _LOCAL_FALLBACKS:
        if (Path(fallback) / "model.safetensors.index.json").exists():
            return fallback
    return _DEFAULT_HF_REPO


def _try_load_real_gate_weights(n_routed_experts: int, dim: int) -> dict | None:
    """Try to load real gate weights from HF; return None on failure."""
    model_id = _resolve_model_id()
    try:
        gate_w = load_gate_weights_from_hf(model_id, layer_idx=3, dtype=torch.bfloat16)
        gate_w["weight"] = gate_w["weight"][:n_routed_experts, :dim]
        gate_w["e_score_correction_bias"] = gate_w["e_score_correction_bias"][:n_routed_experts]
        return gate_w
    except (FileNotFoundError, KeyError) as e:
        logger.warning(f"Could not load real gate weights ({model_id}): {e}. Using random weights.")
        return None


def _try_load_real_gate_input(max_seq_len: int, dim: int) -> torch.Tensor | None:
    """Try to load the gate input from the golden GPU prefill trace; return None on failure.

    Mirrors test_ttnn_moe / test_prefill_transformer: the gate input is the
    post-attention RMSNorm output (``post_attn_norm_layer_{i}``) of the first MoE
    layer in the bit_sculpt golden trace. ``DEEPSEEK_V3_GATE_INPUT_CACHE`` may
    override the trace directory. Returns ``None`` when the trace is unavailable
    so the caller can fall back to synthetic input.
    """
    trace_dir_env = os.environ.get("DEEPSEEK_V3_GATE_INPUT_CACHE")
    trace_dir = Path(trace_dir_env) if trace_dir_env else GOLDEN_LONGBOOK_TRACE
    return load_trace_gate_input(trace_dir, layer_idx=_MOE_LAYER_IDX, max_seq_len=max_seq_len, dim=dim)


# Mesh topologies shared by the regular-gate and hash-gate PCC tests.
MESH_CONFIGS = [
    pytest.param(
        (2, 2),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        },
        2,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
        id="mesh-2x2",
    ),
    pytest.param(
        (2, 2),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        2,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
        id="fabric2d-mesh-2x2",
    ),
    pytest.param(
        (4, 2),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        },
        2,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
        id="mesh-4x2",
    ),
    pytest.param(
        (4, 2),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        2,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(4, 2), topology="mesh-4x2"),
        id="fabric2d-mesh-4x2",
    ),
    pytest.param(
        (2, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        },
        2,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="mesh-2x4",
    ),
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
        },
        2,
        ttnn.Topology.Linear,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="mesh-8x4",
    ),
]

# (gate_model id, gate compute mode). Sigmoid models (V3/Kimi) exercise both the host and on-device
# gates; V4 (sqrtsoftplus) runs the regular gate on device, where the kernel applies the activation.
REGULAR_GATE_CASES = [
    pytest.param("deepseek_v3", GateComputeMode.HOST_ALL, id="deepseek_v3-host_all"),
    pytest.param("deepseek_v3", GateComputeMode.DEVICE_FP32, id="deepseek_v3-device_fp32"),
    pytest.param("kimi", GateComputeMode.HOST_ALL, id="kimi-host_all"),
    pytest.param("kimi", GateComputeMode.DEVICE_FP32, id="kimi-device_fp32"),
    pytest.param("dsv4_pro", GateComputeMode.DEVICE_FP32, id="dsv4_pro-device_fp32"),
    pytest.param("dsv4_flash", GateComputeMode.DEVICE_FP32, id="dsv4_flash-device_fp32"),
]


def _make_gate_input(config, total_seq_len, allow_real_input: bool) -> torch.Tensor:
    """Gate input: the real V3 prefill trace for the 256-expert sigmoid path, else synthetic."""
    torch_input = _try_load_real_gate_input(total_seq_len, config.dim) if allow_real_input else None
    if torch_input is None:
        # 0.1147 is the std of the real gate input; scale it to adjust for smaller test dims.
        torch_input = torch.randn(total_seq_len, config.dim, dtype=torch.bfloat16) * 0.1147 * (7168 / config.dim)
    return torch_input


def _shard_gate_input(config, mesh_device, torch_input: torch.Tensor) -> ttnn.Tensor:
    sharded_mem_config = get_input_mem_config(config, mesh_device.shape)
    return ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=sharded_mem_config,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(0, -1),  # tensor parallel
            mesh_shape=mesh_device.shape,
        ),
    )


def _validate_gate(
    mesh_device,
    tt_topk_scores,
    tt_topk_indices,
    tt_logits,
    reference_topk_indices,
    reference_topk_scores,
    reference_logits,
    recall_threshold,
    logits_pcc_threshold,
    scores_pcc_threshold,
):
    """Shared SP-composed recall/PCC validation for indices, logits and weights."""
    n_sp_devices = mesh_device.shape[0]
    n_tp_devices = mesh_device.shape[1]
    seq_len_per_device = reference_logits.shape[0] // n_sp_devices
    sp_composer = get_sp_mesh_composer(mesh_device)

    # SP-replicated checks: compose into [1, n_sp_devices, ...] for validate_composed. Keep the topk
    # indices and their gate scores position-aligned (do NOT sort the indices) so the tie-aware recall
    # can map each selected expert to its weight.
    host_tt_topk_indices = ttnn.to_torch(tt_topk_indices, mesh_composer=sp_composer).view(
        1, n_sp_devices, seq_len_per_device, -1
    )
    reference_topk_indices = reference_topk_indices.view(1, n_sp_devices, seq_len_per_device, -1)
    host_tt_topk_scores = ttnn.to_torch(tt_topk_scores, mesh_composer=sp_composer).view(
        1, n_sp_devices, seq_len_per_device, -1
    )
    reference_topk_scores = reference_topk_scores.view(1, n_sp_devices, seq_len_per_device, -1)

    # DeepSeek uses grouped top-k gating: at a crowded selection boundary the device fp32 gate and the
    # torch reference can pick different experts that carry near-equal gate weight. Such a swap does not
    # change the routed output (block-level PCC stays ~0.999), so credit it in the recall when the
    # swapped-in expert's weight is within RECALL_WEIGHT_RTOL of the missed expert's. pcc_scores (below)
    # remains the correctness backstop: a genuine mis-route shifts the selected-weight distribution.
    recall_topk_indices = validate_composed(
        host_tt_topk_indices,
        reference_topk_indices,
        1,
        n_sp_devices,
        compare_recall(
            recall_threshold,
            predicted_weights=host_tt_topk_scores,
            reference_weights=reference_topk_scores,
            weight_rtol=RECALL_WEIGHT_RTOL,
        ),
        name="recall_topk_indices",
        broadcast_groups=n_tp_devices,
    )

    host_tt_logits = ttnn.to_torch(tt_logits, mesh_composer=sp_composer)
    host_tt_logits = host_tt_logits.view(1, n_sp_devices, seq_len_per_device, -1)
    reference_logits = reference_logits.view(1, n_sp_devices, seq_len_per_device, -1)

    pcc_logits = validate_composed(
        host_tt_logits,
        reference_logits,
        1,
        n_sp_devices,
        compare_pcc(logits_pcc_threshold),
        name="pcc_logits",
        broadcast_groups=n_tp_devices,
    )

    pcc_scores = validate_composed(
        host_tt_topk_scores,
        reference_topk_scores,
        1,
        n_sp_devices,
        compare_pcc(scores_pcc_threshold),
        name="pcc_scores",
        broadcast_groups=n_tp_devices,
    )

    all_results = [recall_topk_indices, pcc_logits, pcc_scores]
    for res in all_results:
        res.log_mismatches()
    log_validation_results(
        results=all_results,
        num_dispatch_groups=n_tp_devices,
        dispatch_group_size=n_sp_devices,
        title="Gate Prefill2D Validation",
    )
    merged = ValidationResult.merge(all_results, name="gate_prefill2d")
    merged.assert_passed("Gate prefill2d validation failed")


@pytest.mark.parametrize("gate_model, gate_fallback_mode", REGULAR_GATE_CASES)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
def test_forward_pass(
    gate_model,
    mesh_device,
    num_links,
    topology,
    gate_fallback_mode,
):
    """Gate PCC across model variants and compute modes. The gate picks grouped vs plain top-k from
    n_expert_groups (Kimi / V4 have one group) and sigmoid vs sqrtsoftplus from SCORE_FUNC, so the
    test just passes the model config and the compute mode."""
    random.seed(42)
    torch.manual_seed(42)

    config = TtMoEGateConfig.from_model_cfg(GATE_MODELS[gate_model])
    config.ccl_config["NUM_LINKS"] = num_links
    adjust_shapes_for_testing(config, mesh_device)

    # Real DeepSeek gate weights/input (V3, 256 experts, sigmoid) can't be reshaped to other expert
    # counts or activations, so only attempt the real load for the 256-expert sigmoid path.
    use_real = config.n_routed_experts == 256 and config.score_func == "sigmoid"
    gate_w = _try_load_real_gate_weights(config.n_routed_experts, config.dim) if use_real else None
    if gate_w is None:
        gate_w = create_gate_weights(config.n_routed_experts, config.dim)

    n_sp_devices = mesh_device.shape[0]
    total_seq_len = config.sp_dim * n_sp_devices
    torch_input = _make_gate_input(config, total_seq_len, allow_real_input=use_real)

    # Reference forward pass: V4 (sqrtsoftplus) routes through the activation-parametrized grouped
    # gate (single group -> plain top-k, matching the V4 reference router); V3/Kimi use the V3 gate.
    reference_logits = torch_input @ gate_w["weight"].T
    if config.score_func == "sqrtsoftplus":
        logits_fp32 = torch_input.float() @ gate_w["weight"].float().T
        reference_topk_indices, reference_topk_scores = grouped_gate_golden_act(
            logits_fp32,
            gate_w["e_score_correction_bias"].float(),
            config.route_scale,
            1e-20,
            config.n_expert_groups,
            config.n_expert_groups // config.n_limited_groups,
            config.n_limited_groups,
            config.n_activated_experts,
            score_func=config.score_func,
        )
    else:
        ref_config = SimpleNamespace(
            num_experts_per_tok=config.n_activated_experts,
            n_routed_experts=config.n_routed_experts,
            routed_scaling_factor=config.route_scale,
            scoring_func=config.score_func,
            topk_method="noaux_tc",
            n_group=config.n_expert_groups,
            topk_group=config.n_limited_groups,
            norm_topk_prob=True,
            hidden_size=config.dim,
        )
        reference_model = ReferenceMoEGate(ref_config, use_bitonic_sort=True)
        reference_model.weight.data = gate_w["weight"]
        reference_model.e_score_correction_bias.data = gate_w["e_score_correction_bias"]
        reference_model.eval()
        reference_model.to(torch.bfloat16)
        reference_topk_indices, reference_topk_scores = reference_model.grouped_forward(torch_input.unsqueeze(0))

    tt_input = _shard_gate_input(config, mesh_device, torch_input)

    tt_model = TtMoEGatePrefill(
        config,
        mesh_device,
        weight=gate_w["weight"],
        bias=gate_w["e_score_correction_bias"],
        fallback_mode=gate_fallback_mode,
    )
    tt_topk_scores, tt_topk_indices, tt_logits = tt_model(tt_input)

    # Validation thresholds depend on gate compute mode
    if gate_fallback_mode == GateComputeMode.HOST_ALL:
        recall_threshold = 0.997
        logits_pcc_threshold = 0.997
        scores_pcc_threshold = 0.99
    else:
        recall_threshold = 0.95
        logits_pcc_threshold = 0.997
        scores_pcc_threshold = 0.93

    _validate_gate(
        mesh_device,
        tt_topk_scores,
        tt_topk_indices,
        tt_logits,
        reference_topk_indices,
        reference_topk_scores,
        reference_logits,
        recall_threshold,
        logits_pcc_threshold,
        scores_pcc_threshold,
    )


# Hash gate compute modes: HASH_HOST reuses the reference HashRouter on host and ships results to
# device; HASH_DEVICE runs the fully on-device moe_hash_gate (fused tid2eid[input_ids] lookup).
HASH_GATE_MODES = [
    pytest.param(GateComputeMode.HASH_HOST, id="hash_host"),
    pytest.param(GateComputeMode.HASH_DEVICE, id="hash_device"),
]


@pytest.mark.parametrize("gate_model", ["dsv4_pro", "dsv4_flash"])
@pytest.mark.parametrize("gate_compute_mode", HASH_GATE_MODES)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    MESH_CONFIGS,
    indirect=["mesh_device", "device_params"],
)
def test_hash_gate_forward_pass(
    gate_model,
    gate_compute_mode,
    mesh_device,
    num_links,
    topology,
):
    """DeepSeek-V4 hash-routing gate PCC (host-first HASH_HOST and on-device HASH_DEVICE).

    Expert selection is a static tid2eid[input_ids] lookup (not top-k); the learned gate still
    produces sqrtsoftplus(x@W) scores that are gathered at those experts, normalized and scaled.
    Golden is computed independently from the seeded tid2eid table and synthetic input_ids so the
    test validates the routing math plus the device shipping/sharding (HASH_HOST) or the fully
    on-device fused-lookup op (HASH_DEVICE).
    """
    random.seed(42)
    torch.manual_seed(42)

    config = TtMoEGateConfig.from_model_cfg(GATE_MODELS[gate_model])
    config.ccl_config["NUM_LINKS"] = num_links
    adjust_shapes_for_testing(config, mesh_device)

    gate_w = create_gate_weights(config.n_routed_experts, config.dim)

    n_sp_devices = mesh_device.shape[0]
    total_seq_len = config.sp_dim * n_sp_devices
    torch_input = _make_gate_input(config, total_seq_len, allow_real_input=False)

    # Seeded hash table (tid2eid: vocab -> n_activated experts) and synthetic per-token ids.
    vocab_size = 1024
    tid2eid = torch.randint(0, config.n_routed_experts, (vocab_size, config.n_activated_experts), dtype=torch.long)
    input_ids = torch.randint(0, vocab_size, (total_seq_len,), dtype=torch.long)

    # Golden hash routing: indices from the lookup; weights from sqrtsoftplus(logits) gathered there.
    logits_fp32 = torch_input.float() @ gate_w["weight"].float().T
    scores = score_activation(logits_fp32, config.score_func)
    reference_topk_indices = tid2eid[input_ids]
    reference_topk_scores = scores.gather(1, reference_topk_indices)
    reference_topk_scores = (
        reference_topk_scores / (reference_topk_scores.sum(dim=-1, keepdim=True) + 1e-20) * config.route_scale
    )
    reference_logits = torch_input @ gate_w["weight"].T

    tt_input = _shard_gate_input(config, mesh_device, torch_input)

    tt_model = TtMoEGatePrefill(
        config,
        mesh_device,
        weight=gate_w["weight"],
        bias=gate_w["e_score_correction_bias"],
        fallback_mode=gate_compute_mode,
        hash_table=tid2eid,
    )
    tt_topk_scores, tt_topk_indices, tt_logits = tt_model(tt_input, input_ids=input_ids)

    # Hash indices are a deterministic lookup shared by golden and device, so recall is ~1.0. HASH_HOST
    # only diverges via the bf16 round-trip; HASH_DEVICE computes logits with a device bf16 matmul, so
    # its score PCC is looser (matches the DEVICE_FP32 regular-gate tolerance).
    scores_pcc_threshold = 0.99 if gate_compute_mode == GateComputeMode.HASH_HOST else 0.93
    _validate_gate(
        mesh_device,
        tt_topk_scores,
        tt_topk_indices,
        tt_logits,
        reference_topk_indices,
        reference_topk_scores,
        reference_logits,
        recall_threshold=0.997,
        logits_pcc_threshold=0.997,
        scores_pcc_threshold=scores_pcc_threshold,
    )
