# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Trace-based PCC test for TtPrefillTransformer.

Compares TT output against pre-computed reference tensors from a bit_sculpt
debug trace directory. Loads real intermediate hidden states and KVPE from
safetensors files, runs the TT model on the same input tokens, and reports
per-layer PCC.

TTNN weight cache is built automatically from HF pretrained weights on the
first run (via load_and_compute_layer_by_layer) and reused on subsequent runs.
Set TT_DS_PREFILL_TTNN_CACHE or DEEPSEEK_V3_HF_MODEL to control cache/model paths.
"""

import gc
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import load_and_compute_layer_by_layer, load_debug_trace
from tests.ttnn.utils_for_testing import comp_pcc

ILLIAD_1024_TRACE = Path("/data/ddjekic/bit_sculpt/results/deepseek-r1-0528/debug_trace/illiad_prefill_fa2")
ILLIAD_25024_TRACE = Path("/data/ddjekic/bit_sculpt/results/deepseek-r1-0528/debug_trace/illiad_prefill_fa2_25024")
ABC_1k_PADD_RIGHT_1024 = Path(
    "/data/ddjekic/bit_sculpt/results/deepseek-r1-0528/debug_trace/ABC_1k_prefill_padd_right_1024"
)
ABC_1k_PADD_LEFT_1024 = Path(
    "/data/ddjekic/bit_sculpt/results/deepseek-r1-0528/debug_trace/ABC_1k_prefill_padd_left_1024"
)
LONGBOOK_QA_ENG_25024 = Path("/data/ddjekic/bit_sculpt/results/deepseek-r1-0528/debug_trace/longbook_qa_eng_25088")
PCC_THRESHOLD = 0.97


@pytest.mark.skipif(not is_blackhole(), reason="Requires Blackhole.")
@pytest.mark.parametrize(
    "trace_dir, max_seq_len, pad_left",
    [
        # (ILLIAD_1024_TRACE, None, False),
        (LONGBOOK_QA_ENG_25024, None, False),
        # (ABC_1k_PADD_RIGHT_1024, 1024, False),
        # (ABC_1k_PADD_LEFT_1024, 1024, True),
    ],
    ids=["longbook_25k_qa_eng"],  # "illiad_25024"],
)
@pytest.mark.parametrize(
    "num_layers",
    [
        # pytest.param(61, marks=pytest.mark.skipif(not is_galaxy(), reason="Full model only on Galaxy")),
        61,
    ],
)
@pytest.mark.parametrize(
    "n_routed_experts, capacity_factor, gate_fallback_mode",
    [
        (256, 2, GateComputeMode.HOST_ALL),
        (256, 2, GateComputeMode.DEVICE),
    ],
    ids=["e256_cf32_host", "e256_cf32_device"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (2, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            1,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="mesh-2x4",
        ),
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=DeepSeekV3Config.EMB_SIZE),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_prefill_transformer_from_trace(
    config_only,
    mesh_device,
    device_params,
    trace_dir,
    max_seq_len,
    pad_left,
    num_layers,
    n_routed_experts,
    capacity_factor,
    gate_fallback_mode,
    num_links,
    topology,
    weight_cache_path,
    request,
):
    """
    PCC test using pre-computed reference tensors from a bit_sculpt debug trace.

    Loads real intermediate hidden states (per-layer decoder outputs) and KVPE
    from safetensors files, runs the TT model on the same input tokens, and
    compares per-layer PCC.

    TTNN weight cache is built automatically if missing (requires pretrained
    weights accessible via model_path fixture).
    """
    torch.manual_seed(42)

    profiler.clear()
    profiler.start("total_test_time")

    config = config_only
    trace = load_debug_trace(trace_dir, num_layers=num_layers)

    # Reconstruct padded token sequence to match trace generator input.
    # The trace stores only the real (unpadded) token IDs, but its reference
    # tensors have max_seq_len positions because the HF model was fed padded input.
    n_real_tokens = trace.token_ids.shape[1]

    if max_seq_len is not None and max_seq_len > n_real_tokens:
        tok = request.getfixturevalue("tokenizer")
        pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        pad_len = max_seq_len - n_real_tokens
        padding = torch.full((1, pad_len), pad_token_id, dtype=torch.int64)

        if pad_left:
            token_ids = torch.cat([padding, trace.token_ids], dim=1)
        else:
            token_ids = torch.cat([trace.token_ids, padding], dim=1)

        isl_total = max_seq_len
        logger.info(
            f"Padded {n_real_tokens} tokens to {max_seq_len} "
            f"({'left' if pad_left else 'right'}-padded, pad_token_id={pad_token_id})"
        )
    else:
        token_ids = trace.token_ids
        isl_total = n_real_tokens

    config.max_seq_len = isl_total

    sp_axis = 0
    tp_axis = 1
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[sp_axis]
    tp_factor = mesh_shape[tp_axis]
    emb_dim = config.hidden_size
    isl_per_chip = isl_total // sp_factor

    logger.info(
        f"Trace-based test: trace={trace_dir.name}, isl={isl_total}, "
        f"num_layers={num_layers}, mesh={mesh_shape}, "
        f"n_routed_experts={n_routed_experts}, capacity_factor={capacity_factor}, "
        f"gate_fallback_mode={gate_fallback_mode}"
    )

    orig_num_routed_experts = DeepSeekV3Config.NUM_ROUTED_EXPERTS
    DeepSeekV3Config.NUM_ROUTED_EXPERTS = n_routed_experts

    # --- Weight cache: build if missing, reuse if present ---
    if weight_cache_path is not None:
        rows, cols = mesh_shape
        effective_cache_path = weight_cache_path / f"{rows}x{cols}"
        effective_cache_path.mkdir(parents=True, exist_ok=True)
    else:
        pytest.skip("weight_cache_path is None (pretrained weights unavailable)")

    profiler.start("cache_check")
    experts_per_chip = 256 // (mesh_shape[0] * mesh_shape[1])
    ttnn_cache_complete = TtPrefillTransformer.check_cache_complete(effective_cache_path, num_layers, experts_per_chip)
    profiler.end("cache_check")

    from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import report_and_clear

    report_and_clear()

    logger.info(f"Cache status: TTNN={ttnn_cache_complete}")

    if not ttnn_cache_complete:
        logger.info("TTNN weight cache incomplete — building from HF weights...")
        model_path = request.getfixturevalue("model_path")
        profiler.start("cache_build")
        load_and_compute_layer_by_layer(
            model_path=model_path,
            config=config,
            num_layers=num_layers,
            token_ids=None,
            attention_mask=None,
            compute_reference=False,
            build_ttnn_cache=True,
            weight_cache_path=effective_cache_path,
            mesh_device=mesh_device,
            seq_len=isl_total,
            num_links=num_links,
            topology=topology,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            gate_fallback_mode=gate_fallback_mode,
        )
        gc.collect()
        profiler.end("cache_build")
        logger.info(f"TTNN weight cache built at {effective_cache_path}")
    else:
        logger.info(f"TTNN weight cache found at {effective_cache_path}")

    # --- TT transformer ---
    cache_entries_before = mesh_device.num_program_cache_entries()
    logger.info(f"Program cache entries BEFORE transformer creation: {cache_entries_before}")

    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        state_dict={},
        num_layers=num_layers,
        seq_len=isl_total,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        gate_fallback_mode=gate_fallback_mode,
        dispatch_buffer_capacity_factor=capacity_factor,
        weight_cache_path=effective_cache_path,
    )
    ttnn.ReadDeviceProfiler(mesh_device)
    ttnn.synchronize_device(mesh_device)

    cache_entries_after = mesh_device.num_program_cache_entries()
    logger.info(f"Program cache entries AFTER transformer creation: {cache_entries_after}")
    logger.info(f"Program cache entries ADDED during creation: {cache_entries_after - cache_entries_before}")
    profiler.end("tt_transformer_creation")

    # --- Create external KVPE cache ---
    kvpe_cache_head_dim = config.qk_rope_head_dim + config.kv_lora_rank
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_cache_head_dim,
        mesh_device=mesh_device,
        seq_len=isl_total,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
    )

    # --- Shard token_ids to device ---
    token_ids_reshaped = token_ids.reshape(sp_factor, 1, isl_per_chip)
    tt_tokens = ttnn.from_torch(
        token_ids_reshaped,
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
    )

    # --- Forward ---
    profiler.start("tt_forward")
    logger.info("Running TtPrefillTransformer forward...")
    _, _, tt_intermediates = transformer(
        tt_tokens,
        tt_kvpe_cache,
        actual_isl=isl_total,
        return_intermediates=True,
        read_profiler=True,
    )
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_forward")
    logger.info("Forward pass completed successfully")

    # --- PCC check: TT intermediates vs trace references ---
    profiler.start("pcc_validation")
    pcc_results = []

    for label, tt_host in tt_intermediates.items():
        if label not in trace.ref_snapshots:
            logger.debug(f"Skipping {label} (no trace reference)")
            continue
        ref_host = trace.ref_snapshots[label]
        try:
            _, pcc = comp_pcc(ref_host.float(), tt_host.float())
            logger.debug(f"{label:<20s}  PCC = {pcc:.6f}")
            pcc_results.append((label, pcc))
        except Exception as e:
            logger.error(f"{label:<20s}  PCC comparison failed: {e}")
            pcc_results.append((label, -1.0))

    # --- KVPE PCC check ---
    kv_lora_rank = config.kv_lora_rank
    tt_kvpe_all = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)

    for i, ref_kvpe in enumerate(trace.ref_kvpe_list):
        tt_kvpe_layer = tt_kvpe_all[i : i + 1, :1, :, :]
        label = f"layer_{i}_kvpe"

        # if label == "layer_0_kvpe":
        #     breakpoint()

        try:
            _, kv_pcc = comp_pcc(
                ref_kvpe[:, :, :, :kv_lora_rank].float(),
                tt_kvpe_layer[:, :, :, :kv_lora_rank].float(),
            )
            _, pe_pcc = comp_pcc(
                ref_kvpe[:, :, :, kv_lora_rank:].float(),
                tt_kvpe_layer[:, :, :, kv_lora_rank:].float(),
            )
            logger.info(f"{label:<20s}  KV PCC = {kv_pcc:.6f}, PE PCC = {pe_pcc:.6f}")
            pcc_results.append((f"{label}_kv", kv_pcc))
            pcc_results.append((f"{label}_pe", pe_pcc))
        except Exception as e:
            logger.error(f"{label:<20s}  KVPE PCC comparison failed: {e}")
            pcc_results.append((f"{label}_kv", -1.0))
            pcc_results.append((f"{label}_pe", -1.0))

    profiler.end("pcc_validation")

    # --- Summary table ---
    logger.info(f"\n{'='*50}")
    logger.info(f"{'Stage':<20s}  {'PCC':>10s}  {'Status':>8s}")
    logger.info(f"{'-'*50}")
    failures = []
    for label, pcc in pcc_results:
        status = "PASS" if pcc > PCC_THRESHOLD else ("FAIL" if pcc >= 0 else "ERROR")
        logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
        if pcc <= PCC_THRESHOLD:
            failures.append((label, pcc))
    logger.info(f"{'='*50}")

    profiler.end("total_test_time")

    # --- Timing report ---
    logger.info(f"\n{'='*60}")
    logger.info("Timing Report")
    logger.info(f"{'='*60}")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

    plot_pcc_results(
        pcc_results,
        trace_dir,
        dataset_name=trace_dir.name,
        gate_fallback_mode=str(gate_fallback_mode),
    )

    DeepSeekV3Config.NUM_ROUTED_EXPERTS = orig_num_routed_experts

    if failures:
        pcc_failure_msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
        pytest.fail(f"PCC below {PCC_THRESHOLD} at: {pcc_failure_msg}")


def plot_pcc_results(
    pcc_results: list[tuple[str, float]],
    trace_dir: Path,
    dataset_name: str = "",
    gate_fallback_mode: str = "",
    threshold: float = PCC_THRESHOLD,
    filename: str = "pcc_results.png",
    title_prefix: str = "",
) -> None:
    """Save a bar chart of per-layer and logits PCC values to disk next to the trace directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filtered = [
        (label, pcc)
        for label, pcc in pcc_results
        if label.startswith("layer_")
        and "_kvpe" not in label
        and "_kv" not in label
        and "_pe" not in label
        or label in ("lm_head", "logits")
    ]
    if not filtered:
        logger.warning("No layer/logits PCC results to plot")
        return

    labels = [label for label, _ in filtered]
    values = [pcc for _, pcc in filtered]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.35), 6))
    colors = ["#2ecc71" if v > threshold else "#e74c3c" for v in values]
    ax.bar(range(len(values)), values, color=colors, edgecolor="none", width=0.8)

    ax.axhline(y=threshold, color="orange", linestyle="--", linewidth=1, label=f"threshold={threshold}")
    ax.set_ylabel("PCC")
    ax.set_xlabel("Layer")
    title = f"{dataset_name}    gate_mode = {gate_fallback_mode}"
    if title_prefix:
        title = f"{title_prefix}\n{title}"
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylim(min(0, min(values) - 0.02), 1.01)
    ax.legend(loc="lower left")
    fig.tight_layout()

    out_path = trace_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"PCC plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Trace-vs-host diagnostic tests
# ---------------------------------------------------------------------------

HOST_VS_TRACE_PCC_THRESHOLD = 0.999


def _reconstruct_token_ids(trace, max_seq_len, pad_left, request):
    """Reconstruct (possibly padded) token sequence from trace metadata."""
    n_real_tokens = trace.token_ids.shape[1]

    if max_seq_len is not None and max_seq_len > n_real_tokens:
        tok = request.getfixturevalue("tokenizer")
        pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        pad_len = max_seq_len - n_real_tokens
        padding = torch.full((1, pad_len), pad_token_id, dtype=torch.int64)

        if pad_left:
            token_ids = torch.cat([padding, trace.token_ids], dim=1)
        else:
            token_ids = torch.cat([trace.token_ids, padding], dim=1)

        isl_total = max_seq_len
        logger.info(
            f"Padded {n_real_tokens} tokens to {max_seq_len} "
            f"({'left' if pad_left else 'right'}-padded, pad_token_id={pad_token_id})"
        )
    else:
        token_ids = trace.token_ids
        isl_total = n_real_tokens

    return token_ids, isl_total


# ---- Test A: Weight dequantization comparison ----


@pytest.mark.skipif(not is_blackhole(), reason="Requires Blackhole for ttnn imports.")
@pytest.mark.parametrize(
    "trace_dir",
    [ILLIAD_1024_TRACE],
    ids=["illiad_1024"],
)
@pytest.mark.parametrize("layer_indices", [[0, 1, 2, 3, 30, 60]])
@pytest.mark.timeout(0)
def test_weight_dequantization_pcc(
    config_only,
    trace_dir,
    layer_indices,
    request,
):
    """
    Compare dequantized weights from the TT path (LazyStateDict + dequantize_state_dict)
    against the trace's actual outputs to verify the FP8 dequantization produces
    the same bfloat16 values.

    Loads weights for a few representative layers and compares the embedding
    output and per-layer forward output using matched inputs from the trace.
    A single-layer forward isolates weight differences from error accumulation.
    """
    from copy import deepcopy

    from transformers import DynamicCache
    from transformers.modeling_utils import no_init_weights

    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Model
    from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
    from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

    config = config_only
    trace = load_debug_trace(trace_dir, num_layers=max(layer_indices) + 1)
    model_path = request.getfixturevalue("model_path")

    lazy_sd = LazyStateDict(Path(model_path))
    num_layers = max(layer_indices) + 1

    test_config = deepcopy(config)
    test_config.num_hidden_layers = num_layers
    test_config._attn_implementation = "eager"
    test_config.max_seq_len = trace.token_ids.shape[1]

    pcc_results = []

    # --- Check embedding produces the same output ---
    logger.info("Comparing embedding output...")
    embed_sd = sub_state_dict(lazy_sd, "model.embed_tokens.")
    embed_dequant = dequantize_state_dict(embed_sd, config)

    with no_init_weights():
        hf_model = DeepseekV3Model(test_config)
    hf_model.eval()

    embed_with_prefix = {f"embed_tokens.{k}": v for k, v in embed_dequant.items()}
    hf_model.load_state_dict(embed_with_prefix, strict=False)
    with torch.no_grad():
        h_embed = hf_model.embed_tokens(trace.token_ids).to(torch.bfloat16)

    # The trace's layer_0 input is the embedding output — we can't compare directly
    # since the trace doesn't store it, but we use it as the baseline.
    del embed_with_prefix, embed_dequant
    for k in embed_sd.keys():
        lazy_sd.evict(k)
    gc.collect()

    # --- Per-layer single-layer forward with trace inputs ---
    isl_total = trace.token_ids.shape[1]
    position_ids = torch.arange(isl_total, dtype=torch.long).unsqueeze(0)
    from models.demos.deepseek_v3_d_p.utils.transformer_helpers import get_4d_causal_mask

    attention_mask_4d = get_4d_causal_mask(torch.ones(1, isl_total, dtype=torch.int64), ignore_padding=True)

    for layer_idx in layer_indices:
        logger.info(f"Running single-layer forward for layer {layer_idx}...")

        layer_sd = sub_state_dict(lazy_sd, f"model.layers.{layer_idx}.")
        layer_dequant = dequantize_state_dict(layer_sd, config)

        layer_with_prefix = {f"layers.{layer_idx}.{k}": v for k, v in layer_dequant.items()}
        hf_model.load_state_dict(layer_with_prefix, strict=False)

        # Input: embedding for layer 0, trace's previous layer output otherwise
        if layer_idx == 0:
            layer_input = h_embed
        else:
            layer_input = trace.ref_snapshots[f"layer_{layer_idx - 1}"]

        ref_cache = DynamicCache()
        with torch.no_grad():
            layer_out = hf_model.layers[layer_idx](
                layer_input,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_value=ref_cache,
                use_cache=True,
            )
            host_output = layer_out[0]

        trace_output = trace.ref_snapshots[f"layer_{layer_idx}"]
        label = f"layer_{layer_idx}"

        try:
            _, pcc = comp_pcc(trace_output.float(), host_output.float())
            logger.info(f"{label:<20s}  PCC = {pcc:.6f}")
            pcc_results.append((label, pcc))
        except Exception as e:
            logger.error(f"{label:<20s}  PCC comparison failed: {e}")
            pcc_results.append((label, -1.0))

        # Clean up
        for param in hf_model.layers[layer_idx].parameters():
            param.data = torch.empty(0)
        for k in layer_sd.keys():
            lazy_sd.evict(k)
        del layer_with_prefix, layer_dequant, layer_sd
        gc.collect()

    lazy_sd.close()
    del hf_model
    gc.collect()

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("Single-Layer Forward: Trace Input -> TT-Dequantized Weights -> Compare Trace Output")
    logger.info(f"{'='*60}")
    logger.info(f"{'Layer':<20s}  {'PCC':>10s}  {'Status':>8s}")
    logger.info(f"{'-'*60}")
    failures = []
    for label, pcc in pcc_results:
        status = "PASS" if pcc > HOST_VS_TRACE_PCC_THRESHOLD else ("FAIL" if pcc >= 0 else "ERROR")
        logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
        if pcc <= HOST_VS_TRACE_PCC_THRESHOLD:
            failures.append((label, pcc))
    logger.info(f"{'='*60}")

    plot_pcc_results(
        pcc_results,
        trace_dir,
        dataset_name=trace_dir.name,
        threshold=HOST_VS_TRACE_PCC_THRESHOLD,
        filename="weight_dequant_pcc.png",
        title_prefix="Single-layer PCC (trace input, TT-dequantized weights)",
    )

    if failures:
        pcc_failure_msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
        pytest.fail(
            f"PCC below {HOST_VS_TRACE_PCC_THRESHOLD} at: {pcc_failure_msg}\n"
            "This indicates weight dequantization differences between the TT path "
            "(LazyStateDict + dequantize_state_dict) and the bit_sculpt trace path "
            "(transformers Fp8Dequantize on GPU)."
        )


# ---- Test B: Per-layer matched-input comparison ----


@pytest.mark.skipif(not is_blackhole(), reason="Requires Blackhole for ttnn imports.")
@pytest.mark.parametrize(
    "trace_dir, max_seq_len, pad_left",
    [
        # (ILLIAD_1024_TRACE, None, False),
        # (ABC_1k_PADD_RIGHT_1024, 1024, False),
        (ABC_1k_PADD_LEFT_1024, 1024, True),
    ],
    ids=["abc_1k_padd_left_1024"],
)
@pytest.mark.parametrize("num_layers", [61])
@pytest.mark.timeout(0)
def test_trace_vs_host_matched_inputs(
    config_only,
    trace_dir,
    max_seq_len,
    pad_left,
    num_layers,
    request,
):
    """
    Per-layer comparison feeding trace's decoder output as input to the next host layer.

    Unlike the end-to-end test, this isolates per-layer error from accumulation:
    each layer receives the TRACE's output from the previous layer (not the host's
    diverging output).  Low PCC here means the weights or numerics differ for that
    specific layer; high PCC means the layer is correct and any end-to-end divergence
    is purely from error accumulation.
    """
    from copy import deepcopy

    from transformers import DynamicCache
    from transformers.modeling_utils import no_init_weights

    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Model
    from models.demos.deepseek_v3.utils.config_helpers import sub_state_dict
    from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
    from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict
    from models.demos.deepseek_v3_d_p.utils.transformer_helpers import get_4d_causal_mask

    torch.manual_seed(42)

    profiler.clear()
    profiler.start("total_test_time")

    config = config_only
    trace = load_debug_trace(trace_dir, num_layers=num_layers)

    token_ids, isl_total = _reconstruct_token_ids(trace, max_seq_len, pad_left, request)
    config.max_seq_len = isl_total

    model_path = request.getfixturevalue("model_path")
    lazy_sd = LazyStateDict(Path(model_path))

    test_config = deepcopy(config)
    test_config.num_hidden_layers = num_layers
    test_config._attn_implementation = "eager"

    logger.info("Creating empty HF model structure...")
    with no_init_weights():
        hf_model = DeepseekV3Model(test_config)
    hf_model.eval()

    position_ids = torch.arange(isl_total, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(1, isl_total, dtype=torch.int64)
    attention_mask_4d = get_4d_causal_mask(attention_mask, ignore_padding=True)

    # --- Compute embedding (needed as input to layer 0) ---
    profiler.start("embedding")
    embed_sd = sub_state_dict(lazy_sd, "model.embed_tokens.")
    embed_dequant = dequantize_state_dict(embed_sd, config)
    embed_with_prefix = {f"embed_tokens.{k}": v for k, v in embed_dequant.items()}
    hf_model.load_state_dict(embed_with_prefix, strict=False)
    with torch.no_grad():
        h_embed = hf_model.embed_tokens(token_ids).to(torch.bfloat16)
    hf_model.embed_tokens.weight.data = torch.empty(0)
    del embed_with_prefix, embed_dequant
    for k in embed_sd.keys():
        lazy_sd.evict(k)
    gc.collect()
    profiler.end("embedding")

    # --- Per-layer forward with matched (trace) inputs ---
    profiler.start("per_layer_forward")
    pcc_results = []

    for i in range(num_layers):
        logger.info(f"Layer {i}/{num_layers}: loading weights and running forward...")

        layer_sd = sub_state_dict(lazy_sd, f"model.layers.{i}.")
        layer_dequant = dequantize_state_dict(layer_sd, config)
        layer_with_prefix = {f"layers.{i}.{k}": v for k, v in layer_dequant.items()}
        hf_model.load_state_dict(layer_with_prefix, strict=False)

        # Feed trace's previous layer output (not host's accumulated output)
        if i == 0:
            layer_input = h_embed
        else:
            layer_input = trace.ref_snapshots[f"layer_{i - 1}"]

        # Each layer gets a fresh DynamicCache (we only care about the hidden state output)
        ref_cache = DynamicCache()
        with torch.no_grad():
            layer_out = hf_model.layers[i](
                layer_input,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_value=ref_cache,
                use_cache=True,
            )
            host_output = layer_out[0]

        trace_output = trace.ref_snapshots[f"layer_{i}"]
        label = f"layer_{i}"

        try:
            _, pcc = comp_pcc(trace_output.float(), host_output.float())
            logger.info(f"{label:<20s}  PCC = {pcc:.6f}")
            pcc_results.append((label, pcc))
        except Exception as e:
            logger.error(f"{label:<20s}  PCC comparison failed: {e}")
            pcc_results.append((label, -1.0))

        for param in hf_model.layers[i].parameters():
            param.data = torch.empty(0)
        del layer_with_prefix, layer_dequant, layer_sd, ref_cache
        for k in sub_state_dict(lazy_sd, f"model.layers.{i}.").keys():
            lazy_sd.evict(k)
        gc.collect()

    profiler.end("per_layer_forward")

    lazy_sd.close()
    del hf_model, h_embed
    gc.collect()

    # --- Summary ---
    profiler.end("total_test_time")

    logger.info(f"\n{'='*60}")
    logger.info("Per-Layer Matched-Input: Trace(layer_{i-1}) -> Host(layer_i) vs Trace(layer_i)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Layer':<20s}  {'PCC':>10s}  {'Status':>8s}")
    logger.info(f"{'-'*60}")
    failures = []
    for label, pcc in pcc_results:
        status = "PASS" if pcc > HOST_VS_TRACE_PCC_THRESHOLD else ("FAIL" if pcc >= 0 else "ERROR")
        logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
        if pcc <= HOST_VS_TRACE_PCC_THRESHOLD:
            failures.append((label, pcc))
    logger.info(f"{'='*60}")

    logger.info(f"\n{'='*60}")
    logger.info("Timing Report")
    logger.info(f"{'='*60}")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

    plot_pcc_results(
        pcc_results,
        trace_dir,
        dataset_name=trace_dir.name,
        threshold=HOST_VS_TRACE_PCC_THRESHOLD,
        filename="trace_vs_host_matched_inputs_pcc.png",
        title_prefix="Per-layer matched-input PCC (trace input, TT-dequantized weights)",
    )

    if failures:
        pcc_failure_msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
        pytest.fail(
            f"PCC below {HOST_VS_TRACE_PCC_THRESHOLD} at: {pcc_failure_msg}\n"
            "Per-layer PCC with matched inputs is low — this indicates weight differences "
            "between the TT dequantization path and the bit_sculpt GPU trace."
        )


# ---- Test C: End-to-end with host reference (error accumulation expected) ----


@pytest.mark.skipif(not is_blackhole(), reason="Requires Blackhole for ttnn imports.")
@pytest.mark.parametrize(
    "trace_dir, max_seq_len, pad_left",
    [
        (ILLIAD_1024_TRACE, None, False),
    ],
    ids=["illiad_1024"],
)
@pytest.mark.parametrize("num_layers", [61])
@pytest.mark.timeout(0)
def test_trace_vs_host_reference(
    config_only,
    trace_dir,
    max_seq_len,
    pad_left,
    num_layers,
    request,
):
    """
    End-to-end comparison: host CPU reference (eager attention, TT-dequantized
    weights) vs bit_sculpt GPU trace (SDPA attention, transformers Fp8Dequantize).

    This test runs the host model end-to-end (each layer feeds into the next)
    so error accumulates across layers.  It is a diagnostic tool — see
    test_trace_vs_host_matched_inputs for the per-layer isolated comparison
    and test_weight_dequantization_pcc for the weight-level comparison.
    """
    torch.manual_seed(42)

    profiler.clear()
    profiler.start("total_test_time")

    config = config_only
    trace = load_debug_trace(trace_dir, num_layers=num_layers)

    token_ids, isl_total = _reconstruct_token_ids(trace, max_seq_len, pad_left, request)
    config.max_seq_len = isl_total
    attention_mask = torch.ones(1, isl_total, dtype=torch.int64)

    logger.info(
        f"Trace-vs-host end-to-end: trace={trace_dir.name}, isl={isl_total}, "
        f"num_layers={num_layers}, pad_left={pad_left}"
    )

    model_path = request.getfixturevalue("model_path")
    profiler.start("host_reference")
    logger.info(f"Computing host reference for {num_layers} layers, isl={isl_total}...")

    host_result = load_and_compute_layer_by_layer(
        model_path=model_path,
        config=config,
        num_layers=num_layers,
        token_ids=token_ids,
        attention_mask=attention_mask,
        compute_reference=True,
        build_ttnn_cache=False,
    )
    profiler.end("host_reference")

    # host_result.ref_snapshots: [embed, layer_0, ..., layer_{N-1}, norm]
    profiler.start("pcc_validation")
    pcc_results = []

    for i in range(num_layers):
        label = f"layer_{i}"
        trace_tensor = trace.ref_snapshots[label]
        host_tensor = host_result.ref_snapshots[i + 1]

        try:
            _, pcc = comp_pcc(trace_tensor.float(), host_tensor.float())
            logger.info(f"{label:<20s}  PCC = {pcc:.6f}")
            pcc_results.append((label, pcc))
        except Exception as e:
            logger.error(f"{label:<20s}  PCC comparison failed: {e}")
            pcc_results.append((label, -1.0))

    profiler.end("pcc_validation")

    # --- Summary ---
    logger.info(f"\n{'='*60}")
    logger.info("End-to-End: Trace (GPU) vs Host Reference (CPU)")
    logger.info(f"{'='*60}")
    logger.info(f"{'Layer':<20s}  {'PCC':>10s}  {'Status':>8s}")
    logger.info(f"{'-'*60}")
    failures = []
    for label, pcc in pcc_results:
        status = "PASS" if pcc > HOST_VS_TRACE_PCC_THRESHOLD else ("FAIL" if pcc >= 0 else "ERROR")
        logger.info(f"{label:<20s}  {pcc:>10.6f}  {status:>8s}")
        if pcc <= HOST_VS_TRACE_PCC_THRESHOLD:
            failures.append((label, pcc))
    logger.info(f"{'='*60}")

    profiler.end("total_test_time")

    logger.info(f"\n{'='*60}")
    logger.info("Timing Report")
    logger.info(f"{'='*60}")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

    plot_pcc_results(
        pcc_results,
        trace_dir,
        dataset_name=trace_dir.name,
        threshold=HOST_VS_TRACE_PCC_THRESHOLD,
        filename="trace_vs_host_pcc.png",
        title_prefix="End-to-end: Trace (GPU/SDPA) vs Host Reference (CPU/eager)",
    )

    if failures:
        pcc_failure_msg = "; ".join(f"{label}: {pcc:.6f}" for label, pcc in failures)
        pytest.fail(f"PCC below {HOST_VS_TRACE_PCC_THRESHOLD} at: {pcc_failure_msg}")
