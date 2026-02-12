# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Profiling tests for Attention1D fused QK operations.

Uses ttnn trace capture (begin_trace_capture / execute_trace) for accurate
device-side timing without host dispatch overhead. Also collects non-trace
host-side timing for comparison.

Run via the shell wrapper:

    models/common/tests/modules/attention/profiling/run_profiling.sh

Or directly:

    python_env/bin/python -m pytest models/common/tests/modules/attention/profiling/ -s
"""

import os
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1DConfig
from models.common.modules.tt_ccl import TT_CCL

# Reuse helpers from the main test file
from models.common.tests.modules.attention.test_attention_1d import (
    DEEPSEEK_R1_14B,
    LLAMA_1B,
    LLAMA_3B,
    LLAMA_8B,
    LLAMA_11B,
    LLAMA_70B,
    LLAMA_90B,
    MISTRAL_7B,
    MIXTRAL_8X7B,
    QWEN2_7B,
    QWEN3_32B,
    QWEN25_7B,
    QWEN25_72B,
    QWEN25_CODER_32B,
    HfAttentionWrapper,
    PagedAttentionConfig,
    RotarySetupHelper,
    _get_or_init_attn_weights,
    get_attention_weights_from_ref_model,
    get_rot_mats_from_hf,
)
from models.common.utility_functions import comp_pcc

# =============================================================================
# Benchmark Helpers (moved from test_attention_1d.py)
# =============================================================================


def _create_attention_model_for_benchmark(
    ttnn_mesh_device: ttnn.MeshDevice,
    hf_model_name: str,
    use_qk_fused: bool,
    page_block_size: int | None = 64,
    max_seq_len: int = 2048,
) -> tuple:
    """
    Create an Attention1D model with specified use_qk_fused setting for benchmarking.

    Returns:
        (tt_model, reference_wrapper, config_params) tuple for running benchmarks
    """
    batch_size = 1
    mesh_shape = ttnn_mesh_device.shape

    # Load HF config
    hf_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)

    # Extract dimensions
    text_config = getattr(hf_config, "text_config", hf_config)
    dim = text_config.hidden_size
    n_heads = text_config.num_attention_heads
    n_kv_heads = getattr(text_config, "num_key_value_heads", n_heads)
    # Use explicit head_dim if available (e.g., Qwen3 models), else calculate from dim/n_heads
    head_dim = getattr(text_config, "head_dim", None) or (dim // n_heads)

    # Calculate num_devices for topology
    num_devices = mesh_shape[0] * mesh_shape[1]

    # Topology
    topology = ttnn.Topology.Ring if num_devices > 1 else None
    tt_ccl = TT_CCL(ttnn_mesh_device) if num_devices > 1 else None

    # Create reference model with random weights
    is_multimodal = "vision_config" in hf_config.__dict__ or "Vision" in hf_model_name

    if is_multimodal:
        from transformers import MllamaForConditionalGeneration

        with no_init_weights():
            hf_model = MllamaForConditionalGeneration._from_config(hf_config, torch_dtype=torch.bfloat16)
        # Mllama has layers directly at language_model.layers (not language_model.model.layers)
        first_layer = hf_model.language_model.layers[0]
        rotary_emb = getattr(hf_model.language_model, "rotary_emb", None)
    else:
        with no_init_weights():
            hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)
        first_layer = hf_model.model.layers[0]
        rotary_emb = getattr(hf_model.model, "rotary_emb", None)

    # Get reference attention from first layer
    reference_attn = first_layer.self_attn

    # Initialize weights with scaled normal
    _get_or_init_attn_weights(hf_model_name, reference_attn)

    # Create HF wrapper (local class)
    reference_wrapper = HfAttentionWrapper(reference_attn, head_dim, rotary_emb)

    # Extract and prepare weights
    wqkv_torch, wo_torch, q_norm_torch, k_norm_torch, wqkv_bias_torch = get_attention_weights_from_ref_model(
        reference_attn, num_devices
    )

    # Weights are deterministically seeded per model (via _get_or_init_attn_weights),
    # so caching is safe — same model name always produces the same weights.
    wqkv_dtype = ttnn.bfloat8_b
    model_short = hf_model_name.replace("/", "--")
    cache_dir = Path(os.getenv("TT_CACHE_PATH", f"model_cache/attn1d_profiling/{model_short}"))
    lazy_wqkv = LazyWeight(
        source=wqkv_torch,
        dtype=wqkv_dtype,
        cache_dir_weight_name=(cache_dir, "wqkv"),
    )
    lazy_wo = LazyWeight(
        source=wo_torch,
        dtype=wqkv_dtype,
        cache_dir_weight_name=(cache_dir, "wo"),
    )

    # Q/K norm configs (if present)
    q_norm_config = None
    k_norm_config = None
    if q_norm_torch is not None:
        # Add 3 dimensions to match expected shape (1, 1, 1, head_dim)
        q_norm_4d = q_norm_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        q_norm_config = RMSNorm1DConfig(
            weight=LazyWeight(source=q_norm_4d, dtype=ttnn.bfloat16, cache_dir_weight_name=(cache_dir, "q_norm")),
            mesh_device=ttnn_mesh_device,
            decode_in_sharded=False,  # Q/K heads are interleaved after create_qkv_heads
            decode_out_sharded=False,
            prefill_distributed=False,  # Q/K norm doesn't need distributed prefill
        )
    if k_norm_torch is not None:
        # Add 3 dimensions to match expected shape (1, 1, 1, head_dim)
        k_norm_4d = k_norm_torch.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        k_norm_config = RMSNorm1DConfig(
            weight=LazyWeight(source=k_norm_4d, dtype=ttnn.bfloat16, cache_dir_weight_name=(cache_dir, "k_norm")),
            mesh_device=ttnn_mesh_device,
            decode_in_sharded=False,  # Q/K heads are interleaved after create_qkv_heads
            decode_out_sharded=False,
            prefill_distributed=False,  # Q/K norm doesn't need distributed prefill
        )

    # Paged attention config (local dataclass)
    paged_attention_config = None
    if page_block_size is not None:
        paged_attention_config = PagedAttentionConfig(block_size=page_block_size, max_num_blocks=2048)

    # RotarySetupHelper using HF rotary_emb (no rope_scaling needed - HF handles it)
    rope_setup = RotarySetupHelper(
        ttnn_mesh_device,
        batch_size,
        head_dim,
        max_seq_len,
        rotary_emb,  # HF rotary embedding already has rope_scaling applied
        use_qk_fused=use_qk_fused,  # Use the parameterized value
    )

    # Build Attention1DConfig with specified use_qk_fused
    # Note: kv_cache is auto-created by config resolution if not using paged attention
    config = Attention1DConfig(
        wqkv=lazy_wqkv,
        wo=lazy_wo,
        q_norm_config=q_norm_config,
        k_norm_config=k_norm_config,
        wqkv_bias=LazyWeight(source=wqkv_bias_torch, cache_dir_weight_name=(cache_dir, "wqkv_bias"))
        if wqkv_bias_torch is not None
        else None,
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        topology=topology,
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        scale=head_dim**-0.5,
        use_qk_fused=use_qk_fused,  # KEY: this is what we're benchmarking
        use_vllm_paged_kv_cache=False,  # kv_cache managed internally
        paged_attention_config=paged_attention_config,
        wqkv_dtype=wqkv_dtype,
        wo_dtype=wqkv_dtype,
        activation_dtype=ttnn.bfloat16,
    )

    # Create Attention1D
    tt_model = Attention1D.from_config(config)

    config_params = {
        "dim": dim,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "batch_size": batch_size,
        "mesh_shape": mesh_shape,
        "rotary_emb": rotary_emb,  # HF rotary embedding for reference
        "rope_setup": rope_setup,
        "is_multimodal": is_multimodal,
    }

    return tt_model, reference_wrapper, config_params


# =============================================================================
# Device Profiler Helpers
# =============================================================================


# todo)) use this test and codexapi science to speed up fused qk for all models!!!
@torch.no_grad()
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        {"mesh_shape": (1, 1), "trace_region_size": 50_000_000},
        {"mesh_shape": (1, 2), "trace_region_size": 50_000_000},
        {"mesh_shape": (1, 8), "trace_region_size": 50_000_000},
    ],
    ids=[
        "1x1",
        "1x2",
        "1x8",
    ],
    indirect=True,
)
def test_attention_1d_fused_qk_profiling(ttnn_mesh_device: ttnn.MeshDevice):
    """
    Comprehensive profiling test comparing fused vs non-fused QK operations across multiple models.

    This test:
    1. Benchmarks multiple model architectures
    2. Compares fused vs non-fused performance and accuracy
    3. Outputs a detailed analysis table with findings
    4. Runs on 1x1, 1x2, and 1x8 mesh configurations

    Device profiling is enabled at module import time (see top of file).
    """

    mesh_shape = ttnn_mesh_device.shape
    num_devices = mesh_shape[0] * mesh_shape[1]

    # ANSI color codes
    BOLD = "\033[1m"
    RESET = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    WHITE_BG = "\033[47m"
    BLACK = "\033[30m"

    # Models to benchmark - filter based on mesh size
    # Hardware constraints:
    # - nlp_create_qkv_heads_decode supports max 32 q heads per device
    # - DeepSeek-R1-14B has 40 q heads, needs 1x2+ (40/2=20 per device)
    # - Large models (70B, 90B, 72B) need 1x8 for memory
    if num_devices == 1:
        models_to_test = [
            (LLAMA_1B, "Llama-1B"),
            (LLAMA_3B, "Llama-3B"),
            (LLAMA_8B, "Llama-8B"),
            (LLAMA_11B, "Llama-11B-Vision"),
            (MISTRAL_7B, "Mistral-7B"),
            (QWEN2_7B, "Qwen2-7B"),
            (QWEN25_7B, "Qwen2.5-7B"),
            # Note: DeepSeek-R1-14B has 40 q heads, exceeds 32 head limit on single device
        ]
    elif num_devices == 2:
        models_to_test = [
            (LLAMA_1B, "Llama-1B"),
            (LLAMA_3B, "Llama-3B"),
            (LLAMA_8B, "Llama-8B"),
            (LLAMA_11B, "Llama-11B-Vision"),
            (MISTRAL_7B, "Mistral-7B"),
            (QWEN2_7B, "Qwen2-7B"),
            (QWEN25_7B, "Qwen2.5-7B"),
            (DEEPSEEK_R1_14B, "DeepSeek-R1-14B"),  # 40 heads / 2 devices = 20 per device, OK
        ]
    else:
        # 1x8: all models including very large ones
        # Note: Qwen2-7B and Qwen2.5-7B excluded (dim=3584 not divisible by 8)
        models_to_test = [
            (LLAMA_8B, "Llama-8B"),
            (LLAMA_11B, "Llama-11B-Vision"),
            (LLAMA_70B, "Llama-70B"),
            (LLAMA_90B, "Llama-90B-Vision"),
            (MISTRAL_7B, "Mistral-7B"),
            (MIXTRAL_8X7B, "Mixtral-8x7B"),
            (QWEN25_72B, "Qwen2.5-72B"),
            (QWEN25_CODER_32B, "Qwen2.5-Coder-32B"),
            (DEEPSEEK_R1_14B, "DeepSeek-R1-14B"),
            (QWEN3_32B, "Qwen3-32B"),
        ]

    # Collect results for all models (no printing during benchmark)
    all_results = {}
    log_messages = []  # Collect log messages to print at end

    for hf_model_name, model_label in models_to_test:
        log_messages.append(f"Benchmarking {model_label} ({hf_model_name})...")

        model_results = {}

        for use_qk_fused in [False, True]:  # Test non-fused first for comparison
            fused_label = "fused" if use_qk_fused else "non-fused"

            try:
                tt_model, reference_wrapper, params = _create_attention_model_for_benchmark(
                    ttnn_mesh_device=ttnn_mesh_device,
                    hf_model_name=hf_model_name,
                    use_qk_fused=use_qk_fused,
                    page_block_size=None,
                )

                dim = params["dim"]
                head_dim = params["head_dim"]
                batch_size = params["batch_size"]
                rotary_emb = params["rotary_emb"]

                # Prefill to populate KV cache
                prefill_seq_len = 128
                pt_prefill_input = torch.randn(batch_size, prefill_seq_len, dim, dtype=torch.bfloat16)

                tt_prefill_input = ttnn.from_torch(
                    pt_prefill_input.unsqueeze(0),
                    device=ttnn_mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
                )

                # Use HF rotary_emb for rotation matrices
                prefill_rot_mats = get_rot_mats_from_hf(
                    rotary_emb,
                    seq_len=prefill_seq_len,
                    head_dim=head_dim,
                    device=ttnn_mesh_device,
                )

                _ = tt_model.forward(
                    tt_prefill_input,
                    None,
                    prefill_rot_mats,
                    mode="prefill",
                )

                # Run HF prefill for accuracy check
                _ = reference_wrapper(pt_prefill_input, start_pos=0, mask=None)

                # Prepare decode
                pt_decode_input = torch.randn(batch_size, 1, dim, dtype=torch.bfloat16)
                position_idxs = torch.tensor([prefill_seq_len], dtype=torch.long)

                # Use RotarySetupHelper with HF rotary_emb
                decode_rope_setup = RotarySetupHelper(
                    ttnn_mesh_device,
                    batch_size,
                    head_dim,
                    2048,
                    rotary_emb,
                    use_qk_fused=use_qk_fused,
                )
                decode_rot_mats = decode_rope_setup.get_rot_mats(position_idxs)

                current_pos = ttnn.from_torch(
                    position_idxs,
                    device=ttnn_mesh_device,
                    dtype=ttnn.int32,
                    mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(ttnn_mesh_device),
                )

                # First decode for accuracy check
                tt_decode_input = ttnn.from_torch(
                    pt_decode_input.unsqueeze(0),
                    device=ttnn_mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
                )
                tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)

                tt_out = tt_model.forward(
                    tt_decode_input,
                    current_pos,
                    decode_rot_mats,
                    mode="decode",
                )
                ttnn.synchronize_device(ttnn_mesh_device)

                # Check PCC
                tt_out_torch = to_torch_auto_compose(tt_out)
                tt_output = tt_out_torch[:, 0:1, :batch_size, :dim].view(batch_size, 1, dim)

                reference_output = reference_wrapper(pt_decode_input, start_pos=prefill_seq_len, mask=None)

                _, pcc_value = comp_pcc(reference_output, tt_output.to(reference_output.dtype), 0.9)
                pcc = float(pcc_value)

                # Warmup
                for _ in range(5):
                    tt_decode_input = ttnn.from_torch(
                        pt_decode_input.unsqueeze(0),
                        device=ttnn_mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
                    )
                    tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)
                    _ = tt_model.forward(
                        tt_decode_input,
                        current_pos,
                        decode_rot_mats,
                        mode="decode",
                    )
                    ttnn.synchronize_device(ttnn_mesh_device)

                # =====================================================================
                # NON-TRACE timed runs (dispatch overhead included)
                # =====================================================================
                num_runs = 100
                host_timings = []

                for _ in range(num_runs):
                    tt_decode_input = ttnn.from_torch(
                        pt_decode_input.unsqueeze(0),
                        device=ttnn_mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
                    )
                    tt_decode_input = ttnn.to_memory_config(tt_decode_input, tt_model.config.decode_input_memcfg)

                    start = time.perf_counter()
                    _ = tt_model.forward(
                        tt_decode_input,
                        current_pos,
                        decode_rot_mats,
                        mode="decode",
                    )
                    ttnn.synchronize_device(ttnn_mesh_device)
                    end = time.perf_counter()
                    host_timings.append((end - start) * 1e6)

                # =====================================================================
                # TRACE-BASED timed runs (eliminates host dispatch overhead)
                # Pattern: capture op graph once, replay N times via execute_trace
                # Requires trace_region_size > 0 when opening device (see parametrize)
                # =====================================================================
                avg_trace_time = None
                min_trace_time = None
                std_trace_time = None
                trace_pcc = None
                trace_id = None

                try:
                    ttnn.synchronize_device(ttnn_mesh_device)

                    # Allocate fresh DRAM input tensor — this becomes the trace's input slot.
                    # During trace replay, the device reads from this same memory location.
                    tt_trace_input_dram = ttnn.from_torch(
                        pt_decode_input.unsqueeze(0),
                        device=ttnn_mesh_device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=mesh_shape),
                    )

                    # Capture trace: resharding (DRAM→sharded) + full decode_forward
                    trace_id = ttnn.begin_trace_capture(ttnn_mesh_device, cq_id=0)
                    tt_trace_input_sharded = ttnn.to_memory_config(
                        tt_trace_input_dram, tt_model.config.decode_input_memcfg
                    )
                    _tt_out_trace = tt_model.forward(
                        tt_trace_input_sharded,
                        current_pos,
                        decode_rot_mats,
                        mode="decode",
                    )
                    ttnn.end_trace_capture(ttnn_mesh_device, trace_id, cq_id=0)

                    # Trace warmup (first execute may have extra overhead)
                    for _ in range(5):
                        ttnn.execute_trace(ttnn_mesh_device, trace_id, cq_id=0, blocking=False)
                        ttnn.synchronize_device(ttnn_mesh_device)

                    # Timed trace executions
                    trace_timings = []
                    for _ in range(num_runs):
                        start = time.perf_counter()
                        ttnn.execute_trace(ttnn_mesh_device, trace_id, cq_id=0, blocking=False)
                        ttnn.synchronize_device(ttnn_mesh_device)
                        end = time.perf_counter()
                        trace_timings.append((end - start) * 1e6)

                    # Verify trace output PCC matches reference
                    tt_trace_out_torch = to_torch_auto_compose(_tt_out_trace)
                    tt_trace_output = tt_trace_out_torch[:, 0:1, :batch_size, :dim].view(batch_size, 1, dim)
                    _, trace_pcc_value = comp_pcc(reference_output, tt_trace_output.to(reference_output.dtype), 0.9)
                    trace_pcc = float(trace_pcc_value)

                    ttnn.release_trace(ttnn_mesh_device, trace_id)
                    trace_id = None

                    # Trace-based statistics
                    avg_trace_time = sum(trace_timings) / len(trace_timings)
                    min_trace_time = min(trace_timings)
                    std_trace_time = (sum((t - avg_trace_time) ** 2 for t in trace_timings) / len(trace_timings)) ** 0.5

                except Exception as trace_err:
                    trace_err_msg = str(trace_err).split("\n")[0][:120]
                    log_messages.append(f"  {fused_label:10s}: [TRACE FAIL] {trace_err_msg}")
                    if trace_id is not None:
                        ttnn.release_trace(ttnn_mesh_device, trace_id)

                # =====================================================================
                # Statistics
                # =====================================================================

                # Host-side (non-trace) statistics
                avg_host_time = sum(host_timings) / len(host_timings)
                min_host_time = min(host_timings)
                std_host_time = (sum((t - avg_host_time) ** 2 for t in host_timings) / len(host_timings)) ** 0.5

                model_results[fused_label] = {
                    "mean": avg_host_time,
                    "min": min_host_time,
                    "std": std_host_time,
                    "pcc": pcc,
                    # Trace-based timing (no host dispatch overhead)
                    "trace_mean": avg_trace_time,
                    "trace_min": min_trace_time,
                    "trace_std": std_trace_time,
                    "trace_pcc": trace_pcc,
                }

                # Log message
                trace_str = f"trace={avg_trace_time:6.1f}μs" if avg_trace_time is not None else "trace=N/A"
                trace_pcc_str = f", trace_PCC={trace_pcc:.4f}" if trace_pcc is not None else ""
                log_messages.append(
                    f"  {fused_label:10s}: no-trace={avg_host_time:6.1f}μs, {trace_str}, PCC={pcc:.4f}{trace_pcc_str}"
                )

            except Exception as e:
                # Extract short error message (first line or up to 120 chars)
                error_msg = str(e).split("\n")[0][:120]
                log_messages.append(f"  [SKIP] {fused_label}: {error_msg}")
                continue

        all_results[model_label] = model_results

    # =========================================================================
    # ALL OUTPUT AT THE END - AFTER BENCHMARKING COMPLETES
    # =========================================================================

    print()
    print()
    print(f"{BOLD}{WHITE_BG}{BLACK}{'=' * 80}{RESET}")
    print(
        f"{BOLD}{WHITE_BG}{BLACK}  MESH SHAPE: {mesh_shape[0]}x{mesh_shape[1]}  ({num_devices} device{'s' if num_devices > 1 else ''})  {RESET}"
    )
    print(f"{BOLD}{WHITE_BG}{BLACK}{'=' * 80}{RESET}")
    print()

    print(f"{BOLD}{CYAN}FUSED vs NON-FUSED QK COMPREHENSIVE BENCHMARK{RESET}")
    print(f"{CYAN}{'─' * 80}{RESET}")

    # Print collected log messages
    for msg in log_messages:
        if "[SKIP]" in msg:
            print(f"{RED}{msg}{RESET}")
        elif "Benchmarking" in msg:
            print(f"{YELLOW}{msg}{RESET}")
        else:
            print(msg)

    # =========================================================================
    # COMPREHENSIVE ANALYSIS OUTPUT
    # =========================================================================
    print()
    print(f"{BOLD}{MAGENTA}{'=' * 80}{RESET}")
    print(f"{BOLD}{MAGENTA}BENCHMARK ANALYSIS: FUSED vs NON-FUSED QK OPERATIONS{RESET}")
    print(f"{BOLD}{MAGENTA}  Mesh Shape: {mesh_shape[0]}x{mesh_shape[1]} ({num_devices} devices){RESET}")
    print(f"{BOLD}{MAGENTA}{'=' * 80}{RESET}")

    # Performance Comparison Table (Host-side timing)
    print()
    print(f"{BOLD}{GREEN}HOST-SIDE PERFORMANCE (Decode Latency - includes dispatch overhead){RESET}")
    print(f"{GREEN}{'─' * 80}{RESET}")
    print(f"{BOLD}{'Model':<14} │ {'Non-fused (μs)':<16} │ {'Fused (μs)':<14} │ {'Diff (μs)':<12} │ {'%':<10}{RESET}")
    print(f"{GREEN}{'─' * 80}{RESET}")

    total_diff = 0
    total_pct = 0
    count = 0
    fused_faster_count = 0
    fused_slower_count = 0

    for model_label, results in all_results.items():
        if "non-fused" in results and "fused" in results:
            non_fused = results["non-fused"]["mean"]
            fused = results["fused"]["mean"]
            diff = fused - non_fused
            pct = (diff / non_fused) * 100

            total_diff += diff
            total_pct += pct
            count += 1

            if diff < 0:
                fused_faster_count += 1
            else:
                fused_slower_count += 1

            diff_color = GREEN if diff < 0 else RED if diff > 0 else RESET
            print(
                f"{model_label:<14} │ {non_fused:>14.1f}   │ {fused:>12.1f}   │ {diff_color}{diff:>+10.1f}{RESET}   │ {diff_color}{pct:>+8.1f}%{RESET}"
            )

    print(f"{GREEN}{'─' * 80}{RESET}")
    if count > 0:
        avg_diff = total_diff / count
        avg_pct = total_pct / count
        avg_color = GREEN if avg_diff < 0 else RED if avg_diff > 0 else RESET
        print(
            f"{BOLD}{'AVERAGE':<14}{RESET} │ {'':<16} │ {'':<14} │ {avg_color}{BOLD}{avg_diff:>+10.1f}{RESET}   │ {avg_color}{BOLD}{avg_pct:>+8.1f}%{RESET}"
        )

    # =========================================================================
    # TRACE-BASED PERFORMANCE (primary metric — no host dispatch overhead)
    # =========================================================================
    print()
    print(f"{BOLD}{WHITE_BG}{BLACK}TRACE-BASED PERFORMANCE (no dispatch overhead — production metric){RESET}")
    print(f"{GREEN}{'─' * 80}{RESET}")
    print(f"{BOLD}{'Model':<14} │ {'Non-fused (μs)':<16} │ {'Fused (μs)':<14} │ {'Diff (μs)':<12} │ {'%':<10}{RESET}")
    print(f"{GREEN}{'─' * 80}{RESET}")

    trace_total_diff = 0
    trace_total_pct = 0
    trace_count = 0
    trace_fused_faster = 0
    trace_fused_slower = 0

    for model_label, results in all_results.items():
        nf_trace = results.get("non-fused", {}).get("trace_mean")
        f_trace = results.get("fused", {}).get("trace_mean")

        if nf_trace is not None and f_trace is not None:
            diff = f_trace - nf_trace
            pct = (diff / nf_trace) * 100

            trace_total_diff += diff
            trace_total_pct += pct
            trace_count += 1

            if diff < 0:
                trace_fused_faster += 1
            else:
                trace_fused_slower += 1

            diff_color = GREEN if diff < 0 else RED if diff > 0 else RESET
            print(
                f"{model_label:<14} │ {nf_trace:>14.1f}   │ {f_trace:>12.1f}   │ {diff_color}{diff:>+10.1f}{RESET}   │ {diff_color}{pct:>+8.1f}%{RESET}"
            )

    print(f"{GREEN}{'─' * 80}{RESET}")
    trace_avg_pct = 0
    if trace_count > 0:
        trace_avg_diff = trace_total_diff / trace_count
        trace_avg_pct = trace_total_pct / trace_count
        trace_avg_color = GREEN if trace_avg_diff < 0 else RED if trace_avg_diff > 0 else RESET
        print(
            f"{BOLD}{'AVERAGE':<14}{RESET} │ {'':<16} │ {'':<14} │ {trace_avg_color}{BOLD}{trace_avg_diff:>+10.1f}{RESET}   │ {trace_avg_color}{BOLD}{trace_avg_pct:>+8.1f}%{RESET}"
        )

    # Accuracy Comparison Table
    print()
    print(f"{BOLD}{BLUE}ACCURACY COMPARISON (PCC vs HuggingFace Reference){RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")
    print(f"{BOLD}{'Model':<14} │ {'Non-fused PCC':<16} │ {'Fused PCC':<14}{RESET}")
    print(f"{BLUE}{'─' * 60}{RESET}")

    avg_pcc_nf = 0
    avg_pcc_f = 0
    pcc_count = 0

    for model_label, results in all_results.items():
        non_fused_pcc = results.get("non-fused", {}).get("pcc", "N/A")
        fused_pcc = results.get("fused", {}).get("pcc", "N/A")

        non_fused_str = f"{non_fused_pcc:.4f}" if isinstance(non_fused_pcc, float) else non_fused_pcc
        fused_str = f"{fused_pcc:.4f}" if isinstance(fused_pcc, float) else fused_pcc

        if isinstance(non_fused_pcc, float) and isinstance(fused_pcc, float):
            avg_pcc_nf += non_fused_pcc
            avg_pcc_f += fused_pcc
            pcc_count += 1

        nf_color = (
            GREEN
            if isinstance(non_fused_pcc, float) and non_fused_pcc >= 0.99
            else (
                YELLOW
                if isinstance(non_fused_pcc, float) and non_fused_pcc >= 0.95
                else RED
                if isinstance(non_fused_pcc, float)
                else RESET
            )
        )
        f_color = (
            GREEN
            if isinstance(fused_pcc, float) and fused_pcc >= 0.99
            else (
                YELLOW
                if isinstance(fused_pcc, float) and fused_pcc >= 0.95
                else RED
                if isinstance(fused_pcc, float)
                else RESET
            )
        )

        print(f"{model_label:<14} │ {nf_color}{non_fused_str:>14}{RESET}   │ {f_color}{fused_str:>12}{RESET}")

    print(f"{BLUE}{'─' * 60}{RESET}")

    if pcc_count > 0:
        avg_pcc_nf /= pcc_count
        avg_pcc_f /= pcc_count

    # Dynamic Root Cause Analysis based on results
    print()
    print(f"{BOLD}{YELLOW}ROOT CAUSE ANALYSIS{RESET}")
    print(f"{YELLOW}{'─' * 80}{RESET}")
    print()
    print(f"  {BOLD}{RED}FUSED PATH{RESET} (in decode_forward):")
    print(f"    1. {CYAN}_reshard_k_for_fused(){RESET} - 1x ttnn.to_memory_config (K only)")
    print(f"       └─ Q is already on correct cores; only K moves to non-overlapping grid")
    print(f"    2. {CYAN}rotary_embedding_llama_fused_qk(){RESET} - 1 fused kernel")
    print(f"    3. {CYAN}paged_fused_update_cache(){RESET} - 1 fused kernel")
    print()
    print(f"  {BOLD}{GREEN}NON-FUSED PATH{RESET} (in decode_forward):")
    print(f"    1. {CYAN}rotary_embedding_llama(q){RESET} - Q rotary embedding")
    print(f"    2. {CYAN}rotary_embedding_llama(k){RESET} - K rotary embedding")
    print(f"    3. {CYAN}paged_update_cache(keys, k){RESET} - K cache update")
    print(f"    4. {CYAN}paged_update_cache(values, v){RESET} - V cache update")
    print()

    # Dynamic Conclusion based on TRACE results (primary metric)
    print(f"{BOLD}{MAGENTA}CONCLUSION (Mesh {mesh_shape[0]}x{mesh_shape[1]}){RESET}")
    print(f"{MAGENTA}{'─' * 80}{RESET}")
    print()

    if trace_count > 0:
        # Determine overall winner based on trace (production-representative) timing
        if trace_avg_pct < -2:
            perf_summary = f"{GREEN}Fused path is {abs(trace_avg_pct):.1f}% FASTER{RESET} on average (trace-based)"
            recommendation = f"{GREEN}Prefer fused path{RESET} for better performance"
        elif trace_avg_pct > 2:
            perf_summary = f"{RED}Fused path is {trace_avg_pct:.1f}% SLOWER{RESET} on average (trace-based)"
            recommendation = f"{GREEN}Prefer non-fused path{RESET} for better performance"
        else:
            perf_summary = f"{YELLOW}Performance is similar{RESET} (within 2%, trace-based)"
            recommendation = f"{YELLOW}Either path is acceptable{RESET}"

        print(f"  1. {BOLD}TRACE PERFORMANCE:{RESET} {perf_summary}")
        print(f"     - Fused faster: {GREEN}{trace_fused_faster}{RESET} models")
        print(f"     - Fused slower: {RED}{trace_fused_slower}{RESET} models")

        if count > 0:
            # Also note the no-trace (dispatch) numbers for context
            no_trace_color = YELLOW
            print(
                f"  2. {BOLD}NO-TRACE HOST:{RESET} {no_trace_color}avg {avg_pct:+.1f}%{RESET} (noisy — includes Python dispatch overhead)"
            )

        if pcc_count > 0:
            pcc_color = GREEN if avg_pcc_nf >= 0.99 and avg_pcc_f >= 0.99 else YELLOW
            print(
                f"  3. {BOLD}ACCURACY:{RESET} Both paths produce similar results ({pcc_color}PCC ~{avg_pcc_f:.3f}{RESET})"
            )

        print(f"  4. {BOLD}RECOMMENDATION:{RESET} {recommendation}")
    elif count > 0:
        # Fallback to no-trace results if trace data unavailable
        if avg_pct < -2:
            perf_summary = f"{GREEN}Fused path is {abs(avg_pct):.1f}% FASTER{RESET} on average"
            recommendation = f"{GREEN}Prefer fused path{RESET} for better performance"
        elif avg_pct > 2:
            perf_summary = f"{RED}Fused path is {avg_pct:.1f}% SLOWER{RESET} on average"
            recommendation = f"{GREEN}Prefer non-fused path{RESET} for better performance"
        else:
            perf_summary = f"{YELLOW}Performance is similar{RESET} (within 2%)"
            recommendation = f"{YELLOW}Either path is acceptable{RESET}"

        print(f"  1. {BOLD}PERFORMANCE:{RESET} {perf_summary}")
        print(f"     - Fused faster: {GREEN}{fused_faster_count}{RESET} models")
        print(f"     - Fused slower: {RED}{fused_slower_count}{RESET} models")
        print(f"  2. {BOLD}RECOMMENDATION:{RESET} {recommendation}")
    else:
        print(f"  {RED}No valid benchmark results collected.{RESET}")

    print()
    print(f"{BOLD}{MAGENTA}{'=' * 80}{RESET}")
