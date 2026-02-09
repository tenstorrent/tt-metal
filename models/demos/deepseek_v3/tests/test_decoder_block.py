# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path
from typing import Dict

import pytest
import torch
from loguru import logger
from tracy import signpost
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3DecoderLayer
from models.demos.deepseek_v3.tests.pytest_utils import (
    build_expanded_test_ids,
    expand_test_cases_with_position_ids_ranges,
)

# Import synthetic weight generators from existing tests
from models.demos.deepseek_v3.tests.test_mla import generate_synthetic_mla_weights
from models.demos.deepseek_v3.tests.test_mlp import generate_synthetic_mlp_weights
from models.demos.deepseek_v3.tests.test_moe_experts import generate_synthetic_moe_expert_weights
from models.demos.deepseek_v3.tests.test_moe_gate import ExpertDistribution, generate_synthetic_moe_weights
from models.demos.deepseek_v3.tests.test_rms_norm import generate_synthetic_rms_norm_weights
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d import DecoderBlock2D
from models.demos.deepseek_v3.tt.decoder_block.decoder_block_2d_base import DecoderBlock2DBase
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import MoEDecoderBlock2D
from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    dequantize_state_dict,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_cache_from_torch,
    run_reference_with_attention,
    torch_cache_from_transformers_single_layer,
)
from models.perf.benchmarking_utils import BenchmarkProfiler


def generate_synthetic_decoder_block_weights(
    hf_config,
    layer_idx: int,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic weights for a complete decoder block.

    This function combines synthetic weights from all sub-components to create
    a complete decoder block with realistic weight distributions.

    Args:
        hf_config: HuggingFace model configuration
        layer_idx: Layer index (determines if it's MoE or regular MLP)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing weight tensors for all decoder block components
    """
    weights = {}

    # Generate MLA (attention) weights
    mla_weights = generate_synthetic_mla_weights(hf_config, seed=seed + layer_idx)
    for key, value in mla_weights.items():
        weights[f"self_attn.{key}"] = value

    # Determine if this layer uses MoE or regular MLP
    # Based on DeepSeek V3 architecture:
    # - First few layers (< first_k_dense_replace) use regular MLP
    # - After that, layers at moe_layer_freq intervals use MoE
    use_moe = (
        hf_config.n_routed_experts is not None
        and layer_idx >= hf_config.first_k_dense_replace
        and layer_idx % hf_config.moe_layer_freq == 0
    )

    if use_moe:
        # Generate MoE weights (gate + experts + shared expert)
        # MoE gate weights
        moe_gate_weights = generate_synthetic_moe_weights(
            hf_config, distribution=ExpertDistribution.UNIFORM, seed=seed + layer_idx + 1000
        )
        for key, value in moe_gate_weights.items():
            weights[f"mlp.gate.{key}"] = value

        # MoE expert weights (all 256 experts)
        moe_expert_weights = generate_synthetic_moe_expert_weights(hf_config, seed=seed + layer_idx + 2000)
        for key, value in moe_expert_weights.items():
            # The expert weights already have "experts.N." prefix, just add "mlp."
            weights[f"mlp.{key}"] = value

        # Shared expert weights (uses same structure as regular MLP but different size)
        shared_expert_weights = generate_synthetic_mlp_weights(
            hf_config, mlp_type="shared_expert", seed=seed + layer_idx + 3000
        )
        for key, value in shared_expert_weights.items():
            weights[f"mlp.shared_experts.{key}"] = value
    else:
        # Generate regular MLP weights
        # Determine MLP type based on layer index
        if layer_idx < hf_config.first_k_dense_replace:
            # First few layers use non-expert MLP
            mlp_type = "non_expert"
        else:
            # Later non-MoE layers use regular MLP
            mlp_type = "regular"

        mlp_weights = generate_synthetic_mlp_weights(hf_config, mlp_type=mlp_type, seed=seed + layer_idx + 1000)
        for key, value in mlp_weights.items():
            weights[f"mlp.{key}"] = value

    # Generate RMS norm weights
    # Input layernorm
    input_norm_weights = generate_synthetic_rms_norm_weights(
        norm_type="input_layernorm", hidden_size=hf_config.hidden_size, seed=seed + layer_idx + 4000
    )
    for key, value in input_norm_weights.items():
        weights[f"input_layernorm.{key}"] = value

    # Post-attention layernorm
    post_norm_weights = generate_synthetic_rms_norm_weights(
        norm_type="post_attention_layernorm", hidden_size=hf_config.hidden_size, seed=seed + layer_idx + 5000
    )
    for key, value in post_norm_weights.items():
        weights[f"post_attention_layernorm.{key}"] = value

    return weights


def generate_reference_io(
    model_path: Path,
    module_path: str | None,
    hf_config: PretrainedConfig,
    layer_idx: int,
    seq_len: int,
    batch_size: int,
    mode: str,
    state_dict: dict[str, torch.Tensor],
    decode_position_id: int | None = None,
    use_synthetic_weights: bool = False,
):
    reference_model = DeepseekV3DecoderLayer(hf_config, layer_idx=layer_idx).eval().to(torch.bfloat16)

    if use_synthetic_weights:
        # Use synthetic weights that match real DeepSeek V3 distributions
        synthetic_weights = generate_synthetic_decoder_block_weights(hf_config, layer_idx=layer_idx, seed=42)
        # Dequantize synthetic weights for the reference model
        state_dict = dequantize_state_dict(synthetic_weights, hf_config)
        reference_model.load_state_dict(state_dict)
        # Keep the quantized version for TTNN
        state_dict = synthetic_weights
    elif module_path is not None:
        state_dict = sub_state_dict(state_dict, module_path + ".")
        reference_model.load_state_dict(dequantize_state_dict(state_dict, hf_config))
    else:
        # This needs to be disabled as deterministic way to quantize weights is not supported
        torch.use_deterministic_algorithms(False)
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )

    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    position_ids = None
    if mode == "prefill":
        position_ids_or_seq_lens = torch.tensor([seq_len])
    else:
        if decode_position_id is None:
            position_ids = position_ids_or_seq_lens = torch.randint(
                0, hf_config.max_seq_len - 1, (batch_size,), dtype=torch.long
            )
        else:
            if not isinstance(decode_position_id, int):
                raise ValueError(f"decode_position_id must be int or None, got {type(decode_position_id)}")
            if not (0 <= decode_position_id < hf_config.max_seq_len):
                raise ValueError(
                    f"decode_position_id must be in [0, {hf_config.max_seq_len - 1}], got {decode_position_id}"
                )
            position_ids = position_ids_or_seq_lens = torch.ones(batch_size, dtype=torch.long) * decode_position_id
    reference_output, input_cache, output_cache = run_reference_with_attention(
        reference_model, torch_input, position_ids_or_seq_lens, layer_idx, hf_config, mode, False
    )
    input_cache = torch_cache_from_transformers_single_layer(input_cache, layer_idx)
    output_cache = torch_cache_from_transformers_single_layer(output_cache, layer_idx)
    if mode == "decode":
        torch_input = torch_input.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        reference_output = reference_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
    return state_dict, position_ids, torch_input, reference_output, input_cache, output_cache


def run_decoder_with_trace(
    mesh_device,
    forward_fn,
    forward_args,
    num_iters=20,
    warmup_iters=5,
    profiler=BenchmarkProfiler(),
):
    """Run decoder block with trace mode for on-device performance measurement.

    Args:
        mesh_device: The mesh device to run on
        forward_fn: The forward function to call (prefill or decode)
        forward_args: Arguments to pass to the forward function
        num_iters: Number of iterations to run
        warmup_iters: Number of warmup iterations
        profiler: BenchmarkProfiler instance

    Returns:
        The output tensor from the forward pass
    """
    # Compile Run
    logger.info("Compiling decoder block")
    tt_output = forward_fn(*forward_args)
    ttnn.synchronize_device(mesh_device)

    # Capture trace
    logger.info("Capturing trace")
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(num_iters):
        tt_output = forward_fn(*forward_args)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    # Capture warmup trace if needed
    trace_id_warmup = None
    if warmup_iters > 0:
        logger.info("Capturing warmup trace")
        trace_id_warmup = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(warmup_iters):
            tt_output = forward_fn(*forward_args)
        ttnn.end_trace_capture(mesh_device, trace_id_warmup, cq_id=0)
        ttnn.synchronize_device(mesh_device)

    # Run the trace
    logger.info(f"Starting trace performance test (warmup: {warmup_iters}, main: {num_iters} iterations)")

    # Run warmup trace
    if warmup_iters > 0:
        profiler.start("decoder-block-trace-warmup")
        ttnn.execute_trace(mesh_device, trace_id_warmup, blocking=False)
        ttnn.release_trace(mesh_device, trace_id_warmup)
        ttnn.synchronize_device(mesh_device)
        profiler.end("decoder-block-trace-warmup")

    # Run main trace with signposts for device-side measurement
    profiler.start("decoder-block-trace")
    signpost("start")
    ttnn.execute_trace(mesh_device, trace_id, blocking=False)
    ttnn.release_trace(mesh_device, trace_id)
    ttnn.synchronize_device(mesh_device)
    signpost("stop")
    profiler.end("decoder-block-trace")

    # Calculate and report timing
    warmup_time = profiler.get_duration("decoder-block-trace-warmup") if warmup_iters > 0 else 0
    total_time = profiler.get_duration("decoder-block-trace")
    effective_time = total_time - warmup_time
    effective_iters = num_iters - warmup_iters

    logger.info(f"On-device performance results:")
    logger.info(f"  Total time: {total_time:.6f} s")
    logger.info(f"  Effective time (excluding warmup): {effective_time:.6f} s")
    logger.info(f"  Time per iteration: {effective_time / effective_iters * 1000:.3f} ms")
    logger.info(f"  Iterations per second: {effective_iters / effective_time:.2f}")

    return tt_output


def run_test_forward_pass_decoder2d(
    DecoderBlockClass: type[DecoderBlock2DBase],
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
    force_recalculate_weight_config,
    state_dict,
    decode_position_ids: int | None = None,
    use_synthetic_weights: bool = False,
    trace_mode: bool = False,
    num_iters: int = 20,
    warmup_iters: int = 5,
    profiler: BenchmarkProfiler = None,
):
    # Check params
    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports a batch size of 1"
        batch_size = batch_size_per_row
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"
        batch_size = batch_size_per_row * mesh_device.shape[0]

    state_dict, position_ids, torch_input, reference_output, input_cache, _ = generate_reference_io(
        model_path,
        module_path,
        hf_config_short,
        reference_layer_idx,
        seq_len,
        batch_size,
        mode,
        state_dict,
        decode_position_ids,
        use_synthetic_weights,
    )

    # Set up page config
    logger.info("Setting up model configs")
    user_id = None if mode == "decode" else torch.randint(0, USERS_PER_ROW * mesh_device.shape[0], ()).item()
    paged_config = MLA1D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache, tuple(mesh_device.shape), paged_config, user_id
    )

    # Set up model config
    weight_config = get_test_weight_config(
        DecoderBlockClass,
        hf_config_short,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate_weight_config,
    )
    model_config = get_model_config(DecoderBlockClass, mode, hf_config_short, mesh_device)
    model_state = DecoderBlockClass.create_state(
        hf_config_short,
        paged_config,
        mesh_device,
        ccl,
        mla_cache=paged_input_cache,
    )
    model_shared_state = DecoderBlockClass.create_shared_state(hf_config_short, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    position_ids_tensor = (
        ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            dtype=ttnn.int32,
        )
        if mode == "decode"
        else None
    )

    tt_page_table = MLA2D.create_page_table(
        page_table=torch_page_table, paged_config=paged_config, mesh_device=mesh_device
    )
    rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])

    # Forward pass
    logger.info("Running TTNN forward pass")

    if trace_mode:
        # Use trace execution for performance measurement
        if profiler is None:
            profiler = BenchmarkProfiler()

        if mode == "prefill":
            forward_fn = lambda *args: DecoderBlockClass.forward_prefill(*args)
            forward_args = (tt_input, user_id, run_config, rope_tensors, tt_page_table)
        else:
            forward_fn = lambda *args: DecoderBlockClass.forward_decode(*args)
            forward_args = (tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_table)

        tt_output = run_decoder_with_trace(
            mesh_device,
            forward_fn,
            forward_args,
            num_iters=num_iters,
            warmup_iters=warmup_iters,
            profiler=profiler,
        )
    else:
        # Regular forward pass without trace
        if mode == "prefill":
            tt_output = DecoderBlockClass.forward_prefill(tt_input, user_id, run_config, rope_tensors, tt_page_table)
        else:
            tt_output = DecoderBlockClass.forward_decode(
                tt_input, position_ids_tensor, run_config, rope_tensors, tt_page_table
            )

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape)
    )

    # Check output PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.9899)


# Base test cases - ranges will be expanded into individual test cases
# see documentation for expand_test_cases_with_position_ids_ranges for more details
BASE_TEST_CASES = [
    # mode, seq_len, batch_size_per_row, decode_position_ids
    ("decode", 1, USERS_PER_ROW, None),
] + [
    ("prefill", seq_len, 1, None)
    if seq_len == 128
    else pytest.param(
        "prefill",
        seq_len,
        1,
        None,
        marks=pytest.mark.skip(
            f"Skipping prefilling with seq_len={seq_len} since this would cause us to exceed our available CI workload time"
        ),
    )
    for seq_len in PREFILL_SEQ_LENS
]
EXPANDED_TEST_CASES = expand_test_cases_with_position_ids_ranges(BASE_TEST_CASES)
EXPANDED_TEST_IDS = build_expanded_test_ids(EXPANDED_TEST_CASES)


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 17068032},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "use_synthetic_weights",
    [True, False],
)
@pytest.mark.parametrize(
    "trace_mode",
    [False, True],
)
@pytest.mark.parametrize(
    "DecoderBlockClass, module_path, reference_layer_idx, test_closure",
    [
        pytest.param(
            DecoderBlock2D,
            None,
            0,
            run_test_forward_pass_decoder2d,
            marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
        ),
        pytest.param(
            MoEDecoderBlock2D,
            None,
            3,
            run_test_forward_pass_decoder2d,
            marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
        ),
        pytest.param(
            DecoderBlock2D,
            "model.layers.0",
            0,
            run_test_forward_pass_decoder2d,
            marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
        ),
        pytest.param(
            MoEDecoderBlock2D,
            "model.layers.3",
            3,
            run_test_forward_pass_decoder2d,
            marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"]),
        ),
    ],
)
@pytest.mark.parametrize(
    "num_iters, warmup_iters",
    [
        (20, 5),
    ],
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row, decode_position_ids",
    EXPANDED_TEST_CASES,
    ids=EXPANDED_TEST_IDS,
)
def test_forward_pass(
    use_synthetic_weights,
    trace_mode,
    DecoderBlockClass: type[DecoderBlock2DBase],
    module_path,
    reference_layer_idx,
    mode,
    seq_len,
    batch_size_per_row,
    decode_position_ids,
    num_iters,
    warmup_iters,
    hf_config_short,
    cache_path,
    mesh_device,
    model_path,
    ccl,
    force_recalculate_weight_config,
    test_closure,
    set_deterministic_env,
    state_dict,
):
    if mode != "decode":
        decode_position_ids = None

    # Create profiler for trace mode
    profiler = BenchmarkProfiler() if trace_mode else None

    test_closure(
        DecoderBlockClass,
        module_path,
        reference_layer_idx,
        mode,
        seq_len,
        batch_size_per_row,
        hf_config_short,
        cache_path,
        mesh_device,
        model_path,
        ccl,
        force_recalculate_weight_config,
        state_dict,
        decode_position_ids,
        use_synthetic_weights,
        trace_mode,
        num_iters,
        warmup_iters,
        profiler,
    )


if __name__ == "__main__":
    pytest.main([__file__])
