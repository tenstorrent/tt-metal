# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Dict

import pytest
import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3Attention
from models.demos.deepseek_v3.tests.pytest_utils import (
    build_expanded_test_ids,
    expand_test_cases_with_position_ids_ranges,
)
from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW, dequantize, sub_state_dict
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    dequantize_state_dict,
    get_model_config,
    get_rope_tensors,
    get_test_weight_config,
    paged_cache_from_torch,
    run_reference_with_attention,
    torch_cache_from_paged,
    torch_cache_from_transformers_single_layer,
)

PCC_REQUIRED = 0.99
PCC_REQUIRED_KVPE = 0.999


def generate_synthetic_mla_weights(
    hf_config,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Generate synthetic weights for MLA layer that resemble real trained weights.

    This function generates weights with distributions similar to real DeepSeek V3 MLA weights
    based on empirical analysis of the actual model weights from HuggingFace.

    Args:
        hf_config: HuggingFace model configuration
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing weight tensors for all MLA components
    """
    torch.manual_seed(seed)

    # Extract dimensions from config
    hidden_size = hf_config.hidden_size
    num_heads = hf_config.num_attention_heads
    kv_lora_rank = hf_config.kv_lora_rank
    q_lora_rank = hf_config.q_lora_rank
    qk_nope_head_dim = hf_config.qk_nope_head_dim
    qk_rope_head_dim = hf_config.qk_rope_head_dim
    v_head_dim = hf_config.v_head_dim
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # Weight statistics based on empirical analysis of real DeepSeek V3 weights:
    # Analysis shows different layers use different initialization scales

    def create_weight_with_std(shape, target_std):
        """Create weight with specific standard deviation."""
        return torch.randn(shape) * target_std

    # Generate weights for each component
    weights = {}

    # Projection weights (will be quantized to fp8)
    # Create weights that when dequantized will have the target std
    # FP8 E4M3FN has max value of 448, so we create weights in FP8 range
    # and set inv_scale such that dequantized weights have correct std

    def create_quantized_weight(shape, target_std_after_dequant):
        """Create FP8 weight and scale that produces target std after dequantization."""
        # Create weights in FP8 range with reasonable distribution
        fp8_std = 30.0  # Use a good portion of FP8 range
        weight_fp8 = (torch.randn(shape) * fp8_std).to(torch.float8_e4m3fn)

        # Calculate inv_scale to achieve target std after dequantization
        # After dequant: weight_float = weight_fp8 * inv_scale
        # We want: std(weight_float) = target_std_after_dequant
        # So: inv_scale ≈ target_std_after_dequant / fp8_std
        inv_scale = target_std_after_dequant / fp8_std
        return weight_fp8, inv_scale

    # These std values are from actual DeepSeek V3 weight analysis:
    q_a_weight, q_a_scale_base = create_quantized_weight((q_lora_rank, hidden_size), 0.0187)
    q_b_weight, q_b_scale_base = create_quantized_weight((num_heads * q_head_dim, q_lora_rank), 0.0085)
    kv_a_weight, kv_a_scale_base = create_quantized_weight((kv_lora_rank + qk_rope_head_dim, hidden_size), 0.0390)
    kv_b_weight, kv_b_scale_base = create_quantized_weight(
        (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank), 0.0049
    )
    o_weight, o_scale_base = create_quantized_weight((hidden_size, num_heads * v_head_dim), 0.0059)

    weights["q_a_proj.weight"] = q_a_weight
    weights["q_b_proj.weight"] = q_b_weight
    weights["kv_a_proj_with_mqa.weight"] = kv_a_weight
    weights["kv_b_proj.weight"] = kv_b_weight
    weights["o_proj.weight"] = o_weight

    # Generate scale tensors for quantization
    # These represent the inverse scale factors used in quantization
    # For block_shape (128, 128), we need scale tensors with shape matching the number of blocks
    block_size = 128

    def create_scale_tensor_from_base(weight_shape, base_inv_scale):
        """Create scale tensor matching the blocked dimensions of the weight."""
        # Calculate number of blocks in each dimension
        num_blocks_0 = (weight_shape[0] + block_size - 1) // block_size
        num_blocks_1 = (weight_shape[1] + block_size - 1) // block_size
        # Create scale tensor with small variation around the base value
        # This ensures consistency between reference and TTNN implementations
        scale = torch.ones(num_blocks_0, num_blocks_1) * base_inv_scale
        # Add small variation (±10%) to simulate block-wise quantization
        scale = scale * (1.0 + torch.randn(num_blocks_0, num_blocks_1) * 0.1)
        # Ensure positive values
        scale = torch.clamp(scale, min=1e-6)
        return scale

    # Use the inv_scale values calculated from the quantized weights
    # This ensures consistency between reference model and TTNN
    weights["q_a_proj.weight_scale_inv"] = create_scale_tensor_from_base((q_lora_rank, hidden_size), q_a_scale_base)
    weights["q_b_proj.weight_scale_inv"] = create_scale_tensor_from_base(
        (num_heads * q_head_dim, q_lora_rank), q_b_scale_base
    )
    weights["kv_a_proj_with_mqa.weight_scale_inv"] = create_scale_tensor_from_base(
        (kv_lora_rank + qk_rope_head_dim, hidden_size), kv_a_scale_base
    )
    weights["kv_b_proj.weight_scale_inv"] = create_scale_tensor_from_base(
        (num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank), kv_b_scale_base
    )
    weights["o_proj.weight_scale_inv"] = create_scale_tensor_from_base(
        (hidden_size, num_heads * v_head_dim), o_scale_base
    )

    # Layer norm weights (not quantized)
    # Based on real DeepSeek V3 analysis:
    # q_a_layernorm: mean=0.444, std=0.083 (NOT centered at 1.0!)
    # kv_a_layernorm: mean=0.007, std=0.0076 (close to 0, NOT 1!)
    weights["q_a_layernorm.weight"] = (torch.randn(q_lora_rank) * 0.083 + 0.444).to(torch.bfloat16)
    # For kv_a_layernorm, the size should be kv_lora_rank (512)
    weights["kv_a_layernorm.weight"] = (torch.randn(kv_lora_rank) * 0.0076 + 0.007).to(torch.bfloat16)

    # Ensure proper dtypes
    for key in weights:
        if "scale_inv" in key:
            weights[key] = weights[key].to(torch.float32)

    return weights


def get_cache_on_host(tt_cache: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """
    Get the KVPE cache on the host from the TTNN cache.

    Args:
        tt_cache (ttnn.Tensor): The TTNN cache tensor.
        mesh_device (ttnn.MeshDevice): The mesh device to get the cache from.
        row_idx (int | None): The row index to get the cache from. If None, gets the entire cache.
    Returns:
        torch.Tensor: The cache tensor on the host.
    """
    return ttnn.to_torch(
        tt_cache,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )


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
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate reference input/output for testing.

    Args:
        decode_position_id: Configuration for position_ids generation (only used in decode mode):
            - None: Generate random position_ids in range [0, max_seq_len - 1)
            - int: Use this specific position for all batches
        use_synthetic_weights: If True, use synthetic weights instead of loading from state_dict
    """
    reference_model = DeepseekV3Attention(hf_config, layer_idx=layer_idx).eval().to(torch.bfloat16)

    if use_synthetic_weights:
        # Generate synthetic weights
        synthetic_weights = generate_synthetic_mla_weights(hf_config)

        # Dequantize synthetic weights for the reference model
        # The reference model expects float weights, not quantized FP8
        dequantized_weights = {}
        block_shape = hf_config.quantization_config["weight_block_size"]

        for name, tensor in synthetic_weights.items():
            if name.endswith("_scale_inv"):
                continue  # Skip scale tensors
            elif tensor.dtype == torch.float8_e4m3fn:
                # Dequantize FP8 weights using their scale tensors
                scale_name = name + "_scale_inv"
                if scale_name in synthetic_weights:
                    scale_tensor = synthetic_weights[scale_name]
                    # Dequantize using the scale
                    dequantized_tensor = dequantize(tensor, scale_tensor, block_shape)
                    dequantized_weights[name] = dequantized_tensor.to(torch.bfloat16)
                else:
                    dequantized_weights[name] = tensor.to(torch.bfloat16)
            else:
                # Keep non-quantized weights as-is (e.g., layernorm weights)
                dequantized_weights[name] = tensor

        # Load dequantized weights into reference model
        reference_model.load_state_dict(dequantized_weights, strict=False)

        # Return synthetic weights (with quantization) for TTNN weight conversion
        # The MLA1D.convert_weights function expects quantized weights with scale_inv
        state_dict = synthetic_weights
    elif module_path is None:
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(),
            block_shape=hf_config.quantization_config["weight_block_size"],
        )
    else:
        state_dict = sub_state_dict(state_dict, module_path + ".")
        dequantized_state_dict = dequantize_state_dict(state_dict, hf_config)
        reference_model.load_state_dict(dequantized_state_dict)

    # Use deterministic input for reproducibility, especially important for synthetic weights
    torch.manual_seed(42)  # Fixed seed for input generation
    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)
    position_ids = None
    if mode == "prefill":
        position_ids_or_seq_lens = torch.tensor([seq_len])
    else:
        # Handle decode_position_ids for decode mode
        if decode_position_id is None:
            # Generate random position_ids
            position_ids = position_ids_or_seq_lens = torch.randint(
                0, hf_config.max_seq_len - 1, (batch_size,), dtype=torch.long
            )
        else:
            # Must be an int, use that value for all batches
            if not isinstance(decode_position_id, int):
                raise ValueError(f"decode_position_id must be int or None, got {type(decode_position_id)}")
            if not (0 <= decode_position_id < hf_config.max_seq_len):
                raise ValueError(
                    f"decode_position_id must be in [0, {hf_config.max_seq_len - 1}], got {decode_position_id}"
                )
            position_ids = position_ids_or_seq_lens = torch.ones(batch_size, dtype=torch.long) * decode_position_id
    reference_output, input_cache, output_cache = run_reference_with_attention(
        reference_model, torch_input, position_ids_or_seq_lens, layer_idx, hf_config, mode, zeroed_cache=True
    )
    input_cache = torch_cache_from_transformers_single_layer(input_cache, layer_idx)
    output_cache = torch_cache_from_transformers_single_layer(output_cache, layer_idx)
    if mode == "decode":
        torch_input = torch_input.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
        reference_output = reference_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
    return state_dict, position_ids, torch_input, reference_output, input_cache, output_cache


def check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED):
    passing, pcc = comp_pcc(reference_output, tt_output_torch, pcc_required)
    logger.info(f"PCC: {pcc}")
    return passing


def check_cache_matches(tt_cache, reference_cache, lora_rank, pcc_required=PCC_REQUIRED_KVPE):
    # Check correct shapes
    dims_correct = True
    if len(tt_cache.shape) != 4:
        logger.error(
            f"Expected tt_cache of shape (bsz, num_heads, seq_len, head_dim + rope_head_dim), got {tt_cache.shape=}"
        )
        dims_correct = False
    if len(reference_cache.shape) != 4:
        logger.error(
            f"Expected reference_cache of shape (bsz, num_heads, seq_len, head_dim + rope_head_dim), got {reference_cache.shape=}"
        )
        dims_correct = False
    if not dims_correct:
        return False

    if tt_cache.shape != reference_cache.shape:
        logger.error(f"Cache shape mismatch: {tt_cache.shape=} vs {reference_cache.shape=}")
        return False

    if torch.all(tt_cache == 0) != torch.all(reference_cache == 0):
        if torch.all(tt_cache == 0):
            logger.error("TTNN cache is all zeros, but reference cache is not.")
        else:
            logger.error("Reference cache is all zeros, but TTNN cache is not.")
        return False

    # Check PCC
    tt_cache_kv = tt_cache[..., :lora_rank]
    tt_cache_pe = tt_cache[..., lora_rank:]

    ref_cache_kv = reference_cache[..., :lora_rank]  # [bsz, _, _, head_dim]
    ref_cache_pe = reference_cache[..., lora_rank:]  # [bsz, _, _, rope_head_dim]

    kv_passing, kv_pcc_message = comp_pcc(ref_cache_kv, tt_cache_kv, pcc_required)
    pe_passing, pe_pcc_message = comp_pcc(ref_cache_pe, tt_cache_pe, pcc_required)

    logger.info(f"Cache KV PCC: {kv_pcc_message}")
    logger.info(f"Cache PE PCC: {pe_pcc_message}")
    return kv_passing and pe_passing


def check_cache_unchanged(tt_cache, exclusion_area: tuple[slice, slice, slice, slice]):
    if len(tt_cache.shape) != 4:
        logger.error(
            f"Expected tt_cache of shape (bsz, num_heads, seq_len, head_dim + rope_head_dim), got {tt_cache.shape=}"
        )
        return False

    tt_cache = tt_cache.clone()
    tt_cache[exclusion_area] = 0  # Zero out the area we don't want to check
    all_zeros = True
    for user_id in range(tt_cache.shape[0]):
        if not torch.all(tt_cache[user_id] == 0):
            logger.error(
                f"Cache for user {user_id % USERS_PER_ROW} row {user_id // USERS_PER_ROW} not empty outside exclusion area"
            )
            all_zeros = False

    return all_zeros


def run_test_forward_pass_mla2d(
    layer_idx,
    mode,
    seq_len,
    batch_size_per_row,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    force_recalculate_weight_config,
    state_dict,
    decode_position_ids: int | None = None,
    use_synthetic_weights: bool = False,
):
    # Check params
    if mode == "prefill":
        assert batch_size_per_row == 1, "Prefill only supports a batch size of 1"
        batch_size = batch_size_per_row
    else:
        assert mode == "decode" and seq_len == 1, "Decode only supports a sequence length of 1"
        batch_size = batch_size_per_row * mesh_device.shape[0]

    # Get reference IO
    logger.info("Setting up reference IO")
    state_dict, position_ids, torch_input, reference_output, input_cache, output_cache = generate_reference_io(
        model_path,
        module_path,
        hf_config_short,
        layer_idx,
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
    paged_config = MLA2D.get_valid_paged_config(hf_config_short.max_seq_len, USERS_PER_ROW, mesh_device.shape[1])
    paged_input_cache, torch_page_table = paged_cache_from_torch(
        input_cache, tuple(mesh_device.shape), paged_config, user_id
    )

    # Set up model config - force recalculation when using synthetic weights
    weight_config = get_test_weight_config(
        MLA2D,
        hf_config_short,
        (state_dict,),
        cache_path,
        mesh_device,
        use_synthetic_weights or force_recalculate_weight_config,
    )
    model_config = get_model_config(MLA2D, mode, hf_config_short, mesh_device)
    model_state = MLA2D.create_state(hf_config_short, paged_config, mesh_device, ccl, paged_input_cache)
    run_config = create_run_config(model_config, weight_config, model_state)

    # Set up ttnn inputs
    logger.info("Setting up model inputs")
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=run_config["input_memory_config"],
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
    tt_rope_tensors = get_rope_tensors(hf_config_short, batch_size_per_row, seq_len, position_ids, mesh_device)

    # Forward pass
    logger.info("Running TTNN forward pass")

    if mode == "prefill":
        tt_output = MLA2D.forward_prefill(tt_input, user_id, run_config, tt_rope_tensors, tt_page_table)
    else:
        tt_output = MLA2D.forward_decode(tt_input, position_ids_tensor, run_config, tt_rope_tensors, tt_page_table)

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape)
    ).reshape(
        -1, seq_len, hf_config_short.hidden_size
    )  # Concatenate all batches together

    # Check PCC
    tt_cache = torch_cache_from_paged(
        get_cache_on_host(run_config["mla1d"]["kvpe_cache"], mesh_device),
        torch_page_table,
        mesh_device.get_num_devices(),
    )
    if mode == "prefill":
        assert (
            check_output_matches(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED)
            and check_cache_matches(
                tt_cache[user_id : user_id + 1, :, :seq_len],
                output_cache,
                hf_config_short.kv_lora_rank,
                pcc_required=PCC_REQUIRED_KVPE,
            )
            and check_cache_unchanged(
                tt_cache, (slice(user_id, user_id + 1), slice(None), slice(None, seq_len), slice(None))
            )
        ), f"MLA output for prefill {seq_len=} {user_id=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"
    else:
        assert check_output_matches(
            tt_output_torch, reference_output, pcc_required=PCC_REQUIRED
        ) and check_cache_matches(
            tt_cache[torch.arange(batch_size), :, position_ids, :].unsqueeze(2),
            output_cache[:, :, -1:, :],
            hf_config_short.kv_lora_rank,
            pcc_required=PCC_REQUIRED_KVPE,
        ), f"MLA output for decode {batch_size=} {position_ids=} does not meet PCC requirement {PCC_REQUIRED} or KVPE Cache PCC requirement {PCC_REQUIRED_KVPE} or has been modified outside user area"


# Base test cases - ranges will be expanded into individual test cases
# see documentation for expand_test_cases_with_position_ids_ranges for more details
BASE_TEST_CASES = [
    # mode, seq_len, batch_size_per_row, decode_position_ids
    ("decode", 1, USERS_PER_ROW, None),
    # ("decode", 1, USERS_PER_ROW, (4096, 8192, 32)), # Example.
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
]  # decode_position_ids is not applicable for prefill

# Expand ranges into individual position_ids for pytest
EXPANDED_TEST_CASES = expand_test_cases_with_position_ids_ranges(BASE_TEST_CASES)
EXPANDED_TEST_IDS = build_expanded_test_ids(EXPANDED_TEST_CASES)


@pytest.mark.parametrize(
    "mode, seq_len, batch_size_per_row, decode_position_ids",
    EXPANDED_TEST_CASES,
    ids=EXPANDED_TEST_IDS,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "module_path",
    [None, "model.layers.0.self_attn"],
)
@pytest.mark.parametrize(
    "use_synthetic_weights",
    [True, False],  # Test both synthetic and real weights
)
@pytest.mark.parametrize(
    "test_closure",
    [
        pytest.param(run_test_forward_pass_mla2d, marks=pytest.mark.requires_device(["TG", "DUAL", "QUAD"])),
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    batch_size_per_row,
    decode_position_ids,
    hf_config_short,
    cache_path,
    mesh_device,
    ccl,
    model_path,
    module_path,
    use_synthetic_weights,
    force_recalculate_weight_config,
    test_closure,
    set_deterministic_env,
    state_dict,
):
    # Hardcoded arguments; can later change them to test arguments if needed
    layer_idx = 0

    # Only use decode_position_ids for decode mode
    if mode != "decode":
        decode_position_ids = None

    # Skip loading state_dict from file if using synthetic weights
    if use_synthetic_weights:
        logger.info("Using synthetic weights for testing")
        # Pass None as state_dict when using synthetic weights
        state_dict = None
    elif module_path is None:
        # When using real weights without module_path, we still need state_dict
        logger.info("Using real weights for testing")

    test_closure(
        layer_idx,
        mode,
        seq_len,
        batch_size_per_row,
        hf_config_short,
        cache_path,
        mesh_device,
        ccl,
        model_path,
        module_path,
        force_recalculate_weight_config,
        state_dict,
        decode_position_ids,
        use_synthetic_weights,
    )


if __name__ == "__main__":
    pytest.main([__file__])
