# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for throughput-optimized MoE experts with all_to_all operations.

This module follows the structure of tests/nightly/t3000/ccl/test_all_to_all_dispatch.py
and tests/nightly/t3000/ccl/test_moe_expert_token_remap.py for consistency.

Test structure:
1. Tensor generation helpers (gen_*)
2. Golden reference computation helpers
3. Test runner function
4. Parameterized test cases
"""

import os
import random
from math import prod
from typing import Optional

import pytest
import torch
from loguru import logger

import ttnn

from ...tt.experts_throughput import ThroughputExpertConfig, ThroughputExperts, ThroughputProgramConfig
from ...tt.experts_throughput.config import create_expert_mapping_tensors, create_remap_topk_mask
from ..test_factory import parametrize_mesh_with_fabric

# =============================================================================
# TENSOR GENERATION HELPERS
# =============================================================================


def gen_hidden_states(
    batch: int,
    seq_len: int,
    hidden_size: int,
    scheme: str = "random",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate input hidden states tensor.

    Args:
        batch: Total batch size (across all devices)
        seq_len: Sequence length per token
        hidden_size: Hidden dimension size
        scheme: Generation scheme ("random", "sequential", or "fixed")
        dtype: Output tensor dtype

    Returns:
        Hidden states tensor [batch, 1, seq_len, hidden_size]
    """
    if scheme == "random":
        return torch.randn(batch, 1, seq_len, hidden_size, dtype=dtype)
    elif scheme == "fixed":
        # Fixed values for debugging - all tokens identical
        return torch.ones(batch, 1, seq_len, hidden_size, dtype=dtype)
    elif scheme == "sequential":
        # Sequential values for debugging - each token has unique pattern
        tokens = []
        factor = 1
        for b in range(batch):
            for s in range(seq_len):
                tokens.append(torch.ones(1, 1, 1, hidden_size, dtype=dtype) * factor)
                factor += 1
        return torch.cat(tokens, dim=0).reshape(batch, 1, seq_len, hidden_size)
    else:
        raise ValueError(f"Invalid scheme: {scheme}")


def gen_expert_mapping(
    num_experts: int,
    num_devices: int,
    scheme: str = "sequential",
) -> torch.Tensor:
    """Generate expert-to-device mapping tensor.

    Creates a one-hot mapping indicating which device owns each expert.

    Args:
        num_experts: Total number of experts
        num_devices: Number of devices in mesh
        scheme: Distribution scheme ("sequential", "random", or "fixed")

    Returns:
        Expert mapping tensor [1, 1, num_experts, num_devices] with dtype int16
    """
    assert num_experts % num_devices == 0, "num_experts must be divisible by num_devices"
    experts_per_device = num_experts // num_devices

    expert_mapping = torch.zeros(1, 1, num_experts, num_devices, dtype=torch.int16)
    device_expert_count = {d: 0 for d in range(num_devices)}
    for expert_id in range(num_experts):
        if scheme in ["sequential", "fixed"]:
            # Experts distributed evenly: experts 0-3 -> device 0, 4-7 -> device 1, etc.
            # "fixed" uses same distribution as "sequential" for device mapping
            device_id = expert_id // experts_per_device
            expert_mapping[0, 0, expert_id, device_id] = 1
        elif scheme == "random":
            # Random distribution while maintaining equal experts per device
            available_devices = [d for d, count in device_expert_count.items() if count < experts_per_device]
            device_id = random.choice(available_devices)
            expert_mapping[0, 0, expert_id, device_id] = 1
            device_expert_count[device_id] += 1
        else:
            raise ValueError(f"Invalid scheme: {scheme}")

    return expert_mapping


def gen_expert_indices(
    batch: int,
    seq_len: int,
    num_experts: int,
    num_experts_per_tok: int,
    scheme: str = "random",
) -> torch.Tensor:
    """Generate expert selection indices for each token.

    Args:
        batch: Batch size
        seq_len: Sequence length
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts selected per token (top-k)
        scheme: Selection scheme ("random", "sequential", or "fixed")

    Returns:
        Expert indices tensor [batch, 1, seq_len, num_experts_per_tok] with dtype int16
    """
    expert_indices = torch.ones(batch, 1, seq_len, num_experts_per_tok, dtype=torch.int16) * -1
    current_expert = 0

    for b in range(batch):
        for s in range(seq_len):
            for k in range(num_experts_per_tok):
                if scheme == "fixed":
                    # All tokens use the same first N experts (0, 1, 2, ..., k-1)
                    expert_indices[b, 0, s, k] = k
                elif scheme == "sequential":
                    expert_indices[b, 0, s, k] = current_expert % num_experts
                    current_expert += 1 + (k % 2)
                elif scheme == "random":
                    # Ensure unique experts per token
                    current_indices = expert_indices[b, 0, s, :].tolist()
                    available = [e for e in range(num_experts) if e not in current_indices]
                    expert_indices[b, 0, s, k] = random.choice(available)
                else:
                    raise ValueError(f"Invalid scheme: {scheme}")

    return expert_indices


def gen_routing_weights(
    batch: int,
    seq_len: int,
    num_experts: int,
    num_experts_per_tok: int,
    expert_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate routing weights for expert selection.

    Creates both sparse full routing_weights matrix and dense topk_weights.

    Args:
        batch: Batch size
        seq_len: Sequence length
        num_experts: Total number of experts
        num_experts_per_tok: Number of experts per token
        expert_indices: Pre-generated expert indices [batch, 1, seq_len, k]

    Returns:
        Tuple of:
        - routing_weights: [batch * seq_len, num_experts] - sparse full weights
        - topk_weights: [batch, 1, seq_len, k] - dense top-k weights
    """
    routing_weights = torch.zeros(batch * seq_len, num_experts, dtype=torch.float32)
    topk_weights = torch.zeros(batch, 1, seq_len, num_experts_per_tok, dtype=torch.bfloat16)

    for b in range(batch):
        for s in range(seq_len):
            # Generate random weights for selected experts
            weights = torch.rand(num_experts_per_tok)
            weights = weights / weights.sum()  # Normalize

            flat_idx = b * seq_len + s
            for k in range(num_experts_per_tok):
                expert_id = expert_indices[b, 0, s, k].item()
                routing_weights[flat_idx, expert_id] = weights[k]
                topk_weights[b, 0, s, k] = weights[k]

    return routing_weights, topk_weights


def gen_expert_weights(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    std: float = 0.02,
) -> dict:
    """Generate random expert weights in GPT-OSS fused format.

    Args:
        num_experts: Number of experts
        hidden_size: Hidden dimension
        intermediate_size: Intermediate (FFN) dimension
        std: Standard deviation for weight initialization

    Returns:
        Dictionary with gate_up_proj, gate_up_proj_bias, down_proj, down_proj_bias
    """
    return {
        "gate_up_proj": torch.randn(num_experts, hidden_size, 2 * intermediate_size) * std,
        "gate_up_proj_bias": torch.zeros(num_experts, 2 * intermediate_size),
        "down_proj": torch.randn(num_experts, intermediate_size, hidden_size) * std,
        "down_proj_bias": torch.zeros(num_experts, hidden_size),
    }


def gen_remap_topk_mask(
    num_dispatch_rows: int,
    num_experts: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Generate remap topk mask tensor.

    This mask is used by moe_expert_token_remap to create sparsity patterns.

    Args:
        num_dispatch_rows: Number of dispatch device rows
        num_experts: Total number of experts
        dtype: Output tensor dtype

    Returns:
        Mask tensor [1, num_dispatch_rows, 1, num_experts]
    """
    return torch.ones(1, num_dispatch_rows, 1, num_experts, dtype=dtype)


def gen_metadata_tensor(
    expert_indices: torch.Tensor,
    num_devices: int,
) -> torch.Tensor:
    """Generate metadata tensor (simulates all_to_all_dispatch output).

    This is the expert indices duplicated for each device.

    Args:
        expert_indices: Expert indices [batch, 1, seq_len, k]
        num_devices: Number of devices

    Returns:
        Metadata tensor [num_devices, batch, seq_len, k]
    """
    batch = expert_indices.shape[0]
    seq_len = expert_indices.shape[2]
    k = expert_indices.shape[3]

    metadata = expert_indices.reshape(1, batch, seq_len, k)
    return metadata.repeat(num_devices, 1, 1, 1)


# =============================================================================
# GOLDEN REFERENCE COMPUTATION
# =============================================================================


def compute_reference_expert_output(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
) -> torch.Tensor:
    """Compute reference MoE expert output using PyTorch.

    Implements: output = sum_k(weight_k * down(silu(gate(x)) * up(x)))

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        expert_indices: [batch, seq_len, k] - which experts each token routes to
        topk_weights: [batch, seq_len, k] - routing weights
        gate_proj: [num_experts, hidden_size, intermediate_size]
        up_proj: [num_experts, hidden_size, intermediate_size]
        down_proj: [num_experts, intermediate_size, hidden_size]

    Returns:
        Output tensor [batch, seq_len, hidden_size]
    """
    batch, seq_len, hidden_size = hidden_states.shape
    num_experts_per_tok = expert_indices.shape[-1]

    output = torch.zeros(batch, seq_len, hidden_size, dtype=hidden_states.dtype)

    for b in range(batch):
        for s in range(seq_len):
            x = hidden_states[b, s]  # [hidden_size]

            for k in range(num_experts_per_tok):
                expert_id = expert_indices[b, s, k].item()
                weight = topk_weights[b, s, k].item()

                # Expert forward: silu(gate) * up, then down
                gate_out = torch.matmul(x.float(), gate_proj[expert_id].float())
                up_out = torch.matmul(x.float(), up_proj[expert_id].float())
                activated = torch.nn.functional.silu(gate_out) * up_out
                expert_out = torch.matmul(activated, down_proj[expert_id].float())

                output[b, s] += weight * expert_out.to(hidden_states.dtype)

    return output


def get_experts_on_device(
    num_experts: int,
    expert_mapping: torch.Tensor,
    device_id: int,
) -> list:
    """Get list of expert indices assigned to a device.

    Args:
        num_experts: Total number of experts
        expert_mapping: Expert mapping tensor [1, 1, num_experts, num_devices]
        device_id: Device index

    Returns:
        List of expert indices on this device
    """
    experts = []
    for e in range(num_experts):
        if expert_mapping[0, 0, e, device_id] == 1:
            experts.append(e)
    return experts


# =============================================================================
# COMBINED TENSOR GENERATION
# =============================================================================


def gen_all_tensors(
    batch: int,
    seq_len: int,
    num_experts: int,
    num_experts_per_tok: int,
    hidden_size: int,
    intermediate_size: int,
    num_devices: int,
    scheme: str = "random",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """Generate all tensors needed for throughput experts testing.

    Args:
        batch: Total batch size
        seq_len: Sequence length
        num_experts: Total number of experts
        num_experts_per_tok: Experts per token (top-k)
        hidden_size: Hidden dimension
        intermediate_size: FFN intermediate dimension
        num_devices: Number of devices
        scheme: Generation scheme ("random", "sequential", or "fixed")
               - "random": random inputs and expert selection
               - "sequential": sequential patterns for debugging
               - "fixed": all tokens use same inputs and first N experts
        dtype: Tensor dtype

    Returns:
        Dictionary containing all generated tensors and expected outputs
    """
    # Generate inputs
    hidden_states = gen_hidden_states(batch, seq_len, hidden_size, scheme, dtype)
    expert_mapping = gen_expert_mapping(num_experts, num_devices, scheme)
    expert_indices = gen_expert_indices(batch, seq_len, num_experts, num_experts_per_tok, scheme)
    routing_weights, topk_weights = gen_routing_weights(
        batch, seq_len, num_experts, num_experts_per_tok, expert_indices
    )

    # Generate weights
    weights = gen_expert_weights(num_experts, hidden_size, intermediate_size)

    # Split fused gate_up_proj into separate tensors for reference
    gate_up = weights["gate_up_proj"]
    gate_proj = gate_up[:, :, :intermediate_size]
    up_proj = gate_up[:, :, intermediate_size:]
    down_proj = weights["down_proj"]

    # Compute reference output
    hidden_3d = hidden_states.squeeze(1)  # [batch, seq_len, hidden]
    indices_3d = expert_indices.squeeze(1)  # [batch, seq_len, k]
    weights_3d = topk_weights.squeeze(1)  # [batch, seq_len, k]

    reference_output = compute_reference_expert_output(hidden_3d, indices_3d, weights_3d, gate_proj, up_proj, down_proj)

    return {
        "hidden_states": hidden_states,
        "expert_mapping": expert_mapping,
        "expert_indices": expert_indices,
        "routing_weights": routing_weights,
        "topk_weights": topk_weights,
        "weights": weights,
        "gate_proj": gate_proj,
        "up_proj": up_proj,
        "down_proj": down_proj,
        "reference_output": reference_output,
    }


# =============================================================================
# HUGGINGFACE MODEL LOADING
# =============================================================================


def load_hf_reference_layer(
    model_path: str,
):
    """Load HuggingFace reference layer and extract experts.

    Args:
        model_path: Path to HuggingFace model

    Returns:
        Tuple of (config, reference_experts, state_dict)
    """
    try:
        from transformers import AutoConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssDecoderLayer
    except ImportError:
        raise ImportError("transformers library required to load HuggingFace models")

    logger.info(f"Loading HuggingFace model from: {model_path}")

    # Load config to validate dimensions
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Create reference decoder layer
    reference_layer = GptOssDecoderLayer(config, layer_idx=0)

    # Initialize weights (either load from checkpoint or use random for testing)
    # # For now, use random initialization with fixed seed for reproducibility
    with torch.no_grad():
        std = 0.02
        reference_layer.mlp.experts.gate_up_proj.data.normal_(0.0, std)
        reference_layer.mlp.experts.gate_up_proj_bias.data.normal_(0.0, std)
        reference_layer.mlp.experts.down_proj.data.normal_(0.0, std)
        reference_layer.mlp.experts.down_proj_bias.data.normal_(0.0, std)

    # Extract experts module and set to eval mode
    reference_experts = reference_layer.mlp.experts.eval()
    breakpoint()

    # Extract state dict for TT implementation
    state_dict = {
        "gate_up_proj": reference_experts.gate_up_proj.data.clone(),
        "gate_up_proj_bias": reference_experts.gate_up_proj_bias.data.clone(),
        "down_proj": reference_experts.down_proj.data.clone(),
        "down_proj_bias": reference_experts.down_proj_bias.data.clone(),
    }

    logger.info(f"Loaded expert weights:")
    logger.info(f"  gate_up_proj: {state_dict['gate_up_proj'].shape}")
    logger.info(f"  down_proj: {state_dict['down_proj'].shape}")

    return config, reference_experts, state_dict


# =============================================================================
# TEST RUNNER FUNCTIONS
# =============================================================================


def run_throughput_experts_test(
    mesh_device,
    mesh_shape: tuple,
    global_batch_size: int,
    seq_len: int,
    num_iters: int,
    scheme: str = "random",
    dtype=ttnn.bfloat16,
    weight_dtype=ttnn.bfloat16,
    input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    output_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    dispatch_cluster_axis: Optional[int] = None,
    num_links: int = 1,
    pcc_threshold: float = 0.93,
):
    """Run throughput experts test with validation against reference.

    Args:
        mesh_device: TTNN mesh device
        mesh_shape: Mesh shape tuple (rows, cols)
        global_batch_size: Batch size
        num_experts: Total number of experts
        num_experts_per_tok: Experts per token
        hidden_size: Hidden dimension
        intermediate_size: FFN intermediate dimension
        seq_len: Sequence length
        num_iters: Number of iterations
        scheme: Tensor generation scheme
        dtype: Activation dtype
        weight_dtype: Weight dtype
        input_memory_config: Input memory config
        output_memory_config: Output memory config
        dispatch_cluster_axis: Dispatch axis (0 for rows, 1 for cols)
        num_links: Number of CCL links
        pcc_threshold: PCC threshold for validation

    Returns:
        True if all iterations pass
    """
    torch.manual_seed(2024)
    random.seed(2024)

    # Load reference experts from HuggingFace model if specified, otherwise generate random
    hf_model_path = os.environ.get("HF_MODEL")
    if hf_model_path:
        logger.info(f"Loading reference model from HuggingFace: {hf_model_path}")
        hf_config, reference_experts, weights = load_hf_reference_layer(
            model_path=hf_model_path,
        )
        num_experts = hf_config.num_local_experts
        hidden_size = hf_config.hidden_size
        intermediate_size = hf_config.intermediate_size
        num_experts_per_tok = hf_config.num_experts_per_tok
    else:
        logger.info("HF_MODEL not set, generating random weights")
        num_experts = 128
        hidden_size = 2880
        intermediate_size = 2880
        num_experts_per_tok = 4
        weights = gen_expert_weights(num_experts, hidden_size, intermediate_size)
        reference_experts = None  # Will skip reference comparison
    num_devices = prod(mesh_shape)
    # we shard the tokens across the rows of the mesh
    batch_per_device = global_batch_size // mesh_shape[0]

    logger.info(f"Running throughput experts test:")
    logger.info(f"  Mesh: {mesh_shape[0]}x{mesh_shape[1]} ({num_devices} devices)")
    logger.info(f"  Batch: {global_batch_size} ({batch_per_device} per device)")
    logger.info(f"  Experts: {num_experts} ({num_experts // num_devices} per device)")
    logger.info(f"  Hidden: {hidden_size}, Intermediate: {intermediate_size}")
    logger.info(f"  Scheme: {scheme}, Iterations: {num_iters}")

    # Generate test tensors for all iterations (inputs/routing, but use loaded weights)
    all_tensors = []
    for _ in range(num_iters):
        # Generate inputs and routing
        hidden_states = gen_hidden_states(global_batch_size, seq_len, hidden_size, scheme).to(torch.float)
        expert_mapping = gen_expert_mapping(num_experts, num_devices, scheme)
        expert_indices = gen_expert_indices(global_batch_size, seq_len, num_experts, num_experts_per_tok, scheme)
        routing_weights_sparse, topk_weights = gen_routing_weights(
            global_batch_size, seq_len, num_experts, num_experts_per_tok, expert_indices
        )

        # Compute reference output using HuggingFace experts if available
        reference_output = None
        if reference_experts is not None:
            hidden_3d = hidden_states.squeeze(1)  # [batch, seq_len, hidden]

            # Reshape expert_indices for HuggingFace (expects [batch*seq_len, k])
            expert_indices_2d = expert_indices.squeeze(1).reshape(-1, num_experts_per_tok).to(torch.long)

            with torch.no_grad():
                reference_output = reference_experts(
                    hidden_3d,
                    router_indices=expert_indices_2d,
                    routing_weights=routing_weights_sparse,
                )

        tensors = {
            "hidden_states": hidden_states,
            "expert_mapping": expert_mapping,
            "expert_indices": expert_indices,
            "routing_weights": routing_weights_sparse,
            "topk_weights": topk_weights,
            "reference_output": reference_output,
        }
        all_tensors.append(tensors)

    # Create TT config
    config = ThroughputExpertConfig(
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        hidden_size=hidden_size,
        num_experts_per_tok=num_experts_per_tok,
        num_devices=num_devices,
    )

    # Create TT experts module
    breakpoint()
    tt_experts = ThroughputExperts(
        mesh_device=mesh_device,
        config=config,
        state_dict=weights,
        weight_dtype=weight_dtype,
        dispatch_cluster_axis=dispatch_cluster_axis,
        decode_memory_config=output_memory_config,
    )

    # Run test iterations
    all_passed = True
    for iter_idx, tensors in enumerate(all_tensors):
        logger.info(f"Iteration {iter_idx + 1}/{num_iters}")

        reference_output = tensors["reference_output"]

        # Convert to TT tensors
        tt_hidden = ttnn.from_torch(
            tensors["hidden_states"],
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
        )

        tt_indices = ttnn.from_torch(
            tensors["expert_indices"].to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
        )

        tt_weights = ttnn.from_torch(
            tensors["topk_weights"],
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=input_memory_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(dims=(0, None), mesh_shape=mesh_device.shape, mesh_device=mesh_device),
        )

        # Run forward pass
        tt_output = tt_experts.forward_decode(tt_hidden, tt_indices, tt_weights)

        # Convert output to torch
        breakpoint()
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, 0), mesh_shape=tuple(mesh_device.shape)),
        )[0]

        # Compare using PCC if reference is available
        if reference_output is not None:
            from models.common.utility_functions import comp_pcc

            passing, pcc_value = comp_pcc(reference_output, tt_output_torch, pcc_threshold)

            logger.info(f"  PCC: {pcc_value:.6f} (threshold: {pcc_threshold})")
            logger.info(f"  Reference mean: {reference_output.mean().item():.6f}")
            logger.info(f"  TT mean: {tt_output_torch.mean().item():.6f}")
            logger.info(f"  Max diff: {(reference_output - tt_output_torch).abs().max().item():.6f}")

            if not passing:
                logger.error(f"  FAILED: PCC {pcc_value} < {pcc_threshold}")
                all_passed = False
            else:
                logger.info(f"  PASSED")
        else:
            logger.info(f"  TT output mean: {tt_output_torch.mean().item():.6f}")
            logger.info(f"  Skipping PCC comparison (no reference available)")

        # Cleanup
        ttnn.deallocate(tt_hidden)
        ttnn.deallocate(tt_indices)
        ttnn.deallocate(tt_weights)
        ttnn.deallocate(tt_output)

    return all_passed


# =============================================================================
# CONFIG UNIT TESTS
# =============================================================================


class TestThroughputExpertConfig:
    """Tests for ThroughputExpertConfig."""

    def test_config_creation(self):
        """Test basic config creation."""
        config = ThroughputExpertConfig(
            intermediate_size=5632,
            num_experts=128,
            hidden_size=2048,
            num_experts_per_tok=8,
            num_devices=32,
        )

        assert config.intermediate_size == 5632
        assert config.num_experts == 128
        assert config.hidden_size == 2048
        assert config.num_experts_per_tok == 8
        assert config.num_devices == 32
        assert config.num_experts_per_device == 4  # 128 / 32

    def test_config_validation_experts_divisibility(self):
        """Test that num_experts must be divisible by num_devices."""
        with pytest.raises(ValueError, match="must be divisible"):
            ThroughputExpertConfig(
                intermediate_size=5632,
                num_experts=100,  # Not divisible by 32
                hidden_size=2048,
                num_experts_per_tok=8,
                num_devices=32,
            )

    def test_config_default_values(self):
        """Test default values are set correctly."""
        config = ThroughputExpertConfig(
            intermediate_size=5632,
            num_experts=128,
            hidden_size=2048,
            num_experts_per_tok=8,
            num_devices=32,
        )

        assert config.sparsity_block_size == 32
        assert config.swiglu_limit == 88.0
        assert config.alpha == 1.702


# =============================================================================
# TENSOR GENERATION TESTS
# =============================================================================


class TestTensorGeneration:
    """Tests for tensor generation helpers."""

    @pytest.mark.parametrize("scheme", ["random", "sequential", "fixed"])
    def test_gen_hidden_states(self, scheme):
        """Test hidden states generation."""
        batch, seq_len, hidden = 8, 2, 64
        hidden_states = gen_hidden_states(batch, seq_len, hidden, scheme)

        assert hidden_states.shape == (batch, 1, seq_len, hidden)
        assert hidden_states.dtype == torch.bfloat16

        if scheme == "fixed":
            # All hidden states should be ones
            assert torch.allclose(hidden_states, torch.ones_like(hidden_states))

    @pytest.mark.parametrize("scheme", ["random", "sequential", "fixed"])
    def test_gen_expert_mapping(self, scheme):
        """Test expert mapping generation."""
        num_experts, num_devices = 32, 8
        mapping = gen_expert_mapping(num_experts, num_devices, scheme)

        assert mapping.shape == (1, 1, num_experts, num_devices)
        assert mapping.dtype == torch.int16

        # Each expert maps to exactly one device
        for e in range(num_experts):
            assert mapping[0, 0, e, :].sum() == 1

        # Each device has equal experts
        for d in range(num_devices):
            assert mapping[0, 0, :, d].sum() == num_experts // num_devices

    @pytest.mark.parametrize("scheme", ["random", "sequential", "fixed"])
    def test_gen_expert_indices(self, scheme):
        """Test expert indices generation."""
        batch, seq_len, num_experts, k = 4, 2, 16, 4
        indices = gen_expert_indices(batch, seq_len, num_experts, k, scheme)

        assert indices.shape == (batch, 1, seq_len, k)
        assert indices.dtype == torch.int16

        # All indices are valid
        assert (indices >= 0).all()
        assert (indices < num_experts).all()

        # Each token has unique experts
        for b in range(batch):
            for s in range(seq_len):
                token_experts = indices[b, 0, s, :].tolist()
                assert len(set(token_experts)) == k

        if scheme == "fixed":
            # All tokens should use experts [0, 1, 2, 3]
            expected_indices = list(range(k))
            for b in range(batch):
                for s in range(seq_len):
                    token_experts = indices[b, 0, s, :].tolist()
                    assert token_experts == expected_indices

    def test_gen_routing_weights(self):
        """Test routing weights generation."""
        batch, seq_len, num_experts, k = 4, 2, 16, 4
        indices = gen_expert_indices(batch, seq_len, num_experts, k, "random")
        routing_weights, topk_weights = gen_routing_weights(batch, seq_len, num_experts, k, indices)

        assert routing_weights.shape == (batch * seq_len, num_experts)
        assert topk_weights.shape == (batch, 1, seq_len, k)

        # Weights sum to 1 for each token
        for i in range(batch * seq_len):
            assert abs(routing_weights[i].sum() - 1.0) < 1e-5

    def test_gen_all_tensors(self):
        """Test combined tensor generation."""
        tensors = gen_all_tensors(
            batch=8,
            seq_len=2,
            num_experts=32,
            num_experts_per_tok=4,
            hidden_size=64,
            intermediate_size=128,
            num_devices=8,
            scheme="random",
        )

        assert "hidden_states" in tensors
        assert "expert_mapping" in tensors
        assert "expert_indices" in tensors
        assert "routing_weights" in tensors
        assert "topk_weights" in tensors
        assert "weights" in tensors
        assert "reference_output" in tensors

        # Reference output has correct shape
        assert tensors["reference_output"].shape == (8, 2, 64)


# =============================================================================
# REFERENCE IMPLEMENTATION TESTS
# =============================================================================


class TestReferenceImplementation:
    """Tests for reference implementation correctness."""

    def test_reference_expert_forward(self):
        """Test reference expert forward pass."""
        torch.manual_seed(42)

        batch, seq_len, hidden = 4, 1, 64
        intermediate = 128
        num_experts, k = 8, 2

        # Create inputs
        hidden_states = torch.randn(batch, seq_len, hidden)
        expert_indices = torch.zeros(batch, seq_len, k, dtype=torch.long)
        topk_weights = torch.zeros(batch, seq_len, k)

        for b in range(batch):
            experts = torch.randperm(num_experts)[:k]
            weights = torch.rand(k)
            weights = weights / weights.sum()
            expert_indices[b, 0] = experts
            topk_weights[b, 0] = weights

        # Create weights
        gate_proj = torch.randn(num_experts, hidden, intermediate) * 0.02
        up_proj = torch.randn(num_experts, hidden, intermediate) * 0.02
        down_proj = torch.randn(num_experts, intermediate, hidden) * 0.02

        # Run reference
        output = compute_reference_expert_output(
            hidden_states,
            expert_indices,
            topk_weights,
            gate_proj,
            up_proj,
            down_proj,
        )

        assert output.shape == (batch, seq_len, hidden)
        assert not torch.allclose(output, torch.zeros_like(output))

        logger.info("Reference implementation test passed")


# =============================================================================
# DEVICE TESTS - MAPPING TENSORS
# =============================================================================


@parametrize_mesh_with_fabric()
class TestMappingTensorCreation:
    """Tests for expert mapping tensor creation on device."""

    def test_expert_mapping_tensors(self, mesh_device, device_params):
        """Test expert-to-device mapping tensor creation."""
        num_devices = mesh_device.get_num_devices()
        num_experts_per_device = 4
        num_experts = num_devices * num_experts_per_device

        mapping = create_expert_mapping_tensors(
            num_devices=num_devices,
            num_experts_per_device=num_experts_per_device,
            mesh_device=mesh_device,
        )

        assert mapping.shape == [1, 1, num_experts, num_devices]
        assert mapping.dtype == ttnn.uint16

        # Verify correctness
        mapping_torch = ttnn.to_torch(ttnn.get_device_tensors(mapping)[0])
        for e in range(num_experts):
            assert mapping_torch[0, 0, e, :].sum() == 1
            expected_device = e // num_experts_per_device
            assert mapping_torch[0, 0, e, expected_device] == 1

    def test_remap_topk_mask(self, mesh_device, device_params):
        """Test remap mask creation."""
        num_dispatch_rows = mesh_device.shape[0]
        num_experts = 128

        mask = create_remap_topk_mask(
            num_dispatch_device_rows=num_dispatch_rows,
            num_experts=num_experts,
            mesh_device=mesh_device,
        )

        assert mask.shape == [1, num_dispatch_rows, 1, num_experts]
        assert mask.dtype == ttnn.bfloat16


# =============================================================================
# DEVICE TESTS - INTEGRATION
# =============================================================================


@parametrize_mesh_with_fabric()
class TestThroughputExpertsIntegration:
    """Integration tests for ThroughputExperts module."""

    def test_experts_creation(self, mesh_device, device_params):
        """Test ThroughputExperts initialization."""
        num_devices = mesh_device.get_num_devices()
        num_experts = num_devices * 4

        config = ThroughputExpertConfig(
            intermediate_size=1024,
            num_experts=num_experts,
            hidden_size=512,
            num_experts_per_tok=2,
            num_devices=num_devices,
        )
        weights = gen_expert_weights(num_experts, 512, 1024)

        experts = ThroughputExperts(
            mesh_device=mesh_device,
            config=config,
            state_dict=weights,
        )

        assert experts.config == config
        assert experts.num_experts == num_experts
        assert experts.weights.w1 is not None
        assert experts.expert_mapping_tensors is not None

    def test_experts_with_custom_config(self, mesh_device, device_params):
        """Test ThroughputExperts with custom configs."""
        num_devices = mesh_device.get_num_devices()
        num_experts = num_devices * 4

        config = ThroughputExpertConfig(
            intermediate_size=1024,
            num_experts=num_experts,
            hidden_size=512,
            num_experts_per_tok=2,
            num_devices=num_devices,
        )
        weights = gen_expert_weights(num_experts, 512, 1024)

        program_config = ThroughputProgramConfig(
            gate_up_cores=(6, 6),
            down_cores=(6, 5),
        )

        experts = ThroughputExperts(
            mesh_device=mesh_device,
            config=config,
            state_dict=weights,
            program_config=program_config,
            decode_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        assert experts.program_config == program_config


# =============================================================================
# MAIN PARAMETRIZED TESTS (following CCL test structure)
# =============================================================================


# @pytest.mark.parametrize(
#     "device_params",
#     [
#         {
#             "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
#             "fabric_config": ttnn.FabricConfig.FABRIC_1D,
#         },
#     ],
#     ids=["fabric_1d"],
#     indirect=True,
# )
# @pytest.mark.parametrize(
#     "mesh_shape, mesh_device",
#     [
#         pytest.param((2, 4), (2, 4), id="2x4_grid"),
#     ],
#     indirect=["mesh_device"],
# )
# @pytest.mark.parametrize("batch_per_device", [32])
# @pytest.mark.parametrize("experts_per_device", [4])
# @pytest.mark.parametrize("num_experts_per_tok", [2])
# @pytest.mark.parametrize("hidden_size", [512])
# @pytest.mark.parametrize("intermediate_size", [1024])
# @pytest.mark.parametrize("seq_len", [1])
# @pytest.mark.parametrize("num_iters", [2])
# @pytest.mark.parametrize("scheme", ["random"])
# @pytest.mark.parametrize("cluster_axis", [1])
# @pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
# @pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
# def test_throughput_experts_t3000(
#     mesh_device,
#     device_params,
#     mesh_shape,
#     batch_per_device,
#     experts_per_device,
#     num_experts_per_tok,
#     hidden_size,
#     intermediate_size,
#     seq_len,
#     num_iters,
#     scheme,
#     cluster_axis,
#     input_memory_config,
#     output_memory_config,
# ):
#     """Test throughput experts on T3000 (2x4 grid)."""
#     num_devices = prod(mesh_shape)
#     num_experts = num_devices * experts_per_device

#     passed = run_throughput_experts_test(
#         mesh_device=mesh_device,
#         mesh_shape=mesh_shape,
#         batch_per_device=batch_per_device,
#         num_experts=num_experts,
#         num_experts_per_tok=num_experts_per_tok,
#         hidden_size=hidden_size,
#         intermediate_size=intermediate_size,
#         seq_len=seq_len,
#         num_iters=num_iters,
#         scheme=scheme,
#         cluster_axis=cluster_axis,
#         input_memory_config=input_memory_config,
#         output_memory_config=output_memory_config,
#     )

#     assert passed, "Throughput experts test failed"


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize(
    "mesh_shape",
    [
        (4, 8),
    ],
    ids=["mesh_4x8"],
)
@pytest.mark.parametrize("global_batch_size", [128])
@pytest.mark.parametrize("seq_len", [1])
@pytest.mark.parametrize("num_iters", [1])
@pytest.mark.parametrize("scheme", ["random"])
@pytest.mark.parametrize("dispatch_cluster_axis", [0])
@pytest.mark.parametrize("input_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
@pytest.mark.parametrize("output_memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram"])
def test_throughput_experts_galaxy(
    device_params,
    mesh_device,
    mesh_shape,
    global_batch_size,
    seq_len,
    num_iters,
    scheme,
    dispatch_cluster_axis,
    input_memory_config,
    output_memory_config,
):
    """Test throughput experts on Galaxy (4x8 grid)."""
    num_devices = prod(mesh_shape)

    passed = run_throughput_experts_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        global_batch_size=global_batch_size,
        seq_len=seq_len,
        num_iters=num_iters,
        scheme=scheme,
        dispatch_cluster_axis=dispatch_cluster_axis,
        input_memory_config=input_memory_config,
        output_memory_config=output_memory_config,
    )

    assert passed, "Throughput experts test failed"
