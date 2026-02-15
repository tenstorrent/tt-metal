#!/usr/bin/env python3
"""
Modified reference decoder that captures intermediate tensors.
This uses the EXACT same flow as test_decoder_block.py but captures intermediates.
"""

import sys

# Add paths exactly as test_decoder_block.py does
sys.path.append("/home/ntarafdar/tt-moe/tt-metal")
sys.path.append("/home/ntarafdar/tt-moe/tt-metal/models")
sys.path.append("/home/ntarafdar/tt-moe/tt-metal/models/demos/deepseek_v3")

import torch
from loguru import logger

import ttnn

# Import exactly what test_decoder_block.py imports
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import MoEDecoderBlock2D
from models.demos.deepseek_v3.tt.mlp.shared_expert import SharedExpert
from models.demos.deepseek_v3.tt.moe import MoE


class IntermediateCapture:
    """Helper class to capture intermediate tensors during forward pass"""

    def __init__(self):
        self.intermediates = {}

    def capture(self, tensor, name):
        """Capture a tensor with given name"""
        self.intermediates[name] = tensor
        if hasattr(tensor, "shape"):
            logger.info(f"Captured {name}: shape {tensor.shape}")
        return tensor


# Global instance for capturing intermediates
capture = IntermediateCapture()


def run_decoder_block_with_intermediates(input_tensor, position_ids, run_config, rope_tensors, page_table):
    """
    Run the EXACT same forward_decode as test_decoder_block.py but capture intermediates.
    This calls MoEDecoderBlock2D.forward_decode directly.

    Args:
        input_tensor: Input tensor (TTNN)
        position_ids: Position IDs tensor
        run_config: Complete run configuration with weights
        rope_tensors: RoPE tensors
        page_table: Page table

    Returns:
        tuple: (output, intermediates_dict)
    """
    global capture
    capture = IntermediateCapture()

    logger.info("=" * 60)
    logger.info("Running MoEDecoderBlock2D.forward_decode with intermediate capture")
    logger.info("=" * 60)

    # Capture input
    capture.capture(input_tensor, "decoder_input")

    # Run the EXACT same forward_decode as test_decoder_block.py
    output = MoEDecoderBlock2D.forward_decode(input_tensor, position_ids, run_config, rope_tensors, page_table)

    # Capture output
    capture.capture(output, "decoder_output")

    return output, capture.intermediates


def run_mlp_only_with_intermediates(input_tensor, run_config):
    """
    Run just the MLP part (MoE + SharedExpert) with intermediate capture.
    This is what MoEDecoderBlock2D.forward_mlp_decode does.

    Captures exactly 6 intermediate tensors:
    1. tp_input: TP=8 input (before all-gather)
    2. all_gather_output: After all-gather (replicated across row)
    3. moe_gate_input: Tensor being fed into MoE gate (same as all_gather_output)
    4. moe_output: MoE module output (after combine)
    5. shared_expert_output: SharedExpert output
    6. reduce_scatter_output: Final output after reduce-scatter

    Args:
        input_tensor: Input tensor (TTNN)
        run_config: Full run configuration from create_test_setup

    Returns:
        tuple: (output, intermediates_dict)
    """
    global capture
    capture = IntermediateCapture()

    logger.info("=" * 60)
    logger.info("Running MoEDecoderBlock2D.forward_mlp_decode with intermediate capture")
    logger.info("=" * 60)

    # 1. Capture TP input (before all-gather)
    capture.capture(input_tensor, "tp_input")

    # Extract the mlp config from the full run_config
    # The run_config structure has mlp config nested under "mlp"
    mlp_config = run_config["mlp"]

    # This is the exact logic from MoEDecoderBlock2D.forward_mlp_decode
    # Handle all_gather if input is TP-sharded
    hidden_size = mlp_config["moe"]["hidden_size"]
    tp_size = mlp_config["moe"]["mesh_device"].shape[1]
    x_dim = input_tensor.shape[-1]

    if x_dim == hidden_size // tp_size:
        # Input is TP-sharded, need to gather
        ccl_shared = mlp_config["shared_expert"]["ccl"]
        x_gathered = ttnn.experimental.all_gather_async(
            input_tensor, **ccl_shared.populate_all_gather_runtime_args(mlp_config["shared_expert"]["all_gather"])
        )
        # 2. After all-gather (replicated across row)
        capture.capture(x_gathered, "all_gather_output")
    else:
        # Already full hidden size
        x_gathered = input_tensor
        # 2. Still save as all_gather_output for consistency
        capture.capture(x_gathered, "all_gather_output")

    # 3. MoE gate input (same as x_gathered)
    capture.capture(x_gathered, "moe_gate_input")

    # Run MoE forward
    logger.info("Running MoE.forward_decode...")
    moe_output = MoE.forward_decode(x_gathered, mlp_config["moe"])
    # 4. MoE module output (after combine)
    capture.capture(moe_output, "moe_output")

    # Run SharedExpert forward
    logger.info("Running SharedExpert.forward_decode...")
    shared_expert_out = SharedExpert.forward_decode(x_gathered, mlp_config["shared_expert"])
    # 5. SharedExpert output
    capture.capture(shared_expert_out, "shared_expert_output")

    # Add outputs (we don't capture combined_out since we have both moe_output and shared_expert_output)
    combined_out = ttnn.add(moe_output, shared_expert_out)

    ttnn.deallocate(moe_output)
    ttnn.deallocate(shared_expert_out)

    # Handle reduce_scatter if input was TP-sharded
    if x_dim == hidden_size // tp_size:
        # Single reduce_scatter on combined output
        output = ttnn.experimental.reduce_scatter_minimal_async(
            combined_out,
            **ccl_shared.populate_reduce_scatter_runtime_args(mlp_config["shared_expert"]["reduce_scatter_async"]),
        )
        # 6. After reduce-scatter (final output)
        capture.capture(output, "reduce_scatter_output")

        ttnn.deallocate(combined_out)
        # Cleanup gathered tensor
        if x_gathered is not input_tensor:
            ttnn.deallocate(x_gathered)
    else:
        # If not TP-sharded, combined output is the final output
        output = combined_out
        # 6. Save as reduce_scatter_output for consistency
        capture.capture(output, "reduce_scatter_output")

    return output, capture.intermediates


def run_moe_only_with_intermediates(input_tensor, moe_config):
    """
    Run just the MoE forward with intermediate capture.

    Args:
        input_tensor: Input tensor (TTNN)
        moe_config: MoE configuration

    Returns:
        tuple: (output, intermediates_dict)
    """
    global capture
    capture = IntermediateCapture()

    logger.info("=" * 60)
    logger.info("Running MoE.forward_decode with intermediate capture")
    logger.info("=" * 60)

    # Capture input
    capture.capture(input_tensor, "moe_input")

    # Import MoEGate to capture router outputs
    from models.demos.deepseek_v3.tt.moe_gate import MoEGate

    # Get router outputs
    logger.info("Running MoEGate.forward...")
    topk_experts_weights, topk_experts_indices = MoEGate.forward(input_tensor, moe_config["moe_gate"])
    capture.capture(topk_experts_weights, "router_weights")
    capture.capture(topk_experts_indices, "router_indices")

    # Run full MoE forward to get final output
    logger.info("Running full MoE.forward_decode...")
    moe_output = MoE.forward_decode(input_tensor, moe_config)
    capture.capture(moe_output, "moe_final_output")

    return moe_output, capture.intermediates


def create_test_setup(hf_config, mesh_device, ccl, layer_idx, state_dict, cache_path, mode="decode"):
    """
    Create the exact same test setup as test_decoder_block.py.
    This handles all the weight loading and configuration creation.

    Args:
        hf_config: HuggingFace config
        mesh_device: TTNN mesh device
        ccl: CCL object
        layer_idx: Which layer to test
        state_dict: Already loaded state dict
        cache_path: Path for caching converted weights
        mode: "decode" or "prefill"

    Returns:
        dict: Complete run configuration with weights loaded
    """
    from models.demos.deepseek_v3.tt.mla.mla2d import MLA2D
    from models.demos.deepseek_v3.utils.config_helpers import USERS_PER_ROW
    from models.demos.deepseek_v3.utils.run_config import create_run_config
    from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config

    logger.info(f"Creating test setup for layer {layer_idx} in {mode} mode...")

    # Get layer state dict
    module_path = f"model.layers.{layer_idx}"
    layer_state_dict = {}

    # Extract only the weights we need for this specific layer
    # This avoids iterating over all keys which can trigger lazy loading of corrupted files
    expected_keys = [
        # MLP weights
        f"{module_path}.mlp.gate.weight",
        f"{module_path}.mlp.gate.e_score_correction_bias",
        # Shared expert weights
        f"{module_path}.mlp.shared_experts.gate_proj.weight",
        f"{module_path}.mlp.shared_experts.up_proj.weight",
        f"{module_path}.mlp.shared_experts.down_proj.weight",
        # MLA projection weights
        f"{module_path}.self_attn.q_a_proj.weight",
        f"{module_path}.self_attn.q_a_proj.weight_scale_inv",
        f"{module_path}.self_attn.q_b_proj.weight",
        f"{module_path}.self_attn.q_b_proj.weight_scale_inv",
        f"{module_path}.self_attn.kv_a_proj_with_mqa.weight",
        f"{module_path}.self_attn.kv_a_proj_with_mqa.weight_scale_inv",
        f"{module_path}.self_attn.kv_b_proj.weight",
        f"{module_path}.self_attn.kv_b_proj.weight_scale_inv",
        f"{module_path}.self_attn.o_proj.weight",
        f"{module_path}.self_attn.o_proj.weight_scale_inv",
        # Norm weights
        f"{module_path}.self_attn.q_a_layernorm.weight",
        f"{module_path}.self_attn.q_b_layernorm.weight",
        f"{module_path}.self_attn.kv_a_layernorm.weight",
        f"{module_path}.self_attn.kv_b_layernorm.weight",
        f"{module_path}.input_layernorm.weight",
        f"{module_path}.post_attention_layernorm.weight",  # This is the correct norm weight for DeepSeek
    ]

    # Add expert weights
    for i in range(256):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            expected_keys.append(f"{module_path}.mlp.experts.{i}.{proj}.weight")
            expected_keys.append(f"{module_path}.mlp.experts.{i}.{proj}.weight_scale_inv")  # If quantized

    # Try to load each expected key
    for key in expected_keys:
        try:
            if key in state_dict:
                # Remove the layer prefix
                new_key = key.replace(f"model.layers.{layer_idx}.", "")
                layer_state_dict[new_key] = state_dict[key]
        except Exception as e:
            # Skip keys that fail to load
            logger.debug(f"Skipping key {key}: {e}")
            continue

    logger.info(f"Loaded {len(layer_state_dict)} weights for layer {layer_idx}")

    # Get weight config using the same function as test_decoder_block.py
    # But we need to handle the missing norm weights case
    try:
        weight_config = get_test_weight_config(
            MoEDecoderBlock2D, hf_config, (layer_state_dict,), cache_path, mesh_device, force_recalculate=False
        )
    except KeyError as e:
        # If we hit a missing weight error, create a minimal weight config
        # with just the MLP weights we need
        logger.warning(f"Weight loading error: {e}, creating minimal weight config for MLP only")

        # For MLP only test, we just need the MoE and SharedExpert weights
        from models.demos.deepseek_v3.tt.mlp.shared_expert import SharedExpert
        from models.demos.deepseek_v3.tt.moe import MoE

        # Convert just the MoE weights
        moe_weight_config = MoE.convert_weights(hf_config, (layer_state_dict,), cache_path, mesh_device)

        # Convert just the shared expert weights
        shared_expert_weight_config = SharedExpert.convert_weights(
            hf_config, (layer_state_dict,), cache_path, mesh_device
        )

        # Create a minimal weight config
        weight_config = {"mlp": {"moe": moe_weight_config, "shared_expert": shared_expert_weight_config}}

    # Get model config
    model_config = get_model_config(MoEDecoderBlock2D, mode, hf_config, mesh_device)

    # Create paged config
    max_seq_len = getattr(hf_config, "max_seq_len", getattr(hf_config, "max_position_embeddings", 4096))
    paged_config = MLA2D.get_valid_paged_config(max_seq_len, USERS_PER_ROW, mesh_device.shape[1])

    # Create state
    model_state = MoEDecoderBlock2D.create_state(
        hf_config, paged_config, mesh_device, ccl, mla_cache=None  # No cache for simplified test
    )

    # Create shared state
    model_shared_state = MoEDecoderBlock2D.create_shared_state(hf_config, mesh_device)

    # Create run config - this merges everything together
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    logger.info("Test setup complete with weights loaded")

    return run_config


def create_dummy_inputs(batch_size, seq_len, hidden_size, mesh_device, mode="decode"):
    """
    Create dummy inputs for testing, matching test_decoder_block.py format.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_size: Hidden dimension
        mesh_device: TTNN mesh device
        mode: "decode" or "prefill"

    Returns:
        tuple: (tt_input, position_ids, rope_tensors, page_table)
    """

    # Create torch input
    if mode == "decode":
        # For decode: [1, seq_len, batch_size, hidden_size]
        torch_input = torch.randn(1, seq_len, batch_size, hidden_size, dtype=torch.bfloat16)
    else:
        # For prefill: [1, batch_size, seq_len, hidden_size]
        torch_input = torch.randn(1, batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Convert to TTNN
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Create position IDs for decode
    if mode == "decode":
        position_ids = torch.arange(batch_size, dtype=torch.int32).reshape(1, batch_size)
        position_ids_tensor = ttnn.from_torch(
            position_ids,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
            dtype=ttnn.int32,
        )
    else:
        position_ids_tensor = None

    # Create rope tensors (simplified - would need proper implementation)
    rope_tensors = {}  # Would need get_rope_tensors() implementation

    # Create page table (simplified - would need proper implementation)
    page_table = None  # Would need proper page table creation

    return tt_input, position_ids_tensor, rope_tensors, page_table


# Monkey-patch functions to capture intermediates
# This allows us to capture intermediates without modifying the original code

_original_moe_forward = MoE.forward_decode
_original_shared_expert_forward = SharedExpert.forward_decode


def moe_forward_with_capture(x, cfg):
    """Wrapper for MoE.forward_decode that captures output"""
    output = _original_moe_forward(x, cfg)
    if hasattr(capture, "capture"):
        capture.capture(output, "moe_forward_output")
    return output


def shared_expert_forward_with_capture(x, cfg):
    """Wrapper for SharedExpert.forward_decode that captures output"""
    output = _original_shared_expert_forward(x, cfg)
    if hasattr(capture, "capture"):
        capture.capture(output, "shared_expert_forward_output")
    return output


# Apply monkey patches when needed
def enable_intermediate_capture():
    """Enable intermediate capture by monkey-patching forward functions"""
    MoE.forward_decode = moe_forward_with_capture
    SharedExpert.forward_decode = shared_expert_forward_with_capture
    logger.info("Enabled intermediate capture via monkey-patching")


def disable_intermediate_capture():
    """Restore original forward functions"""
    MoE.forward_decode = _original_moe_forward
    SharedExpert.forward_decode = _original_shared_expert_forward
    logger.info("Disabled intermediate capture")
