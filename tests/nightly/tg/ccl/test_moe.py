# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

# Load DeepSeek conftest so fixtures (set_deterministic_env, hf_config, cache_path, ccl, etc.) are available
pytest_plugins = ["models.demos.deepseek_v3.conftest"]

import pytest
import torch
from loguru import logger

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.conftest import PREFILL_SEQ_LENS
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoEInferTest
from models.demos.deepseek_v3.tt.moe import MoE
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.utils.config_helpers import dequantize
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)
from tests.nightly.tg.ccl.test_moe_compute_6U import (
    create_sharded_memory_config,
    create_torch_w0,
    create_torch_w1,
    create_torch_w2,
    gen_expert_mapping,
    gen_sparse_buffer_and_indices,
    prepare_w0_w1_tensor,
    prepare_w2_tensor,
    tt_to_torch_dtype,
)


def load_expert_weights_from_state_dict(
    state_dict,
    num_layers,
    experts_per_device,
    hidden_size,
    N,
    expert_start_index=0,
    block_shape=None,
):
    """
    Load expert weights from a reference model state_dict (e.g. DeepseekV3MoEInferTest).

    state_dict keys expected: experts.{e}.gate_proj.weight, experts.{e}.up_proj.weight,
    experts.{e}.down_proj.weight for e in [expert_start_index, expert_start_index + experts_per_device).
    When quantized, experts.{e}.{proj}.weight_scale_inv must also be present.

    Args:
        block_shape: e.g. (128, 128) for dequantization. If None and weight_scale_inv exists,
            weights are used as-is (may be wrong for quantized weights).

    Returns:
        torch_w0: (num_layers, experts_per_device, hidden_size, N) - gate_proj per expert
        torch_w1: (num_layers, experts_per_device, hidden_size, N) - up_proj per expert
        torch_w2: (num_layers, experts_per_device, N, hidden_size) - down_proj per expert
    """

    def _load_weight(weight_key, scale_inv_key):
        w = state_dict[weight_key]
        if scale_inv_key in state_dict and block_shape is not None:
            w = dequantize(w, state_dict[scale_inv_key], block_shape)
        return w.to(torch.bfloat16)

    torch_w0 = torch.empty((num_layers, experts_per_device, hidden_size, N), dtype=torch.bfloat16)
    torch_w1 = torch.empty((num_layers, experts_per_device, hidden_size, N), dtype=torch.bfloat16)
    torch_w2 = torch.empty((num_layers, experts_per_device, N, hidden_size), dtype=torch.bfloat16)

    for layer_id in range(num_layers):
        for local_e in range(experts_per_device):
            e = expert_start_index + local_e
            # PyTorch Linear stores (out_features, in_features)
            gate = _load_weight(f"experts.{e}.gate_proj.weight", f"experts.{e}.gate_proj.weight_scale_inv")
            up = _load_weight(f"experts.{e}.up_proj.weight", f"experts.{e}.up_proj.weight_scale_inv")
            down = _load_weight(f"experts.{e}.down_proj.weight", f"experts.{e}.down_proj.weight_scale_inv")
            torch_w0[layer_id, local_e] = gate.T  # (hidden_size, N)
            torch_w1[layer_id, local_e] = up.T  # (hidden_size, N)
            torch_w2[layer_id, local_e] = down.T  # (N, hidden_size)

    return torch_w0, torch_w1, torch_w2


def get_moe_compute_result(
    input_tensor,
    expert_indices,
    expert_scores,
    mesh_device,
    total_tokens,
    experts,
    selected_experts_k,
    num_layers,
    hidden_size,
    N,
    cluster_axis=1,
    dtype=ttnn.bfloat16,
    enable_trace=False,
    state_dict=None,
    block_shape=None,
):
    """
    input_tensor is the input tensor to the moe_compute op, the same as sparse_buffer in test_moe_compute_6U.py
    total_tokens is the total number of tokens in the batch, equal to num_tokens below
    experts is the total number of experts, equal to hf_config.n_routed_experts
    selected_experts_k is the number of experts to select for each token, equal to hf_config.num_experts_per_tok
    num_layers is 1
    hidden_size is the hidden size of the model, equal to hf_config.hidden_size
    N is the intermediate dimenion of each expert, equal to hf_config.moe_intermediate_size
    cluster axis is the axis along which the tokens are sharded, this should correspond to line 82 in moe.py

    Unsure:
    expert_mapping
    num_layers
    cluster_axis
    """
    mesh_shape = mesh_device.shape
    num_devices = mesh_shape[0] * mesh_shape[1]
    num_dispatch_devices = mesh_shape[cluster_axis] if cluster_axis is not None else num_devices

    tokens_per_device = total_tokens // num_dispatch_devices
    experts_per_device = experts // num_devices

    logger.info(f"Test configuration:")
    logger.info(f"  mesh_shape: {mesh_shape}")
    logger.info(f"  cluster_axis: {cluster_axis}")
    logger.info(f"  num_devices: {num_devices}, num_dispatch_devices: {num_dispatch_devices}")
    logger.info(f"  tokens_per_device: {tokens_per_device}, total_tokens: {total_tokens}")
    logger.info(
        f"  experts: {experts}, selected_experts_k: {selected_experts_k}, experts_per_device: {experts_per_device}"
    )
    logger.info(f"  hidden_size: {hidden_size}")
    logger.info(f"  dtype: {dtype}")

    #########################################
    # CREATE TILIZE INPUT TENSORS AND GOLDENS
    #########################################

    # Drain tilize core is core (5,9) where indices and scores are sharded
    tilize_drain_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(5, 9), ttnn.CoreCoord(5, 9))})

    #### Expert mapping - per-device [num_devices, experts], replicated on every device ###
    # Each device gets its own row after sharding, but since it's replicated,
    # we give each device the full tensor and it uses its own row.
    # Expert mapping is constant across all runs.
    expert_mapping = gen_expert_mapping(experts, mesh_shape, cluster_axis)
    expert_mapping_mem_config = ttnn.DRAM_MEMORY_CONFIG
    tt_expert_mapping = ttnn.from_torch(
        expert_mapping,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint16,
        memory_config=expert_mapping_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Sparse memory config
    sparse_mem_config = ttnn.L1_MEMORY_CONFIG

    # Create L1 sharded memory config for indices on drain tilizer core
    # expert_indices shape per device: [tokens_per_device, selected_experts_k] (after shard along dispatch axis)
    # But we need the all-gathered version, so shape is [num_dispatch_devices * tokens_per_device, selected_experts_k]
    # which is [total_tokens, selected_experts_k]
    expert_indices_shard_shape = [total_tokens, selected_experts_k]
    expert_indices_mem_config = create_sharded_memory_config(tilize_drain_core, expert_indices_shard_shape, ttnn.uint16)

    # Create L1 sharded memory config for indices and scores on drain tilizer core
    expert_scores_shard_shape = [total_tokens, selected_experts_k]
    expert_scores_mem_config = create_sharded_memory_config(tilize_drain_core, expert_scores_shard_shape, dtype)

    tt_sparse_buffers = []
    tt_expert_indices_buffers = []
    tt_expert_scores_buffers = []
    expert_indices = expert_indices.to(torch.uint16)
    expert_scores = expert_scores.bfloat16()

    # Build sparse buffer: [num_devices, total_tokens, hidden_size] - each device gets tokens routed to its experts
    sparse_buffer = input_tensor.repeat(num_devices, 1, 1)

    for layer_id in range(num_layers):
        # NOTE:
        # - when running multiple layers we initially create tt_sparse_buffer, tt_expert_indices and tt_expert_scores in DRAM, we'll move to L1 before running moe_compute
        # - we're extremely tight on L1 for a single invocation of the op
        if num_layers == 1:
            init_sparse_mem_config = sparse_mem_config
            init_expert_indices_mem_config = expert_indices_mem_config
            init_expert_scores_mem_config = expert_scores_mem_config
        else:
            init_sparse_mem_config = ttnn.DRAM_MEMORY_CONFIG
            init_expert_indices_mem_config = ttnn.DRAM_MEMORY_CONFIG
            init_expert_scores_mem_config = ttnn.DRAM_MEMORY_CONFIG

        ### Sparse buffer is sharded across devices (dim 0) ###
        tt_sparse_buffer = ttnn.from_torch(
            sparse_buffer,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=init_sparse_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_sparse_buffers.append(tt_sparse_buffer)

        ### Expert indices - all-gathered (replicated on all devices) ###
        # Shape: [num_dispatch_devices, tokens_per_device, K]
        # Flatten to [num_dispatch_devices * tokens_per_device, K] = [total_tokens, K] per device
        # Replicate on all devices
        expert_indices_flat = expert_indices.reshape(total_tokens, selected_experts_k)
        expert_indices_replicated = expert_indices_flat.unsqueeze(0).repeat(num_devices, 1, 1)
        tt_expert_indices = ttnn.from_torch(
            expert_indices_replicated,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=init_expert_indices_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_expert_indices_buffers.append(tt_expert_indices)

        ### Expert scores - same distribution as indices ###
        # expert_scores_flat = expert_scores.reshape(total_tokens, selected_experts_k)
        expert_scores_flat = torch.zeros((total_tokens, selected_experts_k))
        expert_scores_replicated = expert_scores_flat.unsqueeze(0).repeat(num_devices, 1, 1)
        tt_expert_scores = ttnn.from_torch(
            expert_scores_replicated,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            memory_config=init_expert_scores_mem_config,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        tt_expert_scores_buffers.append(tt_expert_scores)

    logger.info(f"Done creating goldens and input tensors")

    #########################################
    # CREATE MATMUL INPUT TENSORS
    #########################################

    in0_dtype = ttnn.bfloat16
    w0_dtype = ttnn.bfloat4_b

    # --------------------------------------------------------------------------
    # Shard grid
    # --------------------------------------------------------------------------
    MATMUL_FULL_CORES = {0, 1, 8, 9}
    MATMUL_PAD_CORES = {2, 3, 4, 5, 6, 7, 10, 11}

    in0_core_coords = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, 0)
    core2dram = {}
    for dram_bank_id, core_coords in enumerate(in0_core_coords):
        core2dram[core_coords] = dram_bank_id

    in0_num_cores = len(in0_core_coords)

    # Make a new list of core coords that are sorted in decreasing order by y coordinate and then x coordinate.
    in0_core_coords_sorted = sorted(in0_core_coords, key=lambda x: (x.y, x.x), reverse=True)

    ring2cores = {}
    for ring_pos, core_coord in enumerate(in0_core_coords_sorted):
        # key: ring_pos, value: (core_coord, dram_bank_id, pad_flag)
        ring2cores[ring_pos] = (core_coord, core2dram[core_coord], 1 if ring_pos in MATMUL_PAD_CORES else 0)

    in0_core_range = [ttnn.CoreRange(ring2cores[i][0], ring2cores[i][0]) for i in range(in0_num_cores)]
    in0_core_range_set = ttnn.CoreRangeSet(in0_core_range)

    num_dram_banks = 12

    dram_core_coords = [ttnn.CoreCoord(ring2cores[i][1], 0) for i in range(in0_num_cores)]
    dram_core_range = [ttnn.CoreRange(dram_core_coord, dram_core_coord) for dram_core_coord in dram_core_coords]
    dram_core_range_set = ttnn.CoreRangeSet(dram_core_range)

    if state_dict is not None:
        # Load reference weights per device and shard across mesh so each device has its experts
        w0_w1_list = []
        w2_list = []
        for device_id in range(num_devices):
            expert_start = device_id * experts_per_device
            torch_w0_d, torch_w1_d, torch_w2_d = load_expert_weights_from_state_dict(
                state_dict,
                num_layers,
                experts_per_device,
                hidden_size,
                N,
                expert_start_index=expert_start,
                block_shape=block_shape,
            )
            w0_w1_list.append(
                prepare_w0_w1_tensor(torch_w0_d, torch_w1_d, num_layers, experts_per_device, hidden_size, N, ring2cores)
            )
            w2_list.append(prepare_w2_tensor(torch_w2_d, num_layers, experts_per_device, N, hidden_size, ring2cores))
        torch_w0_w1_reordered = torch.stack(w0_w1_list, dim=0)
        torch_w2_reordered = torch.stack(w2_list, dim=0)
        w0_w1_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        w2_mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=0)
        logger.info("Loaded expert weights from state_dict (sharded by device)")
    else:
        torch_w0 = create_torch_w0(num_layers, experts_per_device, hidden_size, N)
        torch_w1 = create_torch_w1(num_layers, experts_per_device, hidden_size, N)
        torch_w2 = create_torch_w2(num_layers, experts_per_device, N, hidden_size)
        torch_w0_w1_reordered = prepare_w0_w1_tensor(
            torch_w0, torch_w1, num_layers, experts_per_device, hidden_size, N, ring2cores
        )
        torch_w2_reordered = prepare_w2_tensor(torch_w2, num_layers, experts_per_device, N, hidden_size, ring2cores)
        w0_w1_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        w2_mesh_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w0_w1
    # Tensor shape: (num_layers, experts_per_device, hidden_size, 4608) -> padded and reordered to (12, num_layers, experts_per_device, 6, hidden_size, 64)
    # ------------------------------------------------------------------------
    w0_w1_shard_height = num_layers * experts_per_device * 3 * hidden_size
    w0_w1_shard_width = 4 * ttnn.TILE_SIZE

    w0_w1_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w0_w1_shard_height, w0_w1_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w0_w1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w0_w1_shard_spec)

    # ------------------------------------------------------------------------
    # Create DRAM shard spec for w2
    # Tensor shape: (num_layers, experts_per_device, N, hidden_size) -> padded and reordered to (12, num_layers, experts_per_device, 5, N + 192, 128)
    # ------------------------------------------------------------------------
    w2_shard_height = num_layers * experts_per_device * 5 * (N + 192)
    w2_shard_width = 4 * ttnn.TILE_SIZE

    w2_shard_spec = ttnn.ShardSpec(
        dram_core_range_set, (w2_shard_height, w2_shard_width), ttnn.ShardOrientation.ROW_MAJOR
    )

    w2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, w2_shard_spec)

    # Create tt_w0_w1 tensor with DRAM sharding
    tt_w0_w1 = ttnn.from_torch(
        torch_w0_w1_reordered,
        dtype=w0_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w0_w1_mem_config,
        mesh_mapper=w0_w1_mesh_mapper,
    )

    # Create tt_w2 tensor with DRAM sharding
    tt_w2 = ttnn.from_torch(
        torch_w2_reordered,
        dtype=w0_dtype,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=w2_mem_config,
        mesh_mapper=w2_mesh_mapper,
    )

    #########################################
    # RUN OP
    #########################################

    def run_op():
        moe_compute_outputs = []
        for layer_id in range(num_layers):
            # if only running a single layer, we can fit the single set of inputs in L1 initially
            # otherwise with multiple layers, and multiple sets of inputs, we need to move inputs into L1 before a given
            if num_layers == 1:
                tt_sparse_buffer = tt_sparse_buffers[0]
                tt_expert_indices = tt_expert_indices_buffers[0]
                tt_expert_scores = tt_expert_scores_buffers[0]
            else:
                tt_sparse_buffer = ttnn.to_memory_config(tt_sparse_buffers[layer_id], memory_config=sparse_mem_config)
                tt_expert_indices = ttnn.to_memory_config(
                    tt_expert_indices_buffers[layer_id], memory_config=expert_indices_mem_config
                )
                tt_expert_scores = ttnn.to_memory_config(
                    tt_expert_scores_buffers[layer_id], memory_config=expert_scores_mem_config
                )

            # run the op
            (
                l1_per_expert_total_tokens_output_tensor,
                l1_expert_activation_output_tensor,
                l1_e_t_output_tensor,
                l1_output_tensor,  # 1,2,32,7168
            ) = ttnn.experimental.moe_compute(
                tt_sparse_buffer,  # 1, 128, 7168 (1, num_tokens, hidden_size)
                tt_expert_indices,  # 1, 128, 1 (1, num_tokens, selected_experts_k)
                tt_expert_scores,  # 1, 128, 1 (1, num_tokens, selected_experts_k)
                tt_expert_mapping,  # 16, 32 (num_devices, experts)
                tt_w0_w1,  # 1, 12, 1, 2, 3, 7168, 128
                tt_w2,  # 1, 12, 1, 2, 5, 2240, 128
                layer_id=layer_id,  # 0
                cluster_axis=cluster_axis,  # 1
            )

            # dram_output_tensor = ttnn.to_memory_config(l1_output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            output_tensor = ttnn.to_torch(l1_output_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
            moe_compute_outputs.append(output_tensor)

            # deallocate L1 inputs
            # if running with multiple layers, we have to deallocate previous inputs to free up L1 space
            # we still have the DRAM version of the input tensor after deallocating the L1 version
            if num_layers != 1:
                ttnn.deallocate(tt_sparse_buffer)
                ttnn.deallocate(tt_expert_indices)
                ttnn.deallocate(tt_expert_scores)

            # deallocate L1 outputs and save output to verify later
            ttnn.deallocate(l1_per_expert_total_tokens_output_tensor)
            ttnn.deallocate(l1_expert_activation_output_tensor)
            ttnn.deallocate(l1_e_t_output_tensor)
            ttnn.deallocate(l1_output_tensor)

        return moe_compute_outputs

    if enable_trace:
        # Compile the op
        run_op()
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        moe_compute_outputs = run_op()
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")
    else:
        moe_compute_outputs = run_op()

    return moe_compute_outputs


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek MLP model using local implementation."""
    torch.use_deterministic_algorithms(True)
    # Note : Running Reference MoE without shared experts
    hf_config.n_shared_experts = None
    hf_config.num_experts_per_tok = 1  # the same as selected_experts_k
    hf_config.n_routed_experts = 32  # the same as experts
    hf_config.moe_intermediate_size = 2048  # the same as N
    hf_config.hidden_size = 7168  # the same as hidden_size
    return DeepseekV3MoEInferTest(hf_config).eval()


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "trace_region_size": 500000,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape, mesh_device",
    [
        pytest.param((1, 16), (1, 16), id="1x16_grid"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,
    ],
)
@pytest.mark.parametrize(
    "mode,num_tokens",
    [
        ("decode", 512),
    ],
)
def test_forward_pass(
    mode,
    num_tokens,
    set_deterministic_env,
    reference_model,
    hf_config,
    cache_path,
    mesh_shape,
    mesh_device,
    ccl,
    topk_fallback,
):
    """Test forward pass against reference model."""
    # Get state dict from actual model - pass directly to convert_weights
    state_dict = add_inv_scale_to_state_dict(
        reference_model.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Create input tensor
    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_output, expert_indices, expert_scores = reference_model(torch_input)  # 1, 128, 7168

    weight_config = get_test_weight_config(
        MoE, hf_config, (state_dict,), cache_path, mesh_device, force_recalculate=False
    )

    # Generate appropriate config using utility function
    model_config = get_model_config(MoE, mode, hf_config, mesh_device, topk_fallback=topk_fallback)

    # Create a new model state with CCL
    model_state = MoE.create_state(hf_config, mesh_device, ccl)

    # Create a new model shared state
    model_shared_state = MoE.create_shared_state(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # get_moe_compute_result uses 6U-specific layout (1x16 mesh, 12 DRAM banks). Skip on other meshes to avoid segfault.
    if tuple(mesh_device.shape) == (1, 16):
        # Use reference model weights so moe_compute output can be verified against reference.
        # state_dict may have quantized weights + weight_scale_inv; block_shape enables dequantization.
        tt_output_moe_compute = get_moe_compute_result(
            input_tensor=torch_input,
            expert_indices=expert_indices,
            expert_scores=expert_scores,
            mesh_device=mesh_device,
            total_tokens=num_tokens,
            experts=hf_config.n_routed_experts,
            selected_experts_k=hf_config.num_experts_per_tok,
            num_layers=1,
            hidden_size=hf_config.hidden_size,
            N=hf_config.moe_intermediate_size,
            cluster_axis=1,
            dtype=ttnn.bfloat16,
            enable_trace=False,
            state_dict=state_dict,
            block_shape=hf_config.quantization_config["weight_block_size"],
        )
        # moe_compute returns (16, 2, 32, 7168) after ConcatMeshToTensor: 16 devices, 2 experts/device, 32 tokens/device
        # Reduce expert dim (for topk=1 only one expert contributes per token) and reshape to (1, 512, 7168)
        output_tensor = tt_output_moe_compute[0]
        if isinstance(output_tensor, (list, tuple)):
            output_tensor = output_tensor[0]
        # (16, 2, 32, 7168) -> sum over experts -> (16, 32, 7168) -> (1, 512, 7168)
        tt_output_torch = output_tensor.sum(dim=1).reshape(1, num_tokens, hf_config.hidden_size)
        reference_flat = reference_output.reshape(1, -1, reference_output.shape[-1])
        _pcc_passed, pcc_val = comp_pcc(tt_output_torch, reference_flat, 0.98)
        assert_hidden_dim_pcc(tt_output_torch, reference_flat, 0.98)

    # Verify output memory config matches expected
    """
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"MoE output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )


    # Compare outputs using utility function
    logger.info(f"Mode: {mode}, Num tokens: {num_tokens}")
    assert_hidden_dim_pcc(tt_output_torch, reference_output.unsqueeze(0), pcc_required=0.98)
    """
    # Cleanup: tt_output_moe_compute holds torch tensors from to_torch, no ttnn deallocation needed


if __name__ == "__main__":
    pytest.main([__file__])
