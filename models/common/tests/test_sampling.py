# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.sampling import LogProbsCalculator
from models.common.sampling.tt_log_probs import MAX_TOP_LOGPROBS, LogProbsResult
from models.common.utility_functions import comp_pcc


# ---------------------------------------------------------------------------
# Helper: simulate per-device top-k gather (mirrors TTSampling behaviour)
# ---------------------------------------------------------------------------
def _simulate_gathered_topk(torch_logits, num_devices, top_k=32):
    """Simulate the per-device top-k + all-gather that TTSampling performs.

    Args:
        torch_logits: Full logits tensor, shape (1, 1, B, V).
        num_devices: Number of TP devices.
        top_k: Per-device top-k count.

    Returns:
        gathered_values: (1, 1, B, num_devices * top_k) raw logit values.
        gathered_indices: (1, 1, B, num_devices * top_k) global vocab indices.
    """
    V = torch_logits.shape[-1]
    shard_size = V // num_devices
    all_values = []
    all_indices = []
    for d in range(num_devices):
        shard = torch_logits[:, :, :, d * shard_size : (d + 1) * shard_size]
        vals, local_idx = torch.topk(shard, top_k, dim=-1)
        global_idx = local_idx + d * shard_size
        all_values.append(vals)
        all_indices.append(global_idx)
    gathered_values = torch.cat(all_values, dim=-1)
    gathered_indices = torch.cat(all_indices, dim=-1)
    return gathered_values, gathered_indices


# ===========================================================================
# Existing tests (sampled-token-only logprobs)
# ===========================================================================


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_log_probs_calculation(shape, mesh_device):
    seed = 1234
    torch.manual_seed(seed)

    log_probs_calculator = LogProbsCalculator(mesh_device)

    torch_tensor = torch.randn(shape)
    # shuffle the tensor in last 2 dimensions
    for i in range(shape[-2]):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )
    # Push inputs to device
    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_indices_tensor = ttnn.from_torch(
        indices_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator.set_log_probs_mode(True)
    tt_log_probs = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    log_probs_tt_host = ttnn.to_torch(tt_log_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    log_probs_tt_host = log_probs_tt_host[:, :, :1, :32]

    # Calculate log-probs for each user on each chip using torch
    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)
    log_probs_torch_argmax = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)
    log_probs_torch_argmax = torch.reshape(log_probs_torch_argmax, (1, 1, 1, 32))

    passing, pcc = comp_pcc(log_probs_torch_argmax, log_probs_tt_host, pcc=0.99)
    print(f"pcc={pcc}")

    assert passing, f"Assertion failed, PCC={pcc}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_log_probs_returns_none_when_disabled(shape, mesh_device):
    """Test that calculate_log_probs returns None when enable_log_probs is False."""
    log_probs_calculator = LogProbsCalculator(mesh_device)

    torch_tensor = torch.randn(shape)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_indices_tensor = ttnn.from_torch(
        indices_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # Log probs disabled (default) - should return None
    log_probs_calculator.set_log_probs_mode(False)
    result = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    assert result is None, f"Expected None when log_probs disabled, got {type(result)}"

    # Log probs enabled - should return a tensor (not None)
    log_probs_calculator.set_log_probs_mode(True)
    num_devices = mesh_device.get_num_devices()
    result = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    if num_devices in (8, 32) and log_probs_calculator.num_devices_for_sharding >= 2:
        assert result is not None, "Expected tensor when log_probs enabled on supported device"
    else:
        assert result is None, "Expected None on unsupported device count"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 16032],  # llama on TG with 8 chips sharded vocab
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            }
        ),
    ],
    indirect=True,
    ids=["fabric_linear"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_log_probs_with_sub_core_grids_on_galaxy(shape, mesh_device):
    seed = 1234
    torch.manual_seed(seed)

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    log_probs_calculator = LogProbsCalculator(mesh_device, sub_core_grids)

    torch_tensor = torch.randn(shape)
    # shuffle the tensor in last 2 dimensions
    for i in range(shape[-2]):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )

    if mesh_device.get_num_devices() == 8:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
    elif mesh_device.get_num_devices() == 32:
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=list(mesh_device.shape))
    else:
        raise ValueError(f"Unsupported number of devices: {mesh_device.get_num_devices()}")

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=mesh_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_indices_tensor = ttnn.from_torch(
        indices_tensor,
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    log_probs_calculator.set_log_probs_mode(True)
    tt_log_probs = log_probs_calculator.calculate_log_probs(logits_tensor, ttnn_indices_tensor)
    log_probs_tt_host = ttnn.to_torch(tt_log_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    # slice from (1,1,32,256) -> (1,1,1,32)
    log_probs_tt_host = log_probs_tt_host[:, :, :1, :32]

    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)
    log_probs_torch_argmax = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)
    log_probs_torch_argmax = torch.reshape(log_probs_torch_argmax, (1, 1, 1, 32))

    passing, pcc = comp_pcc(log_probs_torch_argmax, log_probs_tt_host, pcc=0.99)
    print(f"pcc={pcc}")

    assert passing, f"Assertion failed, PCC={pcc}"


# ===========================================================================
# New tests: top-K logprobs
# ===========================================================================


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K with 8 TP shards
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_top_k_log_probs_on_t3k(shape, mesh_device):
    """Test top-K logprobs computation on T3K (8-device 1D mesh).

    Verifies that device-computed top-K logprobs match PyTorch log_softmax
    values for the same tokens. Uses PCC >= 0.99 threshold for bfloat16.
    """
    seed = 1234
    torch.manual_seed(seed)
    batch_size = shape[2]
    num_devices = 8
    top_k = 32

    log_probs_calculator = LogProbsCalculator(mesh_device, batch_size=batch_size)

    # Generate random logits and shuffle per batch item
    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    # Ground truth: full log-softmax from PyTorch
    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)

    # Simulate gathered top-k (what TTSampling produces after all-gather)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, num_devices, top_k)
    # gathered_values shape: (1, 1, 32, 256)
    # gathered_indices shape: (1, 1, 32, 256)

    # Get sampled indices (argmax for this test)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    # Push tensors to device
    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_values_tt = ttnn.from_torch(
        gathered_values,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_indices_tt = ttnn.from_torch(
        gathered_indices.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Enable top-K logprobs (num_logprobs=5 for all users)
    enable_log_probs = [True] * batch_size
    num_logprobs_list = [5] * batch_size
    log_probs_calculator.set_log_probs_mode(enable_log_probs, num_logprobs=num_logprobs_list)
    assert log_probs_calculator.top_k_logprobs_needed, "top_k_logprobs_needed should be True"

    # Calculate top-K logprobs on device
    result = log_probs_calculator.calculate_top_k_log_probs(logits_tensor, topk_values_tt, topk_indices_tt)

    assert result is not None, "Expected LogProbsResult, got None"
    assert isinstance(result, LogProbsResult), f"Expected LogProbsResult, got {type(result)}"
    assert result.top_k_logprobs is not None, "top_k_logprobs should not be None when requested"
    assert result.top_k_indices is not None, "top_k_indices should not be None when requested"

    # 1. Verify sampled token logprob by looking it up from top-k arrays
    topk_logprobs_host_all = ttnn.to_torch(
        result.top_k_logprobs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)
    )
    topk_indices_host_all = ttnn.to_torch(
        result.top_k_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)
    )
    K = result.top_k_logprobs.shape[-1]
    topk_logprobs_for_lookup = topk_logprobs_host_all[:, :, :batch_size, :K].squeeze(0).squeeze(0)
    topk_indices_for_lookup = topk_indices_host_all[:, :, :batch_size, :K].squeeze(0).squeeze(0)

    torch_sampled_logprob = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)
    torch_sampled_logprob = torch.reshape(torch_sampled_logprob, (batch_size,))

    for user_idx in range(batch_size):
        sampled_token_id = argmax_tensor[0, 0, user_idx, 0].item()
        match_mask = topk_indices_for_lookup[user_idx] == sampled_token_id
        assert match_mask.any(), f"User {user_idx}: sampled token {sampled_token_id} not found in top-k indices"
        device_sampled_lp = topk_logprobs_for_lookup[user_idx][match_mask][0].item()
        torch_sampled_lp = torch_sampled_logprob[user_idx].item()
        assert abs(device_sampled_lp - torch_sampled_lp) < 0.05, (
            f"User {user_idx} sampled token logprob mismatch: device={device_sampled_lp:.6f}, "
            f"torch={torch_sampled_lp:.6f}"
        )

    # 2. Verify top-K logprobs match PyTorch for the same token indices
    topk_logprobs_host = ttnn.to_torch(result.top_k_logprobs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    K = result.top_k_logprobs.shape[-1]
    topk_logprobs_host = topk_logprobs_host[:, :, :batch_size, :K].squeeze(0).squeeze(0)

    # Compute expected logprobs for the same gathered token indices using PyTorch
    expected_logprobs = torch.gather(
        log_probs_torch.squeeze(0).squeeze(0), dim=-1, index=gathered_indices.squeeze(0).squeeze(0).long()
    )

    passing, pcc = comp_pcc(expected_logprobs, topk_logprobs_host, pcc=0.99)
    print(f"Top-K logprobs PCC={pcc}")
    assert passing, f"Top-K logprobs PCC failed: {pcc}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 16032],  # llama on TG Galaxy with 8 chips sharded vocab
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        (
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            }
        ),
    ],
    indirect=True,
    ids=["fabric_linear"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_top_k_log_probs_on_galaxy(shape, mesh_device):
    """Test top-K logprobs computation on TG Galaxy (32-device 2D mesh).

    Verifies that device-computed top-K logprobs match PyTorch log_softmax
    values for the same tokens on a Galaxy mesh with sub_core_grids.
    """
    seed = 1234
    torch.manual_seed(seed)
    batch_size = shape[2]
    num_devices = 8  # TP dimension for Galaxy
    top_k = 32

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    log_probs_calculator = LogProbsCalculator(mesh_device, sub_core_grids, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, num_devices, top_k)

    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    if mesh_device.get_num_devices() == 32:
        logits_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=list(mesh_device.shape))
    else:
        logits_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=logits_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_values_tt = ttnn.from_torch(
        gathered_values,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_indices_tt = ttnn.from_torch(
        gathered_indices.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    enable_log_probs = [True] * batch_size
    num_logprobs_list = [10] * batch_size
    log_probs_calculator.set_log_probs_mode(enable_log_probs, num_logprobs=num_logprobs_list)

    result = log_probs_calculator.calculate_top_k_log_probs(logits_tensor, topk_values_tt, topk_indices_tt)

    assert result is not None, "Expected LogProbsResult, got None"
    assert result.top_k_logprobs is not None, "top_k_logprobs should not be None"

    # Verify sampled token logprob by looking it up from top-k arrays
    topk_logprobs_host_all = ttnn.to_torch(
        result.top_k_logprobs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)
    )
    topk_indices_host_all = ttnn.to_torch(
        result.top_k_indices, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)
    )
    K = result.top_k_logprobs.shape[-1]
    topk_logprobs_for_lookup = topk_logprobs_host_all[:, :, :batch_size, :K].squeeze(0).squeeze(0)
    topk_indices_for_lookup = topk_indices_host_all[:, :, :batch_size, :K].squeeze(0).squeeze(0)

    torch_sampled_logprob = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)
    torch_sampled_logprob = torch.reshape(torch_sampled_logprob, (batch_size,))

    for user_idx in range(batch_size):
        sampled_token_id = argmax_tensor[0, 0, user_idx, 0].item()
        match_mask = topk_indices_for_lookup[user_idx] == sampled_token_id
        assert match_mask.any(), f"User {user_idx}: sampled token {sampled_token_id} not found in top-k indices"
        device_sampled_lp = topk_logprobs_for_lookup[user_idx][match_mask][0].item()
        torch_sampled_lp = torch_sampled_logprob[user_idx].item()
        assert abs(device_sampled_lp - torch_sampled_lp) < 0.05, (
            f"User {user_idx} sampled token logprob mismatch: device={device_sampled_lp:.6f}, "
            f"torch={torch_sampled_lp:.6f}"
        )

    # Verify top-K logprobs
    topk_logprobs_host = ttnn.to_torch(result.top_k_logprobs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    K = result.top_k_logprobs.shape[-1]
    topk_logprobs_host = topk_logprobs_host[:, :, :batch_size, :K].squeeze(0).squeeze(0)

    expected_logprobs = torch.gather(
        log_probs_torch.squeeze(0).squeeze(0), dim=-1, index=gathered_indices.squeeze(0).squeeze(0).long()
    )

    passing, pcc = comp_pcc(expected_logprobs, topk_logprobs_host, pcc=0.99)
    print(f"Galaxy top-K logprobs PCC={pcc}")
    assert passing, f"Galaxy top-K logprobs PCC failed: {pcc}"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_top_k_log_probs_returns_none_when_not_needed(shape, mesh_device):
    """Test that calculate_top_k_log_probs returns None when logprobs disabled,
    and returns LogProbsResult with top_k=None when only sampled token logprob
    is requested (num_logprobs=0 for all users).
    """
    batch_size = shape[2]
    log_probs_calculator = LogProbsCalculator(mesh_device, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, 8, 32)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_values_tt = ttnn.from_torch(
        gathered_values,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_indices_tt = ttnn.from_torch(
        gathered_indices.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Case 1: logprobs disabled entirely → should return None
    log_probs_calculator.set_log_probs_mode(False, num_logprobs=0)
    result = log_probs_calculator.calculate_top_k_log_probs(logits_tensor, topk_values_tt, topk_indices_tt)
    assert result is None, "Expected None when logprobs disabled"

    # Case 2: logprobs enabled but num_logprobs=0 for all users → sampled token only
    log_probs_calculator.set_log_probs_mode(True, num_logprobs=0)
    assert not log_probs_calculator.top_k_logprobs_needed, "top_k_logprobs_needed should be False"
    result = log_probs_calculator.calculate_top_k_log_probs(logits_tensor, topk_values_tt, topk_indices_tt)
    # With top_k_logprobs_needed=False, this still returns a result with top_k=None
    assert result is not None, "Expected LogProbsResult when logprobs enabled"
    assert result.top_k_logprobs is None, "top_k_logprobs should be None when num_logprobs=0"
    assert result.top_k_indices is None, "top_k_indices should be None when num_logprobs=0"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_per_user_logprobs_enabled(shape, mesh_device):
    """Test that per-user logprobs_enabled array works correctly.

    When only some users have logprobs enabled, the computation still runs
    for the whole batch (since at least one user needs it), and get_host_results
    returns None for users with logprobs disabled.
    """
    seed = 42
    torch.manual_seed(seed)
    batch_size = shape[2]

    log_probs_calculator = LogProbsCalculator(mesh_device, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, 8, 32)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_values_tt = ttnn.from_torch(
        gathered_values,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_indices_tt = ttnn.from_torch(
        gathered_indices.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Enable logprobs for only half the users (even indices), with varying num_logprobs
    enable_log_probs = [i % 2 == 0 for i in range(batch_size)]
    num_logprobs_list = [5 if i % 2 == 0 else 0 for i in range(batch_size)]
    log_probs_calculator.set_log_probs_mode(enable_log_probs, num_logprobs=num_logprobs_list)

    assert log_probs_calculator.enable_log_probs, "enable_log_probs should be True (some users need it)"
    assert log_probs_calculator.top_k_logprobs_needed, "top_k_logprobs_needed should be True"
    assert log_probs_calculator.logprobs_enabled == enable_log_probs, "Per-user logprobs_enabled mismatch"

    result = log_probs_calculator.calculate_top_k_log_probs(logits_tensor, topk_values_tt, topk_indices_tt)
    assert result is not None, "Expected LogProbsResult"

    # Verify get_host_results returns None for disabled users
    sampled_ids = argmax_tensor.squeeze()
    host_results = log_probs_calculator.get_host_results(result, sampled_ids, mesh_device)
    assert len(host_results) == batch_size, f"Expected {batch_size} results, got {len(host_results)}"

    for i in range(batch_size):
        if enable_log_probs[i]:
            assert host_results[i] is not None, f"User {i} should have logprobs result"
            assert "returned_token" in host_results[i], "Missing returned_token key"
            assert "top_logprobs" in host_results[i], "Missing top_logprobs key"
            assert host_results[i]["returned_token"]["token_idx"] == int(sampled_ids[i].item())
            # Check top_logprobs has at most num_logprobs entries
            assert len(host_results[i]["top_logprobs"]["token_indices"]) <= num_logprobs_list[i]
        else:
            assert host_results[i] is None, f"User {i} should have None logprobs result"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_get_host_results_response_format(shape, mesh_device):
    """Test that get_host_results returns correctly structured response objects.

    Verifies the OpenAI-compatible response format:
    {
        "returned_token": {"token_idx": int, "logprob": float},
        "top_logprobs": {"token_indices": [int], "logprobs": [float]}
    }
    """
    seed = 7777
    torch.manual_seed(seed)
    batch_size = shape[2]

    log_probs_calculator = LogProbsCalculator(mesh_device, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, 8, 32)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_values_tt = ttnn.from_torch(
        gathered_values,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_indices_tt = ttnn.from_torch(
        gathered_indices.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Request different num_logprobs per user
    num_logprobs_list = [(i % MAX_TOP_LOGPROBS) + 1 for i in range(batch_size)]  # 1 to 20
    enable_log_probs = [True] * batch_size
    log_probs_calculator.set_log_probs_mode(enable_log_probs, num_logprobs=num_logprobs_list)

    result = log_probs_calculator.calculate_top_k_log_probs(logits_tensor, topk_values_tt, topk_indices_tt)
    assert result is not None

    sampled_ids = argmax_tensor.squeeze()
    host_results = log_probs_calculator.get_host_results(result, sampled_ids, mesh_device)

    for i in range(batch_size):
        r = host_results[i]
        assert r is not None, f"User {i} result should not be None"

        # Verify returned_token structure
        assert isinstance(r["returned_token"]["token_idx"], int), "token_idx should be int"
        assert isinstance(r["returned_token"]["logprob"], float), "logprob should be float"
        assert r["returned_token"]["logprob"] <= 0.0, "logprob should be <= 0 (log-probability)"

        # Verify top_logprobs structure
        n = num_logprobs_list[i]
        top_lp = r["top_logprobs"]
        assert isinstance(top_lp["token_indices"], list), "token_indices should be a list"
        assert isinstance(top_lp["logprobs"], list), "logprobs should be a list"
        assert len(top_lp["token_indices"]) == len(top_lp["logprobs"]), "lengths should match"
        assert (
            len(top_lp["token_indices"]) <= n
        ), f"Expected at most {n} top logprobs, got {len(top_lp['token_indices'])}"

        # Verify logprobs are sorted descending
        if len(top_lp["logprobs"]) > 1:
            for j in range(len(top_lp["logprobs"]) - 1):
                assert (
                    top_lp["logprobs"][j] >= top_lp["logprobs"][j + 1]
                ), f"Top logprobs should be sorted descending, got {top_lp['logprobs'][j]} < {top_lp['logprobs'][j+1]}"

        # All token indices should be valid vocabulary indices
        for idx in top_lp["token_indices"]:
            assert isinstance(idx, int), "Each token index should be int"
            assert 0 <= idx < shape[-1], f"Token index {idx} out of range [0, {shape[-1]})"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_set_log_probs_mode_validation(shape, mesh_device):
    """Test set_log_probs_mode correctly sets internal state for various inputs."""
    batch_size = shape[2]
    calc = LogProbsCalculator(mesh_device, batch_size=batch_size)

    # Test 1: scalar enable_log_probs=True, no num_logprobs
    calc.set_log_probs_mode(True)
    assert calc.enable_log_probs is True
    assert all(calc.logprobs_enabled), "All users should be enabled"
    assert not calc.top_k_logprobs_needed, "top_k not needed when num_logprobs not set"

    # Test 2: scalar enable_log_probs=True, scalar num_logprobs=5
    calc.set_log_probs_mode(True, num_logprobs=5)
    assert calc.enable_log_probs is True
    assert calc.top_k_logprobs_needed is True
    assert all(n == 5 for n in calc.num_logprobs)

    # Test 3: per-user list with mixed values
    enable_list = [True, False, True] + [False] * (batch_size - 3)
    num_lp_list = [10, 0, 3] + [0] * (batch_size - 3)
    calc.set_log_probs_mode(enable_list, num_logprobs=num_lp_list)
    assert calc.enable_log_probs is True  # at least one user enabled
    assert calc.top_k_logprobs_needed is True  # user 0 and 2 need top-k
    assert calc.logprobs_enabled == enable_list
    assert calc.num_logprobs == num_lp_list

    # Test 4: all disabled
    calc.set_log_probs_mode(False, num_logprobs=0)
    assert calc.enable_log_probs is False
    assert not calc.top_k_logprobs_needed

    # Test 5: num_logprobs=0 with enable=True (sampled token only, no top-k)
    calc.set_log_probs_mode(True, num_logprobs=0)
    assert calc.enable_log_probs is True
    assert not calc.top_k_logprobs_needed

    # Test 6: empty_slots partial update — only specified positions are modified
    calc.set_log_probs_mode(False, num_logprobs=0)  # start clean
    assert not calc.enable_log_probs

    # Update only slots 2 and 5
    calc.set_log_probs_mode(
        enable_log_probs=[True, True],
        num_logprobs=[10, 15],
        empty_slots=[2, 5],
    )
    assert calc.logprobs_enabled[2] is True
    assert calc.logprobs_enabled[5] is True
    assert calc.logprobs_enabled[0] is False  # untouched
    assert calc.logprobs_enabled[1] is False  # untouched
    assert calc.num_logprobs[2] == 10
    assert calc.num_logprobs[5] == 15
    assert calc.num_logprobs[0] == 0  # untouched
    assert calc.enable_log_probs is True  # at least one slot enabled
    assert calc.top_k_logprobs_needed is True  # slots 2 and 5 need top-k

    # Test 7: empty_slots with scalar values — broadcasts to all specified slots
    calc.set_log_probs_mode(False, num_logprobs=0)  # reset
    calc.set_log_probs_mode(
        enable_log_probs=True,
        num_logprobs=7,
        empty_slots=[0, 3, 4],
    )
    assert calc.logprobs_enabled[0] is True
    assert calc.logprobs_enabled[3] is True
    assert calc.logprobs_enabled[4] is True
    assert calc.logprobs_enabled[1] is False  # untouched
    assert all(calc.num_logprobs[i] == 7 for i in [0, 3, 4])
    assert calc.num_logprobs[1] == 0  # untouched

    # Test 8: second empty_slots call preserves earlier partial state
    calc.set_log_probs_mode(
        enable_log_probs=[True],
        num_logprobs=[20],
        empty_slots=[1],
    )
    # slot 0,3,4 from previous call should still be set
    assert calc.logprobs_enabled[0] is True
    assert calc.logprobs_enabled[1] is True  # newly set
    assert calc.num_logprobs[0] == 7  # preserved
    assert calc.num_logprobs[1] == 20  # newly set


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 8 * 18992],  # Qwen3 on T3K with 8 TP shards
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
    ],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_top_k_logprobs_pcc_host_vs_device_20_logprobs(shape, mesh_device):
    """Compare host (PyTorch) vs device logprobs for a full batch with 20 top logprobs.

    For every user in the batch this test:
    1. Computes the full log-softmax on host (PyTorch float32 – ground truth).
    2. Runs calculate_top_k_log_probs on device (bfloat16).
    3. Transfers the device top-k logprobs to host via get_host_results.
    4. For each user, looks up the PyTorch logprobs at the same token indices
       that the device returned and checks PCC >= 0.99.

    This covers the full end-to-end path: global stats computation, top-k
    logprob formula, host transfer, sorting, and per-user trimming to 20.
    """
    seed = 9999
    torch.manual_seed(seed)
    batch_size = shape[2]
    num_devices = 8
    top_k = 32
    requested_logprobs = MAX_TOP_LOGPROBS  # 20

    log_probs_calculator = LogProbsCalculator(mesh_device, batch_size=batch_size)

    # Generate random logits and shuffle per batch item for realistic distribution
    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    # Ground truth: full log-softmax from PyTorch (float32)
    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)

    # Simulate gathered top-k (mirrors TTSampling all-gather)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, num_devices, top_k)

    # Sampled token indices (argmax)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    # Push tensors to device
    logits_tensor = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_values_tt = ttnn.from_torch(
        gathered_values,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    topk_indices_tt = ttnn.from_torch(
        gathered_indices.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # All users request 20 logprobs
    enable_log_probs = [True] * batch_size
    num_logprobs_list = [requested_logprobs] * batch_size
    log_probs_calculator.set_log_probs_mode(enable_log_probs, num_logprobs=num_logprobs_list)

    # Run device computation
    result = log_probs_calculator.calculate_top_k_log_probs(logits_tensor, topk_values_tt, topk_indices_tt)
    assert result is not None, "Expected LogProbsResult"

    # Get structured host results (sorted, trimmed to 20 per user)
    sampled_ids = argmax_tensor.squeeze()
    host_results = log_probs_calculator.get_host_results(result, sampled_ids, mesh_device)
    assert len(host_results) == batch_size

    # Compare each user's top-20 device logprobs against PyTorch reference
    for user in range(batch_size):
        r = host_results[user]
        assert r is not None, f"User {user} result is None"

        # -- Sampled token logprob --
        device_sampled_lp = r["returned_token"]["logprob"]
        token_idx = r["returned_token"]["token_idx"]
        torch_sampled_lp = log_probs_torch[0, 0, user, token_idx].item()
        # Allow small tolerance for bfloat16 vs float32
        assert abs(device_sampled_lp - torch_sampled_lp) < 0.05, (
            f"User {user} sampled token logprob mismatch: device={device_sampled_lp:.6f}, "
            f"torch={torch_sampled_lp:.6f}"
        )

        # -- Top-20 logprobs PCC --
        top_indices = r["top_logprobs"]["token_indices"]
        top_lps_device = torch.tensor(r["top_logprobs"]["logprobs"], dtype=torch.float32)
        assert (
            len(top_indices) == requested_logprobs
        ), f"User {user}: expected {requested_logprobs} top logprobs, got {len(top_indices)}"

        # Look up the PyTorch logprobs for the exact same token indices
        top_lps_torch = log_probs_torch[0, 0, user, top_indices].float()

        passing, pcc = comp_pcc(top_lps_torch.unsqueeze(0), top_lps_device.unsqueeze(0), pcc=0.99)
        assert passing, (
            f"User {user} top-{requested_logprobs} logprobs PCC failed: {pcc}\n"
            f"  device: {top_lps_device[:5].tolist()}...\n"
            f"  torch:  {top_lps_torch[:5].tolist()}..."
        )
