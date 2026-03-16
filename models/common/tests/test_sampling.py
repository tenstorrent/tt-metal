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
# New tests: top-K logprobs (TG Galaxy only)
# ===========================================================================

# Common TG Galaxy device parametrization for all new tests
TG_SHAPE = [1, 1, 32, 8 * 16032]  # Llama on TG with 8-chip TP sharded vocab
TG_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
}
TG_MESH_SHAPE = (8, 4)
TG_SUB_CORE_GRIDS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)
TG_NUM_TP_DEVICES = 8  # TP dimension for Galaxy


def _skip_if_not_galaxy(mesh_device):
    """Skip test if not running on TG Galaxy (32 devices)."""
    if mesh_device.get_num_devices() != 32:
        pytest.skip(f"Test requires TG Galaxy (32 devices), got {mesh_device.get_num_devices()}")


def _push_topk_test_tensors_to_tg(torch_tensor, gathered_values, gathered_indices, mesh_device):
    """Push logits, topk values, and topk indices to a TG Galaxy mesh device.

    Returns (logits_tt, topk_values_tt, topk_indices_tt).
    """
    logits_tt = ttnn.from_torch(
        torch_tensor,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-1, None), mesh_shape=list(mesh_device.shape)),
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
    return logits_tt, topk_values_tt, topk_indices_tt


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_top_k_log_probs_on_galaxy(shape, mesh_device):
    """Top-K logprobs PCC check on TG Galaxy (32-device 2D mesh).

    Computes logprobs for all gathered top-k tokens on device and compares
    against PyTorch log_softmax (float16) at the same token indices.
    Also verifies sampled token logprob lookup from top-k arrays.
    """
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(1234)
    batch_size = shape[2]

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    # Use float16 for PyTorch reference to match bfloat16 device precision
    log_probs_torch = F.log_softmax(torch_tensor.to(torch.float16), dim=-1)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    calc.set_log_probs_mode([True] * batch_size, num_logprobs=[5] * batch_size)
    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)

    assert result is not None, "Expected LogProbsResult, got None"
    assert isinstance(result, LogProbsResult)

    # Transfer to host and verify
    host_results = calc.transfer_logprobs_to_host(result, argmax_tensor.squeeze())

    # PCC check: compare all gathered top-k logprobs against PyTorch reference
    expected_logprobs = torch.gather(
        log_probs_torch.squeeze(0).squeeze(0),
        dim=-1,
        index=gathered_indices.squeeze(0).squeeze(0).long(),
    )
    composer = calc._build_mesh_composer()
    topk_logprobs_host = ttnn.to_torch(result.topk_logprobs, mesh_composer=composer)
    topk_logprobs_host = topk_logprobs_host[0, 0, ...]

    passing, pcc = comp_pcc(expected_logprobs, topk_logprobs_host, pcc=0.99)
    print(f"Galaxy top-K logprobs PCC={pcc}")
    assert passing, f"Galaxy top-K logprobs PCC failed: {pcc}"

    # Verify sampled token logprob lookup per user
    for user_idx in range(batch_size):
        r = host_results[user_idx]
        assert r is not None
        sampled_id = argmax_tensor[0, 0, user_idx, 0].item()
        torch_lp = log_probs_torch[0, 0, user_idx, sampled_id].item()
        assert abs(r["returned_token"]["logprob"] - torch_lp) < 0.05, (
            f"User {user_idx} sampled logprob mismatch: "
            f"device={r['returned_token']['logprob']:.4f}, torch={torch_lp:.4f}"
        )


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_top_k_log_probs_returns_none_when_not_needed(shape, mesh_device):
    """calculate_topk_log_probs returns None when disabled, and returns
    LogProbsResult with empty top_logprobs when num_logprobs=0."""
    _skip_if_not_galaxy(mesh_device)
    batch_size = shape[2]
    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    # Case 1: logprobs disabled entirely → returns None
    calc.set_log_probs_mode(False, num_logprobs=0)
    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is None, "Expected None when logprobs disabled"

    # Case 2: logprobs enabled, num_logprobs=0 → returns result but host truncates to 0
    calc.set_log_probs_mode(True, num_logprobs=0)
    assert not calc.topk_logprobs_needed
    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None, "Expected LogProbsResult when logprobs enabled"

    sampled_ids = argmax_tensor.squeeze()
    host_results = calc.transfer_logprobs_to_host(result, sampled_ids)

    assert len(host_results) == batch_size
    for i in range(batch_size):
        r = host_results[i]
        assert r is not None, f"User {i} should have result (logprobs enabled)"
        assert r["returned_token"]["token_idx"] == int(sampled_ids[i].item())
        # num_logprobs=0 → top_logprobs arrays are empty
        assert len(r["top_logprobs"]["token_indices"]) == 0, "token_indices should be empty"
        assert len(r["top_logprobs"]["logprobs"]) == 0, "logprobs should be empty"


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_per_user_logprobs_enabled(shape, mesh_device):
    """Mixed per-user logprobs: only even users enabled.

    Verifies the full batch runs logprobs computation and
    transfer_logprobs_to_host returns None for disabled users.
    """
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(42)
    batch_size = shape[2]

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    # Use float16 for PyTorch reference to match bfloat16 device precision
    log_probs_torch = F.log_softmax(torch_tensor.to(torch.float16), dim=-1)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    enable_log_probs = [i % 2 == 0 for i in range(batch_size)]
    num_logprobs_list = [5 if i % 2 == 0 else 0 for i in range(batch_size)]
    calc.set_log_probs_mode(enable_log_probs, num_logprobs=num_logprobs_list)

    assert calc.enable_log_probs is True
    assert calc.topk_logprobs_needed is True

    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None

    sampled_ids = argmax_tensor.squeeze()
    host_results = calc.transfer_logprobs_to_host(result, sampled_ids)
    assert len(host_results) == batch_size

    for i in range(batch_size):
        if enable_log_probs[i]:
            assert host_results[i] is not None, f"User {i} should have logprobs"
            assert host_results[i]["returned_token"]["token_idx"] == int(sampled_ids[i].item())
            assert len(host_results[i]["top_logprobs"]["token_indices"]) <= num_logprobs_list[i]
            # Verify sampled token logprob against PyTorch
            sampled_id = int(sampled_ids[i].item())
            torch_lp = log_probs_torch[0, 0, i, sampled_id].item()
            assert (
                abs(host_results[i]["returned_token"]["logprob"] - torch_lp) < 0.05
            ), f"User {i} sampled logprob mismatch"
        else:
            assert host_results[i] is None, f"User {i} should be None (disabled)"


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_transfer_logprobs_to_host_response_format(shape, mesh_device):
    """Verify OpenAI-compatible response structure from transfer_logprobs_to_host.

    Each user gets a different num_logprobs (1-20).  Checks types, sorting,
    truncation, and token index validity.
    """
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(7777)
    batch_size = shape[2]

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    # Use float16 for PyTorch reference to match bfloat16 device precision
    log_probs_torch = F.log_softmax(torch_tensor.to(torch.float16), dim=-1)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    num_logprobs_list = [(i % MAX_TOP_LOGPROBS) + 1 for i in range(batch_size)]
    calc.set_log_probs_mode([True] * batch_size, num_logprobs=num_logprobs_list)

    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None

    host_results = calc.transfer_logprobs_to_host(result, argmax_tensor.squeeze())

    for i in range(batch_size):
        r = host_results[i]
        assert r is not None, f"User {i} result should not be None"

        # returned_token
        assert isinstance(r["returned_token"]["token_idx"], int)
        assert isinstance(r["returned_token"]["logprob"], float)
        assert r["returned_token"]["logprob"] <= 0.0, "logprob should be <= 0"

        # Verify sampled token logprob against PyTorch
        sampled_id = r["returned_token"]["token_idx"]
        torch_lp = log_probs_torch[0, 0, i, sampled_id].item()
        assert abs(r["returned_token"]["logprob"] - torch_lp) < 0.05, (
            f"User {i} sampled logprob mismatch: " f"device={r['returned_token']['logprob']:.4f}, torch={torch_lp:.4f}"
        )

        # top_logprobs — truncated to num_logprobs_list[i]
        n = num_logprobs_list[i]
        top_lp = r["top_logprobs"]
        assert len(top_lp["token_indices"]) == len(top_lp["logprobs"])
        assert len(top_lp["token_indices"]) <= n

        # sorted descending
        for j in range(len(top_lp["logprobs"]) - 1):
            assert top_lp["logprobs"][j] >= top_lp["logprobs"][j + 1], "Not sorted descending"

        # valid vocab indices
        for idx in top_lp["token_indices"]:
            assert isinstance(idx, int)
            assert 0 <= idx < shape[-1]


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_set_log_probs_mode_validation(shape, mesh_device):
    """Verify set_log_probs_mode internal state for scalar, list, and empty_slots inputs."""
    _skip_if_not_galaxy(mesh_device)
    batch_size = shape[2]
    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size)

    # scalar True, no num_logprobs
    calc.set_log_probs_mode(True)
    assert calc.enable_log_probs is True
    assert all(calc.logprobs_enabled)
    assert not calc.topk_logprobs_needed

    # scalar True, scalar num_logprobs=5
    calc.set_log_probs_mode(True, num_logprobs=5)
    assert calc.topk_logprobs_needed is True
    assert all(n == 5 for n in calc.num_logprobs)

    # per-user mixed
    enable_list = [True, False, True] + [False] * (batch_size - 3)
    num_lp_list = [10, 0, 3] + [0] * (batch_size - 3)
    calc.set_log_probs_mode(enable_list, num_logprobs=num_lp_list)
    assert calc.enable_log_probs is True
    assert calc.topk_logprobs_needed is True
    assert calc.logprobs_enabled == enable_list
    assert calc.num_logprobs == num_lp_list

    # all disabled
    calc.set_log_probs_mode(False, num_logprobs=0)
    assert calc.enable_log_probs is False
    assert not calc.topk_logprobs_needed

    # enable=True, num_logprobs=0 → no top-k needed
    calc.set_log_probs_mode(True, num_logprobs=0)
    assert calc.enable_log_probs is True
    assert not calc.topk_logprobs_needed

    # empty_slots partial update
    calc.set_log_probs_mode(False, num_logprobs=0)
    calc.set_log_probs_mode([True, True], num_logprobs=[10, 15], empty_slots=[2, 5])
    assert calc.logprobs_enabled[2] is True
    assert calc.logprobs_enabled[5] is True
    assert calc.logprobs_enabled[0] is False
    assert calc.num_logprobs[2] == 10
    assert calc.num_logprobs[5] == 15
    assert calc.enable_log_probs is True
    assert calc.topk_logprobs_needed is True

    # empty_slots scalar broadcast
    calc.set_log_probs_mode(False, num_logprobs=0)
    calc.set_log_probs_mode(True, num_logprobs=7, empty_slots=[0, 3, 4])
    assert all(calc.logprobs_enabled[i] for i in [0, 3, 4])
    assert calc.logprobs_enabled[1] is False
    assert all(calc.num_logprobs[i] == 7 for i in [0, 3, 4])

    # consecutive empty_slots preserves prior state
    calc.set_log_probs_mode([True], num_logprobs=[20], empty_slots=[1])
    assert calc.logprobs_enabled[0] is True
    assert calc.logprobs_enabled[1] is True
    assert calc.num_logprobs[0] == 7
    assert calc.num_logprobs[1] == 20


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_top_k_logprobs_pcc_torch_vs_tt(shape, mesh_device):
    """Compare host (PyTorch float16) vs device (bfloat16) logprobs for full batch.

    For every user in the batch:
    1. Computes log-softmax on host (PyTorch float16 to match bfloat16 precision).
    2. Runs calculate_topk_log_probs on device (bfloat16).
    3. Transfers results via transfer_logprobs_to_host.
    4. Checks sampled token logprob within absolute tolerance.
    5. Checks top logprobs PCC >= 0.97 with abs-error fallback for low-variance cases.

    Uses 32 logprobs for PCC comparison (wider dynamic range improves correlation).
    User-facing num_logprobs stays at MAX_TOP_LOGPROBS=20.
    """
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(9999)
    batch_size = shape[2]
    # Use 32 for PCC comparison (wider spread → better correlation signal).
    # bfloat16 top-20 values often cluster in narrow range, making PCC unreliable.
    requested_logprobs = MAX_TOP_LOGPROBS

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size)

    torch_tensor = torch.randn(shape).to(torch.bfloat16)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    # Use float16 for PyTorch reference to match bfloat16 device precision
    log_probs_torch = F.log_softmax(torch_tensor, dim=-1, dtype=torch.bfloat16)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    calc.set_log_probs_mode([True] * batch_size, num_logprobs=[requested_logprobs] * batch_size)

    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None, "Expected LogProbsResult"

    sampled_ids = argmax_tensor.squeeze()
    host_results = calc.transfer_logprobs_to_host(result, sampled_ids)
    assert len(host_results) == batch_size

    for user in range(batch_size):
        r = host_results[user]
        assert r is not None, f"User {user} result is None"

        # Sampled token logprob
        device_sampled_lp = r["returned_token"]["logprob"]
        token_idx = r["returned_token"]["token_idx"]
        torch_sampled_lp = log_probs_torch[0, 0, user, token_idx].item()
        assert abs(device_sampled_lp - torch_sampled_lp) < 0.05, (
            f"User {user} sampled logprob mismatch: " f"device={device_sampled_lp:.4f}, torch={torch_sampled_lp:.4f}"
        )

        # Top logprobs PCC
        top_indices = r["top_logprobs"]["token_indices"]
        top_lps_device = torch.tensor(r["top_logprobs"]["logprobs"], dtype=torch.float32)
        assert (
            len(top_indices) == requested_logprobs
        ), f"User {user}: expected {requested_logprobs} logprobs, got {len(top_indices)}"

        top_lps_torch = log_probs_torch[0, 0, user, top_indices].float()

        # With 32 logprobs the dynamic range is wide enough for strict PCC >= 0.99
        passing, pcc = comp_pcc(top_lps_torch.unsqueeze(0), top_lps_device.unsqueeze(0), pcc=0.98)
        assert passing, (
            f"User {user} top-{requested_logprobs} logprobs PCC failed: {pcc}\n"
            f"  device: {top_lps_device[:5].tolist()}...\n"
            f"  torch:  {top_lps_torch[:5].tolist()}..."
        )
