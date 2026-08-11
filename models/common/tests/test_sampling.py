# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.common.sampling import (
    LogProbsCalculator,
    SamplingParams,
    SeedManager,
    broadcast_sampling_params,
    format_sampling_params,
)
from models.common.sampling.generator import _hash_request_seed_to_device_seed
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
# Top-K logprobs tests (TG Galaxy only)
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


def _make_host_only_seed_manager(max_batch_size=4):
    return SeedManager(SimpleNamespace(_sampling_dp=1), max_batch_size=max_batch_size)


def test_seed_manager_seed_params_do_not_fallback_to_slot_zero():
    seed_manager = _make_host_only_seed_manager()

    assert seed_manager._seed_from_slot_params([11], 0) == 11
    assert seed_manager._seed_from_slot_params([11], 1) is None
    assert seed_manager._seed_from_slot_params(torch.tensor([22]), 1) is None
    assert seed_manager._seed_from_slot_params(33, 3) == 33


def test_seed_counter_position_alignment_skips_out_of_bounds_slots():
    seed_manager = _make_host_only_seed_manager()

    seed_manager.align_seed_counters_to_positions([101, None, 303], [0, 2], [5], offset=1)

    assert seed_manager.seed_counters == [6, 0, 0, 0]


def test_slot_remap_condense_relabels_destination_and_vacates_source():
    """A condense map moves the source slot's RNG state to its new slot and leaves
    the vacated source unseeded.

    vLLM's condense moves the highest live request down into the lowest empty slot
    (``InputBatch.condense``: ``_slot_remap[empty_index] = _slot_remap[last_req_index]``),
    so a source that is not itself a destination has genuinely been vacated.
    """
    seed_manager = _make_host_only_seed_manager(max_batch_size=4)
    seed_manager.reset_seed([42, 99], [0, 3])  # slot0=42, slot3=99
    assert seed_manager.seeds == [42, None, None, 99]

    # Condense: the request in slot3 moves into empty slot1. remap[1]=3; indices
    # 0/2/3 keep their identity values (the map does not mark slot3 as empty).
    seed_manager.apply_slot_remap(torch.tensor([0, 3, 2, 3], dtype=torch.int32))

    assert seed_manager.seeds[1] == 99  # relabelled into its new slot
    assert seed_manager.seeds[3] is None  # source vacated
    assert seed_manager.seed_counters[3] == 0
    assert seed_manager.seeds[0] == 42  # untouched slot keeps its seed
    assert seed_manager._seed_active is True


def test_slot_remap_identity_is_a_noop():
    """The steady state is an identity map (vLLM pops the remap and resets it to
    identity every step, and lane-DP never condenses at all), which must not touch
    any slot's RNG state."""
    seed_manager = _make_host_only_seed_manager(max_batch_size=4)
    seed_manager.reset_seed([42, 43], [0, 1])

    identity = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    for _ in range(50):
        seed_manager.apply_slot_remap(identity)

    assert seed_manager.seeds == [42, 43, None, None]
    assert seed_manager._seed_active is True


def test_broadcast_sampling_params_preserves_none_list_fields():
    params = SamplingParams(temperature=[1.0, 1.0], top_k=[1, 1], top_p=[1.0, 1.0], seed=[None, 42])

    broadcast = broadcast_sampling_params(params, 0, slot_len=4)

    assert broadcast.seed == [None, None, None, None]


def test_format_sampling_params_uses_device_argmax_sentinel_for_greedy_rows():
    params = format_sampling_params(
        SamplingParams(temperature=0.0, top_k=32, top_p=0.95),
        max_batch_size=32,
    )

    assert params.temperature[0] == 1.0
    assert params.top_k[0] == 1
    assert params.top_p[0] == 0.0


# ---------------------------------------------------------------------------
# Seeded decode reproducibility under async scheduling (#51981).
# Host-only: the generator is built with __new__ and driven with stub modules.
# ---------------------------------------------------------------------------
SEED_TEST_BATCH = 32  # format_sampling_params requires a multiple of 32


class _RecordingSeedManager(SeedManager):
    """SeedManager that records each step's device seed vector instead of pushing it."""

    def __init__(self, max_batch_size=SEED_TEST_BATCH):
        super().__init__(SimpleNamespace(_sampling_dp=1), max_batch_size=max_batch_size)
        self.pushed = []

    def _copy_seeds_to_device(self, new_seeds):
        self.pushed.append(list(new_seeds))


class _StubSamplingModule:
    def __init__(self, max_batch_size=SEED_TEST_BATCH):
        self.seed_manager = _RecordingSeedManager(max_batch_size)
        self.tt_sampling = SimpleNamespace(max_batch_size=max_batch_size)

    def apply_decode_state(self, sampling_params_chunks, *, reset_batch=False, prompt_tokens=None, output_tokens=None):
        pass

    def sample(self, logits=None, tt_out_tok=None, enable_trace=False):
        return logits


def _make_stub_generator(max_batch_size=SEED_TEST_BATCH):
    from models.tt_transformers.tt.generator import Generator

    generator = Generator.__new__(Generator)
    generator.data_parallel = 1
    sampling = _StubSamplingModule(max_batch_size)
    generator.model = [SimpleNamespace(sampling=sampling, sampling_dp=1)]
    return generator, sampling


def _decode_sampling_step(generator, seeds, positions, reload_inputs, max_batch_size=SEED_TEST_BATCH):
    """Run one device-sampling decode step; return the per-slot device seeds pushed."""
    seeds = list(seeds) + [None] * (max_batch_size - len(seeds))
    positions = list(positions) + [-1] * (max_batch_size - len(positions))
    params = SamplingParams(
        temperature=[1.0] * max_batch_size,
        top_k=[32] * max_batch_size,
        top_p=[1.0] * max_batch_size,
        seed=seeds,
    )
    generator.sample_decode_on_device(
        [None],
        sampling_params=params,
        start_pos=[torch.tensor(positions, dtype=torch.int32)],
        reload_inputs=reload_inputs,
    )
    return generator.model[0].sampling.seed_manager.pushed[-1]


def _expected_seed_stream(request_seed, first_position, num_steps):
    """Device seeds for `num_steps` consecutive tokens starting at `first_position`."""
    positions = range(first_position, first_position + num_steps)
    return [_hash_request_seed_to_device_seed(request_seed, pos + 1) for pos in positions]


def test_seed_stream_is_independent_of_host_position_lag():
    """The counter must self-advance from the last authoritative anchor; re-anchoring
    to a lagging host position replays device seeds (#51981)."""
    generator, sampling = _make_stub_generator()

    pushed = [_decode_sampling_step(generator, [7], [100], reload_inputs=True)[0]]
    # Device is at 101, 102, 103; the host reports the previous position and
    # stalls entirely when a readback is late.
    for lagging_host_pos in (100, 101, 101):
        pushed.append(_decode_sampling_step(generator, [7], [lagging_host_pos], reload_inputs=False)[0])

    assert pushed == _expected_seed_stream(7, 100, 4)
    assert len(set(pushed)) == 4  # no replayed seed
    assert sampling.seed_manager.seed_counters[0] == 105  # anchored at 101, one per token


def test_seed_stream_matches_across_different_host_lags():
    """Same seed and true positions, but one run has async overlap engaged (host
    lags) and the other does not. Equal streams is what `seed=` promises."""
    lagged, _ = _make_stub_generator()
    exact, _ = _make_stub_generator()

    lagged_stream = [_decode_sampling_step(lagged, [7], [100], reload_inputs=True)[0]]
    exact_stream = [_decode_sampling_step(exact, [7], [100], reload_inputs=True)[0]]
    for true_pos in (101, 102, 103):
        lagged_stream.append(_decode_sampling_step(lagged, [7], [true_pos - 1], reload_inputs=False)[0])
        exact_stream.append(_decode_sampling_step(exact, [7], [true_pos], reload_inputs=False)[0])

    assert lagged_stream == exact_stream


def test_seed_counters_realign_when_host_inputs_are_authoritative():
    """A batch reset re-anchors every active slot: vLLM may have evicted and
    re-admitted the request elsewhere, so the resident counter is untrustworthy."""
    generator, sampling = _make_stub_generator()

    _decode_sampling_step(generator, [7], [100], reload_inputs=True)
    sampling.seed_manager.seed_counters[0] = 0  # state moved behind our back
    pushed = _decode_sampling_step(generator, [7], [200], reload_inputs=True)

    assert pushed[0] == _hash_request_seed_to_device_seed(7, 201)


def test_newly_seeded_slot_is_aligned_even_on_a_non_authoritative_step():
    """A freshly admitted slot's position comes from its prefill, so it is
    authoritative even when the rest of the batch's host inputs are stale;
    otherwise its reset-to-zero counter starts the stream at the wrong offset."""
    generator, _ = _make_stub_generator()

    _decode_sampling_step(generator, [7], [100], reload_inputs=True)
    # Slot 1 admitted mid-flight at position 5; slot 0 keeps decoding with a lag.
    pushed = _decode_sampling_step(generator, [7, 11], [100, 5], reload_inputs=False)

    assert pushed[0] == _hash_request_seed_to_device_seed(7, 102)  # unmoved, self-advanced
    assert pushed[1] == _hash_request_seed_to_device_seed(11, 6)  # anchored to its prefill position


def test_reset_seed_from_slots_if_needed_reports_the_slots_it_reset():
    seed_manager = _make_host_only_seed_manager()
    seed_manager.reset_seed_from_slots([42, 43, None, None], [0, 1, 2, 3])

    assert seed_manager.reset_seed_from_slots_if_needed([42, 43, None, None], [0, 1, 2, 3]) == []
    assert seed_manager.reset_seed_from_slots_if_needed([42, 99, None, None], [0, 1, 2, 3]) == [1]


def _skip_if_not_galaxy(mesh_device):
    """Skip test if not running on TG Galaxy (32 devices)."""
    if mesh_device.get_num_devices() != 32:
        pytest.skip(f"Test requires TG Galaxy (32 devices), got {mesh_device.get_num_devices()}")


def _push_topk_test_tensors_to_tg(torch_tensor, gathered_values, gathered_indices, mesh_device):
    """Push logits, topk values, and topk indices to a TG Galaxy mesh device."""
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


def _shard_logits_2d_mesh(logits_host, mesh_device):
    """Shard vocab along mesh TP axis (matches test_sampling_1d._make_logits_tt)."""
    cluster_shape = tuple(mesh_device.shape)
    if cluster_shape[-1] >= cluster_shape[-2]:
        shard_dims = (None, -1)
    else:
        shard_dims = (-1, None)
    return ttnn.from_torch(
        logits_host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=shard_dims, mesh_shape=cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=["device_params"],
    ids=["fabric_linear"],
)
def test_log_probs_calculation_shard_tensor_2d_mesh_1x8(mesh_device):
    """LogProbsCalculator with ShardTensor2dMesh on 1×8 — the path Sampling1D uses.

    test_log_probs_calculation shards via ShardTensorToMesh(dim=-1), which does not
    exercise the 1×N _all_gather_cluster_axis bug fixed in tt_log_probs.py.
    """
    if mesh_device.get_num_devices() != 8:
        pytest.skip(f"Test targets 1×8 mesh, got {mesh_device.get_num_devices()} devices")

    batch_size = 32
    vocab_size = 32768
    shape = [1, 1, batch_size, vocab_size]

    torch.manual_seed(42)
    log_probs_calculator = LogProbsCalculator(mesh_device)

    torch_tensor = torch.randn(shape, dtype=torch.bfloat16)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(vocab_size)]

    # Pin a few batch slots to tokens on different chips (4096 tokens/chip on 1×8).
    pinned_tokens = [(0, 100), (1, 20000), (2, 30000), (3, 5000)]  # chips 0, 4, 7, 1
    for batch_idx, token_id in pinned_tokens:
        torch_tensor[:, :, batch_idx, token_id] = 10.0

    argmax_tensor = torch.argmax(torch_tensor.float(), dim=-1, keepdim=True)
    for batch_idx, token_id in pinned_tokens:
        assert argmax_tensor[0, 0, batch_idx, 0].item() == token_id
    indices_tensor = argmax_tensor.reshape(
        argmax_tensor.shape[0], argmax_tensor.shape[1], argmax_tensor.shape[-1], argmax_tensor.shape[-2]
    )

    logits_tensor = _shard_logits_2d_mesh(torch_tensor, mesh_device)
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
    assert tt_log_probs is not None

    log_probs_tt_host = ttnn.to_torch(tt_log_probs, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    log_probs_tt_host = log_probs_tt_host[:, :, :1, :batch_size]

    log_probs_torch = F.log_softmax(torch_tensor.float(), dim=-1)
    log_probs_torch_argmax = torch.gather(log_probs_torch, dim=-1, index=argmax_tensor)
    log_probs_torch_argmax = log_probs_torch_argmax.reshape(1, 1, 1, batch_size)

    passing, pcc = comp_pcc(log_probs_torch_argmax, log_probs_tt_host, pcc=0.99)
    assert passing, f"logprobs PCC below threshold: {pcc}"


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
# New top-K logprobs tests (TG Galaxy only)
# ===========================================================================


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_top_k_log_probs_on_galaxy(shape, mesh_device):
    """Top-K logprobs PCC check on TG Galaxy (32-device 2D mesh)."""
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(1234)
    batch_size = shape[2]

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

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

    host_results = calc.transfer_logprobs_to_host(result, argmax_tensor.squeeze())

    composer = calc._build_mesh_composer()
    topk_logprobs_host = ttnn.to_torch(result.topk_logprobs, mesh_composer=composer)
    topk_logprobs_host = topk_logprobs_host[0, 0, ...]
    topk_indices_host = ttnn.to_torch(result.topk_indices, mesh_composer=composer)
    topk_indices_host = topk_indices_host[0, 0, ...].long()

    expected_logprobs = torch.gather(
        log_probs_torch.squeeze(0).squeeze(0),
        dim=-1,
        index=topk_indices_host,
    )

    passing, pcc = comp_pcc(expected_logprobs, topk_logprobs_host, pcc=0.99)
    print(f"Galaxy top-K logprobs PCC={pcc}")
    assert passing, f"Galaxy top-K logprobs PCC failed: {pcc}"

    for user_idx in range(batch_size):
        r = host_results[user_idx]
        assert r is not None
        sampled_id = argmax_tensor[0, 0, user_idx, 0].item()
        torch_lp = log_probs_torch[0, 0, user_idx, sampled_id].item()
        assert abs(r["returned_token"]["logprob"] - torch_lp) < 0.05


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_top_k_log_probs_returns_none_when_not_needed(shape, mesh_device):
    """calculate_topk_log_probs returns None when disabled."""
    _skip_if_not_galaxy(mesh_device)
    batch_size = shape[2]
    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    torch_tensor = torch.randn(shape)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    calc.set_log_probs_mode(False, num_logprobs=0)
    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is None, "Expected None when logprobs disabled"

    calc.set_log_probs_mode(True, num_logprobs=0)
    assert calc.topk_logprobs_needed  # needed for sampled token logprob
    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None, "Expected LogProbsResult when logprobs enabled"

    sampled_ids = argmax_tensor.squeeze()
    host_results = calc.transfer_logprobs_to_host(result, sampled_ids)
    assert len(host_results) == batch_size
    for i in range(batch_size):
        r = host_results[i]
        assert r is not None
        assert r["returned_token"]["token_idx"] == int(sampled_ids[i].item())
        assert len(r["top_logprobs"]["token_indices"]) == 0


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_per_user_logprobs_enabled(shape, mesh_device):
    """Mixed per-user logprobs: only even users enabled."""
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(42)
    batch_size = shape[2]

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    torch_tensor = torch.randn(shape)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    log_probs_torch = F.log_softmax(torch_tensor.to(torch.float16), dim=-1)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    enable_log_probs = [i % 2 == 0 for i in range(batch_size)]
    num_logprobs_list = [5 if i % 2 == 0 else 0 for i in range(batch_size)]
    calc.set_log_probs_mode(enable_log_probs, num_logprobs=num_logprobs_list)

    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None

    sampled_ids = argmax_tensor.squeeze()
    host_results = calc.transfer_logprobs_to_host(result, sampled_ids)

    for i in range(batch_size):
        if enable_log_probs[i]:
            assert host_results[i] is not None
            sampled_id = int(sampled_ids[i].item())
            torch_lp = log_probs_torch[0, 0, i, sampled_id].item()
            assert abs(host_results[i]["returned_token"]["logprob"] - torch_lp) < 0.05
        else:
            assert host_results[i] is None


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_set_log_probs_mode_validation(shape, mesh_device):
    """Verify set_log_probs_mode internal state."""
    _skip_if_not_galaxy(mesh_device)
    batch_size = shape[2]
    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    calc.set_log_probs_mode(True)
    assert calc.enable_log_probs is True
    assert all(calc.logprobs_enabled)
    assert calc.topk_logprobs_needed  # needed for sampled token logprob

    calc.set_log_probs_mode(True, num_logprobs=5)
    assert calc.topk_logprobs_needed is True
    assert all(n == 5 for n in calc.num_logprobs)

    enable_list = [True, False, True] + [False] * (batch_size - 3)
    num_lp_list = [10, 0, 3] + [0] * (batch_size - 3)
    calc.set_log_probs_mode(enable_list, num_logprobs=num_lp_list)
    assert calc.enable_log_probs is True
    assert calc.topk_logprobs_needed is True
    assert calc.logprobs_enabled == enable_list
    assert calc.num_logprobs == num_lp_list

    calc.set_log_probs_mode(False, num_logprobs=0)
    assert calc.enable_log_probs is False

    calc.set_log_probs_mode(True, num_logprobs=0)
    assert calc.enable_log_probs is True
    assert calc.topk_logprobs_needed  # needed for sampled token logprob

    calc.set_log_probs_mode(False, num_logprobs=0)
    calc.set_log_probs_mode([True, True], num_logprobs=[10, 15], empty_slots=[2, 5])
    assert calc.logprobs_enabled[2] is True
    assert calc.logprobs_enabled[5] is True
    assert calc.logprobs_enabled[0] is False
    assert calc.num_logprobs[2] == 10
    assert calc.num_logprobs[5] == 15

    calc.set_log_probs_mode(False, num_logprobs=0)
    calc.set_log_probs_mode(True, num_logprobs=7, empty_slots=[0, 3, 4])
    assert all(calc.logprobs_enabled[i] for i in [0, 3, 4])
    assert calc.logprobs_enabled[1] is False

    calc.set_log_probs_mode([True], num_logprobs=[20], empty_slots=[1])
    assert calc.logprobs_enabled[1] is True
    assert calc.num_logprobs[0] == 7
    assert calc.num_logprobs[1] == 20


@pytest.mark.parametrize("shape", [TG_SHAPE])
@pytest.mark.parametrize("device_params", [TG_DEVICE_PARAMS], indirect=True, ids=["tg"])
@pytest.mark.parametrize("mesh_device", [TG_MESH_SHAPE], indirect=True)
def test_top_k_logprobs_pcc_torch_vs_tt(shape, mesh_device):
    """Compare host (PyTorch bfloat16) vs device (bfloat16) logprobs for full batch."""
    _skip_if_not_galaxy(mesh_device)
    torch.manual_seed(9999)
    batch_size = shape[2]
    requested_logprobs = MAX_TOP_LOGPROBS

    calc = LogProbsCalculator(mesh_device, TG_SUB_CORE_GRIDS, batch_size=batch_size, use_topk_logprobs=True)

    torch_tensor = torch.randn(shape).to(torch.bfloat16)
    for i in range(batch_size):
        torch_tensor[:, :, i, :] = torch_tensor[:, :, i, torch.randperm(shape[-1])]

    log_probs_torch = F.log_softmax(torch_tensor, dim=-1, dtype=torch.bfloat16)
    gathered_values, gathered_indices = _simulate_gathered_topk(torch_tensor, TG_NUM_TP_DEVICES)
    argmax_tensor = torch.argmax(torch_tensor, dim=-1, keepdim=True)

    logits_tt, topk_values_tt, topk_indices_tt = _push_topk_test_tensors_to_tg(
        torch_tensor, gathered_values, gathered_indices, mesh_device
    )

    calc.set_log_probs_mode([True] * batch_size, num_logprobs=[requested_logprobs] * batch_size)

    result = calc.calculate_topk_log_probs(logits_tt, topk_values_tt, topk_indices_tt)
    assert result is not None

    sampled_ids = argmax_tensor.squeeze()
    host_results = calc.transfer_logprobs_to_host(result, sampled_ids)

    for user in range(batch_size):
        r = host_results[user]
        assert r is not None

        device_sampled_lp = r["returned_token"]["logprob"]
        token_idx = r["returned_token"]["token_idx"]
        torch_sampled_lp = log_probs_torch[0, 0, user, token_idx].item()
        assert abs(device_sampled_lp - torch_sampled_lp) < 0.05

        top_indices = r["top_logprobs"]["token_indices"]
        top_lps_device = torch.tensor(r["top_logprobs"]["logprobs"], dtype=torch.float32)
        assert len(top_indices) == requested_logprobs

        top_lps_torch = log_probs_torch[0, 0, user, top_indices].float()
        passing, pcc = comp_pcc(top_lps_torch.unsqueeze(0), top_lps_device.unsqueeze(0), pcc=0.98)
        assert passing, (
            f"User {user} top-{requested_logprobs} logprobs PCC failed: {pcc}\n"
            f"  device: {top_lps_device[:5].tolist()}...\n"
            f"  torch:  {top_lps_torch[:5].tolist()}..."
        )
