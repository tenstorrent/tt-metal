# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device numerical, placement, provenance, and ownership tests for sampling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.checkpoint import CHECKPOINT_FILE_MANIFEST_SHA256, INDEX_SHA256
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import Qwen38BuildProvenance, Qwen38LiveBuildIdentity
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import Qwen38MeshContract, TensorPlacement
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import (
    PINNED_CHECKPOINT_REVISION,
    PINNED_TENSOR_MANIFEST_SHA256,
    VOCAB_SIZE,
    Qwen38ShardedLogits,
    Qwen38TTNNLMHead,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import (
    EXPECTED_VOCAB_RANGES,
    Qwen38SamplingCleanupError,
    Qwen38SamplingError,
    Qwen38SamplingParameters,
    Qwen38SamplingProfile,
    Qwen38TTNNHostSampler,
    _RuntimeOps,
    sample_host_logits,
)

PHYSICAL_IDS = (10, 11, 12, 13)


class PlacementReplicate:
    pass


class PlacementShard:
    def __init__(self, dim: int) -> None:
        self.dim = dim


class _Topology:
    def __init__(self, *, replicated: bool) -> None:
        self._placements = (
            PlacementReplicate(),
            PlacementReplicate() if replicated else PlacementShard(3),
        )

    def distribution_shape(self) -> tuple[int, int]:
        return (1, 4)

    def mesh_coords(self) -> tuple[tuple[int, int], ...]:
        return tuple((0, column) for column in range(4))

    def placements(self):
        return self._placements


class _FakeTensor:
    def __init__(
        self,
        shape: tuple[int, ...],
        *,
        replicated: bool,
        physical_ids: tuple[int, int, int, int] = PHYSICAL_IDS,
    ) -> None:
        self.shape = shape
        self.dtype = ttnn.bfloat16
        self.layout = ttnn.TILE_LAYOUT
        self._topology = _Topology(replicated=replicated)
        self._allocated = True
        self._physical_ids = physical_ids

    def device(self) -> object:
        return _FakeMesh(self._physical_ids)

    def tensor_topology(self) -> _Topology:
        return self._topology

    def is_allocated(self) -> bool:
        return self._allocated


class _FakeMesh:
    shape = (1, 4)

    def __init__(self, physical_ids: tuple[int, int, int, int] = PHYSICAL_IDS) -> None:
        self._physical_ids = physical_ids

    def get_num_devices(self) -> int:
        return 4

    def get_device_ids(self) -> list[int]:
        return list(self._physical_ids)


def test_mesh_contract_rejects_permuted_coordinate_order(expect_error) -> None:
    tensor = _FakeTensor((1, 1, 1, 640), replicated=False)
    tensor._topology.mesh_coords = lambda: ((0, 1), (0, 0), (0, 2), (0, 3))

    with expect_error(RuntimeError, "mesh-coordinate order"):
        Qwen38MeshContract(PHYSICAL_IDS).validate_tensor(
            tensor,
            placement=TensorPlacement.HIDDEN_SHARDED,
            shard_dim=3,
        )


@pytest.mark.parametrize("bad_ids", ([10, 11, 12, 13], (True, 11, 12, 13), (10, 11, 12, -1)))
def test_mesh_contract_rejects_noncanonical_physical_ids(bad_ids, expect_error) -> None:
    with expect_error(ValueError, "four distinct physical IDs"):
        Qwen38MeshContract(bad_ids)


class _FakeRuntime:
    def __init__(self, host_logits: torch.Tensor) -> None:
        self.host_logits = host_logits
        self.gather_calls = 0
        self.readback_calls = 0
        self.deallocate_calls = 0
        self.borrowed: _FakeTensor | None = None
        self.last_gathered: _FakeTensor | None = None
        self.alias_borrowed = False
        self.readback_error: BaseException | None = None
        self.deallocate_error_before_release: BaseException | None = None
        self.deallocate_error_after_release: BaseException | None = None

    def all_gather(self, tensor: _FakeTensor, **kwargs) -> _FakeTensor:
        assert kwargs == {
            "dim": 3,
            "cluster_axis": 1,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "topology": "Ring",
        }
        self.gather_calls += 1
        self.borrowed = tensor
        if self.alias_borrowed:
            return tensor
        rows = tensor.shape[2]
        self.last_gathered = _FakeTensor((1, 1, rows, VOCAB_SIZE), replicated=True)
        return self.last_gathered

    @staticmethod
    def make_concat_composer(mesh: _FakeMesh, dim: int):
        assert isinstance(mesh, _FakeMesh)
        assert dim == 0
        return (mesh, dim)

    def to_torch(self, tensor: _FakeTensor, *, mesh_composer) -> torch.Tensor:
        assert tensor is self.last_gathered
        assert mesh_composer[1] == 0
        self.readback_calls += 1
        if self.readback_error is not None:
            error, self.readback_error = self.readback_error, None
            raise error
        return self.host_logits.repeat(4, 1, 1, 1)

    def deallocate(self, tensor: _FakeTensor) -> None:
        assert tensor is self.last_gathered
        self.deallocate_calls += 1
        if self.deallocate_error_before_release is not None:
            error, self.deallocate_error_before_release = self.deallocate_error_before_release, None
            raise error
        tensor._allocated = False
        if self.deallocate_error_after_release is not None:
            error, self.deallocate_error_after_release = self.deallocate_error_after_release, None
            raise error

    def ops(self) -> _RuntimeOps:
        return _RuntimeOps(
            all_gather=self.all_gather,
            make_concat_composer=self.make_concat_composer,
            to_torch=self.to_torch,
            deallocate=self.deallocate,
        )


def _provenance(*, runtime_digit: str = "2") -> Qwen38BuildProvenance:
    return Qwen38BuildProvenance(
        checkpoint_revision=PINNED_CHECKPOINT_REVISION,
        checkpoint_index_sha256=INDEX_SHA256,
        checkpoint_config_sha256=CONFIG_SHA256,
        checkpoint_file_manifest_sha256=CHECKPOINT_FILE_MANIFEST_SHA256,
        checkpoint_hash_manifest_sha256=PINNED_TENSOR_MANIFEST_SHA256,
        tt_metal_sha="1" * 40,
        ttnn_runtime_sha256=runtime_digit * 64,
    )


def _identity(provenance: Qwen38BuildProvenance) -> Qwen38LiveBuildIdentity:
    return Qwen38LiveBuildIdentity(
        provenance=provenance,
        mesh_shape=(1, 4),
        physical_ids=PHYSICAL_IDS,
        collective_topology="Ring",
        dram_bank_ring_order=(6, 5, 4, 3, 2, 1, 0),
        ring_size=7,
    )


def _host_logits(*, rows: int = 1, fill: float = -100.0) -> torch.Tensor:
    return torch.full((1, 1, rows, VOCAB_SIZE), fill, dtype=torch.float32)


def _sampler(
    host_logits: torch.Tensor,
    *,
    clock_ns=None,
) -> tuple[Qwen38TTNNHostSampler, _FakeRuntime, Qwen38ShardedLogits, Qwen38LiveBuildIdentity]:
    provenance = _provenance()
    identity = _identity(provenance)
    mesh = _FakeMesh()
    head = object.__new__(Qwen38TTNNLMHead)
    head.mesh_device = mesh
    head.mesh_contract = Qwen38MeshContract(PHYSICAL_IDS)
    head.weights = SimpleNamespace(vocab_ranges=EXPECTED_VOCAB_RANGES)
    head.collective_topology = "Ring"
    runtime = _FakeRuntime(host_logits)
    sampler_kwargs = {} if clock_ns is None else {"clock_ns": clock_ns}
    sampler = Qwen38TTNNHostSampler(
        head,
        identity,
        expected_provenance=provenance,
        expected_identity_key=identity.key,
        eos_token_ids=(2, 248044),
        _runtime_ops=runtime.ops(),
        **sampler_kwargs,
    )
    rows = host_logits.shape[2]
    borrowed = _FakeTensor((1, 1, rows, VOCAB_SIZE // 4), replicated=False)
    logits = Qwen38ShardedLogits(
        tensor=borrowed,
        vocab_ranges=EXPECTED_VOCAB_RANGES,
        global_shape=(1, 1, rows, VOCAB_SIZE),
    )
    return sampler, runtime, logits, identity


def _custom(
    *,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    seed: int = 0,
) -> Qwen38SamplingParameters:
    return Qwen38SamplingParameters(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        seed=seed,
    )


def test_official_sampling_defaults_and_greedy_profile_are_exact() -> None:
    thinking = Qwen38SamplingParameters.official_thinking(seed=17)
    assert (thinking.temperature, thinking.top_p, thinking.top_k, thinking.presence_penalty, thinking.seed) == (
        1.0,
        0.95,
        20,
        0.0,
        17,
    )
    assert thinking.profile is Qwen38SamplingProfile.THINKING

    non_thinking = Qwen38SamplingParameters.official_non_thinking(seed=19)
    assert (
        non_thinking.temperature,
        non_thinking.top_p,
        non_thinking.top_k,
        non_thinking.presence_penalty,
        non_thinking.seed,
    ) == (0.7, 0.8, 20, 1.5, 19)
    assert non_thinking.profile is Qwen38SamplingProfile.NON_THINKING

    greedy = Qwen38SamplingParameters.greedy()
    assert (greedy.temperature, greedy.top_p, greedy.top_k, greedy.presence_penalty) == (0.0, 1.0, 0, 0.0)
    assert greedy.profile is Qwen38SamplingProfile.GREEDY


def test_greedy_is_exact_torch_argmax_including_lowest_global_tie() -> None:
    logits = _host_logits(rows=2)
    logits[0, 0, 0, 90] = 10
    logits[0, 0, 0, 91] = 10
    logits[0, 0, 1, VOCAB_SIZE - 1] = 20
    expected = torch.argmax(logits, dim=-1)
    actual = sample_host_logits(logits, Qwen38SamplingParameters.greedy(), token_histories=((), ()))
    assert actual.dtype == torch.int64
    assert actual.shape == (1, 1, 2)
    assert torch.equal(actual, expected)
    assert actual.tolist() == [[[90, VOCAB_SIZE - 1]]]


def test_top_k_and_shifted_top_p_filters_bound_the_support() -> None:
    logits = _host_logits()
    logits[0, 0, 0, 3] = 4.0
    logits[0, 0, 0, 4] = 3.0
    logits[0, 0, 0, 5] = 2.0

    for seed in range(20):
        token = sample_host_logits(logits, _custom(temperature=1.0, top_k=2, seed=seed)).item()
        assert token in (3, 4)

    # Token 3 alone crosses p=0.5; shifted nucleus masking must retain it.
    for seed in range(20):
        token = sample_host_logits(logits, _custom(temperature=1.0, top_p=0.5, seed=seed)).item()
        assert token == 3


def test_seeded_sampling_is_repeatable() -> None:
    logits = _host_logits()
    logits[0, 0, 0, :20] = torch.linspace(0.0, 1.0, 20)
    parameters = _custom(temperature=0.9, top_p=0.93, top_k=20, seed=123456)
    first = sample_host_logits(logits, parameters)
    second = sample_host_logits(logits.clone(), parameters)
    assert torch.equal(first, second)


def test_request_generator_advances_one_reproducible_seeded_stream() -> None:
    logits = _host_logits()
    logits[0, 0, 0, :20] = torch.linspace(0.0, 1.0, 20)
    parameters = _custom(temperature=0.9, top_p=0.93, top_k=20, seed=123456)
    first_generator = torch.Generator(device="cpu").manual_seed(parameters.seed)
    second_generator = torch.Generator(device="cpu").manual_seed(parameters.seed)
    initial_state = first_generator.get_state().clone()

    first_stream = [
        sample_host_logits(logits, parameters, generator=first_generator).item(),
        sample_host_logits(logits, parameters, generator=first_generator).item(),
    ]
    second_stream = [
        sample_host_logits(logits, parameters, generator=second_generator).item(),
        sample_host_logits(logits, parameters, generator=second_generator).item(),
    ]
    assert first_stream == second_stream
    assert not torch.equal(first_generator.get_state(), initial_state)
    assert torch.equal(first_generator.get_state(), second_generator.get_state())


def test_presence_penalty_subtracts_once_for_each_seen_token() -> None:
    logits = _host_logits()
    logits[0, 0, 0, 5] = 3.0
    logits[0, 0, 0, 6] = 2.0
    parameters = _custom(temperature=0.0, presence_penalty=1.5)
    once = sample_host_logits(logits, parameters, token_histories=(5,))
    repeated = sample_host_logits(logits, parameters, token_histories=(5, 5, 5))
    assert once.item() == 6
    assert torch.equal(once, repeated)


def test_sampler_gathers_four_vocab_shards_reports_eos_and_releases_owned_tensor() -> None:
    host = _host_logits()
    host[0, 0, 0, 2] = 10
    sampler, runtime, logits, identity = _sampler(host)

    result = sampler.sample(
        logits,
        Qwen38SamplingParameters.greedy(),
        source_identity_key=identity.key,
    )
    assert result.token_ids.tolist() == [[[2]]]
    assert result.eos_mask.tolist() == [[[True]]]
    assert result.any_eos
    assert result.source_identity_key == identity.key
    assert result.source_provenance_key == identity.provenance.key
    assert result.readback == "explicit-tp4-full-logit-host-gather"
    assert result.rng_mode == "per-call-seed"
    assert result.borrowed_logits_ownership == "caller-retained"
    assert result.owned_ttnn_tensors_after_return == 0
    assert result.timing.full_logit_gather_readback_ns >= 0
    assert result.timing.host_filter_sample_ns >= 0
    assert result.timing.end_to_end_ns >= 0
    assert result.timing.validation_and_packaging_ns >= 0
    assert runtime.gather_calls == runtime.readback_calls == runtime.deallocate_calls == 1
    assert runtime.borrowed is logits.tensor and runtime.borrowed.is_allocated()
    assert runtime.last_gathered is not None and not runtime.last_gathered.is_allocated()
    assert sampler.pending_owned_tensor_count == 0


@pytest.mark.parametrize(
    "bad_ranges",
    [
        ((0, VOCAB_SIZE),) * 4,
        ((0, VOCAB_SIZE // 4),) * 4,
        ((1, 1 + VOCAB_SIZE // 4),) + EXPECTED_VOCAB_RANGES[1:],
    ],
)
def test_replicated_or_noncontiguous_vocab_ranges_fail_before_gather(bad_ranges, expect_error) -> None:
    sampler, runtime, logits, identity = _sampler(_host_logits())
    invalid = Qwen38ShardedLogits(
        tensor=logits.tensor,
        vocab_ranges=bad_ranges,
        global_shape=logits.global_shape,
    )
    with expect_error(ValueError, "four distinct contiguous"):
        sampler.sample(invalid, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert runtime.gather_calls == 0
    assert logits.tensor.is_allocated()


def test_replicated_tensor_topology_is_not_accepted_as_vocab_sharding(expect_error) -> None:
    sampler, runtime, logits, identity = _sampler(_host_logits())
    replicated = _FakeTensor(tuple(logits.tensor.shape), replicated=True)
    invalid = Qwen38ShardedLogits(
        tensor=replicated,
        vocab_ranges=EXPECTED_VOCAB_RANGES,
        global_shape=logits.global_shape,
    )
    with expect_error(RuntimeError, "unexpected placements"):
        sampler.sample(invalid, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert runtime.gather_calls == 0
    assert replicated.is_allocated()


def test_logit_tensor_from_another_physical_mesh_fails_provenance_check(expect_error) -> None:
    sampler, runtime, logits, identity = _sampler(_host_logits())
    foreign = _FakeTensor(tuple(logits.tensor.shape), replicated=False, physical_ids=(20, 21, 22, 23))
    invalid = Qwen38ShardedLogits(
        tensor=foreign,
        vocab_ranges=EXPECTED_VOCAB_RANGES,
        global_shape=logits.global_shape,
    )
    with expect_error(RuntimeError, "physical order"):
        sampler.sample(invalid, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert runtime.gather_calls == 0
    assert foreign.is_allocated()


def test_declared_source_identity_and_constructor_provenance_fail_closed(expect_error) -> None:
    sampler, runtime, logits, identity = _sampler(_host_logits())
    with expect_error(ValueError, "source identity differs"):
        sampler.sample(logits, Qwen38SamplingParameters.greedy(), source_identity_key="f" * 64)
    assert runtime.gather_calls == 0

    with expect_error(ValueError, "independently supplied provenance differ"):
        Qwen38TTNNHostSampler(
            sampler.lm_head,
            identity,
            expected_provenance=_provenance(runtime_digit="3"),
            expected_identity_key=identity.key,
            eos_token_ids=(2,),
            _runtime_ops=runtime.ops(),
        )


def test_readback_fault_releases_only_owned_gather_and_owner_is_reusable(expect_error) -> None:
    host = _host_logits()
    host[0, 0, 0, 7] = 10
    sampler, runtime, logits, identity = _sampler(host)
    runtime.readback_error = RuntimeError("injected readback failure")

    with expect_error(RuntimeError, "injected readback failure"):
        sampler.sample(logits, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert runtime.deallocate_calls == 1
    assert sampler.pending_owned_tensor_count == 0
    assert logits.tensor.is_allocated()

    result = sampler.sample(logits, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert result.token_ids.item() == 7
    assert runtime.deallocate_calls == 2
    assert logits.tensor.is_allocated()


def test_post_sampling_failure_rolls_back_request_generator_and_keeps_no_owned_tensor(expect_error) -> None:
    class _FailingClock:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self) -> int:
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError("injected post-sampling clock failure")
            return self.calls * 10

    host = _host_logits()
    host[0, 0, 0, :20] = torch.linspace(0.0, 1.0, 20)
    sampler, runtime, logits, identity = _sampler(host, clock_ns=_FailingClock())
    parameters = _custom(temperature=1.0, top_k=20, seed=77)
    generator = torch.Generator(device="cpu").manual_seed(parameters.seed)
    original_state = generator.get_state().clone()

    with expect_error(RuntimeError, "post-sampling clock failure"):
        sampler.sample(
            logits,
            parameters,
            source_identity_key=identity.key,
            generator=generator,
        )
    assert torch.equal(generator.get_state(), original_state)
    assert sampler.pending_owned_tensor_count == 0
    assert runtime.deallocate_calls == 1
    assert logits.tensor.is_allocated()


def test_cleanup_failure_remains_pending_and_can_be_retried_without_double_free(expect_error) -> None:
    host = _host_logits()
    host[0, 0, 0, 8] = 10
    sampler, runtime, logits, identity = _sampler(host)
    runtime.deallocate_error_after_release = RuntimeError("injected post-release failure")

    with expect_error(Qwen38SamplingCleanupError, "retry release before reuse"):
        sampler.sample(logits, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert sampler.pending_owned_tensor_count == 1
    assert runtime.last_gathered is not None and not runtime.last_gathered.is_allocated()
    assert runtime.deallocate_calls == 1
    assert logits.tensor.is_allocated()

    sampler.release_owned_tensors()
    assert sampler.pending_owned_tensor_count == 0
    # is_allocated() proved the first call released it; retry did not double-free.
    assert runtime.deallocate_calls == 1
    sampler.release_owned_tensors()


def test_cleanup_failure_before_release_is_automatically_retried_before_next_gather(expect_error) -> None:
    host = _host_logits()
    host[0, 0, 0, 9] = 10
    sampler, runtime, logits, identity = _sampler(host)
    runtime.deallocate_error_before_release = RuntimeError("injected pre-release failure")

    with expect_error(Qwen38SamplingCleanupError):
        sampler.sample(logits, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert sampler.pending_owned_tensor_count == 1
    assert runtime.last_gathered is not None and runtime.last_gathered.is_allocated()

    result = sampler.sample(logits, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert result.token_ids.item() == 9
    assert runtime.gather_calls == 2
    assert runtime.deallocate_calls == 3  # failed old release, successful retry, new gather
    assert sampler.pending_owned_tensor_count == 0


def test_all_gather_alias_fault_never_deallocates_borrowed_logits(expect_error) -> None:
    sampler, runtime, logits, identity = _sampler(_host_logits())
    runtime.alias_borrowed = True
    with expect_error(RuntimeError, "aliases the borrowed"):
        sampler.sample(logits, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert runtime.deallocate_calls == 0
    assert sampler.pending_owned_tensor_count == 0
    assert logits.tensor.is_allocated()


def test_close_is_idempotent_and_disallows_future_sampling(expect_error) -> None:
    sampler, runtime, logits, identity = _sampler(_host_logits())
    sampler.close()
    sampler.close()
    with expect_error(Qwen38SamplingError, "closed"):
        sampler.sample(logits, Qwen38SamplingParameters.greedy(), source_identity_key=identity.key)
    assert runtime.gather_calls == 0
