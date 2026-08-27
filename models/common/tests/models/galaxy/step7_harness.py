# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host harness for the Milestone B step-7 coverage suites.

Step 7 asks for paged KV, concat-32, prefix-cache, device-sampling and
long-context coverage. Most of what makes those *correct* is decided on the
host, before a single TTNN call:

- which blocks a slot owns, and therefore whether one slot can touch another's;
- which of the two page-table layouts a call stages, and how it is mapped;
- which tokens and which source rows a concatenated prefill plans;
- what values ``Sampling2D`` writes into its per-slot buffers.

This module makes those decisions observable without a mesh. It fakes exactly
the TTNN surface :mod:`models.common.models.galaxy.direct_runner` uses -
``from_torch`` and the two mesh mappers - and records the host tensor, the
mapper and the placement of everything the runner stages. The recorded torch
tensors are the *real* ones the runner computed, not stand-ins, so an assertion
here is an assertion about the value that would reach the device.

What this harness cannot tell you is anything about the mesh: the partition, the
collectives, L1 capacity, or whether an op accepts the layout it is handed. Those
need silicon. Every suite that uses this module says so in its own docstring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Sequence
from unittest.mock import MagicMock

import torch

import ttnn
from models.common.models.galaxy import direct_runner as direct_runner_module
from models.common.models.galaxy.kv_contract import GalaxyAttentionKVSpec, GalaxyPagedAttentionConfig

GALAXY_MESH_SHAPE = (8, 4)
GALAXY_PHYSICAL_BATCH = 32
GALAXY_USERS_PER_COLUMN = 8

#: Block size every Milestone B paged pool uses.
BLOCK_SIZE = 32

#: Llama-3.3-70B and Qwen3-32B vocabularies. Both appear in the sampling suite
#: because their padding behaviour differs and the difference is the point.
LLAMA_VOCAB_SIZE = 128256
QWEN_VOCAB_SIZE = 151936


def mock_mesh() -> Any:
    """Return a mesh stand-in that satisfies every Galaxy static geometry check."""

    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = GALAXY_MESH_SHAPE
    mesh.get_num_devices.return_value = 32
    mesh.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    return mesh


@dataclass
class StagedTensor:
    """One tensor the runner handed to ``ttnn.from_torch``."""

    host: torch.Tensor
    mapper: Any
    dtype: Any
    layout: Any
    memory_config: Any
    allocated: bool = True

    #: ``shape`` and ``dtype`` make this quack like a ttnn tensor for the
    #: module-side validators, which read both by duck typing.
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.host.shape)

    def is_allocated(self) -> bool:
        return self.allocated

    def deallocate(self, *_args: Any, **_kwargs: Any) -> None:
        self.allocated = False

    def __getitem__(self, key: Any) -> "StagedTensor":
        return StagedTensor(
            host=self.host[key],
            mapper=self.mapper,
            dtype=self.dtype,
            layout=self.layout,
            memory_config=self.memory_config,
        )


@dataclass
class ShardMapper:
    """Stand-in for ``ttnn.ShardTensor2dMesh``."""

    dims: tuple[Any, ...]
    mesh_shape: tuple[int, int]

    def device_local(self, host: torch.Tensor) -> torch.Tensor:
        """Return the shard one device owns, for the mesh dims this mapper sets.

        ``ttnn`` reports a distributed tensor's ``shape`` as the *shard* shape,
        not the global one: ``TensorToMesh::Impl::create_tensor`` builds the
        output ``Tensor`` from ``compute_tensor_spec_for_shards``. Every module
        validator that reads ``page_table.shape`` therefore sees this, which is
        why the harness models it explicitly.
        """

        result = host
        for axis, dim in enumerate(self.dims):
            if dim is None:
                continue
            shards = self.mesh_shape[axis]
            size = result.shape[dim]
            if size % shards:
                raise ValueError(f"cannot shard dim {dim} of size {size} over {shards}")
            result = result.narrow(dim, 0, size // shards)
        return result


@dataclass
class ReplicateMapper:
    """Stand-in for ``ttnn.ReplicateTensorToMesh``."""

    def device_local(self, host: torch.Tensor) -> torch.Tensor:
        return host


@dataclass
class StagingRecorder:
    """Every tensor the runner staged, in order."""

    staged: list[StagedTensor] = field(default_factory=list)

    def of_rank(self, rank: int) -> list[StagedTensor]:
        return [tensor for tensor in self.staged if tensor.host.dim() == rank]

    @property
    def page_tables(self) -> list[StagedTensor]:
        """Rank-2 int32 stagings, i.e. the page tables."""

        return [tensor for tensor in self.staged if tensor.host.dim() == 2 and tensor.host.dtype == torch.int32]


def patch_direct_runner(monkeypatch: Any) -> StagingRecorder:
    """Fake the TTNN surface the direct runner stages through.

    Returns the recorder holding every staged host tensor with the mapper and
    placement it was given.
    """

    recorder = StagingRecorder()

    def from_torch(tensor, *, device=None, mesh_mapper=None, dtype=None, layout=None, memory_config=None, **_kwargs):
        staged = StagedTensor(
            host=tensor.clone(),
            mapper=mesh_mapper,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
        )
        recorder.staged.append(staged)
        return staged

    monkeypatch.setattr(direct_runner_module.ttnn, "from_torch", from_torch)
    monkeypatch.setattr(
        direct_runner_module.ttnn,
        "ShardTensor2dMesh",
        lambda mesh_device, dims, mesh_shape: ShardMapper(dims=tuple(dims), mesh_shape=tuple(mesh_shape)),
    )
    monkeypatch.setattr(
        direct_runner_module.ttnn,
        "ReplicateTensorToMesh",
        lambda mesh_device: ReplicateMapper(),
    )
    return recorder


class RecordingModel:
    """A Galaxy graph stand-in that records what the runner asks it to run.

    It implements exactly the contract :class:`GalaxyDirectRunner` uses. Nothing
    here computes anything numerical; the point is the *plan* - which user ids,
    which sequence length, which page table, which token indices.
    """

    def __init__(
        self,
        *,
        paged: bool = True,
        n_layers: int = 2,
        max_num_blocks: int | None = None,
        vocab_size: int = LLAMA_VOCAB_SIZE,
        max_seq_len: int = 2048,
        prefill_sequence_lengths: Sequence[int] = (128, 512),
        batched_prefill_sequence_lengths: Sequence[int] = (128,),
        n_local_kv_heads: int = 1,
        head_dim: int = 128,
    ):
        self.geometry = SimpleNamespace(
            max_batch_size=GALAXY_PHYSICAL_BATCH,
            users_per_column=GALAXY_USERS_PER_COLUMN,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            prefill_sequence_lengths=tuple(prefill_sequence_lengths),
            batched_prefill_sequence_lengths=tuple(batched_prefill_sequence_lengths),
        )
        blocks = (max_seq_len // BLOCK_SIZE) * GALAXY_PHYSICAL_BATCH if max_num_blocks is None else max_num_blocks
        paged_config = GalaxyPagedAttentionConfig(block_size=BLOCK_SIZE, max_num_blocks=blocks) if paged else None
        spec = GalaxyAttentionKVSpec(
            n_local_kv_heads=n_local_kv_heads,
            head_dim=head_dim,
            kv_cache_dtype=ttnn.bfloat8_b,
            paged_attention_config=paged_config,
        )
        self.mesh_device = mock_mesh()
        self.kv_specs = (spec,) * n_layers
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.bound_cache: Any = None
        self.bind_calls: list[Any] = []
        self.modes: list[str] = []
        self.prefill_calls: list[dict[str, Any]] = []
        #: ``_stage_tokens`` returns a ``LazyWeight``, which never reaches
        #: ``ttnn.from_torch`` on the host, so the staged token rows are
        #: recorded here instead of in the staging recorder.
        self.prefill_token_rows: list[Any] = []
        self.decode_token_rows: list[Any] = []
        self.decode_calls: list[dict[str, Any]] = []
        self.projection_calls: list[dict[str, Any]] = []
        self.sample_calls: list[dict[str, Any]] = []
        #: Per-slot deterministic logits, so a test can assert which row the
        #: runner read back for which slot.
        self.logit_bias: dict[int, float] = {}

    # -- KV binding ---------------------------------------------------------

    def set_kv_cache(self, cache: Any) -> None:
        self.bind_calls.append(cache)
        self.bound_cache = cache

    # -- lifecycle ----------------------------------------------------------

    def activate(self, mode: str) -> Any:
        self.modes.append(mode)
        return None

    # -- staging ------------------------------------------------------------

    def embed_prefill(self, tokens: Any) -> Any:
        self.prefill_token_rows.append(tokens.source)
        return SimpleNamespace(name="prefill-embed", tokens=tokens)

    def embed_decode(self, tokens: Any) -> Any:
        self.decode_token_rows.append(tokens.source)
        return SimpleNamespace(name="decode-embed", tokens=tokens)

    def prepare_prefill_rot_mats(self, start_pos: int, seq_len: int) -> list[Any]:
        return [SimpleNamespace(name=f"rot-prefill-{start_pos}-{seq_len}")]

    def prepare_decode_rot_mats(self, positions: Any) -> list[Any]:
        return [SimpleNamespace(name="rot-decode", positions=positions)]

    # -- graph --------------------------------------------------------------

    def prefill_forward(self, x_embed, rot_mats, **kwargs) -> Any:
        self.prefill_calls.append(dict(kwargs))
        return SimpleNamespace(name="prefill-hidden")

    def project_prefill_logits(self, hidden, *, rows=1, sequence_length=None, token_indices=None) -> tuple[Any, ...]:
        self.projection_calls.append(
            {"rows": rows, "sequence_length": sequence_length, "token_indices": tuple(token_indices or ())}
        )
        return tuple(SimpleNamespace(name=f"logits-{index}") for index in range(rows))

    def decode_forward(self, x_embed, current_pos, rot_mats, page_table) -> Any:
        self.decode_calls.append({"positions": current_pos, "page_table": page_table})
        return SimpleNamespace(name="decode-logits")

    def sample_decode(self, logits, **kwargs) -> Any:
        self.sample_calls.append(dict(kwargs))
        return SimpleNamespace(name="sampled")


def patch_compose(monkeypatch: Any, rows_factory) -> None:
    """Fake the runner's logits composition so its readback is inspectable.

    Patches `compose_galaxy_logits`, not `to_torch_auto_compose`.
    `GalaxyDirectRunner._compose_rows` stopped using auto-composition because it
    composed the logits along the wrong mesh axis and then narrowed without
    raising (D-B23 in `tttv2_milestone_b_evidence/llama/REPORT.md`), and the
    replacement builds a real `ttnn.ConcatMesh2dToTensor`, which a `MagicMock` mesh
    cannot satisfy:

        TypeError: create_mesh_composer(): incompatible function arguments
        Invoked with types: unittest.mock.MagicMock, MeshComposerConfig

    The kwargs are swallowed so every existing caller's one-argument factory keeps
    working unchanged.
    """

    monkeypatch.setattr(
        direct_runner_module,
        "compose_galaxy_logits",
        lambda tensor, **_: rows_factory(tensor),
    )
    # The runner has *two* readback paths and this helper fakes both. The logits go
    # through `compose_galaxy_logits`; the device-sampled token ids still go through
    # `to_torch_auto_compose`, which is correct for them - `Sampling2D`'s output
    # placement is set by a mapper, not produced by a matmul, so its declared
    # topology is trustworthy. Patching only the first left the sampling tests
    # reaching a real `tensor_topology()` on a `SimpleNamespace`.
    monkeypatch.setattr(direct_runner_module, "to_torch_auto_compose", rows_factory)


__all__ = [
    "BLOCK_SIZE",
    "GALAXY_MESH_SHAPE",
    "GALAXY_PHYSICAL_BATCH",
    "GALAXY_USERS_PER_COLUMN",
    "LLAMA_VOCAB_SIZE",
    "QWEN_VOCAB_SIZE",
    "RecordingModel",
    "ReplicateMapper",
    "ShardMapper",
    "StagedTensor",
    "StagingRecorder",
    "mock_mesh",
    "patch_compose",
    "patch_direct_runner",
]
