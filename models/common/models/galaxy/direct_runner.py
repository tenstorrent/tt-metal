# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Direct prefill/decode driver for a Galaxy `(8, 4)` 2D tensor model.

Milestone B validates the reconstructed tensor models *before* the model-owned
executors exist, so its demos and hardware tests drive the graph directly. This
module owns the mechanical parts of that: paged KV allocation, the two page
table layouts, position and token staging, last-token extraction, sampling, and
deterministic teardown. It is model neutral — it uses only the Galaxy graph
contract both reconstructions expose — and it is *not* the Milestone C runtime.
It never imports a model-named package.

Two page table layouts exist because the two modes address the cache
differently:

- prefill fills a named user's blocks with ``paged_fill_cache(..., batch_idx=u)``
  and therefore needs every user's row on every device, so its table is a
  replicated ``[32, blocks_per_user]``;
- decode attends to one mesh column's users, and both ``paged_update_cache`` and
  the paged decode SDPA require the device-local table to carry exactly that
  batch, so its table is a ``[32, blocks_per_user]`` host tensor sharded over the
  four columns into ``[8, blocks_per_user]`` shards.

Blocks are statically owned: user ``u`` holds blocks
``[u * blocks_per_user, (u + 1) * blocks_per_user)``. That is the simplest
allocation that satisfies the module contract and keeps every slot isolated,
which is what the Milestone B cross-slot contamination gate asks for.

**Unqualified.** Nothing in this file has run on a Galaxy mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.models.galaxy.collectives import compose_galaxy_logits, deallocate_if_allocated
from models.common.models.galaxy.recipes import GALAXY_MESH_SHAPE
from models.common.modules.lazy_weight import LazyWeight


@dataclass(frozen=True)
class GalaxySamplingPolicy:
    """Per-request sampling values. ``temperature == 0`` means greedy."""

    top_k: int = 1
    top_p: float = 1.0
    temperature: float = 0.0
    seed: int | None = None
    on_device: bool = False

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0 or self.top_k == 1


@dataclass
class GalaxyDirectGeneration:
    """One slot's result from a direct generation run."""

    slot: int
    prompt_tokens: tuple[int, ...]
    generated_tokens: list[int] = field(default_factory=list)
    finished: bool = False
    prefill_logits: torch.Tensor | None = None


def _deallocate_all(tensors: Iterable[Any]) -> None:
    for tensor in tensors:
        deallocate_if_allocated(tensor)


class GalaxyDirectRunner:
    """Drive one Galaxy tensor model through prefill, decode, and sampling."""

    def __init__(
        self,
        model: Any,
        *,
        stop_token_ids: Sequence[int] = (),
        page_table_dtype: Any = ttnn.int32,
        active_slots: int | None = None,
        page_table_column_alignment: int = 8,
    ):
        geometry = model.geometry
        if tuple(model.mesh_device.shape) != GALAXY_MESH_SHAPE:
            raise ValueError(f"the direct runner requires a {GALAXY_MESH_SHAPE} mesh")
        self.model = model
        self.geometry = geometry
        self.mesh_device = model.mesh_device
        self.max_batch_size = geometry.max_batch_size
        self.users_per_column = geometry.users_per_column
        self.max_seq_len = geometry.max_seq_len
        self.vocab_size = geometry.vocab_size
        self.stop_token_ids = frozenset(int(value) for value in stop_token_ids)
        self.page_table_dtype = page_table_dtype
        # Chunked SDPA reads the page table as 32-byte sticks, so its row width
        # must be a multiple of eight int32 entries.
        self.page_table_column_alignment = page_table_column_alignment
        self.active_slots = self.max_batch_size if active_slots is None else int(active_slots)
        if not 1 <= self.active_slots <= self.max_batch_size:
            raise ValueError(f"active_slots must be in [1, {self.max_batch_size}], got {self.active_slots}")

        spec = model.kv_specs[0]
        self.kv_spec = spec
        self.paged = spec.paged_attention_config is not None
        if self.paged:
            paged = spec.paged_attention_config
            self.block_size = paged.block_size
            # Every inactive slot owns one sink block of its own. The decode
            # graph always runs the full physical batch, so inactive users still
            # write KV; giving each its own block keeps those writes off every
            # active slot's pages without two users racing on one address.
            self.sink_blocks = self.max_batch_size - self.active_slots
            usable = paged.max_num_blocks - self.sink_blocks
            self.blocks_per_user = usable // self.active_slots if usable > 0 else 0
            if self.blocks_per_user * self.block_size < self.max_seq_len:
                raise ValueError(
                    f"paged pool {paged.max_num_blocks} gives {self.blocks_per_user} blocks to each of "
                    f"{self.active_slots} slots, which cannot hold max_seq_len {self.max_seq_len}"
                )
        else:
            if self.active_slots != self.max_batch_size:
                raise ValueError("a contiguous KV cache serves every slot; active_slots applies to paged pools only")
            self.block_size = 0
            self.blocks_per_user = 0
            self.sink_blocks = 0

        self._kv_cache: list[list[Any]] = []
        self._prefill_page_table: Any = None
        self._decode_page_table: Any = None
        self._bound = False
        self._open = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "GalaxyDirectRunner":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def open(self) -> "GalaxyDirectRunner":
        """Allocate the KV cache and page tables and bind them to the model."""

        if self._open:
            return self
        try:
            self._kv_cache = self._allocate_kv_cache()
            self.model.set_kv_cache(self._kv_cache)
            # Recorded before the page tables so a staging failure still unbinds.
            self._bound = True
            if self.paged:
                self._prefill_page_table = self._stage_page_table(self.prefill_page_table_rows(), sharded=False)
                self._decode_page_table = self._stage_page_table(self.decode_page_table_rows(), sharded=True)
            self._open = True
            return self
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release everything the runner allocated. Idempotent and terminal."""

        failures: list[BaseException] = []

        def attempt(action: Callable[[], None]) -> None:
            try:
                action()
            except BaseException as error:  # noqa: BLE001 - collect, then raise the first
                failures.append(error)

        if self._bound:
            attempt(lambda: self.model.set_kv_cache(None))
        attempt(lambda: _deallocate_all((self._prefill_page_table, self._decode_page_table)))
        attempt(lambda: _deallocate_all(tensor for pair in self._kv_cache for tensor in pair))
        self._prefill_page_table = self._decode_page_table = None
        self._kv_cache = []
        self._bound = False
        self._open = False
        if failures:
            raise failures[0]

    def _require_open(self) -> None:
        if not self._open:
            raise RuntimeError("the direct runner is not open")

    # ------------------------------------------------------------------
    # Device state
    # ------------------------------------------------------------------

    def _allocate_kv_cache(self) -> list[list[Any]]:
        """Return one zeroed K/V pair per layer.

        A paged cache is replicated: every device owns the whole block pool and
        writes only the users its column serves, which is what makes one page
        table valid on every device. A contiguous cache instead holds one mesh
        column's users. In both cases the row shards start identical because the
        source is zero; each mesh row then fills its own KV head slice.
        """

        spec = self.kv_spec
        if self.paged:
            shape = spec.local_cache_shape()
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        else:
            shape = (self.users_per_column, spec.n_local_kv_heads, self.max_seq_len, spec.head_dim)
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        source = torch.zeros(shape, dtype=torch.bfloat16)
        cache: list[list[Any]] = []
        try:
            for _ in range(self.model.n_layers):
                cache.append(
                    [
                        ttnn.from_torch(
                            source,
                            device=self.mesh_device,
                            mesh_mapper=mapper,
                            dtype=spec.kv_cache_dtype,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        )
                        for _ in range(2)
                    ]
                )
            return cache
        except BaseException:
            _deallocate_all(tensor for pair in cache for tensor in pair)
            raise

    def _page_table_rows(self) -> torch.Tensor:
        """Return the static ``[32, blocks_per_user]`` block ownership table.

        Active slot ``u`` owns the contiguous run
        ``[u * blocks_per_user, (u + 1) * blocks_per_user)``; every inactive slot
        repeats its own sink block across the whole row so that any position it
        is asked to write lands there.
        """

        rows = torch.empty((self.max_batch_size, self.blocks_per_user), dtype=torch.int32)
        active_total = self.active_slots * self.blocks_per_user
        for slot in range(self.max_batch_size):
            if slot < self.active_slots:
                rows[slot] = torch.arange(
                    slot * self.blocks_per_user, (slot + 1) * self.blocks_per_user, dtype=torch.int32
                )
            else:
                rows[slot] = active_total + (slot - self.active_slots)
        return rows

    def prefill_page_table_rows(self) -> torch.Tensor:
        """Return the replicated prefill table, stick-aligned for chunked SDPA."""

        return self._pad_columns(self._page_table_rows())

    def decode_page_table_rows(self) -> torch.Tensor:
        """Return the column-sharded decode table, deliberately unpadded.

        The paged decode SDPA derives its KV length from the table's row width,
        so padding here would claim more cached context than each slot owns.
        Only the prefill table needs the 32-byte stick alignment.
        """

        return self._page_table_rows()

    def _pad_columns(self, rows: torch.Tensor) -> torch.Tensor:
        """Right-pad a page table to the chunked-SDPA stick alignment."""

        alignment = self.page_table_column_alignment
        width = rows.shape[1]
        padded_width = ((width + alignment - 1) // alignment) * alignment
        if padded_width == width:
            return rows
        padded = torch.zeros((rows.shape[0], padded_width), dtype=rows.dtype)
        padded[:, :width] = rows
        return padded

    def stage_chunk_page_table(self, *, chunk_start: int, length: int) -> Any:
        """Stage the replicated page table covering one prefill chunk.

        ``paged_fill_cache`` walks this table from entry zero, so it must start
        at the chunk's first block rather than at the sequence's.
        """

        self._require_open()
        if not self.paged:
            raise RuntimeError("chunked prefill requires a paged KV cache")
        if chunk_start % self.block_size:
            raise ValueError(f"chunk_start {chunk_start} must be a multiple of block size {self.block_size}")
        first = chunk_start // self.block_size
        blocks = -(-length // self.block_size)
        if first + blocks > self.blocks_per_user:
            raise ValueError(f"chunk [{chunk_start}, {chunk_start + length}) exceeds each slot's block allocation")
        rows = self._page_table_rows()[:, first : first + blocks]
        return self._stage_page_table(self._pad_columns(rows), sharded=False)

    def _stage_page_table(self, rows: torch.Tensor, *, sharded: bool) -> Any:
        mapper = (
            ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 0), mesh_shape=GALAXY_MESH_SHAPE)
            if sharded
            else ttnn.ReplicateTensorToMesh(self.mesh_device)
        )
        return ttnn.from_torch(
            rows,
            device=self.mesh_device,
            mesh_mapper=mapper,
            dtype=self.page_table_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _stage_tokens(self, tokens: Sequence[int]) -> LazyWeight:
        """Stage a replicated ``[1, n]`` token row for the embedding."""

        row = torch.tensor(list(tokens), dtype=torch.int32).reshape(1, -1)
        return LazyWeight(source=row, device=self.mesh_device)

    def _stage_positions(self, positions: Sequence[int]) -> Any:
        """Stage ``[32]`` decode positions as one column's eight per device."""

        if len(positions) != self.max_batch_size:
            raise ValueError(f"decode needs {self.max_batch_size} positions, got {len(positions)}")
        return ttnn.from_torch(
            torch.tensor(list(positions), dtype=torch.int32),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 0), mesh_shape=GALAXY_MESH_SHAPE),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------
    # Shapes and padding
    # ------------------------------------------------------------------

    def padded_prefill_length(self, token_count: int, *, batched: bool = False) -> int:
        """Return the smallest resolved recipe length covering ``token_count``."""

        lengths = self.geometry.batched_prefill_sequence_lengths if batched else self.geometry.prefill_sequence_lengths
        for length in lengths:
            if length >= token_count:
                return length
        kind = "batched" if batched else "single-row"
        raise ValueError(f"no {kind} prefill recipe covers {token_count} tokens; resolved lengths are {lengths}")

    def _compose_rows(self, tensor: Any, rows: int) -> torch.Tensor:
        """Compose device logits and return ``[rows, vocab_size]`` on host.

        The vocabulary is sharded over mesh rows and replicated over columns.
        This used to call `to_torch_auto_compose`, which reads the tensor's own
        topology - and a matmul output carries its *activation's* topology, so it
        concatenated the four columns along the vocabulary axis instead of the
        eight rows and returned four copies of row 0's vocabulary slice. Slicing
        `[:, :vocab_size]` then narrowed silently rather than raising, so every
        logit here was wrong with no symptom. See `compose_galaxy_logits`, which
        carries the measurement.
        """

        flat = compose_galaxy_logits(tensor, mesh_device=self.mesh_device, vocab_size=self.vocab_size)
        if flat.shape[0] < rows:
            raise ValueError(f"composed logits have {flat.shape[0]} rows, expected at least {rows}")
        if flat.shape[1] != self.vocab_size:
            raise ValueError(f"composed logits are {flat.shape[1]} wide, expected the {self.vocab_size} vocabulary")
        return flat[:rows, : self.vocab_size]

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill_row(
        self,
        tokens: Sequence[int],
        *,
        slot: int,
        sequence_length: int | None = None,
        chunk_start: int | None = None,
        chunk_page_table: Any = None,
        prefix_user_id: int | None = None,
    ) -> torch.Tensor:
        """Prefill one user and return its last-token logits ``[1, vocab]``."""

        self._require_open()
        if not 0 <= slot < self.active_slots:
            raise ValueError(f"slot must be in [0, {self.active_slots}), got {slot}")
        length = sequence_length or self.padded_prefill_length(len(tokens))
        if len(tokens) > length:
            raise ValueError(f"{len(tokens)} tokens do not fit the {length}-token recipe")
        padded = list(tokens) + [0] * (length - len(tokens))

        self.model.activate("prefill")
        rot_mats = self.model.prepare_prefill_rot_mats(chunk_start or 0, length)
        # The graph consumes the embedding; the release below is the failure path.
        hidden = logits = x_embed = None
        try:
            x_embed = self.model.embed_prefill(self._stage_tokens(padded))
            hidden = self.model.prefill_forward(
                x_embed,
                rot_mats,
                sequence_length=length,
                user_ids=(slot,),
                page_table=self._prefill_page_table,
                chunk_page_table=chunk_page_table,
                chunk_start=chunk_start,
                prefix_user_id=prefix_user_id,
                return_hidden_state=True,
            )
            (logits,) = self.model.project_prefill_logits(
                hidden, rows=1, sequence_length=length, token_indices=(len(tokens) - 1,)
            )
            hidden = None  # consumed by the projection
            return self._compose_rows(logits, 1)
        finally:
            _deallocate_all((logits, hidden, x_embed, *rot_mats))

    def prefill_chunked(self, tokens: Sequence[int], *, slot: int = 0, chunk_length: int) -> torch.Tensor:
        """Prefill one user in chunks and return its last-token logits.

        The first chunk is an ordinary prefill; every later chunk is a
        prefix-cached request whose SDPA reads the blocks the earlier chunks
        wrote. This is the only way to prefill a context longer than one
        resolved recipe, which is what the long-context smokes need, and it is
        also the exact path prefix caching uses when a request resumes.
        """

        self._require_open()
        if chunk_length <= 0 or len(tokens) == 0:
            raise ValueError("chunked prefill needs a positive chunk length and at least one token")
        chunks = [tokens[start : start + chunk_length] for start in range(0, len(tokens), chunk_length)]
        logits: torch.Tensor | None = None
        for index, chunk in enumerate(chunks):
            chunk_start = index * chunk_length
            if index == 0:
                logits = self.prefill_row(chunk, slot=slot, sequence_length=chunk_length)
                continue
            chunk_table = None
            try:
                chunk_table = self.stage_chunk_page_table(chunk_start=chunk_start, length=chunk_length)
                logits = self.prefill_row(
                    chunk,
                    slot=slot,
                    sequence_length=chunk_length,
                    chunk_start=chunk_start,
                    chunk_page_table=chunk_table,
                    prefix_user_id=slot,
                )
            finally:
                deallocate_if_allocated(chunk_table)
        if logits is None:
            raise RuntimeError("chunked prefill produced no logits")
        return logits

    def prefill_batched(self, token_rows: Sequence[Sequence[int]]) -> torch.Tensor:
        """Prefill all 32 slots as one concatenated request.

        Every row is padded to one common recipe length, which is the Galaxy
        batched-prefill policy: physical batch 32, no cached prefixes, and a
        common padded sequence length.
        """

        self._require_open()
        if len(token_rows) != self.max_batch_size or self.active_slots != self.max_batch_size:
            raise ValueError(f"concatenated prefill needs exactly {self.max_batch_size} active rows")
        length = self.padded_prefill_length(max(len(row) for row in token_rows), batched=True)
        flat: list[int] = []
        for row in token_rows:
            if len(row) > length:
                raise ValueError(f"{len(row)} tokens do not fit the {length}-token batched recipe")
            flat.extend(list(row) + [0] * (length - len(row)))

        self.model.activate("prefill")
        rot_mats = self.model.prepare_prefill_rot_mats(0, length)
        hidden = x_embed = None
        outputs: tuple[Any, ...] = ()
        try:
            x_embed = self.model.embed_prefill(self._stage_tokens(flat))
            hidden = self.model.prefill_forward(
                x_embed,
                rot_mats,
                sequence_length=length,
                user_ids=tuple(range(self.max_batch_size)),
                page_table=self._prefill_page_table,
                return_hidden_state=True,
            )
            outputs = self.model.project_prefill_logits(
                hidden,
                rows=self.max_batch_size,
                sequence_length=length,
                token_indices=tuple(len(row) - 1 for row in token_rows),
            )
            hidden = None
            return torch.cat([self._compose_rows(output, 1) for output in outputs], dim=0)
        finally:
            _deallocate_all((*outputs, hidden, x_embed, *rot_mats))

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode_logits(self, tokens: Sequence[int], positions: Sequence[int]) -> torch.Tensor:
        """Run one decode step and return host logits ``[32, vocab]``."""

        self._require_open()
        device_logits = None
        try:
            device_logits = self._decode_device_logits(tokens, positions)
            return self._compose_rows(device_logits, self.max_batch_size)
        finally:
            deallocate_if_allocated(device_logits)

    def decode_sampled(
        self, tokens: Sequence[int], positions: Sequence[int], policy: GalaxySamplingPolicy
    ) -> torch.Tensor:
        """Run one decode step and sample on device, returning ``[32]`` tokens."""

        self._require_open()
        device_logits = sampled = None
        try:
            device_logits = self._decode_device_logits(tokens, positions)
            sampled = self.model.sample_decode(
                device_logits,
                top_k=policy.top_k,
                top_p=policy.top_p,
                temperature=policy.temperature,
                seed=policy.seed,
                forced_argmax=policy.greedy,
            )
            composed = to_torch_auto_compose(sampled).reshape(-1)[: self.max_batch_size]
            return composed.to(torch.int64)
        finally:
            _deallocate_all((sampled, device_logits))

    def _decode_device_logits(self, tokens: Sequence[int], positions: Sequence[int]) -> Any:
        if len(tokens) != self.max_batch_size:
            raise ValueError(f"decode needs {self.max_batch_size} tokens, got {len(tokens)}")
        self.model.activate("decode")
        position_tensor = x_embed = None
        rot_mats: list[Any] = []
        try:
            position_tensor = self._stage_positions(positions)
            rot_mats = self.model.prepare_decode_rot_mats(torch.tensor(list(positions), dtype=torch.int64))
            x_embed = self.model.embed_decode(self._stage_tokens(tokens))
            return self.model.decode_forward(x_embed, position_tensor, rot_mats, self._decode_page_table)
        finally:
            # The graph consumes the embedding; this is the failure path.
            _deallocate_all((position_tensor, x_embed, *rot_mats))

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_host(self, logits: torch.Tensor, policy: GalaxySamplingPolicy) -> list[int]:
        """Sample one token per logits row on host."""

        if policy.greedy:
            return [int(value) for value in torch.argmax(logits, dim=-1)]
        generator = torch.Generator(device="cpu")
        if policy.seed is not None:
            generator.manual_seed(int(policy.seed))
        tokens: list[int] = []
        for row in logits:
            k = min(policy.top_k, row.shape[-1])
            values, indices = torch.topk(row / policy.temperature, k=k)
            probabilities = torch.softmax(values, dim=-1)
            if policy.top_p < 1.0:
                cumulative = probabilities.cumsum(dim=-1)
                probabilities = probabilities.masked_fill(cumulative - probabilities > policy.top_p, 0.0)
                probabilities = probabilities / probabilities.sum()
            choice = torch.multinomial(probabilities, 1, generator=generator)
            tokens.append(int(indices[choice].item()))
        return tokens

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: Sequence[Sequence[int]],
        *,
        max_new_tokens: int,
        policy: GalaxySamplingPolicy = GalaxySamplingPolicy(),
        batched_prefill: bool = False,
        on_token: Callable[[int, int], None] | None = None,
    ) -> list[GalaxyDirectGeneration]:
        """Prefill every prompt, then decode the physical batch in lockstep.

        Slots are the prompt order. Inactive slots hold position 0 and token 0;
        they occupy their own KV blocks, so they can neither read nor write an
        active slot's cache.
        """

        self._require_open()
        if not prompts or len(prompts) > self.active_slots:
            raise ValueError(f"between 1 and {self.active_slots} prompts are required")
        if any(len(prompt) == 0 for prompt in prompts):
            raise ValueError("every prompt needs at least one token")
        results = [
            GalaxyDirectGeneration(slot=slot, prompt_tokens=tuple(prompt)) for slot, prompt in enumerate(prompts)
        ]

        if batched_prefill:
            if len(prompts) != self.max_batch_size:
                raise ValueError("concatenated prefill requires all 32 slots")
            prefill_logits = self.prefill_batched(prompts)
        else:
            prefill_logits = torch.cat(
                [self.prefill_row(prompt, slot=slot) for slot, prompt in enumerate(prompts)], dim=0
            )
        first_tokens = self.sample_host(prefill_logits, policy)
        for result, logits_row, token in zip(results, prefill_logits, first_tokens):
            result.prefill_logits = logits_row
            result.generated_tokens.append(token)
            result.finished = token in self.stop_token_ids
            if on_token is not None:
                on_token(result.slot, token)

        tokens = [0] * self.max_batch_size
        positions = [0] * self.max_batch_size
        for result in results:
            tokens[result.slot] = result.generated_tokens[-1]
            positions[result.slot] = len(result.prompt_tokens)

        for _ in range(max_new_tokens - 1):
            if all(result.finished for result in results):
                break
            if max(positions) >= self.max_seq_len:
                break
            if policy.on_device:
                sampled = [int(value) for value in self.decode_sampled(tokens, positions, policy)]
            else:
                sampled = self.sample_host(self.decode_logits(tokens, positions), policy)
            for result in results:
                slot = result.slot
                positions[slot] += 1
                if result.finished:
                    continue
                token = sampled[slot]
                result.generated_tokens.append(token)
                tokens[slot] = token
                result.finished = token in self.stop_token_ids
                if on_token is not None:
                    on_token(slot, token)
        return results

    # ------------------------------------------------------------------
    # Teacher forcing
    # ------------------------------------------------------------------

    def teacher_forced_decode(
        self, prompt_tokens: Sequence[int], reference_tokens: Sequence[int], *, slot: int = 0
    ) -> torch.Tensor:
        """Return ``[len(reference_tokens), vocab]`` logits for a forced sequence.

        The prompt is prefilled once; every reference token is then fed back as
        the next decode input regardless of what the model predicted, which is
        what the Milestone B top-1/top-5 accuracy gate measures.
        """

        self._require_open()
        prefill_logits = self.prefill_row(prompt_tokens, slot=slot)
        rows = [prefill_logits[0]]
        tokens = [0] * self.max_batch_size
        positions = [0] * self.max_batch_size
        position = len(prompt_tokens)
        for index, forced in enumerate(reference_tokens[:-1]):
            tokens[slot] = int(forced)
            positions[slot] = position + index
            rows.append(self.decode_logits(tokens, positions)[slot])
        return torch.stack(rows, dim=0)


__all__ = [
    "GalaxyDirectGeneration",
    "GalaxyDirectRunner",
    "GalaxySamplingPolicy",
]
