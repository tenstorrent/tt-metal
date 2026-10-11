# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exclusive serving lifecycle for complete hybrid prefix checkpoints.

This coordinator consumes scheduler-owned private pages and recurrent slots; it
never allocates vLLM blocks or advertises an attention-only APC hit. A runner
must hold ``execution()`` across model submissions, slot moves and these hooks.
Restore is synchronous and finishes before the suffix can use its destination.
The serving capability stays disabled until the paired runner is qualified.
"""

import threading
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace

from models.demos.qwen38_27b_qb2.tt.prefix_checkpoint import Checkpoint, Identity, capture, find_prefix, restore


class CancelledPrefixRequest(RuntimeError):
    pass


class PrefixDeviceFailure(RuntimeError):
    """Device completion is uncertain; cold fallback or slot reuse is unsafe."""


@dataclass(frozen=True)
class RequestHandle:
    request_id: str
    generation: int


@dataclass
class _Request:
    handle: RequestHandle
    slot: int
    tokens: tuple[int, ...]
    pages: tuple[int, ...]
    identity: Identity
    consumed: int = 0
    cancelled: bool = False


class PrefixServingCache:
    """Request lifecycle and leases around an existing opaque blob backend.

    The driver supplies ``prepare()``, ``fence()``, ``reset(slot)`` and
    ``transfer(checkpoint, slot, pages)``. prepare fences queued model work and
    publishes resident decode-bucket state into canonical slots. reset clears
    only the private recurrent slot; cold prefill overwrites private KV pages.

    Async cancellation marks the generation without acquiring the execution
    lock. Publication checks that mark after the final all-rank fence. The
    slot remains held until the runner releases it under execution ownership.
    """

    def __init__(self, store, identity, layout, driver, *, slots, num_pages):
        if type(slots) is not int or slots < 1 or type(num_pages) is not int or num_pages < 1:
            raise ValueError("Serving cache needs positive slot and physical-page capacities")
        self.store, self.identity, self.layout, self.driver = store, identity, layout, driver
        self.slots, self.num_pages = slots, num_pages
        self._execution = threading.RLock()
        self._metadata = threading.RLock()
        self._local = threading.local()
        self._generation = 0
        self._requests = {}
        self._poisoned = False
        self._quarantined_transfers = []
        self.stats = Counter()

    @contextmanager
    def execution(self):
        """Cover every model submission, not just cache copy calls."""
        with self._execution:
            if self._poisoned:
                raise PrefixDeviceFailure("A failed device transfer quarantined this cache")
            self._local.depth = getattr(self._local, "depth", 0) + 1
            try:
                yield
            finally:
                self._local.depth -= 1

    def _owned(self):
        if not getattr(self._local, "depth", 0):
            raise RuntimeError("Prefix lifecycle requires the shared model execution lease")
        if self._poisoned:
            raise PrefixDeviceFailure("A failed device transfer quarantined this cache")

    def _request(self, handle, *, allow_cancelled=False):
        request = self._requests.get(handle.request_id)
        if request is None or request.handle != handle:
            raise CancelledPrefixRequest("Prefix request generation no longer owns a slot")
        if request.cancelled and not allow_cancelled:
            raise CancelledPrefixRequest("Prefix request was cancelled before publication")
        return request

    def _allocation(self, slot, pages, *, excluding=None):
        if type(slot) is not int or not 0 <= slot < self.slots:
            raise ValueError("Prefix recurrent slot is outside the allocated cache")
        if (
            not pages
            or any(type(page) is not int or not 0 <= page < self.num_pages for page in pages)
            or len(set(pages)) != len(pages)
        ):
            raise ValueError("Prefix allocation requires distinct private physical pages")
        for request in self._requests.values():
            if request.handle == excluding:
                continue
            if request.slot == slot or set(request.pages).intersection(pages):
                raise ValueError("Prefix allocation aliases another admitted request")

    def admit(self, request_id, *, slot, tokens, pages, namespace, multimodal=False):
        """Claim an already allocated fresh destination before any prefill.

        Namespace must include the authenticated tenant/cache salt. Multimodal
        prefixes need content/processor identity and rotary state; reject them
        until those are represented rather than hashing placeholder tokens.
        """
        self._owned()
        if multimodal:
            raise ValueError("Multimodal prefix reuse requires media identity and MRoPE checkpoint state")
        if not isinstance(request_id, str) or not request_id or not isinstance(namespace, str) or not namespace:
            raise ValueError("Prefix admission requires request and tenant/cache-salt identities")
        tokens, pages = tuple(tokens), tuple(pages)
        if not tokens or any(type(token) is not int or not 0 <= token < 2**32 for token in tokens):
            raise ValueError("Prefix token history requires uint32 IDs")
        with self._metadata:
            if request_id in self._requests:
                raise ValueError("A live request ID cannot be admitted as a fresh generation")
            self._allocation(slot, pages)
            self._generation += 1
            handle = RequestHandle(request_id, self._generation)
            identity = replace(self.identity, namespace=namespace)
            self._requests[request_id] = _Request(handle, slot, tokens, pages, identity)
            return handle

    def update(self, handle, *, tokens, pages):
        """Grow history/pages without changing already-consumed token ownership."""
        self._owned()
        tokens, pages = tuple(tokens), tuple(pages)
        with self._metadata:
            request = self._request(handle)
            if (
                len(tokens) < request.consumed
                or tokens[: request.consumed] != request.tokens[: request.consumed]
                or any(type(token) is not int or not 0 <= token < 2**32 for token in tokens)
            ):
                raise ValueError("Consumed prefix history cannot change within a request generation")
            used = (request.consumed + self.layout.page_tokens - 1) // self.layout.page_tokens
            if pages[:used] != request.pages[:used]:
                raise ValueError("Consumed prefix pages cannot move without a new allocation lease")
            self._allocation(request.slot, pages, excluding=handle)
            request.tokens, request.pages = tokens, pages

    def advance(self, handle, consumed):
        """Record the ordered model input frontier, excluding sampled output."""
        self._owned()
        with self._metadata:
            request = self._request(handle)
            if (
                type(consumed) is not int
                or not request.consumed <= consumed <= len(request.tokens)
                or consumed > len(request.pages) * self.layout.page_tokens
            ):
                raise ValueError("Consumed frontier exceeds known tokens/private pages or moves backward")
            request.consumed = consumed

    def cancel(self, handle):
        """Mark cancellation immediately; the runner must still release its slot."""
        with self._metadata:
            request = self._request(handle, allow_cancelled=True)
            request.cancelled = True

    def release(self, handle):
        self._owned()
        with self._metadata:
            request = self._request(handle, allow_cancelled=True)
            del self._requests[request.handle.request_id]

    def remap_slots(self, old_to_new):
        """Commit the same complete permutation already submitted by the model."""
        self._owned()
        if set(old_to_new) != set(range(self.slots)) or set(old_to_new.values()) != set(range(self.slots)):
            raise ValueError("Prefix slot remap must be a complete permutation")
        with self._metadata:
            for request in self._requests.values():
                request.slot = old_to_new[request.slot]

    def frontier(self, handle):
        self._owned()
        with self._metadata:
            return self._request(handle).consumed

    def _device(self, operation, *args, transfer=None):
        try:
            return operation(*args)
        except BaseException as error:
            self._poisoned = True
            if transfer is not None:
                self._quarantined_transfers.append(transfer)
            raise PrefixDeviceFailure("Device prefix operation failed; destination is quarantined") from error

    def capture(self, handle):
        """Publish the exact consumed frontier, or return False on storage miss.

        Storage exhaustion is an ordinary non-admission. Device failures remain
        fatal: do not confuse a failed fence with a full disk and keep decoding.
        """
        self._owned()
        with self._metadata:
            request = self._request(handle)
            checkpoint = Checkpoint.for_tokens(request.identity, self.layout, request.tokens, request.consumed)
        try:
            capture(self.store, checkpoint, _ServingSource(self, handle))
        except FileExistsError:
            self.stats["already_present"] += 1
            return True
        except OSError:
            self.stats["capture_storage_misses"] += 1
            return False
        self.stats["captures"] += 1
        return True

    def restore_longest(self, handle, frontiers):
        """Restore a complete stored prefix into a fresh private allocation.

        Candidates come from scheduler admission, and must fit its current
        allocation. Leave at least one token to produce actual suffix logits.
        No scheduler computed-token count is changed by this method.
        """
        self._owned()
        with self._metadata:
            request = self._request(handle)
            if request.consumed:
                raise ValueError("Restore requires a fresh unobservable request allocation")
            candidates = [n for n in frontiers if n <= len(request.pages) * self.layout.page_tokens]
            checkpoint = find_prefix(self.store, request.identity, self.layout, request.tokens, candidates)
        if checkpoint is None:
            self.stats["misses"] += 1
            return 0
        try:
            restore(self.store, checkpoint, _ServingTarget(self, handle))
        except (OSError, ValueError):
            # _ServingTarget has already reset/fenced any partial destination.
            # A failed reset/fence raises PrefixDeviceFailure instead.
            self.stats["restore_storage_misses"] += 1
            return 0
        self.stats["hits"] += 1
        self.stats["restored_tokens"] += checkpoint.consumed
        return checkpoint.consumed


class _ServingSource:
    def __init__(self, owner, handle):
        self.owner, self.handle = owner, handle

    @contextmanager
    def freeze(self, checkpoint):
        owner = self.owner
        owner._device(owner.driver.prepare)
        with owner._metadata:
            request = owner._request(self.handle)
            if request.consumed != checkpoint.consumed:
                raise ValueError("Request advanced beyond the captured frontier")
            transfer = owner._device(
                owner.driver.transfer,
                checkpoint,
                request.slot,
                request.pages[: checkpoint.consumed // checkpoint.layout.page_tokens],
            )
        try:
            yield _ServingRead(owner, transfer)
            with owner._metadata:
                owner._request(self.handle)
        finally:
            owner._device(transfer.close, transfer=transfer)


class _ServingRead:
    def __init__(self, owner, transfer):
        self.owner, self.transfer = owner, transfer

    def read(self, segment, offset, size):
        return self.owner._device(self.transfer.read, segment, offset, size, transfer=self.transfer)


class _ServingTarget:
    def __init__(self, owner, handle):
        self.owner, self.handle = owner, handle

    @contextmanager
    def begin(self, checkpoint):
        owner = self.owner
        owner._device(owner.driver.prepare)
        with owner._metadata:
            request = owner._request(self.handle)
            slot = request.slot
            transfer = owner._device(
                owner.driver.transfer,
                checkpoint,
                slot,
                request.pages[: checkpoint.consumed // checkpoint.layout.page_tokens],
            )
        committed = False
        try:
            yield _ServingWrite(owner, self.handle, checkpoint, transfer)
            committed = True
        finally:
            owner._device(lambda: transfer.close(discard=not committed), transfer=transfer)
            if not committed:
                owner._device(owner.driver.reset, slot)
                owner._device(owner.driver.fence)


class _ServingWrite:
    def __init__(self, owner, handle, checkpoint, transfer):
        self.owner, self.handle, self.checkpoint, self.transfer = owner, handle, checkpoint, transfer

    def write(self, segment, offset, data):
        self.owner._device(self.transfer.write, segment, offset, data, transfer=self.transfer)

    def commit(self, consumed):
        if consumed != self.checkpoint.consumed:
            raise ValueError("Restore attempted to publish a different consumed frontier")
        self.owner._device(self.transfer.fence, transfer=self.transfer)
        with self.owner._metadata:
            request = self.owner._request(self.handle)
            request.consumed = consumed


class GeneratorPrefixDriver:
    """TT implementation for the canonical generator's stable cache addresses."""

    def __init__(self, generator, *, batched=False):
        self.generator, self.batched = generator, batched

    def prepare(self):
        self.fence()
        self.generator.model.suspend_decode_bucket()
        self.fence()

    def fence(self):
        import ttnn

        ttnn.synchronize_device(self.generator.mesh)

    def reset(self, slot):
        self.generator.reset_recurrent_slots([slot])

    def transfer(self, checkpoint, slot, pages):
        from models.demos.qwen38_27b_qb2.tt.prefix_transfer import PackedCacheTransfer

        return PackedCacheTransfer(
            self.generator.mesh,
            self.generator.cache,
            checkpoint,
            slot=slot,
            pages=pages,
            batched=self.batched,
        )
