# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packing geometry independent of request slots and absolute prefix positions.

Tokenwise work uses a balanced CP partition of the concatenated requests. Ring
attention always uses the cache's original chunk geometry. CP all-gather and
partition implement the two inverse redistributions, including empty cache ranks.
"""

from dataclasses import dataclass


def round_up(value, alignment):
    return -(-value // alignment) * alignment


def map_packed_rows(tensor, max_rows, operation):
    """Bound tokenwise kernel slabs without respecting request boundaries.

    Larger packed batches must not silently select a lower-precision matmul
    fallback or exceed normalization's L1 budget. Inputs remain caller-owned;
    the operation returns a fresh tensor with the same number of rows.
    """
    if max_rows is None or tensor.shape[2] <= max_rows:
        return operation(tensor)
    import ttnn

    outputs = []
    for start in range(0, tensor.shape[2], max_rows):
        rows = RaggedAttentionLayout.slice_rows(tensor, start, min(max_rows, tensor.shape[2] - start))
        outputs.append(operation(rows))
        rows.deallocate(True)
    result = ttnn.concat(outputs, dim=2)
    for output in outputs:
        output.deallocate(True)
    return result


@dataclass(frozen=True)
class PrefillRequest:
    request_id: int
    slot_id: int
    actual_start: int
    token_ids: tuple[int, ...]
    # Migration's request_id is a chunk sequence number. Keep it distinct from
    # the stable serving request identity used to own slots and outputs.
    completion_id: int | None = None

    @property
    def actual_end(self):
        return self.actual_start + len(self.token_ids)


def iter_request_batches(prompts, *, num_slots, chunk_size=8192):
    """Schedule ``(request_id, tokens)`` prompts at completed batch boundaries.

    One chunk per live slot, in slot order. Exhausted prompts are replaced on
    the next boundary. The consumer must finish a yielded batch before asking
    for another one; no lookahead reuses a live cache slot.
    """
    if num_slots <= 0 or chunk_size <= 0:
        raise ValueError("Slot count and chunk size must be positive")
    source = iter(prompts)
    active = {}
    seen = set()
    exhausted = False
    while True:
        for slot in range(num_slots):
            if slot not in active and not exhausted:
                try:
                    identity, tokens = next(source)
                except StopIteration:
                    exhausted = True
                    break
                tokens = tuple(tokens)
                if identity in seen or not tokens:
                    raise ValueError("Prompts require unique request IDs and at least one token")
                seen.add(identity)
                active[slot] = (identity, tokens, 0)
        if not active:
            return
        yield tuple(
            PrefillRequest(identity, slot, start, tokens[start : start + chunk_size])
            for slot, (identity, tokens, start) in sorted(active.items())
        )
        for slot, (identity, tokens, start) in list(active.items()):
            end = start + chunk_size
            if end >= len(tokens):
                del active[slot]
            else:
                active[slot] = (identity, tokens, end)


@dataclass(frozen=True)
class RaggedPrefillPlan:
    """One trace shape: ordered tile-rounded segment sizes plus a CP/TP bucket.

    Slots, request identities, prefixes and exact lengths are runtime values.
    Occupancy changes select another plan; there are no inactive device lanes.
    """

    segment_sizes: tuple[int, ...]
    chunk_size: int = 8192
    cp: int = 8
    tp: int = 4

    def __post_init__(self):
        if (self.cp, self.tp) != (8, 4):
            raise ValueError("Ragged prefill currently requires CP8/TP4")
        if self.chunk_size <= 0 or self.chunk_size % (self.cp * self.tp * 32):
            raise ValueError("Cache chunks must contain whole CP/TP-local tiles")
        if not self.segment_sizes or any(n <= 0 or n > self.chunk_size or n % 32 for n in self.segment_sizes):
            raise ValueError("Active segments must contain 1 to chunk_size tokens, rounded to tiles")

    @classmethod
    def for_requests(cls, requests, **kwargs):
        return cls(tuple(round_up(len(request.token_ids), 32) for request in requests), **kwargs)

    @property
    def packed_size(self):
        return round_up(sum(self.segment_sizes), self.cp * self.tp * 32)

    @property
    def offsets(self):
        offset = 0
        result = []
        for size in self.segment_sizes:
            result.append(offset)
            offset += size
        return tuple(result)

    def pack(self, requests, *, positions=False):
        if len(requests) != len(self.segment_sizes):
            raise ValueError("Request occupancy does not match the trace")
        values = [0] * self.packed_size
        for request, offset, size in zip(requests, self.offsets, self.segment_sizes):
            length = len(request.token_ids)
            if round_up(length, 32) != size:
                raise ValueError("Request length does not match the trace segment")
            values[offset : offset + length] = (
                range(request.actual_start, request.actual_end) if positions else request.token_ids
            )
        return values

    def unpack(self, packed, requests):
        """Return request-owned valid rows from a host tensor with sequence at -2."""
        return {
            request.request_id: packed[..., offset : offset + len(request.token_ids), :].clone()
            for request, offset in zip(requests, self.offsets)
        }


class RaggedAttentionLayout:
    """Traceable split/concat with an explicit inverse CP mapping.

    All intermediates are caller-owned. The ring receive buffers/semaphores stay
    owned by CCLManager and are reused in command-queue order, just as between
    consecutive layers on the single-request path.
    """

    def __init__(self, plan, mesh_config, metadata):
        self.plan = plan
        self.mesh_config = mesh_config
        self.metadata = metadata

    def gather(self, tensor):
        import ttnn

        # TP heads stay local; only CP token rows move here.
        return ttnn.all_gather(
            tensor, dim=2, cluster_axis=self.mesh_config.cp_axis, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    @staticmethod
    def slice_rows(tensor, start, size):
        import ttnn

        # TTNN returns an alias for a full-extent slice. Routing intermediates
        # have independent ownership because their source is freed immediately.
        if start == 0 and size == tensor.shape[2]:
            return ttnn.clone(tensor)
        return ttnn.slice(tensor, (0, 0, start, 0), (*tuple(tensor.shape)[:2], start + size, tensor.shape[-1]))

    @staticmethod
    def pad_rows(tensor, size):
        import ttnn

        if tensor.shape[2] == size:
            return tensor
        result = ttnn.pad(tensor, [(0, 0), (0, 0), (0, size - tensor.shape[2]), (0, 0)], 0.0)
        tensor.deallocate(True)
        return result

    def split(self, gathered, lane):
        import ttnn

        segment = self.slice_rows(gathered, self.plan.offsets[lane], self.plan.segment_sizes[lane])
        padded = self.pad_rows(segment, self.plan.chunk_size)
        local = ttnn.mesh_partition(padded, dim=2, cluster_axis=self.mesh_config.cp_axis)
        padded.deallocate(True)
        return local

    def compact(self, local, lane):
        gathered = self.gather(local)
        local.deallocate(True)
        segment = self.slice_rows(gathered, 0, self.plan.segment_sizes[lane])
        gathered.deallocate(True)
        return segment

    def concatenate(self, segments):
        import ttnn

        packed = ttnn.concat(segments, dim=2) if len(segments) > 1 else segments[0]
        if len(segments) > 1:
            for tensor in segments:
                tensor.deallocate(True)
        padded = self.pad_rows(packed, self.plan.packed_size)
        local = ttnn.mesh_partition(padded, dim=2, cluster_axis=self.mesh_config.cp_axis)
        padded.deallocate(True)
        return local
