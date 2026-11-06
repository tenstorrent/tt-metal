# SPDX-License-Identifier: Apache-2.0
"""Fixed BF16 backing for request-owned prefill histories (experimental)."""
import gemma_history_reservation as _history_reservation
import torch
import ttnn


class HistoryTail:
    """Temporary aligned K/V sources plus the original true tail coordinates.

    A descriptor is consumed at the model's bank commit, never by attention.
    Only its first length rows are logical history; filler is never consumed.
    """
    def __init__(self, sources, true_end, *, row=0):
        if len(sources) != 2:
            raise ValueError("Expected one aligned K/V source pair")
        shape = tuple(sources[0].shape)
        if (len(shape) != 4 or shape[1] != 4 or shape[3] != 256 or
                not 128 <= shape[2] <= 2048 or shape[2] % 128 or
                not 0 <= row < shape[0] or not 0 < true_end <= shape[2]):
            raise ValueError("Invalid aligned history-tail source or true span")
        for value in sources:
            if (tuple(value.shape) != shape or tuple(value.padded_shape) != shape or
                    value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT or
                    value.memory_config() != ttnn.DRAM_MEMORY_CONFIG):
                raise ValueError("Preserve exact interleaved BF16 native head sources")
        self.sources = tuple(sources)
        self.row = int(row)
        self.length = min(1152, int(true_end))
        self.begin = int(true_end) - self.length


class HistoryBank:
    """Setup-owned roots and indices; callers must use the default CQ0 only.

    Each refresh, gather and copy is enqueued on that same ordered queue.
    There is no multi-CQ sharing protocol for the mutable index buffers.
    """
    CAPACITY = 1152

    def __init__(self, model, capacity):
        if not 1 <= capacity <= 32:
            raise ValueError("History bank capacity must match bounded state slots")
        self.sliding = tuple(layer.sliding for layer in model.layers)
        self.roots = []
        # Allocate before any trace capture. No lazy persistent tensor owners.
        for _ in range(capacity):
            row = []
            for layer in model.layers:
                if not layer.sliding:
                    row.append(None)
                    continue
                if layer.kv_heads != 4 or layer.head_dim != 256 or layer.window != 1024:
                    raise ValueError("History bank requires original Gemma sliding geometry")
                row.append(tuple(ttnn.empty(
                    [1, 4, self.CAPACITY, 256], dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT, device=model.mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ) for _ in range(2)))
            self.roots.append(row)
        self.leases = {}
        self.generations = {}
        self.next_generation = 0
        self.free = list(reversed(range(capacity)))
        # Runtime contents change, but every retained index owner is setup-only.
        def index_buffer(count):
            return ttnn.from_torch(
                torch.zeros((1, count), dtype=torch.int32), dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT, device=model.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(model.mesh),
            )
        self.read_indices = {length: index_buffer(4 * length) for length in range(128, 1025, 128)}
        self.write_indices = index_buffer(4 * self.CAPACITY)
        self.normalization_counters = {"attention_reads": 0, "descriptor_writes": 0, "legacy_writes": 0}

    @staticmethod
    def _refresh(destination, values):
        # Default CQ0 write precedes its readers. Refresh again only after the
        # previous readers were enqueued on CQ0; no cross-queue use is allowed.
        source = ttnn.from_torch(values.contiguous(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        identity = destination.buffer_unique_id()
        ttnn.copy_host_to_device_tensor(source, destination)
        assert destination.buffer_unique_id() == identity

    def index_buffer_ids(self):
        return {**{str(length): value.buffer_unique_id() for length, value in self.read_indices.items()},
                "write": self.write_indices.buffer_unique_id()}

    @staticmethod
    def _lookup(source, index, rows):
        # Format the full physical head stride before flattening. No shortened
        # tiled-bank reshape is permitted, and no RM table/result is retained.
        rm = ttnn.to_layout(source, ttnn.ROW_MAJOR_LAYOUT)
        table = ttnn.reshape(rm, [source.shape[0] * 4 * source.shape[2], 256])
        assert table.buffer_unique_id() == rm.buffer_unique_id()
        selected = ttnn.embedding(index, table, layout=ttnn.TILE_LAYOUT,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.reshape(selected, [1, 4, rows, 256])
        assert output.buffer_unique_id() == selected.buffer_unique_id()
        return output

    def read_attention(self, roots, begin, length):
        if length not in self.read_indices or not 0 <= begin <= self.CAPACITY - length:
            raise ValueError("Expected a generated page-aligned attention history")
        index = self.read_indices[length]
        offsets = torch.arange(4, dtype=torch.int32)[:, None] * self.CAPACITY
        offsets = (offsets + begin + torch.arange(length, dtype=torch.int32)[None, :]).reshape(1, 4 * length)
        self._refresh(index, offsets)
        result = tuple(self._lookup(root, index, length) for root in roots)
        self.normalization_counters["attention_reads"] += 1
        return result

    def write_descriptor(self, roots, tail):
        height = int(tail.sources[0].shape[2])
        columns = torch.arange(self.CAPACITY, dtype=torch.int32)[None, :]
        selected = torch.where(columns < tail.length, columns + tail.begin, torch.zeros_like(columns))
        offsets = (torch.arange(4, dtype=torch.int32)[:, None] + tail.row * 4) * height
        indices = (offsets + selected).reshape(1, 4 * self.CAPACITY)
        assert int(indices.min()) >= 0 and int(indices.max()) < tail.sources[0].shape[0] * 4 * height
        self._refresh(self.write_indices, indices)
        for source, root in zip(tail.sources, roots):
            packed = self._lookup(source, self.write_indices, self.CAPACITY)
            ttnn.copy(packed, root)
            del packed
        self.normalization_counters["descriptor_writes"] += 1

    def acquire(self, key):
        if key not in self.leases:
            if not self.free:
                raise RuntimeError("History bank exhausted; cannot evict a live identity")
            self.leases[key] = self.free.pop()
            self.next_generation += 1
            self.generations[key] = self.next_generation
        reservation = _history_reservation.context()
        if reservation is not None and reservation.scheduler is not None:
            reservation.observe_bank_acquire(self)
        return self.leases[key]

    def release(self, key):
        index = self.leases.pop(key, None)
        if index is not None:
            self.generations.pop(key)
            self.free.append(index)

    def clear(self):
        self.leases.clear()
        self.generations.clear()
        self.free = list(reversed(range(len(self.roots))))

    def session(self, key, *, previous=None, rewind=0, restore=None, retain=True):
        index = self.acquire(key)
        if previous is not None:
            if previous.bank is not self or previous.key != key or previous.index != index:
                raise ValueError("History bank identity mismatch")
            previous._check(0)
            lengths = previous.lengths.copy()
        else:
            lengths = [0] * len(self.sliding)
        return BankHistory(self, key, index, lengths, rewind, restore, retain)


class BankHistory:
    """Layer lookup materializes a compact temporary; assignment commits a tail.

    No materialized read, restoration, pad or returned tail is cached here.
    """
    normalized_tail_publication = True

    def __init__(self, bank, key, index, lengths, rewind, restore, retain):
        self.bank, self.key, self.index = bank, key, index
        self.generation = bank.generations[key]
        self.lengths, self.rewind, self.restore = lengths, rewind, restore
        self.retain = retain
        self.committed = set()

    def __len__(self):
        return len(self.lengths)

    def _check(self, layer):
        if not 0 <= layer < len(self):
            raise IndexError(layer)
        if self.bank.leases.get(self.key) != self.index or self.bank.generations.get(self.key) != self.generation:
            raise RuntimeError("Attempt to consume a released history lease")

    def __getitem__(self, layer):
        self._check(layer)
        if not self.bank.sliding[layer]:
            return None
        if layer in self.committed:
            # Published state introspection reads the saved tail without rewind.
            valid, begin = self.lengths[layer], 0
        elif self.restore is not None:
            return self.restore(layer)
        else:
            valid = self.lengths[layer] - self.rewind
            begin = max(0, valid - 1024)
        if valid <= 0:
            return None
        if layer not in self.committed:
            return self.bank.read_attention(self.bank.roots[self.index][layer], begin, valid - begin)
        # Published-state diagnostic reads keep their original logical shape,
        # including arbitrary saved-tail lengths. They are not runtime attention.
        return tuple(root[:, :, begin:valid, :] for root in self.bank.roots[self.index][layer])

    def __setitem__(self, layer, tail):
        self._check(layer)
        if layer in self.committed:
            raise RuntimeError("History layer committed twice in one model call")
        if tail is None or not self.retain:
            self.lengths[layer] = 0
        elif isinstance(tail, HistoryTail):
            if not self.bank.sliding[layer]:
                raise ValueError("Full attention cannot publish a sliding tail")
            self.bank.write_descriptor(self.bank.roots[self.index][layer], tail)
            self.lengths[layer] = tail.length
        else:
            if not self.bank.sliding[layer] or len(tail) != 2:
                raise ValueError("Unexpected history tail")
            length = int(tail[0].shape[2])
            if not 0 < length <= self.bank.CAPACITY:
                raise ValueError("History exceeds original 1152-row bound")
            for value, root in zip(tail, self.bank.roots[self.index][layer]):
                if tuple(value.shape) != (1, 4, length, 256) or value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT:
                    raise ValueError("History bank must preserve BF16 tiled native heads")
                packed = value if length == self.bank.CAPACITY else ttnn.pad(
                    value, [(0, 0), (0, 0), (0, self.bank.CAPACITY-length), (0, 0)], value=0.0,
                )
                ttnn.copy(packed, root)
                del packed
            self.lengths[layer] = length
            self.bank.normalization_counters["legacy_writes"] += 1
        self.committed.add(layer)

    def publish(self):
        if len(self.committed) != len(self):
            raise RuntimeError("Cannot publish a partially committed history")
        # Drop the restoration closure and its captured per-call page tables.
        self.restore = None
        self.rewind = 0
        return self
