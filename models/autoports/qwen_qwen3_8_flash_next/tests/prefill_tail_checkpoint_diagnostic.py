# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic-only pre-singleton checkpoint; no production policy changes.

A committed odd prefill remains usable by decode. Only an immediate same-owner
prefill continuation rewinds one token, then ordinary dyadic chunks recompute it
with new input. Missing/invalid checkpoints preserve the original negative path.
No KV or compressed-index tensor is copied. No device runs at import.
"""
import hashlib
from pathlib import Path

import torch


def compact_state(cache):
    result = []
    for index, layer in enumerate(cache.layers):
        for field in ("conv", "recurrent", "ple_conv", "token_history"):
            tensor = getattr(layer, field, None)
            if tensor is not None:
                result.append((f"layer{index}.{field}", tensor))
        if layer.attention is not None:
            for field in ("raw_history", "position"):
                result.append((f"layer{index}.attention.{field}", getattr(layer.attention, field)))
    return result


class TailCheckpointDiagnostic:
    def __init__(self, gen, memory):
        import ttnn

        self.tt, self.gen = ttnn, gen
        self.valid, self.active = {}, None
        self.original = {
            name: getattr(gen, name) for name in ("prefill_forward", "_prefill_chunk", "_reset_cache", "decode_forward")
        }
        self.report = dict(
            scope="diagnostic only; not production or stage acceptance",
            helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            excluded=["K cache", "V cache", "compressed index keys", "page tables", "sampler state"],
            saved=[],
            restored=[],
            missing_or_invalid=[],
            invalidated=[],
            tensor_inventory=[],
        )
        before = memory(gen.mesh)
        self.buffers = []
        for slot, cache in enumerate(gen.caches):
            row = []
            for name, tensor in compact_state(cache):
                backup = ttnn.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    device=gen.mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                row.append((name, backup))
                self.report["tensor_inventory"].append(
                    dict(
                        slot=slot,
                        name=name,
                        shape=list(tensor.shape),
                        padded_shape=list(backup.padded_shape),
                        dtype=str(tensor.dtype),
                    )
                )
            self.buffers.append(row)
        after = memory(gen.mesh)
        self.report["allocation_memory_before"] = before
        self.report["allocation_memory_after"] = after
        self.report["persistent_allocation_bytes_per_rank"] = after["num_banks"] * (
            after["total_bytes_allocated_per_bank"] - before["total_bytes_allocated_per_bank"]
        )
        # Compile both directions before the harness records its prepared count.
        for cache, row in zip(gen.caches, self.buffers, strict=True):
            self.copy(compact_state(cache), row)
            self.copy(row, compact_state(cache))
        gen.prefill_forward = self.prefill
        gen._prefill_chunk = self.chunk
        gen._reset_cache = self.reset_cache
        gen.decode_forward = self.decode

    def copy(self, source, destination):
        for (left, src), (right, dst) in zip(source, destination, strict=True):
            assert left == right
            self.tt.copy(src, dst)

    def reset_cache(self, cache):
        for slot, owned in enumerate(self.gen.caches):
            if owned is cache:
                self.valid.pop(slot, None)
        return self.original["_reset_cache"](cache)

    def decode(self, *args, **kwargs):
        # Diagnostic harness has no slot migration; invalidate all rather than
        # pretend an old prefill checkpoint describes a decoded request.
        self.report["invalidated"].append(dict(reason="decode", slots=list(self.valid)))
        self.valid.clear()
        return self.original["decode_forward"](*args, **kwargs)

    def chunk(self, count, slot):
        assert self.active is not None
        row = self.active[slot]
        offset, end = row["offset"], row["end"]
        if count == 1 and offset == end - 1 and end % 2 == 1:
            assert offset % 2 == 0  # This singleton cannot write compressed_keys.
            self.copy(compact_state(self.gen.prefill_cache), self.buffers[slot])
            self.valid[slot] = dict(end=end, token=row["last_token"])
            self.report["saved"].append(dict(slot=slot, position=offset, logical_end=end))
        self.original["_prefill_chunk"](count, slot)
        row["offset"] += count

    def prefill(self, tokens, *args, **kwargs):
        assert not args and not kwargs.get("slot_remap"), "Diagnostic accepts keyword prefill metadata, no migration"
        slots = tuple(kwargs.get("slots") or range(len(kwargs["prompt_lens"])))
        ends = tuple(map(int, kwargs["prompt_lens"]))
        starts = tuple(map(int, kwargs.get("start_pos") or [0] * len(slots)))
        rewritten = list(starts)
        dropped = [0] * len(slots)
        for row, (slot, start, end) in enumerate(zip(slots, starts, ends, strict=True)):
            cache = self.gen.caches[slot]
            saved = self.valid.get(slot)
            # This narrow diagnostic covers unchanged mappings. Do not restore
            # old page-table ownership or claim a remap/COW implementation.
            incoming = kwargs.get("page_table")
            previous = self.gen.page_tables.get(slot)
            mapping_matches = incoming is None or (
                previous is not None and all(torch.equal(value, incoming[row : row + 1]) for value in previous)
            )
            if start and start % 2:
                if (
                    mapping_matches
                    and saved
                    and saved["end"] == start
                    and cache.position == start
                    and saved["token"] == int(tokens[row, start - 1])
                ):
                    self.copy(self.buffers[slot], compact_state(cache))
                    cache.position = start - 1
                    rewritten[row], dropped[row] = start - 1, 1
                    self.report["restored"].append(dict(slot=slot, start=start, replay_from=start - 1, end=end))
                else:
                    self.report["missing_or_invalid"].append(dict(slot=slot, start=start, end=end))
            self.valid.pop(slot, None)
        self.active = {
            slot: dict(offset=start, end=end, last_token=int(tokens[row, end - 1]))
            for row, (slot, start, end) in enumerate(zip(slots, rewritten, ends, strict=True))
        }
        try:
            result = self.original["prefill_forward"](tokens, **dict(kwargs, start_pos=rewritten))
        finally:
            self.active = None
        if kwargs.get("return_all_logits") and any(dropped):
            rows = [
                result[row, drop : drop + end - start].clone()
                for row, (drop, end, start) in enumerate(zip(dropped, ends, starts, strict=True))
            ]
            result = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
        return result

    def close(self):
        for name, original in self.original.items():
            setattr(self.gen, name, original)
