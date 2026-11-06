# SPDX-License-Identifier: Apache-2.0
"""Prepared equal-length short-prefill groups over distinct request-owned pages."""

import torch

import ttnn


class PreparedBatchedPrefill:
    def __init__(self, prepared, batch, *, max_rows=1024):
        # Keep hidden RMSNorm in the same reduction family as solo prefill.
        # Larger row counts select a numerically different interleaved kernel.
        self.solo = prepared
        self.model = prepared.model
        self.inputs = {}
        for size in (32, 64, 96, 128):
            if size > prepared.context:
                continue
            for count in (2, 4, 8, 16, 32):
                if count > batch or count * size > max_rows:
                    continue
                self.inputs[count, size] = {
                    "tokens": prepared.upload(torch.zeros(count, size, dtype=torch.int32)),
                    "positions": prepared.upload(torch.arange(size, dtype=torch.int32).repeat(count, 1)),
                    "indices": prepared.upload(torch.zeros(1, count, dtype=torch.int32)),
                    "fill": [
                        prepared.upload(torch.zeros(count, size // 32, dtype=torch.int32), signed=True)
                        for _ in self.model.layers
                    ],
                    "attention": [
                        prepared.upload(torch.zeros(count, table.shape[1], dtype=torch.int32), signed=True)
                        for table in prepared.attention.values()
                    ],
                }

    def allocation_bytes_per_device(self):
        """Physical DRAM bank reservations for the prepared input buffers."""
        banks = ttnn.get_memory_view(self.model.mesh, ttnn.BufferType.DRAM).num_banks
        seen = set()

        def size(value):
            if isinstance(value, dict):
                return sum(size(item) for item in value.values())
            if isinstance(value, list):
                return sum(size(item) for item in value)
            part = ttnn.get_device_tensors(value)[0]
            uid = part.buffer_unique_id()
            if uid in seen:
                return 0
            seen.add(uid)
            return ((part.buffer_num_pages() + banks - 1) // banks) * part.buffer_aligned_page_size() * banks

        return int(size(self.inputs))

    def groups(self, lengths, starts):
        by_size = {}
        for row, (length, start) in enumerate(zip(lengths, starts)):
            if start == 0 and 0 < length <= 128:
                size = ((length + 31) // 32) * 32
                by_size.setdefault(size, []).append(row)
        result = []
        for size, rows in by_size.items():
            while len(rows) >= 2:
                counts = [n for n, s in self.inputs if s == size and n <= len(rows)]
                if not counts:
                    break
                count = max(counts)
                result.append((size, rows[:count]))
                rows = rows[count:]
        return result

    def run(self, tokens, lengths, slots, *, size, kv_cache):
        count = len(lengths)
        inputs = self.inputs[count, size]
        padded = torch.zeros(count, size, dtype=torch.int32)
        positions = torch.arange(size, dtype=torch.int32).repeat(count, 1)
        for row, length in enumerate(lengths):
            padded[row, :length] = tokens[row, :length]
            positions[row, length:] = length - 1
        self.solo.copy(padded, inputs["tokens"])
        self.solo.copy(positions, inputs["positions"])
        self.solo.copy(
            torch.tensor([[row * size + length - 1 for row, length in enumerate(lengths)]], dtype=torch.int32),
            inputs["indices"],
        )
        for index, pages in enumerate(self.solo.host_pages):
            selected = pages[slots]
            self.solo.copy(selected[:, : size // 32], inputs["fill"][index])
            width = inputs["attention"][index].shape[1]
            self.solo.copy(selected[:, :width], inputs["attention"][index])
            self.solo.copy(torch.zeros(1, dtype=torch.int32), self.solo.offsets[index])
        # Every row starts at zero; padding stays inside its final owned page.
        hidden = self.model.prefill_chunk(
            inputs["tokens"],
            inputs["positions"],
            kv_cache=kv_cache,
            page_tables=inputs["attention"],
            chunk_tables=inputs["fill"],
            chunk_start=0,
            valid_tokens=size,
            attention_tables=inputs["attention"],
            start_tensors=list(self.solo.offsets.values()),
            return_hidden=True,
        )
        flat = ttnn.reshape(ttnn.untilize(hidden, use_multicore=True), (count * size, self.model.config.hidden_size))
        selected = ttnn.embedding(inputs["indices"], flat, layout=ttnn.TILE_LAYOUT)
        logits = self.model.terminal(ttnn.reshape(selected, (1, 1, count, self.model.config.hidden_size)))
        # These offsets are bounded by prepared group size and warmed before capture.
        return [ttnn.slice(logits, (0, 0, row, 0), (1, 1, row + 1, logits.shape[-1])) for row in range(count)]

    def warm(self, kv_cache):
        for count, size in self.inputs:
            print("WARM_BATCHED_PREFILL", count, size, flush=True)
            outputs = self.run(
                torch.zeros(count, size, dtype=torch.int32),
                [size] * count,
                list(range(count)),
                size=size,
                kv_cache=kv_cache,
            )
            del outputs
