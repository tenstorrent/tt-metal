# SPDX-License-Identifier: Apache-2.0
"""Prepared prefill inputs: logical lengths and page ownership remain runtime data."""

import torch

import ttnn


class PreparedPrefill:
    def __init__(self, model, context, host_page_tables):
        self.model = model
        self.context = context
        self.host_pages = host_page_tables
        self.buckets = tuple(range(32, min(1024, ((context + 31) // 32) * 32) + 1, 32))
        self.inputs = {}
        for size in self.buckets:
            self.inputs[size] = dict(
                tokens=self.upload(torch.zeros(1, size, dtype=torch.int32)),
                positions=self.upload(torch.zeros(1, size, dtype=torch.int32)),
                fill={
                    index: self.upload(torch.zeros(1, size // 32, dtype=torch.int32), signed=True)
                    for index in range(len(model.layers))
                },
            )
        self.attention = {
            index: self.upload(
                torch.zeros(1, 64 if layer.sliding_window else ((context + 127) // 128) * 4, dtype=torch.int32),
                signed=True,
            )
            for index, layer in enumerate(model.layers)
        }
        self.offsets = {
            index: self.upload(torch.zeros(1, dtype=torch.int32), signed=True) for index in range(len(model.layers))
        }
        self.partial_pages = {
            index: (self.upload(torch.zeros(512, dtype=torch.int32)), self.upload(torch.zeros(512, dtype=torch.int32)))
            for index in range(len(model.layers))
        }
        self.partial_masks = {}
        for layer in model.layers:
            key = (layer.nkv, layer.head_dim)
            if key not in self.partial_masks:
                self.partial_masks[key] = model.upload(
                    torch.zeros(1, layer.nkv, 32, layer.head_dim, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
        # Tensor-indexed slice reads a full source tile from the index buffers.
        self.selection_start = self.upload(torch.zeros(512, dtype=torch.int32))
        self.selection_end = self.upload(torch.zeros(512, dtype=torch.int32))
        self.selection_index = self.upload(torch.zeros(1, 1, dtype=torch.int32))
        self.selection_indices = self.upload(torch.zeros(1, 32, dtype=torch.int32))

    def upload(self, value, *, signed=False):
        return self.model.upload(value, dtype=ttnn.int32 if signed else ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def copy(self, value, target):
        host = ttnn.from_torch(
            value, dtype=target.dtype, layout=target.layout, mesh_mapper=ttnn.ReplicateTensorToMesh(self.model.mesh)
        )
        ttnn.copy_host_to_device_tensor(host, target)

    def stage(self, tokens, *, slot, start, size, valid):
        inputs = self.inputs[size]
        padded = torch.zeros(1, size, dtype=torch.int32)
        padded[0, :valid] = tokens[:valid]
        self.copy(padded, inputs["tokens"])
        self.copy(torch.arange(start, start + size, dtype=torch.int32).reshape(1, -1), inputs["positions"])
        for kind, (layer, pages) in enumerate(zip(self.model.layers, self.host_pages)):
            row = pages[slot : slot + 1]
            self.copy(row[:, start // 32 : (start + size) // 32], inputs["fill"][kind])
            offset = start
            if layer.sliding_window:
                base = max(0, start - 1024)
                # Every entry remains a valid page owned by this request.
                # Absolute page ownership supports both standalone cyclic maps
                # and scheduler-owned sliding-window blocks.
                indices = (torch.arange(64) + base // 32).clamp(max=row.shape[1] - 1)
                attention = row[:, indices]
                offset = start - base
            else:
                attention = row
            self.copy(attention, self.attention[kind])
            self.copy(torch.tensor([offset], dtype=torch.int32), self.offsets[kind])
        last = valid - 1
        aligned = last // 32 * 32
        self.copy_slice_indices([0, 0, aligned, 0], self.selection_start)
        self.copy_slice_indices([1, 1, aligned + 32, self.model.config.hidden_size], self.selection_end)
        self.copy(torch.tensor([[last % 32]], dtype=torch.int32), self.selection_index)

    def hidden(self, *, size, start, kv_cache, preserve_prefix=False):
        inputs = self.inputs[size]
        kinds = range(len(self.model.layers))
        return self.model.prefill_chunk(
            inputs["tokens"],
            inputs["positions"],
            kv_cache=kv_cache,
            page_tables=[self.attention[k] for k in kinds],
            chunk_tables=[inputs["fill"][k] for k in kinds],
            chunk_start=start,
            valid_tokens=size,
            attention_tables=[self.attention[k] for k in kinds],
            start_tensors=[self.offsets[k] for k in kinds],
            return_hidden=True,
            partial_cache_inputs=(
                [
                    (*self.partial_pages[k], self.partial_masks[(layer.nkv, layer.head_dim)])
                    for k, layer in enumerate(self.model.layers)
                ]
                if preserve_prefix
                else None
            ),
        )

    def last_row(self, hidden):
        size = hidden.shape[2]
        tile = ttnn.slice(hidden, self.selection_start, self.selection_end, slice_dim=2, num_devices=size // 32)
        rows = ttnn.reshape(ttnn.untilize(tile, use_multicore=True), (32, self.model.config.hidden_size))
        selected = ttnn.embedding(self.selection_index, rows, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(selected, (1, 1, 1, self.model.config.hidden_size))

    def select_rows(self, hidden, indices):
        """Select up to 32 hidden rows using one fixed-shape runtime index input."""
        if not 1 <= len(indices) <= 32 or any(not 0 <= index < hidden.shape[2] for index in indices):
            raise ValueError("Hidden-row selection outside physical chunk")
        padded = torch.full((1, 32), int(indices[-1]), dtype=torch.int32)
        padded[0, : len(indices)] = torch.tensor(indices, dtype=torch.int32)
        self.copy(padded, self.selection_indices)
        rows = ttnn.reshape(ttnn.untilize(hidden, use_multicore=True), (hidden.shape[2], self.model.config.hidden_size))
        selected = ttnn.embedding(self.selection_indices, rows, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(selected, (1, 1, 32, self.model.config.hidden_size))

    def copy_slice_indices(self, indices, target):
        values = torch.zeros(512, dtype=torch.int32)
        values[:4] = torch.tensor(indices, dtype=torch.int32)
        self.copy(values, target)

    def partial_hidden(self, tokens, *, slot, start, kv_cache):
        """Prefill only new rows in a partial page, preserving cached prefix K/V."""
        leading = start % 32
        valid = len(tokens)
        if not 0 < leading < 32 or not 0 < valid <= 32 - leading:
            raise ValueError("Partial prefill must fit the unfinished physical page")
        base = start - leading
        tile_tokens = torch.zeros(32, dtype=torch.int32)
        tile_tokens[leading : leading + valid] = tokens
        self.stage(tile_tokens, slot=slot, start=base, size=32, valid=32)
        for key, mask in self.partial_masks.items():
            heads, width = key
            values = torch.zeros(1, heads, 32, width, dtype=torch.bfloat16)
            values[:, :, :leading, :] = 1
            self.copy(values, mask)
        for index, (layer, pages) in enumerate(zip(self.model.layers, self.host_pages)):
            physical_page = int(pages[slot, base // 32])
            begin, end = self.partial_pages[index]
            self.copy_slice_indices([physical_page, 0, 0, 0], begin)
            self.copy_slice_indices([physical_page + 1, layer.nkv, 32, layer.head_dim], end)
        hidden = self.hidden(size=32, start=base, kv_cache=kv_cache, preserve_prefix=True)
        selected = self.select_rows(hidden, list(range(leading, leading + valid)))
        del hidden
        # Returned rows are compacted; terminal sampling selects the last new row.
        self.copy(torch.tensor([[valid - 1]], dtype=torch.int32), self.selection_index)
        return selected
