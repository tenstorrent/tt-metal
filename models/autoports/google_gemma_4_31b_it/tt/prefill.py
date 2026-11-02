# SPDX-License-Identifier: Apache-2.0
"""Prepared prefill inputs: logical lengths and page ownership remain runtime data."""

import torch

import ttnn


def prefill_intervals(start, end, *, exact=True):
    """Yield (absolute start, logical count, physical size) without prefix replay.

    Size one selects masked decode updates. Bulk fills own complete 32-token
    pages. Only standalone callers may pad a final private page.
    """
    if start < 0 or end < start:
        raise ValueError("Invalid prefill interval")
    while start < end:
        if exact and (start % 32 or end - start < 32):
            yield start, 1, 1
            start += 1
        else:
            count = min(1024, end - start)
            if exact:
                count = count // 32 * 32
            size = (count + 31) // 32 * 32
            yield start, count, size
            start += count


class PreparedPrefill:
    def __init__(self, model, context, host_page_tables):
        self.model = model
        self.context = context
        self.host_pages = host_page_tables
        self.external_pages = any(
            layer.sliding_window and layer.cache_position_modulo is None for layer in model.layers
        )
        self.keys = (
            list(range(len(model.layers))) if self.external_pages else [layer.layer_type for layer in model.layers]
        )
        kinds = dict(zip(self.keys, [layer.layer_type for layer in model.layers]))
        self.buckets = tuple(range(32, min(1024, ((context + 31) // 32) * 32) + 1, 32))
        self.inputs = {}
        for size in self.buckets:
            self.inputs[size] = dict(
                tokens=self.upload(torch.zeros(1, size, dtype=torch.int32)),
                positions=self.upload(torch.zeros(1, size, dtype=torch.int32)),
                fill={kind: self.upload(torch.zeros(1, size // 32, dtype=torch.int32), signed=True) for kind in kinds},
            )
        self.attention = {
            kind: self.upload(
                torch.zeros(
                    1, 64 if layer_kind == "sliding_attention" else ((context + 127) // 128) * 4, dtype=torch.int32
                ),
                signed=True,
            )
            for kind, layer_kind in kinds.items()
        }
        self.offsets = {kind: self.upload(torch.zeros(1, dtype=torch.int32), signed=True) for kind in kinds}
        self.selection_start = self.upload(torch.tensor([0, 0, 0, 0], dtype=torch.int32))
        self.selection_end = self.upload(torch.tensor([1, 1, 32, model.config.hidden_size], dtype=torch.int32))
        self.selection_index = self.upload(torch.zeros(1, 1, dtype=torch.int32))

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
        done = set()
        for kind, layer, pages in zip(self.keys, self.model.layers, self.host_pages):
            if kind in done:
                continue
            done.add(kind)
            row = pages[slot : slot + 1]
            self.copy(row[:, start // 32 : (start + size) // 32], inputs["fill"][kind])
            offset = start
            if layer.sliding_window:
                base = max(0, start - 1024)
                # Every entry remains a valid page owned by this request.
                if self.external_pages:
                    attention = torch.zeros(1, 64, dtype=torch.int32)
                    valid_pages = row[:, base // 32 : base // 32 + 64]
                    attention[:, : valid_pages.shape[1]] = valid_pages
                else:
                    physical = row[:, :64]
                    attention = physical[:, (torch.arange(64) + base // 32) % 64]
                offset = start - base
            else:
                attention = row
            self.copy(attention, self.attention[kind])
            self.copy(torch.tensor([offset], dtype=torch.int32), self.offsets[kind])
        last = valid - 1
        aligned = last // 32 * 32
        self.copy(torch.tensor([0, 0, aligned, 0], dtype=torch.int32), self.selection_start)
        self.copy(
            torch.tensor([1, 1, aligned + 32, self.model.config.hidden_size], dtype=torch.int32), self.selection_end
        )
        self.copy(torch.tensor([[last % 32]], dtype=torch.int32), self.selection_index)

    def hidden(self, *, size, start, kv_cache):
        inputs = self.inputs[size]
        kinds = self.keys
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
        )

    def last_row(self, hidden):
        size = hidden.shape[2]
        tile = ttnn.slice(hidden, self.selection_start, self.selection_end, slice_dim=2, num_devices=size // 32)
        rows = ttnn.reshape(ttnn.untilize(tile, use_multicore=True), (32, self.model.config.hidden_size))
        selected = ttnn.embedding(self.selection_index, rows, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(selected, (1, 1, 1, self.model.config.hidden_size))
