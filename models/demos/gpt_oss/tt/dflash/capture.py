# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The drafter's context features from the gpt-oss target: the outputs of target layers 1, 6, 11, 16, 21 (each the input
residual of the next layer, DecoderLayer.capture_input) in the prompt's prefill, read back to the host as they are
produced (eager)."""

import torch

import ttnn


class TargetCapture:
    def __init__(self, model, layer_ids):
        self.model = model
        self.layer_ids = list(layer_ids)
        self.hidden = model.hf_config.hidden_size
        self.prefill = {L: [] for L in self.layer_ids}
        for L in self.layer_ids:
            model.layers[L + 1].capture_input = lambda t, rows_split=False, L=L: self._record(L, t, rows_split)

    def _host(self, t, rows_split=False):
        devs = ttnn.get_device_tensors(t)
        if rows_split:  # sequence-parallel prefill: device d holds rows d * S / TP onward
            return torch.cat([ttnn.to_torch(d).float() for d in devs], dim=-2)
        x = ttnn.to_torch(devs[0]).float()
        if x.shape[-1] != self.hidden and x.shape[-1] * len(devs) == self.hidden:
            x = torch.cat([ttnn.to_torch(d).float() for d in devs], dim=-1)
        return x

    # on_device (the batch-32 runner): the features stay on the device (the host round trip of 5 x [n, hidden] FP32
    # per user dominated a 32-user prefill at long prompts)
    on_device = False

    def _record(self, L, t, rows_split=False):
        if self.on_device:
            if rows_split:  # (sequence-parallel prefill: device d holds rows d * S / TP onward, the whole hidden)
                x = ttnn.all_gather(t, dim=2, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            elif t.shape[-1] != self.hidden:  # (TP-sharded hidden: gathered over the mesh row)
                x = ttnn.all_gather(t, dim=3, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            else:
                x = ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            self.prefill[L].append(ttnn.reshape(x, [1, 1, -1, self.hidden]))
            return
        x = self._host(t, rows_split)
        self.prefill[L].append(x.reshape(-1, x.shape[-1]))

    def prefill_features_device(self):
        """(on_device) [1, 1, rows, len(layer_ids) * hidden] BF16 on the device (rows: the prefill's, >= the prompt's);
        clears the prefill record."""
        parts = []
        for L in self.layer_ids:
            chunks = self.prefill[L]
            parts.append(chunks[0] if len(chunks) == 1 else ttnn.concat(chunks, dim=2))
            if len(chunks) > 1:
                for t in chunks:
                    t.deallocate(True)
        out = ttnn.concat(parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in parts:
            t.deallocate(True)
        self.prefill = {L: [] for L in self.layer_ids}
        return out

    def prefill_features(self, n):
        """[n, len(layer_ids) * hidden] for the first n prompt positions; clears the prefill record."""
        out = torch.cat([torch.cat(self.prefill[L])[:n] for L in self.layer_ids], dim=-1)
        self.prefill = {L: [] for L in self.layer_ids}
        return out
