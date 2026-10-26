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
            model.layers[L + 1].capture_input = lambda t, L=L: self._record(L, t)

    def _host(self, t):
        devs = ttnn.get_device_tensors(t)
        x = ttnn.to_torch(devs[0]).float()
        if x.shape[-1] != self.hidden and x.shape[-1] * len(devs) == self.hidden:
            x = torch.cat([ttnn.to_torch(d).float() for d in devs], dim=-1)
        return x

    def _record(self, L, t):
        x = self._host(t)
        self.prefill[L].append(x.reshape(-1, x.shape[-1]))

    def prefill_features(self, n):
        """[n, len(layer_ids) * hidden] for the first n prompt positions; clears the prefill record."""
        out = torch.cat([torch.cat(self.prefill[L])[:n] for L in self.layer_ids], dim=-1)
        self.prefill = {L: [] for L in self.layer_ids}
        return out
