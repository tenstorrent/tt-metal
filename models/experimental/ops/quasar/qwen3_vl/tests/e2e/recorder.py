# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Capture per-stage TT outputs by wrapping module instances' forward."""
import time

import torch


def to_host(t):
    import ttnn

    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class StageRecorder:
    def __init__(self, progress, to_host=to_host):
        self.progress = progress
        self.to_host = to_host
        self.tensors = {}
        self.seconds = {}

    def wrap(self, monkeypatch, obj, stage, transform, when=lambda a, k: True, append_dim=None):
        orig = obj.forward

        def forward(*args, **kwargs):
            if not when(args, kwargs):
                return orig(*args, **kwargs)
            prev, self.progress.stage = self.progress.stage, stage
            t0 = time.time()
            try:
                out = orig(*args, **kwargs)
            finally:
                self.progress.stage = prev
            self.seconds[stage] = self.seconds.get(stage, 0.0) + time.time() - t0
            value = transform(self.to_host(_first(out)))
            if append_dim is not None and stage in self.tensors:
                value = torch.cat([self.tensors[stage], value], dim=append_dim)
            self.tensors[stage] = value
            return out

        monkeypatch.setattr(obj, "forward", forward)
