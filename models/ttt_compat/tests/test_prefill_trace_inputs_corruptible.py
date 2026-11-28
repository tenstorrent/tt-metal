# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The persistent prefill trace inputs are acknowledged to the trace allocation tracker.

``_prefill_forward_trace`` rewrites them from host before every replay, so another live trace replaying over them is
harmless. Without the acknowledgment the tracker reports them against every trace that was live when they were
allocated, e.g. another model's decode trace when two models share a device in one process. Host only, no device.
"""

import contextlib
from types import SimpleNamespace

import torch

import models.ttt_compat.tt.generator as generator_module
from models.ttt_compat.tt.generator import Generator

HOST_INPUTS = ("tokens", "rot_mats_global", "rot_mats_local", "page_table", "chunk_page_table", "chunk_start_idx")
PERSISTENT_INPUTS = ("tokens", "page_table", "chunk_page_table", "chunk_start_idx")


class _Model:
    def prepare_prefill_inputs_trace(self, tokens, **kwargs):
        return HOST_INPUTS


def test_prefill_trace_inputs_are_allocated_as_corruptible(monkeypatch):
    mesh_device = object()
    events = []

    @contextlib.contextmanager
    def corruptible_allocation_scope(device):
        assert device is mesh_device
        events.append("scope begin")
        yield
        events.append("scope end")

    def copy_host_to_device(host_inputs, mesh_device=None, device_tensors=None):
        events.append(("allocate", tuple(host_inputs)))
        return list(host_inputs)

    monkeypatch.setattr(
        generator_module.trace_allocation_tracker, "corruptible_allocation_scope", corruptible_allocation_scope
    )
    monkeypatch.setattr(generator_module, "copy_host_to_device", copy_host_to_device)
    monkeypatch.setattr(generator_module.ttnn, "synchronize_device", lambda device: None)

    generator = Generator.__new__(Generator)  # host-only: no mesh device, no weights
    generator.model = [_Model()]
    generator.data_parallel = 1
    generator.model_args = [SimpleNamespace(mesh_device=mesh_device)]
    generator._prefill_trace_forward = lambda prepared, device_inputs: events.append("compile pass") or "output"

    prepared = generator._prepare_trace_prefill(torch.zeros(1, 128, dtype=torch.long), model_id=0)

    # Only the persistent inputs are acknowledged; the compile pass (program binaries, activations) is not.
    assert events == ["scope begin", ("allocate", PERSISTENT_INPUTS), "scope end", "compile pass"]
    assert prepared["device_inputs"] == list(PERSISTENT_INPUTS)
