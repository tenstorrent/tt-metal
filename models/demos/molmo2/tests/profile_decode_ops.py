# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-op device profile of one decode step (run under tracy, see README "Profiling").

    python -m tracy -p -r -v -m pytest models/demos/molmo2/tests/profile_decode_ops.py -s
Ops are identical to the traced step (the trace replays the same program set); eager mode is used so the
profiler can attribute every op.
"""

import torch

import ttnn
from models.demos.molmo2.tests.test_generator_vllm_adapter import adapter, mesh_device  # noqa: F401

PROMPT_LEN = 256


def test_profile_decode_ops(adapter, mesh_device):
    m = adapter.model
    ids = torch.randint(1000, 20000, (1, PROMPT_LEN), dtype=torch.int32)
    adapter.prefill_forward(ids, None, None, torch.tensor([PROMPT_LEN]))
    for i in range(2):  # compile
        m.forward_decode_step(5, PROMPT_LEN + i)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)  # drop everything before the measured step
    m.forward_decode_step(5, PROMPT_LEN + 2)
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)
