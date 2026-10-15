# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode-step breakdown through the vLLM adapter's traced path: host->device input upload,
device execution of the trace (sync'd), and logits readback. Prints median ms per phase and tok/s.

    pytest models/demos/molmo2/tests/bench_decode.py -v -s
"""

import statistics as st
import time

import torch
from loguru import logger

import ttnn
from models.demos.molmo2.tests.test_generator_vllm_adapter import adapter, mesh_device  # noqa: F401

PROMPT_LEN = 256
STEPS = 60


def test_decode_breakdown(adapter, mesh_device):
    m = adapter.model
    logger.info(f"lm_head sharded={m._lm_head_sharded} shape={m.lm_head.shape} devices={mesh_device.get_num_devices()}")
    ids = torch.randint(1000, 20000, (1, PROMPT_LEN), dtype=torch.int32)
    adapter.prefill_forward(ids, None, None, torch.tensor([PROMPT_LEN]))
    adapter.decode_forward(torch.tensor([[5]], dtype=torch.int32), torch.tensor([PROMPT_LEN]), None, None)  # capture
    up, ex, rd, tot = [], [], [], []
    for i in range(STEPS):
        pos = PROMPT_LEN + 1 + i
        t0 = time.perf_counter()
        m._update_decode_inputs(5, pos)
        t1 = time.perf_counter()
        m._dispatch_decode_trace(blocking=False)
        ttnn.synchronize_device(mesh_device)
        t2 = time.perf_counter()
        m._read_decode_output()
        t3 = time.perf_counter()
        up.append(t1 - t0), ex.append(t2 - t1), rd.append(t3 - t2), tot.append(t3 - t0)
    ms = lambda v: 1000 * st.median(v)  # noqa: E731
    logger.info(
        f"decode median ms: upload={ms(up):.2f} device={ms(ex):.2f} readback={ms(rd):.2f} total={ms(tot):.2f} "
        f"-> {1000 / ms(tot):.1f} tok/s"
    )
