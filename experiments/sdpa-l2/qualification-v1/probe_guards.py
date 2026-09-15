# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prove unsupported FP32 requests reject before executing fallback kernels."""

import json

import torch
import ttnn

from qualify import inputs, run_device

torch.set_num_threads(8)
device = ttnn.open_device(device_id=0)
try:
    for n in (2048, 25920, 32769):
        case = dict(heads=5, q_len=n, kv_len=n, seed=1234, distribution="normal", offset=0)
        q, k, v = inputs(case)
        try:
            run_device(device, q, k, v, torch.arange(min(n, 32)), "accurate")
        except RuntimeError as exc:
            if "QUALIFICATION_UNSUPPORTED_FP32_STREAMING" not in str(exc):
                raise
            print(json.dumps(dict(length=n, guard_rejection=True, fallback_executed=False)), flush=True)
        else:
            raise AssertionError("FP32 fallback was not rejected")
finally:
    ttnn.close_device(device)
