# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tiny timing helpers shared by the BEVFormer encoder, TSA, SCA, and
MSDeformAttn so all of them can record into one dict without a circular
import. Gated on TT_UNIAD_TIMING=1, active only from call#2 onward (warm
cache). All helpers are no-ops otherwise so import sites can call them
unconditionally.

Caveat: `record()` / `sync_now()` call `ttnn.synchronize_device`, which
is rejected inside `ttnn.begin_trace_capture`. The encoders therefore
pass `_enc_stats=None` on the trace-capture call (the first warm call of
a given shape), so that call's per-sub-phase timings show up as zeros.
Sub-phase numbers are only meaningful on subsequent (trace-replay) calls
— read the totals, not the breakdown, on the capture call."""

import os
import time

import ttnn

_ENABLED = os.environ.get("TT_UNIAD_TIMING") == "1"
_CALL_IDX = [0]


def bump_call_idx():
    _CALL_IDX[0] += 1


def call_idx():
    return _CALL_IDX[0]


def active():
    return _ENABLED and _CALL_IDX[0] >= 2


def sync_now(device):
    if active():
        ttnn.synchronize_device(device)
        return time.perf_counter()
    return 0.0


def record(stats, key, t0, device):
    if not active():
        return
    ttnn.synchronize_device(device)
    stats[key] = stats.get(key, 0.0) + (time.perf_counter() - t0)
