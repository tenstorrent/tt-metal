# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41_TRACE_ALLOC_LOG=1 (with TT_METAL_TRACE_ALLOC_TRACKING=1, optionally TT_METAL_TRACE_ALLOC_TRACEBACKS=1): the ttnn trace-allocation tracker RAISES at the first ``execute_trace`` that finds a
buffer allocated after the trace's capture and still alive ("will be corrupted on replay"). This turns the raise into a log line (once per distinct buffer set): the trace id, the number of buffers,
and a count per allocating op (+ the model frame when tracebacks are on), then lets the replay go on."""

import os
import re
import sys
from collections import Counter

_SEEN = set()


def install():
    if os.environ.get("DSV41_TRACE_ALLOC_LOG") != "1":
        return
    try:
        from ttnn.tools import trace_allocation_tracker as T
    except Exception as e:  # noqa: BLE001
        print(f"TRACE-ALLOC-DIAG: tracker not importable: {e!r}", flush=True)
        return
    cls = T.TraceAllocationTracker
    if getattr(cls, "_dsv41_diag", False):
        return
    orig = cls.verify_before_replay

    def verify(mesh_device, trace_id):
        try:
            return orig(mesh_device, trace_id)
        except RuntimeError as e:
            msg = str(e)
            if "still alive before trace replay" not in msg:
                raise
            ids = tuple(sorted(re.findall(r"^Buffer (\d+)", msg, re.M)))
            key = (trace_id if isinstance(trace_id, int) else id(trace_id), ids)
            if key in _SEEN:
                return None
            _SEEN.add(key)
            ops = Counter()
            frames = Counter()
            cur, last = None, None
            chain = []

            def flush():
                if cur is not None and chain:
                    frames[(cur, " < ".join(reversed(chain[-3:])))] += 1

            for ln in msg.splitlines():
                m = re.match(r"Buffer (\d+)(?: \[op: (.*)\])?$", ln)
                if m:
                    flush()
                    cur, last = (m.group(2) or "?")[:60], None
                    chain = []
                    ops[cur] += 1
                elif cur is not None and "deepseek_v41_flash/" in ln and "trace_alloc_diag" not in ln:
                    last = ln.strip().split("deepseek_v41_flash/")[-1][:100]
                    chain.append(last.replace('", line ', ":").replace(", in ", ":"))
                elif cur is not None and ln.startswith("  File") and last is not None:
                    pass
            flush()
            print(
                f"TRACE-ALLOC-DIAG trace {trace_id}: {len(ids)} buffers alive before replay (allocated after its capture)",
                file=sys.stdout,
                flush=True,
            )
            for k, v in ops.most_common():
                print(f"TRACE-ALLOC-DIAG   x{v} {k}", flush=True)
            for (op, fr), v in frames.most_common(40):
                print(f"TRACE-ALLOC-DIAG   x{v} {op} <- {fr}", flush=True)
            return None

    cls.verify_before_replay = staticmethod(verify)
    cls._dsv41_diag = True


_CAP = [0]
_UP_SEEN = set()


def install_upload_log():
    """DSV41_TRACE_UPLOAD_LOG=1: log (once per call site) every host -> device write (``from_torch`` with a device, ``copy_host_to_device_tensor``, ``to_device``) issued while a trace capture is open.
    Such a write is EXECUTED at capture time (ops are only recorded): the data sits in a buffer that is NOT written again by the replay, so if the buffer is freed (or was a temporary) the
    replay reads whatever the allocator left there."""
    if os.environ.get("DSV41_TRACE_UPLOAD_LOG") != "1":
        return
    import traceback

    import ttnn

    if getattr(ttnn, "_dsv41_upload_log", False):
        return
    ttnn._dsv41_upload_log = True
    ob, oe = ttnn.begin_trace_capture, ttnn.end_trace_capture

    def bt(*a, **k):
        _CAP[0] += 1
        return ob(*a, **k)

    def et(*a, **k):
        r = oe(*a, **k)
        _CAP[0] = max(0, _CAP[0] - 1)
        return r

    ttnn.begin_trace_capture, ttnn.end_trace_capture = bt, et

    def wrap(name, pick_dev):
        f = getattr(ttnn, name)

        def w(*a, **k):
            if _CAP[0] and pick_dev(a, k):
                fr = [
                    f"{x.filename.split('deepseek_v41_flash/')[-1]}:{x.lineno}:{x.name}"
                    for x in traceback.extract_stack()[:-1]
                    if "deepseek_v41_flash/" in x.filename
                ]
                key = (name, tuple(fr[-3:]))
                if key not in _UP_SEEN:
                    _UP_SEEN.add(key)
                    print(f"TRACE-UPLOAD {name} during capture <- {' > '.join(fr[-4:])}", flush=True)
            return f(*a, **k)

        setattr(ttnn, name, w)

    wrap("from_torch", lambda a, k: k.get("device") is not None)
    wrap("copy_host_to_device_tensor", lambda a, k: True)
    wrap("to_device", lambda a, k: True)
