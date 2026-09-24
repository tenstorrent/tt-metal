import time

import ttnn


def make_traced(D):
    def traced(fn, n=4):
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(D)
        tid = ttnn.begin_trace_capture(D, cq_id=0)
        try:
            outs = [fn() for _ in range(n)]
            ttnn.end_trace_capture(D, tid, cq_id=0)
        except Exception:
            try:
                ttnn.end_trace_capture(D, tid, cq_id=0)
            except Exception:
                pass
            try:
                ttnn.release_trace(D, tid)
            except Exception:
                pass
            raise
        ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
        t0 = time.perf_counter()
        for _ in range(8):
            ttnn.execute_trace(D, tid, cq_id=0, blocking=True)
        us = (time.perf_counter() - t0) / 8 / n * 1e6
        ttnn.release_trace(D, tid)
        for o in outs:
            ttnn.deallocate(o)
        return us

    return traced
