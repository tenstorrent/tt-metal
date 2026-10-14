"""Opt-in, no-sync nested host spans for one Gemma prefill adapter invocation.

Integration: import install from this module; call install(self) at the END
of Gemma4ForCausalLM.__init__. Only enabled with GEMMA_PREFILL_SPANS=1.
No timer/logging is performed by nested wrappers outside an adapter prefill.
Inclusive spans overlap; exclusive spans partition the adapter host duration.
Neither measures device kernels or separates asynchronous dispatch from work.
"""
import functools
import json
import os
import queue
import sys
import threading
import time

_LOCAL = threading.local()
_RECORDS = queue.SimpleQueue()
_WRITER = None
_INSTALL_LOCK = threading.Lock()


def _write_records():
    while True:
        record = _RECORDS.get()
        try:
            sys.stdout.write("GEMMA_PREFILL_SPANS " + json.dumps(record, separators=(",", ":")) + "\n")
            sys.stdout.flush()
        except Exception:
            pass


def _counter_snapshot(adapter):
    return dict(adapter.generator.counters)


def _wrap(cls, method, label, outer=False):
    original = getattr(cls, method)
    if getattr(original, "_gemma_prefill_span_wrapped", False):
        return

    @functools.wraps(original)
    def wrapped(self, *args, **kwargs):
        current = getattr(_LOCAL, "record", None)
        owner = outer and current is None
        if owner:
            counters = _counter_snapshot(self)
            current = {
                "schema": 1,
                "start_wall_s": time.time(),
                "start_mono_s": time.perf_counter(),
                "status": "ok",
                "spans": {},
                "stack": [],
                "shape_count_before": len(self.generator.prefill_shapes),
                "trace_count_before": len(self.generator.traces),
            }
            _LOCAL.record = current
        if current is None:
            return original(self, *args, **kwargs)
        start = time.perf_counter()
        frame = {"child_s": 0.0}
        current["stack"].append(frame)
        try:
            return original(self, *args, **kwargs)
        except BaseException as error:
            if owner:
                current["status"] = type(error).__name__
            raise
        finally:
            elapsed = time.perf_counter() - start
            current["stack"].pop()
            if current["stack"]:
                current["stack"][-1]["child_s"] += elapsed
            span = current["spans"].setdefault(label, {"count": 0, "inclusive_s": 0.0, "exclusive_s": 0.0})
            span["count"] += 1
            span["inclusive_s"] += elapsed
            span["exclusive_s"] += elapsed - frame["child_s"]
            if owner:
                _LOCAL.record = None
                current.pop("stack")
                current["end_wall_s"] = time.time()
                current["shape_count_after"] = len(self.generator.prefill_shapes)
                current["trace_count_after"] = len(self.generator.traces)
                after = _counter_snapshot(self)
                current["counter_delta"] = {k: after.get(k, 0) - counters.get(k, 0) for k in set(counters) | set(after)}
                _RECORDS.put(current)

    wrapped._gemma_prefill_span_wrapped = True
    setattr(cls, method, wrapped)


def install(adapter):
    """Install once per class, only after the adapter has its generator."""
    global _WRITER
    if os.environ.get("GEMMA_PREFILL_SPANS") != "1" and not os.path.exists("/prefix-controlled/enable-spans"):
        return
    with _INSTALL_LOCK:
        generator = type(adapter.generator)
        for method, label in [
            ("prefill_forward", "generator_prefill"),
            ("_retire_traces", "trace_retire"),
            ("_restore_prefill_history", "prefix_restore"),
            ("sample_prefill_outputs", "sample_prefill"),
            ("_entry", "trace_entry"),
            ("_read_tokens", "token_readback"),
            ("_host_logits", "logits_readback"),
        ]:
            if hasattr(generator, method):
                _wrap(generator, method, label)
        _wrap(type(adapter.generator.model), "prefill_device", "model_prefill_device")
        _wrap(type(adapter), "prefill_forward", "adapter_prefill", outer=True)
        if _WRITER is None:
            _WRITER = threading.Thread(target=_write_records, name="gemma-prefill-spans", daemon=True)
            _WRITER.start()
