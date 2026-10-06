# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Records every ttnn call a layer makes, with the range of device-operation ids (== GLOBAL CALL COUNT of the device
profiler digest) it launched, its caller (file:line:function in the model code) and tensor shapes/dtypes/layouts.

Only the OUTERMOST ttnn call is recorded (calls nested in python composites are folded into it).
"""

import os
import sys
import types

import ttnn
from ttnn import _ttnn

SKIP = {
    "synchronize_device",
    "from_torch",
    "to_torch",
    "to_device",
    "from_device",
    "deallocate",
    "begin_trace_capture",
    "end_trace_capture",
    "execute_trace",
    "release_trace",
    "ReadDeviceProfiler",
    "get_memory_view",
    "open_mesh_device",
    "close_mesh_device",
    "get_device_tensors",
    "init_device_compute_kernel_config",
    "dump_tensor",
    "load_tensor",
    "as_tensor",
    "create_global_semaphore",
    "reset_global_semaphore_value",
    "get_tt_metal_version",
    "set_printoptions",
    "tilize_with_zero_padding_cpu",
    "manage_config",
    "register_python_operation",
    "get_arch_name",
    "num_cores_to_corerangeset",
}
NAMESPACES = [
    ("ttnn", lambda: ttnn),
    ("ttnn.experimental", lambda: ttnn.experimental),
    ("ttnn.experimental.deepseek", lambda: ttnn.experimental.deepseek),
    ("ttnn.experimental.deepseek.moe", lambda: ttnn.experimental.deepseek.moe),
    ("ttnn.experimental.deepseek_prefill", lambda: ttnn.experimental.deepseek_prefill),
    ("ttnn.transformer", lambda: ttnn.transformer),
]
HERE = os.path.dirname(os.path.abspath(__file__))


def dev_id():
    return int(_ttnn.get_device_operation_id())


def tdesc(t):
    try:
        shape = "x".join(str(d) for d in t.shape)
        dt = str(t.dtype).split(".")[-1].replace("float", "f").replace("bfloat", "bf").replace("bf16", "bf16")
        lay = "T" if t.layout == ttnn.TILE_LAYOUT else "RM"
        mc = t.memory_config()
        mem = "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
        if mc.is_sharded():
            mem += "-sh"
        return f"[{shape}] {dt} {lay} {mem}"
    except Exception as e:  # noqa
        return f"<tensor {type(e).__name__}>"


def collect(v, out, depth=0):
    if isinstance(v, ttnn.Tensor):
        out.append(v)
    elif depth < 2 and isinstance(v, (list, tuple)):
        for e in v:
            collect(e, out, depth + 1)
    elif depth < 2 and isinstance(v, dict):
        for e in v.values():
            collect(e, out, depth + 1)


def kdesc(k, v):
    if isinstance(v, (bool, int, float, str, type(None))):
        return f"{k}={v}"
    if hasattr(v, "buffer_type") and hasattr(v, "is_sharded"):
        try:
            return f"{k}={'L1' if v.buffer_type == ttnn.BufferType.L1 else 'DRAM'}{'-sh' if v.is_sharded() else ''}"
        except Exception:  # noqa
            return None
    if isinstance(v, (list, tuple)) and len(v) <= 6 and all(isinstance(e, (int, float, str, bool)) for e in v):
        return f"{k}={list(v)}"
    if isinstance(v, ttnn.DataType) or isinstance(v, ttnn.Layout):
        return f"{k}={str(v).split('.')[-1]}"
    return None


def caller():
    """first non-ttnn frame, plus (after '<') the nearest frame of the prefill layer/model/attention/sparse/moe files (the model-level caller)."""
    f = sys._getframe(2)
    first, outer = None, None
    while f is not None:
        fn = f.f_code.co_filename
        if (
            "/ttnn/" not in fn.replace("deepseek_v41_flash", "")
            and not fn.endswith("op_table_recorder.py")
            and "site-packages" not in fn
            and "python_env" not in fn
        ):
            d = f"{os.path.basename(fn)}:{f.f_lineno}:{f.f_code.co_name}"
            first = first or d
            if os.path.basename(fn) in (
                "prefill_layer.py",
                "prefill_model.py",
                "prefill_attention.py",
                "prefill_sparse.py",
                "prefill_unified_moe.py",
                "prefill_dyn.py",
            ):
                outer = d
                break
        f = f.f_back
    if first is None:
        return "?"
    return first if outer is None or outer == first else first + "<" + outer


class OpRecorder:
    def __init__(self):
        self.rows = []
        self.zero = {}  # name|caller -> count of calls that launched no device program
        self.enabled = False
        self.depth = 0
        self.marks = []
        self.installed = False

    def mark(self, name):
        self.marks.append((name, dev_id(), len(self.rows)))

    def _wrap(self, qual, fn):
        rec = self

        def wrapper(*a, **k):
            if not rec.enabled or rec.depth > 0:
                return fn(*a, **k)
            rec.depth += 1
            i0 = dev_id()
            try:
                out = fn(*a, **k)
            finally:
                rec.depth -= 1
            i1 = dev_id()
            try:
                who = caller_wrapped()
                if i1 == i0:
                    key = f"{qual}|{who}"
                    rec.zero[key] = rec.zero.get(key, 0) + 1
                    return out
                ins, outs = [], []
                for v in a:
                    collect(v, ins)
                for v in k.values():
                    collect(v, ins)
                collect(out, outs)
                kw = [s for s in (kdesc(kk, vv) for kk, vv in k.items()) if s]
                extra = ""
                if qual.endswith("generic_op"):
                    try:
                        extra = ",".join(os.path.basename(str(kd.kernel_source)) for kd in a[1].kernels)
                    except Exception:  # noqa
                        pass
                rec.rows.append(
                    dict(
                        idx=len(rec.rows),
                        op=qual,
                        id0=i0,
                        id1=i1,
                        caller=who,
                        ins=[tdesc(t) for t in ins],
                        outs=[tdesc(t) for t in outs],
                        kw=kw,
                        kernels=extra,
                    )
                )
            except Exception as e:  # noqa
                rec.rows.append(
                    dict(
                        idx=len(rec.rows),
                        op=qual,
                        id0=i0,
                        id1=i1,
                        caller="?",
                        ins=[],
                        outs=[],
                        kw=[f"recorder error {e}"],
                        kernels="",
                    )
                )
            return out

        wrapper.__name__ = getattr(fn, "__name__", qual)
        wrapper.__wrapped__ = fn
        return wrapper

    def install(self):
        if self.installed:
            return
        for qual, get in NAMESPACES:
            ns = get()
            for name in list(dir(ns)):
                if name.startswith("_") or name in SKIP:
                    continue
                try:
                    obj = getattr(ns, name)
                except Exception:  # noqa
                    continue
                if isinstance(obj, (type, types.ModuleType)) or not callable(obj):
                    continue
                try:
                    setattr(ns, name, self._wrap(f"{qual}.{name}", obj))
                except Exception:  # noqa
                    pass
        for nm in (
            "__add__",
            "__radd__",
            "__sub__",
            "__rsub__",
            "__mul__",
            "__rmul__",
            "__truediv__",
            "__matmul__",
            "__neg__",
        ):
            try:
                fn = getattr(ttnn.Tensor, nm)
                setattr(ttnn.Tensor, nm, self._wrap(f"Tensor.{nm}", fn))
            except Exception:  # noqa
                pass
        self.installed = True


def caller_wrapped():
    # frames: caller_wrapped <- wrapper <- model code
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if (
            "site-packages" not in fn
            and "python_env" not in fn
            and not fn.endswith("op_table_recorder.py")
            and "/ttnn/ttnn/" not in fn
        ):
            return f"{os.path.basename(fn)}:{f.f_lineno}:{f.f_code.co_name}"
        f = f.f_back
    return "?"
