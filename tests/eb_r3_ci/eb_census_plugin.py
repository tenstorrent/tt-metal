# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary census (CI only): every ttnn add, subtract and multiply call site a test reaches, once per call
site and operand description, written to EB_R3_LOG_CALLS before the call, so binary_ng's own EB_CALL line (one per program
built) follows the first site that builds it. Load with -p eb_census_plugin (PYTHONPATH tests/eb_r3_ci)."""
import os
import sys

_LOG = os.environ.get("EB_R3_LOG_CALLS")
_NAMES = ["add", "add_", "subtract", "subtract_", "sub", "sub_", "multiply", "multiply_", "mul", "mul_", "rsub", "rsub_"]
_KW = ["activations", "input_tensor_a_activations", "input_tensor_b_activations", "dtype", "memory_config",
       "fast_and_approximate_mode", "use_legacy", "output_tensor"]
_seen = set()


def _mc(mc):
    try:
        r = f"{str(mc.memory_layout).split('.')[-1]}/{str(mc.buffer_type).split('.')[-1]}"
        ss = mc.shard_spec
        if ss is not None:
            r += f"/n{ss.grid.num_cores()}/s{ss.shape[0]}x{ss.shape[1]}"
        return r
    except Exception:
        return type(mc).__name__


def _desc(t):
    if isinstance(t, (int, float, bool)):
        return f"scalar({type(t).__name__})"
    try:
        return f"{str(t.dtype).split('.')[-1]}/{_mc(t.memory_config())}/{list(t.shape)}"
    except Exception:
        return type(t).__name__


def _kw(k, v):
    if v is None:
        return None
    if k == "memory_config":
        return f"{k}={_mc(v)}"
    if k == "output_tensor":
        return f"{k}={_desc(v)}"
    if isinstance(v, (list, tuple)):
        return f"{k}=[{','.join(str(getattr(x, 'op_type', x)).split('.')[-1] for x in v)}]"
    return f"{k}={str(v).split('.')[-1]}"


def _site():
    f = sys._getframe(2)
    first_test = None
    while f is not None:
        fn = f.f_code.co_filename
        if "/models/" in fn and "/site-packages/" not in fn:
            return f"models/{fn.split('/models/', 1)[1]}:{f.f_lineno}"
        if first_test is None and "/tests/" in fn and "eb_census" not in fn and "/site-packages/" not in fn:
            first_test = f"tests/{fn.split('/tests/', 1)[1]}:{f.f_lineno}"
        f = f.f_back
    return first_test or "?"


def _wrap(name, fn):
    def w(*args, **kwargs):
        try:
            a = args[0] if args else kwargs.get("input_tensor_a", kwargs.get("input_a"))
            b = args[1] if len(args) > 1 else kwargs.get("input_tensor_b", kwargs.get("input_b", kwargs.get("scalar")))
            kws = [x for x in (_kw(k, kwargs.get(k)) for k in _KW) if x]
            line = f"EB_SITE {name} {_site()} a={_desc(a)} b={_desc(b)} {' '.join(kws)}"
            if line not in _seen:
                _seen.add(line)
                with open(_LOG, "a") as fh:
                    fh.write(line + "\n")
        except Exception as e:  # never break the test
            pass
        return fn(*args, **kwargs)

    w.__name__ = getattr(fn, "__name__", name)
    return w


def pytest_configure(config):
    if not _LOG:
        return
    import ttnn

    for n in _NAMES:
        fn = getattr(ttnn, n, None)
        if fn is not None and not getattr(fn, "_eb_census", False):
            w = _wrap(n, fn)
            w._eb_census = True
            setattr(ttnn, n, w)
