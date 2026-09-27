# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Record every call a model makes to a derived bring-up op (ttnn.bringup.*, see ttnn/ttnn/bringup/INDEX.md).

A pytest plugin: load it with ``-p models.demos.common.bringup.testing.fork_capture`` and set
``BRINGUP_CAPTURE_FORKS=<out.json>``, e.g.

    BRINGUP_CAPTURE_FORKS=<bringup>/results/fork_calls.json BRINGUP_RUNG=last \
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/common/bringup/tests/test_ladder.py \
        -p models.demos.common.bringup.testing.fork_capture

It records each distinct call signature once, with a count:
- the op;
- every tensor argument's per-device shape, dtype, layout, memory placement and mesh size;
- every other argument's value (enums and compute-kernel configs as text).
It also records how the mesh was opened (``device``: the fabric config, and the test's ``device_params`` when the test
has them). Tensor contents are never recorded: fork tests build their own inputs. Each signature has a short id ``sig``, which is
how a fork's test cases say which captured call they cover (``models/demos/common/bringup/testing/fork_cases.py``
checks that).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ENV = "BRINGUP_CAPTURE_FORKS"
PREFIX = "ttnn.bringup."
_CALLS: dict[str, dict] = {}
_DEVICE: dict = {}  # how the mesh was opened: the fabric config, and the test's device_params when it has them


def _tensor(t) -> dict:
    import ttnn

    d = {
        "tensor": True,
        "shape": [int(x) for x in t.shape],
        "dtype": str(t.dtype).split(".")[-1],
        "layout": str(t.layout).split(".")[-1],
    }
    try:
        mc = t.memory_config()
        d["buffer"] = str(mc.buffer_type).split(".")[-1]
        d["memory_layout"] = str(mc.memory_layout).split(".")[-1]
        if mc.shard_spec is not None:
            d["shard_shape"] = [int(x) for x in mc.shard_spec.shape]
            d["shard_cores"] = int(mc.shard_spec.grid.num_cores())
    except Exception:  # host tensor or no memory config
        pass
    try:
        dev = t.device()
        d["mesh"] = [int(x) for x in dev.shape] if hasattr(dev, "shape") else [1]
        d["devices"] = len(ttnn.get_device_tensors(t))
    except Exception:
        pass
    return d


def _value(v):
    import ttnn

    if isinstance(v, ttnn.Tensor):
        return _tensor(v)
    if isinstance(v, ttnn.MemoryConfig):
        d = {"buffer": str(v.buffer_type).split(".")[-1], "memory_layout": str(v.memory_layout).split(".")[-1]}
        if v.shard_spec is not None:
            d["shard_shape"] = [int(x) for x in v.shard_spec.shape]
            d["shard_cores"] = int(v.shard_spec.grid.num_cores())
        return {"type": "MemoryConfig", **d}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _value(x) for k, x in v.items()}
    fields = {}
    for f in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc", "dst_full_sync_en"):
        if hasattr(v, f):  # a compute kernel config
            fields[f] = _value(getattr(v, f))
    if fields:
        return {"type": type(v).__name__, **fields}
    return str(v)


def _device() -> None:
    import ttnn

    if "fabric_config" not in _DEVICE:
        try:
            _DEVICE["fabric_config"] = str(ttnn.get_fabric_config()).split(".")[-1]
        except Exception:
            pass


def record(op: str, args, kwargs) -> None:
    _device()
    call = {
        "op": op,
        "args": [_value(a) for a in args],
        "kwargs": {k: _value(v) for k, v in sorted(kwargs.items())},
    }
    key = json.dumps(call, sort_keys=True)
    sig = hashlib.sha1(key.encode()).hexdigest()[:10]
    if sig in _CALLS:
        _CALLS[sig]["count"] += 1
    else:
        _CALLS[sig] = {"sig": sig, "count": 1, **call}


def patch() -> None:
    import ttnn.decorators as D

    if getattr(D.FastOperation, "_fork_capture_orig", None) is not None:
        return
    orig = D.FastOperation.__call__

    def call(self, *args, **kwargs):
        name = self.python_fully_qualified_name
        if name.startswith(PREFIX):
            record(name, args, kwargs)
        return orig(self, *args, **kwargs)

    D.FastOperation._fork_capture_orig = orig
    D.FastOperation.__call__ = call


def write(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    calls = sorted(_CALLS.values(), key=lambda c: (c["op"], c["sig"]))
    p.write_text(json.dumps({"device": _DEVICE, "calls": calls}, indent=1) + "\n")
    return p


def pytest_configure(config):
    if os.environ.get(ENV):
        patch()


def pytest_runtest_setup(item):
    params = getattr(item, "callspec", None) and item.callspec.params.get("device_params")
    if os.environ.get(ENV) and isinstance(params, dict) and "device_params" not in _DEVICE:
        _DEVICE["device_params"] = {k: _value(v) for k, v in params.items()}


def pytest_unconfigure(config):
    out = os.environ.get(ENV)
    if out:
        p = write(out)
        print(f"FORK_CAPTURE: {len(_CALLS)} distinct ttnn.bringup call(s) -> {p}")
