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

    if isinstance(v, dict) and v.get("type") in ("MeshProgramDescriptor", "PreparedProgram"):
        return v  # (program_summary's, already a record)
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
    if hasattr(
        v, "max_packet_payload_size_bytes"
    ):  # (a FabricRouterConfig in device_params: its value, not its address)
        return {"type": type(v).__name__, "max_packet_payload_size_bytes": _value(v.max_packet_payload_size_bytes)}
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


# F201: generic program runners (plain nanobind functions, not ttnn operations: patch() wraps them by name). Their
# program argument is recorded by program_summary, never by str() (an object address) or by its runtime args / baked
# addresses (an L1 placement)
PROGRAM_RUNNERS = ("prepare_program", "run_prepared")
ADDRESS_MIN = (
    1 << 16
)  # (a literal this large in a kernel's source / defines / compile-time args is taken as an address)
_NUM = None
_PREPARED: dict[int, tuple] = {}  # id(PreparedProgram) -> (it, its program's summary): run_prepared's argument
_PREPARED_CAP = 4096


def _mask(text: str) -> str:
    """Integer literals >= ADDRESS_MIN (decimal or hex, any u / l suffix) -> "A": the addresses a program builder bakes
    into a kernel's source, defines or compile-time args; smaller literals (shapes, counts, coordinates) are kept."""
    import re

    global _NUM
    if _NUM is None:
        _NUM = re.compile(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]*\b")
    return _NUM.sub(lambda m: "A" if int(m.group(1), 0) >= ADDRESS_MIN else m.group(1), text)


def _kernel_summary(k) -> dict:
    src = str(k.kernel_source)
    if "FILE_PATH" in str(k.source_type):
        source = src.split("tt-metal", 1)[-1].split("/ttnn/", 1)[-1] if "/" in src else src
    else:  # (inline source: its text with the addresses masked, hashed)
        source = "inline:" + hashlib.sha1(_mask(src).encode()).hexdigest()[:12]
    cfg = k.config
    cfg_d = {"type": type(cfg).__name__}
    for f in (
        "processor",
        "noc",
        "noc_mode",
        "math_fidelity",
        "fp32_dest_acc_en",
        "dst_full_sync_en",
        "math_approx_mode",
    ):
        if hasattr(cfg, f):
            cfg_d[f] = str(getattr(cfg, f)).split(".")[-1]
    return {
        "source": source,
        "defines": sorted((str(a), _mask(str(b))) for a, b in k.defines),
        "ct_args": [_mask(str(int(v))) for v in k.compile_time_args],
        "named_ct_args": sorted((str(a), _mask(str(int(b)))) for a, b in getattr(k, "named_compile_time_args", [])),
        "cores": str(k.core_ranges),
        "config": cfg_d,
        "rt_cores": len(k.runtime_args),
        "common_rt_args": len(k.common_runtime_args),
    }


def _cb_summary(cb) -> dict:
    return {
        "total_size": int(cb.total_size),
        "cores": str(cb.core_ranges),
        "formats": [
            (int(f.buffer_index), int(f.data_format_as_uint8), int(f.page_size)) for f in cb.format_descriptors
        ],
        "has_buffer": bool(cb.has_buffer()),
        "address_offset": int(cb.address_offset),
    }


def program_summary(md) -> dict:
    """A MeshProgramDescriptor as what makes it that program, the same at any L1 placement: per (mesh range, program)
    its kernels (source - a file's path, or the inline text with addresses masked -, defines, compile-time args with
    addresses masked, core ranges, config, how many cores take runtime args - never their values, which hold the io
    tensors' addresses -), circular buffers (size, cores, formats, buffer-backed) and semaphores. ``digest`` names it;
    ``kernels`` lists each program's kernel sources for a reader."""
    progs = []
    for rng, pd in iter(md):
        progs.append(
            {
                "range": str(rng),
                "kernels": [_kernel_summary(k) for k in pd.kernels],
                "cbs": [_cb_summary(cb) for cb in pd.cbs],
                "semaphores": [(str(s.core_ranges), int(s.initial_value)) for s in pd.semaphores],
            }
        )
    digest = hashlib.sha1(json.dumps(progs, sort_keys=True).encode()).hexdigest()[:12]
    return {
        "type": "MeshProgramDescriptor",
        "programs": len(progs),
        "kernels": sorted({k["source"] for p in progs for k in p["kernels"]}),
        "digest": digest,
    }


def _record_runner(name: str, args, kwargs):
    """prepare_program(desc, io) / run_prepared(prepared, io): the program as program_summary (run_prepared: the
    summary of the descriptor it was prepared from), the io tensors as any tensor."""
    args = list(args)
    prog = args[0] if args else kwargs.get("mesh_program_descriptor", kwargs.get("prepared"))
    io = args[1] if len(args) > 1 else kwargs.get("io_tensors", [])
    if name == "prepare_program":
        summ = program_summary(prog)
    else:
        hit = _PREPARED.get(id(prog))
        summ = hit[1] if hit is not None and hit[0] is prog else {"type": "PreparedProgram", "digest": "unknown"}
    record(PREFIX + name, [summ, list(io)], {})
    return summ


# F179: arguments whose value is a run's identity, not the call's shape (a program-cache key built from buffer
# addresses: DeepSeek-V4.1 flat_routed_expert extra_key changed between two runs, so no case could ever match it)
VOLATILE_KWARGS = frozenset({"extra_key"})


def record(op: str, args, kwargs) -> None:
    _device()
    call = {
        "op": op,
        "args": [_value(a) for a in args],
        "kwargs": {k: _value(v) for k, v in sorted(kwargs.items())},
    }
    full = hashlib.sha1(json.dumps(call, sort_keys=True).encode()).hexdigest()[:10]  # the pre-F179 signature
    stable = dict(call, kwargs={k: v for k, v in call["kwargs"].items() if k not in VOLATILE_KWARGS})
    sig = hashlib.sha1(json.dumps(stable, sort_keys=True).encode()).hexdigest()[:10]
    if sig in _CALLS:
        _CALLS[sig]["count"] += 1
    else:
        _CALLS[sig] = {"sig": sig, "sig_full": full, "count": 1, **call}


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
    patch_runners()


def patch_runners() -> None:
    """F201: wrap ttnn.bringup.prepare_program / run_prepared (callers look them up on ttnn.bringup at each call)."""
    import ttnn

    mod = ttnn.bringup
    for name in PROGRAM_RUNNERS:
        fn = mod.__dict__.get(name) or getattr(mod, name, None)
        if fn is None or getattr(fn, "_fork_capture_orig", None) is not None:
            continue

        def wrapped(*args, _fn=fn, _name=name, **kwargs):
            summ = _record_runner(_name, args, kwargs)
            out = _fn(*args, **kwargs)
            if _name == "prepare_program":
                if len(_PREPARED) >= _PREPARED_CAP:
                    _PREPARED.pop(next(iter(_PREPARED)))
                _PREPARED[id(out)] = (out, summ)
            return out

        wrapped._fork_capture_orig = fn
        setattr(mod, name, wrapped)


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
