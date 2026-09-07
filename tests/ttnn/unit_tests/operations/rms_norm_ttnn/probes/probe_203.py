"""`per_channel_mcast_v2` -- HALF TWO of the round-2 entry condition, proved on the PROGRAM.

check_off_identity.py proves the OFF build compiles the shipped reader SOURCE.  This one
proves it also builds the shipped PROGRAM: same CB list, same semaphore list, same
compile-time args on all three kernels, same runtime args on every core, same defines,
same kernel configs.

It works by recording, not by introspecting nanobind getters: `ttnn.CBDescriptor`,
`ttnn.SemaphoreDescriptor`, `ttnn.KernelDescriptor` and `ttnn.RuntimeArgs` are swapped for
recorders for the duration of one call to each factory, and the two recordings are diffed.

    RMS_CASES=FOCUS,WSHARD,BLOCK,RMW,STREAM python3 check_off_descriptor.py
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(
    os.environ.get(
        "RMS_EXP_DIR",
        "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/"
        "rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_mcast_v2",
    )
)
sys.path.insert(0, str(HERE))

import bench_v2  # noqa: E402
import ttnn  # noqa: E402
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PDM = _load("pd_mcast_v2", HERE / "pd_mcast.py")

_WRAP = (
    "CBDescriptor",
    "CBFormatDescriptor",
    "SemaphoreDescriptor",
    "KernelDescriptor",
    "RuntimeArgs",
    "ReaderConfigDescriptor",
    "WriterConfigDescriptor",
    "DataMovementConfigDescriptor",
)
REAL = {n: getattr(ttnn, n) for n in _WRAP}
_ADDR = __import__("re").compile(r" at 0x[0-9a-f]+")


def val(obj):
    """A repr with the nanobind object ADDRESS scrubbed, plus the recorded kwargs of
    anything this checker wraps -- two descriptors with identical contents must compare
    equal, and two nanobind handles at different addresses must not compare unequal."""
    if isinstance(obj, _Rec):
        return obj.tag
    if isinstance(obj, (list, tuple)):
        return [val(o) for o in obj]
    return _ADDR.sub("", repr(obj))


class _Rec:
    """A recorded constructor call that also builds the real object."""

    def __init__(self, name, real, args, kw):
        self.tag = (name, [val(a) for a in args], sorted((k, val(v)) for k, v in kw.items()))
        args = [_unwrap(a) for a in args]
        kw = {k: _unwrap(v) for k, v in kw.items()}
        self.real = real(*args, **kw)


def _unwrap(v):
    if isinstance(v, _Rec):
        return v.real
    if isinstance(v, list):
        return [_unwrap(o) for o in v]
    return v


class _Row(dict):
    def __setitem__(self, k, v):
        super().__setitem__(k, list(v))


class RecRuntimeArgs:
    def __init__(self):
        self.rows = {}
        self.real = REAL["RuntimeArgs"]()

    def __getitem__(self, x):
        self.rows.setdefault(x, _Row())
        return _Proxy(self, x)

    def snapshot(self):
        return {(x, y): v for x, row in self.rows.items() for y, v in row.items()}


class _Proxy:
    def __init__(self, owner, x):
        self.owner, self.x = owner, x

    def __setitem__(self, y, v):
        self.owner.rows[self.x][y] = v
        self.owner.real[self.x][y] = v


def record(factory, **kw):
    log = {"cbs": [], "sems": [], "kernels": [], "rt": []}

    def wrap(name, sink=None):
        def go(*a, **k):
            rec = _Rec(name, REAL[name], a, k)
            if sink is not None:
                log[sink].append(rec.tag)
            return rec.real

        return go

    def mk_rec(name):
        def go(*a, **k):
            return _Rec(name, REAL[name], a, k)

        return go

    def mk_kernel(**k):
        rt = k.get("runtime_args")
        log["kernels"].append(
            {
                "src": Path(str(k.get("kernel_source"))).name,
                "defines": [tuple(d) for d in (k.get("defines") or [])],
                "ct": list(k.get("compile_time_args") or []),
                "cfg": val(k.get("config")),
                "cores": val(k.get("core_ranges")),
            }
        )
        log["rt"].append(rt.snapshot() if isinstance(rt, RecRuntimeArgs) else None)
        k = dict(k)
        k["runtime_args"] = rt.real if isinstance(rt, RecRuntimeArgs) else rt
        k["config"] = k["config"].real if isinstance(k.get("config"), _Rec) else k.get("config")
        return REAL["KernelDescriptor"](**k)

    ttnn.CBDescriptor = wrap("CBDescriptor", "cbs")
    ttnn.SemaphoreDescriptor = wrap("SemaphoreDescriptor", "sems")
    ttnn.CBFormatDescriptor = mk_rec("CBFormatDescriptor")
    ttnn.ReaderConfigDescriptor = mk_rec("ReaderConfigDescriptor")
    ttnn.WriterConfigDescriptor = mk_rec("WriterConfigDescriptor")
    ttnn.DataMovementConfigDescriptor = mk_rec("DataMovementConfigDescriptor")
    ttnn.KernelDescriptor = mk_kernel
    ttnn.RuntimeArgs = RecRuntimeArgs
    try:
        factory(**kw)
    finally:
        for n, v in REAL.items():
            setattr(ttnn, n, v)
    return log


def diff(a, b, path="", out=None):
    out = [] if out is None else out
    if type(a) is not type(b):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b), key=repr):
            if k not in a:
                out.append(f"{path}[{k!r}]: MISSING in shipped")
            elif k not in b:
                out.append(f"{path}[{k!r}]: MISSING in off")
            else:
                diff(a[k], b[k], f"{path}[{k!r}]", out)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: len {len(a)} != {len(b)}  shipped={a!r} off={b!r}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def main():
    cases = (os.environ.get("RMS_CASES") or "FOCUS,F1024,FGB,STREAM,WSHARD,BLOCK,RMW").split(",")
    device = ttnn.open_device(device_id=0)
    bad = 0
    print(f"STAGE_ZONES shipped={PD.STAGE_ZONES} off={PDM.STAGE_ZONES} env={os.environ.get('RMS_STAGE_ZONES')!r}")
    try:
        for case in cases:
            c = bench_v2.CASES[case]
            # Rebuild the operands WITHOUT running the op: the descriptor is a pure
            # function of the tensors + the config.
            import torch

            shape, ml, mode = c["shape"], c["ml"], c["mode"]
            W = shape[-1]
            lay = ttnn.TILE_LAYOUT
            wlay = ttnn.ROW_MAJOR_LAYOUT if c.get("w_layout") == "rm" else ttnn.TILE_LAYOUT
            if c.get("shard"):
                from eval.sharding import shard_config

                mc = shard_config(c["shard"][0], c["shard"][1], ml, layout=lay, dtype=ttnn.bfloat16, device=device)
            else:
                mc = ttnn.DRAM_MEMORY_CONFIG
            x = ttnn.from_torch(
                torch.zeros(shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=lay,
                device=device,
                memory_config=mc,
            )
            out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), x.dtype, x.layout, device, mc)
            vec = lambda: ttnn.from_torch(  # noqa: E731
                torch.zeros(1, 1, 1, W, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=wlay, device=device
            )
            g = vec() if ("gamma" in mode and mode != "no_gamma") else None
            b = vec() if "bias" in mode else None
            r = (
                ttnn.from_torch(
                    torch.zeros(shape, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=lay,
                    device=device,
                    memory_config=mc,
                )
                if "residual" in mode
                else None
            )
            ckc = bench_v2._cfg(c.get("fp32", False))
            # Resolve the plan EXACTLY the way the op does, so the descriptor under
            # test is the one the measured run builds.
            resolved_pc = PD.resolve_program_config(
                x, program_config=None, memory_config=mc, dest_limit=PD.dest_tile_limit(ckc)
            )
            kw = dict(
                input_tensor=x,
                output_tensor=out,
                weight=g,
                bias=b,
                residual=r,
                epsilon=1e-12,
                compute_kernel_config=ckc,
                program_config=resolved_pc,
            )
            PDM.PC_MCAST_MODE = None
            a = record(PD.create_program_descriptor, **kw)
            bb = record(PDM.create_program_descriptor, **kw)
            d = diff(a, bb)
            if d:
                bad += 1
                print(f"OFF-DIVERGENT  {case}")
                for line in d[:20]:
                    print("   " + line)
            else:
                print(
                    f"OFF-IDENTICAL  {case:8s} cbs={len(a['cbs'])} sems={len(a['sems'])} "
                    f"kernels={len(a['kernels'])} reader_ct={len(a['kernels'][0]['ct'])} "
                    f"cores={len(a['rt'][0])} defines={a['kernels'][0]['defines']}"
                )
            for t in (x, out, g, b, r):
                if t is not None:
                    ttnn.deallocate(t)
    finally:
        ttnn.close_device(device)
    print("ENTRY-CONDITION(descriptor): " + ("PASS" if bad == 0 else f"FAIL ({bad} case(s))"))


main()
