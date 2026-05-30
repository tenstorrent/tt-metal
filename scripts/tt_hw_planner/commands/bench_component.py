"""`tt_hw_planner bench-component <model> [<component>]` — Stage 4
empirical CPU vs device benchmark.

For each graduated stub (or a single one if specified), times the
component on:

  * CPU       — torch reference, the model's submodule .forward()
  * Device    — TT port from the stub's build(device, torch_module),
                INCLUDING host↔device transfer cost (ttnn.from_torch
                for inputs, ttnn.to_torch for outputs)

Compares wall-clock; verdict per the rule

    HOT (device wins) iff  device_time + 2 × transfer_cost  <  cpu_time

Persists the measurement under hot_cold.json as a Signal-4 record so
the categorizer can use this evidence as gold-standard. Annotates the
existing entry's `bench` key (per workload-mode, like the other
evidence signals).

Gold-standard for "does putting this component on device actually
add value" — catches cases where the kernel exists but is slower
than CPU for the call's shape/dtype/batch, which the heuristic
density + affinity signals only approximate.

Prerequisite: component must have a GRADUATED stub on disk (i.e.
.py.last_good_native snapshot present). Components without a working
device port can't be benched against — for those, only Signals 1-3
contribute.
"""

from __future__ import annotations

import importlib
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BenchResult:
    """One component's CPU vs device measurement."""

    component: str
    cpu_mean_ms: float = 0.0
    cpu_p50_ms: float = 0.0
    cpu_std_ms: float = 0.0
    device_mean_ms: float = 0.0
    device_p50_ms: float = 0.0
    device_std_ms: float = 0.0
    transfer_in_ms: float = 0.0
    transfer_out_ms: float = 0.0
    n_iters: int = 0
    verdict: str = "UNKNOWN"  # "DEVICE_WINS" | "CPU_WINS" | "BREAKEVEN" | "ERROR"
    speedup: float = 0.0  # cpu_mean / (device_mean + 2*transfer)
    error: str = ""
    workload_mode: str = "default"


def cmd_bench_component(args) -> int:
    """Run Stage 4 bench. Returns 0 on success, non-zero on usage errors.

    A bench failure for a single component (e.g. transfer issues) is
    logged and treated as ERROR for that component; other components
    continue."""
    from ..bringup_loop import _safe_id, find_demo_dir

    model_id = args.model_id
    component_arg = (getattr(args, "component", None) or "").strip() or None
    n_iters = int(getattr(args, "n_iters", 5) or 5)
    n_warmup = int(getattr(args, "n_warmup", 2) or 2)
    workload_mode = (getattr(args, "workload_mode", None) or "default").strip().lower() or "default"

    demo_dir = find_demo_dir(model_id)
    if demo_dir is None:
        print(f"error: no scaffolded demo dir for `{model_id}`. Run `up` first.", file=sys.stderr)
        return 2

    status_path = demo_dir / "bringup_status.json"
    if not status_path.is_file():
        print(f"error: no bringup_status.json at {status_path}", file=sys.stderr)
        return 2

    try:
        status = json.loads(status_path.read_text())
    except Exception as exc:
        print(f"error: failed to read bringup_status.json: {exc}", file=sys.stderr)
        return 2

    # Collect candidates: graduated stubs (have .py.last_good_native).
    candidates = _collect_graduated_for_bench(demo_dir, status, component_arg)
    if not candidates:
        if component_arg:
            print(
                f"error: component `{component_arg}` has no graduated stub "
                f"(no .py.last_good_native snapshot). Bench requires a working "
                f"device port.",
                file=sys.stderr,
            )
        else:
            print(
                f"error: no graduated components found for `{model_id}`. "
                f"Bench requires components with .py.last_good_native snapshots.",
                file=sys.stderr,
            )
        return 2

    print(f"bench-component: loading HF model `{model_id}`…")
    try:
        import transformers

        hf_model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
        hf_model.eval()
    except Exception as exc:
        print(f"error: HF load failed: {exc}", file=sys.stderr)
        return 2

    # Lazy ttnn import: bench MUST have a TT device available.
    try:
        import ttnn  # noqa: F401
    except ImportError:
        print(
            "error: ttnn is not importable. bench-component requires a TT environment.",
            file=sys.stderr,
        )
        return 2

    device = _open_device_for_bench()
    if device is None:
        print(
            "error: could not open a TT device. Bench requires hardware.",
            file=sys.stderr,
        )
        return 2

    print(
        f"bench-component: {len(candidates)} component(s) to bench "
        f"(n_iters={n_iters}, n_warmup={n_warmup}, mode={workload_mode})"
    )
    print()

    results: List[BenchResult] = []
    for comp in candidates:
        comp_name = comp["name"]
        print(f"  benching {comp_name}…", flush=True)
        try:
            result = _bench_one(
                comp_name=comp_name,
                comp_entry=comp,
                hf_model=hf_model,
                demo_dir=demo_dir,
                device=device,
                n_iters=n_iters,
                n_warmup=n_warmup,
                workload_mode=workload_mode,
            )
        except Exception as exc:
            result = BenchResult(
                component=comp_name,
                verdict="ERROR",
                error=f"{type(exc).__name__}: {exc}",
                workload_mode=workload_mode,
            )
        results.append(result)

    try:
        _close_device(device)
    except Exception:
        pass

    _print_bench_summary(results)
    _persist_bench_results(model_id, results, workload_mode=workload_mode)
    return 0


def _collect_graduated_for_bench(
    demo_dir: Path,
    status: dict,
    only_component: Optional[str],
) -> List[Dict[str, Any]]:
    """Walk bringup_status.json and return only NEW components whose
    stub has a `.py.last_good_native` snapshot (verified graduated)
    AND has captured inputs (args.pt, kwargs.pt). Optionally filter
    to a single named component."""
    from ..bringup_loop import _safe_id

    out: List[Dict[str, Any]] = []
    for c in status.get("components", []) or []:
        name = c.get("name") or ""
        if not name:
            continue
        if only_component is not None and name != only_component:
            continue
        if c.get("status") != "NEW":
            continue
        safe = _safe_id(name)
        snapshot = demo_dir / "_stubs" / f"{safe}.py.last_good_native"
        if not snapshot.is_file():
            continue
        cap_dir = demo_dir / "_captured" / safe
        if not all((cap_dir / fname).is_file() for fname in ("args.pt", "kwargs.pt", "manifest.json")):
            continue
        # Pick up submodule_path from status OR capture manifest.
        sub_path = (c.get("submodule_path") or "").strip()
        if not sub_path:
            try:
                sub_path = (json.loads((cap_dir / "manifest.json").read_text()).get("submodule_path") or "").strip()
            except Exception:
                sub_path = ""
        if not sub_path:
            continue
        out.append({"name": name, "safe": safe, "submodule_path": sub_path})
    return out


def _open_device_for_bench():
    """Open a TT device for the bench duration. Returns the device
    handle, or None if open failed.

    Uses the same path as the demo template's pytest fixture
    (l1_small_size=24576) so the bench measurements reflect the
    same allocator config the demo will use."""
    try:
        import ttnn

        device = ttnn.open_device(device_id=0, l1_small_size=24576)
        return device
    except Exception as exc:
        print(f"  [bench] failed to open device: {exc}", file=sys.stderr)
        return None


def _close_device(device) -> None:
    try:
        import ttnn

        ttnn.close_device(device)
    except Exception:
        pass


def _resolve_torch_submodule(model, dotted: str):
    """Walk dotted/indexed path on the model."""
    cur = model
    for tok in dotted.replace("[", ".").replace("]", "").split("."):
        if not tok:
            continue
        if tok.isdigit():
            cur = cur[int(tok)]
        else:
            cur = getattr(cur, tok)
    return cur


def _bench_one(
    *,
    comp_name: str,
    comp_entry: Dict[str, Any],
    hf_model,
    demo_dir: Path,
    device,
    n_iters: int,
    n_warmup: int,
    workload_mode: str,
) -> BenchResult:
    """Bench a single component. Returns BenchResult with verdict."""
    import torch
    import ttnn

    safe = comp_entry["safe"]
    sub_path = comp_entry["submodule_path"]

    # Resolve the torch submodule on the HF model.
    torch_module = _resolve_torch_submodule(hf_model, sub_path)
    torch_module.eval()

    # Load captured inputs.
    cap_dir = demo_dir / "_captured" / safe
    captured_args = torch.load(cap_dir / "args.pt", map_location="cpu", weights_only=False) or ()
    captured_kwargs = torch.load(cap_dir / "kwargs.pt", map_location="cpu", weights_only=False) or {}

    # Build the TT port via the graduated stub.
    from ..bringup_loop import _stub_import_path
    from ..discovery import BRINGUP_ROOT

    repo_root = BRINGUP_ROOT()
    stub_module_path = _stub_import_path(demo_dir, safe, repo_root)
    stub_mod = importlib.import_module(stub_module_path)
    if not hasattr(stub_mod, "build"):
        return BenchResult(
            component=comp_name,
            verdict="ERROR",
            error=f"stub `{stub_module_path}` has no build(device, torch_module)",
            workload_mode=workload_mode,
        )
    tt_port = stub_mod.build(device, torch_module)

    # ---------- CPU timing ----------
    cpu_times: List[float] = []
    for _ in range(n_warmup + n_iters):
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            try:
                _cpu_invoke(torch_module, captured_args, captured_kwargs)
            except Exception as exc:
                return BenchResult(
                    component=comp_name,
                    verdict="ERROR",
                    error=f"CPU forward failed: {type(exc).__name__}: {exc}",
                    workload_mode=workload_mode,
                )
        cpu_times.append((time.perf_counter_ns() - t0) / 1e6)
    cpu_times = cpu_times[n_warmup:]  # drop warmup

    # ---------- Transfer cost timing (in only — measured separately) ----------
    transfer_in_times: List[float] = []
    for _ in range(n_warmup + n_iters):
        t0 = time.perf_counter_ns()
        try:
            _convert_args_to_tt(captured_args, captured_kwargs, device)
        except Exception as exc:
            return BenchResult(
                component=comp_name,
                verdict="ERROR",
                error=f"input transfer failed: {type(exc).__name__}: {exc}",
                workload_mode=workload_mode,
            )
        transfer_in_times.append((time.perf_counter_ns() - t0) / 1e6)
    transfer_in_times = transfer_in_times[n_warmup:]

    # ---------- Device timing (no transfer in measurement) ----------
    device_times: List[float] = []
    transfer_out_times: List[float] = []
    for _ in range(n_warmup + n_iters):
        tt_args, tt_kwargs = _convert_args_to_tt(captured_args, captured_kwargs, device)
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            try:
                out = tt_port(*tt_args, **tt_kwargs)
            except Exception as exc:
                return BenchResult(
                    component=comp_name,
                    verdict="ERROR",
                    error=f"device forward failed: {type(exc).__name__}: {exc}",
                    workload_mode=workload_mode,
                )
        device_times.append((time.perf_counter_ns() - t0) / 1e6)

        # Transfer-out timing
        t0_out = time.perf_counter_ns()
        try:
            _convert_output_to_torch(out)
        except Exception:
            pass
        transfer_out_times.append((time.perf_counter_ns() - t0_out) / 1e6)
    device_times = device_times[n_warmup:]
    transfer_out_times = transfer_out_times[n_warmup:]

    cpu_mean = statistics.mean(cpu_times)
    device_mean = statistics.mean(device_times)
    xfer_in = statistics.mean(transfer_in_times)
    xfer_out = statistics.mean(transfer_out_times)
    total_device = device_mean + xfer_in + xfer_out
    speedup = cpu_mean / total_device if total_device > 0 else 0.0

    if speedup > 1.10:
        verdict = "DEVICE_WINS"
    elif speedup < 0.90:
        verdict = "CPU_WINS"
    else:
        verdict = "BREAKEVEN"

    return BenchResult(
        component=comp_name,
        cpu_mean_ms=cpu_mean,
        cpu_p50_ms=statistics.median(cpu_times),
        cpu_std_ms=statistics.stdev(cpu_times) if len(cpu_times) > 1 else 0.0,
        device_mean_ms=device_mean,
        device_p50_ms=statistics.median(device_times),
        device_std_ms=statistics.stdev(device_times) if len(device_times) > 1 else 0.0,
        transfer_in_ms=xfer_in,
        transfer_out_ms=xfer_out,
        n_iters=n_iters,
        verdict=verdict,
        speedup=speedup,
        workload_mode=workload_mode,
    )


def _cpu_invoke(module, captured_args, captured_kwargs):
    """Invoke a torch module with captured I/O. Strips kwargs the
    forward doesn't accept (mirrors the demo template's _call_tt_module
    convention so CPU and device timing exercise the same arg subset)."""
    import inspect

    sig = inspect.signature(module.forward)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
    accepted = {p.name for p in params if p.kind != p.VAR_POSITIONAL}
    if has_var_kwargs:
        return module(*captured_args, **captured_kwargs)
    filtered = {k: v for k, v in captured_kwargs.items() if k in accepted}
    return module(*captured_args, **filtered)


def _convert_args_to_tt(captured_args, captured_kwargs, device):
    """Convert captured torch tensors to ttnn. Mirrors the demo
    template's _to_tt convention."""
    import torch
    import ttnn

    def _to_tt(x):
        if isinstance(x, torch.Tensor):
            if x.is_floating_point():
                x = x.to(torch.bfloat16)
                return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
        return x

    tt_args = tuple(_to_tt(a) for a in captured_args)
    tt_kwargs = {k: _to_tt(v) for k, v in captured_kwargs.items()}
    return tt_args, tt_kwargs


def _convert_output_to_torch(out):
    """Convert device output back to torch. Used to measure round-trip
    transfer cost end-to-end."""
    import torch
    import ttnn

    if isinstance(out, ttnn.Tensor):
        return ttnn.to_torch(ttnn.from_device(out))
    if isinstance(out, tuple) and out:
        first = out[0]
        if isinstance(first, ttnn.Tensor):
            return ttnn.to_torch(ttnn.from_device(first))
    return out


def _print_bench_summary(results: List[BenchResult]) -> None:
    print()
    print(
        f"  {'component':<35} {'verdict':<14} {'cpu_ms':>8} {'dev_ms':>8} "
        f"{'xfer_in':>8} {'xfer_out':>8} {'speedup':>7}"
    )
    print(f"  {'-'*35} {'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for r in results:
        if r.verdict == "ERROR":
            print(f"  {r.component:<35} ERROR          {r.error[:60]}")
            continue
        print(
            f"  {r.component:<35} {r.verdict:<14} {r.cpu_mean_ms:>8.2f} "
            f"{r.device_mean_ms:>8.2f} {r.transfer_in_ms:>8.2f} "
            f"{r.transfer_out_ms:>8.2f} {r.speedup:>7.2f}x"
        )
    print()
    device_wins = sum(1 for r in results if r.verdict == "DEVICE_WINS")
    cpu_wins = sum(1 for r in results if r.verdict == "CPU_WINS")
    breakeven = sum(1 for r in results if r.verdict == "BREAKEVEN")
    errors = sum(1 for r in results if r.verdict == "ERROR")
    print(f"  DEVICE_WINS: {device_wins}, CPU_WINS: {cpu_wins}, " f"BREAKEVEN: {breakeven}, ERROR: {errors}")


def _persist_bench_results(
    model_id: str,
    results: List[BenchResult],
    *,
    workload_mode: str,
) -> None:
    """Persist bench results under the per-component multi-mode schema
    in hot_cold.json. Each component's ``modes[<workload_mode>]`` gets
    a ``bench`` key with the BenchResult. The top-level ``kind`` is
    updated if the bench verdict strongly disagrees with prior signals:

      bench=DEVICE_WINS  → strengthens HOT
      bench=CPU_WINS     → upgrades verdict to COLD (kernel exists but
                            CPU is faster — no point on device)
      bench=BREAKEVEN    → neutral
      bench=ERROR        → no change
    """
    from ..overlay_manager import _load_hot_cold_raw, persist_hot_cold

    raw = _load_hot_cold_raw(model_id)
    if not isinstance(raw, dict):
        raw = {}

    for r in results:
        existing = raw.get(r.component)
        if existing is None:
            existing = {}
        elif not isinstance(existing, dict):
            # Legacy string-only entry → promote
            existing = {"kind": str(existing)}
        # Promote single-mode to multi-mode if needed
        if "modes" not in existing:
            old_mode = existing.get("workload_mode") or "default"
            existing = {"kind": existing.get("kind", "UNKNOWN"), "modes": {old_mode: existing}}

        mode_entry = existing["modes"].setdefault(workload_mode, {"workload_mode": workload_mode})
        mode_entry["bench"] = asdict(r)
        mode_entry["bench_verdict"] = r.verdict

        # Update top-level kind based on bench verdict + prior verdicts.
        if r.verdict == "CPU_WINS":
            existing["kind"] = "COLD"
            existing.setdefault("evidence", [])
            existing["evidence"].append(
                f"[{workload_mode}] bench: CPU_WINS (speedup {r.speedup:.2f}x; " f"transfer cost dominates)"
            )
        elif r.verdict == "DEVICE_WINS":
            if existing.get("kind") not in ("HOT",):
                existing["kind"] = "HOT"
            existing.setdefault("evidence", [])
            existing["evidence"].append(f"[{workload_mode}] bench: DEVICE_WINS (speedup {r.speedup:.2f}x)")
        # BREAKEVEN / ERROR don't change the top-level kind.

        raw[r.component] = existing

    persist_hot_cold(model_id, raw)
