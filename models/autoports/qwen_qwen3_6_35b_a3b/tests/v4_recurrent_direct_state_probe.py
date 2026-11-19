# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic complete recurrence: optional direct stable-state writer versus final copy.

Saved real operands; bounded eager/trace continuation, reset and independent-state checks.
Only the serialized hardware owner runs the device branch.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import runpy
import statistics
import time
import traceback
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder/endurance"
KEYS = ("delta_q", "delta_k", "delta_v", "delta_g", "delta_beta")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def boundary(path):
    tree = ast.parse(path.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_linear_attention")
    branch = next(
        n
        for n in method.body
        if isinstance(n, ast.If)
        and any(
            isinstance(c, ast.Call) and ast.unparse(c.func) == "self._recurrent_step"
            for x in n.body
            for c in ast.walk(x)
        )
    )
    store = (
        next(
            n
            for n in method.body
            if not isinstance(n, ast.If)
            and isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == "ttnn.copy"
            and ast.unparse(n.value.args[-1]) == "state['recurrent']"
        )
        if 'output_state=state["recurrent"]' not in path.read_text()
        else next(
            n
            for n in method.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "not decode"
            and any(isinstance(c, ast.Call) and ast.unparse(c.func) == "ttnn.copy" for c in ast.walk(n))
        )
    )
    fn = ast.parse(
        "def complete(self,q,k,v,g,beta,state,decay_value,cache_boundary):\n    decode=True\n    return o,new\n"
    ).body[0]
    fn.body = fn.body[:1] + copy.deepcopy(branch.body) + [copy.deepcopy(store)] + fn.body[1:]
    return ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))


def run(args):
    torch.set_num_threads(2)
    assert not args.output.exists(), f"Preserve evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    base = DOC.parent / "selected_fused_decoder_v4.py.snapshot"
    candidate = DOC / "v4_recurrent_direct_state.py.snapshot"
    prov = json.loads(candidate.with_suffix(".provenance.json").read_text())
    assert sha(base) == prov["base_sha256"] and sha(candidate) == prov["candidate_sha256"]
    definitions = [boundary(p) for p in (base, candidate)]
    captures = []
    fixtures = []
    for pos in args.positions:
        p = Path(f"{args.capture_base}.pos{pos}.pt")
        values = torch.load(p, map_location="cpu", weights_only=False)["tt_values"]
        inputs = [values[k].contiguous() for k in KEYS]
        initial = values["delta_state_input"].float().contiguous()
        assert initial.shape == (1, 32, 128, 128) and all(torch.isfinite(v).all() for v in inputs + [initial])
        fixtures.append((inputs, initial))
        captures.append(
            dict(position=pos, path=str(p), sha256=sha(p), shapes={k: list(v.shape) for k, v in zip(KEYS, inputs)})
        )
    result = dict(
        status="cpu_setup_pass",
        diagnostic_only=True,
        acceptance_eligible=False,
        probe_sha256=sha(__file__),
        base_sha256=sha(base),
        candidate_sha256=sha(candidate),
        captures=captures,
        extracted_complete_boundary={p.name: ast.unparse(d) for p, d in zip((base, candidate), definitions)},
        checks=[],
        timing=[],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def check(name, a, b):
        c = compare(a, b)
        result["checks"].append(dict(name=name, **c))
        save()
        assert c["exact"], name

    # Structural proof: same arithmetic method except output target and explicit original L1 Q-product residency.
    trees = [ast.parse(p.read_text()) for p in (base, candidate)]
    methods = [{n.name: n for n in ast.walk(t) if isinstance(n, ast.FunctionDef)} for t in trees]
    old, new = copy.deepcopy(methods[0]["_recurrent_step"]), copy.deepcopy(methods[1]["_recurrent_step"])
    new.args.kwonlyargs.pop()
    new.args.kw_defaults.pop()
    for n in ast.walk(new):
        if isinstance(n, ast.Call):
            n.keywords = [
                k
                for k in n.keywords
                if k.arg != "output_tensor"
                and not (
                    k.arg == "memory_config"
                    and any(k2.arg == "input_tensor_b_activations" for k2 in n.keywords)
                    and ast.unparse(n.func) == "ttnn.multiply"
                    and "MUL_UNARY_SFPU" in ast.unparse(n)
                )
            ]
    assert ast.dump(old) == ast.dump(new)
    assert ast.dump(methods[0]["_recurrent_sum"]) == ast.dump(methods[1]["_recurrent_sum"])
    result["arithmetic_ast_unchanged_except_writer_and_q_product_placement"] = True
    if args.oracle_only:
        save()
        print("DIRECT_STATE_CPU_SETUP_PASS")
        return
    for key in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert os.environ.get(key) in (None, "", "0"), f"Clean timing requires {key} disabled"
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(args.output)
    mesh = None
    try:
        params = dict(physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768)
        result["device_open_parameters"] = params
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **params)
        modules = []
        objects = []
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        for path, definition in zip((base, candidate), definitions):
            m = runpy.run_path(str(path))
            exec(compile(definition, str(path) + ":complete", "exec"), m)
            modules.append(m)
            o = m["FusedDecoder"]()
            o.compute = compute
            o.round_widen_chain = [
                ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value)),
                ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value)),
            ]
            objects.append(o)

        def upload(v):
            return ttnn.from_torch(
                v,
                device=mesh,
                dtype=ttnn.bfloat16 if v.dtype == torch.bfloat16 else ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        device_inputs = [[upload(v) for v in inputs] for inputs, _ in fixtures]
        decays = [upload(inputs[3].float().exp()) for inputs, _ in fixtures]
        initial = upload(fixtures[0][1])
        states = [{"recurrent": ttnn.clone(initial)} for _ in range(2)]
        addresses = [s["recurrent"].buffer_address() for s in states]
        identities = [id(s["recurrent"]) for s in states]

        def stable():
            for j, s in enumerate(states):
                assert id(s["recurrent"]) == identities[j] and s["recurrent"].buffer_address() == addresses[j]
                assert (
                    s["recurrent"].memory_config() == ttnn.DRAM_MEMORY_CONFIG and s["recurrent"].dtype == ttnn.float32
                )

        def reset():
            for s in states:
                ttnn.copy(initial, s["recurrent"])

        def step(j, index, rounded=True, use_decay=True, state=None):
            destination = states[j] if state is None else state
            with device_only():
                out, new = modules[j]["complete"](
                    objects[j], *device_inputs[index], destination, decays[index] if use_decay else None, rounded
                )
            if j == 1:
                assert new.buffer_address() == destination["recurrent"].buffer_address()
            return out

        # Complete single-step checks include the final persistent state store and Q contraction.
        for index in range(len(fixtures)):
            for rounded in (False, True):
                for use_decay in (False, True):
                    reset()
                    outs = [step(j, index, rounded, use_decay) for j in range(2)]
                    label = f"single{index}_round{rounded}_decay{use_decay}"
                    check(label + "_core", ttnn.to_torch(outs[0]), ttnn.to_torch(outs[1]))
                    check(
                        label + "_state", ttnn.to_torch(states[0]["recurrent"]), ttnn.to_torch(states[1]["recurrent"])
                    )
                    stable()
                    del outs
        flags = [True, False, False, True] * 3
        # Boundary/no-boundary suffix, save after four steps, then clone/copy restore and repeat it.
        reset()
        prefix_saved = None
        suffix = []
        for i, rounded in enumerate(flags):
            outs = [step(j, i % len(fixtures), rounded) for j in range(2)]
            host = [ttnn.to_torch(x) for x in outs]
            check(f"chain{i}_core", *host)
            check(f"chain{i}_state", ttnn.to_torch(states[0]["recurrent"]), ttnn.to_torch(states[1]["recurrent"]))
            stable()
            if i == 3:
                prefix_saved = [ttnn.clone(s["recurrent"]) for s in states]
            if i >= 4:
                suffix.append(host)
            del outs
        suffix_state = [ttnn.to_torch(s["recurrent"]) for s in states]
        for j in range(2):
            ttnn.copy(prefix_saved[j], states[j]["recurrent"])
        for i in range(4, len(flags)):
            for j in range(2):
                out = step(j, i % len(fixtures), flags[i])
                check(f"restored_suffix{i}_{j}", suffix[i - 4][j], ttnn.to_torch(out))
                del out
        for j in range(2):
            check(f"restored_final{j}", suffix_state[j], ttnn.to_torch(states[j]["recurrent"]))
        stable()
        # Program-cache hits targeting another state must not overwrite the first request.
        independent = [{"recurrent": ttnn.clone(initial)} for _ in range(2)]
        held = [ttnn.to_torch(s["recurrent"]) for s in states]
        outs = [step(j, 1 % len(fixtures), state=independent[j]) for j in range(2)]
        check("independent_core", ttnn.to_torch(outs[0]), ttnn.to_torch(outs[1]))
        check(
            "independent_state", ttnn.to_torch(independent[0]["recurrent"]), ttnn.to_torch(independent[1]["recurrent"])
        )
        for j in range(2):
            check(f"other_request_unchanged{j}", held[j], ttnn.to_torch(states[j]["recurrent"]))
        del outs, independent, prefix_saved
        # Timed trace includes full recurrence/Q contraction and copy or direct store.
        # Identical replay counts preserve comparable final states; resets stay outside measured calls.
        timing_ends = []
        for j, label in enumerate(("baseline", "candidate")):
            trace = None
            capture_open = False
            out = None
            try:
                ttnn.copy(initial, states[j]["recurrent"])
                for _ in range(args.warmup):
                    out = step(j, 0)
                    ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                capture_open = True
                out = step(j, 0)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                capture_open = False
                ttnn.copy(initial, states[j]["recurrent"])
                for _ in range(args.warmup):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                first = (ttnn.to_torch(out), ttnn.to_torch(states[j]["recurrent"]))
                samples = []
                for _ in range(args.samples):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.replays):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append(1000 * (time.perf_counter() - start) / args.replays)
                end = (ttnn.to_torch(out), ttnn.to_torch(states[j]["recurrent"]))
                timing_ends.append((first, end))
                result["timing"].append(
                    dict(
                        name=label,
                        median_ms=statistics.median(samples),
                        samples_ms=samples,
                        scope="Complete actual decode recurrence including Q contraction and persistent copy/direct store; reset/readback outside timing",
                    )
                )
                # Reset and replay full timed iteration count; verify state address and repeatability.
                ttnn.copy(initial, states[j]["recurrent"])
                for _ in range(args.warmup + args.samples * args.replays):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                check(label + "_trace_reset_core", end[0], ttnn.to_torch(out))
                check(label + "_trace_reset_state", end[1], ttnn.to_torch(states[j]["recurrent"]))
                stable()
            finally:
                if trace is not None:
                    if capture_open:
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    ttnn.release_trace(mesh, trace)
                out = None
                save()
        for stage in range(2):
            for value in range(2):
                check(f"trace_cross_{stage}_{value}", timing_ends[0][stage][value], timing_ends[1][stage][value])
        # Captured mixed-boundary continuation chain, replayed twice from the same initial buffers.
        captured = []
        for j in range(2):
            trace = None
            capture_open = False
            try:
                reset()
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                capture_open = True
                outs = [step(j, i % len(fixtures), flag) for i, flag in enumerate(flags)]
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                capture_open = False
                snapshots = []
                for _ in range(2):
                    ttnn.copy(initial, states[j]["recurrent"])
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    snapshots.append(([ttnn.to_torch(o) for o in outs], ttnn.to_torch(states[j]["recurrent"])))
                for i, (a, b) in enumerate(zip(snapshots[0][0], snapshots[1][0])):
                    check(f"chain_trace_reset{j}_{i}", a, b)
                check(f"chain_trace_final_reset{j}", snapshots[0][1], snapshots[1][1])
                captured.append(snapshots[0])
                stable()
            finally:
                if trace is not None:
                    if capture_open:
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    ttnn.release_trace(mesh, trace)
        for i, (a, b) in enumerate(zip(captured[0][0], captured[1][0])):
            check(f"chain_trace_cross{i}", a, b)
        check("chain_trace_cross_final_state", captured[0][1], captured[1][1])
        stable()
        result["status"] = "exact"
        result["persistent_addresses"] = addresses
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
            except Exception:
                result.update(status="failed", close_error=traceback.format_exc())
        save()
    print(result["status"], args.output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--capture-base", type=Path, default=DOC.parent / "endurance_precise_probe.json")
    p.add_argument("--positions", type=int, nargs="+", default=[128, 130, 7703])
    p.add_argument("--output", type=Path, default=DOC / "v4_recurrent_direct_state_probe.json")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=15)
    p.add_argument("--replays", type=int, default=20)
    a = p.parse_args()
    assert min(a.warmup, a.samples, a.replays) > 0 and a.positions
    run(a)
