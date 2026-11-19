# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact complete recurrent adapter gate with matched evolving-state timing.

Source/CPU preparation is the default. Hardware execution is parent-owned opt-in.
No HF state is injected. Native reductions remain unchanged in the candidate.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
BASE_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"
COMPONENT = Path(__file__).with_name("v6_recurrent_six_stage_compilefix_component.py")
COMPONENT_SHA = "b1b3a9a34ea4a8b3e826e7d8406d462cb2053316fce050e8ae1900fac6fe77eb"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_component():
    assert sha(COMPONENT) == COMPONENT_SHA
    spec = importlib.util.spec_from_file_location("recurrent_six_stage", COMPONENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native_functions(t, debug):
    assert sha(BASE) == BASE_SHA
    tree = ast.parse(BASE.read_text())
    reshape_fn = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reshape"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    functions = [
        copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
        for name in ("_recurrent_sum", "_recurrent_step")
    ]
    hashes = {fn.name: hashlib.sha256(ast.dump(fn).encode()).hexdigest() for fn in functions}
    if debug:
        functions[0].body.insert(0, ast.parse("self.products.append(products)").body[0])
        observed = {"q", "k", "v", "beta", "decayed", "memory", "delta", "update", "new", "output"}

        class Observe(ast.NodeTransformer):
            def visit_Assign(self, node):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in observed:
                    name = node.targets[0].id
                    return [node, ast.parse(f"self.observed[{name!r}] = {name}").body[0]]
                return node

        functions[1] = Observe().visit(functions[1])
    module = ast.fix_missing_locations(ast.Module([reshape_fn, *functions], type_ignores=[]))
    scope = {"ttnn": t}
    exec(compile(module, str(BASE) + ":recurrent", "exec"), scope)
    return scope["_recurrent_sum"], scope["_recurrent_step"], hashes


def compare(a, b):
    import torch

    assert a.shape == b.shape and a.dtype == b.dtype == torch.float32, (a.shape, b.shape, a.dtype, b.dtype)
    unequal = a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)
    finite = torch.isfinite(a) & torch.isfinite(b)
    return dict(
        bits_different=int(unequal.sum()),
        elements=a.numel(),
        first_indices=unequal.nonzero()[:8].tolist(),
        max_finite_abs=float((a[finite] - b[finite]).abs().max()) if bool(finite.any()) else 0,
    )


def mapping_proof():
    import torch

    # Labels identify every logical state word. The formulas below match both
    # literal kernels' original-state gather/scatter and grouped tile addresses.
    h = torch.arange(32).reshape(32, 1, 1).expand(32, 128, 128)
    k = torch.arange(128).reshape(1, 128, 1).expand(32, 128, 128)
    v = torch.arange(128).reshape(1, 1, 128).expand(32, 128, 128)
    page = h * 16 + (k // 32) * 4 + v // 32
    lane = ((k % 32) // 16) * 512 + ((v % 32) // 16) * 256 + (k % 16) * 16 + v % 16
    state_address = page * 1024 + lane
    group_page = k * 4 + v // 32
    group_lane = (h // 16) * 512 + ((v % 32) // 16) * 256 + (h % 16) * 16 + v % 16
    grouped_address = group_page * 1024 + group_lane
    labels = torch.arange(32 * 128 * 128).reshape(32, 128, 128)
    storage = torch.empty_like(labels).flatten()
    storage[state_address.flatten()] = labels.flatten()
    grouped = torch.empty_like(storage)
    grouped[grouped_address.flatten()] = storage[state_address.flatten()]
    restored = torch.empty_like(storage)
    restored[state_address.flatten()] = grouped[grouped_address.flatten()]
    assert torch.equal(storage, restored)
    assert torch.unique(state_address).numel() == torch.unique(grouped_address).numel() == labels.numel()
    for group in range(8):
        for head in range(32):
            for i in range(16):
                feature = group * 16 + i
                expected = (feature // 32) * 1024 + ((feature % 32) // 16) * 256 + (head // 2) * 16 + feature % 16
                reader = (group // 2) * 1024 + (group % 2) * 256 + (head // 2) * 16 + i
                assert reader == expected
    return dict(
        state_words=labels.numel(),
        gather_scatter_inverse=True,
        no_duplicate_state_writes=True,
        grouped_layout_covers_every_word=True,
        gqa_head_mapping="value_head//2",
        qk_address_all_pairs_exact=True,
    )


def prepare(args):
    import torch

    torch.set_num_threads(4)
    component = load_component()
    _, _, hashes = native_functions(None, False)
    fixtures, provenance = [], {}
    generator = torch.Generator().manual_seed(2026111306)

    def random_fixture(name):
        return dict(
            name=name,
            q=(torch.randn(1, 1, 16, 128, generator=generator) * 0.1).bfloat16().float(),
            k=(torch.randn(1, 1, 16, 128, generator=generator) * 0.1).bfloat16().float(),
            v=torch.randn(1, 1, 32, 128, generator=generator).bfloat16(),
            beta=torch.sigmoid(torch.randn(1, 1, 32, generator=generator)).bfloat16().float(),
            decay=torch.exp(-torch.rand(1, 1, 32, generator=generator)),
            state=torch.randn(1, 32, 128, 128, generator=generator) * 0.1,
        )

    for group in args.cases.split(","):
        if group.startswith("actual"):
            positions = (227, 738, 5799) if group == "actual" else (int(group.removeprefix("actual")),)
            assert all(p in (227, 738, 5799) for p in positions), "Only non-transplanted captures in this probe"
            for pos in positions:
                path = E / f"v6_inline_precision_stream_probe.json.pos{pos}.pt"
                capture = torch.load(path, weights_only=False, map_location="cpu")
                values = capture["tt_values"]
                fixtures.append(
                    dict(
                        name=f"actual{pos}",
                        **{
                            name: values[key].contiguous()
                            for name, key in (
                                ("q", "delta_q"),
                                ("k", "delta_k"),
                                ("v", "delta_v"),
                                ("beta", "delta_beta"),
                                ("decay", "delta_decay"),
                                ("state", "delta_state_input"),
                            )
                        },
                    )
                )
                provenance[str(path)] = sha(path)
        elif group in ("random", "zero"):
            fixture = random_fixture(group)
            if group == "zero":
                fixture["state"].zero_()
            fixtures.append(fixture)
        elif group == "boundary":
            fixture = random_fixture(group)
            codes = (
                torch.tensor(
                    [
                        0,
                        0x8000,
                        1,
                        0x8001,
                        0x7F,
                        0x807F,
                        0x80,
                        0x8080,
                        0x3F00,
                        0xBF00,
                        0x3F80,
                        0xBF80,
                        0x3D00,
                        0xBD00,
                        0x600,
                        0x8600,
                    ],
                    dtype=torch.int32,
                )
                .to(torch.int16)
                .view(torch.bfloat16)
            )
            fixture["q"] = codes.repeat(128).reshape(1, 1, 16, 128).float()
            fixture["k"] = codes.roll(5).repeat(128).reshape(1, 1, 16, 128).float()
            fixture["v"] = codes.roll(7).repeat(256).reshape(1, 1, 32, 128)
            words = (
                torch.tensor(
                    [
                        0,
                        0x80000000,
                        1,
                        0x80000001,
                        0x7FFFFF,
                        0x807FFFFF,
                        0x800000,
                        0x80800000,
                        0x3F808000,
                        0xBF808000,
                        0x3F818000,
                        0xBF818000,
                        0x33800000,
                        0xB3800000,
                        0x3F000000,
                        0xBF000000,
                    ],
                    dtype=torch.int64,
                )
                .to(torch.int32)
                .view(torch.float32)
            )
            fixture["state"] = words.repeat(32768).reshape(1, 32, 128, 128)
            fixtures.append(fixture)
        else:
            raise ValueError(group)
    # Ordinary random initial state for changing-input and timing trajectories.
    ordinary = random_fixture("ordinary_initial")
    kernel_names = [name for name in vars(component) if name.startswith("PHASE")]
    report = dict(
        status="cpu_setup_pass",
        source_sha256=sha(__file__),
        component_sha256=COMPONENT_SHA,
        base_sha256=BASE_SHA,
        native_method_ast_sha256=hashes,
        fixture_sources=provenance,
        fixtures=[f["name"] for f in fixtures],
        cases=[],
        trajectory=[],
        timing=[],
        device_executed=args.run_device,
        device_closed=False,
        acceptance_eligible=False,
        oracle_assisted=False,
        source_mapping_proof=mapping_proof(),
        kernel_sha256={name: hashlib.sha256(getattr(component, name).encode()).hexdigest() for name in kernel_names},
        phase1_cb_bytes=45056,
        phase2_cb_bytes=63488,
        phase1_debug_cb_bytes=49152,
        phase2_debug_cb_bytes=83968,
        persistent_component_bytes=0,
        product_bytes=2097152,
        decayed_bytes=2097152,
        native_reduction_stages="16-term left fold, FP32 pack, then8-group left fold, FP32 pack; unchanged fast_reduce_nc",
        timing_contract="Identical ordinary initial state and logical update counts for native/fused warmup/eager/capture/traced groups; compare matching checkpoints. Replay determinism resets to a device backup taken after capture, outside timing.",
    )
    return component, fixtures, ordinary, report


def run(args):
    assert not args.output.exists()
    component, fixtures, ordinary, result = prepare(args)
    args.output.with_suffix(".source.py.snapshot").write_bytes(Path(__file__).read_bytes())
    args.output.with_suffix(".component.py.snapshot").write_bytes(COMPONENT.read_bytes())
    for name in result["kernel_sha256"]:
        args.output.with_suffix(f".{name.lower()}.cpp.snapshot").write_text(getattr(component, name))

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("RECURRENT_SIX_STAGE_CPU_SETUP_PASS", result["source_sha256"])
        return
    for name in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(name) in (None, "", "0"), name
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=64 * 1024 * 1024
        )
        compute = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        narrow = t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value))
        widen = t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value))
        rounds = [narrow, widen]
        candidate = component.RecurrentSixStage(t, mesh, compute)
        # Compile/extract the frozen oracle once, outside all measured calls.
        native_compiled = {flag: native_functions(t, flag) for flag in (False, True)}
        plain_sum, plain_step, _ = native_compiled[False]
        plain_obj = SimpleNamespace(compute=compute, round_widen_chain=rounds)
        plain_obj._recurrent_sum = lambda products: plain_sum(plain_obj, products)

        def upload(x):
            # Preserve tiny/signed BF16 values on the input side of this test.
            tile = t.from_torch(
                x.contiguous(), dtype=t.bfloat16 if x.dtype == torch.bfloat16 else t.float32, layout=t.TILE_LAYOUT
            )
            return t.to_device(tile, mesh, memory_config=t.DRAM_MEMORY_CONFIG)

        def inputs(fixture):
            return {name: upload(fixture[name]) for name in ("q", "k", "v", "beta", "decay")}

        def native(inp, state, output_state, round_state=True, debug=False):
            if not debug:
                core, new = plain_step(
                    plain_obj,
                    inp["q"],
                    inp["k"],
                    inp["v"],
                    None,
                    inp["beta"],
                    state,
                    decay_value=inp["decay"],
                    round_state=round_state,
                    output_state=output_state,
                )
                return core, new, {}
            sum_fn, step_fn, _ = native_compiled[True]
            obj = SimpleNamespace(compute=compute, round_widen_chain=rounds, observed={}, products=[])
            obj._recurrent_sum = lambda products: sum_fn(obj, products)
            if debug:
                rounded = t.unary_chain(state, rounds) if round_state else t.clone(state)
                scaled = t.unary_chain(inp["q"], [t.UnaryWithParam(t.UnaryOpType.MUL_UNARY_SFPU, 128**-0.5)])
            core, new = step_fn(
                obj,
                inp["q"],
                inp["k"],
                inp["v"],
                None,
                inp["beta"],
                state,
                decay_value=inp["decay"],
                round_state=round_state,
                output_state=output_state,
            )
            if debug:
                obj.observed.update(
                    rounded=rounded,
                    scaled_q=scaled,
                    memory_products=obj.products[0],
                    output_products=obj.products[1],
                    subtract=t.subtract(obj.observed["v"], obj.observed["memory"], memory_config=t.L1_MEMORY_CONFIG),
                )
            return core, new, obj.observed

        def fused(inp, state, output_state, round_state=True, debug=False):
            return candidate(
                inp["q"],
                inp["k"],
                inp["v"],
                inp["beta"],
                inp["decay"],
                state,
                output_state=output_state,
                round_state=round_state,
                debug=debug,
            )

        def state_order(x):
            return x.reshape(1, 128, 32, 128).permute(0, 2, 1, 3).contiguous()

        def host_pair(output):
            return (t.to_torch(output[0]).clone(), t.to_torch(output[1]).clone())

        def pair_check(a, b):
            return dict(core=compare(a[0], b[0]), state=compare(a[1], b[1]))

        def pair_exact(checks):
            return all(c["bits_different"] == 0 for c in checks.values())

        differences = {}
        for fixture in fixtures:
            inp = inputs(fixture)
            for round_state in (True, False) if args.both_round_modes else (True,):
                for alias in (True, False):
                    for debug in (True, False):
                        old_state, new_state = upload(fixture["state"]), upload(fixture["state"])
                        old_out = old_state if alias else upload(torch.zeros_like(fixture["state"]))
                        new_out = new_state if alias else upload(torch.zeros_like(fixture["state"]))
                        with device_only():
                            old = native(inp, old_state, old_out, round_state, debug)
                            new = fused(inp, new_state, new_out, round_state, debug)
                        old_pair, new_pair = host_pair(old), host_pair(new)
                        checks = pair_check(old_pair, new_pair)
                        if not alias:
                            checks["input_state_unchanged"] = compare(fixture["state"], t.to_torch(new_state))
                        if debug:
                            oh = {k: t.to_torch(v) for k, v in old[2].items()}
                            nh = {k: t.to_torch(v) for k, v in new[2].items()}
                            for key in ("rounded", "decayed", "memory_products", "output_products"):
                                checks[key] = compare(oh[key], state_order(nh[key]))
                            checks["memory"] = compare(oh["memory"], nh["memory"].reshape(1, 32, 1, 128))
                            for target, key in (("value_wide", "v"), ("subtract", "subtract"), ("delta", "delta")):
                                checks[target] = compare(
                                    oh[key].reshape(1, 1, 32, 128).expand(8, 1, 32, 128).contiguous(), nh[target]
                                )
                            checks["update"] = compare(oh["update"].reshape(1, 32, 128, 128), state_order(nh["update"]))
                            scaled = (
                                oh["scaled_q"]
                                .reshape(16, 128)
                                .repeat_interleave(2, dim=0)
                                .unsqueeze(-1)
                                .expand(32, 128, 128)
                                .unsqueeze(0)
                                .contiguous()
                            )
                            checks["scaled_q"] = compare(scaled, state_order(nh["scaled_q"]))
                        row = dict(
                            name=fixture["name"],
                            round_state=round_state,
                            alias=alias,
                            debug=debug,
                            checks=checks,
                            output_alias_matches=new[1].buffer_address() == new_out.buffer_address(),
                        )
                        result["cases"].append(row)
                        if not pair_exact(checks):
                            key = f'{fixture["name"]}_r{int(round_state)}_a{int(alias)}_d{int(debug)}'
                            differences[key] = dict(
                                native_core=old_pair[0],
                                new_core=new_pair[0],
                                native_state=old_pair[1],
                                new_state=new_pair[1],
                                native_intermediates=oh if debug else {},
                                new_intermediates=nh if debug else {},
                            )
                        save()
                        print(json.dumps(row), flush=True)
                        del old, new, old_state, new_state, old_out, new_out
        if differences:
            path = args.output.with_suffix(".differences.pt")
            torch.save(differences, path)
            result.update(differences=str(path), differences_sha256=sha(path))
        exact = all(pair_exact(r["checks"]) and r["output_alias_matches"] for r in result["cases"])
        result["status"] = "component_exact" if exact else "component_differences"
        save()
        assert exact, "Strict intermediate/core/state bit gate failed"
        # Changing inputs start from an ordinary generated state, not a captured
        # reference state. Both live states advance once for each chosen input.
        old_state, new_state = upload(ordinary["state"]), upload(ordinary["state"])
        trajectory_inputs = [inputs(f) for f in fixtures if f["name"] != "boundary"]
        for step in range(12):
            inp = trajectory_inputs[step % len(trajectory_inputs)]
            with device_only():
                old = native(inp, old_state, old_state)
                new = fused(inp, new_state, new_state)
            checks = pair_check(host_pair(old), host_pair(new))
            result["trajectory"].append(dict(step=step, checks=checks))
            save()
            assert pair_exact(checks), checks
        del old, new, old_state, new_state, trajectory_inputs
        if args.time:
            assert set(args.cases.split(",")) == {"actual", "random", "zero", "boundary"} and args.both_round_modes
            inp = inputs(ordinary)
            references = {}
            for name, function in (("native", native), ("fused", fused)):
                state = upload(ordinary["state"])
                backup = upload(torch.zeros_like(ordinary["state"]))
                checks = {}

                def checkpoint(label, output):
                    value = host_pair(output)
                    if name == "native":
                        references[label] = value
                    check = pair_check(references[label], value)
                    assert pair_exact(check), check
                    checks[label] = check

                for _ in range(5):
                    with device_only():
                        output = function(inp, state, state)
                checkpoint("warm5", output)
                del output
                eager = []
                for sample in range(3):
                    t.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(20):
                        with device_only():
                            output = function(inp, state, state)
                    t.synchronize_device(mesh)
                    eager.append(1000 * (time.perf_counter() - start) / 20)
                    checkpoint(f"eager{sample}", output)
                del output
                trace, capturing = None, False
                try:
                    trace = t.begin_trace_capture(mesh, cq_id=0)
                    capturing = True
                    with device_only():
                        output = function(inp, state, state)
                    t.end_trace_capture(mesh, trace, cq_id=0)
                    capturing = False
                    trace_allocation_tracker.acknowledge_corruptible(output[0])
                    # Backup taken after capture; capture-time state updates need
                    # not be inferred to reproduce the subsequent trajectory.
                    t.copy(state, backup)
                    for _ in range(5):
                        t.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    checkpoint("trace_first", output)
                    samples = []
                    for sample in range(3):
                        t.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(100):
                            t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        t.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / 100)
                        checkpoint(f"trace{sample}", output)
                    final = host_pair(output)
                    t.copy(backup, state)
                    for _ in range(305):
                        t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    t.synchronize_device(mesh)
                    repeat = pair_check(final, host_pair(output))
                    assert pair_exact(repeat), repeat
                    result["timing"].append(
                        dict(
                            variant=name,
                            warmed_ms=eager,
                            warmed_median_ms=statistics.median(eager),
                            traced_ms=samples,
                            traced_median_ms=statistics.median(samples),
                            replays_per_sample=100,
                            matched_native_checkpoints=checks,
                            restarted_305_replay_determinism=repeat,
                        )
                    )
                    save()
                finally:
                    if trace is not None:
                        if capturing:
                            t.end_trace_capture(mesh, trace, cq_id=0)
                        t.synchronize_device(mesh)
                        t.release_trace(mesh, trace)
                    del output, state, backup
            result["status"] = "completed_microbenchmark"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--time", action="store_true")
    p.add_argument("--both-round-modes", action="store_true")
    p.add_argument("--cases", default="actual,random,zero,boundary")
    run(p.parse_args())


if __name__ == "__main__":
    main()
