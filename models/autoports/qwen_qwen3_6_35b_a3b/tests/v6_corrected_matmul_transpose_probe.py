# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Corrected matmul transpose-address reader; frozen arithmetic, opt-in hardware."""

import argparse
import ast
import hashlib
import itertools
import json
import math
import os
import statistics
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
SNAPSHOT = E / "v6_transport_conv_norm_terminal_scan_integration.py.snapshot"
SNAPSHOT_SHA = "803685e105d7ef1d46a6f9a8ceafb1758d9b1131e1c8c35aed1ab05cbf1475a5"
CPU_FIXTURE = E / "v6_chunk64_corrected_fma_output_device.json.cpu_matrices.pt"
CPU_SHA = "28c3d2e0aa18af8a17ee251b49bb2e9ea428e18da680ffa0065b12d6217e2670"
DEVICE_FIXTURE = E / "v6_chunk64_corrected_fma_output_device.json.device_matrices.pt"
DEVICE_SHA = "19b2c810db56cd0b37d361e0a2a7e65be45042d8c7205d04d36303993f5536fc"
NAMES = {
    "FUSED_CORRECTED_MM_READER",
    "FUSED_CORRECTED_MM_COMPUTE",
    "FUSED_CORRECTED_MM_WRITER",
    "fused_corrected_mm_shape_plan",
    "fused_corrected_mm_schedule",
    "FusedCorrectedMatmul",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def source_bundle():
    """Extract only six pinned static definitions; no model setup/import side effects."""
    assert sha(SNAPSHOT) == SNAPSHOT_SHA
    tree = ast.parse(SNAPSHOT.read_text())
    nodes = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in NAMES:
            nodes.append(node)
    assert len(nodes) == len(NAMES)
    baseline_source = ast.unparse(ast.Module(nodes, [])) + "\n"
    baseline = {"math": math}
    exec(compile(baseline_source, "<pinned-803-corrected-matmul>", "exec"), baseline)
    reader = baseline["FUSED_CORRECTED_MM_READER"]
    reader_edits = [
        (
            "constexpr uint32_t b_accessor_offset=get_compile_time_arg_val(0);",
            "constexpr uint32_t b_accessor_offset=get_compile_time_arg_val(0);\n"
            " constexpr bool transpose_a=get_compile_time_arg_val(1)!=0;\n"
            " constexpr bool transpose_b=get_compile_time_arg_val(2)!=0;",
        ),
        ("TensorAccessorArgs<1>()", "TensorAccessorArgs<3>()"),
        (".page_id=(abatch*Mt+mr)*Kt+k/32", ".page_id=transpose_a ? (abatch*Kt+k/32)*Mt+mr : (abatch*Mt+mr)*Kt+k/32"),
        (".page_id=(bbatch*Kt+k/32)*Nt+nc", ".page_id=transpose_b ? (bbatch*Nt+nc)*Kt+k/32 : (bbatch*Kt+k/32)*Nt+nc"),
        ("sa[face_offset(r,k%32)]", "sa[transpose_a ? face_offset(k%32,r) : face_offset(r,k%32)]"),
        ("sb[face_offset(k%32,c)]", "sb[transpose_b ? face_offset(c,k%32) : face_offset(k%32,c)]"),
    ]
    for old, new in reader_edits:
        reader = replace_once(reader, old, new)
    restored = reader
    for old, new in reversed(reader_edits):
        restored = replace_once(restored, new, old)
    assert restored == baseline["FUSED_CORRECTED_MM_READER"]
    cls = next(n for n in nodes if isinstance(n, ast.ClassDef))
    class_source = ast.unparse(cls)
    class_edits = [
        ("class FusedCorrectedMatmul:", "class TransposeCorrectedMatmul:"),
        ("def __call__(self, a, b):", "def __call__(self, a, b, *, transpose_a=False, transpose_b=False):"),
        (
            "plan = fused_corrected_mm_shape_plan(tuple(a.shape), tuple(b.shape))",
            "ashape, bshape = tuple(a.shape), tuple(b.shape)\n"
            "        if transpose_a:\n            ashape = ashape[:-2] + (ashape[-1], ashape[-2])\n"
            "        if transpose_b:\n            bshape = bshape[:-2] + (bshape[-1], bshape[-2])\n"
            "        plan = fused_corrected_mm_shape_plan(ashape, bshape)",
        ),
        ("[1 + len(ac)] + ac + bc", "[3 + len(ac), int(transpose_a), int(transpose_b)] + ac + bc"),
    ]
    candidate_class = class_source
    for old, new in class_edits:
        candidate_class = replace_once(candidate_class, old, new)
    restored_class = candidate_class
    for old, new in reversed(class_edits):
        restored_class = replace_once(restored_class, new, old)
    assert ast.dump(ast.parse(restored_class)) == ast.dump(ast.parse(class_source))
    candidate = dict(baseline)
    candidate["FUSED_CORRECTED_MM_READER"] = reader
    exec(compile(candidate_class, "<transpose-address-corrected-matmul>", "exec"), candidate)
    runtime = next(
        n for n in ast.parse(candidate_class).body[0].body if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    calls = [ast.unparse(n.func) for n in ast.walk(runtime) if isinstance(n, ast.Call)]
    assert calls.count("t.generic_op") == 1
    assert not any(s.startswith("torch.") or s.endswith((".transpose", ".from_torch", ".to_torch")) for s in calls)
    proof = dict(
        snapshot=str(SNAPSHOT),
        snapshot_sha256=SNAPSHOT_SHA,
        baseline_extraction_sha256=digest(baseline_source),
        candidate_class_sha256=digest(candidate_class),
        baseline_reader_sha256=digest(baseline["FUSED_CORRECTED_MM_READER"]),
        candidate_reader_sha256=digest(reader),
        compute_sha256=digest(baseline["FUSED_CORRECTED_MM_COMPUTE"]),
        writer_sha256=digest(baseline["FUSED_CORRECTED_MM_WRITER"]),
        reader_inverse_exact=True,
        class_inverse_ast_exact=True,
        compute_writer_cb_schedule_geometry_unchanged=True,
        single_generic_op_no_host_conversion=True,
        reader_edits=reader_edits,
        class_edits=class_edits,
    )
    return baseline, candidate, proof, baseline_source, candidate_class


def face(r, c):
    return (r // 16) * 512 + (c // 16) * 256 + (r % 16) * 16 + c % 16


def physical_address(batch, row, col, height, width):
    return ((batch * ((height + 31) // 32) + row // 32) * ((width + 31) // 32) + col // 32) * 1024 + face(
        row % 32, col % 32
    )


def mapping_checks(baseline):
    """Independently address every logical operand coordinate for all four modes."""
    assert {face(r, c) for r in range(32) for c in range(32)} == set(range(1024))
    cases = [
        ((1, 32), (1, 32), 64, 128, 64),
        ((1, 32), (1, 32), 128, 64, 128),
        ((2, 1), (1, 3), 33, 65, 17),
        ((), (2, 3), 1, 1, 1),
        ((2, 1, 4), (1, 3, 1), 31, 33, 65),
    ]
    results = []
    for alead, blead, m, k, n in cases:
        plan = baseline["fused_corrected_mm_shape_plan"](alead + (m, k), blead + (k, n))
        count = 0
        for ta, tb in itertools.product((False, True), repeat=2):
            for batch in range(math.prod(plan["leading"])):
                quotient, abatch, bbatch = batch, 0, 0
                coords = []
                for j in reversed(range(len(plan["leading"]))):
                    coordinate = quotient % plan["leading"][j]
                    quotient //= plan["leading"][j]
                    abatch += coordinate * plan["a_strides"][j]
                    bbatch += coordinate * plan["b_strides"][j]
                    coords.insert(0, coordinate)
                # Independent row-major broadcast batch indexing.
                for lead, got in ((alead, abatch), (blead, bbatch)):
                    padded = (1,) * (len(coords) - len(lead)) + lead
                    want = 0
                    for coord, dim in zip(coords, padded):
                        want = want * dim + (coord if dim > 1 else 0)
                    assert want == got
                for kk in range(k):
                    for row in range(m):
                        mr, rr = divmod(row, 32)
                        page = (
                            (abatch * plan["Kt"] + kk // 32) * plan["Mt"] + mr
                            if ta
                            else (abatch * plan["Mt"] + mr) * plan["Kt"] + kk // 32
                        )
                        got = page * 1024 + (face(kk % 32, rr) if ta else face(rr, kk % 32))
                        want = (
                            physical_address(abatch, kk, row, k, m) if ta else physical_address(abatch, row, kk, m, k)
                        )
                        assert got == want
                        count += 1
                    for col in range(n):
                        nc, cc = divmod(col, 32)
                        page = (
                            (bbatch * plan["Nt"] + nc) * plan["Kt"] + kk // 32
                            if tb
                            else (bbatch * plan["Kt"] + kk // 32) * plan["Nt"] + nc
                        )
                        got = page * 1024 + (face(cc, kk % 32) if tb else face(kk % 32, cc))
                        want = (
                            physical_address(bbatch, col, kk, n, k) if tb else physical_address(bbatch, kk, col, k, n)
                        )
                        assert got == want
                        count += 1
        results.append(dict(a_leading=alead, b_leading=blead, M=m, N=n, K=k, raw_address_checks=count))
    return results


def fixtures():
    import torch

    assert sha(CPU_FIXTURE) == CPU_SHA and sha(DEVICE_FIXTURE) == DEVICE_SHA
    cpu = torch.load(CPU_FIXTURE, weights_only=False, map_location="cpu")
    device = torch.load(DEVICE_FIXTURE, weights_only=False, map_location="cpu")
    operands = cpu["cases"][128]["operands"]
    inter = device[128]["intermediates"]
    cases = []
    for chunk in range(2):
        key = operands["key"][:, :, chunk * 64 : (chunk + 1) * 64, :].float().contiguous()
        for name, a, b, ta, tb in (
            ("gram", inter[f"chunk{chunk}/k_beta"], key, False, True),
            ("state_update", inter[f"chunk{chunk}/weighted_k"], inter[f"chunk{chunk}/v_new"], True, False),
            ("output_attention", inter[f"chunk{chunk}/query_scaled"], key, False, True),
        ):
            cases.append(
                dict(
                    name=f"actual_chunk{chunk}_{name}",
                    a=a.float().contiguous(),
                    b=b.float().contiguous(),
                    transpose_a=ta,
                    transpose_b=tb,
                    timing=chunk == 0,
                    source="saved fullgraph operands",
                )
            )
    gen = torch.Generator().manual_seed(827153)
    for ta, tb in itertools.product((False, True), repeat=2):
        for m, k, n, al, bl, tag in (
            (64, 128, 64, (1, 2), (1, 2), "rectangular"),
            (33, 65, 17, (2, 1), (1, 3), "broadcast_partial"),
            (1, 1, 1, (), (2, 3), "singleton"),
        ):
            ashape = al + ((k, m) if ta else (m, k))
            bshape = bl + ((n, k) if tb else (k, n))
            cases.append(
                dict(
                    name=f"random_{tag}_ta{int(ta)}_tb{int(tb)}",
                    a=torch.randn(ashape, generator=gen) * 0.03,
                    b=torch.randn(bshape, generator=gen) * 0.02,
                    transpose_a=ta,
                    transpose_b=tb,
                    timing=False,
                    source="seed827153",
                )
            )
    # Deliberate exceptional finite encodings. Native transpose remains part of the strict gate.
    codes = torch.tensor(
        [0, -2147483648, 1, -2147483647, 0x007FFFFF, -2139095041, 0x00800000, -2139095040, 0x3F800000, -1082130432],
        dtype=torch.int32,
    ).view(torch.float32)
    for ta, tb in itertools.product((False, True), repeat=2):
        logical_a = codes.repeat(103)[:1024].reshape(32, 32)
        logical_b = torch.ones((32, 32)) * 0.5
        cases.append(
            dict(
                name=f"finite_zero_subnormal_ta{int(ta)}_tb{int(tb)}",
                a=(logical_a.T if ta else logical_a).contiguous(),
                b=(logical_b.T if tb else logical_b).contiguous(),
                transpose_a=ta,
                transpose_b=tb,
                timing=False,
                source="explicit signed zero/subnormal controls",
            )
        )
    assert all(c["a"].dtype == c["b"].dtype == torch.float32 for c in cases)
    return cases


def bitdiff(a, b):
    import torch

    assert tuple(a.shape) == tuple(b.shape)
    a, b = a.float().contiguous(), b.float().contiguous()
    return dict(
        fp32_different=int((a.view(torch.int32) != b.view(torch.int32)).sum()),
        bf16_different=int((a.bfloat16().view(torch.int16) != b.bfloat16().view(torch.int16)).sum()),
        max_abs=float((a - b).abs().max()),
        exact=bool(torch.equal(a.view(torch.int32), b.view(torch.int32))),
    )


def benchmark(t, mesh, call, expected, args):
    for _ in range(3):
        out = call()
    assert bitdiff(expected, t.to_torch(out))["exact"]
    eager = []
    for _ in range(args.samples):
        t.synchronize_device(mesh)
        start = time.perf_counter()
        out = call()
        t.synchronize_device(mesh)
        eager.append((time.perf_counter() - start) * 1000)
    assert bitdiff(expected, t.to_torch(out))["exact"]
    out = None
    trace = t.begin_trace_capture(mesh, cq_id=0)
    out = call()
    t.end_trace_capture(mesh, trace, cq_id=0)
    try:
        samples = []
        for _ in range(args.samples):
            t.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(args.replays):
                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
            t.synchronize_device(mesh)
            samples.append((time.perf_counter() - start) * 1000 / args.replays)
        replay_check = bitdiff(expected, t.to_torch(out))
        assert replay_check["exact"]
        return dict(
            warmed_eager_ms=statistics.median(eager),
            eager_samples_ms=eager,
            traced_warmed_ms=statistics.median(samples),
            trace_samples_ms=samples,
            replay_count=args.samples * args.replays,
            replay_check=replay_check,
        )
    finally:
        t.release_trace(mesh, trace)


def device_run(args, cases, baseline, candidate, report, save):
    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    assert "TT_VISIBLE_DEVICES" not in os.environ, "Topology requires unset TT_VISIBLE_DEVICES on this host"
    if args.timing:
        assert not any(
            k in os.environ for k in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_PROFILER_ENABLE")
        )
    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=67108864, l1_small_size=32768
        )
        native = baseline["FusedCorrectedMatmul"](t, mesh)
        fused = candidate["TransposeCorrectedMatmul"](t, mesh)
        report["device_cases"] = []
        for case in cases:

            def upload(x):
                return t.from_torch(
                    x, device=mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
                )

            a, b = upload(case["a"]), upload(case["b"])
            ta, tb = case["transpose_a"], case["transpose_b"]
            row = dict(name=case["name"], transpose_a=ta, transpose_b=tb)
            report["device_cases"].append(row)
            row["input_a_roundtrip"] = bitdiff(case["a"], t.to_torch(a))
            row["input_b_roundtrip"] = bitdiff(case["b"], t.to_torch(b))
            assert row["input_a_roundtrip"]["exact"] and row["input_b_roundtrip"]["exact"]
            # Require native transpose itself to preserve all logical bits, including controls.
            with device_only():
                at = t.transpose(a, -1, -2) if ta else a
                bt = t.transpose(b, -1, -2) if tb else b
            row["native_a_transpose"] = bitdiff(case["a"].transpose(-1, -2) if ta else case["a"], t.to_torch(at))
            row["native_b_transpose"] = bitdiff(case["b"].transpose(-1, -2) if tb else case["b"], t.to_torch(bt))
            save()
            assert row["native_a_transpose"]["exact"] and row["native_b_transpose"]["exact"]

            def base_call():
                with device_only():
                    aa = t.transpose(a, -1, -2) if ta else a
                    bb = t.transpose(b, -1, -2) if tb else b
                    return native(aa, bb)

            def candidate_call():
                with device_only():
                    return fused(a, b, transpose_a=ta, transpose_b=tb)

            expected = t.to_torch(base_call())
            out = candidate_call()
            row["output"] = bitdiff(expected, t.to_torch(out))
            row["plan"] = fused.last_plan
            save()
            assert row["output"]["exact"]
            if args.timing and case["timing"]:
                row["baseline_timing"] = benchmark(t, mesh, base_call, expected, args)
                row["candidate_timing"] = benchmark(t, mesh, candidate_call, expected, args)
                save()
            t.deallocate(a)
            t.deallocate(b)
            if ta:
                t.deallocate(at)
            if tb:
                t.deallocate(bt)
            t.deallocate(out)
        report["device_status"] = "pass"
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            report["device_closed"] = True
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve previous evidence"
    assert args.samples > 0 and args.replays > 0
    assert not args.timing or args.run_device
    baseline, candidate, proof, baseline_source, candidate_class = source_bundle()
    cases = fixtures()
    report = dict(
        classification="isolated transpose-address fusion; not whole-decoder qualification",
        script_sha256=sha(__file__),
        source_proof=proof,
        mapping_checks=mapping_checks(baseline),
        fixture_sources={str(CPU_FIXTURE): CPU_SHA, str(DEVICE_FIXTURE): DEVICE_SHA},
        fixtures=[{k: list(v.shape) if k in ("a", "b") else v for k, v in c.items()} for c in cases],
        source_only=not args.run_device,
        cb_bytes_per_core=40960,
        max_cores=64,
        no_persistent_device_allocation=True,
        status="source_setup_pass",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for suffix, content in (
        ("source.py.snapshot", Path(__file__).read_text()),
        ("baseline.py.snapshot", baseline_source),
        ("candidate_class.py.snapshot", candidate_class),
        ("reader.cpp.snapshot", candidate["FUSED_CORRECTED_MM_READER"]),
        ("compute.cpp.snapshot", candidate["FUSED_CORRECTED_MM_COMPUTE"]),
        ("writer.cpp.snapshot", candidate["FUSED_CORRECTED_MM_WRITER"]),
    ):
        path = Path(str(args.output) + "." + suffix)
        assert not path.exists()
        path.write_text(content)
    save()
    if args.run_device:
        try:
            device_run(args, cases, baseline, candidate, report, save)
            report["status"] = "pass"
        except Exception:
            report["status"] = "failed"
            report["error"] = traceback.format_exc()
            save()
            raise
    assert sha(__file__) == report["script_sha256"]
    save()
    print(json.dumps(dict(status=report["status"], output=str(args.output), cases=len(cases))))


if __name__ == "__main__":
    main()
