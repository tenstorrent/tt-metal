# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic exact prefill scan extracted verbatim from immutable snapshot ASTs.

Saved real per-token operands are repeated to construct deterministic short scans.
This isolates the graph rewrite; it is not a canonical prefill accuracy claim.
"""

import argparse
import ast
import json
import math
import runpy
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.exact_next_fusion_probe import DOC, compare, digest
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.rank5_recurrence_probe import KEYS, cpu_step


def extract_scan(path):
    tree = ast.parse(path.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_linear_attention")
    branch = next(
        n
        for n in fn.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "decode"
        and any(isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) in ("o, new", "(o, new)") for x in n.orelse)
    )
    start = next(
        i for i, n in enumerate(branch.orelse) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "kk"
    )
    body = branch.orelse[start:] + [ast.Return(ast.Name("new", ast.Load()))]
    definition = ast.parse("def scan(self,k,v,beta,decay_values,state,s):\n    pass").body[0]
    definition.body = body
    return ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[]))


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    assert not output.exists(), f"Preserve evidence: {output}"
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    base = DOC / "selected_fused_decoder_v2.py.snapshot"
    candidate = DOC / "endurance/v2_prefill_rank5_state.py.snapshot"
    prov = json.loads(candidate.with_suffix(".provenance.json").read_text())
    assert digest(base) == prov["base_sha256"] and digest(candidate) == prov["candidate_sha256"]
    paths = [Path(f"{args.capture_base}.pos{p}.pt") for p in args.positions]
    captures = [torch.load(p, map_location="cpu", weights_only=False)["tt_values"] for p in paths]
    result = dict(
        status="running",
        diagnostic_only=True,
        acceptance_eligible=False,
        probe_sha256=digest(__file__),
        base_sha256=digest(base),
        candidate_sha256=digest(candidate),
        extracted_scans={p.name: ast.unparse(extract_scan(p)) for p in (base, candidate)},
        fixtures=[dict(path=str(p), sha256=digest(p)) for p in paths],
        checks=[],
    )
    mesh = None
    stage = "cpu_scan"

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    def check(name, a, b):
        stats = compare(a, b)
        result["checks"].append(dict(name=name, **stats))
        save()
        print("V2_PREFILL_RANK5", name, stats["fp32_code_differences"], flush=True)

    save()
    try:
        for length in args.lengths:
            old = captures[0]["delta_state_input"].float()
            new = old.clone()
            for i in range(length):
                values = captures[i % len(captures)]
                inputs = [values[k].float() for k in KEYS]
                inputs[-1] = old
                old = cpu_step(inputs, False, False)["new_state"]
                inputs[-1] = new
                new = cpu_step(inputs, True, False)["new_state"]
            check(f"cpu_scan{length}_final", old, new)
        if args.oracle_only:
            result["status"] = "cpu_proof_passed"
            return True
        import ttnn

        stage = "open_device"
        params = dict(physical_device_ids=[0], trace_region_size=0, l1_small_size=32768)
        result["device_open_parameters"] = params
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **params)
        modules = [runpy.run_path(str(p)) for p in (base, candidate)]
        scans = []
        for path, module in zip((base, candidate), modules):
            exec(compile(extract_scan(path), str(path) + ":prefill_scan", "exec"), module)
            scans.append(module["scan"])
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        obj = SimpleNamespace(compute=compute)

        def upload(x):
            return ttnn.from_torch(
                x.contiguous(),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        for length in args.lengths:
            stage = f"device_scan{length}"
            values = [captures[i % len(captures)] for i in range(length)]
            inputs = [
                upload(torch.cat([v[k].float() for v in values], dim=1)) for k in ("delta_k", "delta_v", "delta_beta")
            ]
            inputs[1] = ttnn.reshape(inputs[1], (1, length, 4096))
            inputs.append(upload(torch.cat([v["delta_g"].float().exp() for v in values], dim=1)))
            initial = upload(captures[0]["delta_state_input"].float())

            def captured_scan(fn):
                multiply, add = ttnn.multiply, ttnn.add
                large_count = 0
                sum_count = 0
                held = {}
                selected = {0, min(1, length - 1), length - 1}

                def capture_mul(a, b, *pos, **kwargs):
                    nonlocal large_count
                    out = multiply(a, b, *pos, **kwargs)
                    if math.prod(tuple(out.shape)) == 32 * 128 * 128:
                        row, part = divmod(large_count, 3)
                        large_count += 1
                        if row in selected:
                            held[f'row{row}_{("decayed","memory_product","update_product")[part]}'] = out
                    return out

                def capture_add(a, b, *pos, **kwargs):
                    out = add(a, b, *pos, **kwargs)
                    row = large_count // 3 - 1
                    if math.prod(tuple(out.shape)) == 32 * 128 * 128 and row in selected:
                        held[f"row{row}_updated_state"] = out
                    return out

                def capture_sum(p):
                    nonlocal sum_count
                    out = modules[0]["FusedDecoder"]._recurrent_sum(obj, p)
                    if sum_count in selected:
                        held[f"row{sum_count}_memory"] = out
                    sum_count += 1
                    return out

                obj._recurrent_sum = capture_sum
                ttnn.multiply = capture_mul
                ttnn.add = capture_add
                try:
                    out = fn(obj, *inputs, {"recurrent": initial}, length)
                finally:
                    ttnn.multiply = multiply
                    ttnn.add = add
                assert large_count == 3 * length and sum_count == length, (large_count, sum_count)
                ttnn.synchronize_device(mesh)
                cpu = {
                    k: (
                        ttnn.to_torch(v).reshape(1, 32, 128, 128)
                        if math.prod(tuple(v.shape)) == 32 * 128 * 128
                        else ttnn.to_torch(v)
                    )
                    for k, v in held.items()
                }
                cpu["final_state"] = ttnn.to_torch(out)
                return cpu

            old = captured_scan(scans[0])
            new = captured_scan(scans[1])
            assert old.keys() == new.keys()
            for key in old:
                check(stage + "_" + key, old[key], new[key])
            del inputs, initial
        result["status"] = "passed" if all(c["passed"] for c in result["checks"]) else "failed"
        return result["status"] == "passed"
    except Exception:
        result.update(status="error", failed_stage=stage, traceback=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
            except Exception:
                result["status"] = "error"
                result["close_error"] = traceback.format_exc()
        save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--positions", nargs="+", type=int, default=[128, 130, 7703])
    p.add_argument("--lengths", nargs="+", type=int, default=[1, 32])
    p.add_argument("--capture-base", default=str(DOC / "endurance_precise_probe.json"))
    p.add_argument("--output", default=str(DOC / "endurance/v2_prefill_rank5_device.json"))
    raise SystemExit(0 if run(p.parse_args()) else 1)
