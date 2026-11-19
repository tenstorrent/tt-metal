# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bitwise complete rank-5 head-broadcast recurrence comparison; no performance claim.

--oracle-only checks CPU products and ordered recurrence on saved real inputs.
Only the serialized hardware owner runs the default device branch.
"""

import argparse
import json
import math
import runpy
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.exact_next_fusion_probe import BASE_SHA, DOC, compare, digest
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

KEYS = ("delta_q", "delta_k", "delta_v", "delta_g", "delta_beta", "delta_state_input")


def ordered_sum(product):
    chunks = product.permute(0, 2, 1, 3).reshape(8, 16, 32, 128)
    inner = torch.zeros_like(chunks[:, 0])
    for j in range(16):
        inner = inner + chunks[:, j]
    outer = torch.zeros_like(inner[0])
    for j in range(8):
        outer = outer + inner[j]
    return outer.reshape(1, 1, 32, 128).permute(0, 2, 1, 3)


def cpu_step(inputs, rank5, rounded):
    q, k, v, g, beta, state = inputs
    if rounded:
        state = state.bfloat16().float()
    if rank5:
        q = (q * (128**-0.5)).permute(0, 2, 3, 1).reshape(1, 16, 1, 128, 1)
        k = k.permute(0, 2, 3, 1).reshape(1, 16, 1, 128, 1)
    else:
        q = (q.repeat_interleave(2, dim=2) * (128**-0.5)).permute(0, 2, 3, 1)
        k = k.repeat_interleave(2, dim=2).permute(0, 2, 3, 1)
    decayed = state * g.exp().reshape(1, 32, 1, 1)
    memory_product = ((decayed.reshape(1, 16, 2, 128, 128) if rank5 else decayed) * k).reshape(1, 32, 128, 128)
    memory = ordered_sum(memory_product)
    delta = (v.permute(0, 2, 1, 3) - memory) * beta.reshape(1, 32, 1, 1)
    update_product = (k * (delta.reshape(1, 16, 2, 1, 128) if rank5 else delta)).reshape(1, 32, 128, 128)
    new = decayed + update_product
    output_product = ((new.reshape(1, 16, 2, 128, 128) if rank5 else new) * q).reshape(1, 32, 128, 128)
    reduction = ordered_sum(output_product)
    return dict(
        decayed=decayed,
        memory_product=memory_product,
        memory=memory,
        update_product=update_product,
        output_product=output_product,
        reduced_output=reduction,
        new_state=new,
        core=reduction.permute(0, 2, 1, 3),
    )


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    base = DOC / "selected_fused_decoder.py.snapshot"
    candidate = Path(args.candidate_snapshot)
    assert digest(base) == BASE_SHA
    provenance = json.loads(candidate.with_suffix(".provenance.json").read_text())
    assert digest(candidate) == provenance["candidate_sha256"]
    result = dict(
        status="running",
        diagnostic_only=True,
        acceptance_eligible=False,
        execution="CPU only" if args.oracle_only else "serialized physical chip0",
        base_sha256=digest(base),
        candidate_sha256=digest(candidate),
        candidate_snapshot=str(candidate),
        probe_sha256=digest(__file__),
        checks=[],
        captures=[],
    )
    mesh = None
    stage = "fixtures"

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    def check(label, old, new):
        stats = compare(old, new)
        result["checks"].append(dict(name=label, **stats))
        save()
        print("RANK5_RECURRENCE", label, stats["fp32_code_differences"], flush=True)

    save()
    try:
        fixtures = []
        for pos in args.positions:
            path = Path(f"{args.capture_base}.pos{pos}.pt")
            values = torch.load(path, map_location="cpu", weights_only=False)["tt_values"]
            inputs = [values[k].float().contiguous() for k in KEYS]
            assert all(torch.isfinite(x).all() for x in inputs)
            fixtures.append((pos, inputs))
            result["captures"].append(
                dict(
                    position=pos,
                    path=str(path),
                    sha256=digest(path),
                    shapes={k: list(x.shape) for k, x in zip(KEYS, inputs)},
                )
            )
            for rounded in (False, True):
                old, new = cpu_step(inputs, False, rounded), cpu_step(inputs, True, rounded)
                for key in old:
                    check(f"cpu_pos{pos}_round{rounded}_{key}", old[key], new[key])
        if args.oracle_only:
            result["status"] = "cpu_proof_passed" if all(c["passed"] for c in result["checks"]) else "failed"
            return result["status"] == "cpu_proof_passed"

        import ttnn

        base_cls = runpy.run_path(str(base))["FusedDecoder"]
        new_cls = runpy.run_path(str(candidate))["FusedDecoder"]
        stage = "open_device"
        params = dict(physical_device_ids=[0], trace_region_size=0, l1_small_size=32768)
        result["device_open_parameters"] = params
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **params)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        rw = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value)),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value)),
        ]
        obj = SimpleNamespace(compute=compute, round_widen_chain=rw)

        def upload(x):
            return ttnn.from_torch(
                x.contiguous(),
                device=mesh,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def captured_step(cls, inputs, rounded, decay):
            # Keep diagnostic references until the full call finishes; no host transfers inside the graph.
            products, reductions = [], []
            multiply = ttnn.multiply

            def capture_multiply(a, b, *aargs, **kwargs):
                out = multiply(a, b, *aargs, **kwargs)
                if math.prod(tuple(out.shape)) == 32 * 128 * 128:
                    products.append(out)
                return out

            def capture_sum(p):
                out = base_cls._recurrent_sum(obj, p)
                reductions.append(out)
                return out

            obj._recurrent_sum = capture_sum
            ttnn.multiply = capture_multiply
            try:
                core, new_state = cls._recurrent_step(obj, *inputs, round_state=rounded, decay_value=decay)
            finally:
                ttnn.multiply = multiply
            assert len(products) == 4, len(products)
            assert len(reductions) == 2, len(reductions)
            ttnn.synchronize_device(mesh)
            captured = {
                key: ttnn.to_torch(value).reshape(1, 32, 128, 128).contiguous()
                for key, value in zip(("decayed", "memory_product", "update_product", "output_product"), products)
            }
            captured.update(
                memory=ttnn.to_torch(reductions[0]),
                reduced_output=ttnn.to_torch(reductions[1]),
                core=ttnn.to_torch(core),
                new_state=ttnn.to_torch(new_state),
            )
            return captured

        for pos, cpu_inputs in fixtures:
            inputs = [upload(x) for x in cpu_inputs]
            # The explicit-decay fixture is identical for both adapters, not an EXP accuracy assertion.
            explicit_decay = upload(cpu_inputs[3].exp())
            for rounded in (False, True):
                for mode, decay in (("g_exp", None), ("explicit_decay", explicit_decay)):
                    stage = f"device_pos{pos}_round{rounded}_{mode}"
                    old = captured_step(base_cls, inputs, rounded, decay)
                    new = captured_step(new_cls, inputs, rounded, decay)
                    for key in old:
                        check(stage + "_" + key, old[key], new[key])
            del inputs, explicit_decay
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
                result["device_close_error"] = traceback.format_exc()
                result["status"] = "error"
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--positions", nargs="+", type=int, default=[128, 130, 7703])
    parser.add_argument("--capture-base", default=str(DOC / "endurance_precise_probe.json"))
    parser.add_argument("--candidate-snapshot", default=str(DOC / "endurance/exact_next_rank5_heads.py.snapshot"))
    parser.add_argument("--output", default=str(DOC / "endurance/rank5_recurrence_device.json"))
    raise SystemExit(0 if run(parser.parse_args()) else 1)
