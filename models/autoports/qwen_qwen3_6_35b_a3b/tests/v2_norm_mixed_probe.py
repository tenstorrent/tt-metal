# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic: v2 mixed BF16 norm operands, exact FP32 cuts and complete output.

No hardware is used with --oracle-only. Default device execution is reserved for
its serialized owner. This probe supplies no performance or decoder acceptance claim.
"""

import argparse
import json
import runpy
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.exact_next_fusion_probe import DOC, compare, digest, fixtures
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    assert not output.exists(), f"Preserve existing evidence: {output}"
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    base = DOC / "selected_fused_decoder_v2.py.snapshot"
    candidate = Path(args.candidate_snapshot)
    provenance = json.loads(candidate.with_suffix(".provenance.json").read_text())
    assert digest(base) == provenance["base_sha256"] and digest(candidate) == provenance["candidate_sha256"]
    rows, _, gamma, eps = fixtures(args)
    oracle = torch.load(args.oracle, map_location="cpu", weights_only=False)
    rows.append(("prefill_first128", oracle["prefix"][:, :128].contiguous()))
    codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    finite = codes[torch.isfinite(codes)].reshape(1, -1, 32)
    result = dict(
        status="running",
        diagnostic_only=True,
        acceptance_eligible=False,
        base_sha256=digest(base),
        candidate_sha256=digest(candidate),
        probe_sha256=digest(__file__),
        oracle_sha256=digest(args.oracle),
        finite_bf16_codes=finite.numel(),
        checks=[],
    )
    mesh = None
    stage = "cpu_fixtures"

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    def check(label, a, b):
        stats = compare(a, b)
        result["checks"].append(dict(name=label, **stats))
        save()
        print("V2_NORM_MIXED", label, stats["fp32_code_differences"], flush=True)

    save()
    try:
        assert all(x.dtype == torch.bfloat16 and torch.isfinite(x).all() for _, x in rows)
        result["fixtures"] = [dict(name=n, shape=list(x.shape)) for n, x in rows]
        if args.oracle_only:
            result["status"] = "cpu_fixtures_passed"
            return True
        import ttnn

        params = dict(physical_device_ids=[0], trace_region_size=0, l1_small_size=32768)
        result["device_open_parameters"] = params
        stage = "open_device"
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **params)
        classes = [runpy.run_path(str(p))["FusedDecoder"] for p in (base, candidate)]
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

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        obj = SimpleNamespace(
            kind="linear_attention",
            compute=compute,
            round_widen_chain=rw,
            mesh_device=mesh,
            config=SimpleNamespace(rms_norm_eps=eps),
            w={"input_layernorm.weight": upload(gamma, ttnn.float32)},
        )
        stage = "all_finite_bf16_square"
        x = upload(finite)
        wide = ttnn.typecast(x, ttnn.float32)
        old = ttnn.multiply(wide, wide)
        new = ttnn.multiply(x, x, dtype=ttnn.float32, fast_and_approximate_mode=False)
        check(stage, ttnn.to_torch(old), ttnn.to_torch(new))
        del x, wide, old, new

        def norm_with_cuts(cls, x):
            original = ttnn.multiply
            cuts = {}

            def multiply(a, b, *pos, **kw):
                out = original(a, b, *pos, **kw)
                if hasattr(b, "shape") and a.shape[-1] == 2048:
                    if a is b:
                        cuts["squares"] = out
                    elif b.shape[-1] == 1:
                        cuts["normalized_fp32"] = out
                return out

            ttnn.multiply = multiply
            try:
                out = cls.norm(obj, x, "input_layernorm.weight")
            finally:
                ttnn.multiply = original
            assert set(cuts) == {"squares", "normalized_fp32"}, cuts.keys()
            ttnn.synchronize_device(mesh)
            return {**{k: ttnn.to_torch(v) for k, v in cuts.items()}, "complete_norm": ttnn.to_torch(out)}

        for name, row in rows:
            stage = name
            x = upload(row)
            old = norm_with_cuts(classes[0], x)
            new = norm_with_cuts(classes[1], x)
            for key in old:
                check(name + "_" + key, old[key], new[key])
            del x
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
    p.add_argument("--candidate-snapshot", default=str(DOC / "endurance/v2_norm_mixed_operands.py.snapshot"))
    p.add_argument("--oracle", default=str(DOC / "oracle_cache/stress_8192_0.pt"))
    p.add_argument("--positions", nargs="+", type=int, default=[128, 130, 3979, 5991, 7703])
    p.add_argument("--capture-positions", nargs="*", type=int, default=[])
    p.add_argument("--capture-base", default=str(DOC / "endurance_precise_probe.json"))
    p.add_argument("--output", default=str(DOC / "endurance/v2_norm_mixed_device.json"))
    raise SystemExit(0 if run(p.parse_args()) else 1)
