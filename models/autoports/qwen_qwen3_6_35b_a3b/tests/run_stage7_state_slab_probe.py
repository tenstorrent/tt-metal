# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated full state slab versus selected production recurrence, parent-only hardware."""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch


def fixtures(root):
    cases, provenance = [], []
    for pos in (128, 8157):
        path = root / f"doc/fused_decoder/endurance_precise_probe.json.pos{pos}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)["tt_values"]
        values = {k: saved["delta_" + k].contiguous() for k in ("q", "k", "v", "beta")}
        values["decay"] = saved["delta_g"].exp().contiguous()
        values["state"] = saved["delta_state_input"].contiguous()
        cases.append((f"real_pos{pos}", values))
        provenance.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    signed = {k: torch.zeros_like(v) for k, v in cases[0][1].items()}
    signed["state"][:, 0::2] = -0.0
    signed["q"][:, :, 0::2] = -0.0
    signed["k"][:, :, 1::2] = -0.0
    signed["beta"].fill_(1)
    signed["decay"].fill_(1)
    cases.append(("signed_zero", signed))
    cancellation = {k: v.clone() for k, v in signed.items()}
    cancellation["q"].fill_(1)
    cancellation["k"].fill_(1)
    cancellation["state"][:, :, 0::16] = 2**24
    cancellation["state"][:, :, 1::16] = 1
    cancellation["state"][:, :, 2::16] = -(2**24)
    cancellation["state"][:, :, 3::16] = 1
    cancellation["beta"].fill_(2**-20)
    cases.append(("cancellation", cancellation))
    tiny = {k: v.clone() for k, v in signed.items()}
    tiny["state"].fill_(2**-126)
    tiny["state"][:, 0::2] *= -1
    tiny["k"].fill_(0.5)
    tiny["q"].fill_(0.25)
    tiny["decay"].fill_(0.5)
    cases.append(("subnormal_product", tiny))
    midpoint = {k: v.clone() for k, v in signed.items()}
    # Exact FP32 bit neighbours on either side of BF16 ties, both signs.
    bits = torch.tensor(
        [0x3F807FFF, 0x3F808000, 0x3F808001, 0x3F817FFF, 0x3F818000, 0x3F818001, -1082097665, -1082097664, -1082097663],
        dtype=torch.int32,
    ).view(torch.float32)
    flat = midpoint["state"].flatten()
    flat.copy_(bits.repeat((flat.numel() + bits.numel() - 1) // bits.numel())[: flat.numel()])
    midpoint["k"].fill_(2**-8)
    midpoint["q"].fill_(2**-7)
    midpoint["beta"].fill_(0.5)
    cases.append(("bf16_round_midpoints", midpoint))
    return cases, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("acceptance", "timing", "cpu"), default="acceptance")
    parser.add_argument("--rounded", type=int, choices=(0, 1), default=1)
    parser.add_argument("--state-output", choices=("persistent", "temporary"), default="persistent")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    assert args.steps > 0 and args.iterations > 0
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    record = dict(
        passed=False,
        command=sys.argv,
        mode=args.mode,
        rounded=args.rounded,
        checks=[],
        timings={},
        state_output=args.state_output,
        scope="isolated full recurrence; no full-model acceptance",
        decay_fixture="CPU exp of captured delta_g; same exact values supplied to both paths",
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                root / "tt/state_slab_recurrence_probe.py",
                root / "tt/multichip_decoder.py",
                root / "tt/mesh_product_recurrent_sum.py",
                root / "tt/direct_recurrent_sum_probe.py",
                root / "tt/mesh_direct_recurrent_sum.py",
            )
        },
    )
    mesh, traces = None, {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cases, record["fixture_provenance"] = fixtures(root)
        assert all(torch.isfinite(v).all() for _, values in cases for v in values.values())
        record["fixture_names"] = [name for name, _ in cases]
        if args.mode == "cpu":
            record.update(passed=True, hardware_verified=False)
            return
        import ttnn as t
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_direct_recurrent_sum import DirectRecurrentSum
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_product_recurrent_sum import ProductRecurrentSum
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import LinearTP4Mixin
        from models.autoports.qwen_qwen3_6_35b_a3b.tt.state_slab_recurrence_probe import install

        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == ("1" if args.mode == "acceptance" else "0")
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
        if args.mode == "timing":
            assert not any(
                os.environ.get(key)
                for key in (
                    "TT_METAL_WATCHER",
                    "TT_METAL_DEVICE_PROFILER",
                    "TTNN_OP_PROFILER",
                    "TT_METAL_PROFILER_TRACE_TRACKING",
                )
            )
        t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(t.MeshShape(1, 4), trace_region_size=200000000)

        def upload(key, x, device=False):
            kw = dict(
                dtype=t.bfloat16 if key == "v" else t.float32,
                layout=t.TILE_LAYOUT,
                mesh_mapper=t.ShardTensorToMesh(mesh, dim=1 if key == "state" else 2),
            )
            if device:
                kw.update(device=mesh, memory_config=t.DRAM_MEMORY_CONFIG)
            return t.from_torch(x, **kw)

        inputs = {key: upload(key, x, True) for key, x in cases[0][1].items() if key != "state"}
        states = {name: upload("state", cases[0][1]["state"], True) for name in ("old", "new")}
        chain = [
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value)),
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value)),
        ]
        old = SimpleNamespace(
            kind="linear_attention",
            mesh_device=mesh,
            round_widen_chain=chain,
            full_model_optimized=True,
            full_model_recurrent_sum=DirectRecurrentSum(t, mesh),
            full_model_product_sum=ProductRecurrentSum(t, mesh),
            compute=t.WormholeComputeKernelConfig(
                math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
            ),
        )
        for name in ("_recurrent_step", "_recurrent_sum", "_recurrent_sum_terminal"):
            setattr(old, name, MethodType(getattr(LinearTP4Mixin, name), old))
        new = SimpleNamespace(**vars(old))
        new._recurrent_step = MethodType(LinearTP4Mixin._recurrent_step, new)
        install(new)
        engines = dict(old=old, new=new)

        def call(name):
            output, state = engines[name]._recurrent_step(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                None,
                inputs["beta"],
                states[name],
                decay_value=inputs["decay"],
                round_state=bool(args.rounded),
                output_state=states[name] if args.state_output == "persistent" else None,
            )
            if args.state_output == "temporary":
                assert state.memory_config() == t.L1_MEMORY_CONFIG
                t.copy(state, states[name])
            return output

        def host(x):
            return [t.to_torch(rank).contiguous() for rank in t.get_device_tensors(x)]

        def compare(label, a, b):
            assert len(a) == len(b) == 4
            for rank, (left, right) in enumerate(zip(a, b)):
                assert left.shape == right.shape and left.dtype == right.dtype == torch.float32
                count = int((left.view(torch.int32) != right.view(torch.int32)).sum())
                record["checks"].append(
                    dict(name=label, rank=rank, differences=count, max_abs=float((left - right).abs().max()))
                )
                assert count == 0, (label, rank, count)

        def reset(values):
            for key, x in values.items():
                if key == "state":
                    for target in states.values():
                        t.copy_host_to_device_tensor(upload(key, x), target)
                else:
                    t.copy_host_to_device_tensor(upload(key, x), inputs[key])

        outputs = {}
        for name in engines:
            value = call(name)
            outputs[name] = t.empty_like(value)
            t.copy(value, outputs[name])
            host(outputs[name])
            host(states[name])
        t.synchronize_device(mesh)
        for name in engines:
            trace = t.begin_trace_capture(mesh, cq_id=0)
            try:
                t.copy(call(name), outputs[name])
            finally:
                t.end_trace_capture(mesh, trace, cq_id=0)
                traces[name] = trace
        entries = mesh.num_program_cache_entries()
        for fixture, values in cases:
            for mode in ("eager", "trace"):
                reset(values)
                for step in range(args.steps):
                    # Change every persistent operand between trace invocations.
                    changed = cases[step % 2][1] if fixture.startswith("real") else values
                    for key, x in changed.items():
                        if key != "state":
                            t.copy_host_to_device_tensor(upload(key, x), inputs[key])
                    if mode == "eager":
                        for name in engines:
                            t.copy(call(name), outputs[name])
                    else:
                        for name in engines:
                            t.execute_trace(mesh, traces[name], cq_id=0, blocking=False)
                    compare(f"{fixture}/{mode}/{step}/output", host(outputs["new"]), host(outputs["old"]))
                    compare(f"{fixture}/{mode}/{step}/state", host(states["new"]), host(states["old"]))
        assert mesh.num_program_cache_entries() == entries
        reset(cases[0][1])
        record["timing_fixture"] = cases[0][0]
        for name, trace in traces.items():
            for _ in range(3):
                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
            t.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(args.iterations):
                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
            t.synchronize_device(mesh)
            record["timings"][name] = (time.perf_counter() - start) * 1000 / args.iterations
        record.update(
            passed=True, hardware_verified=True, timing_eligible=args.mode == "timing", program_cache_entries=entries
        )
    except Exception:
        record["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        if mesh is not None:
            import ttnn as t

            for trace in traces.values():
                t.release_trace(mesh, trace)
            t.close_mesh_device(mesh)
        print(
            json.dumps(dict(passed=record["passed"], checks=len(record["checks"]), timings=record["timings"])),
            flush=True,
        )


if __name__ == "__main__":
    main()
