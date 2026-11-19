# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic same-input recurrent state rounding fusion; never a decoder fallback.

--oracle-only builds and validates CPU cases without importing TTNN. Device mode
must be run by the serialized hardware owner. Full FP32 state tensors and
broadcast per-head decay match the actual decode geometry. This is not a latency
benchmark or full-decoder acceptance test.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

ROOT = Path(__file__).parents[1]
DOC = ROOT / "doc/fused_decoder"
SHAPE = (1, 32, 128, 128)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(reference, actual):
    a = reference.float().contiguous().reshape(SHAPE)
    b = actual.float().contiguous().reshape(SHAPE)
    different = a.view(torch.int32) != b.view(torch.int32)
    indices = different.flatten().nonzero().flatten()[:8]
    aa, bb = a.double().flatten(), b.double().flatten()
    return {
        "fp32_code_differences": int(different.sum()),
        "signed_zero_code_differences": int((different & (a == 0) & (b == 0)).sum()),
        "reference_subnormal_actual_zero": int(
            ((a.abs() > 0) & (a.abs() < torch.finfo(torch.float32).tiny) & (b == 0)).sum()
        ),
        "max_abs": float((aa - bb).abs().max()),
        "relative_l2": float((aa - bb).norm() / aa.norm().clamp_min(1e-300)),
        "all_finite": bool(torch.isfinite(b).all()),
        "per_head_code_differences": different.reshape(32, -1).sum(-1).tolist(),
        "first_differences": [
            dict(
                flat_index=int(i),
                reference=float(aa[i]),
                actual=float(bb[i]),
                reference_code=f"{int(a.flatten().view(torch.int32)[i]) & 0xffffffff:08x}",
                actual_code=f"{int(b.flatten().view(torch.int32)[i]) & 0xffffffff:08x}",
            )
            for i in indices
        ],
    }


def build_cases(path, seed):
    capture = torch.load(path, map_location="cpu", weights_only=False)["tt_values"]
    # new_state is raw FP32 before the next public call's BF16 cache boundary;
    # delta_state_input in this older capture has already crossed that boundary.
    actual = capture["new_state"].float().contiguous().reshape(SHAPE)
    decay = capture["delta_decay"].float().contiguous().reshape(1, 32, 1, 1)
    generator = torch.Generator().manual_seed(seed)
    scale = actual.reshape(32, -1).std(-1).reshape(1, 32, 1, 1).clamp_min(1e-8)
    random = torch.randn(SHAPE, generator=generator) * scale
    count = actual.numel()
    exponent = torch.randint(97, 143, (count,), generator=generator, dtype=torch.int64)
    mantissa = torch.randint(0, 128, (count,), generator=generator, dtype=torch.int64)
    sign = torch.randint(0, 2, (count,), generator=generator, dtype=torch.int64)
    offsets = torch.tensor([0, 0x7FFF, 0x8000, 0x8001, 0xFFFF], dtype=torch.int64)
    bits = (sign << 31) | (exponent << 23) | (mantissa << 16) | offsets[torch.arange(count) % 5]
    ties = bits.to(torch.int32).view(torch.float32).reshape(SHAPE)
    ties.flatten()[:2] = torch.tensor([0.0, -0.0])
    # Includes even/odd BF16 LSB, both signs, and just below/at/above halfway.
    cases = [("captured_raw_new_state", actual), ("random_real_head_scales", random), ("rne_halfway_neighbors", ties)]
    decays = [
        ("captured_decay", decay),
        (
            "unit_and_varied_decay",
            torch.cat((torch.ones(8), torch.linspace(0.03125, 0.96875, 24))).reshape(1, 32, 1, 1),
        ),
    ]
    for _, state in cases:
        assert state.shape == SHAPE and torch.isfinite(state).all()
        code = state.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        rounded = ((code + 0x7FFF + ((code >> 16) & 1)) & 0xFFFF0000).to(torch.int32).view(torch.float32)
        assert torch.equal(rounded.view(torch.int32), state.bfloat16().float().view(torch.int32))
    assert all(torch.isfinite(d).all() and ((d > 0) & (d <= 1)).all() for _, d in decays)
    return cases, decays


def main(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    source = ROOT / "tt/fused_decoder.py"
    result = dict(
        diagnostic_only=True,
        acceptance_eligible=False,
        execution="CPU only" if args.oracle_only else "serialized physical chip0",
        input=str(args.input),
        input_sha256=digest(args.input),
        seed=args.seed,
        source_sha256=digest(source),
        state_shape=SHAPE,
        decay_shape=(1, 32, 1, 1),
        cases=[],
        device_comparison_pass=None,
        comparison_contract="Complete old two-cast multiply versus fused multiply: bitwise FP32 equality and finite output. Standalone rounding and CPU products are diagnostics.",
    )
    cases, decays = build_cases(args.input, args.seed)
    mesh = None
    start = time.monotonic()
    try:
        if not args.oracle_only:
            import ttnn

            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
            round_op = ttnn.UnaryWithParam(
                ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value)
            )
            widen_op = ttnn.UnaryWithParam(
                ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value)
            )
            chain = [round_op, widen_op]

            def upload(value):
                return ttnn.from_torch(
                    value.contiguous(),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            def host(value):
                return ttnn.to_torch(value).reshape(SHAPE).contiguous()

        for name, state in cases:
            rounded_cpu = state.bfloat16().float()
            item = dict(
                name=name,
                elements=state.numel(),
                min=float(state.min()),
                max=float(state.max()),
                cpu_round_changes=int((state.view(torch.int32) != rounded_cpu.view(torch.int32)).sum()),
                round_comparison=None,
                decays=[],
            )
            result["cases"].append(item)
            if args.oracle_only:
                continue
            device_state = upload(state)
            old_round = ttnn.typecast(ttnn.typecast(device_state, ttnn.bfloat16), ttnn.float32)
            fused_round = ttnn.unary_chain(device_state, chain)
            old_round_host, fused_round_host = host(old_round), host(fused_round)
            item["round_comparison"] = dict(
                old_vs_chain=metrics(old_round_host, fused_round_host),
                old_vs_cpu=metrics(rounded_cpu, old_round_host),
                chain_vs_cpu=metrics(rounded_cpu, fused_round_host),
            )
            for decay_name, decay in decays:
                device_decay = upload(decay)
                old = ttnn.multiply(old_round, device_decay)
                fused = ttnn.multiply(device_state, device_decay, input_tensor_a_activations=chain)
                actual_old, actual_fused = host(old), host(fused)
                expected = rounded_cpu * decay
                # Explicit diagnostic model of device FTZ inputs/outputs and +0
                # canonicalization. It never changes the old-vs-fused gate.
                tiny = torch.finfo(torch.float32).tiny
                ftz_state = torch.where(rounded_cpu.abs() < tiny, 0.0, rounded_cpu)
                ftz_decay = torch.where(decay.abs() < tiny, 0.0, decay)
                expected_ftz = ftz_state * ftz_decay
                expected_ftz = torch.where(expected_ftz.abs() < tiny, 0.0, expected_ftz)
                item["decays"].append(
                    dict(
                        name=decay_name,
                        decay_subnormal_heads=int(((decay.abs() > 0) & (decay.abs() < tiny)).sum()),
                        old_vs_fused=metrics(actual_old, actual_fused),
                        old_vs_cpu=metrics(expected, actual_old),
                        fused_vs_cpu=metrics(expected, actual_fused),
                        old_vs_cpu_ftz_model=metrics(expected_ftz, actual_old),
                        fused_vs_cpu_ftz_model=metrics(expected_ftz, actual_fused),
                    )
                )
                del device_decay, old, fused
            del device_state, old_round, fused_round
            output.write_text(json.dumps(result, indent=2) + "\n")
        if not args.oracle_only:
            result["device_comparison_pass"] = all(
                all(
                    d["old_vs_fused"]["fp32_code_differences"] == 0 and d["old_vs_fused"]["all_finite"]
                    for d in case["decays"]
                )
                for case in result["cases"]
            )
    except Exception as error:
        result["error"] = repr(error)
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        result["elapsed_seconds"] = time.monotonic() - start
        result["source_unchanged"] = digest(source) == result["source_sha256"]
        if args.oracle_only:
            result["ttnn_imported"] = "ttnn" in sys.modules
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                output=str(output),
                cases=len(result["cases"]),
                device_comparison_pass=result["device_comparison_pass"],
                execution=result["execution"],
            )
        ),
        flush=True,
    )
    if result["device_comparison_pass"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DOC / "endurance/routing_seed154.json.pos5991.pt"))
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
