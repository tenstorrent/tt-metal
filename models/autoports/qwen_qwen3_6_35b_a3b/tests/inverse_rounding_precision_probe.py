# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Localize square-root/reciprocal refinement error with identical FP32 inputs."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder"


def datum(expected, actual):
    expected, actual = expected.float().reshape(()), actual.float().reshape(())
    return dict(
        expected=float(expected),
        actual=float(actual),
        expected_bits=int(expected.view(torch.int32)),
        actual_bits=int(actual.view(torch.int32)),
        equal=bool(expected.view(torch.int32) == actual.view(torch.int32)),
    )


def run(args):
    torch.set_num_threads(8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    source = torch.load(args.stream, map_location="cpu", weights_only=False)
    stream = torch.cat([source["prefix"], source["tokens"]], dim=1).reshape(-1, 2048).float()
    x = stream[args.positions]
    a = x.square().mean(-1) + 1e-6
    correct_sqrt = a.double().sqrt().float()
    correct_inverse = (1 / correct_sqrt.double()).float()
    hf = torch.rsqrt(a)
    assert torch.equal(correct_inverse, hf), "Two-rounding CPU model failed"
    result = dict(
        purpose="Unmeasured same-input unary and cancellation probe; FP64 CPU computes primitive oracles only",
        stream_sha256=hashlib.sha256(Path(args.stream).read_bytes()).hexdigest(),
        positions=args.positions,
        rows=[],
    )
    if args.oracle_only:
        result["rows"] = [
            dict(position=p, shifted_variance=float(v), correct_sqrt=float(s), hf_inverse=float(r))
            for p, v, s, r in zip(args.positions, a, correct_sqrt, hf)
        ]
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return

    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:

        def upload(value):
            return ttnn.from_torch(
                value.reshape(1, 1, -1, 1).expand(-1, -1, -1, 32).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
            )

        ta = upload(a)
        ones = upload(torch.ones_like(a))
        root = ttnn.sqrt(ta, fast_and_approximate_mode=False)
        negroot = ttnn.neg(root)
        residual = ttnn.addcmul(ta, negroot, root)
        mac_residual = ttnn.mac(negroot, root, ta)
        split_residual = ttnn.subtract(ta, ttnn.multiply(root, root))
        root_reciprocal = ttnn.reciprocal(root)
        half_reciprocal = ttnn.multiply(root_reciprocal, 0.5)
        correction = ttnn.multiply(residual, half_reciprocal)
        refined_root = ttnn.add(root, correction)
        mac_refined_root = ttnn.add(root, ttnn.multiply(mac_residual, half_reciprocal))
        reciprocal = ttnn.reciprocal(refined_root)
        reciprocal_residual = ttnn.addcmul(ones, ttnn.neg(refined_root), reciprocal)
        reciprocal_mac_residual = ttnn.mac(ttnn.neg(refined_root), reciprocal, ones)
        reciprocal_correction = ttnn.multiply(reciprocal, reciprocal_residual)
        refined_reciprocal = ttnn.add(reciprocal, reciprocal_correction)

        def split(value):
            high = ttnn.typecast(ttnn.typecast(value, ttnn.bfloat16), ttnn.float32)
            return high, ttnn.subtract(value, high)

        high, low = split(root)
        split_sqrt_residual = ttnn.subtract(
            ttnn.subtract(ttnn.subtract(ta, ttnn.multiply(high, high)), ttnn.multiply(ttnn.multiply(high, 2.0), low)),
            ttnn.multiply(low, low),
        )
        split_sqrt = ttnn.add(root, ttnn.multiply(split_sqrt_residual, half_reciprocal))
        split_inverse = ttnn.reciprocal(split_sqrt)
        rh, rl = split(split_sqrt)
        ih, il = split(split_inverse)
        split_inverse_residual = ttnn.subtract(
            ttnn.subtract(
                ttnn.subtract(ttnn.subtract(ones, ttnn.multiply(rh, ih)), ttnn.multiply(rh, il)), ttnn.multiply(rl, ih)
            ),
            ttnn.multiply(rl, il),
        )
        split_refined_inverse = ttnn.add(split_inverse, ttnn.multiply(split_inverse, split_inverse_residual))
        tensors = dict(
            split_sqrt_residual=split_sqrt_residual,
            split_sqrt=split_sqrt,
            split_inverse=split_inverse,
            split_inverse_residual=split_inverse_residual,
            split_refined_inverse=split_refined_inverse,
            raw_sqrt=root,
            sqrt_residual=residual,
            sqrt_mac_residual=mac_residual,
            sqrt_split_residual=split_residual,
            root_reciprocal=root_reciprocal,
            half_reciprocal=half_reciprocal,
            sqrt_correction=correction,
            refined_sqrt=refined_root,
            mac_refined_sqrt=mac_refined_root,
            raw_reciprocal=reciprocal,
            reciprocal_residual=reciprocal_residual,
            reciprocal_mac_residual=reciprocal_mac_residual,
            reciprocal_correction=reciprocal_correction,
            refined_reciprocal=refined_reciprocal,
        )
        got = {key: ttnn.to_torch(value).reshape(-1, 32)[:, 0].contiguous() for key, value in tensors.items()}
        # Compare each primitive against correctly rounded arithmetic on its actual device operands.
        r = got["raw_sqrt"]
        rr = got["refined_sqrt"]
        inv = got["raw_reciprocal"]
        exact_residual = (a.double() - r.double() * r.double()).float()
        exact_reciprocal_residual = (1 - rr.double() * inv.double()).float()
        h, l = r.bfloat16().float(), r - r.bfloat16().float()
        sr, si = got["split_sqrt"], got["split_inverse"]
        rh, rl = sr.bfloat16().float(), sr - sr.bfloat16().float()
        ih, il = si.bfloat16().float(), si - si.bfloat16().float()
        expected = dict(
            split_sqrt_residual=((a - h * h) - (2 * h) * l) - l * l,
            split_sqrt=r + got["split_sqrt_residual"] * got["half_reciprocal"],
            split_inverse=(1 / sr.double()).float(),
            split_inverse_residual=(((1 - rh * ih) - rh * il) - rl * ih) - rl * il,
            split_refined_inverse=si + si * got["split_inverse_residual"],
            raw_sqrt=correct_sqrt,
            sqrt_residual=exact_residual,
            sqrt_mac_residual=exact_residual,
            sqrt_split_residual=a - r * r,
            root_reciprocal=(1 / r.double()).float(),
            half_reciprocal=got["root_reciprocal"] * 0.5,
            sqrt_correction=got["sqrt_residual"] * got["half_reciprocal"],
            refined_sqrt=r + got["sqrt_correction"],
            mac_refined_sqrt=r + got["sqrt_mac_residual"] * got["half_reciprocal"],
            raw_reciprocal=(1 / rr.double()).float(),
            reciprocal_residual=exact_reciprocal_residual,
            reciprocal_mac_residual=exact_reciprocal_residual,
            reciprocal_correction=inv * got["reciprocal_residual"],
            refined_reciprocal=inv + got["reciprocal_correction"],
        )
        cpu_refined_root = r + exact_residual * ((1 / r.double()).float() * 0.5)
        cpu_raw_inverse = (1 / cpu_refined_root.double()).float()
        cpu_inverse_residual = (1 - cpu_refined_root.double() * cpu_raw_inverse.double()).float()
        cpu_refined_inverse = cpu_raw_inverse + cpu_raw_inverse * cpu_inverse_residual
        for i, position in enumerate(args.positions):
            stages = {key: datum(expected[key][i], got[key][i]) for key in tensors}
            result["rows"].append(
                dict(
                    position=position,
                    shifted_variance=float(a[i]),
                    stages=stages,
                    split_refined_sqrt_vs_correct=datum(correct_sqrt[i], sr[i]),
                    split_refined_inverse_vs_hf=datum(hf[i], got["split_refined_inverse"][i]),
                    refined_sqrt_vs_correct=datum(correct_sqrt[i], rr[i]),
                    refined_inverse_vs_hf=datum(hf[i], got["refined_reciprocal"][i]),
                    ieee_emulated_refined_sqrt_vs_correct=datum(correct_sqrt[i], cpu_refined_root[i]),
                    ieee_emulated_refined_inverse_vs_hf=datum(hf[i], cpu_refined_inverse[i]),
                    first_primitive_difference=next((key for key, value in stages.items() if not value["equal"]), None),
                )
            )
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", default=str(DOC / "endurance/mlp_norm_shard_8192_0_results.json.tensors.pt"))
    parser.add_argument("--positions", nargs="+", type=int, default=[1, 19, 39, 52, 58, 89, 94, 119, 130, 430, 1387])
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
