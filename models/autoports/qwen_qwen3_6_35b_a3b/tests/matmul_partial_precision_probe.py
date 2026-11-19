# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolate Blackhole MVMUL summation with same-product pairs and residue masks.

All matmuls use ordinary 32x32 tiles, original BF16 operands, HiFi4 and FP32
output/destination. Masking weights is diagnostic setup, never a model change.
Run serially on physical chip 0. --oracle-only performs no TTNN import/device work.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.frontend_precision_probe import compare
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import SNAPSHOT

SEGMENTS = (("qkv", 0, 8192), ("z", 8192, 12288), ("a", 12288, 12320), ("b", 12320, 12352))
PAIRS = ((0, 1), (2, 17), (31, 32), (128, 129), (256, 1024), (1024, 1025), (1536, 1792), (2046, 2047))


def load_inputs(path):
    captured = torch.load(path, map_location="cpu", weights_only=False)
    x = captured["hf_values"]["input_norm"].contiguous()
    if x.dtype != torch.bfloat16 or x.numel() != 2048:
        raise ValueError(f"Expected one BF16 2048-element norm, got {x.shape}/{x.dtype}")
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    pieces = []
    for suffix in ("qkv", "z", "a", "b"):
        key = f"model.language_model.layers.0.linear_attn.in_proj_{suffix}.weight"
        with safe_open(SNAPSHOT / index[key], framework="pt", device="cpu") as f:
            pieces.append(f.get_tensor(key))
    packed = torch.cat(pieces).contiguous()
    assert packed.dtype == torch.bfloat16 and packed.shape == (12352, 2048)
    ref32 = torch.nn.functional.linear(x.float(), packed.float())
    ref16 = torch.cat([torch.nn.functional.linear(x, w) for w in pieces], dim=-1)
    return x, packed, ref32, ref16


def pair_inputs(x, packed, slots, live=(True, True)):
    # Same sampled products in every placement: move BOTH activation and weight.
    channels = torch.tensor(list(range(16)) + list(range(4096, 4104)) + list(range(8192, 8200)))
    a = torch.zeros(len(PAIRS), 32, 32, dtype=torch.bfloat16)
    b = torch.zeros_like(a)
    products = []
    flat = x.reshape(-1)
    for batch, source_pair in enumerate(PAIRS):
        terms = []
        for term, (source, slot) in enumerate(zip(source_pair, slots)):
            value, weight = flat[source], packed[channels, source]
            terms.append(value.float() * weight.float())
            if live[term]:
                a[batch, 0, slot] = value
                b[batch, slot, :] = weight
        products.append(torch.stack(terms))
    products = torch.stack(products)
    expected = sum((products[:, i] for i in range(2) if live[i]))
    assert torch.equal((a.float() @ b.float())[:, 0, :], expected)
    return a, b, expected, products, channels


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    x, packed, ref32, ref16 = load_inputs(args.input)
    rows = []
    tensors = dict(input_norm=x, reference_float32=ref32, reference_bf16=ref16, pairs=list(PAIRS))
    report = dict(
        input=str(args.input),
        input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
        packed_weights_sha256=hashlib.sha256(packed.view(torch.uint16).numpy().tobytes()).hexdigest(),
        packed_weights_shape=list(packed.shape),
        packed_weights_dtype=str(packed.dtype),
        source_contract="Blackhole MVMUL groups K=16; standard padded32 tiles throughout",
        execution="CPU oracle only" if args.oracle_only else "serial physical chip 0",
        rows=rows,
    )

    def save(row):
        rows.append(row)
        output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save(tensors, str(output) + ".tensors.pt")
        print("MATMUL_PARTIAL_PRECISION", json.dumps(row), flush=True)

    cases = (
        ("pair_same_face", (0, 1), (True, True)),
        ("pair_cross_face", (0, 16), (True, True)),
        ("single_first", (0, 1), (True, False)),
        ("single_second", (0, 1), (False, True)),
    )
    for label, slots, live in cases:
        a, b, expected, products, channels = pair_inputs(x, packed, slots, live)
        tensors[label] = dict(a=a, b=b, expected=expected, products=products, channels=channels)
    if args.oracle_only:
        save(dict(candidate="oracle_pair_placement", verified=True, cases=len(cases), sampled_pairs=len(PAIRS)))
        return

    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        report["compute"] = str(compute)
        report["architecture"] = str(mesh.arch())

        def upload(value):
            return ttnn.from_torch(value.contiguous(), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def matmul(a, b):
            return ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=compute)

        for label, _, _ in cases:
            case = tensors[label]
            ta, tb = upload(case["a"]), upload(case["b"])
            out = matmul(ta, tb)
            got = ttnn.to_torch(out)[:, 0, :].contiguous()
            case["actual"] = got
            save(
                dict(
                    candidate=label,
                    float32=compare(case["expected"], got),
                    per_pair=[compare(case["expected"][i], got[i]) for i in range(len(PAIRS))],
                )
            )
            for tensor in (out, ta, tb):
                ttnn.deallocate(tensor)
        single_sum = tensors["single_first"]["actual"] + tensors["single_second"]["actual"]
        save(
            dict(
                candidate="pair_vs_independent_device_products_cpu_sum",
                same_face=compare(single_sum, tensors["pair_same_face"]["actual"]),
                cross_face=compare(single_sum, tensors["pair_cross_face"]["actual"]),
            )
        )

        tx = upload(x)
        assert torch.equal(ttnn.to_torch(tx).reshape(x.shape), x), "input upload differs"
        for groups in (1, *args.groups):
            total = None
            cpu_total = torch.zeros_like(ref32)
            partials = []
            # Only one masked matrix resides on device at a time. No timing claims:
            # transfers/setup are intentionally outside the precision comparison.
            for residue in range(groups):
                masked = packed.clone()
                masked[:, torch.arange(packed.shape[1]) % groups != residue] = 0
                tw = upload(masked.T)
                assert torch.equal(ttnn.to_torch(tw), masked.T), "masked weight upload differs"
                part = matmul(tx, tw)
                host_part = ttnn.to_torch(part).reshape(ref32.shape)
                cpu_total += host_part
                partials.append(host_part)
                if total is None:
                    total = part
                else:
                    # FP32 operands explicitly select accurate SFPU addition.
                    combined = ttnn.add(total, part)
                    ttnn.deallocate(total)
                    ttnn.deallocate(part)
                    total = combined
                ttnn.deallocate(tw)
            got = ttnn.to_torch(total).reshape(ref32.shape)
            label = "unmasked" if groups == 1 else f"residue_masks_{groups}"
            tensors[label] = dict(actual=got, cpu_combined=cpu_total, partials=partials)
            save(
                dict(
                    candidate=label,
                    groups=groups,
                    float32=compare(ref32, got),
                    canonical_bf16=compare(ref16, got.bfloat16()),
                    cpu_combined_float32=compare(ref32, cpu_total),
                    device_vs_cpu_combination=compare(cpu_total, got),
                    segments={
                        name: compare(ref16[..., lo:hi], got[..., lo:hi].bfloat16()) for name, lo, hi in SEGMENTS
                    },
                )
            )
            ttnn.deallocate(total)
        ttnn.deallocate(tx)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance_exact_frontend_probe.json.pos3979.pt",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups", nargs="+", type=int, choices=(2, 4, 8, 16), default=[2, 4, 8, 16])
    parser.add_argument("--oracle-only", action="store_true")
    arguments = parser.parse_args()
    if len(set(arguments.groups)) != len(arguments.groups):
        parser.error("--groups must not contain duplicates")
    run(arguments)
