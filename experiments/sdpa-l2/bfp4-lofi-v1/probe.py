# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device packer and operand-fidelity probes; no SDPA dispatch changes."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent


def metrics(actual, reference):
    a, r = actual.double().flatten(), reference.double().flatten()
    delta = a - r
    ac, rc = a - a.mean(), r - r.mean()
    denom = ac.norm() * rc.norm()
    return dict(
        finite=bool(torch.isfinite(a).all()),
        l2_pct=float(100 * delta.norm() / r.norm()) if r.norm() else None,
        pcc=float(ac.dot(rc) / denom) if denom else None,
        max_abs=float(delta.abs().max()),
        unequal=int((a != r).sum()),
    )


def quantize(x, bits, mode):
    """Finite, ordinary-range model with native row-wise groups of 16.

    Candidate models only: device probes select which is applicable. This does
    not model subnormals, exceptional values, or near-overflow exponent groups.
    """
    assert x.shape[-1] % 16 == 0 and torch.isfinite(x).all()
    groups = x.float().reshape(-1, 16)
    if mode == "host":
        # Match blockfloat_common.cpp, including mantissa alignment before RNE.
        # Alignment discards low sticky bits; ideal real-valued RNE can differ
        # at rare thresholds for transformed FP32 inputs.
        raw = groups.abs().contiguous().view(torch.int32).long()
        exponents = (raw >> 23) & 255
        shared = exponents.amax(-1, keepdim=True)
        mantissa = torch.where(exponents == 0, 0, (raw & 0x7FFFFF) | 0x800000)
        aligned = mantissa >> (shared - exponents).clamp_max(63)
        shift = 24 - bits
        integer = aligned >> shift
        remainder = aligned & (2**shift - 1)
        tie = 2 ** (shift - 1)
        integer += ((remainder > tie) | ((remainder == tie) & ((integer & 1) != 0))).long()
        values = integer.clamp_max(2**bits - 1).float() * torch.exp2((shared - 127 - (bits - 1)).float())
        return (values * groups.sign()).reshape(x.shape)
    if mode == "device":
        if bits == 3:
            # Current BFP4 typecast configuration first rounds per datum to
            # E8M6, then rounds to shared-exponent BFP8, then truncates to BFP4.
            # The first rounding can increase the group's maximum exponent.
            per_datum_step = torch.exp2(torch.frexp(groups.abs().clamp_min(2.0**-100))[1].float() - 7)
            groups = (groups.abs().double() / per_datum_step + 0.5).floor().float() * per_datum_step * groups.sign()
            mode = "rna8_trunc"
        elif bits == 7:
            mode = "rna"
        else:
            raise ValueError("Device model validated only for BFP4/BFP8")
    magnitude = groups.abs()
    exponent = torch.frexp(magnitude.amax(-1, keepdim=True).clamp_min(2.0**-100))[1].float() - 1
    step = torch.exp2(exponent - (bits - 1))
    scaled = magnitude / step
    if mode == "rne":
        integer = scaled.round()
    elif mode == "rna":
        integer = (scaled.double() + 0.5).floor().float()
    elif mode == "trunc":
        integer = scaled.floor()
    elif mode in ("rna8_trunc", "rne8_trunc"):
        intermediate = scaled * (2 ** (7 - bits))
        intermediate = (intermediate.double() + 0.5).floor().float() if mode == "rna8_trunc" else intermediate.round()
        integer = intermediate.clamp_max(127).div(2 ** (7 - bits)).floor()
    else:
        raise ValueError(mode)
    return (integer.clamp_max(2**bits - 1) * step * groups.sign()).reshape(x.shape)


def upload(x, dtype, device):
    return ttnn.from_torch(x.float(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def unpack(x):
    return ttnn.to_torch(x).float()


def pack_probe(device, emit):
    # Sweep thresholds densely, keeping each exponent group anchored in [1,2).
    values = torch.arange(1024).float() / 512
    groups = values[:, None].repeat(1, 16)
    groups[:, 0] = 1.75
    boundary = torch.cat([groups * sign * 2.0**e for e in (-8, 0, 8) for sign in (-1, 1)]).reshape(-1, 32)
    generator = torch.Generator().manual_seed(20260915)
    random = torch.randn((512, 128), generator=generator)
    random *= torch.exp2(torch.randint(-8, 9, (512, 1), generator=generator).float())
    for name, original in (("thresholds", boundary), ("random", random)):
        for source in ("bf16", "fp32"):
            x = original.bfloat16().float() if source == "bf16" else original
            src = upload(x, ttnn.bfloat16 if source == "bf16" else ttnn.float32, device)
            for bits, dtype in ((3, ttnn.bfloat4_b), (7, ttnn.bfloat8_b)):
                host = upload(x, dtype, device)
                packed = ttnn.typecast(src, dtype)
                host_values, device_values = unpack(host), unpack(packed)
                torch.save(
                    dict(input=x, host=host_values, device=device_values), HERE / f"pack-{name}-{source}-{bits}.pt"
                )
                candidates = {
                    mode: metrics(device_values, quantize(x, bits, mode))
                    for mode in ("rne", "rna", "trunc", "rna8_trunc", "rne8_trunc", "device")
                }
                mismatch = (device_values != host_values).flatten().nonzero().flatten()[:12]
                emit(
                    dict(
                        kind="pack",
                        case=name,
                        source=source,
                        magnitude_bits=bits,
                        device_vs_host=metrics(device_values, host_values),
                        host_vs_rne=metrics(host_values, quantize(x, bits, "rne")),
                        host_vs_model=metrics(host_values, quantize(x, bits, "host")),
                        candidates=candidates,
                        examples=[
                            dict(
                                input=float(x.flatten()[i]),
                                host=float(host_values.flatten()[i]),
                                device=float(device_values.flatten()[i]),
                            )
                            for i in mismatch
                        ],
                    )
                )
                ttnn.deallocate(host)
                ttnn.deallocate(packed)
            ttnn.deallocate(src)


def matmul_probe(device, emit):
    generator = torch.Generator().manual_seed(20260916)
    formats = {"4": ttnn.bfloat4_b, "8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}
    for geometry in ("single_product", "qk", "pv"):
        reduction = 512 if geometry == "pv" else 128
        a = torch.randn((64, reduction), generator=generator).bfloat16().float()
        b = (
            torch.randn((64, reduction) if geometry == "qk" else (reduction, 64), generator=generator)
            .bfloat16()
            .float()
        )
        if geometry == "single_product":
            a[:, 1:] = 0
            b[1:, :] = 0
        for left, right in (("4", "4"), ("8", "4"), ("4", "8"), ("8", "8"), ("bf16", "bf16")):
            ta, tb = upload(a, formats[left], device), upload(b, formats[right], device)
            da, db = unpack(ta).double(), unpack(tb).double()
            ref = da @ (db.T if geometry == "qk" else db)
            outputs = {}
            for fidelity in ("LoFi", "HiFi4"):
                config = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=getattr(ttnn.MathFidelity, fidelity),
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                )
                out = ttnn.matmul(
                    ta,
                    tb,
                    transpose_b=geometry == "qk",
                    dtype=ttnn.float32,
                    compute_kernel_config=config,
                    core_grid=ttnn.CoreGrid(y=1, x=1),
                )
                outputs[fidelity] = unpack(out)
                emit(
                    dict(
                        kind="matmul",
                        geometry=geometry,
                        left=left,
                        right=right,
                        fidelity=fidelity,
                        metrics=metrics(outputs[fidelity], ref),
                    )
                )
                ttnn.deallocate(out)
            emit(
                dict(
                    kind="fidelity_difference",
                    geometry=geometry,
                    left=left,
                    right=right,
                    metrics=metrics(outputs["LoFi"], outputs["HiFi4"]),
                )
            )
            ttnn.deallocate(ta)
            ttnn.deallocate(tb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--only", choices=("pack", "matmul", "all"), default="all")
    args = parser.parse_args()
    torch.set_num_threads(8)
    path = HERE / (args.label + ".jsonl")
    assert not path.exists(), "Use a fresh label"
    with path.open("x") as output:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                hostname=platform.node(),
                args=vars(args),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            )
        )
        device = ttnn.open_device(device_id=0)
        started = time.monotonic()
        try:
            if args.only in ("pack", "all"):
                pack_probe(device, emit)
            if args.only in ("matmul", "all"):
                matmul_probe(device, emit)
        finally:
            ttnn.close_device(device)
        emit(dict(kind="completed", seconds=time.monotonic() - started))


if __name__ == "__main__":
    main()
