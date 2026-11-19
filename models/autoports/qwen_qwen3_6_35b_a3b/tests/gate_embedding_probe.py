# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Encoded gate LUT diagnostic; CPU-only setup with --oracle-only.

Device mode owns physical chip 0 serially. It checks nine canonical captures,
then traces the entire input-bitcast/embedding/byte-selection/reassembly adapter
at S=1 and S=128. No decoder modification or acceptance waiver.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import REVISION, SNAPSHOT

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder"
POSITIONS = (128, 129, 130, 1024, 3979, 5011, 7551, 7703, 8157)


def numeric(x):
    x = float(x)
    return x if torch.isfinite(torch.tensor(x, dtype=torch.float64)) else str(x)


def compare(expected, actual, limit=8):
    expected, actual = expected.float().contiguous().reshape(-1), actual.float().contiguous().reshape(-1)
    assert expected.shape == actual.shape
    ec, ac = expected.view(torch.int32), actual.view(torch.int32)
    different = ec != ac
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    error = (expected[finite].double() - actual[finite].double()).abs()
    return dict(
        elements=expected.numel(),
        exact_fp32_code_differences=int(different.sum()),
        finite_numeric_differences=int((finite & (expected != actual)).sum()),
        finite_pairs=int(finite.sum()),
        excluded_nonfinite_pairs=int((~finite).sum()),
        expected_nan=int(torch.isnan(expected).sum()),
        actual_nan=int(torch.isnan(actual).sum()),
        expected_inf=int(torch.isinf(expected).sum()),
        actual_inf=int(torch.isinf(actual).sum()),
        max_absolute_error=float(error.max()) if error.numel() else None,
        first_differences=[
            dict(
                index=i,
                expected=numeric(expected[i]),
                actual=numeric(actual[i]),
                expected_code=f"0x{int(ec[i]) & 0xffffffff:08x}",
                actual_code=f"0x{int(ac[i]) & 0xffffffff:08x}",
            )
            for i in different.nonzero().flatten()[:limit].tolist()
        ],
    )


def load_tables():
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name in ("A_log", "dt_bias"):
        key = f"model.language_model.layers.0.linear_attn.{name}"
        with safe_open(SNAPSHOT / index[key], framework="pt", device="cpu") as f:
            weights[name] = f.get_tensor(key).float()
    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    scale = -weights["A_log"].exp().reshape(1, 32)
    bias = weights["dt_bias"].reshape(1, 32)
    gate = scale * F.softplus(raw.float().reshape(65536, 1) + bias)
    return raw, scale, bias, dict(gate=gate, decay=gate.exp())


def decode_selected(parts, encoding):
    if encoding == "scaled_bf16":
        parts = parts.float()
        return parts[..., 0] + (parts[..., 1] + parts[..., 2]) * parts[..., 3]
    parts = parts.to(torch.int64)
    bits = parts[..., 0] | (parts[..., 1] << 8) | (parts[..., 2] << 16) | (parts[..., 3] << 24)
    return bits.to(torch.int32).contiguous().view(torch.float32)


def encode(values, encoding="bytes"):
    if encoding == "scaled_bf16":
        # Avoid native UINT32->FP32 bitcast, whose current input-unpack path narrows
        # the reinterpreted Float32 DFB through SrcA. Preserve all FP32 bits using
        # high + (middle + low) * inverse_scale instead. Scaling tiny residuals
        # keeps even FP32-subnormal low bits representable as BF16 components.
        def truncate(x):
            return x.contiguous().view(torch.int32).bitwise_and(-65536).view(torch.float32)

        high = truncate(values)
        scale = torch.where(values.abs() < 2.0**-100, 2.0**32, 1.0)
        residual = torch.where(torch.isfinite(values), values - high, torch.zeros_like(values)) * scale
        middle = truncate(residual)
        low = residual - middle
        parts = [high.bfloat16(), middle.bfloat16(), low.bfloat16()]
        # Gate contains negative zeros: all zero addends must retain its sign.
        for i in (1, 2):
            parts[i] = torch.where(parts[i] == 0, torch.copysign(parts[i], values), parts[i]).bfloat16()
        parts.append(scale.reciprocal().bfloat16())
        return torch.stack(parts, dim=-1).reshape(65536, 128)
    # Arithmetic bit extraction is endian independent. Every byte is exactly BF16-representable.
    bits = values.contiguous().view(torch.int32).to(torch.int64).bitwise_and(0xFFFFFFFF)
    shifts = torch.tensor([0, 8, 16, 24], dtype=torch.int64)
    return ((bits[..., None] >> shifts) & 255).reshape(65536, 128).bfloat16()


def decode_cpu(encoded, encoding="bytes"):
    return decode_selected(encoded.reshape(65536, 32, 4), encoding)


def make_adapter(ttnn, mesh, encoded_tables, lengths, encoding):
    def upload(x, dtype, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(
            x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tables = {
        name: upload(value.reshape(1, 1, 65536, 128), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        for name, value in encoded_tables.items()
    }
    columns = {}
    for length in lengths:
        head_columns = torch.arange(128, dtype=torch.int32).reshape(32, 4).repeat(length, 1)
        columns[length] = upload(head_columns.reshape(1, length * 32, 4), ttnn.uint32)

    def adapter(a, name, intermediate=None):
        # All tables/index columns are setup constants. No Torch, uploads, or host lookup here.
        length = a.shape[1]
        indices = ttnn.typecast(ttnn.bitcast(a, ttnn.uint16), ttnn.uint32)
        indices = ttnn.to_layout(ttnn.reshape(indices, (1, length * 32)), ttnn.ROW_MAJOR_LAYOUT)
        embedded = ttnn.embedding(indices, tables[name], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        selected = ttnn.gather(embedded, -1, columns[length])
        if encoding == "bytes":
            byte_values = ttnn.typecast(selected, ttnn.uint32)
            word = byte_values[:, :, :1]
            for byte in range(1, 4):
                shifted = ttnn.bitwise_left_shift(byte_values[:, :, byte : byte + 1], byte * 8)
                word = ttnn.bitwise_or(word, shifted)
            value = ttnn.bitcast(word, ttnn.float32)
            if intermediate is not None:
                intermediate.update(numeric_bytes=byte_values, assembled_word=word, after_bitcast=value)
        else:
            components = ttnn.typecast(selected, ttnn.float32)
            residual = ttnn.add(components[:, :, 1:2], components[:, :, 2:3])
            residual = ttnn.multiply(residual, components[:, :, 3:4])
            value = ttnn.add(components[:, :, :1], residual)
        result = ttnn.reshape(value, (1, length, 32))
        if intermediate is not None:
            intermediate.update(selected=selected, before_reshape=value, final=result)
        return result

    return adapter, upload


def trace_benchmark(ttnn, mesh, fn, expected, args):
    from ttnn.tools import trace_allocation_tracker

    for _ in range(args.warmup):
        warm = fn()
        ttnn.synchronize_device(mesh)
        del warm
    trace_id, trace_output, capture_open = None, None, False
    try:
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        capture_open = True
        trace_output = fn()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        capture_open = False
        trace_allocation_tracker.acknowledge_corruptible(trace_output)
        for _ in range(args.warmup):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        first = ttnn.to_torch(trace_output).clone()
        samples = []
        for _ in range(args.iterations):
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            for _ in range(args.trace_replays):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - begin) * 1000 / args.trace_replays)
        last = ttnn.to_torch(trace_output).clone()
        return dict(
            median_ms=statistics.median(samples),
            minimum_ms=min(samples),
            samples_ms=samples,
            first_vs_cpu=compare(expected, first),
            last_vs_cpu=compare(expected, last),
            first_vs_last=compare(first, last),
            replays_per_sample=args.trace_replays,
            scope="Complete adapter: BF16 input through FP32 output; same input each replay; no host conversion/upload in timing",
        )
    finally:
        if trace_id is not None:
            try:
                if capture_open:
                    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                ttnn.synchronize_device(mesh)
            finally:
                ttnn.release_trace(mesh, trace_id)
        del trace_output


def main(args):
    if any(v < 1 for v in (args.warmup, args.iterations, args.trace_replays)):
        raise ValueError("warmup/iterations/trace-replays must be positive")
    if args.trace_region_size <= 0 or args.trace_region_size % 32:
        raise ValueError("trace-region-size must be a positive multiple of 32")
    torch.set_num_threads(8)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Preserving {output}; choose another output or --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    raw, scale, bias, tables = load_tables()
    encoded = {name: encode(value, args.encoding) for name, value in tables.items()}
    finite = torch.isfinite(raw)
    result = dict(
        classification="diagnostic only; no acceptance waiver",
        status="in_progress",
        execution="CPU-only" if args.oracle_only else "serial physical chip 0",
        checkpoint_revision=REVISION,
        finite_bf16_codes=int(finite.sum()),
        heads=32,
        encoded_table_shape=[65536, 128],
        bytes_per_encoded_table=65536 * 128 * 2,
        original_fp32_bytes_per_table=65536 * 32 * 4,
        encoding=args.encoding,
        setup_encoding=(
            "Four exact numeric bytes, least significant first"
            if args.encoding == "bytes"
            else "Four BF16 components: high + (middle + low) * inverse_scale; tiny residuals scaled by 2**32"
        ),
        time_inexact_diagnostic=args.time_inexact_diagnostic,
        cpu_all_finite_codes={},
        cpu_adapter_shapes=[],
        captures=[],
        device_captures=[],
        device_trace=[],
        device_domain="Nine canonical captures plus S=1/S=128 trace inputs; exhaustive finite-code verification is CPU-only",
    )

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    for name, table in tables.items():
        reconstructed = decode_cpu(encoded[name], args.encoding)
        result["cpu_all_finite_codes"][name] = compare(table[finite], reconstructed[finite])
        assert result["cpu_all_finite_codes"][name]["exact_fp32_code_differences"] == 0

    captures = []
    for position in POSITIONS:
        path = Path(f"{args.capture_prefix}.pos{position}.pt")
        captured = torch.load(path, map_location="cpu", weights_only=False)
        a = captured["hf_values"]["projection_a"].reshape(1, 1, 32).bfloat16()
        gate = scale * F.softplus(a.float() + bias)
        expected = dict(gate=gate, decay=gate.exp())
        indices = a.contiguous().view(torch.int16).long() & 0xFFFF
        row = dict(position=position, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cpu={})
        for name, values in expected.items():
            got = tables[name][indices.reshape(32), torch.arange(32)].reshape(1, 1, 32)
            row["cpu"][name] = compare(values, got)
            assert row["cpu"][name]["exact_fp32_code_differences"] == 0
        captures.append((position, a, expected))
        result["captures"].append(row)
    pool = torch.cat([a for _, a, _ in captures], dim=1)
    for length in (1, 128):
        a = pool.repeat(1, (length + pool.shape[1] - 1) // pool.shape[1], 1)[:, :length, :].contiguous()
        indices = (a.view(torch.int16).long() & 0xFFFF).reshape(-1)
        columns = torch.arange(128).reshape(32, 4).repeat(length, 1)
        gate = scale * F.softplus(a.float() + bias)
        for name, table in encoded.items():
            selected = table[indices].gather(-1, columns)
            got = decode_selected(selected, args.encoding).reshape(1, length, 32)
            row = dict(
                sequence_length=length, kind=name, comparison=compare(gate if name == "gate" else gate.exp(), got)
            )
            assert row["comparison"]["exact_fp32_code_differences"] == 0
            result["cpu_adapter_shapes"].append(row)
    save()
    if args.oracle_only:
        result["status"] = "complete"
        save()
        print(json.dumps(dict(output=str(output), cpu_finite_codes_per_head=int(finite.sum()), captures=len(captures))))
        return

    import ttnn

    lengths = [1] if args.decode_only else [1, 128]
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=args.trace_region_size
    )
    try:
        adapter, upload = make_adapter(ttnn, mesh, encoded, lengths, args.encoding)
        for position, a, expected in captures:
            ta = upload(a, ttnn.bfloat16)
            row = dict(position=position)
            for name in tables:
                intermediate = {} if args.capture_intermediates and position == POSITIONS[0] else None
                got = adapter(ta, name, intermediate)
                row[name] = compare(expected[name], ttnn.to_torch(got))
                if intermediate is not None:
                    host = {key: ttnn.to_torch(value).clone() for key, value in intermediate.items()}
                    row.setdefault("intermediates", {})[name] = {
                        key: compare(expected[name], host[key]) for key in ("before_reshape", "final")
                    }
                    idx = (a.contiguous().view(torch.int16).long() & 0xFFFF).reshape(32)
                    selected_expected = encoded[name][idx].gather(-1, torch.arange(128).reshape(32, 4))
                    row["intermediates"][name]["selected"] = compare(selected_expected, host["selected"])
                    if args.encoding == "bytes":
                        expected_word = (
                            expected[name].contiguous().view(torch.int32).long().bitwise_and(0xFFFFFFFF).reshape(-1)
                        )
                        actual_word = host["assembled_word"].long().bitwise_and(0xFFFFFFFF).reshape(-1)
                        row["intermediates"][name]["assembled_word"] = dict(
                            exact_uint32_code_differences=int((expected_word != actual_word).sum()),
                            expected_codes=[f"0x{int(v):08x}" for v in expected_word],
                            actual_codes=[f"0x{int(v):08x}" for v in actual_word],
                        )
                        row["intermediates"][name]["numeric_bytes"] = compare(
                            selected_expected, host["numeric_bytes"].float()
                        )
                        row["intermediates"][name]["after_bitcast"] = compare(expected[name], host["after_bitcast"])
                    torch.save(host, str(output) + f".pos{position}.{name}.intermediates.pt")
            result["device_captures"].append(row)
            save()
        # A bounded S=128 layout check mixes every captured token across heads/tokens.
        # It tests adapter sequence geometry, not a full prefill recurrence.
        captures_exact = all(
            row[name]["exact_fp32_code_differences"] == 0 for row in result["device_captures"] for name in tables
        )
        trace_lengths = lengths if (captures_exact or args.time_inexact_diagnostic) and not args.skip_trace else []
        result["timing_after_inexact_captures"] = bool(trace_lengths and not captures_exact)
        result["timing_acceptance_note"] = (
            "Diagnostic timing only. Microprobe mismatches remain reported and device_exact remains false; "
            "no model PCC/acceptance conclusion follows from this probe."
        )
        result["trace_skipped_reason"] = (
            "explicit --skip-trace"
            if args.skip_trace
            else None
            if trace_lengths
            else "canonical capture mismatch; use --time-inexact-diagnostic to measure without changing correctness results"
        )
        for length in trace_lengths:
            a = pool.repeat(1, (length + pool.shape[1] - 1) // pool.shape[1], 1)[:, :length, :].contiguous()
            ta = upload(a, ttnn.bfloat16)
            gate = scale * F.softplus(a.float() + bias)
            for name in args.trace_kinds:
                expected = gate if name == "gate" else gate.exp()
                eager = adapter(ta, name)
                row = dict(sequence_length=length, kind=name, eager_vs_cpu=compare(expected, ttnn.to_torch(eager)))
                row["trace"] = trace_benchmark(ttnn, mesh, lambda: adapter(ta, name), expected, args)
                result["device_trace"].append(row)
                save()
                print(
                    json.dumps(
                        dict(
                            sequence_length=length,
                            kind=name,
                            median_ms=row["trace"]["median_ms"],
                            code_differences=row["trace"]["last_vs_cpu"]["exact_fp32_code_differences"],
                        )
                    ),
                    flush=True,
                )
    finally:
        ttnn.close_mesh_device(mesh)
    result["status"] = "complete"
    result["device_exact"] = all(
        row[name]["exact_fp32_code_differences"] == 0 for row in result["device_captures"] for name in tables
    )
    result["device_exact"] &= all(
        row["eager_vs_cpu"]["exact_fp32_code_differences"] == 0
        and all(
            row["trace"][key]["exact_fp32_code_differences"] == 0
            for key in ("first_vs_cpu", "last_vs_cpu", "first_vs_last")
        )
        for row in result["device_trace"]
    )
    save()
    print(json.dumps(dict(output=str(output), device_exact=result["device_exact"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--encoding", choices=("bytes", "scaled_bf16"), default="scaled_bf16")
    parser.add_argument(
        "--capture-intermediates", action="store_true", help="Save first capture stages after adapter completion"
    )
    parser.add_argument("--skip-trace", action="store_true")
    parser.add_argument(
        "--time-inexact-diagnostic",
        action="store_true",
        help="Time despite capture mismatches; retain all errors and device_exact=false, with no model acceptance claim",
    )
    parser.add_argument("--capture-prefix", default=str(DOC / "endurance_precise_probe.json"))
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--trace-kinds", nargs="+", choices=("gate", "decay"), default=["decay", "gate"])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trace-replays", type=int, default=20)
    parser.add_argument("--trace-region-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to a separate inexact-timing filename when --time-inexact-diagnostic is used",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        filename = (
            "gate_embedding_probe_inexact_timing.json"
            if args.time_inexact_diagnostic
            else "gate_embedding_probe_repaired.json"
        )
        args.output = str(DOC / "endurance" / filename)
    main(args)
