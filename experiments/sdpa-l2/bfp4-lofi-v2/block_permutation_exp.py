# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exp-only LoFi FP32/B8 whole-KV-block permutation diagnostic; no timing.

Native and macro-LUT share the existing LUT builder; optional cubic uses
native_storage.build(native_exp=False). Original Q7/KV5 preparation, FP32
P/numerator/denominator, BF16 rowmax, chunks and one KV slot stay unchanged.
Native must bit-match recorded original block_permutation outputs separately.
Online rescale still uses the existing accurate exponential in all cases.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
import ttnn

import grid7_fullchip as BF16
import native_storage_fullchip as FP32
import exp_lut_macro_streaming as LUT

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPRO, PREP, ORACLE = BF16.REPRO, BF16.PREP, BF16.ORACLE


def tensor_hash(x):
    assert x.dtype == torch.bfloat16
    return hashlib.sha256(x.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()


def reorder(x, indices):
    assert x.shape[2] % 512 == 0
    blocked = x.reshape(1, 2, x.shape[2] // 512, 512, 128)
    return blocked.index_select(2, indices).reshape_as(x).contiguous()


def orders(k):
    count = k.shape[2] // 512
    # Descending RMS over BOTH heads, all 512 tokens and D. This is a proxy
    # for block magnitude, NOT a query-dependent score maximum or oracle sort.
    rms = k.double().reshape(1, 2, count, 512, 128).square().mean(dim=(0, 1, 3, 4)).sqrt()
    return [
        ("identity", torch.arange(count)),
        ("reverse", torch.arange(count - 1, -1, -1)),
        ("k_rms_descending", torch.argsort(rms, descending=True, stable=True)),
    ], rms.tolist()


def configuration(variant, length, args):
    common = dict(length=length, heads=2, cores=args.cores, check_preprocess=True, read_barrier_tiles=2)
    if variant in ("native", "macro_lut"):
        return argparse.Namespace(
            **common,
            destination="fp32",
            kv_formats="b8_b8",
            lut_exp=variant == "macro_lut",
            raw_lut=False,
        )
    assert variant == "cubic"
    return argparse.Namespace(
        **common,
        variant="lofi_fp32_b8",
        q_chunk=256,
        q_prescale=1.0,
        center_k=False,
        mean_mode="bf16_fpu",
        b8_rne=False,
        bfp8_pack_precise=False,
        fix_correction=False,
        exp_degree=3,
        native_exp=False,
        reader_chain=True,
        reader_split=False,
        reader_linear_k=False,
    )


def native_evidence(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]["kind"] == "complete" and rows[-1]["sources_unchanged"]
    result = {}
    for row in rows:
        if row.get("kind") == "result" and row["variant"] == "lofi_fp32_b8":
            key = (row["length"], row["distribution"], row["order"])
            assert key not in result
            assert row["trace_bitwise_equal"] and row["combined_trace_replays"] == 2
            assert row["all_output_finite"] and row["sources_unchanged"]
            result[key] = row
    return result


def compare_native_record(current_hash, original_hashes, ordered_hashes, prepared, info, old):
    assert original_hashes == old["original_input_sha256"], "Cross-driver original inputs differ"
    assert ordered_hashes == old["ordered_input_sha256"], "Cross-driver ordering differs"
    assert [x["output_sha256"] for x in prepared] == [
        x["output_sha256"] for x in old["preprocessing_checks"]
    ], "Cross-driver represented inputs differ"
    assert info["cb_bytes_per_core"] == old["kernel"]["cb_bytes_per_core"], "Cross-driver CB bytes differ"
    assert info["input_slots"] == old["kernel"]["input_slots"] == 1
    assert current_hash == old["output_sha256"], "Native LUT-off wrapper differs from original native driver"
    return dict(
        original_and_ordered_inputs_identical=True,
        prepared_bits_identical=True,
        cb_bytes_identical=True,
        output_bitwise_identical=True,
        previous_output_sha256=old["output_sha256"],
    )


def source_hashes():
    pins = FP32.source_hashes()
    paths = LUT.source_files("fp32") + [
        Path(__file__).resolve(),
        Path(FP32.__file__).resolve(),
        Path(BF16.__file__).resolve(),
        Path(LUT.__file__).resolve(),
    ]
    pins.update({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return pins


def check_prepared(prepared, expected):
    records = []
    for name, tensor, golden in zip(("Q", "K", "V"), prepared, expected):
        actual = ttnn.to_torch(tensor).bfloat16()
        golden = golden.bfloat16()
        # Native pack canonicalizes zeros. Do not suppress any nonzero mismatch.
        zero_pair = (actual == 0) & (golden == 0)
        different_bits = actual.view(torch.int16) != golden.view(torch.int16)
        nonzero_mismatch = int((different_bits & ~zero_pair).sum())
        zero_sign_disagreements = int((different_bits & zero_pair).sum())
        assert nonzero_mismatch == 0, f"{name} preparation differs from original-order oracle"
        records.append(
            dict(
                input=name,
                nonzero_bit_mismatch=nonzero_mismatch,
                zero_sign_disagreements=zero_sign_disagreements,
                output_sha256=tensor_hash(actual),
                contract="Exact represented values, signed zeros value-equivalent; no nonzero tolerance",
            )
        )
    return records


def check_inputs(device_originals, ordered, ordered_hashes, originals, original_hashes):
    assert [tensor_hash(x) for x in originals] == original_hashes, "Original CPU inputs mutated"
    assert [tensor_hash(x) for x in ordered] == ordered_hashes, "Ordered CPU inputs mutated"
    actual = [tensor_hash(ttnn.to_torch(t)) for t in device_originals]
    assert actual == ordered_hashes, "Original device input bits mutated"
    return dict(
        original_cpu_unchanged=True,
        ordered_cpu_unchanged=True,
        original_device_unchanged=True,
        device_input_sha256=actual,
    )


def interorder(actual, identity):
    a, b = actual.double(), identity.double()
    delta = a - b
    norm = b.norm()
    return dict(
        l2_pct=float(100 * delta.norm() / norm) if norm else None,
        absolute_error_rms=float(delta.square().mean().sqrt()),
        absolute_error_max=float(delta.abs().max()),
        bit_mismatches=int((actual.view(torch.int16) != identity.view(torch.int16)).sum()),
        scope="Every output element, compared with same variant's identity-order device output; no gain alignment",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", nargs="+", type=int, choices=(4096, 32768), default=[4096, 32768])
    parser.add_argument(
        "--exp-modes",
        dest="variants",
        nargs="+",
        choices=("native", "macro_lut", "cubic"),
        default=["native", "macro_lut"],
    )
    parser.add_argument(
        "--native-evidence",
        default=str(HERE / "block-permutation-v1.jsonl"),
        help="Required original native-control evidence; native output must match exactly",
    )
    parser.add_argument("--cores", type=int, default=22)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument(
        "--block-scaled-k", action="store_true", help="Also test repeating per-block K scales .5,1,2,4; V unchanged"
    )
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    assert args.cores >= 2 and args.cores % 2 == 0 and args.sample_rows > 0
    torch.set_num_threads(4)
    pins = source_hashes()
    evidence_path = Path(args.native_evidence)
    evidence_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    previous_native = native_evidence(evidence_path)
    with (HERE / (args.label + ".jsonl")).open("x") as stream:

        def emit(record):
            line = json.dumps(record, allow_nan=False)
            stream.write(line + "\n")
            stream.flush()
            print(line, flush=True)

        emit(
            dict(
                kind="provenance",
                args=vars(args),
                source_sha256=pins,
                scope="Accuracy only; H2 D128 Q256/K512; complete joint512token KV block permutations",
                max_sorted_definition="Stable descending K block RMS over both heads/tokens/features, NOT actual query-score maxima",
                reference_rtol=1e-11,
                reference_atol=1e-12,
                native_evidence_file=str(evidence_path),
                native_evidence_sha256=evidence_hash,
                exp_scope="Native vs macro-LUT; optional existing cubic. Accurate online rescale unchanged; isolates approximation family, not a proof of causality",
                source_scope="Principal selected experiment and API dependencies, not full compiler/firmware closure",
            )
        )
        count = 0
        for length in args.lengths:
            distributions = ["normal", "block_scaled_k"] if args.block_scaled_k else ["normal"]
            for distribution in distributions:
                originals = REPRO.make_inputs(2, length, length, 128, args.seed, "normal")
                scales = None
                if distribution == "block_scaled_k":
                    scales = torch.tensor([0.5, 1.0, 2.0, 4.0]).repeat((length // 512 + 3) // 4)[: length // 512]
                    originals[1] = (
                        (originals[1].reshape(1, 2, length // 512, 512, 128).float() * scales.reshape(1, 1, -1, 1, 1))
                        .reshape_as(originals[1])
                        .bfloat16()
                    )
                original_hashes = [tensor_hash(x) for x in originals]
                rows = torch.linspace(0, length - 1, min(args.sample_rows, length)).long().unique()
                reference = REPRO.reference(originals[0][..., rows, :], originals[1], originals[2])
                expected_original = [PREP.MODEL.round_significand(originals[0], 7)] + [
                    ORACLE.native_bfp8_rne5(x) for x in originals[1:]
                ]
                order_list, block_rms = orders(originals[1])
                # Compute each ordered FP64 reference once, before opening a device.
                agreement = {}
                for name, indices in order_list:
                    assert sorted(indices.tolist()) == list(range(length // 512))
                    ordered_ref = REPRO.reference(
                        originals[0][..., rows, :], reorder(originals[1], indices), reorder(originals[2], indices)
                    )
                    assert torch.allclose(
                        ordered_ref, reference, rtol=1e-11, atol=1e-12
                    ), "FP64 permutation invariance failed"
                    agreement[name] = dict(
                        rtol=1e-11,
                        atol=1e-12,
                        passed=True,
                        max_abs=float((ordered_ref - reference).abs().max()),
                        l2_pct=float(100 * (ordered_ref - reference).norm() / reference.norm()),
                    )
                for name, _ in order_list:
                    assert (length, distribution, name) in previous_native, "Missing original native-control row"
                for variant in args.variants:
                    identity_output = None
                    for order_name, indices in order_list:
                        ordered = [originals[0], reorder(originals[1], indices), reorder(originals[2], indices)]
                        ordered_hashes = [tensor_hash(x) for x in ordered]
                        expected = [
                            expected_original[0],
                            reorder(expected_original[1], indices),
                            reorder(expected_original[2], indices),
                        ]
                        config = configuration(variant, length, args)
                        device = ttnn.open_device(device_id=0, trace_region_size=16777216)
                        try:
                            builder = FP32 if variant == "cubic" else LUT
                            dev_inputs, prepared, out, attention, preprocess, combined, info = builder.build(
                                device, config, ordered
                            )
                            combined()
                            actual = ttnn.to_torch(out).bfloat16()
                            assert bool(torch.isfinite(actual).all()), "Nonfinite output"
                            prep_check = check_prepared(prepared, expected)
                            before = check_inputs(dev_inputs, ordered, ordered_hashes, originals, original_hashes)
                            output_hash = tensor_hash(actual)
                            native_check = None
                            if variant == "native":
                                native_check = compare_native_record(
                                    output_hash,
                                    original_hashes,
                                    ordered_hashes,
                                    prep_check,
                                    info,
                                    previous_native[(length, distribution, order_name)],
                                )
                            trace = ttnn.begin_trace_capture(device, cq_id=0)
                            combined()
                            ttnn.end_trace_capture(device, trace, cq_id=0)
                            replay_hashes = []
                            try:
                                for _ in range(2):
                                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                                    replay_hashes.append(tensor_hash(ttnn.to_torch(out)))
                                    assert replay_hashes[-1] == output_hash, "Combined replay changed output bits"
                            finally:
                                ttnn.release_trace(device, trace)
                            after = check_inputs(dev_inputs, ordered, ordered_hashes, originals, original_hashes)
                            if identity_output is None:
                                assert order_name == "identity"
                                identity_output = actual.clone()
                            assert source_hashes() == pins, "Selected sources changed during run"
                            emit(
                                dict(
                                    kind="result",
                                    variant=variant,
                                    distribution=distribution,
                                    length=length,
                                    order=order_name,
                                    block_indices=indices.tolist(),
                                    k_block_rms=block_rms,
                                    block_k_scales=scales.tolist() if scales is not None else None,
                                    original_input_sha256=original_hashes,
                                    ordered_input_sha256=ordered_hashes,
                                    reference_agreement=agreement[order_name],
                                    config=vars(config),
                                    kernel=info,
                                    sampled_query_rows=rows.tolist(),
                                    accuracy=REPRO.metrics(actual[..., rows, :], reference),
                                    interorder_vs_identity=interorder(actual, identity_output),
                                    original_reference_scope="Original unpermuted BF16 Q/K/V; all heads and KV, explicit sampled Q rows",
                                    preprocessing_checks=prep_check,
                                    all_output_finite=True,
                                    native_cross_driver_check=native_check,
                                    combined_trace_replays=2,
                                    replay_output_sha256=replay_hashes,
                                    trace_bitwise_equal=True,
                                    input_immutability_before=before,
                                    input_immutability_after=after,
                                    output_sha256=output_hash,
                                    sources_unchanged=True,
                                    timing_performed=False,
                                )
                            )
                            count += 1
                        finally:
                            ttnn.close_device(device)
                        del dev_inputs, prepared, out, attention, preprocess, combined, actual, ordered, expected
                        gc.collect()
                    del identity_output
                assert [tensor_hash(x) for x in originals] == original_hashes
        assert count == len(args.lengths) * (2 if args.block_scaled_k else 1) * len(args.variants) * 3
        assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == evidence_hash, "Native evidence changed"
        emit(
            dict(
                kind="complete", cases=count, sources_unchanged=source_hashes() == pins, native_evidence_unchanged=True
            )
        )


if __name__ == "__main__":
    main()
