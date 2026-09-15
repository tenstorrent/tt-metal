# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""BF16 LoFi native/grid7 resident-input comparison, fixed Q256/K512/D128.

Repeated copies of one K/V chunk give the same exact normalized reference.
The unchanged reader fills two input slots once per invocation; no recurring
input reads occur in the inner loop. Startup reads/final writes remain timed.
This is a throughput microbenchmark, not distinct-KV long-context qualification.
"""

import argparse
import gc
import json
import math
import struct
from pathlib import Path

import torch
import ttnn

import grid7_fullchip as G

HERE, ROOT = G.HERE, G.ROOT
READER = "experiments/sdpa-l2/bfp4-lofi-v2/resident/reader.cpp"
WRITER = "experiments/sdpa-l2/bfp4-lofi-v2/resident/writer.cpp"
COMPUTE = "experiments/sdpa-l2/bfp4-lofi-v2/grid7_streaming/compute_resident.cpp"


def build(device, args, inputs):
    assert args.destination in ("main_bf16", "fast_bf16")
    fast = args.destination == "fast_bf16"
    assert not args.denom_only or fast
    numerator_compensation = fast and not args.denom_only
    kfmt, vfmt = args.kv_formats.split("_")
    originals = [ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT) for x in inputs]
    tensors, calls, checks, input_specs = [], [], [], []
    for i, (src, fmt) in enumerate(zip(originals, ("bf16", kfmt, vfmt))):
        if fmt == "b4":
            tensor, invoke, _ = G.B4_PREP.build(device, src, ncores=1)
            dtype, size = ttnn.bfloat4_b, 576
        else:
            tensor, invoke, _ = G.PREP.build(
                device, src, bits=7 if i == 0 else 5, output_format=fmt, ncores=1, bfp8_pack_precise=False
            )
            dtype, size = (ttnn.bfloat16, 2048) if i == 0 else (ttnn.bfloat8_b, 1088)
        invoke()
        assert tensor.dtype == dtype
        if args.check_preprocess:
            expected = (
                G.PREP.MODEL.round_significand(inputs[i], 7)
                if i == 0
                else (G.B4_PREP.host_rne_bfp4(inputs[i]) if fmt == "b4" else G.ORACLE.native_bfp8_rne5(inputs[i]))
            )
            mismatch = int((ttnn.to_torch(tensor).float() != expected).sum())
            assert mismatch == 0, f"Resident input {i} oracle mismatch: {mismatch}"
            checks.append(dict(input=("Q", "K", "V")[i], format=fmt, mismatch=mismatch))
        tensors.append(tensor)
        calls.append(invoke)
        input_specs.append((dtype, size))
    out = ttnn.allocate_tensor_on_device(
        [1, 1, 256, 128], ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    specs = [
        (0, 64),
        (1, 128),
        (2, 128),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 128),
        (8, 32 * (2 if numerator_compensation else 1)),
        (9, 32 * (2 if numerator_compensation else 1)),
        (10, 8),
        (11, 8),
        (12, 8 * (2 if fast else 1)),
        (13, 8 * (2 if fast else 1)),
        (14, 8),
        (16, 16),
    ]
    specs = [
        (
            index,
            count,
            input_specs[index][1] if index < 3 else 2048,
            input_specs[index][0] if index < 3 else ttnn.bfloat16,
        )
        for index, count in specs
    ]
    total_bytes = sum(count * size for _, count, size, _ in specs)
    assert total_bytes < 1536 * 1024
    cbs = [
        ttnn.CBDescriptor(
            total_size=count * size,
            core_ranges=grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
        )
        for index, count, size, dtype in specs
    ]
    defines = dict(
        EXP_APPROX_MODE="1",
        STATS_GRANULARITY="8",
        SUB_EXP_GRANULARITY="8",
        MUL_BCAST_GRANULARITY="8",
        DHT_GRANULARITY="4",
        REDUCE_GRANULARITY="4",
        SDPA_LOFI_SAFE_RESCALE="1",
    )
    if fast:
        defines.update(SDPA_STREAMING_ACCURACY="1", SDPA_OUT_A_CB="8", SDPA_OUT_B_CB="9", SDPA_LOFI_FIX_CORRECTION="1")
        if numerator_compensation:
            defines["SDPA_STREAMING_NUMERATOR_COMPENSATION"] = "1"
    else:
        defines["RESIDENT_MAIN"] = "1"
    if args.grid7_exp:
        defines["SDPA_LOFI_EXP_GRID7"] = "1"
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [t.buffer_address() for t in tensors]
    writer[0][0] = [out.buffer_address()]
    reader_cta = [args.q_repeats, args.k_chunks, 2]
    for tensor in tensors:
        reader_cta += ttnn.TensorAccessorArgs(tensor).get_compile_time_args()
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=READER,
                core_ranges=grid,
                compile_time_args=reader_cta,
                runtime_args=reader,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=WRITER,
                core_ranges=grid,
                compile_time_args=[args.q_repeats] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=COMPUTE,
                core_ranges=grid,
                compile_time_args=[
                    args.q_repeats,
                    args.k_chunks,
                    struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0],
                ],
                defines=list(defines.items()),
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=False,
                    dst_full_sync_en=False,
                    math_approx_mode=True,
                ),
            ),
        ],
    )

    def attention():
        ttnn.generic_op(tensors + [out], desc)

    def preprocess():
        for invoke in calls:
            invoke()

    def combined():
        preprocess()
        attention()

    combined.buffers = (originals, tensors, out, calls)
    info = dict(
        defines=defines,
        fidelity="LoFi",
        fp32_dst=False,
        q_chunk=256,
        k_chunk=512,
        head_dim=128,
        actual_cores=1,
        input_slots=2,
        cb_bytes_per_core=total_bytes,
        cb_specs=[(i, n, b, str(f)) for i, n, b, f in specs],
        k_format=kfmt,
        v_format=vfmt,
        grid7_exp=args.grid7_exp,
        exp_grid_significant_bits=7 if args.grid7_exp else 9,
        numerator_compensation=numerator_compensation,
        denominator_compensation=fast,
        safe_rescale=True,
        fix_correction=fast,
        preprocessing_checks=checks,
        device_preprocessing=True,
        q_preprocessing="RNE7/BF16",
        kv_preprocessing="B4 native-group RNE or per-value RNE5/native BFP8 RNA, independent K/V",
        input_contract="Original BF16 Q256 and one K/V512 chunk; repeated copies leave ideal attention unchanged",
        dm_scope="Two ring slots initialized once per invocation; no recurring input DRAM reads; final Q output written",
        output_dtype="BF16",
    )
    return originals, tensors, out, attention, preprocess, combined, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--destination", choices=("main_bf16", "fast_bf16"), default="main_bf16")
    parser.add_argument("--denom-only", action="store_true")
    parser.add_argument("--grid7-exp", action="store_true")
    parser.add_argument("--kv-formats", choices=("b8_b8", "b4_b8", "b8_b4", "b4_b4"), default="b8_b8")
    parser.add_argument("--q-repeats", type=int, default=1)
    parser.add_argument("--k-chunks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1240)
    parser.add_argument("--distributions", "--distribution", nargs="+", default=["normal", "constant_v"])
    parser.add_argument("--check-preprocess", action="store_true")
    parser.add_argument("--max-l2", type=float)
    parser.add_argument("--iters", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--trace-repeats", type=int, default=1)
    args = parser.parse_args()
    assert Path(args.label).name == args.label
    assert args.q_repeats > 0 and args.k_chunks > 0
    assert args.iters >= 0 and args.warmup >= 0 and args.trace_repeats > 0
    assert not args.denom_only or args.destination == "fast_bf16"
    torch.set_num_threads(4)
    paths = sorted(set(G.source_files(args.destination) + [Path(__file__).resolve()]))
    pinned = G.hashes(paths)
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
                source_sha256=pinned,
                scope="Resident repeated-input benchmark, NOT distinct-KV context qualification; one core, unchanged Q256/K512/two slots",
            )
        )
        for distribution in args.distributions:
            inputs = G.REPRO.make_inputs(1, 256, 512, 128, args.seed, distribution)
            original_hashes = [G.tensor_hash(x) for x in inputs]
            reference = G.REPRO.reference(*inputs)
            device = ttnn.open_device(device_id=0, trace_region_size=16777216)
            try:
                originals, tensors, out, attention, preprocess, combined, info = build(device, args, inputs)
                combined()
                actual = ttnn.to_torch(out).bfloat16()
                assert bool(torch.isfinite(actual).all()), "Nonfinite output; no timing"
                accuracy = G.REPRO.metrics(actual, reference)
                if args.max_l2 is not None:
                    assert accuracy["l2_pct"] < args.max_l2, "Accuracy gate; no timing"
                G.verify_trace(device, combined, out, actual)
                timings = {
                    name: G.timed(device, call, args)
                    for name, call in (("attention", attention), ("preprocessing", preprocess), ("combined", combined))
                }
                assert torch.equal(actual, ttnn.to_torch(out).bfloat16())
                assert all(torch.equal(ttnn.to_torch(t).bfloat16(), x) for t, x in zip(originals, inputs))
                assert [G.tensor_hash(x) for x in inputs] == original_hashes
                assert G.hashes(paths) == pinned, "Pinned sources changed during run"
                flops = 4 * args.q_repeats * 256 * (args.k_chunks * 512) * 128
                emit(
                    dict(
                        kind="result",
                        distribution=distribution,
                        **vars(args),
                        kernel=info,
                        accuracy=accuracy,
                        centered_output_accuracy=G.centered_output_metrics(actual, reference, inputs[2]),
                        bf16_output_rounding_floor=G.REPRO.metrics(reference.bfloat16(), reference),
                        useful_flops=flops,
                        **timings,
                        tflops_per_core=flops / (timings["attention"]["median_ms"] * 1e9) if args.iters else None,
                        combined_tflops_per_core=(
                            flops / (timings["combined"]["median_ms"] * 1e9) if args.iters else None
                        ),
                        all_output_finite=True,
                        trace_equal=True,
                        sources_unchanged=True,
                        original_input_sha256=original_hashes,
                        output_sha256=G.tensor_hash(actual),
                        accuracy_scope="All256 rows of final repeated Q block, original BF16 reference; duplicates leave exact softmax/PV unchanged",
                        timing_scope="Attention includes one-time resident input fill and final output write; combined also includes actual device quantization",
                    )
                )
            finally:
                ttnn.close_device(device)
            del originals, tensors, out, attention, preprocess, combined
            gc.collect()
        emit(dict(kind="complete", sources_unchanged=G.hashes(paths) == pinned))


if __name__ == "__main__":
    main()
