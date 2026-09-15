# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident identity4 off/on bitwise comparison; native FP32 LoFi streaming.
Repeated KV dominates identity branch; use identity4_streaming.py for distinct KV.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import struct
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = "experiments/sdpa-l2/bfp4-lofi-v2/identity4_streaming/"
import identity4_streaming as DRIVER

v2_spec = importlib.util.spec_from_file_location("lofi_model", HERE / "numerics.py")
V2 = importlib.util.module_from_spec(v2_spec)
v2_spec.loader.exec_module(V2)
# Private Q7/BF16, RNE5/BFP8 K/V point; do not mutate the frozen source file.
V2.VARIANTS['identity_q7b8'] = ('e7', ('e5_b8',), 'fp32', ('e5_b8',))


def build(device, args):
    assert not args.distinct_kv, "Use identity4_streaming.py for distinct-KV qualification"
    assert args.destination == 'fp32' and args.native_exp and args.fidelity == 'LoFi'
    assert args.variant == 'identity_q7b8' and args.exp_quality == 'cheap'
    accurate = args.mode in (
        "accurate",
        "fp32_hifi2",
        "fp32_hifi2_cheap",
        "qk4_pv2",
        "qk4_pv2_fullsub",
        "qk2_pv2_fullsub",
    )
    fast = args.mode == "fast"
    numerator_compensation = fast and not args.denom_only
    assert not args.denom_only or (fast and args.fix_correction)
    hybrid = args.mode == "hybrid"
    torch.manual_seed(args.seed)
    q = torch.randn(1, 1, 256, 128).bfloat16()
    k = torch.randn(1, 1, 512, 128).bfloat16()
    v = torch.randn(1, 1, 512, 128).bfloat16()
    if args.distribution == "constant_v":
        v.fill_(1)
    # Duplicate resident K/V chunks leave the exact normalized attention unchanged.
    ref = torch.softmax(q.double() @ k.double().transpose(-2, -1) / math.sqrt(128), dim=-1) @ v.double()
    if args.distinct_kv:
        spec = importlib.util.spec_from_file_location(
            "sdpa_repro", ROOT / "tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py"
        )
        repro = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(repro)
        torch.set_num_threads(8)
        q, k, v = repro.make_inputs(1, 256, 512 * args.k_chunks, 128, args.seed, args.distribution)
        ref = repro.reference(q, k, v)
    qfmt, kfmts, pfmt, vfmts = V2.VARIANTS[args.variant]
    residual = len(kfmts) == 2
    assert len(kfmts) == len(vfmts) and pfmt in ("fp32", "e7")
    assert not residual or accurate
    formats = (qfmt, kfmts[0], vfmts[0])
    prepared = (q, k, v)
    if residual:
        assert args.exp_quality == "accurate" or args.no_p_round
        kc = V2.components(k, kfmts, args.qkv_route)
        vc = V2.components(v, vfmts, args.qkv_route)
        prepared = (q, kc[0], vc[0], kc[1], vc[1])
        formats += (kfmts[1], vfmts[1])
    tensors, input_specs = [], []
    for x, fmt in zip(prepared, formats):
        if fmt in ("e7", "e5", "e5_b8"):
            x = V2.round_significand(x, 7 if fmt == "e7" else 5).bfloat16()
        if fmt == "b4":
            dtype, tile_bytes = ttnn.bfloat4_b, 576
        elif fmt in ("b8", "e5_b8"):
            dtype, tile_bytes = ttnn.bfloat8_b, 1088
        else:
            dtype, tile_bytes = ttnn.bfloat16, 2048
        tensors.append(ttnn.from_torch(x.float(), dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT))
        input_specs.append((dtype, tile_bytes))
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 256, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    kv_slots = 1 if accurate else 2
    fp = ttnn.float32 if accurate else ttnn.bfloat16
    page = 4096 if accurate else 2048
    specs = [
        (0, 64, 2048, ttnn.bfloat16),
        (1, 64 * kv_slots, 2048, ttnn.bfloat16),
        (2, 64 * kv_slots, 2048, ttnn.bfloat16),
        (3, 1, 2048, ttnn.bfloat16),
        (4, 1, 2048, ttnn.bfloat16),
        (5, 1, page, fp),
        (6, 128, page, fp),
        (8, 32 * (2 if numerator_compensation else 1), page, fp),
        (9, 32 * (2 if numerator_compensation else 1), page, fp),
        (10, 8, 2048, ttnn.bfloat16),
        (11, 8, 2048, ttnn.bfloat16),
        (12, 8 * (2 if fast else 1), page, fp),
        (13, 8 * (2 if fast else 1), page, fp),
        (14, 8, page, fp),
        (16, 8 if accurate else 16, 2048, ttnn.bfloat16),
    ]
    specs = [(idx, n, input_specs[idx][1], input_specs[idx][0]) if idx < 3 else (idx, n, size, dtype)
             for idx, n, size, dtype in specs]
    if residual:
        specs += [(17 + i, 64 * kv_slots, input_specs[3 + i][1], input_specs[3 + i][0]) for i in range(2)]
    cbs = []
    if hybrid:
        specs = [
            (idx, n, 4096, ttnn.float32) if idx in (5, 8, 9, 12, 13, 14) else (idx, n, size, dtype)
            for idx, n, size, dtype in specs
        ]
    for idx, n, tile_bytes, dtype in specs:
        formats = [ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=tile_bytes)]
        if idx == 6 and accurate:
            formats.append(ttnn.CBFormatDescriptor(buffer_index=7, data_format=ttnn.float32, page_size=tile_bytes))
        if hybrid and idx in (12, 13):
            formats.append(ttnn.CBFormatDescriptor(buffer_index=idx + 5, data_format=ttnn.float32, page_size=4096))
        cbs.append(ttnn.CBDescriptor(total_size=n * tile_bytes, core_ranges=grid, format_descriptors=formats))
    defines = {
        "EXP_APPROX_MODE": "1",
        "STATS_GRANULARITY": "4" if accurate else "8",
        "SUB_EXP_GRANULARITY": "4" if accurate else "8",
        "MUL_BCAST_GRANULARITY": "4" if accurate else "8",
        "DHT_GRANULARITY": "4",
        "REDUCE_GRANULARITY": "2" if accurate else "4",
    }
    if args.mode == "main":
        defines["RESIDENT_MAIN"] = "1"
    if fast:
        defines.update(
            SDPA_STREAMING_ACCURACY="1", SDPA_STREAMING_NUMERATOR_COMPENSATION="1", SDPA_OUT_A_CB="8", SDPA_OUT_B_CB="9"
        )
        if args.denom_only:
            defines.pop("SDPA_STREAMING_NUMERATOR_COMPENSATION")
        if args.fix_correction:
            defines["SDPA_LOFI_FIX_CORRECTION"] = "1"
    if accurate:
        defines.update(
            {
                name: "1"
                for name in (
                    "SDPA_FP32_STREAMING",
                    "SDPA_FP32_STATE",
                    "SDPA_HIFI2_ROUND",
                    "SDPA_FP32_FUSED_EXP",
                    "SDPA_FP32_REUSE_EXP",
                    "SDPA_FP32_EXTRA_CONST",
                    "SDPA_FP32_PAIRED_UNPACK",
                    "SDPA_FP32_PAIRED_PACK",
                    "SDPA_FP32_L1_SUB",
                    "SDPA_FP32_L1_MACRO",
                    "SDPA_FP32_REFINE_MACRO",
                )
            }
        )
        defines.update(SDPA_DIAG_EXP_MODE="4", SDPA_FP32_SUB_BATCH="2", SDPA_DENOM_PHASES="2", SDPA_DIAG_SCORE_CB="7")
    if args.mode in ("fp32_hifi2_cheap", "qk4_pv2"):
        defines = {k: v for k, v in defines.items() if not k.startswith("SDPA_")}
        defines.update(SDPA_FP32_STREAMING="1", SDPA_FP32_STATE="1", SDPA_HIFI2_ROUND="1")
    if args.mode.startswith("qk4_"):
        defines["SDPA_QK4"] = "1"
    if args.mode.endswith("_fullsub"):
        defines["SDPA_MATCH_HIFI2"] = "1"
    defines.pop("SDPA_DENOM_PHASES", None)
    if getattr(args, "safe_rescale", False):
        defines["SDPA_LOFI_SAFE_RESCALE"] = "1"
    if hybrid:
        defines["SDPA_HYBRID_STATE"] = "1"
        if args.hybrid_block_pack:
            defines["SDPA_HYBRID_BLOCK_PACK"] = "1"
    if accurate:
        defines["SDPA_MATCH_HIFI2"] = "1"
        defines.pop("SDPA_DENOM_PHASES", None)
        if input_specs[0][0] != ttnn.bfloat16:
            defines["SDPA_LOFI_Q_RECONFIG"] = "1"
    if residual:
        defines.pop("SDPA_MATCH_HIFI2", None)
        defines.update(SDPA_LOFI_RESIDUALS="1", SDPA_LOFI_DENOM="1")
        if args.skip_residual_reconfig:
            assert kfmts[0] == kfmts[1] and vfmts[0] == vfmts[1]
            defines["SDPA_LOFI_SAME_FORMAT_RESIDUAL"] = "1"
    if pfmt == "e7":
        assert args.exp_quality == "accurate" or args.no_p_round
        defines.pop("SDPA_MATCH_HIFI2", None)
        defines["SDPA_LOFI_DENOM"] = "1"
        if not args.no_p_round:
            defines["SDPA_LOFI_ROUND_P"] = "1"
    if args.fidelity == "HiFi4" and pfmt != "e7":
        defines.pop("SDPA_MATCH_HIFI2", None)
        defines["SDPA_DENOM_PHASES"] = "2"
    if args.exp_degree != 3:
        assert args.destination == "fp32" and args.exp_quality == "cheap", "Exp refiner is FP32-only; BF16 calls native exp"
        defines["SDPA_LOFI_EXP_DEGREE"] = str(args.exp_degree)
    if args.native_exp:
        assert args.destination == "fp32" and args.exp_quality == "cheap" and args.exp_degree == 3
        defines["SDPA_LOFI_NATIVE_EXP"] = "1"
    if args.identity4:
        defines['SDPA_IDENTITY4'] = '1'
    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=getattr(ttnn.MathFidelity, args.fidelity),
        fp32_dest_acc_en=accurate,
        dst_full_sync_en=False,
        math_approx_mode=True,
    )
    if accurate or hybrid:
        pd = ttnn._ttnn.program_descriptor
        modes = pd.VectorUnpackToDestMode([pd.UnpackToDestMode.Default] * 64)
        for cb in ((5, 8, 9, 12, 13, 14) if hybrid else (5, 7, 8, 9, 12, 13, 14)):
            modes[cb] = pd.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes
    reader, writer = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    reader[0][0] = [t.buffer_address() for t in tensors]
    writer[0][0] = [out.buffer_address()]
    reader_cta = [args.q_repeats, args.k_chunks, kv_slots]
    for t in tensors:
        reader_cta += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    desc = ttnn.ProgramDescriptor(
        cbs=cbs,
        semaphores=[],
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "reader_resident.cpp",
                core_ranges=grid,
                compile_time_args=reader_cta,
                runtime_args=reader,
                defines=[("SDPA_LOFI_RESIDUALS", "1")] if residual else [],
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "writer_resident.cpp",
                core_ranges=grid,
                compile_time_args=[args.q_repeats] + ttnn.TensorAccessorArgs(out).get_compile_time_args(),
                runtime_args=writer,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=PREFIX + "compute_resident.cpp",
                core_ranges=grid,
                compile_time_args=[
                    args.q_repeats,
                    args.k_chunks,
                    struct.unpack("I", struct.pack("f", 1 / math.sqrt(128)))[0],
                ],
                defines=list(defines.items()),
                config=config,
            ),
        ],
    )

    def invoke():
        ttnn.generic_op(tensors + [out], desc)
    info = dict(
        cores=1, q_chunk=256, k_chunk=512, dim=128, kv_slots=kv_slots,
        cb_bytes_per_core=sum(n * size for _, n, size, _ in specs),
        cb_specs=[(idx, n, size, str(fmt)) for idx, n, size, fmt in specs],
        defines=defines, input_slots=dict(q=2, k=1, v=1),
        input_storage="Q7 BF16; K/V RNE5 BFP8; FP32 P/DST/state",
        input_preprocessing="Host; excluded from resident timing",
        resident_repeated_kv=not args.distinct_kv)
    assert info["cb_bytes_per_core"] == 1212416
    info.update(reserved_l1_bytes_assumed=111616, estimated_allocator_headroom_bytes=248832)
    return out, invoke, None, None, info, ref, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    DRIVER.add_common_arguments(parser)
    parser.add_argument("--q-repeats", type=int, default=16)
    parser.add_argument("--k-chunks", type=int, default=512)
    args = parser.parse_args()
    DRIVER.validate_args(args)
    assert args.q_repeats > 0 and args.k_chunks > 0
    vars(args).update(variant="identity_q7b8", destination="fp32", mode="fp32_hifi2_cheap",
        native_exp=True, exp_quality="cheap", exp_degree=3, fidelity="LoFi",
        qkv_route="host", no_p_round=True, skip_residual_reconfig=False, denom_only=False,
        fix_correction=False, hybrid_block_pack=False, safe_rescale=False, distinct_kv=False)
    device = ttnn.open_device(device_id=0, trace_region_size=16777216 if args.iters else 0)
    try:
        DRIVER.qualify(device, args, lambda: build(device, args),
                       4 * 256 * 512 * 128 * args.q_repeats * args.k_chunks, resident=True)
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
