# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reduction driver for the face_compressed_mm split-K call-boundary hang.

Cases come from the JSON file named by FCMM_SWEEP (a list of dicts), or from DEFAULT_CASES.
Each case: id, M, K, N, kt_per_call, call_formats, hold, unp_nops, unp_pos, entry_stall, mode.
"""

import json
import os

import pytest
from conftest import blackhole_only
from helpers.compressed_utils import FMT_CODE, assign_clustered, run_compressed
from helpers.test_variant_parameters import CUSTOM_MM_CALLS, FCMM_PERTURB
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from test_matmul_face_compressed import (
    COMPRESSION_GRANULARITY,
    MULTI_CALL_PCC,
    SUPPORTED_FORMATS,
    SUPPORTED_M,
    buffer_b_words,
    encode_meta,
    meta_words_per_call,
    pack_b,
    promote_assignment,
)

KERNEL = "sources/fcmm_hang_min_test.cpp"

DEFAULT_CASES = [
    # the three rows PR 61133 reports as hanging, at the kernel's hold of 2000 cycles
    dict(
        id="alt-ct1",
        M=1,
        K=2048,
        N=32,
        kt_per_call=2,
        call_formats=[["bfp2"], ["bfp4"]],
    ),
    dict(
        id="alt-ct1-stall",
        M=1,
        K=2048,
        N=32,
        kt_per_call=2,
        call_formats=[["bfp2"], ["bfp4"]],
        entry_stall=1,
    ),
]


def load_cases():
    path = os.environ.get("FCMM_SWEEP")
    cases = json.load(open(path)) if path else DEFAULT_CASES
    out = []
    for c in cases:
        c = dict(c)
        c.setdefault("hold", 2000)
        c.setdefault("unp_nops", 0)
        c.setdefault("unp_pos", 0)
        c.setdefault("entry_stall", 0)
        c.setdefault("mode", 0)
        c.setdefault("c1ctx", 0)
        c.setdefault("nfaces", 4)
        c.setdefault("unp_wait", 0)
        out.append(c)
    return out


CASES = load_cases()


@blackhole_only
@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_fcmm_hang_min(case):
    M, K, N = case["M"], case["K"], case["N"]
    kt_per_call = case["kt_per_call"]
    call_formats = [tuple(f) for f in case.get("call_formats", [["bfp2"]])]

    kt, ct = K // DEFAULT_TILE_R_DIM, N // DEFAULT_TILE_C_DIM
    ku, cu = K // COMPRESSION_GRANULARITY, N // COMPRESSION_GRANULARITY
    num_calls = kt // kt_per_call
    rows_per_call = ku // num_calls
    faces_per_call = rows_per_call * cu
    stride = meta_words_per_call(kt_per_call, ct)
    stride_bytes = 4 * stride

    assignment = []
    if case.get("faces"):
        # explicit per-call face strings, one char per face in row-major order: 0 / 2 / 4
        code = {"0": 0, "2": FMT_CODE["bfp2"], "4": FMT_CODE["bfp4"]}
        for c in range(num_calls):
            f = case["faces"][c % len(case["faces"])]
            assert (
                len(f) == faces_per_call
            ), f"call {c}: {len(f)} faces, expected {faces_per_call}"
            assignment += [code[ch] for ch in f]
    else:
        for c in range(num_calls):
            formats = call_formats[c % len(call_formats)]
            assignment += assign_clustered(
                rows_per_call * COMPRESSION_GRANULARITY,
                N,
                formats,
                COMPRESSION_GRANULARITY,
            )

    if os.environ.get("FCMM_DUMP"):
        # save golden/result of every check so a mismatch can be inspected
        import helpers.compressed_utils as _cu
        import numpy as _np

        _orig = _cu.passed_test

        def _wrapped(golden, res, *a, **k):
            _np.savez(
                os.path.join(os.environ["FCMM_DUMP"], f"{case['id']}.npz"),
                golden=golden.float().numpy(),
                res=res.float().numpy(),
            )
            return _orig(golden, res, *a, **k)

        _cu.passed_test = _wrapped

    def pack_b_per_call(faces):
        out = b""
        infos = []
        for c in range(num_calls):
            packed, info = pack_b(faces[c * faces_per_call : (c + 1) * faces_per_call])
            base = len(out) // 16
            infos.append(
                [(b2 + base, b2_y, b4 + base, b4_y) for b2, b2_y, b4, b4_y in info]
            )
            out += packed
        return out, infos

    def encode_meta_per_call(assignment, cu, ku, infos):
        out = b""
        for c in range(num_calls):
            meta = encode_meta(
                assignment[c * faces_per_call : (c + 1) * faces_per_call],
                cu,
                rows_per_call,
                infos[c],
                b_base_words=buffer_b_words(ku),
            )
            assert (
                len(meta) <= stride_bytes
            ), f"call {c}: meta exceeds the kernel's stride"
            out += meta.ljust(stride_bytes, b"\0")
        return out

    run_compressed(
        M,
        K,
        N,
        assignment,
        KERNEL,
        COMPRESSION_GRANULARITY,
        SUPPORTED_M,
        SUPPORTED_FORMATS,
        promote_assignment,
        pack_b_per_call,
        encode_meta_per_call,
        pcc_threshold=MULTI_CALL_PCC,
        calls=[
            CUSTOM_MM_CALLS(num_calls=num_calls, meta_words_per_call=stride),
            FCMM_PERTURB(
                fcmm_math_hold=case["hold"],
                fcmm_unp_entry_nops=case["unp_nops"],
                fcmm_unp_entry_pos=case["unp_pos"],
                fcmm_entry_stall=case["entry_stall"],
                fcmm_kernel_mode=case["mode"],
                fcmm_call1_ctx=case["c1ctx"],
                fcmm_nfaces=case["nfaces"],
                fcmm_unp_wait=case["unp_wait"],
            ),
        ],
    )
