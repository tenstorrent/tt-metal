# Scratch bit dump (not for the PR): raw result tile of mul_reduce_scalar (plain and chunked) for seeded inputs,
# written to $FL_BITS_OUT/<case>.npz; element [0] is the defined result, the full tile is kept as well.
import os

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    MATH_FIDELITY,
    MUL_REDUCE_SCALAR_CHUNK_SIZE,
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
)
from helpers.tile_shape import construct_tile_shape

OUT = os.environ.get("FL_BITS_OUT", "/tmp/fl_bits")
FID = {"lofi": MathFidelity.LoFi, "hifi2": MathFidelity.HiFi2, "hifi4": MathFidelity.HiFi4}
DIMS = {"32x32": [32, 32], "16x32": [16, 32], "16x16": [16, 16]}


def _inputs(n, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    if seed == 0:
        a = torch.rand(n, generator=g)
        b = torch.ones(n)
    else:
        a = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-6, 7, (n,), generator=g).float())
        b = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-6, 7, (n,), generator=g).float())
        if seed == 2:
            k = torch.randint(0, n, (n // 16,), generator=g)
            a[k] = 0.0
            b[k[: len(k) // 2]] = -0.0
    return a.to(torch.bfloat16), b.to(torch.bfloat16)


def _cases():
    out = []
    for fp32 in (False, True):
        for fid in FID:
            for dims in DIMS:
                for nt in ((1, 2, 3, 4) if fp32 else (1, 2, 3, 7, 8)):
                    for seed in (0, 1, 2):
                        out.append(("plain", fp32, fid, dims, nt, 0, seed))
                for chunk in ((4,) if fp32 else (4, 8)):
                    b = chunk - 1
                    for nt in (b + 2, 2 * b + 1, 4 * b):
                        for seed in (1, 2):
                            out.append(("chunked", fp32, fid, dims, nt, chunk, seed))
    return out


@pytest.mark.parametrize("case", _cases(), ids=lambda c: "_".join(str(x) for x in c))
def test_fl2_mrs_bits(case):
    kind, fp32, fid, dims, nt, chunk, seed = case
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32 if fp32 else DataFormat.Float16_b)
    dest_acc = DestAccumulation.Yes if fp32 else DestAccumulation.No
    td = DIMS[dims]
    ts = construct_tile_shape(td)
    ept = ts.total_tile_size()
    a, b = _inputs(nt * ept, seed)
    runtimes = [TILE_COUNT(nt)]
    if kind == "chunked":
        runtimes.append(MUL_REDUCE_SCALAR_CHUNK_SIZE(chunk))
    runtimes += [NUM_FACES_R_DIM(ts.num_faces_r_dim, ts.num_faces_r_dim), NUM_FACES_C_DIM(ts.num_faces_c_dim, ts.num_faces_c_dim)]
    cfg = TestConfig(
        "sources/mul_reduce_scalar_test.cpp" if kind == "plain" else "sources/mul_reduce_scalar_chunked_test.cpp",
        formats,
        templates=[MATH_FIDELITY(FID[fid])],
        runtimes=runtimes,
        variant_stimuli=StimuliConfig(
            a,
            formats.input_format,
            b,
            formats.input_format,
            formats.output_format,
            tile_count_A=nt,
            tile_count_B=nt,
            tile_count_res=1,
            num_faces=ts.total_num_faces(),
            face_r_dim=ts.face_r_dim,
            tile_dimensions=td,
            use_dense_tile_dimensions=True,
            sfpu=False,
        ),
        dest_acc=dest_acc,
    )
    res = cfg.run().result
    raw = cfg.variant_stimuli.collect_raw_result_bytes(TestConfig.TENSIX_LOCATION)
    os.makedirs(OUT, exist_ok=True)
    name = "_".join(str(x) for x in case)
    np.savez(f"{OUT}/{name}.npz", res=np.frombuffer(raw, dtype=np.uint8), r0=np.array([float(res[0])], dtype=np.float64))
