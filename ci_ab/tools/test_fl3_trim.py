# Scratch bit dump (not for the PR): plain, chunked and sum reduce rows with a range of scalers, raw result tile and r0.
import math
import os
from dataclasses import dataclass

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
    TemplateParameter,
)
from helpers.tile_shape import construct_tile_shape

OUT = os.environ.get("FL_BITS_OUT", "/tmp/fl_bits")
DIMS = {"32x32": [32, 32], "16x32": [16, 32], "16x16": [16, 16]}
FID = {"lofi": MathFidelity.LoFi, "hifi2": MathFidelity.HiFi2, "hifi4": MathFidelity.HiFi4}
SC = {"rms": None, "neg": -1.5, "zero": 0.0, "big": 1024.0, "tiny": 3e-4}
SRC = {"plain": "sources/mul_reduce_scalar_test.cpp", "chunked": "sources/mul_reduce_scalar_chunked_test.cpp", "sum": "sources/sum_reduce_scalar_test.cpp"}


@dataclass
class FL3_SC(TemplateParameter):
    scaler: float = 1.0

    def convert_to_cpp(self) -> str:
        return f"#define FL3_SCALER {self.scaler!r}f\n"


def _inputs(n, dist, seed, kind):
    g = torch.Generator().manual_seed(4000 + 37 * seed + sum(map(ord, dist + kind)))
    if dist == "uniform":
        a = torch.rand(n, generator=g)
    else:
        a = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-6, 7, (n,), generator=g).float())
    b = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-6, 7, (n,), generator=g).float())
    if kind == "sum":
        b = torch.zeros(n)
    return a, b


def _cases():
    out = []
    for fp32 in (False, True):
        for fid in FID:
            for dims in DIMS:
                for nt in ((1, 4) if fp32 else (1, 3, 8)):
                    for sc in SC:
                        out.append(("plain", fp32, fid, dims, nt, 0, sc, "wide"))
                if dims != "16x16":
                    for nt, chunk in (((5, 4),) if fp32 else ((9, 8), (14, 8), (7, 4))):
                        for sc in ("rms", "neg", "big"):
                            out.append(("chunked", fp32, fid, dims, nt, chunk, sc, "wide"))
                if fid != "lofi":
                    for nt in ((1, 3) if fp32 else (1, 2, 7)):
                        if dims == "16x16" and nt > 1:
                            continue
                        for sc in ("rms", "neg", "zero", "big"):
                            out.append(("sum", fp32, fid, dims, nt, 0, sc, "uniform"))
    return out


@pytest.mark.parametrize("case", _cases(), ids=lambda c: "_".join(str(x) for x in c))
def test_fl3_trim(case):
    kind, fp32, fid, dims, nt, chunk, sc, dist = case
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32 if fp32 else DataFormat.Float16_b)
    td = DIMS[dims]
    ts = construct_tile_shape(td)
    ept = ts.total_tile_size()
    a, b = _inputs(nt * ept, dist, 0, kind)
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    s = SC[sc] if SC[sc] is not None else 1.0 / math.sqrt(nt * ept)
    runtimes = [TILE_COUNT(nt)]
    if kind == "chunked":
        runtimes.append(MUL_REDUCE_SCALAR_CHUNK_SIZE(chunk))
    runtimes += [NUM_FACES_R_DIM(ts.num_faces_r_dim, ts.num_faces_r_dim), NUM_FACES_C_DIM(ts.num_faces_c_dim, ts.num_faces_c_dim)]
    cfg = TestConfig(
        SRC[kind],
        formats,
        templates=[MATH_FIDELITY(FID[fid]), FL3_SC(s)],
        runtimes=runtimes,
        variant_stimuli=StimuliConfig(
            a, formats.input_format, b, formats.input_format, formats.output_format,
            tile_count_A=nt, tile_count_B=nt, tile_count_res=1, num_faces=ts.total_num_faces(), face_r_dim=ts.face_r_dim,
            tile_dimensions=td, use_dense_tile_dimensions=True, sfpu=False,
        ),
        dest_acc=DestAccumulation.Yes if fp32 else DestAccumulation.No,
    )
    res = cfg.run().result
    raw = cfg.variant_stimuli.collect_raw_result_bytes(TestConfig.TENSIX_LOCATION)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{'_'.join(str(x) for x in case)}.npz", res=np.frombuffer(raw, dtype=np.uint8), r0=np.array([float(res[0])]))
