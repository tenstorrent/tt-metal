# Scratch accuracy dump (not for the PR): mul_reduce_scalar per-tile products against the API's accumulate_in_one_tile,
# RMSNorm-like inputs (a = b = x) and signed products, scaler 1/sqrt(numel) as the DeepSeek RMSNorm passes it.
# Writes $FL_BITS_OUT/<case>.npz with the device result r0 and the fp64 reference.
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
    NUM_FACES_C_DIM,
    NUM_FACES_R_DIM,
    TILE_COUNT,
    TemplateParameter,
)
from helpers.tile_shape import construct_tile_shape

OUT = os.environ.get("FL_BITS_OUT", "/tmp/fl_bits")
DIMS = {"32x32": [32, 32], "16x32": [16, 32]}


@dataclass
class FL3_MODE(TemplateParameter):
    one_tile: int = 0
    scaler: float = 1.0

    def convert_to_cpp(self) -> str:
        s = f"#define FL3_SCALER {self.scaler!r}f\n"
        return s + ("#define FL3_ONE_TILE 1\n" if self.one_tile else "")


def _inputs(n, dist, seed):
    g = torch.Generator().manual_seed(7000 + 97 * seed + hash(dist) % 1000)
    if dist == "normal":
        x = torch.randn(n, generator=g)
        return x, x
    if dist == "wide":
        x = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-6, 7, (n,), generator=g).float())
        return x, x
    if dist == "outlier":
        x = torch.randn(n, generator=g)
        k = torch.randint(0, n, (max(1, n // 100),), generator=g)
        x[k] = x[k] * 64.0
        return x, x
    if dist == "uniform":
        x = torch.rand(n, generator=g)
        return x, x
    a = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-4, 5, (n,), generator=g).float())
    b = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-4, 5, (n,), generator=g).float())
    return a, b


def _cases():
    out = []
    for fp32 in (False, True):
        for dims, nts in (("32x32", (1, 2, 3, 4, 7, 8)), ("16x32", (1, 3))):
            for nt in nts:
                for one in (0, 1):
                    if fp32 and not one and dims == "32x32" and nt > 4:
                        continue
                    for dist in ("normal", "wide", "outlier", "uniform", "signed"):
                        for seed in range(8):
                            out.append((fp32, dims, nt, one, dist, seed))
    return out


@pytest.mark.parametrize("case", _cases(), ids=lambda c: "_".join(str(x) for x in c))
def test_fl3_acc(case):
    fp32, dims, nt, one, dist, seed = case
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32 if fp32 else DataFormat.Float16_b)
    td = DIMS[dims]
    ts = construct_tile_shape(td)
    ept = ts.total_tile_size()
    a, b = _inputs(nt * ept, dist, seed)
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    s = 1.0 / math.sqrt(nt * ept)
    cfg = TestConfig(
        "sources/mul_reduce_scalar_test.cpp",
        formats,
        templates=[MATH_FIDELITY(MathFidelity.LoFi), FL3_MODE(one, s)],
        runtimes=[
            TILE_COUNT(nt),
            NUM_FACES_R_DIM(ts.num_faces_r_dim, ts.num_faces_r_dim),
            NUM_FACES_C_DIM(ts.num_faces_c_dim, ts.num_faces_c_dim),
        ],
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
        dest_acc=DestAccumulation.Yes if fp32 else DestAccumulation.No,
    )
    res = cfg.run().result
    ref = float((a.double() * b.double()).sum()) * s * s
    os.makedirs(OUT, exist_ok=True)
    name = "_".join(str(x) for x in case)
    np.savez(f"{OUT}/{name}.npz", r0=np.array([float(res[0])], dtype=np.float64), ref=np.array([ref]), s=np.array([s]))
