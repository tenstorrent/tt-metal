# Scratch bit dump (not for the PR): mul_reduce_scalar per-tile row against the interleaved row, raw result tile and r0.
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
from helpers.test_variant_parameters import MATH_FIDELITY, NUM_FACES_C_DIM, NUM_FACES_R_DIM, TILE_COUNT, TemplateParameter
from helpers.tile_shape import construct_tile_shape

OUT = os.environ.get("FL_BITS_OUT", "/tmp/fl_bits")
DIMS = {"32x32": [32, 32], "16x32": [16, 32]}


@dataclass
class FL3_ILV(TemplateParameter):
    ilv: int = 0
    scaler: float = 1.0

    def convert_to_cpp(self) -> str:
        return f"#define FL3_SCALER {self.scaler!r}f\n" + ("#define FL3_INTERLEAVE 1\n" if self.ilv else "")


def _inputs(n, dist, seed):
    g = torch.Generator().manual_seed(9000 + 131 * seed + sum(map(ord, dist)))
    if dist == "normal":
        x = torch.randn(n, generator=g)
        return x, x
    if dist == "wide":
        x = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-8, 9, (n,), generator=g).float())
        return x, x
    if dist == "zeros":
        a = torch.randn(n, generator=g)
        b = torch.randn(n, generator=g)
        k = torch.randint(0, n, (n // 8,), generator=g)
        a[k] = 0.0
        b[k[: len(k) // 2]] = -0.0
        return a, b
    a = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-6, 7, (n,), generator=g).float())
    b = torch.randn(n, generator=g) * torch.pow(2.0, torch.randint(-6, 7, (n,), generator=g).float())
    return a, b


def _cases():
    out = []
    for fp32 in (False, True):
        shapes = (("32x32", (2, 3, 4) if fp32 else (2, 3, 4, 5, 7, 8)), ("16x32", (2, 3, 4) if fp32 else (2, 3, 5, 7, 8)))
        for dims, nts in shapes:
            for nt in nts:
                for ilv in (0, 1):
                    for sc in ("rms", "one"):
                        for dist in ("normal", "wide", "signed", "zeros"):
                            for seed in range(3):
                                out.append((fp32, dims, nt, ilv, sc, dist, seed))
    return out


@pytest.mark.parametrize("case", _cases(), ids=lambda c: "_".join(str(x) for x in c))
def test_fl3_ilv(case):
    fp32, dims, nt, ilv, sc, dist, seed = case
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32 if fp32 else DataFormat.Float16_b)
    td = DIMS[dims]
    ts = construct_tile_shape(td)
    ept = ts.total_tile_size()
    a, b = _inputs(nt * ept, dist, seed)
    a, b = a.to(torch.bfloat16), b.to(torch.bfloat16)
    s = 1.0 / math.sqrt(nt * ept) if sc == "rms" else 1.0
    cfg = TestConfig(
        "sources/mul_reduce_scalar_test.cpp",
        formats,
        templates=[MATH_FIDELITY(MathFidelity.LoFi), FL3_ILV(ilv, s)],
        runtimes=[TILE_COUNT(nt), NUM_FACES_R_DIM(ts.num_faces_r_dim, ts.num_faces_r_dim), NUM_FACES_C_DIM(ts.num_faces_c_dim, ts.num_faces_c_dim)],
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
