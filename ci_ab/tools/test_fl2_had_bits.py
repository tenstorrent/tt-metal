# Scratch bit dump (not for the PR): raw result tiles of the H128 transform for seeded inputs over a wide range of
# magnitudes and signs, written to $FL_BITS_OUT/<case>.npz.
import os

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, HADAMARD, MATH_FIDELITY, NUM_FACES, TILE_COUNT
from helpers.golden_generators import HadamardH128Golden

OUT = os.environ.get("FL_BITS_OUT", "/tmp/fl_bits")
FID = {"lofi": MathFidelity.LoFi, "hifi2": MathFidelity.HiFi2, "hifi4": MathFidelity.HiFi4}
BF16 = DataFormat.Float16_b


def _tile(values):
    t = torch.zeros(1024, dtype=torch.bfloat16)
    v = values.reshape(-1).to(torch.bfloat16)
    t[: v.numel()] = v
    return t


def _vectors(seed, n):
    g = torch.Generator().manual_seed(7000 + seed)
    x = torch.randn(n, 128, generator=g) * torch.pow(2.0, torch.randint(-20, 21, (n, 128), generator=g).float())
    if seed % 4 == 1:
        x = torch.randn(n, 128, generator=g) * torch.pow(2.0, torch.randint(-3, 4, (n, 1), generator=g).float())
    if seed % 4 == 2:
        x[:, ::3] = 0.0
    return x


CASES = [(f, nz, s, k, n) for f in FID for nz in (True, False) for s in range(12) for k in ("vec", "blk") for n in (8, 5)]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}_{int(c[1])}_{c[2]}_{c[3]}_{c[4]}")
def test_fl2_had_bits(case):
    fid, normalize, seed, kernel, n = case
    x = _vectors(seed, n)
    h16 = _tile(HadamardH128Golden.sylvester(16))
    cfg = TestConfig(
        "sources/hadamard_test.cpp" if kernel == "vec" else "sources/hadamard_block_test.cpp",
        InputOutputFormat(BF16, BF16),
        templates=[HADAMARD(hadamard_normalize=normalize, h16_tile_index=0), MATH_FIDELITY(FID[fid]), DEST_SYNC(DestSync.Half)],
        runtimes=[TILE_COUNT(n), NUM_FACES(1, 1, 1)],
        variant_stimuli=StimuliConfig(
            h16, BF16, torch.cat([_tile(v) for v in x]), BF16, BF16,
            tile_count_A=1, tile_count_B=n, tile_count_res=n, num_faces=1,
        ),
        dest_acc=DestAccumulation.No,
    )
    cfg.run()
    raw = cfg.variant_stimuli.collect_raw_result_bytes(TestConfig.TENSIX_LOCATION)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{fid}_{int(normalize)}_{seed}_{n}_{kernel}.npz", res=np.frombuffer(raw, dtype=np.uint8), r0=np.array([0.0]))
