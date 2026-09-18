# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Golden outputs from the original experiment code, not the packaged code."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from tt_bfp_quant import HessianFactor, gptq_search, search_linear


def test_original_experiment_fixture():
    root = Path(__file__).parent
    path = root / "frozen-experiment.npz"
    meta = json.loads((root / "frozen-experiment.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["fixture_sha256"]
    with np.load(path) as f:
        w = torch.from_numpy(f["weight"])
        factor = HessianFactor(
            torch.from_numpy(f["factor_order"]),
            torch.from_numpy(f["factor_inverse_order"]),
            torch.from_numpy(f["factor_dead"]),
            torch.from_numpy(f["factor_upper"]),
            0.01,
            True,
            0.0,
        )
        for bits in (4, 8):
            q, _ = search_linear(w, bits, backend="numpy")
            np.testing.assert_array_equal(q.numpy().view(np.uint32), f[f"search{bits}_expected"].view(np.uint32))
        q, info = gptq_search(w, factor=factor, block_size=32, backend="numpy")
        np.testing.assert_array_equal(q.numpy().view(np.uint32), f["gptq_expected"].view(np.uint32))
        assert info["group_choices"] == meta["group_choices"]
