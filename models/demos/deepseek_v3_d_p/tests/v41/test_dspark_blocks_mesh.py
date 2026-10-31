# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The one-pass ``dspark_blocks.mesh_inputs`` against the per-rank reference, on the
workstation: no ttnn, no device. ``python -m pytest --noconftest -p no:cacheprovider
models/demos/deepseek_v3_d_p/tests/v41/test_dspark_blocks_mesh.py``."""

import importlib.util
import pathlib
import random

import torch

_PATH = pathlib.Path(__file__).resolve().parents[2] / "tt" / "v41" / "dspark_blocks.py"
_SPEC = importlib.util.spec_from_file_location("dspark_blocks", _PATH)
db = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(db)  # the module alone: tt.v41's package __init__ imports ttnn


def test_mesh_inputs_match_the_reference():
    """The one-pass ``mesh_inputs`` equals the per-rank ``mesh_inputs_reference`` on random entries (every
    ring
    width, block count and position regime, taps of every length, some ranks without taps)."""
    rng = random.Random(181)
    for ring_slots in (128, 160):
        for blocks in (1, 2, 3, 4):
            for pmax in (5, 200, 70000):
                for _ in range(10):
                    entries = []
                    for _r in range(8):
                        es = []
                        for u in rng.sample(range(4), blocks):
                            p = rng.randint(0, pmax)
                            a = rng.randint(0, 4)
                            taps = [(rng.randint(0, db.ROWS - 1), p - a + j) for j in range(a + 1)]
                            taps = [(t, q) for t, q in taps if 0 <= q <= p] if rng.random() < 0.9 else []
                            es.append({"user": u, "token": rng.randint(0, 100000), "position": p, "taps": taps})
                        entries.append(es)
                    got, ref = db.mesh_inputs(entries, ring_slots, 4), db.mesh_inputs_reference(entries, ring_slots, 4)
                    assert got.keys() == ref.keys()
                    for k in ref:
                        assert got[k].dtype == ref[k].dtype and got[k].shape == ref[k].shape, k
                        assert torch.equal(got[k], ref[k]), (k, ring_slots, blocks, pmax)
