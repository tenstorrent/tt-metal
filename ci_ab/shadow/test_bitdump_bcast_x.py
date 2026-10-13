# r3 unary datacopy, bit dump of the 32-bit unary broadcast (unpack to DEST, then the row, column or scalar broadcast of
# the math thread) against main (not for merge). The unpack_A_test runner of the tree under test runs on fixed stimuli
# that place every upper half (65536 patterns) in the positions each broadcast reads, with seeded lower halves, golden check
# off; per run the sha256 of the raw result bytes and the upper halves found in the result go to $BITDUMP_DIR/<test>.jsonl.
# Comparing two trees' files compares every run's bytes; the coverage line proves the patterns reached the broadcast.
# Two DEST sync modes: half (DEST slots 0 to 3 of a 32-bit DEST, unpack_A_test.cpp) and full (slots 0 to 7, the same test
# with SyncFull, unpack_A_full_test.cpp); a broadcast is per tile, so each run's bytes must also match across the two modes.
import hashlib
import json
import os

import numpy as np
import pytest
import torch

import test_bcast as tb
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import BroadcastType, DestAccumulation
from helpers.stimuli_config import StimuliConfig

OUT = os.environ.get("BITDUMP_DIR", "/tmp/bitdump")
TILES = {"row": 32, "col": 32, "scalar": 64}  # [128, 256] or [256, 256] per run
DIMS = {32: [128, 256], 64: [256, 256]}
_STATE = {"bases": None, "kind": None, "raw": None}
_real_gen = tb.generate_stimuli
_real_collect = StimuliConfig.collect_results


def _uppers(n_tiles, bases, kind):
    i = np.arange(1024, dtype=np.int64)
    face, r, c = i // 256, (i % 256) // 16, i % 16
    if kind == "row":
        local = c + 16 * (face % 2)
    elif kind == "col":
        local = r + 16 * (face // 2)
    else:
        local = np.zeros(1024, dtype=np.int64)
    up = (np.asarray(bases, dtype=np.int64)[:, None] * (1 if kind == "scalar" else 32) + local[None, :]) % 65536
    return up.reshape(-1)[: n_tiles * 1024]


def _gen(*args, **kwargs):
    a, ca, b, cb = _real_gen(*args, **kwargs)
    n = a.numel()
    up = _uppers(n // 1024, _STATE["bases"], _STATE["kind"])
    lo = np.random.default_rng(int(_STATE["bases"][0]) + 17).integers(0, 65536, n, dtype=np.int64)
    bits = (up << 16) | lo
    if a.dtype == torch.int64:  # UInt32
        return torch.from_numpy(bits).reshape(a.shape), ca, b, cb
    i32 = torch.from_numpy(np.where(bits >= 2**31, bits - 2**32, bits).astype(np.int32))
    if a.dtype == torch.int32:
        return i32.reshape(a.shape), ca, b, cb
    return i32.view(torch.float32).reshape(a.shape), ca, b, cb


def _collect(self, location="0,0"):
    _STATE["raw"] = self.collect_raw_result_bytes(location)
    return _real_collect(self, location)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(tb, "generate_stimuli", _gen)
    monkeypatch.setattr(tb, "passed_test", lambda *a, **k: True)
    monkeypatch.setattr(StimuliConfig, "collect_results", _collect)


def _run_bcast(fmt, bcast, input_dimensions, sync):
    # test_bcast.test_unpack_bcast's body for a 32 x 32 tile and a multi-tile input, the exactness asserts off
    tile_dimensions = [32, 32]
    formats = InputOutputFormat(fmt, fmt)
    face_r_dim, num_faces_r_dim, num_faces_c_dim = tb.get_tile_params(tile_dimensions)
    num_faces = num_faces_r_dim * num_faces_c_dim
    src_A, tile_cnt_A, src_B, tile_cnt_B = tb.generate_stimuli(
        stimuli_format_A=fmt, input_dimensions_A=input_dimensions, stimuli_format_B=fmt, input_dimensions_B=input_dimensions,
        tile_dimensions=tile_dimensions)
    dest_sync = tb.DestSync.Full if sync == "full" else tb.DestSync.Half
    num_blocks, num_tiles_in_block = tb.get_num_blocks_and_num_tiles_in_block(
        dest_sync, DestAccumulation.No, formats, input_dimensions, tile_dimensions, tb.BlocksCalculationAlgorithm.Standard)
    configuration = tb.TestConfig(
        "sources/unpack_A_full_test.cpp" if sync == "full" else "sources/unpack_A_test.cpp",
        formats,
        templates=[
            tb.STOCHASTIC_ROUNDING(tb.StochasticRounding.No),
            tb.BROADCAST_TYPE(bcast),
            tb.ACC_TO_DEST(False),
            tb.REUSE_DEST_TYPE(tb.EltwiseBinaryReuseDestType.NONE),
            tb.PARTIAL_FACE(partial_a=False, partial_face_pack=False, partial_b=False, partial_face_math=False),
            tb.DISABLE_SRC_ZERO_FLAG(False),
        ],
        runtimes=[
            tb.UNPACK_TRANS_FACES(tb.Transpose.No),
            tb.UNPACK_TRANS_WITHIN_FACE(tb.Transpose.No),
            tb.NUM_FACES(num_faces),
            tb.NUM_FACES_R_DIM(num_faces_r_dim, num_faces_r_dim),
            tb.NUM_FACES_C_DIM(num_faces_c_dim, num_faces_c_dim),
            tb.TILE_COUNT(tile_cnt_A),
            tb.TEST_FACE_DIMS(face_r_dim=face_r_dim),
            tb.NUM_TILES_IN_BLOCK(num_tiles_in_block),
            tb.NUM_BLOCKS(num_blocks),
            tb.INPUT_DIMENSIONS(input_dimensions[0] // tb.TILE_DIMENSIONS[0], input_dimensions[1] // tb.TILE_DIMENSIONS[1]),
        ],
        variant_stimuli=StimuliConfig(
            src_A, fmt, src_B, fmt, fmt, tile_count_A=tile_cnt_A, tile_count_B=tile_cnt_B, tile_count_res=tile_cnt_A,
            num_faces=num_faces, face_r_dim=face_r_dim, tile_dimensions=tile_dimensions, use_dense_tile_dimensions=True),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )
    configuration.run()


F = DataFormat
BC = {"row": BroadcastType.Row, "col": BroadcastType.Column, "scalar": BroadcastType.Scalar}
# Float32: every upper half for all three broadcasts; Int32 and UInt32: every upper half for row and column, every 32nd
# for scalar (one datum per tile is read); in the full mode, which checks DEST slots 4 to 7, every 32nd for scalar
CASES = [(f, k, y) for y in ("half", "full") for f in (F.Float32, F.Int32, F.UInt32) for k in ("row", "col", "scalar")]


@pytest.mark.parametrize("cfg", CASES, ids=lambda c: f"{c[0].name}-{c[1]}" + ("-full" if c[2] == "full" else ""))
def test_bitdump_bcast32(cfg):
    fmt, kind, sync = cfg
    n_bases = 65536 if kind == "scalar" else 2048
    stride = 1 if (kind != "scalar" or (fmt == F.Float32 and sync == "half")) else 32
    bases = np.arange(0, n_bases, stride, dtype=np.int64)
    os.makedirs(OUT, exist_ok=True)
    seen = np.zeros(65536, dtype=bool)
    with open(os.path.join(OUT, f"bc_{fmt.name}_{kind}{'_full' if sync == 'full' else ''}.jsonl"), "w") as fh:
        nt = TILES[kind]
        for r in range(0, len(bases), nt):
            _STATE["bases"], _STATE["kind"], _STATE["raw"] = bases[r : r + nt], kind, None
            _run_bcast(fmt, BC[kind], DIMS[nt], sync)
            raw = _STATE["raw"]
            if raw is None:  # a compile-only (producer) run
                return
            words = np.frombuffer(raw, dtype=np.uint32)
            seen[np.unique(words >> 16)] = True
            fh.write(json.dumps({"run": r // nt, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}) + "\n")
        fh.write(json.dumps({"coverage_upper_halves": int(seen.sum()), "bases": int(len(bases))}) + "\n")
