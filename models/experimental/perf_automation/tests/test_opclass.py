"""M3 · OP_CLASS_MAP coverage of the real sample (PLAN section 4.2 / 7.4)."""

import csv
from pathlib import Path

from agent.opclass import SIGNPOST_CODES, base_op_code, classify_op

FIXTURE = Path(__file__).parent / "fixtures" / "ops_perf_sample.csv"


def _sample_op_codes():
    with open(FIXTURE, newline="") as f:
        return [r["OP CODE"] for r in csv.DictReader(f)]


def test_opclass_map_covers_sample():
    # Every device OP CODE in the real sample classifies without crashing; the
    # known kernels land in a real class, GenericOp falls to `other` (TBD).
    expected = {
        "EmbeddingsDeviceOperation": "embedding",
        "LayerNormDeviceOperation": "reduction",
        "MatmulDeviceOperation": "matmul",
        "MinimalMatmulDeviceOperation": "matmul",
        "GenericOpDeviceOperation": "other",  # TBD(genericop)
        "BinaryNgDeviceOperation": "eltwise",
        "SDPAOperation": "attention",
    }
    for code in _sample_op_codes():
        if code in SIGNPOST_CODES:
            continue
        assert classify_op(code) == expected[code], code


def test_shape_suffix_is_stripped():
    # tt-perf-report appends shapes to the OP Code; classification ignores them.
    assert base_op_code("MatmulDeviceOperation 512 x 1024 x 3072") == "MatmulDeviceOperation"
    assert classify_op("MatmulDeviceOperation 512 x 1024 x 3072") == "matmul"


def test_unknown_code_is_other():
    assert classify_op("TotallyMadeUpOp") == "other"
