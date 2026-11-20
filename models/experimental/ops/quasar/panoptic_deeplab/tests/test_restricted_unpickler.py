# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the allowlisted checkpoint deserialization (no device, no weights file needed)."""

import io
import pickle

import numpy as np
import pytest

from models.experimental.ops.quasar.panoptic_deeplab.reference.pytorch_model import _RestrictedUnpickler

_SENTINEL = []


def _boom(arg):
    _SENTINEL.append(arg)


def _blob_calling_boom() -> bytes:
    """Hand-assembled protocol-2 pickle that would call _boom("executed") if the global were resolved.

    Equivalent to a __reduce__ payload, but written as raw opcodes (GLOBAL, BINUNICODE, TUPLE1, REDUCE, STOP).
    """
    arg = b"executed"
    return (
        b"\x80\x02" + b"c" + __name__.encode() + b"\n_boom\n" + b"X" + len(arg).to_bytes(4, "little") + arg + b"\x85R."
    )


def _load(blob: bytes):
    return _RestrictedUnpickler(io.BytesIO(blob), encoding="latin1").load()


# Protocol 2 is what Detectron2 checkpoints use and exercises the allowed _codecs.encode path.
@pytest.mark.parametrize("protocol", [2, 4])
def test_numpy_checkpoint_roundtrip(protocol):
    ckpt = {
        "model": {
            "a.weight": np.arange(24, dtype=np.float32).reshape(2, 3, 4),
            "b.bias": np.linspace(0, 1, 5, dtype=np.float64),
            "c.num_batches_tracked": np.array(7, dtype=np.int64),
        },
        "matching_heuristics": True,
    }
    out = _load(pickle.dumps(ckpt, protocol=protocol))
    assert out["matching_heuristics"] is True
    assert out["model"].keys() == ckpt["model"].keys()
    for k, v in ckpt["model"].items():
        assert out["model"][k].dtype == v.dtype
        assert out["model"][k].shape == v.shape
        np.testing.assert_array_equal(out["model"][k], v)


def test_disallowed_callable_is_rejected_without_running(expect_error):
    _SENTINEL.clear()
    with expect_error(pickle.UnpicklingError, "Forbidden global"):
        _load(_blob_calling_boom())
    assert not _SENTINEL, "disallowed callable was invoked during unpickling"
