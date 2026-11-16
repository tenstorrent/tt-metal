# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The checkpoint reader (reference/xtts_gpt_ref._load_restricted).

The checkpoint stores coqui's config objects beside the tensors, so torch's weights_only mode
cannot read it. The reader instead names every class the file may build, which keeps a
substituted file from running code. Host-only: no device.

Run:
    pytest -svv models/experimental/xtts_v2/tests/test_checkpoint_loading.py
"""
import collections
import os

import torch

from models.experimental.xtts_v2.reference.xtts_gpt_ref import (
    _CKPT_GLOBALS,
    _CKPT_STUBBED,
    _load_restricted,
)


def _write(tmp_path, obj, name="c.pth"):
    p = tmp_path / name
    torch.save(obj, p)
    return str(p)


def test_tensors_and_containers_load(tmp_path):
    """What a checkpoint legitimately holds: an OrderedDict of tensors."""
    sd = collections.OrderedDict(a=torch.arange(6, dtype=torch.float32), b=torch.ones(2, 3))
    got = _load_restricted(_write(tmp_path, sd))
    assert list(got) == ["a", "b"]
    assert torch.equal(got["a"], sd["a"]) and torch.equal(got["b"], sd["b"])


def test_a_class_outside_the_list_is_refused(tmp_path, expect_error):
    """The attack: a reducer naming any callable the file was not supposed to need."""

    class Reducer:
        def __reduce__(self):
            return (os.system, ("true",))

    with expect_error(Exception, "unexpected class"):
        _load_restricted(_write(tmp_path, Reducer()))


def test_the_lists_stay_disjoint_and_named():
    """A class must be either rebuilt or discarded, never both."""
    assert not (_CKPT_GLOBALS & _CKPT_STUBBED)
    assert all(mod.startswith("TTS.") for mod, _ in _CKPT_STUBBED)
    assert ("collections", "OrderedDict") in _CKPT_GLOBALS
