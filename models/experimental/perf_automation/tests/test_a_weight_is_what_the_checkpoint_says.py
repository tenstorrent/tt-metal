# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What counts as a weight is decided by the checkpoint, not by a list of attribute names.

The census skipped runtime tensors -- KV caches, page tables, staging buffers -- with a substring
match on the attribute they hung off:

    _CACHE_ATTRS = ("kv_cache", "kvcache", "page_table", "paged_cache", "cache")

defended on the grounds that "the pipeline attaches them as named attributes". It does, in the
models that list was written against. Voxtral calls its cache `kv`, which contains none of those
strings, and run 10's census reported `kv: 83,886,112` inside the model's weight total.

The test that does not care what anything is called was already in the file: a tensor whose element
count appears in the CHECKPOINT was loaded from disk; one whose count appears nowhere in it was made
at runtime. True of a KV cache, an accumulator and a staging copy alike, in any naming convention.

It needed a checkpoint to compare against, and nothing passed one until 2026-08-19 -- so the name
list was the only filter that had ever run.
"""

import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_MID = "mistralai/Voxtral-Mini-3B-2507"


def _model_with_a_cache(numels, cache_numel):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.tower = nn.Module()
            for i, n in enumerate(numels):
                setattr(self.tower, "w%d" % i, nn.Parameter(torch.zeros(n, dtype=torch.bfloat16)))
            # named `kv` -- exactly what the deleted list did NOT match
            self.kv = nn.Parameter(torch.zeros(cache_numel, dtype=torch.bfloat16))

    return M()


def _real_numels(n):
    from agent.checkpoint_sections import hf_cache_dir
    from agent.weight_census import _checkpoint_tensor_sections

    if not hf_cache_dir(_MID):
        pytest.skip("voxtral not in the local HF cache")
    return [numel for numel, _sec, _nm in list(_checkpoint_tensor_sections(_MID))[:n]]


def test_a_cache_the_name_list_missed_is_excluded_by_the_checkpoint(monkeypatch):
    import agent.weight_census as WC

    monkeypatch.setattr(WC, "_on_device", lambda *a, **k: True)
    numels = _real_numels(6)
    cache = 41_943_056  # a size that appears nowhere in the checkpoint
    assert cache not in set(numels)

    c = WC.census(_model_with_a_cache(numels, cache), scope="pipeline", checkpoint=_MID)

    assert c["weight_bytes"] == sum(numels) * 2, "the cache was counted as a weight"
    assert c["complete"] is True


def test_without_a_checkpoint_the_census_says_it_cannot_tell(monkeypatch):
    """It counts everything -- an OVERCOUNT, which reads as too LOW a ceiling, the direction that
    lets a run believe in headroom it does not have. Incomplete is refused by perf_target."""
    import agent.weight_census as WC

    monkeypatch.setattr(WC, "_on_device", lambda *a, **k: True)
    numels = _real_numels(6)
    cache = 41_943_056

    c = WC.census(_model_with_a_cache(numels, cache), scope="pipeline", checkpoint=None)

    assert c["weight_bytes"] > sum(numels) * 2, "the cache is somehow already excluded"
    assert c["complete"] is False, "an unclassifiable census must not present itself as the answer"


def test_the_name_list_is_gone():
    src = (_PA / "agent" / "weight_census.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "_CACHE_ATTRS" not in code, "the attribute-name list is back"
    assert "_is_cache_attr" not in code


def test_completeness_requires_a_checkpoint():
    src = (_PA / "agent" / "weight_census.py").read_text()
    # the returned dict's entry, not the docstring's mention of the field
    i = src.index('"complete": unknown ==')
    assert "_have_ckpt" in src[i : i + 120], "a census with nothing to classify against reads complete"
