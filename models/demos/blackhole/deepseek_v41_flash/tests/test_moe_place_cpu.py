# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of tt/moe_place.py (no device): file format, validation, cache key, the sequential map equals the YAML 'sequential' (cluster_axis 0) map, and the
prefill relabel identity (placed decode weights + relabelled router ids == sequential prefill layout)."""
import importlib.util
import json
import random
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "tt" / "moe_place.py"


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False


def load_mod():
    spec = importlib.util.spec_from_file_location("moe_place_cpu", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rand_perm(seed, same_col):
    rng = random.Random(seed)
    a = []
    if same_col:
        a = [0] * 384
        for col in range(8):
            es = list(range(col * 48, col * 48 + 48))
            rows = [r for r in range(4) for _ in range(12)]
            rng.shuffle(rows)
            for e, r in zip(es, rows):
                a[e] = r * 8 + col
        return a
    chips = [c for c in range(32) for _ in range(12)]
    rng.shuffle(chips)
    return chips


def test_sequential_matches_yaml_sequential():
    m = load_mod()
    num_dispatch, devices, epd = 4, 32, 12
    num_replicated = devices // num_dispatch
    per_cluster = 384 // num_replicated
    ref = []
    for e in range(384):
        cluster, within = e // per_cluster, e % per_cluster
        ref.append((within // epd) * num_replicated + cluster)  # tt_moe_decode_config.py, cluster_axis 0
    assert m.sequential() == ref


def test_validate_and_tag(tmp_path, monkeypatch):
    m = load_mod()
    a, b = rand_perm(1, False), rand_perm(2, False)
    m.validate(a)
    bad = list(a)
    bad[0] = (bad[0] + 1) % 32
    assert raises(AssertionError, m.validate, bad)
    assert m.tag(a) != m.tag(b) and m.tag(a) == m.tag(list(a))
    p = tmp_path / "pl.json"
    json.dump({"3": a, "4": m.sequential(), "5": rand_perm(3, True)}, open(p, "w"))
    monkeypatch.setenv("DSV41_MOE_PLACE", str(p))
    assert m.enabled() and m.assignment(3) == a and m.cache_suffix(3) == "_" + m.tag(a)
    assert m.assignment(4) is None and m.cache_suffix(4) == ""  # the sequential layout keeps the plain cache directory
    assert m.assignment(9) is None and m.cache_suffix(9) == ""
    assert m.samecol(m.assignment(5)) and not m.samecol(a)
    monkeypatch.delenv("DSV41_MOE_PLACE")
    assert not m.enabled() and m.assignment(3) is None and m.cache_suffix(3) == ""
    monkeypatch.setenv("DSV41_MOE_PLACE", str(p))
    m.check_prefill_compat([3], True)  # relabel on (default): the unified prefill ranks permuted gate columns, compatible
    monkeypatch.setenv("DSV41_MOE_PLACE_PF_RELABEL", "0")
    assert raises(RuntimeError, m.check_prefill_compat, [3], True)  # relabel off: the in-place prefill would read the wrong experts
    m.check_prefill_compat([4, 9], True)
    m.check_prefill_compat([3], False)


@pytest.mark.parametrize("same_col", [False, True])
def test_prefill_relabel_identity(same_col):
    """weights of expert e sit at chip a[e] slot rank(e among the experts of that chip, ascending) (stable argsort of the mapping). The unified prefill expects the
    weights of id e' at chip seq(e') slot e' % 12. With pi(e) = the j-th smallest id of the sequential set of chip a[e], j = rank of e on its chip, the relabelled id pi(e)
    finds exactly the weights of e."""
    m = load_mod()
    a = rand_perm(7, same_col)
    seq = m.sequential()
    on_chip = {d: sorted(e for e in range(384) if a[e] == d) for d in range(32)}
    seq_set = {d: sorted(e for e in range(384) if seq[e] == d) for d in range(32)}
    pi = [0] * 384
    for d in range(32):
        for j, e in enumerate(on_chip[d]):
            pi[e] = seq_set[d][j]
    assert sorted(pi) == list(range(384))
    for e in range(384):
        e2 = pi[e]
        assert seq[e2] == a[e]  # same chip
        assert e2 % 12 == on_chip[a[e]].index(e) or (e2 % 48) % 12 == on_chip[a[e]].index(
            e
        )  # same slot (seq slot = (e' % 48) % 12 = e' % 12)
    if same_col:
        assert all(
            pi[e] // 48 == e // 48 for e in range(384)
        )  # a same-column placement relabels inside the column group
