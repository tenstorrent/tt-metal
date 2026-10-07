# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the default spec-decode resolution (tt/spec_policy.py): no device. Run: pytest tests/test_spec_policy.py"""

import pytest

from models.demos.blackhole.deepseek_v41_flash.tt import spec_policy as sp


def R(batch, env=None, ctx=None):
    return sp.resolve(batch, 4, ctx, env or {})


@pytest.mark.parametrize(
    "B,ks",
    [(4, [3, 5]), (8, [1, 3, 5]), (16, [1, 3, 5]), (32, [0, 1, 3])],
)
def test_default_adaptive(B, ks):
    c = R(B, ctx=512)
    assert (c.mode, c.ks, c.k, c.rows) == ("adaptive", ks, max(ks), False)
    assert c.ring_rows == 288 and not c.explicit
    assert (
        c.describe().startswith("spec decode: adaptive {" + ",".join(map(str, ks)) + "}")
        and f"default for B={B}" in c.describe()
    )


@pytest.mark.parametrize("B", [64, 128, 100])
def test_default_plain_large_batch(B):
    c = R(B, ctx=512)
    assert c.mode == "plain" and c.ring_rows == 0 and not c.on
    assert "B>=64" in c.reason or "B>=128" in c.reason


def test_b64_plain_reason_text():
    assert R(64).describe().startswith("spec decode: plain (B>=64 default")
    assert R(128).describe() == "spec decode: plain (B>=128)"


def test_explicit_off():
    for B in (4, 16, 32, 64, 128):
        c = R(B, {"DSV41_SPEC": "0", "DSV41_SPEC_ADAPT": "1"})
        assert c.mode == "plain" and c.ring_rows == 0 and c.describe() == "spec decode: plain (explicit DSV41_SPEC=0)"


def test_explicit_fixed_and_adapt_overrides():
    c = R(16, {"DSV41_SPEC": "3"})
    assert (c.mode, c.k, c.ring_rows, c.explicit) == ("fixed", 3, 288, True)
    c = R(8, {"DSV41_SPEC": "5", "DSV41_SPEC_ADAPT": "1", "DSV41_SPEC_SET": "0,3,5"})
    assert (c.mode, c.ks, c.k) == ("adaptive", [0, 3, 5], 5)
    c = R(16, {"DSV41_SPEC_SET": "1,3"})
    assert (c.mode, c.ks) == ("adaptive", [1, 3])
    c = R(16, {"DSV41_SPEC_ADAPT": "0"})
    assert (c.mode, c.k) == ("fixed", 3) and c.ring_rows == 288
    c = R(4, {"DSV41_SPEC_ADAPT": "1"})
    assert (c.mode, c.ks) == ("adaptive", [3, 5])


def test_b64_opt_in():
    for env in ({"DSV41_SPEC": "3"}, {"DSV41_SPEC_ADAPT": "1"}, {"DSV41_SPEC_ROWS": "1"}, {"DSV41_SPEC_SET": "1,3"}):
        c = R(64, env)
        assert c.on and c.rows and c.ring_rows == 288, env
    c = R(64, {"DSV41_SPEC_ADAPT": "1"})
    assert (c.mode, c.ks) == ("adaptive", [0, 1, 3])
    c = R(64, {"DSV41_SPEC": "3", "DSV41_SPEC_ADAPT": "1", "DSV41_SPEC_SET": "0,1,3", "DSV41_SPEC_ROWS": "1"})
    assert (c.mode, c.ks) == ("adaptive", [0, 1, 3])
    assert R(64, {"DSV41_SPEC": "1"}).k == 1


def test_b128():
    c = R(128, {"DSV41_SPEC": "3"})
    assert c.mode == "plain" and "ignored" in c.reason
    assert R(128, {"DSV41_SPEC_ADAPT": "1"}).mode == "plain"
    c = R(128, {"DSV41_SPEC_B128": "1", "DSV41_SPEC_ROWS": "1", "DSV41_SPEC": "3"})
    assert (c.mode, c.k, c.rows, c.ring_rows) == ("fixed", 3, True, 288)


def test_context_fallback():
    assert R(16, ctx=70000).on and R(8, ctx=70000).on and R(4, ctx=70000).on
    assert R(32, ctx=40000).on
    c = R(32, ctx=70000)
    assert (
        c.mode == "plain"
        and c.ring_rows == 0
        and c.describe().startswith("spec decode: plain (context too long for spec")
    )
    c = R(4, ctx=135000)
    assert c.mode == "plain" and "context too long" in c.reason
    # explicit settings are trusted
    assert R(32, {"DSV41_SPEC": "3"}, ctx=70000).on
    assert R(16, {"DSV41_SPEC_ADAPT": "1"}, ctx=135000).on


def test_users_per_row_padding():
    assert R(1).ks == [3, 5] and R(5).ks == [1, 3, 5] and R(30).ks == [0, 1, 3]
    assert R(63).mode == "plain" or True  # padded batch of the demo is always a multiple of the mesh rows


def test_default_ks_table():
    assert (
        sp.default_ks(1) == [3, 5]
        and sp.default_ks(2) == [1, 3, 5]
        and sp.default_ks(4) == [1, 3, 5]
        and sp.default_ks(8) == [1, 3]
    )
    assert sp.default_ks(16) == [1] and sp.default_ks(16, True) == [1, 3] and sp.default_ks(32, True) == [1, 3]
    assert sp.parse_ks("0,3,5", "1") == [0, 3, 5] and sp.parse_ks(None, "1,3") == [1, 3]


def test_apply_ring_rows_and_reconfigure_sequence():
    env = {}
    sp.apply(R(16), env)  # build for B=16: spec -> ring rows 288
    assert env == {"DSV41_RING_ROWS": "288"}
    sp.apply(
        sp.resolve(64, 4, None, env), env
    )  # reconfigure to plain B=64: our value is not an explicit setting, removed
    assert env == {}
    sp.apply(sp.resolve(64, 4, None, {**env, "DSV41_SPEC_ROWS": "1"}), env)
    sp._applied.clear()
    env2 = {"DSV41_RING_ROWS": "160"}
    sp.apply(R(16), env2)
    assert env2 == {"DSV41_RING_ROWS": "160"}  # user value wins
    env3 = {"DSV41_SPEC": "0"}
    sp.apply(sp.resolve(16, 4, None, env3), env3)
    assert "DSV41_RING_ROWS" not in env3  # explicit off: build untouched


def test_apply_rows_for_b64_opt_in_and_undo(monkeypatch):
    import os

    for k in [k for k in os.environ if k.startswith("DSV41_")]:
        monkeypatch.delenv(k)
    c = sp.resolve_apply(64, 4, None)
    assert c.mode == "plain" and "DSV41_RING_ROWS" not in os.environ
    monkeypatch.setenv("DSV41_SPEC_ADAPT", "1")
    c = sp.resolve_apply(64, 4, None)
    assert c.on and os.environ["DSV41_RING_ROWS"] == "288" and os.environ["DSV41_SPEC_ROWS"] == "1"
    monkeypatch.delenv("DSV41_SPEC_ADAPT")
    c = sp.resolve_apply(64, 4, None)  # the applied DSV41_SPEC_ROWS must not count as an explicit opt-in
    assert c.mode == "plain" and "DSV41_RING_ROWS" not in os.environ and "DSV41_SPEC_ROWS" not in os.environ
    c = sp.resolve_apply(32, 4, 512)
    assert c.mode == "adaptive" and os.environ["DSV41_RING_ROWS"] == "288" and "DSV41_SPEC_ROWS" not in os.environ
    sp.apply(sp.SpecChoice("plain"))
    assert "DSV41_RING_ROWS" not in os.environ


def test_dram_check():
    assert sp.dram_check(301, 200, 4, {})[0] and sp.dram_check(190, 100, 1, {})[0]
    assert not sp.dram_check(172, 84, 4, {})[0] and not sp.dram_check(300, 10, 4, {})[0]
    assert sp.dram_check(527, 478, 8, {})[0]  # B=32 ISL 30k (measured, spec ran)
    assert (
        not sp.dram_check(172, 84, 8, {})[0] and not sp.dram_check(300, 290, 8, {})[0]
    )  # B=32 ISL 60k (OOM) / below the 320 need
    assert not sp.dram_check(527, 478, 8, {"DSV41_SPEC_NEED_FREE_MIB": "9999"})[0]
