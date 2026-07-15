#!/usr/bin/env python3
# Offline (NO device) tests for the regime_a_bench harness persistence + timeout classification.
# Run: python3 tools/mm_sweep/test_bench_harness.py   (exits nonzero on failure).
import json, os, tempfile, types

import regime_a_bench as b


def _tmp():
    d = tempfile.mkdtemp(prefix="rabench_")
    return d


def test_finalized_report_never_overwritten():
    """A finalized {corpus,cache} OUT must never be touched by cache persistence (run_cfg writes CACHE)."""
    d = _tmp()
    out_path = os.path.join(d, "report.json")
    cache_path = os.path.join(d, "cache.json")
    finalized = {"corpus": [{"M": 1, "K": 1, "N": 1}], "cache": {"1x1x1:auto": {"cls": "ok"}}}
    json.dump(finalized, open(out_path, "w"))
    before = open(out_path).read()
    store = b.CacheStore(cache_path)
    store.put("32x2048x1024:1,2,1,2,1", {"cls": "ok", "us_med": 1.0})
    after = open(out_path).read()
    assert after == before, "finalized OUT was modified by a CacheStore.put()"
    assert json.load(open(cache_path)), "CACHE was not written"
    return "finalized {corpus,cache} untouched; CACHE written separately"


def test_interrupted_sweep_resumes():
    """A resumed store loads prior records; a sweep loop skips completed keys."""
    d = _tmp()
    cache_path = os.path.join(d, "cache.json")
    s1 = b.CacheStore(cache_path)
    s1.put("k1", {"cls": "ok"})
    s1.put("k2", {"cls": "validation"})
    # New process: fresh store bound to same path.
    s2 = b.CacheStore(cache_path)
    assert "k1" in s2 and "k2" in s2, "resumed store did not load prior records"
    todo = ["k1", "k2", "k3"]
    ran = [k for k in todo if k not in s2]  # sweep only runs the not-yet-done keys
    assert ran == ["k3"], f"resume re-ran completed records: would run {ran}"
    return "resume loads prior records; only uncached keys re-run"


def test_interactive_cannot_clobber():
    """run_cfg(..., {}) with a plain dict must NOT write the on-disk CACHE (interactive = non-persistent)."""
    d = _tmp()
    cache_path = os.path.join(d, "cache.json")
    orig = b.CACHE
    b.CACHE = cache_path
    try:
        store = b.CacheStore(cache_path)
        store.put("sweep-key", {"cls": "ok", "us_med": 42.0})
        disk_before = open(cache_path).read()
        # Interactive call with a plain dict + an INFEASIBLE cfg (Sm>Mt) -> validation path, NO subprocess.
        plain = {}
        rec = b.run_cfg(256, 6144, 1024, (1, 1, 9, 1, 1), plain)  # Mt=8, Sm=9 -> infeasible
        assert rec["cls"] == "validation" and rec["reason"] == "Sm>Mt", f"unexpected rec {rec}"
        assert "256x6144x1024:1,1,9,1,1" in plain, "plain dict did not receive the record in-memory"
        disk_after = open(cache_path).read()
        assert disk_after == disk_before, "interactive run_cfg(...,{}) clobbered the on-disk CACHE"
    finally:
        b.CACHE = orig
    return "interactive plain-dict run_cfg leaves CACHE untouched; record stays in-memory"


def test_exit_124_is_hang(monkeypatched=None):
    """A worker that returns GNU-timeout code 124 must classify as 'hang' (not 'runtime') + reset device."""
    # classify_timeout unit checks
    assert b.classify_timeout(124) is True
    assert b.classify_timeout(137) is True
    assert b.classify_timeout(-15) is True
    assert b.classify_timeout(0) is False
    assert b.classify_timeout(134) is False  # SIGABRT = FATAL crash, stays 'runtime'
    assert b.classify_timeout(139) is False  # SIGSEGV crash

    # integration: fake subprocess.run so no device is touched
    calls = {"reset": 0}
    real_sub = b.subprocess

    class FakeCompleted:
        def __init__(self, rc):
            self.returncode, self.stdout, self.stderr = rc, "", ""

    def fake_run(cmd, **kw):
        if cmd and cmd[0] == "tt-smi":
            calls["reset"] += 1
            return FakeCompleted(0)
        return FakeCompleted(124)  # the worker "times out"

    b.subprocess = types.SimpleNamespace(run=fake_run, TimeoutExpired=real_sub.TimeoutExpired)
    try:
        rec = b.run_cfg(256, 6144, 1024, (1, 12, 1, 2, 1), {})  # feasible cfg -> reaches (faked) subprocess
        assert rec["cls"] == "hang", f"exit 124 classified as {rec['cls']}, expected hang"
        assert rec["returncode"] == 124
        assert calls["reset"] == 1, "device was not reset after inner timeout"
    finally:
        b.subprocess = real_sub
    return "exit 124 -> hang + device reset; 134/139 stay runtime"


def main():
    tests = [
        test_finalized_report_never_overwritten,
        test_interrupted_sweep_resumes,
        test_interactive_cannot_clobber,
        test_exit_124_is_hang,
    ]
    fails = 0
    for t in tests:
        try:
            msg = t()
            print(f"PASS {t.__name__}: {msg}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests)-fails}/{len(tests)} passed")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
