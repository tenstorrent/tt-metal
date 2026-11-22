#!/usr/bin/env python3
"""C0 measurement shim (torrentci#3525): runs ctcache 1.2.0 (e33c063c,
vendored beside this file) unchanged, and appends one JSON line per
translation unit to $C0_LOG: digest, hit, stored, timings.

C0_READ_ONLY=1 drops every store (a read-only lease). C0_REF_DIR names a
second local cache whose membership is recorded (stale-marker hit ratio).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clang_tidy_cache as ctc  # noqa: E402

STATE = {"digest": None, "hit": None, "stored": False, "hash_s": None}
READ_ONLY = os.environ.get("C0_READ_ONLY") == "1"

_hash_inputs = ctc.hash_inputs


def _timed_hash(log, opts):
    start = time.monotonic()
    digest = _hash_inputs(log, opts)
    STATE["hash_s"] = round(time.monotonic() - start, 4)
    STATE["digest"] = digest
    return digest


ctc.hash_inputs = _timed_hash

_is_cached = ctc.ClangTidyCache.is_cached


def _recorded_is_cached(self, digest):
    hit = _is_cached(self, digest)
    STATE["hit"] = bool(hit)
    return hit


ctc.ClangTidyCache.is_cached = _recorded_is_cached

_store = ctc.ClangTidyCache.store_in_cache


def _recorded_store(self, digest):
    if READ_ONLY:
        return
    _store(self, digest)
    STATE["stored"] = True


ctc.ClangTidyCache.store_in_cache = _recorded_store


def main():
    start = time.monotonic()
    rc = ctc.main()
    record = {
        "digest": STATE["digest"],
        "hit": STATE["hit"],
        "stored": STATE["stored"],
        "rc": rc,
        "hash_s": STATE["hash_s"],
        "total_s": round(time.monotonic() - start, 4),
        "src": next((a for a in reversed(sys.argv[1:]) if a.endswith((".cpp", ".cc", ".c", ".cxx"))), None),
    }
    ref_dir = os.environ.get("C0_REF_DIR")
    if ref_dir and STATE["digest"]:
        digest = STATE["digest"]
        record["in_ref"] = os.path.isfile(os.path.join(ref_dir, digest[:2], digest[2:]))
    log_path = os.environ.get("C0_LOG")
    if log_path:
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
