# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F264 (owner 2026-11-12, DeepSeek-V4.1 FIX.26): golden state at the serving handoff point.

A model whose serving path stops a request before the engine's ``actual_end`` (DeepSeek-V4.1 P.49, PREFILL_END=cap:
prefill stops at decode's cap = actual_end // KV_BLOCK_SIZE x KV_BLOCK_SIZE and decode replays the rest) never computes
the state at ``actual_end``. The serving-contract test (testing/contract.py) then judges the state where the model
hands off, which the golden must hold. Two optional hooks say where:

    contract_handoffs(spec, length) -> [positions]   every position other than ``length`` a request of ``length``
                                                     tokens may stop at in any serving mode (the golden's job)
    contract_end(spec, length) -> int                where the model's configured serving mode stops (the test's job)

The golden at a handoff point ``p`` is made by the same pipeline as the generator's own snapshot at
``seq - tests.contract_tail_pad`` (generate_golden.main: the native reference, fp32, every chunk from position 0, the
last chunk ending early, the fixed state stored with store_dtype): it re-runs chunks 0 .. n-2 from position 0, checks
that the state at the last chunk's start equals the golden's stored snapshot and growing states byte for byte (the
same reference and code as the golden; a mismatch stops before anything is written), then runs [last start, p).
Files (``derived``: outside the golden's content_hash, so the pinned golden stays valid; integrity from
``handoff.json``):

    kv_cache/handoff_layer_{i}_at_{p}.safetensors  fixed-size state at p, keys as kv_cache/layer_{i}_at_{p}
    handoff.json                                   points, sha256 per file, the golden content_hash it extends, check
    handoff-drift-<rung>.json                      {str(p): drift entries}: the official reference fed as the device
                                                   is (every chunk from 0, the last ending at p) vs the native golden,
                                                   contract_state_<name>_L<i> (fixed states at p, growing ones over
                                                   [0, p rounded down to 32)), the same entries reference/drift.py
                                                   writes for seq - tail; the contract's limits at p come from it

    python -m models.demos.common.bringup.reference.handoff golden --spec S     # native snapshots (CPU, ~2 chunks)
    python -m models.demos.common.bringup.reference.handoff drift --spec S      # official drift (CPU, ~2 chunks)
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.freeze import sha256_file
from models.demos.common.bringup.core.metrics import cpu_threads

INDEX = "handoff.json"
PREFIX = "handoff"  # golden.derived(): every handoff file is outside the content hash


def snap_name(i: int, p: int) -> str:
    return f"kv_cache/{PREFIX}_layer_{i}_at_{p}.safetensors"


def drift_path(golden_dir, rung: str) -> Path:
    return Path(golden_dir) / f"{PREFIX}-drift-{rung}.json"


def tail_at(spec, seq: int) -> int:
    return seq - int(spec.get("tests.contract_tail_pad"))


def points(spec, length: int) -> list[int]:
    """The model's handoff points of a request of ``length`` tokens other than ``length`` (hooks.contract_handoffs)."""
    hooks = spec.hooks()
    if not hasattr(hooks, "contract_handoffs"):
        return []
    return sorted({int(p) for p in hooks.contract_handoffs(spec, int(length))} - {int(length)})


def contract_end(spec, length: int) -> int:
    """Where the model's configured serving mode stops a request of ``length`` tokens (hooks.contract_end)."""
    hooks = spec.hooks()
    return int(hooks.contract_end(spec, int(length))) if hasattr(hooks, "contract_end") else int(length)


def index(golden_dir) -> dict | None:
    p = Path(golden_dir) / INDEX
    return json.loads(p.read_text()) if p.exists() else None


def load_snapshot(golden, i: int, p: int) -> dict | None:
    """The handoff snapshot of layer i at p ({key: tensor}), sha256-checked against handoff.json and tied to the
    golden's content_hash; None when the golden has no handoff at p."""
    idx = index(golden.dir)
    rel = snap_name(i, p)
    if idx is None or rel not in idx["files"]:
        return None
    if idx["golden_content_hash"] != golden.manifest.get("content_hash"):
        raise ValueError(f"{golden.dir / INDEX}: made for golden {idx['golden_content_hash']}, not this one")
    f = golden.dir / rel
    if sha256_file(f) != idx["files"][rel]:
        raise ValueError(f"{f}: sha256 differs from {INDEX}; regenerate the handoff snapshots")
    return load_file(str(f))


def _chunks(ref, tokens, chunk: int, n: int, state) -> None:
    for c in range(n):
        s = c * chunk
        t0 = time.time()
        ref.forward_chunk(tokens[s : s + chunk], s, state, lambda name, t: None, logits_last_n=0)
        print(f"chunk {c} [{s}, {s + chunk}) {time.time() - t0:.0f}s", flush=True)


def golden(spec) -> dict:
    """The native snapshots at every handoff point of the contract request (see the module docstring)."""
    from models.demos.common.bringup.reference.golden import Golden, reference_at, rung_dir, store_dtype, write_manifest
    from models.demos.common.bringup.testing.contract import contract_rung

    torch.set_num_threads(cpu_threads())
    t0 = time.time()
    rung = spec.rung(contract_rung(spec))
    g = Golden(rung_dir(spec, rung))  # the shared copy: written there, mirrored by `reference.golden sync`
    if g.manifest.get("numerics") != "native":
        raise SystemExit(f"{g.dir}: numerics {g.manifest.get('numerics')!r}, want native (F128)")
    seq, chunk, n = g.seq, g.chunk, g.n_chunks
    last = (n - 1) * chunk
    pts = points(spec, tail_at(spec, seq))
    bad = [p for p in pts if not last < p <= seq]
    if bad:
        raise SystemExit(f"handoff points {bad} outside the last chunk ({last}, {seq}]: not supported")
    have = set(g.meta.get("state_snapshots", [])) | set((index(g.dir) or {}).get("points", []))
    pts = [p for p in pts if p not in have]
    if not pts:
        print("no handoff point the golden does not already hold")
        return {}
    selected, fixed = g.layers, spec.state_fixed
    full_stack = max(selected) == spec.num_layers - 1
    run_layers = None if full_stack else list(range(max(selected) + 1))
    ref = reference_at(spec, "native", layers=run_layers, dtype=torch.float32)  # as generate_golden.main
    tokens = g.tokens()
    state = ref.new_state(seq)
    _chunks(ref, tokens, chunk, n - 1, state)

    # the same reference and code as the golden: its state at the last chunk's start, byte for byte
    diffs = []
    for i in selected:
        want = g.state(i, at=last)
        got = ref.state_tensors(state, i, last)
        for name, w in want.items():
            o = store_dtype(got[name].clone(), True, name)
            if name not in fixed:  # growing: the final golden over the positions before the last chunk
                w = w[: o.shape[0]]
            if o.shape != w.shape or not torch.equal(o, w.to(o.dtype)):
                diffs.append(f"L{i} {name}")
    if diffs:
        raise SystemExit(f"the reference no longer reproduces the golden's state at {last}: {diffs[:8]}; not written")
    print(f"state at {last} equals the golden's ({len(selected)} layers, fixed and growing)", flush=True)

    files = {}
    for p in pts:
        st = copy.deepcopy(state) if len(pts) > 1 else state
        tc = time.time()
        ref.forward_chunk(tokens[last:p], last, st, lambda name, t: None, logits_last_n=0)
        print(f"last chunk [{last}, {p}) {time.time() - tc:.0f}s", flush=True)
        for i in selected:
            names = [x for x in spec.state_names(i) if x in fixed]
            if names:
                tens = ref.state_tensors(st, i, p)
                rel = snap_name(i, p)
                save_file(
                    {f"{x}_cache_layer_{i}": store_dtype(tens[x].clone(), True, x) for x in names}, str(g.dir / rel)
                )
                files[rel] = sha256_file(g.dir / rel)
    old = index(g.dir) or {"files": {}, "points": []}
    rec = {
        "golden_content_hash": g.manifest["content_hash"],
        "rung": rung["name"],
        "request_length": tail_at(spec, seq),
        "points": sorted(set(old["points"]) | set(pts)),
        "files": {**old["files"], **files},
        "pipeline": "reference.handoff golden: generate_golden's native reference (fp32, keep_fp32), chunks from 0, "
        "the last ending at the point; state at the last chunk start checked equal to the golden's",
        "checked_at": last,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cpu_seconds": round(time.time() - t0, 1),
    }
    (g.dir / INDEX).write_text(json.dumps(rec, indent=1, sort_keys=True))
    write_manifest(g.dir)  # F260: the mirror then syncs it
    metrics.record("handoff_points", len(rec["points"]))
    metrics.record("cpu_seconds", rec["cpu_seconds"])
    print(f"handoff snapshots at {pts} -> {g.dir} ({len(files)} files) in {rec['cpu_seconds']:.0f}s")
    return rec


def drift(spec) -> dict:
    """The official reference fed as the device is (every chunk from 0, the last ending at the handoff point) vs the
    native golden: the contract's limits at the point (reference/drift.py's rule, F128 / F134)."""
    from models.demos.common.bringup.reference import drift as D
    from models.demos.common.bringup.reference.golden import Golden, reference_at, rung_dir
    from models.demos.common.bringup.testing.contract import contract_rung
    from models.demos.common.bringup.testing.ladder import rung_layers  # noqa: E402

    torch.set_num_threads(cpu_threads())
    rung = spec.rung(contract_rung(spec))
    out_dir = rung_dir(spec, rung)
    g = Golden(out_dir)  # the shared copy: the native handoff snapshots are written there
    layers = rung_layers(spec, rung, g)
    dtype = getattr(torch, str(spec.get("reference.shipped_dtype")))
    run_layers = None if layers[-1] == spec.num_layers - 1 else list(range(max(layers) + 1))
    ref = reference_at(spec, "official", layers=run_layers, dtype=dtype)
    p_out = drift_path(out_dir, rung["name"])
    out = json.loads(p_out.read_text()) if p_out.exists() else {}
    for p in points(spec, tail_at(spec, g.seq)):
        if str(p) in out:  # (FIX.28: as `handoff golden`, a point the file already holds is not recomputed)
            print(f"handoff {p}: drift already held", flush=True)
            continue
        t0 = time.time()
        rep = D.replay(spec, ref, g, layers, 0, p, full_last=False)
        _wait(g, p)  # `handoff golden` may still be running (the two jobs run side by side)
        out[str(p)] = D.contract_entries(spec, ref, g, layers, rep, p, growing="tail")
        print(f"handoff {p}: {len(out[str(p)])} drift entries in {time.time() - t0:.0f}s", flush=True)
    p_out.write_text(json.dumps(out, indent=1, sort_keys=True))
    from models.demos.common.bringup.reference.golden import write_manifest

    write_manifest(out_dir)
    metrics.record("handoff_drift_metrics", sum(len(v) for v in out.values()))
    return out


def _wait(g, p: int, timeout_s: int = 6 * 3600) -> None:
    t0 = time.time()
    while p not in (index(g.dir) or {}).get("points", []):
        if time.time() - t0 > timeout_s:
            raise SystemExit(f"{g.dir}: no handoff snapshot at {p} after {timeout_s}s; run `handoff golden` first")
        time.sleep(60)


def load_drift(golden_dir, rung: str, p: int) -> dict:
    """The drift entries at handoff point p (testing/e2e_limits.state_limit reads them like drift-<rung>.json)."""
    f = drift_path(golden_dir, rung)
    if not f.exists() or str(int(p)) not in (d := json.loads(f.read_text())):
        raise FileNotFoundError(
            f"{f}: no drift at the handoff point {p}; python -m models.demos.common.bringup.reference.handoff drift "
            "(F264)"
        )
    return d[str(int(p))]


def main(argv=None) -> None:
    from models.demos.common.bringup.reference.golden import load_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("golden", "drift"))
    ap.add_argument("--spec")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    golden(spec) if a.what == "golden" else drift(spec)


if __name__ == "__main__":
    main()
