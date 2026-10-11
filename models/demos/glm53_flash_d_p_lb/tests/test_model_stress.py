# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Whole-model determinism + hang stress: all 45 layers, real weights, the LoudBox mesh, for GLM_STRESS_HOURS hours.

Passes of chunked prefill: each pass picks a token stream (the canonical prompt, or the same tokens reversed) and a
length L of 1 .. target.seq / target.chunk chunks, and runs chunks 0 .. L-1 from position 0 (KDA state and the MLA /
indexer caches start fresh at position 0, so a position's output depends only on the stream). Every pass therefore
revisits (stream, position) pairs, restarting at random points.

Everything stays on device (as tests/test_flat_stress.py): the chunk ids of both streams are uploaded once; the first
output (the last layer's residual, the split layout) of each (stream, position) is cloned as its reference; every later
output of the pair is compared bit-exactly with it on device (ne -> max into a per-position sticky marker; each chip
reduces only its own rows). The markers (one [1,1,1,1] per position and chip) are read once per pass; any nonzero is a
non-determinism, reported with the pass, stream and position. Every 7th chunk a matmul runs in between (other traffic).
Hangs: the dispatch timeout of scripts/run_safe_pytest.sh (triage dump, device reset).

  TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    scripts/run_safe_pytest.sh --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_model_stress.py -s

GLM_STRESS_HOURS (default 3), GLM_STRESS_SEED (default 1), GLM_STRESS_LOG (a progress log file, appended), GLM_STRESS_FAIL
(1 = stop at the first mismatch, default; 0 = keep going and count)."""

import os
import random
import time

import torch

import ttnn
from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec

S = spec()
pytestmark = device_timeout(S)
HOURS = float(os.environ.get("GLM_STRESS_HOURS", "3"))
SEED = int(os.environ.get("GLM_STRESS_SEED", "1"))
LOG = os.environ.get("GLM_STRESS_LOG")
FAIL_FAST = os.environ.get("GLM_STRESS_FAIL", "1") == "1"


def _log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")


@mesh_parametrize
def test_model_stress(mesh_device):
    from models.demos.common.bringup.reference.prompt import tokens as prompt_tokens

    seq, chunk = int(S.get("target.seq")), int(S.get("target.chunk"))
    n_pos = seq // chunk
    layers = S.layers()
    t0 = time.time()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=False)
    tt = model.model
    _log(f"[stress] loaded {len(layers)} layers in {time.time() - t0:.0f} s; {n_pos} positions x {chunk}, {HOURS} h")

    toks = prompt_tokens(S, seq).to(torch.long)[:seq]
    streams = {"prompt": toks, "reversed": toks.flip(0)}
    ids = {
        (name, k): ttnn.from_torch(
            t[k * chunk : (k + 1) * chunk].reshape(1, 1, -1).to(torch.int64).to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        for name, t in streams.items()
        for k in range(n_pos)
    }
    rep = lambda v: ttnn.from_torch(  # noqa: E731
        torch.full((1, 1, 32, 32), v, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    markers = {key: rep(0.0) for key in ids}
    noise_a, noise_b = rep(0.5), rep(0.25)
    refs = {}

    def chunk_out(name, k):
        h = tt.embed(ids[(name, k)])
        for blk in tt.blocks:
            h2 = blk(h, k * chunk)
            ttnn.deallocate(h)
            h = h2
        return h

    def check(key, out):
        if key not in refs:
            refs[key] = ttnn.clone(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return
        ne = ttnn.ne(out, refs[key])
        m = ttnn.max(ttnn.max(ne, dim=-1, keepdim=True), dim=-2, keepdim=True)
        ttnn.deallocate(ne)
        m32 = ttnn.pad(m, [(0, 0), (0, 0), (0, 31), (0, 31)], 0.0) if tuple(m.shape)[-1] != 32 else m
        new = ttnn.maximum(markers[key], m32)
        ttnn.deallocate(markers[key])
        if m32 is not m:
            ttnn.deallocate(m32)
        ttnn.deallocate(m)
        markers[key] = new

    # negative control: the comparator must flag a changed output (position 0 of the prompt, +1 on every element)
    key = ("prompt", 0)
    out = chunk_out(*key)
    check(key, out)
    ttnn.deallocate(out)
    out = chunk_out(*key)
    bumped = ttnn.add(out, 1.0)
    check(key, bumped)
    ttnn.deallocate(out)
    ttnn.deallocate(bumped)
    flagged = [float(ttnn.to_torch(t).flatten()[0]) for t in ttnn.get_device_tensors(markers[key])]
    assert all(v != 0.0 for v in flagged), f"the comparator missed a changed output on chips {flagged}"
    ttnn.deallocate(markers[key])
    markers[key] = rep(0.0)
    out = chunk_out(*key)  # and passes an unchanged one
    check(key, out)
    ttnn.deallocate(out)
    clean = [float(ttnn.to_torch(t).flatten()[0]) for t in ttnn.get_device_tensors(markers[key])]
    assert all(v == 0.0 for v in clean), f"position 0 not deterministic on its second run: {clean}"
    _log("[stress] comparator check: a changed output is flagged on every chip, an unchanged one is not")

    rng = random.Random(SEED)
    deadline = time.time() + HOURS * 3600
    passes = chunks = mismatches = 0
    seen = {key: 0 for key in ids}
    t_start = time.time()
    while time.time() < deadline:
        name = rng.choice(list(streams))
        L = rng.randint(1, n_pos)
        tp = time.time()
        for k in range(L):
            out = chunk_out(name, k)
            check((name, k), out)
            ttnn.deallocate(out)
            seen[(name, k)] += 1
            chunks += 1
            if chunks % 7 == 0:  # other traffic in between
                ttnn.deallocate(ttnn.matmul(noise_a, noise_b))
        # the markers of this pass's positions, read once (8 chips x one tile each)
        bad = []
        for k in range(L):
            vals = [float(ttnn.to_torch(t).flatten()[0]) for t in ttnn.get_device_tensors(markers[(name, k)])]
            if any(v != 0.0 for v in vals):
                bad.append((k, [c for c, v in enumerate(vals) if v != 0.0]))
        passes += 1
        if bad:
            mismatches += len(bad)
            _log(f"[stress] MISMATCH pass {passes} stream {name}: (position, chips) {bad}")
            for k, _ in bad:  # re-arm so a later recurrence is seen again
                ttnn.deallocate(markers[(name, k)])
                markers[(name, k)] = rep(0.0)
            if FAIL_FAST:
                break
        if passes % 10 == 0 or bad:
            el = time.time() - t_start
            _log(
                f"[stress] pass {passes} ({name}, {L} chunks, {time.time() - tp:.1f} s) | {chunks} chunks in "
                f"{el / 60:.1f} min ({chunks / el:.2f} chunks/s) | mismatches {mismatches} | min visits per position "
                f"{min(seen.values())}"
            )
    _log(
        f"[stress] done: {passes} passes, {chunks} chunks in {(time.time() - t_start) / 3600:.2f} h, "
        f"{mismatches} mismatching (pass, position) reports; visits per (stream, position) "
        f"{min(seen.values())} .. {max(seen.values())}"
    )
    assert mismatches == 0, f"{mismatches} non-deterministic (pass, position) reports"
