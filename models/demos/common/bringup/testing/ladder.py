# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Integration: stacked device layers over one ladder rung, against the golden.

Per rung (spec ``ladder`` entry): all chunks on device, or with ``prefix_from_golden`` the golden state of
[0, last chunk) loaded and only the last chunk on device. Device activations propagate; nothing is teacher-forced,
except that with a non-contiguous layer subset each contiguous run starts from the golden block input of its first
layer (the gaps never run on the device). A rung may set ``layers`` to stack fewer layers (one layer, a subset).

Metrics (last chunk unless noted):
    pcc_layer_L{i}       per-layer output trail
    pcc_state_min        worst state tensor over every layer and the whole sequence (and pcc_state_<name>_L{i})
    pcc_final_hidden, top1_match, top5_overlap, pcc_logits_tail     only when the stack ends at the model's last layer
    chunk_seconds_c{c}, prefill_seconds, model_load_s, covered_layers, subset,
    host_transfers_per_layer (warm chunks only: the most host round-trips inside one model.layer call; a deferred
        step's CPU bridge is not counted), deferred_cpu_steps, deferred_cpu_ms (the bridge's steps and host time, F46)
"""

from __future__ import annotations

import time

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.generate_golden import run_starts
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import cpu_bridge
from models.demos.common.bringup.testing.harness import threshold
from models.demos.common.bringup.testing.host_transfers import HostTransfers


def sampled_rows(chunk: int) -> torch.Tensor:
    return torch.unique(torch.cat([torch.arange(0, chunk, 16), torch.arange(chunk - 32, chunk)]))


def run_ladder(s, rung_name: str, mesh) -> dict:
    rung = s.rung(rung_name)
    g = Golden.for_rung(s, rung_name)
    layers = [i for i in (s.layers() if "layers" not in rung else rung["layers"]) if i in g.layers]
    starts = set(run_starts(layers))
    full_stack = layers == list(range(s.num_layers))
    ends_at_last = layers[-1] == s.num_layers - 1  # final norm, logits and top-k are only meaningful then
    tokens = g.tokens()
    chunk, n_chunks = rung["chunk"], rung["seq"] // rung["chunk"]
    last = n_chunks - 1

    model = s.hooks().device_model(mesh, s, layers, lm_head=ends_at_last)
    metrics.record("model_load_s", round(getattr(model, "load_seconds", 0.0), 1))
    metrics.record("device_model_hybrid", int("Hybrid" in type(model).__name__))  # dashboard: which model a row ran on
    metrics.record("rung_seq", rung["seq"])
    metrics.record("rung_chunk", rung["chunk"])
    metrics.record("covered_layers", len(layers))
    metrics.record("subset", int(not full_stack))
    state = model.new_state(rung["seq"])

    first = 0
    metrics.record("rung_start", (n_chunks - 1) * chunk if rung.get("prefix_from_golden") else 0)
    if rung.get("prefix_from_golden"):
        for i in layers:
            state.load_prefix(i, g.state(i, at=last * chunk), last * chunk)
        first = last

    trail, t_total, hidden = {}, 0.0, None
    cpu_bridge.STATS.reset()
    host = {}  # layer -> host round-trips inside model.layer on warm chunks (after the first chunk this rung runs)
    for c in range(first, n_chunks):
        s0 = c * chunk
        t0 = time.time()
        h = None
        for i in layers:
            if i in starts:
                if h is not None:
                    model.free(h)
                h = model.embed(tokens[s0 : s0 + chunk]) if i == 0 else model.from_host(g.layer(c, i)["in"].float())
            if c > first:
                with HostTransfers() as ht:
                    h2 = model.layer(i, h, s0, state)
                if ht.total >= host.get(i, (0, None))[0]:
                    host[i] = (ht.total, dict(ht.calls))
            else:
                h2 = model.layer(i, h, s0, state)
            model.free(h)
            h = h2
            if c == last:
                trail[i] = model.to_host(h).float()
        hidden = model.final_norm(h) if ends_at_last else h
        if ends_at_last:
            model.free(h)
        model.sync()
        dt = time.time() - t0
        t_total += dt
        metrics.record(f"chunk_seconds_c{c:02d}", round(dt, 3))
        print(f"chunk {c} [{s0},{s0 + chunk}) {dt:.2f}s")
        if c != last:
            model.free(hidden)
    metrics.record("prefill_seconds", round(t_total, 3))
    cpu_bridge.record(metrics)
    if cpu_bridge.STATS.steps:
        print(
            f"deferred to op-gen, on the CPU bridge: {cpu_bridge.STATS.step_names} ({cpu_bridge.STATS.ms:.0f} ms host)"
        )
    if host:
        worst = max(n for n, _ in host.values())
        metrics.record("host_transfers_per_layer", worst)
        bad = {i: calls for i, (n, calls) in host.items() if n}
        print(
            f"host transfers per layer (warm): max {worst}"
            + (f"; layers {sorted(bad)}: {bad[min(bad)]}" if bad else "")
        )

    out = {"trail": {}, "failed": []}
    for i in layers:
        p = metrics.pcc(trail[i], g.layer(last, i)["out"].float())
        metrics.record(f"pcc_layer_L{i:02d}", p)
        out["trail"][i] = p
        if p < threshold(s, "layer"):
            out["failed"].append(f"layer {i} pcc {p:.5f}")
    worst_layer = min(out["trail"].values())
    print(f"worst layer pcc {worst_layer:.6f}")

    if ends_at_last:
        gm = g.model(last)
        final = model.to_host(hidden).float()
        p = metrics.pcc(final, gm["final_norm"].float())
        metrics.record("pcc_final_hidden", p)
        rows = sampled_rows(chunk)
        lg = model.logits(hidden, rows).float()
        tt5 = lg.topk(5, dim=-1).indices
        gtop = gm["top32_ids"][rows].long()
        top1 = (tt5[:, 0] == gtop[:, 0]).float().mean().item()
        top5 = (tt5 == gtop[:, :1]).any(-1).float().mean().item()
        metrics.record("top1_match", top1)
        metrics.record("top5_overlap", top5)
        metrics.record("pcc_logits_tail", metrics.pcc(lg[-32:], gm["logits_tail"].float()))
        print(f"final hidden pcc {p:.6f} top1 {top1:.4f} top5 {top5:.4f}")
        if p < threshold(s, "final_hidden"):
            out["failed"].append(f"final hidden {p:.5f}")
        if top5 < threshold(s, "top5"):
            out["failed"].append(f"top5 {top5:.4f}")

    worst = 1.0
    for i in layers:
        got = state.to_torch(i, rung["seq"])
        for name, want in g.state(i).items():
            p = metrics.pcc(got[name].float(), want.float())
            metrics.record(f"pcc_state_{name}_L{i:02d}", p)
            worst = min(worst, p)
    metrics.record("pcc_state_min", worst)
    print(f"worst state pcc {worst:.6f}")
    if worst < threshold(s, "state"):
        out["failed"].append(f"state {worst:.5f}")
    return out
