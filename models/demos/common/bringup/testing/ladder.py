# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Integration: stacked device layers over one ladder rung, against the golden.

Per rung (spec ``ladder`` entry): all chunks on device, or with ``prefix_from_golden`` the golden state of
[0, last chunk) loaded and only the last chunk on device. Device activations propagate; nothing is teacher-forced,
except that with a non-contiguous layer subset each contiguous run starts from the golden block input of its first
layer (the gaps never run on the device). A rung may set ``layers`` to stack fewer layers (one layer, a subset).

Metrics (last chunk unless noted):
    err_layer_L{i}       per-layer output rel L2 (gated, F135; pcc_layer_L{i} the PCC trail, a diagnostic)
    err_state_<name>_L{i}, overlap_state_<name>_L{i}   every state tensor (float: rel L2; index lists: overlap), over the
                         positions the device computed (F134); err_state_max, pcc_state_min, overlap_state_min
    each gated metric also records <metric>_limit: calib x the reference's own drift from full precision (F128,
    testing/e2e_limits.py); the golden is the full-precision reference
    state_bits_<name>    bits per element of each state tensor the ladder ran on (state.formats(), optional hook)
    err_final_hidden, top5_overlap (gated), pcc_final_hidden, top1_match, pcc_logits_tail   only when the stack ends
                         at the model's last layer
    chunk_seconds_c{c}, prefill_seconds, model_load_s, covered_layers, subset,
    host_transfers_per_layer (warm chunks only: the most host round-trips inside one model.layer call; a deferred
        step's CPU bridge is not counted), deferred_cpu_steps, deferred_cpu_ms (the bridge's steps and host time, F46)
"""

from __future__ import annotations

STATE_BITS = {"bfloat16": 16, "float32": 32, "bfloat8_b": 8, "bfloat4_b": 4}

import time

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.generate_golden import run_starts
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import cpu_bridge
from models.demos.common.bringup.testing import e2e_limits as E
from models.demos.common.bringup.testing.harness import begin_chunk
from models.demos.common.bringup.testing.host_transfers import HostTransfers


def rung_layers(s, rung: dict, g: Golden) -> list[int]:
    """The layers a rung stacks: its own ``layers`` or the spec's subset, those the golden holds."""
    return [i for i in (s.layers() if "layers" not in rung else rung["layers"]) if i in g.layers]


def sampled_rows(chunk: int) -> torch.Tensor:
    return torch.unique(torch.cat([torch.arange(0, chunk, 16), torch.arange(chunk - 32, chunk)]))


def run_ladder(s, rung_name: str, mesh) -> dict:
    rung = s.rung(rung_name)
    g = Golden.for_rung(s, rung_name)
    layers = rung_layers(s, rung, g)
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
    metrics.record("scaled_run", int(getattr(s, "scaled", False)))  # F64: the reduced model of a scaled run
    state = model.new_state(rung["seq"])
    # F58: the KV format the ladder actually runs on, so a gate can require the served one (one format for the
    # ladder, the contract and serving). Optional hook: state.formats() -> {name: ttnn dtype}; bits per element.
    for name, dt in (state.formats() if hasattr(state, "formats") else {}).items():
        metrics.record(f"state_bits_{name}", STATE_BITS.get(str(dt).split(".")[-1].lower(), -1))

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
        begin_chunk(model, tokens[s0 : s0 + chunk], s0)  # F65
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
    drift = E.load(g.dir, rung_name)  # F128 / F134: calib x the reference's drift, fed as the device is on this rung
    for i in layers:
        want = g.layer(last, i)["out"].float()
        p = metrics.pcc(trail[i], want)
        metrics.record(f"pcc_layer_L{i:02d}", p)  # F135: a diagnostic; the rel error gates
        out["trail"][i] = p
        e = rel(trail[i], want)
        metrics.record(f"err_layer_L{i:02d}", e)
        E.check(drift, f"layer_L{i:02d}", f"err_layer_L{i:02d}", e, out["failed"])
    worst_layer = min(out["trail"].values())
    print(f"worst layer pcc {worst_layer:.6f}")

    if ends_at_last:
        gm = g.model(last)
        final = model.to_host(hidden).float()
        p = metrics.pcc(final, gm["final_norm"].float())
        metrics.record("pcc_final_hidden", p)
        e = rel(final, gm["final_norm"])
        metrics.record("err_final_hidden", e)
        E.check(drift, "final_hidden", "err_final_hidden", e, out["failed"])
        rows = sampled_rows(chunk)
        lg = model.logits(hidden, rows).float()
        tt5 = lg.topk(5, dim=-1).indices
        gtop = gm["top32_ids"][rows].long()
        top1 = (tt5[:, 0] == gtop[:, 0]).float().mean().item()
        top5 = (tt5 == gtop[:, :1]).any(-1).float().mean().item()
        metrics.record("top1_match", top1)
        metrics.record("top5_overlap", top5)
        metrics.record("pcc_logits_tail", metrics.pcc(lg[-32:], gm["logits_tail"].float()))
        print(f"final hidden pcc {p:.6f} rel {e:.6f} top1 {top1:.4f} top5 {top5:.4f}")
        E.check(drift, "top5", "top5_overlap", top5, out["failed"])

    check_states(state, g, layers, rung["seq"], drift, out["failed"], s.get("tests.waivers"))
    return out


def rel(got: torch.Tensor, want: torch.Tensor) -> float:
    """rel L2 of ``got`` vs ``want`` in float64 (F135: the gated end-to-end error, as per step)."""
    w = want.double()
    return float((got.double().reshape(w.shape) - w).norm() / w.norm().clamp_min(1e-300))


def _state_check(drift: dict, key: str, name: str, value: float, failed: list[str], waivers: dict | None) -> bool:
    """E.check of err_<key>; over its limit with an owner waiver ``state.<name>`` (F226): WAIVED, not failed."""
    n = len(failed)
    if E.check(drift, key, f"err_{key}", value, failed):
        return True
    why = str((waivers or {}).get(f"state.{name}") or "").strip()
    if not why:
        return False
    print(f"WAIVED {failed.pop(n)}; owner waiver: {why}")
    metrics.record(f"err_{key}_waived", 1)
    return True


def check_states(
    state, g: Golden, layers: list[int], seq: int, drift: dict, failed: list[str], waivers: dict | None = None
) -> None:
    """Every layer's state after the rung vs the golden, each with its drift limit (F128), over the positions the
    device computed (the drift entry's region, F134), gated as err_state_<name>_L<i>: float states by rel L2 (F135;
    PCC recorded), integer states (top-k index lists) by the missed fraction 1 - overlap (F99, F138; overlap recorded),
    never the PCC of the ids. F226: an owner waiver ``state.<name>`` in spec tests.waivers (``waivers``) passes a
    state over its limit: printed WAIVED with the reason and recorded <metric>_waived = 1, never hidden."""
    from models.demos.common.bringup.reference.drift import narrow

    worst, worst_err, worst_ov = 1.0, 0.0, 1.0
    for i in layers:
        got = state.to_torch(i, seq)
        for name, want in g.state(i).items():
            key = f"state_{name}_L{i:02d}"
            region = (drift.get(key) or {}).get("region")  # F134: the positions the device computed
            o = got[name].reshape(want.shape) if got[name].numel() == want.numel() else got[name]
            o, want = narrow(o, region), narrow(want, region)
            if want.is_floating_point():
                p, e = metrics.pcc(o.float(), want.float()), rel(o, want)
                metrics.record(f"pcc_{key}", p)
                metrics.record(f"err_{key}", e)
                worst, worst_err = min(worst, p), max(worst_err, e)
                _state_check(drift, key, name, e, failed, waivers)
            else:  # the missed fraction, as per step (F81): every state gates as err_state_* (F138)
                ov = index_overlap(o, want)
                metrics.record(f"overlap_{key}", ov)
                metrics.record(f"err_{key}", 1.0 - ov)
                worst_ov, worst_err = min(worst_ov, ov), max(worst_err, 1.0 - ov)
                _state_check(drift, key, name, 1.0 - ov, failed, waivers)
    metrics.record("pcc_state_min", worst)
    metrics.record("err_state_max", worst_err)
    metrics.record("overlap_state_min", worst_ov)
    print(f"worst state rel {worst_err:.6f} (pcc {worst:.6f}), worst index-state overlap {worst_ov:.6f}")


def index_overlap(got: torch.Tensor, want: torch.Tensor) -> float:
    """Mean per-row share of the golden's selected ids (>= 0; negatives are padding) that the device also selected.
    Plain, not tie-aware: the golden holds no selection scores (a full rung's would be [tokens, keys] per layer); the
    component and swap tests carry the tie-aware recall (F101)."""
    k = want.shape[-1]
    w, gs = want.long().reshape(-1, k), got.long().reshape(-1, k).sort(-1).values
    hit = (gs.gather(-1, torch.searchsorted(gs, w).clamp(max=k - 1)) == w) & (w >= 0)
    valid = (w >= 0).sum(-1)
    rows = valid > 0
    return float((hit.sum(-1)[rows] / valid[rows]).mean()) if bool(rows.any()) else 1.0
