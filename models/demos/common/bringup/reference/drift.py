# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F128 (owner): the end-to-end margin; F134: measured on the comparison it gates. The reference at its own shipped
precision (numerics official, ``reference.shipped_dtype``) is run the way the device runs the rung (testing/ladder.py):
the golden's state prefix loaded for a ``prefix_from_golden`` rung, each run of a gapped layer subset started from the
golden block input of its first layer; its layer outputs, states, final hidden and top-5 are compared with the
full-precision golden (the target) by the metrics the ladder uses. The device may then miss the golden by calib x that
drift (testing/e2e_limits.py), never by a fixed bar.

    python -m models.demos.common.bringup.reference.drift --rung s55296

Writes ``drift-<rung>.json`` into the full-precision golden, one entry per compared tensor (``layer_L<i>``,
``state_<name>_L<i>``, ``final_hidden``, ``top5``): a float tensor {"rel", "rel_res", "pcc", "pcc_res", "region"}, a
selection {"overlap", "overlap_res", "region"}; testing/e2e_limits.py turns them into limits (F135: rel L2 gated, the
same calib convention as per step). A float tensor's floor (_res) is the metric for the target rounded once to the
dtype the official golden would store it in (bf16, or fp32 for golden.keep_fp32; F133), an overlap's one miss among
the selected entries (rows for top-5). A state of a prefix rung is compared over the positions the device computed
(``region``, along the axis the reference's state_tensors slices; the ladder applies the same region). On the contract
rung, ``contract_state_<name>_L<i>`` entries: the state as the contract reads it back (every chunk, the last ending
``tests.contract_tail_pad`` early; growing states over [0, that length rounded down to 32), fixed ones at it).
Records drift_<rung>_<metric> and drift_metrics for the gate. A G task's pass pins every drift file into the frozen
block of each task gated on an end-to-end limit (pin()): a drift changed outside its G task fails those gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.reference.golden import Golden, load_spec, reference_at, store_dtype

BLOCK = 32  # the contract reads its KV back in 32-token blocks (testing/contract.py)
# a gate on an end-to-end limit (ladder, quick profile, contract): its task pins the drift files (F134)
E2E_METRIC = re.compile(r"^(pcc|err|overlap)_(layer|state|final_hidden|producer_kv|contract_state)|^top5_overlap")


def drift_path(golden_dir, rung: str) -> Path:
    return Path(golden_dir) / f"drift-{rung}.json"


def stored(t: torch.Tensor) -> str:
    """The dtype the official golden stores a tensor in (golden.store_dtype): fp32 or bf16."""
    return "fp32" if t.dtype == torch.float32 else "bf16"


def narrow(t: torch.Tensor, region) -> torch.Tensor:
    return t if not region else t.narrow(region[0], region[1], region[2] - region[1])


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().reshape(b.shape), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def _float_entry(official: torch.Tensor, target: torch.Tensor, region=None) -> dict:
    """F135: rel L2 (gated) and PCC (the contract's read-back, diagnostics), each with one rounding's floor."""
    from models.demos.common.bringup.testing.noise_floor import round_to

    t = narrow(target.float(), region)
    o = narrow(official.reshape(target.shape), region)
    r = round_to(t, stored(o))
    if not t.abs().max().item():  # F159: an all-zero target (a state still empty): exact or not, no rounding margin
        return {
            "rel": 0.0 if not o.abs().max().item() else 1.0,
            "rel_res": 0.0,
            "pcc": 1.0,
            "pcc_res": 1.0,
            "region": region,
            "zero_target": True,
        }
    return {
        "rel": rel(o, t),
        "rel_res": rel(r, t),
        "pcc": metrics.pcc(o.float(), t),
        "pcc_res": metrics.pcc(r, t),
        "region": region,
    }


def _overlap_res(ids: torch.Tensor) -> float:
    """One miss among the selected entries (>= 0; negatives are padding), as the per-step index limits."""
    return 1.0 - 1.0 / max(int((ids >= 0).sum().item()), 1)


def _region(full: torch.Tensor, part: torch.Tensor, lo: bool) -> list | None:
    """The positions one slice of the reference's state covers: the axis on which state_tensors at two lengths
    differ; [axis, size at the shorter length, size at the longer] (lo) or [axis, 0, size at the shorter]."""
    axes = [d for d in range(full.dim()) if full.shape[d] != part.shape[d]]
    if not axes:
        return None
    if len(axes) > 1 or full.dim() != part.dim():
        raise SystemExit(f"state_tensors slices {tuple(full.shape)} -> {tuple(part.shape)}: not one position axis")
    a = axes[0]
    return [a, part.shape[a], full.shape[a]] if lo else [a, 0, part.shape[a]]


def replay(s, ref, g: Golden, layers: list[int], first: int, tail_at: int | None = None) -> dict:
    """The official reference over the golden's tokens as the ladder feeds the device: from chunk ``first`` (the
    golden's state prefix loaded before it), each run of the layer subset after a gap from the golden block input of
    its first layer (written into the block input forward_chunk records, which run_block hands to the steps).
    ``tail_at``: also the last chunk ending there, from the state before it (the contract)."""
    from models.demos.common.bringup.reference.generate_golden import run_starts

    seq, chunk = g.seq, g.chunk
    last, tokens = g.n_chunks - 1, g.tokens()
    starts = set(run_starts(layers)) - {0}
    ends_at_last = layers[-1] == s.num_layers - 1
    state = ref.new_state(seq)
    if first:
        for i in layers:
            ref.load_state(state, i, g.state(i, at=first * chunk), first * chunk)
    out, fed, firsts, ins = {"trail": {}}, set(), {}, {}
    out["first_step"] = {i: ref.block_graph(i)[0] for i in starts}

    def run(c: int, end: int, st, keep: bool):
        s0 = c * chunk

        def rec(name, t):
            if not name.startswith("L"):
                if keep and name == "final_norm":
                    out["final_norm"] = t.detach().clone()
                return
            li, key = name[1:].split(".", 1)
            li = int(li)
            if key == "in" and li in starts:
                t.copy_(g.layer(c, li)["in"][: end - s0].reshape(t.shape).to(t.dtype))
                fed.add((c, li))
            if keep and li in layers and key == "out":
                out["trail"][li] = t.detach().clone()
            if keep and key == "in" and li in starts:
                ins[li] = t.detach().clone()
            if keep and li in starts and key == out["first_step"][li].output:
                firsts[li] = t.detach().clone()

        _, logits = ref.forward_chunk(tokens[s0:end], s0, st, rec, logits_last_n=(end - s0) if ends_at_last else 0)
        if keep and logits is not None:
            out["logits"] = logits.float()

    before = None
    for c in range(first, g.n_chunks):
        if tail_at is not None and c == last:
            before = copy.deepcopy(state)
        run(c, (c + 1) * chunk, state, c == last)
    missing = sorted({li for li in starts} - {li for _, li in fed})
    if missing:
        raise SystemExit(f"forward_chunk recorded no block input L<i>.in for layers {missing}: cannot feed them (F134)")
    for li in starts:  # the feed must reach the steps: the first step from the golden input gives the recorded output
        st0 = out["first_step"][li]
        if tuple(st0.inputs) != ("in",) or st0.stateful or li not in firsts:
            print(f"note: layer {li}'s first step {st0.name} reads more than its block input; the feed is not checked")
            continue
        ctx = ref.chunk_context(li, last * chunk, chunk, state)
        ctx.extra.setdefault("tokens", tokens[last * chunk : (last + 1) * chunk])
        y = ref.component(li, st0.name)(ctx, ins[li])
        if not torch.equal(y.reshape(firsts[li].shape), firsts[li]):
            raise SystemExit(
                f"layer {li}: the reference's forward_chunk does not run the block from the block input it records, so "
                "a gapped subset cannot be replayed; run each block through run_block (reference/interface.py; F134)"
            )
    out["state"] = state
    if before is not None:
        run(last, tail_at, before, False)
        out["tail_state"] = before
    return out


def entries(s, ref, g: Golden, layers: list[int], rep: dict, first: int) -> dict:
    """The ladder's metrics, the replay against the target; states of a prefix rung over the positions computed."""
    from models.demos.common.bringup.testing.ladder import index_overlap, sampled_rows

    keep, fixed = tuple(s.get("golden.keep_fp32", [])), set(g.state_fixed)
    last, seq, start = g.n_chunks - 1, g.seq, first * g.chunk
    out = {}
    for i in layers:
        out[f"layer_L{i:02d}"] = _float_entry(store_dtype(rep["trail"][i], keep, "out"), g.layer(last, i)["out"])
        got = ref.state_tensors(rep["state"], i, seq)
        pre = ref.state_tensors(rep["state"], i, start) if start else {}
        for name, want in g.state(i).items():
            region = _region(got[name], pre[name], True) if start and name not in fixed else None
            o = got[name]
            if want.is_floating_point():
                out[f"state_{name}_L{i:02d}"] = _float_entry(store_dtype(o, keep, name), want, region)
            else:
                w, o = narrow(want, region), narrow(o.reshape(want.shape), region)
                out[f"state_{name}_L{i:02d}"] = {
                    "overlap": index_overlap(o, w),
                    "overlap_res": _overlap_res(w),
                    "region": region,
                }
    if layers[-1] == s.num_layers - 1:
        gm = g.model(last)
        out["final_hidden"] = _float_entry(store_dtype(rep["final_norm"], keep, "final_norm"), gm["final_norm"])
        rows = sampled_rows(g.chunk)
        top1 = gm["top32_ids"][rows][:, 0].long()
        top5 = rep["logits"][rows].topk(5, dim=-1).indices
        out["top5"] = {
            "overlap": float((top5 == top1[:, None]).any(-1).float().mean()),
            "overlap_res": 1.0 - 1.0 / max(top1.numel(), 1),
            "region": None,
        }
    return out


def contract_entries(s, ref, g: Golden, layers: list[int], rep: dict, tail_at: int) -> dict:
    """The states as the contract reads them back: growing ones over [0, tail_at rounded down to BLOCK) of the full
    run, fixed ones after the last chunk ending at tail_at."""
    keep, fixed = tuple(s.get("golden.keep_fp32", [])), set(g.state_fixed)
    length = tail_at - tail_at % BLOCK
    out = {}
    for i in layers:
        full = ref.state_tensors(rep["state"], i, g.seq)
        part = ref.state_tensors(rep["state"], i, length)
        tail = ref.state_tensors(rep["tail_state"], i, tail_at)
        want_at = g.state(i, at=tail_at) if fixed & set(full) else {}
        for name, want in g.state(i).items():
            if not want.is_floating_point():
                continue
            if name in fixed:
                o, w, region = tail[name], want_at[name], None
            else:
                region = _region(full[name], part[name], False)
                o, w = full[name], want
            out[f"contract_state_{name}_L{i:02d}"] = _float_entry(store_dtype(o, keep, name), w, region)
    return out


def compute_hf(s, rung_name: str) -> dict:
    """F136, ``reference.margin_source: hf_official`` (the owner's choice for a reference that cannot run at its
    shipped precision): the official modelling code (hooks.hf_model) at reference.shipped_dtype, one pass over the
    rung's tokens, each run after a gap started from the golden block input; layer outputs of the last chunk and
    top-5 only (no states, no final hidden: their gates then stop). Not matched on a prefix rung (no golden prefix)."""
    from models.demos.common.bringup.reference.check_hf import hf_layers, load_hf
    from models.demos.common.bringup.reference.generate_golden import run_starts
    from models.demos.common.bringup.testing.ladder import rung_layers, sampled_rows

    rung = s.rung(rung_name)
    g = Golden.for_rung(s, rung_name)
    layers = rung_layers(s, rung, g)
    ends_at_last = layers[-1] == s.num_layers - 1
    model = load_hf(s, None if ends_at_last else max(layers) + 1).to(
        getattr(torch, str(s.get("reference.shipped_dtype")))
    )
    blocks, outs, handles = hf_layers(s, model), {}, []
    for li in set(run_starts(layers)) - {0}:
        x = torch.cat([g.layer(c, li)["in"] for c in range(g.n_chunks)]).float()

        def feed(m, args, kwargs, x=x):
            h = args[0] if args else kwargs["hidden_states"]
            h = x.reshape(h.shape).to(h.dtype)
            return ((h, *args[1:]), kwargs) if args else (args, {**kwargs, "hidden_states": h})

        handles.append(blocks[li].register_forward_pre_hook(feed, with_kwargs=True))
    for li in layers:
        handles.append(
            blocks[li].register_forward_hook(
                lambda m, i, o, li=li: outs.__setitem__(li, o[0] if isinstance(o, tuple) else o)
            )
        )
    with torch.no_grad():
        logits = model(g.tokens()[None], use_cache=False).logits[0].float()
    for h in handles:
        h.remove()
    keep, a, b = tuple(s.get("golden.keep_fp32", [])), (g.n_chunks - 1) * g.chunk, g.seq
    out = {}
    for i in layers:
        o = outs[i].reshape(-1, outs[i].shape[-1])[a:b]
        out[f"layer_L{i:02d}"] = dict(_float_entry(store_dtype(o, keep, "out"), g.layer(g.n_chunks - 1, i)["out"]))
    if ends_at_last:
        rows = sampled_rows(g.chunk)
        top1 = g.model(g.n_chunks - 1)["top32_ids"][rows][:, 0].long()
        top5 = logits[a:b][rows].topk(5, dim=-1).indices
        out["top5"] = {
            "overlap": float((top5 == top1[:, None]).any(-1).float().mean()),
            "overlap_res": 1.0 - 1.0 / max(top1.numel(), 1),
            "region": None,
        }
    for e in out.values():
        e.update(source="hf_official", matched=not rung.get("prefix_from_golden"))
    return out


def margin(d: dict) -> dict:
    """F136: whether the reference's shipped precision moves anything: layers whose drift exceeds one rounding. None
    (and an identical official run) means no margin source: every limit would be one rounding of the target."""
    lay = {k: e for k, e in d.items() if k.startswith("layer_L")}
    return {
        "official_identical": int(all(e["rel"] == 0 for e in lay.values())),
        "drift_layers": len(lay),
        "drift_layers_above_res": sum(e["rel"] > e["rel_res"] for e in lay.values()),
    }


def compute(s, rung_name: str) -> dict:
    """The drift entries of one rung (and the contract's, on the contract rung)."""
    from models.demos.common.bringup.testing.contract import contract_rung
    from models.demos.common.bringup.testing.ladder import rung_layers

    if s.get("reference.margin_source") == "hf_official":
        return compute_hf(s, rung_name)
    rung = s.rung(rung_name)
    g = Golden.for_rung(s, rung_name)
    if g.manifest.get("numerics", "official") != "native":
        raise SystemExit(f"{g.dir}: numerics {g.manifest.get('numerics', 'official')!r}, want 'native' (F128)")
    layers = rung_layers(s, rung, g)
    first = g.n_chunks - 1 if rung.get("prefix_from_golden") else 0
    dtype = getattr(torch, str(s.get("reference.shipped_dtype")))
    run_layers = None if layers[-1] == s.num_layers - 1 else list(range(max(layers) + 1))
    ref = reference_at(s, "official", layers=run_layers, dtype=dtype)
    tail = None
    if contract_rung(s) == rung_name:
        tail = g.seq - int(s.get("tests.contract_tail_pad"))
    rep = replay(s, ref, g, layers, first, tail if first == 0 else None)
    out = entries(s, ref, g, layers, rep, first)
    if tail is not None:
        if first:  # the contract runs every chunk even when the ladder's rung starts after a golden prefix
            rep = replay(s, ref, g, layers, 0, tail)
        out.update(contract_entries(s, ref, g, layers, rep, tail))
    return out


def readers(s, rung_name: str) -> list[str]:
    """The rungs that read this rung's golden: itself and every rung with ``golden: <it>``."""
    return [r["name"] for r in s.data["ladder"] if r["name"] == rung_name or r.get("golden") == rung_name]


def pin(spec, ledger) -> list[str]:
    """F134: every drift file of the spec's goldens pinned in the frozen block of each device task gated on an
    end-to-end limit. Returns the pinned tasks."""
    from models.demos.common.bringup.core import freeze
    from models.demos.common.bringup.reference.golden import rung_dir

    dirs = {rung_dir(spec, r) for r in spec.data.get("ladder") or []}
    files = sorted(str(p) for d in dirs for p in Path(d).glob("drift-*.json"))
    if not files:
        return []
    pins = freeze.hash_paths(spec.repo, files)
    out = []
    for tid, t in ledger.tasks().items():
        conds = (t.get("gate") or {}).get("metrics") or {}
        if not (t.get("device") and any(E2E_METRIC.match(k) and str(c).endswith("limit") for k, c in conds.items())):
            continue
        frozen = dict(t.get("frozen") or {})
        frozen["files"] = {k: v for k, v in (frozen.get("files") or {}).items() if "/drift-" not in k} | pins
        ledger.update_task_def(tid, frozen=frozen)
        out.append(tid)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    ap.add_argument("--rung", required=True)
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    d = compute(spec, a.rung)
    p = drift_path(Golden.for_rung(spec, a.rung).dir, a.rung)
    p.write_text(json.dumps(d, indent=1, sort_keys=True))
    for k, v in d.items():
        metrics.record(f"drift_{a.rung}_{k}", v.get("rel", 1 - v.get("overlap", 1.0)))
    metrics.record(f"drift_metrics_{a.rung}", len(d))
    worst = max(d, key=lambda k: d[k].get("rel", 1 - d[k].get("overlap", 1.0)))
    print(
        f"drift of the official reference fed as the device on {a.rung}: {len(d)} metrics, worst {worst} "
        f"{d[worst].get('rel', 1 - d[worst].get('overlap', 1.0)):.6g} -> {p.name}"
    )


if __name__ == "__main__":
    main()
