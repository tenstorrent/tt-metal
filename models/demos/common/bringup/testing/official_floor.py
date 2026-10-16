# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The reference arithmetic of the noise floor (F80): the official code at its own dtypes, tapped per step.

    python -m models.demos.common.bringup.testing.official_floor --spec S [--check]

The model hook ``official_taps(spec, layers, tokens, chunk, chunk_size)`` runs the official implementation (its own
modeling code, its shipped dtypes; e.g. through hooks.hf_model) over the first ``chunk + 1`` chunks of ``tokens`` and
returns, for the chunk ``chunk``, ``{layer: {step: {"inputs": [the step's inputs, in the framework step's input order,
as the official code held them], "output": its output}}}``, plus per layer ``"_state": {"start": ..., "end": ...}``:
the layer's state as the official run had it at the chunk's start and end, in the reference's own state format (what
``ref.state_tensors`` returns, the names of ``spec.state_names(layer)``). A stateful step (attention, a compressor)
reads its prefix from that state, and a layer that reads another layer's state reads the source's end state: the
golden's state would mix the official run's drift into the comparison (DeepSeek-V4.1 R.5: attention 2-5x its floor at
layers 20-24 with the golden's caches). The reference role writes the hook together with official_taps.yaml (the
official dtypes of every step's inputs and output, testing/noise_floor.py).

For every tapped step this runs the fp32 CPU step on the official inputs and records, next to the component goldens
(official_floor.json, read by noise_floor.arith):
- vs_step: the official output vs the fp32 step on the same inputs (rel, worst row, row-norm ratios, median bias);
- vs_golden: the official output vs the golden (the same, plus pcc_vs_golden).
It also checks the tap map: the fp32 step on the official inputs must give the official output within calib x the
step's input-nudge floor (a mis-mapped input or output is far outside it), unless official_taps.yaml explains why the
official code rounds more inside the step (`explained`, with the evidence; the overseer reviews each). Gate metrics: official_taps_steps (>= 1),
official_taps_errors (== 0), official_taps_unmapped (== 0: every step of every golden layer is tapped).
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from models.demos.common.bringup.core import metrics


def _stats(got: torch.Tensor, want: torch.Tensor) -> dict:
    from models.demos.common.bringup.testing.component import step_errors

    e = step_errors(got.float(), want.float())
    e["pcc"] = metrics.pcc(got.float(), want.float())
    return e


def _ctx(ref, layer: int, g, c: int, s, taps: dict):
    """The reference context of chunk c for ``layer`` with the official run's state (taps[layer]["_state"]: its own
    at the chunk start, its sources' at the chunk end); the golden's where the hook gives none. (ctx, whose state)."""
    from models.demos.common.bringup.testing.harness import reference_ctx, sources

    own = (taps.get(layer) or {}).get("_state") or {}
    srcs = list(sources(layer, s))
    if not own and not any((taps.get(j) or {}).get("_state") for j in srcs):
        return reference_ctx(ref, layer, g, c, s), "golden"
    start, end = c * g.chunk, (c + 1) * g.chunk
    state = ref.new_state(g.seq)
    if start:
        ref.load_state(state, layer, own["start"] if own.get("start") is not None else g.state(layer, at=start), start)
    for j in srcs:
        js = (taps.get(j) or {}).get("_state") or {}
        ref.load_state(state, j, js["end"] if js.get("end") is not None else g.state(j, at=end), end)
    ctx = ref.chunk_context(layer, start, g.chunk, state)
    ctx.extra.setdefault("tokens", g.tokens()[start:end])
    return ctx, "official"


def generate(s, write: bool = True) -> dict:
    from models.demos.common.bringup.testing import component_checks as CC
    from models.demos.common.bringup.testing import noise_floor as NF
    from models.demos.common.bringup.testing.harness import component_golden

    hooks = s.hooks()
    if not hasattr(hooks, "official_taps"):
        raise SystemExit("the model has no official_taps hook (bringup/hooks.py); the reference role writes it")
    g, c = component_golden(s)
    layers = list(g.layers)
    taps = hooks.official_taps(s, layers, g.tokens(), c, g.chunk)
    ref = hooks.reference(s, layers=layers, dtype=torch.float32)
    k = CC.limits(s)["component_calib"]
    out, errors, unmapped, n = {}, [], [], 0
    for L in layers:
        bt = s.block_type_of(L)
        for st in ref.block_graph(L):
            t = (taps.get(L) or {}).get(st.name)
            whose = _ctx(ref, L, g, c, s, taps)[1]
            if st.stateful and whose == "golden" and t is not None:
                print(f"note L{L:02d}.{st.name}: stateful, but the hook gives no official state; the golden's is used")
            if t is None:
                unmapped.append(f"L{L:02d}.{st.name}")
                continue
            xs = [x.float() if x.is_floating_point() else x for x in t["inputs"]]
            official = t["output"]
            cpu = ref.component(L, st.name)

            def run(*inp, L=L, cpu=cpu):
                ctx = _ctx(ref, L, g, c, s, taps)[0]  # fresh per run: a stateful step writes its state
                return cpu(ctx, *[x.clone() if isinstance(x, torch.Tensor) else x for x in inp])

            try:
                y32 = run(*xs)
            except Exception as e:  # noqa: BLE001 - a wrong input list is a mapping error
                errors.append(f"L{L:02d}.{st.name}: the fp32 step on the tapped inputs raised {type(e).__name__}: {e}")
                continue
            if not (official.is_floating_point() and y32.is_floating_point()):
                continue  # an index output: the floor is the nudge's alone
            if official.numel() != y32.numel():
                errors.append(
                    f"L{L:02d}.{st.name}: tapped output {tuple(official.shape)}, the step's {tuple(y32.shape)}"
                )
                continue
            official = official.float().reshape(y32.shape)
            rec = {"vs_step": _stats(official, y32), "state": whose}
            want = g.layer(c, L).get(st.output)
            if want is not None and want.is_floating_point() and want.numel() == official.numel():
                gs = _stats(official, want.float().reshape(official.shape))
                gs["pcc_vs_golden"] = gs.pop("pcc")
                rec["vs_golden"] = gs
            dtypes = NF.input_dtypes(s, bt, st.name, len(xs))
            nud = max(CC.rel(o, y32) for o in NF.nudged(run, xs, dtypes))
            res = CC.rel(NF.round_to(y32, NF.output_dtype(s, bt, st.name)), y32)
            lim = k * max(nud, res)
            rec["check"] = {"rel": rec["vs_step"]["rel"], "limit": lim, "nudge": nud}
            why = NF.explanation(s, bt, st.name)
            if not rec["vs_step"]["rel"] <= lim and why:
                rec["check"]["explained"] = why  # the official arithmetic is the floor: the overseer reviews `why`
                print(
                    f"note L{L:02d}.{st.name}: official arithmetic rel {rec['vs_step']['rel']:.3g} > {lim:.3g} "
                    f"({k} x the input nudge), explained: {why[:200]}"
                )
            elif not rec["vs_step"]["rel"] <= lim:
                errors.append(
                    f"L{L:02d}.{st.name}: official output vs the fp32 step on the official inputs rel "
                    f"{rec['vs_step']['rel']:.3g} > {lim:.3g} ({k} x the input nudge {nud:.3g}): a mis-mapped input "
                    "or output, or the official code rounds inside the step more than its inputs' nudge shows; if "
                    "so, say why with the evidence in official_taps.yaml (`explained`)"
                )
            out[f"L{L:02d}.{st.name}"] = {str(c): rec}
            n += 1
    doc = {"chunk": c, "steps": out, "unmapped": unmapped, "errors": errors}
    if write:
        (g.dir / "official_floor.json").write_text(json.dumps(doc, indent=1) + "\n")
    metrics.record("official_taps_steps", n)
    metrics.record("official_taps_errors", len(errors))
    metrics.record("official_taps_unmapped", len(unmapped))
    for e in errors:
        print(f"FAIL {e}")
    if unmapped:
        print(f"FAIL not tapped: {', '.join(unmapped)}")
    print(f"{'ok  ' if not errors and not unmapped else 'FAIL'} official taps: {n} steps, chunk {c}")
    return doc


def main(argv=None) -> int:
    from models.demos.common.bringup.core.spec import Spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.environ.get("BRINGUP_SPEC"), help="default: BRINGUP_SPEC (set by the gate)")
    ap.add_argument("--check", action="store_true", help="check the tap map only; do not write official_floor.json")
    a = ap.parse_args(argv)
    doc = generate(Spec.load(a.spec), write=not a.check)
    return 0 if not doc["errors"] and not doc["unmapped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
