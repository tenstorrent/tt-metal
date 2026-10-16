# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The reference arithmetic of the noise floor (F80): the official code at its own dtypes, tapped per step.

    python -m models.demos.common.bringup.testing.official_floor --spec S [--check]

The model hook ``official_taps(spec, layers, tokens, chunk, chunk_size)`` runs the official implementation (its own
modeling code, its shipped dtypes; e.g. through hooks.hf_model) over the first ``chunk + 1`` chunks of ``tokens`` and
returns, for the chunk ``chunk``, ``{layer: {step: {"inputs": [the step's inputs, in the framework step's input order,
as the official code held them], "output": its output}}}``. The reference role writes it together with
official_taps.yaml (the official dtypes of every step's inputs and output, testing/noise_floor.py).

For every tapped step this runs the fp32 CPU step on the official inputs and records, next to the component goldens
(official_floor.json, read by noise_floor.arith):
- vs_step: the official output vs the fp32 step on the same inputs (rel, worst row, row-norm ratios, median bias);
- vs_golden: the official output vs the golden (the same, plus pcc_vs_golden).
It also checks the tap map: the fp32 step on the official inputs must give the official output within calib x the
step's input-nudge floor (a mis-mapped input or output is far outside it). Gate metrics: official_taps_steps (>= 1),
official_taps_errors (== 0), official_taps_unmapped (== 0: every step of every golden layer is tapped).
"""

from __future__ import annotations

import argparse
import json

import torch

from models.demos.common.bringup.core import metrics


def _stats(got: torch.Tensor, want: torch.Tensor) -> dict:
    from models.demos.common.bringup.testing.component import step_errors

    e = step_errors(got.float(), want.float())
    e["pcc"] = metrics.pcc(got.float(), want.float())
    return e


def generate(s, write: bool = True) -> dict:
    from models.demos.common.bringup.testing import component_checks as CC
    from models.demos.common.bringup.testing import noise_floor as NF
    from models.demos.common.bringup.testing.harness import component_golden, reference_ctx

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
            if t is None:
                unmapped.append(f"L{L:02d}.{st.name}")
                continue
            xs = [x.float() if x.is_floating_point() else x for x in t["inputs"]]
            official = t["output"]
            cpu = ref.component(L, st.name)

            def run(*inp, L=L, cpu=cpu):
                return cpu(
                    reference_ctx(ref, L, g, c, s), *[x.clone() if isinstance(x, torch.Tensor) else x for x in inp]
                )

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
            rec = {"vs_step": _stats(official, y32)}
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
            if not rec["vs_step"]["rel"] <= lim:
                errors.append(
                    f"L{L:02d}.{st.name}: official output vs the fp32 step on the official inputs rel "
                    f"{rec['vs_step']['rel']:.3g} > {lim:.3g} ({k} x the input nudge {nud:.3g}): a mis-mapped input "
                    "or output, or the official code is not what the step computes"
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
    ap.add_argument("--spec")
    ap.add_argument("--check", action="store_true", help="check the tap map only; do not write official_floor.json")
    a = ap.parse_args(argv)
    doc = generate(Spec.load(a.spec), write=not a.check)
    return 0 if not doc["errors"] and not doc["unmapped"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
