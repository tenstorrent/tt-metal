# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A plan written for the deployment target, checked on every box the model may run on (F64).

The plan is made for ``deploy.box`` (e.g. a Galaxy 8x4 at 1M tokens). Smaller boxes stand in for it during the
bring-up: each is the same layout on smaller mesh axes. With a ``deploy`` section in the spec, ``plan.yaml`` adds:

    roles:                              # what each mesh axis splits: role -> mesh axes (names are free)
      sequence: [0]                     # required: the axes the tokens are split over ([] = not split)
      heads: [1]
      experts: [0, 1]
    placements:                         # as before; also {placement: split, over: <role>}
    state:                              # per box: {heads: <total>, split_heads: <role>, split_seq: <role>}
      - {layers: "3,7", what: latent, heads: 1, head_dim: 512, tensors: 1, dtype: bf16, split_seq: sequence}
    boxes:                              # per-box differences from the placements (optional)
      loudbox_2x4: {dtypes: {"model.layers.*.experts.*": bfp4}, activations_gb_per_chip: 4.0}
    conditions:                         # per-chip counts that pick a code path, or must hold on every box
      - {what: "experts per chip (flat_routed_expert)", total: 288, over: experts, max: 64, else: "dispatch/combine"}
      - {what: "heads per chip (sparse_sdpa)", total: 64, over: heads, min: 32, else: "all-to-all around it"}
      - {what: "pooled keys per chip per chunk", total: "chunk / 4", over: sequence, multiple_of: 32}

``total`` is a number or an expression of ``seq`` and ``chunk`` (the box's context). A condition must divide evenly
over its role's chips and meet its bounds; when it does not, the box runs ``else`` instead of the plan's path, and a
condition without ``else`` fails the box (an error on the deployment box and on the run's box).

Per box the check reports:
- scale: per role, the box's chips over the deployment's (a partial run on a smaller box scales the sequence, heads
  and experts by these, so each chip sees the deployment's per-chip shapes);
- context: the deployment's seq and chunk scaled by the sequence factor (exact or not);
- full model at real dimensions: weights per chip and the longest context that fits (full mode needs it to fit);
- the run's layers scaled (per chip as on the deployment box) and at real dimensions on the box;
- every condition: the per-chip count, the path, and whether it is the deployment box's path.
"""

from __future__ import annotations

import ast
import fnmatch
import operator

from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.plan.memory import check_plan, layer_of, ways

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}


def evaluate(expr, names: dict) -> float:
    """A number, or an arithmetic expression (+ - * / //) of the given names."""
    if isinstance(expr, (int, float)):
        return expr

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name) and n.id in names:
            return names[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(f"unsupported expression {expr!r} (numbers, {sorted(names)}, + - * / //)")

    return ev(ast.parse(str(expr), mode="eval"))


def _mesh(spec: Spec, box: str) -> list[int]:
    return list(spec.boxes[box]["mesh"])


def factors(spec: Spec, plan: dict, box: str) -> dict[str, float]:
    """Per role: the box's chips over the deployment box's chips for that role."""
    roles, dm, bm = plan.get("roles") or {}, _mesh(spec, spec.deploy_box), _mesh(spec, box)
    return {r: ways(r, bm, roles) / ways(r, dm, roles) for r in roles}


def context(spec: Spec, plan: dict, box: str) -> dict:
    """The deployment's context scaled by the box's sequence factor: {seq, chunk, exact}."""
    f = factors(spec, plan, box).get("sequence", 1.0)
    seq, chunk = spec.deploy_seq * f, spec.deploy_chunk * f
    exact = seq == int(seq) and chunk == int(chunk) and int(chunk) % 32 == 0 and int(seq) % max(int(chunk), 1) == 0
    return {"seq": int(seq), "chunk": int(chunk), "exact": bool(exact)}


def conditions(spec: Spec, plan: dict, box: str, seq: int, chunk: int) -> list[dict]:
    roles, mesh, out = plan.get("roles") or {}, _mesh(spec, box), []
    for c in plan.get("conditions") or []:
        r = {"what": c.get("what", "?"), "over": c.get("over")}
        try:
            per = evaluate(c["total"], {"seq": seq, "chunk": chunk}) / ways(c.get("over"), mesh, roles)
        except (KeyError, ValueError, ZeroDivisionError) as e:
            out.append({**r, "per_chip": None, "ok": False, "path": f"bad condition: {e}", "hard": True})
            continue
        ok = per == int(per)
        if ok and "min" in c:
            ok = per >= c["min"]
        if ok and "max" in c:
            ok = per <= c["max"]
        if ok and "multiple_of" in c:
            ok = int(per) % int(c["multiple_of"]) == 0
        path = "plan" if ok else (c.get("else") or "FAILS")
        out.append({**r, "per_chip": per, "ok": ok, "path": path, "hard": not ok and not c.get("else")})
    return out


def longest_context(spec, plan, tensors, cfg, box: str, cap: int) -> int:
    """The longest context (a multiple of 1024, at most ``cap``) at which the full model fits the box, else 0."""

    def fits(seq):
        every = list(range(spec.num_layers))
        r = check_plan(
            spec, plan, tensors, cfg, mesh=_mesh(spec, box), cap_gb=_cap(spec, box), seq=seq, layers=every, box=box
        )
        return "total_gb" in r and r["total_gb"] <= r["budget_gb"]

    lo, hi = 0, max(cap // 1024, 1)
    if fits(hi * 1024):
        return hi * 1024
    while lo < hi - 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(mid * 1024) else (lo, mid)
    return lo * 1024


def _cap(spec: Spec, box: str) -> float:
    return float(spec.boxes[box]["chip_dram_gb"])


def check_boxes(spec: Spec, plan: dict, tensors: dict, cfg: dict | None = None) -> dict:
    """{rows: one per box, errors}. Needs a spec with ``deploy`` (spec.has_deploy)."""
    errs, roles = [], plan.get("roles")
    if not isinstance(roles, dict) or "sequence" not in roles:
        return {"rows": [], "errors": ["plan.yaml needs roles (role -> mesh axes), with 'sequence' ([] if not split)"]}
    for r, axes in roles.items():
        if not isinstance(axes, list) or any(a not in (0, 1) for a in axes):
            errs.append(f"roles.{r} must list mesh axes 0 / 1")
    for name in set(plan.get("boxes") or {}) - set(spec.boxes):
        errs.append(f"plan boxes.{name} is not a box of the spec ({sorted(spec.boxes)})")
    if errs:
        return {"rows": [], "errors": errs}
    every = list(range(spec.num_layers))
    for p in plan.get("placements") or []:
        if p.get("placement") == "skip":
            hit = sorted(
                {
                    layer_of(n)
                    for n in tensors
                    if (layer_of(n) or 0) < spec.num_layers
                    and layer_of(n) is not None
                    and fnmatch.fnmatchcase(n, p["pattern"])
                }
            )
            if hit:
                errs.append(
                    f"placement {p['pattern']!r} skips decoder layers {hit[:5]}: the deployment check needs every "
                    "layer placed (the run's layer subset is applied by the check, not by skip)"
                )
    dep = spec.deploy_box
    dep_conds = {c["what"]: c for c in conditions(spec, plan, dep, spec.deploy_seq, spec.deploy_chunk)}
    rows = []
    for box in spec.boxes:
        mesh, cap = _mesh(spec, box), _cap(spec, box)
        ctx = (
            {"seq": spec.deploy_seq, "chunk": spec.deploy_chunk, "exact": True}
            if box == dep
            else context(spec, plan, box)
        )
        kw = {"mesh": mesh, "cap_gb": cap, "box": box}
        full = check_plan(spec, plan, tensors, cfg, seq=spec.deploy_seq, layers=every, **kw)
        weights = check_plan(spec, plan, tensors, cfg, seq=0, layers=every, **kw)
        longest = longest_context(spec, plan, tensors, cfg, box, spec.deploy_seq)
        # scaled partial: each chip as on the deployment box (deploy mesh and context), this box's dtypes and DRAM
        scaled = check_plan(
            spec,
            plan,
            tensors,
            cfg,
            mesh=_mesh(spec, dep),
            cap_gb=cap,
            seq=spec.deploy_seq,
            layers=spec.layers(),
            box=box,
        )
        run_seq = int(spec.get("target.seq")) if box == spec.run_box else ctx["seq"]
        real = check_plan(spec, plan, tensors, cfg, seq=run_seq, layers=spec.layers(), **kw)
        chunk = int(spec.get("target.chunk")) if box == spec.run_box and box != dep else ctx["chunk"]
        conds = conditions(spec, plan, box, run_seq if box == spec.run_box else ctx["seq"], chunk)
        for c in conds:
            d = dep_conds.get(c["what"])
            c["same_as_deploy"] = bool(d) and d["path"] == c["path"]
        row = {
            "box": box,
            "mesh": mesh,
            "chips": mesh[0] * mesh[1],
            "deploy": box == dep,
            "run": box == spec.run_box,
            "scale": {r: round(f, 6) for r, f in factors(spec, plan, box).items()},
            "context": ctx,
            "dtypes": (plan.get("boxes") or {}).get(box, {}).get("dtypes") or {},
            "full": {
                "weights_gb": round(weights.get("total_gb", -1), 6),
                "gb_at_deploy_seq": round(full.get("total_gb", -1), 6),
                "fits_deploy_seq": bool(full.get("total_gb", 1e9) <= full.get("budget_gb", 0)),
                "longest_seq": longest,
            },
            "partial_scaled_gb": round(scaled.get("total_gb", -1), 6),
            "partial_real": {"seq": run_seq, "gb": round(real.get("total_gb", -1), 6)},
            "budget_gb": round(full.get("budget_gb", -1), 6),
            "conditions": conds,
        }
        rows.append(row)
        if box == dep:
            errs += [f"deployment box {box}: {e}" for e in full["errors"]]
        for c in conds:
            if c["hard"] and (box == dep or box == spec.run_box):
                errs.append(f"box {box}: condition {c['what']!r} fails (per chip {c['per_chip']}) and has no 'else'")
    return {"rows": rows, "errors": errs}


def table(rows: list[dict]) -> str:
    """The per-box table, as plan.md and the gate log show it."""
    out = [
        "| Box | Mesh | Scale | Context (scaled) | Full model: weights / longest context | Run's layers: scaled / real |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        tag = " (deploy)" if r["deploy"] else " (run)" if r["run"] else ""
        sc = ", ".join(f"{k} {v:g}" for k, v in r["scale"].items())
        ctx = f"{r['context']['seq']:,} / {r['context']['chunk']:,}" + ("" if r["context"]["exact"] else " (inexact)")
        f = r["full"]
        out.append(
            f"| {r['box']}{tag} | {r['mesh'][0]}x{r['mesh'][1]} | {sc} | {ctx} | {f['weights_gb']:.1f} GB / "
            f"{f['longest_seq']:,} | {r['partial_scaled_gb']:.1f} / {r['partial_real']['gb']:.1f} GB "
            f"(budget {r['budget_gb']:.1f}) |"
        )
    conds = [c["what"] for c in (rows[0]["conditions"] if rows else [])]
    if conds:
        out += ["", "| Condition | " + " | ".join(r["box"] for r in rows) + " |", "|---|" + "---|" * len(rows)]
        for i, w in enumerate(conds):
            cells = []
            for r in rows:
                c = r["conditions"][i]
                per = "?" if c["per_chip"] is None else f"{c['per_chip']:g}"
                cells.append(f"{per}: {c['path']}" + ("" if c["same_as_deploy"] else " **(differs)**"))
            out.append(f"| {w} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def scaling_applies(spec: Spec) -> bool:
    """A partial run on a box with fewer chips than the deployment box also runs scaled (plan/scale.py)."""
    if not spec.has_deploy or spec.mode != "partial":
        return False
    run, dep = spec.boxes[spec.run_box]["mesh"], spec.boxes[spec.deploy_box]["mesh"]
    return run[0] * run[1] < dep[0] * dep[1]
