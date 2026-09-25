#!/usr/bin/env python3
"""Append one runs.csv row per timed point of a budget_sweep.py log (and fill.csv rows per fill chunk).

  budget_collect.py <run.log> <run.env> <runs.csv>

A run that did not finish still gets rows: completed points keep their numbers, the point it died in
and the ones after it get the run's STATUS and empty timings. Append-only.
"""
import csv, json, os, sys

COLS = (
    "run_id,timestamp,git_sha,exp,layer_set,layers,n_dense,n_sparse,mesh,sp,tp,ep,W,B,segments_json,"
    "replicated_tokens,traced,synth_history,iters,wall_ms_median,wall_ms_min,wall_ms_max,device_ms,"
    "host_gap_pct,peak_dram_gb,status,notes"
).split(",")
FILL_COLS = ["run_id", "layer_set", "W", "capacity", "h", "n", "wall_ms"]


def append(path, cols, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerows(rows)


def main(log, envf, out):
    env = dict(l.rstrip("\n").split("=", 1) for l in open(envf) if "=" in l)
    recs, status = [], "ERROR"
    for line in open(log, errors="replace"):
        if line.startswith("RESULT "):
            recs.append(json.loads(line[7:]))
        elif line.startswith("STATUS="):
            status = line.split()[0].split("=", 1)[1]
    cfg = next((r for r in recs if r["kind"] == "config"), None)
    if cfg is None:
        layers = [int(x) for x in env["BUDGET_LAYER_IDS"].split(",")]
        cfg = {
            "layers": layers,
            "W": int(env.get("BUDGET_W", 2048)),
            "points": [],
            "capacity": "",
            "stages": 2,
            "fill": env.get("BUDGET_FILL", "real"),
            "iters": "",
        }
    layers = cfg["layers"]
    rows_ = 8 // int(cfg["stages"])
    base = dict(
        run_id=env["run_id"],
        timestamp=env["date"],
        git_sha=env["git_sha"][:12],
        exp=env.get("EXP", ""),
        layer_set=env.get("LAYER_SET", ""),
        layers=" ".join(map(str, layers)),
        n_dense=sum(l < 3 for l in layers),
        n_sparse=sum(l >= 3 for l in layers),
        mesh=f"({rows_},4)",
        sp=rows_,
        tp=4,
        ep=rows_ * 4,
        W=cfg["W"],
        B=1,
        replicated_tokens=0,
        traced=0,
        synth_history=int(cfg["fill"] != "real"),
        iters=cfg["iters"],
    )
    note = f"capacity={cfg['capacity']} fill={cfg['fill']} {env.get('NOTES', '')}".strip()
    done = {(r["h"], r["n"]): r for r in recs if r["kind"] == "point"}
    rows = []
    for h, n in [tuple(p) for p in cfg["points"]] or [(None, None)]:
        r = done.get((h, n))
        row = dict(
            base,
            segments_json=json.dumps([{"slot": 0, "h": h, "n": n}]),
            notes=note,
            status="OK" if r else (status if status != "OK" else "ERROR"),
        )
        if r:
            row.update(wall_ms_median=r["wall_ms_median"], wall_ms_min=r["wall_ms_min"], wall_ms_max=r["wall_ms_max"])
        rows.append(row)
    append(out, COLS, rows)
    fills = [
        dict(
            run_id=env["run_id"],
            layer_set=env.get("LAYER_SET", ""),
            W=cfg["W"],
            capacity=cfg["capacity"],
            h=r["h"],
            n=r["n"],
            wall_ms=r["wall_ms"],
        )
        for r in recs
        if r["kind"] == "fill"
    ]
    if fills:
        append(os.path.join(os.path.dirname(out), "fill.csv"), FILL_COLS, fills)
    print(f"[collect] {env['run_id']}: status={status}, {len(done)}/{len(rows)} points, {len(fills)} fill rows")


if __name__ == "__main__":
    main(*sys.argv[1:4])
