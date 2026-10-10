"""Print the _BLOCKINGS patch (dict literal) of every swept layer whose best md5-identical blocking beats the table
one by >= 3%. Reads the per-layer sweep JSONs (run_sweep); a best blocking that is not bit-identical never wins."""
import glob, json, sys

new = {}
for p in sorted(glob.glob(sys.argv[1] + "/*.json")):
    r = json.load(open(p))
    tbl, best = r["table_us"], r["best_us"]
    key = (r["h_factor"], r["w_factor"], r["C_in"], r["C_out"], tuple(r["kernel"]), *r["table_key"])
    chk = r.get("output_check") or {}
    same_cin = r["best_blocking"] and r["best_blocking"][0] == r["table_blocking"][0]
    win = (best is not None and tbl and r["best_blocking"] != r["table_blocking"] and best <= 0.97 * tbl
           and chk.get("identical") is True and same_cin)
    print(f"{p}: key={key} table={r['table_blocking']} {tbl} us best={r['best_blocking']} {best} us "
          f"check={chk} win={win}", file=sys.stderr)
    if win:
        new[key] = tuple(r["best_blocking"])
print(repr(new) if new else "")
