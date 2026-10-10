"""Print the _BLOCKINGS patch (dict literal) of every swept layer whose best blocking beats the table one by >= 3%."""
import glob, json, sys

new = {}
for p in sorted(glob.glob(sys.argv[1] + "/*.json")):
    r = json.load(open(p))
    tbl, best = r["table_us"], r["best_us"]
    key = (r["h_factor"], r["w_factor"], r["C_in"], r["C_out"], tuple(r["kernel"]), *r["table_key"])
    win = best is not None and tbl and r["best_blocking"] != r["table_blocking"] and best <= 0.97 * tbl
    print(f"{p}: key={key} table={r['table_blocking']} {tbl} us best={r['best_blocking']} {best} us "
          f"check={r['output_check']} win={win}", file=sys.stderr)
    if win:
        new[key] = tuple(r["best_blocking"])
print(repr(new) if new else "")
