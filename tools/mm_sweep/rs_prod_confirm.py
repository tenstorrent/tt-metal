#!/usr/bin/env python3
# Confirm the PRODUCTION path selects reduce-scatter on EVERY gate-selected corpus winner. For each of the 5
# gated shapes (at its deployed picker config), A/B mask 0 (production default) vs mask 256 (DIAG_FORCE_CHAIN =
# pure chain). If the gate fires, mask 0 == reduce-scatter and should match the diagnostic reduce-scatter delta.
import json, os, statistics, sys

sys.path.insert(0, os.path.dirname(__file__))
import regime_a_diag_suite as ds

PROD, CHAIN = 0, 256  # mask 0 = production default; 256 = DIAG_FORCE_CHAIN
N_RELAUNCH = 8
ITERS = 8
# (M,K,N, picker cfg (Ns,Pk,Sm,kb,nsb)) — the 5 shapes the production gate selects.
SHAPES = [
    (64, 2048, 1024, (2, 4, 1, 2, 2)),
    (128, 2048, 1024, (1, 4, 2, 2, 4)),
    (128, 2048, 2048, (3, 4, 1, 2, 3)),
    (256, 2048, 1024, (1, 4, 2, 2, 4)),
    (256, 2048, 2048, (1, 4, 3, 2, 4)),
]


def med(M, K, N, cfg, mask):
    ws = []
    for _ in range(N_RELAUNCH):
        r = ds.run_one(M, K, N, cfg, mask, iters=ITERS, timeout=200)
        if r.get("ok") and r.get("wall_us") is not None:
            ws.append(r["wall_us"])
    return statistics.median(ws) if ws else None, len(ws)


def main():
    rows = []
    for M, K, N, cfg in SHAPES:
        pw, pn = med(M, K, N, cfg, PROD)
        cw, cn = med(M, K, N, cfg, CHAIN)
        delta = (pw - cw) / cw * 100.0 if (pw and cw) else None
        rows.append((M, K, N, cfg, cw, pw, delta, cn, pn))
        print(
            f"[{M}x{K}x{N} cfg={cfg}] chain(256)={cw:.2f} prod(0)={pw:.2f} delta={delta:+.2f}% (n={pn}/{cn})",
            flush=True,
        )
        json.dump(
            [
                {
                    "M": r[0],
                    "K": r[1],
                    "N": r[2],
                    "cfg": list(r[3]),
                    "chain_us": r[4],
                    "prod_us": r[5],
                    "prod_vs_chain_pct": r[6],
                }
                for r in rows
            ],
            open(f"{ds.HERE}/rs_prod_confirm_results.json", "w"),
            indent=2,
        )
    print("\n===== PRODUCTION PATH (mask 0) vs FORCE_CHAIN on all 5 gate winners =====", flush=True)
    print(f"{'shape':16s} {'chain_us':>9s} {'prod_us':>9s} {'prod_vs_chain':>13s}", flush=True)
    for M, K, N, cfg, cw, pw, d, cn, pn in rows:
        print(f"{f'{M}x{K}x{N}':16s} {cw:>9.2f} {pw:>9.2f} {d:>+12.2f}%", flush=True)
    print("(prod_us should be the reduce-scatter time => negative delta on every winner)", flush=True)


if __name__ == "__main__":
    main()
