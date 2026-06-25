# (S,Pk) heuristic for minimal_matmul — derived from the joint sweep oracle, back-tested.
# Inputs: Mt,Nt,Kt (tile counts). Output: (S=N-slices, Pk=K-par bands), S*Pk<=grid.y(=8).
def pick_S_Pk(Mt, Nt, Kt, GY=8):
    small, big = min(Mt, Nt), max(Mt, Nt)
    out = Mt * Nt
    skew = big / small
    if Kt <= 4:  # shallow K: K-par useless; slice only if rows underfilled
        S, Pk = (8 if (skew >= 6 and small < 8) else 1), 1
    elif Kt >= 64:  # deep K: K-par is the primary lever (fill cores + shorten K-loop)
        if out < 16:
            Pk = 8
        elif out <= 64:
            Pk = 8 if skew >= 2.5 else (1 if small >= 8 else 4)
        elif out < 256:
            Pk = 4 if skew >= 6 else 2
        else:
            Pk = 2 if out < 512 else 1
        S = 1
        if skew >= 12 and small < 8:
            S, Pk = 2, min(Pk, 4)  # very skewed + starved: trade some Pk for a slice
    else:  # medium K (8..32)
        if out >= 256:
            S, Pk = (8 if (skew >= 24 and small < 8) else 2 if skew >= 2.5 else 1), 1
        elif out >= 64:
            S, Pk = (8, 1) if skew >= 24 else (4, 2) if skew >= 12 else (2, 1) if skew >= 6 else (1, 1)
        elif out >= 16:
            S, Pk = (4, 2) if skew >= 12 else (2, 4) if skew >= 6 else (2, 1) if skew >= 2.5 else (1, 1)
        else:
            S, Pk = (4, 2) if skew >= 6 else (1, 4)  # out<16: very starved
    while Pk > 1 and Kt // Pk < 2:
        Pk //= 2  # keep >=2 K-tiles per band
    S = max(1, min(S, GY // Pk))  # enforce budget S*Pk<=GY
    return (S, Pk)


if __name__ == "__main__":
    import json, collections, math, statistics

    SRC = [
        "mm_jointsweep/results_shard0.jsonl",
        "mm_fluxltx_sweep/results_shard0.jsonl",
        "mm_model_shapes/transferred_results.jsonl",
        "mm_model_shapes/results.jsonl",
    ]
    SH = collections.defaultdict(dict)
    AUTO = {}
    FEAT = {}
    for f in SRC:
        try:
            fh = open("/localdev/cglagovich/" + f)
        except:
            continue
        for l in fh:
            try:
                r = json.loads(l)
            except:
                continue
            if not r.get("ok") or r.get("pcc") is None or r["pcc"] < 0.99 or r.get("us") is None:
                continue
            k = (r["M"], r["K"], r["N"])
            FEAT[k] = (r["Mt"], r["Nt"], r["Kt"])
            if not str(r["S"]).isdigit():
                AUTO[k] = min(AUTO.get(k, 1e18), r["us"])
                continue
            sp = (int(r["S"]), int(r["Pk"]))
            SH[k][sp] = min(SH[k].get(sp, 1e18), r["us"])
    ORC = {k: min(d.items(), key=lambda kv: kv[1]) for k, d in SH.items() if d}

    def bt(keys, name):
        rg = []
        ar = []
        ex = 0
        w5 = 0
        for k in keys:
            Mt, Nt, Kt = FEAT[k]
            d = SH[k]
            osp, ob = ORC[k]
            sp = pick_S_Pk(Mt, Nt, Kt)
            if sp not in d:
                c = [x for x in d if x[0] * x[1] == sp[0] * sp[1]]
                sp = c[0] if c else ((1, 1) if (1, 1) in d else min(d, key=lambda x: d[x]))
            rg.append(d[sp] / ob)
            ex += sp == osp
            w5 += d[sp] / ob <= 1.05
            if k in AUTO:
                ar.append(AUTO[k] / ob)
        g = math.exp(statistics.mean(math.log(x) for x in rg))
        ag = math.exp(statistics.mean(math.log(x) for x in ar))
        print(
            f"{name:28} n={len(keys):4}  heuristic={g:.4f}  auto={ag:.4f}  exact={100*ex/len(keys):.0f}% w5%={100*w5/len(keys):.0f}%"
        )

    bt(list(ORC), "ALL")
    bt([k for k in ORC if FEAT[k][0] * FEAT[k][1] >= 256], "large (out>=256)")
