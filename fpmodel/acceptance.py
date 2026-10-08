"""Score a time_picks.py run: per case, the model's pick and the rules' pick against legacy, timed in the same session.
Counts only (held-out data is never used to tune). usage: acceptance.py TIMED_CSV [OTHER_TIMED_CSV label]"""
import sys, numpy as np, pandas as pd

gm = lambda x: float(np.exp(np.mean(np.log(np.asarray(x, float)))))


def table(path):
    d = pd.read_csv(path, low_memory=False)
    t = d.pivot_table(index="case", columns="origin", values="device_ns", aggfunc="last")
    st = d.pivot_table(index="case", columns="origin", values="status", aggfunc="last")
    return t, st


def report(path, label):
    t, st = table(path)
    print(f"== {label}: {len(t)} cases")
    print("   status:", {o: st[o].value_counts().to_dict() for o in st})
    ok = t.dropna(subset=["legacy", "heuristic", "model"])
    ok = ok[(st.loc[ok.index] == "ok").all(axis=1)]
    m, r = ok.model / ok.legacy, ok.heuristic / ok.legacy
    mr = ok.model / ok.heuristic
    print(f"   n={len(ok)} with all three timed ok")
    print(f"   vs legacy (geomean):    model {gm(m):.3f}   rules {gm(r):.3f}")
    print(f"   >5% slower than legacy: model {(m > 1.05).sum():3d}   rules {(r > 1.05).sum():3d}")
    print(f"   >10% slower:            model {(m > 1.10).sum():3d}   rules {(r > 1.10).sum():3d}")
    print(f"   worst vs legacy:        model {m.max():.2f}   rules {r.max():.2f}")
    print(f"   best vs legacy:         model {m.min():.2f}   rules {r.min():.2f}")
    print(
        f"   model vs rules: geomean {gm(mr):.3f}; model >5% faster {(mr < 1 / 1.05).sum()}, >5% slower {(mr > 1.05).sum()}, same config {(ok.model == ok.heuristic).sum()}"
    )
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    print("   quantiles (p5 p25 p50 p75 p95):")
    for name, x in (("model/legacy", m), ("rules/legacy", r), ("model/rules", mr)):
        print(f"     {name:13s} " + "  ".join(f"{v:.2f}" for v in x.quantile(q)))
    bins = [0, 0.5, 0.8, 0.95, 1.05, 1.10, 1.25, 1.5, np.inf]
    labels = ["<0.50", "0.50-0.80", "0.80-0.95", "0.95-1.05", "1.05-1.10", "1.10-1.25", "1.25-1.50", ">1.50"]
    h = pd.DataFrame(
        {
            k: pd.cut(x, bins, labels=labels, right=False).value_counts().reindex(labels)
            for k, x in (("model/legacy", m), ("rules/legacy", r), ("model/rules", mr))
        }
    )
    print("   distribution (ratio, cases):")
    print("     " + h.to_string().replace("\n", "\n     "))
    return ok


if __name__ == "__main__":
    ok = report(sys.argv[1], sys.argv[1].split("/")[-1])
    if len(sys.argv) > 3:
        report(sys.argv[2], sys.argv[3])
