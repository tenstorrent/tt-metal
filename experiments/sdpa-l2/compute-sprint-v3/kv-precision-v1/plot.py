"""Plot measured E-family KV ablation alongside frozen D/C/B/A evidence."""
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
ORDER = ("D", "C", "B", "A", "E_bf16", "E_bfp8", "E_bfp4")
COLORS = dict(D="#3658a2", C="#8155a6", B="#258657", A="#64748b",
              E_bf16="#006b66", E_bfp8="#00a9ca", E_bfp4="#d87d24")


def read(path):
    return json.loads(path.read_text())


def main():
    old = read(V3 / "pareto/matched-v1.json")
    new = read(HERE / "accuracy-v1.json")
    perf = read(HERE / "perf-v1.json")
    for data in (old, new, perf):
        assert data["complete"] and data["selected_sources_immutable"]
    assert len(new["records"]) == 63 and len(perf["records"]) == 3
    rows = [r for r in old["records"] if r["variant"] in "DCBA"] + new["records"]
    assert len(rows) == 147
    for r in rows:
        assert r["raw_trace_equal"] and r["inputs_immutable"]
    for r in new["records"]:
        assert r["preprocessing_exact"] and r["prepared_immutable"]
        v = {"E_bfp8": "E", "E_bfp4": "G"}.get(r["variant"], "E")
        prior = next(x for x in old["records"] if x["variant"] == v and
                     (x["suite"], x["k_length"], x["distribution"]) ==
                     (r["suite"], r["k_length"], r["distribution"]))
        assert prior["input_sha256"] == r["input_sha256"]
        if r["variant"] != "E_bf16":
            assert r["legacy_equal"] and prior["output_sha256"] == r["output_sha256"]
    sources = [V3 / "fp32/integrity-early-sustained-v2.json",
               V3 / "review/B-valid-perf-final.json", V3 / "unchanged-A-resident-v2.json",
               V3 / "pareto/matched-v1.json", HERE / "accuracy-v1.json", HERE / "perf-v1.json"]
    speed = {}
    for v in "DC":
        timings = [r for r in read(sources[0]) if r["variant"] == v and r["algorithm"] == "l1_early"]
        speed[v] = timings[0]["useful_flops"] / (statistics.median(r["median_ms"] for r in timings)*1e9)
    speed["B"] = read(sources[1])["results"]["group2_valid"]["tflops_per_core"]
    speed["A"] = read(sources[2])["tflops_per_core"]
    speed.update({r["variant"]: r["tflops_per_core"] for r in perf["records"]})
    summary = dict(throughput_tflops_per_core=speed, panels={}, input_sha256={
        str(p.relative_to(V3)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    for suite in ("core", "stress"):
        summary["panels"][suite] = {}
        for v in ORDER:
            selected = [r for r in rows if r["variant"] == v and r["suite"] == suite]
            assert len(selected) == (18 if suite == "core" else 3)
            ys = np.array([r["metrics"]["l2_pct"] for r in selected])
            assert np.isfinite(ys).all() and (ys > 0).all()
            summary["panels"][suite][v] = dict(n=len(ys), med=float(np.median(ys)),
                q1=float(np.quantile(ys,.25)), q3=float(np.quantile(ys,.75)),
                whislo=float(ys.min()), whishi=float(ys.max()), fliers=[])

    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11,
                         "axes.spines.top":False, "axes.spines.right":False})
    fig, axes = plt.subplots(1,2,figsize=(17,11),gridspec_kw={"width_ratios":[1.12,1]})
    fig.patch.set_facecolor("#f8fafc")
    fig.subplots_adjust(left=.07,right=.965,bottom=.405,top=.83,wspace=.22)
    fig.text(.07,.951,"SDPA frontier · KV precision ablation",fontsize=26,fontweight="bold",color="#17243a")
    fig.text(.07,.910,"E_bfp8 = previous E     •     E_bfp4 = previous G     •     E_bf16 = new RNE5 BF16 KV",fontsize=14,color="#475569")
    fig.text(.07,.876,"Same LoFi compute, BF16 DST, compensated state and exp; Q RNE7. Only KV representation changes within E.",fontsize=12,color="#475569")
    core_markers = {4096:"o",32768:"s",262144:"^"}
    stress_markers = {"common_q":"^","common_k":"s","common_v":"o"}
    offsets = {"core":dict(D=(8,-25),C=(8,-25),B=(-25,23),A=(-27,49),
                           E_bf16=(17,-42),E_bfp8=(-115,3),E_bfp4=(-80,32)),
               "stress":dict(D=(8,12),C=(8,-22),B=(-25,25),A=(-25,38),
                             E_bf16=(18,-40),E_bfp8=(-103,-4),E_bfp4=(-85,-35))}
    for ax,suite in zip(axes,("core","stress")):
        ax.set_facecolor("white")
        ax.set_yscale("log")
        ax.set_xlim(.76,2.36)
        ax.set_xticks([.9,1.2,1.5,1.8,2.1])
        ax.set_ylim((.07,130) if suite == "core" else (.002,160))
        ax.grid(axis="y",which="major",color="#e2e8f0",lw=.9)
        ax.grid(axis="x",color="#f1f5f9",lw=.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("Resident useful TFLOP/s per core  →  faster",labelpad=10)
        ax.set_ylabel("Relative L2 error (%) · log scale  ←  better",labelpad=10)
        ax.axhline(.5,color="#9aa7b5",ls=(0,(4,4)),lw=1)
        ax.text(.785,.54,"0.5% reference",fontsize=9,color="#718096")
        for v in ORDER:
            selected = [r for r in rows if r["variant"] == v and r["suite"] == suite]
            st = summary["panels"][suite][v]
            x,color = speed[v],COLORS[v]
            if suite == "core":
                ax.bxp([st],positions=[x],widths=.022 if v.startswith("E_") else .045,
                       manage_ticks=False,showfliers=False,patch_artist=True,
                       boxprops=dict(facecolor=color+"30",edgecolor=color,lw=1.5),
                       medianprops=dict(color=color,lw=2.5),whiskerprops=dict(color=color,lw=1.2),
                       capprops=dict(color=color,lw=1.2))
            else:
                ax.vlines(x,st["whislo"],st["whishi"],color=color,alpha=.6,lw=1.2)
                ax.hlines(st["med"],x-.012,x+.012,color=color,lw=2.5)
            for r in selected:
                marker = core_markers[r["k_length"]] if suite == "core" else stress_markers[r["distribution"]]
                ax.scatter([x],[r["metrics"]["l2_pct"]],marker=marker,s=24 if suite == "core" else 48,
                           color=color,edgecolors="white",linewidths=.4,alpha=.75,zorder=4)
            ax.annotate(v,(x,st["med"]),xytext=offsets[suite][v],textcoords="offset points",
                        color=color,fontsize=11,fontweight="bold",
                        arrowprops=dict(arrowstyle="-",color=color,lw=.8))
        markers = core_markers if suite == "core" else stress_markers
        labels = [f"{n//1024}K KV" for n in markers] if suite == "core" else ["Q +32","K +32","V +32"]
        ax.legend(handles=[Line2D([],[],marker=m,ls="none",color="#64748b",label=l)
                           for m,l in zip(markers.values(),labels)],loc="upper left",frameon=False,fontsize=9)
        ax.set_title("Broad suite · 18 cases / variant" if suite == "core" else "Common-mode stress · 3 cases / variant",
                     loc="left",fontsize=14,pad=15,fontweight="bold")
    axes[0].annotate("A: uniform attention, 256K",(speed["A"],summary["panels"]["core"]["A"]["whishi"]),
                     xytext=(-165,-4),textcoords="offset points",fontsize=9,color=COLORS["A"],
                     arrowprops=dict(arrowstyle="-",lw=.7,color=COLORS["A"]))
    tax = fig.add_axes([.07,.155,.895,.195]); tax.axis("off")
    labels = dict(D="HiFi4 / FP32",C="QK4-PV2 / FP32",B="HiFi2 / compensated BF16",A="Stock BF16 streaming",
                  E_bf16="LoFi / RNE5 BF16 KV",E_bfp8="LoFi / RNE5 → BFP8 KV",E_bfp4="LoFi / BFP4-grid RNE KV")
    table_rows = []
    for v in ORDER:
        c,s = (summary["panels"][suite][v] for suite in ("core","stress"))
        table_rows.append([v,labels[v],f"{speed[v]:.3f}",f"{c['med']:.3f}%",
                           f"{c['whislo']:.3f}–{c['whishi']:.3f}%",f"{s['whishi']:.3f}%"])
    table = tax.table(cellText=table_rows,colLabels=["Variant","Recipe","TFLOP/s/core","Broad median L2","Broad min–max L2","Stress max L2"],
                      colWidths=[.105,.285,.13,.15,.18,.15],cellLoc="left",colLoc="left",bbox=[0,0,1,1])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0"); cell.set_linewidth(.5)
        cell.set_facecolor("#e7edf4" if r==0 else ("white" if r%2 else "#f1f5f9"))
        if r==0: cell.set_text_props(fontweight="bold",color="#334155")
        elif c==0: cell.set_text_props(fontweight="bold",color=COLORS[ORDER[r-1]])
    notes = [
        "Boxes: Q1–Q3, median, min–max; dots: cases at true measured x (no jitter). E_bf16 and E_bfp8 largely overlap.",
        "Broad: normal, clipped ±2, Q/K ×0.25, Q/K ×2, sparse outliers, uniform attention; each at 4K / 32K / 256K KV.",
        "One head, 256 queries, D128, noncausal; same original BF16 inputs and FP64 reference. Stress: Q/K/V +32 at 32K KV.",
        "Resident Q256/K512, one core, repeated KV; preprocessing excluded. E trio remeasured together; D/C/B/A carried over.",
        "Case spread ≠ uncertainty. These points need not be nondominated; packed KV also reduces storage and ring communication.",
    ]
    for y,note in zip((.125,.101,.077,.053,.029),notes):
        fig.text(.07,y,note,fontsize=10,color="#475569")
    target = HERE / "sdpa_pareto_kv_precision.png"
    fig.savefig(target,dpi=190,facecolor=fig.get_facecolor())
    (HERE / "plot_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(target)


if __name__ == "__main__":
    main()
