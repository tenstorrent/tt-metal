"""Paired accuracy/performance points with and without external rounding."""
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
V3 = HERE.parent
FAMILY = ("E_bf16", "E_bfp8", "E_bfp4")
COLORS = dict(D="#3658a2", C="#8155a6", B="#258657", A="#64748b",
              E_bf16="#006b66", E_bfp8="#00a9ca", E_bfp4="#d87d24")


def read(p):
    return json.loads(p.read_text())


def main():
    paths = [V3/"kv-precision-v1/accuracy-v1.json", HERE/"accuracy-v1.json",
             V3/"kv-precision-v1/plot_summary.json", HERE/"perf-v1.json"]
    rounded, plain, prior, perf = map(read, paths)
    for x in (rounded, plain, perf):
        assert x["complete"] and x["selected_sources_immutable"]
    assert len(plain["records"]) == 63
    for r in plain["records"]:
        old = next(x for x in rounded["records"] if (x["variant"],x["suite"],x["k_length"],x["distribution"]) ==
                   (r["base_variant"],r["suite"],r["k_length"],r["distribution"]))
        assert r["input_sha256"] == old["input_sha256"] and r["metadata"] == old["metadata"]
        assert r["raw_trace_equal"] and r["inputs_immutable"] and r["prepared_immutable"]
    rows = rounded["records"] + plain["records"]
    speed = dict(prior["throughput_tflops_per_core"])
    speed.update({r["variant"]:r["tflops_per_core"] for r in perf["records"]})
    stats = {}
    for v in FAMILY:
        for name in (v,v+"_plain"):
            stats[name] = {}
            for suite in ("core","stress"):
                ys = np.array([r["metrics"]["l2_pct"] for r in rows if r["variant"] == name and r["suite"] == suite])
                assert len(ys) == (18 if suite == "core" else 3)
                stats[name][suite] = dict(med=float(np.median(ys)),q1=float(np.quantile(ys,.25)),
                    q3=float(np.quantile(ys,.75)),whislo=float(ys.min()),whishi=float(ys.max()),fliers=[])
    summary = dict(throughput_tflops_per_core=speed,statistics=stats,
                  sources={str(p.relative_to(V3)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axes = plt.subplots(1,2,figsize=(17,10),gridspec_kw={"width_ratios":[1.15,1]})
    fig.patch.set_facecolor("#f8fafc")
    fig.subplots_adjust(left=.07,right=.965,bottom=.36,top=.82,wspace=.20)
    fig.text(.07,.95,"SDPA frontier · removing input rounding",fontsize=26,fontweight="bold",color="#17243a")
    fig.text(.07,.905,"Solid / filled: special rounding     •     Dashed / hollow: ordinary inputs and standard casts",fontsize=14,color="#475569")
    fig.text(.07,.87,"No changes to compute, internal arithmetic, Q256/K512/D128 or buffering. Only the three E variants need external rounding.",fontsize=11,color="#475569")
    for ax,suite in zip(axes,("core","stress")):
        ax.set_facecolor("white"); ax.set_yscale("log")
        ax.grid(axis="y",which="major",color="#e2e8f0"); ax.set_axisbelow(True)
        ax.set_ylabel("Relative L2 error (%) · log scale")
        ax.axhline(.5,color="#94a3b8",ls=":",lw=1)
        if suite == "core":
            ax.set_xlim(.77,2.3); ax.set_ylim(.08,140)
            ax.set_xlabel("Resident useful TFLOP/s per core  →  faster")
            for v in "DCBA":
                st = prior["panels"]["core"][v]
                ax.bxp([st],positions=[speed[v]],widths=.034,manage_ticks=False,showfliers=False,
                       patch_artist=True,boxprops=dict(facecolor=COLORS[v]+"30",edgecolor=COLORS[v]),
                       medianprops=dict(color=COLORS[v],lw=2),whiskerprops=dict(color=COLORS[v]),
                       capprops=dict(color=COLORS[v]))
                ax.annotate(v,(speed[v],st["med"]),xytext=(-17,-24),textcoords="offset points",
                            color=COLORS[v],fontweight="bold",fontsize=12)
        else:
            ax.set_xlim(-.5,2.5)
            ax.set_ylim(.15,max(stats[n]["stress"]["whishi"] for n in stats)*2)
            ax.set_xticks(range(3),FAMILY)
            ax.set_xlabel("Paired E recipes (categorical x; throughput shown at left)")
        for i,v in enumerate(FAMILY):
            for plain_flag in (False,True):
                name = v+"_plain" if plain_flag else v
                st = stats[name][suite]; color=COLORS[v]
                x = speed[name] if suite == "core" else i+(.12 if plain_flag else -.12)
                ls = "--" if plain_flag else "-"
                if suite == "core":
                    ax.bxp([st],positions=[x],widths=.028 if plain_flag else .012,manage_ticks=False,showfliers=False,patch_artist=True,
                           boxprops=dict(facecolor="none" if plain_flag else color+"45",edgecolor=color,ls=ls,lw=1.5),
                           medianprops=dict(color=color,lw=2),whiskerprops=dict(color=color,ls=ls,lw=1.2),
                           capprops=dict(color=color,lw=1.2))
                else:
                    ax.vlines(x,st["whislo"],st["whishi"],color=color,linestyles=ls,alpha=.6)
                    for r in rows:
                        if r["variant"] == name and r["suite"] == suite:
                            marker = {"common_q":"^","common_k":"s","common_v":"o"}[r["distribution"]]
                            ax.scatter([x],[r["metrics"]["l2_pct"]],marker=marker,s=65,
                                       facecolors="none" if plain_flag else color,edgecolors=color,lw=1.4)
            if suite == "core":
                ax.plot([speed[v],speed[v+"_plain"]],[stats[v][suite]["med"],stats[v+"_plain"][suite]["med"]],
                        color=COLORS[v],alpha=.45,lw=.9)
        ax.set_title("Broad suite · 18 cases / point" if suite == "core" else "Common-mode stress · Q / K / V +32",
                     loc="left",fontsize=14,fontweight="bold",pad=15)
    axes[0].legend(handles=[Line2D([],[],color=COLORS[v],lw=3,label=v) for v in FAMILY],
                   loc="upper left",frameon=False)
    axes[1].legend(handles=[Line2D([],[],color="#64748b",ls="none",marker=m,label=n)
                           for m,n in (("^","Q +32"),("s","K +32"),("o","V +32"))],loc="upper left",frameon=False)
    worst = max((r for r in plain["records"] if r["suite"] == "stress"),key=lambda r:r["metrics"]["l2_pct"])
    wx = FAMILY.index(worst["base_variant"])+.12
    axes[1].annotate(f"{worst['metrics']['l2_pct']:,.0f}% · standard BFP4 cast",(wx,worst["metrics"]["l2_pct"]),
                     xytext=(-205,-6),textcoords="offset points",fontsize=10,color=COLORS[worst["base_variant"]],
                     arrowprops=dict(arrowstyle="-",color=COLORS[worst["base_variant"]],lw=.8))
    tax=fig.add_axes([.07,.19,.895,.105]); tax.axis("off")
    values=[]
    for v in FAMILY:
        a,b=stats[v]["core"],stats[v+"_plain"]["core"]
        values.append([v,f"{speed[v]:.3f} → {speed[v+'_plain']:.3f}",f"{a['med']:.3f}% → {b['med']:.3f}%",
                       f"{a['whislo']:.3f}–{a['whishi']:.3f}%",f"{b['whislo']:.3f}–{b['whishi']:.3f}%"])
    table=tax.table(cellText=values,colLabels=["Variant","TFLOP/s/core: rounded → plain","Median L2: rounded → plain","Rounded L2 min–max","Plain L2 min–max"],
                    colWidths=[.10,.25,.25,.20,.20],cellLoc="left",colLoc="left",bbox=[0,0,1,1])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor("#e2e8f0"); cell.set_facecolor("#e7edf4" if r==0 else "white")
        if r==0:cell.set_text_props(fontweight="bold")
        elif c==0:cell.set_text_props(color=COLORS[FAMILY[r-1]],fontweight="bold")
    notes=["Broad: normal, clipped ±2, Q/K ×0.25, Q/K ×2, sparse outliers, uniform attention; each at 4K / 32K / 256K KV.",
           "Plain: unmodified BF16 Q; KV unmodified BF16 or standard device typecast to BFP8/BFP4. No Q RNE7 / KV RNE5 / custom BFP4 quantizer.",
           "One head, 256 queries, D128, noncausal; matched original BF16 inputs, FP64 reference. Stress uses 32K KV, excluded from broad boxes.",
           "Resident timing excludes all preprocessing/casts; boxes show case Q1–Q3 and min–max, not uncertainty. Prior rounded/D/C/B/A evidence retained."]
    for y,note in zip((.145,.11,.075,.04),notes):fig.text(.07,y,note,fontsize=10,color="#475569")
    path=HERE/"sdpa_pareto_no_rounding.png"
    fig.savefig(path,dpi=190,facecolor=fig.get_facecolor())
    (HERE/"plot_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
