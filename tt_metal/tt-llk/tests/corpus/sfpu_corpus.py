#!/usr/bin/env python3
"""Inventory and execute the complete tt-llk SFPU header corpus."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, pathlib, re, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
LLK = ROOT / "tt_metal/tt-llk"
MANIFEST = HERE / "sfpu_corpus_v1.tsv"
EXPECTED = {"bh": 41, "wh": 32, "raw": 20, "typed": 38, "replay": 6, "mop": 2}
FIELDS = ["version","id","arches","header_bh","header_wh","raw_tti","typed_sfpi","replay","mop",
          "functional_modules","perf_modules","mapping_state","notes"]

def headers(arch):
    base = LLK / f"tt_llk_{'blackhole' if arch == 'bh' else 'wormhole_b0'}/common/inc/sfpu"
    return {p.relative_to(base).as_posix(): p for p in base.rglob("*.h")}

def modules(prefix):
    p = LLK / "tests/python_tests"
    return sorted(x.relative_to(p).as_posix() for x in p.rglob(f"{prefix}*.py") if "__pycache__" not in x.parts)

def classify(text):
    return (bool(re.search(r"\bTTI_[A-Z0-9_]+", text)), bool(re.search(r"\bsfpi::|__builtin_rvtt_|using namespace sfpi", text)),
            bool(re.search(r"lltt::(?:record|replay)", text)), bool(re.search(r"\bTTI_MOP|\bMOP\b", text)))

def seed_maps():
    out = {}
    with (HERE / "f1_candidates.tsv").open() as f:
        for row in csv.reader(f, delimiter="\t"):
            if not row or row[0].startswith("#"): continue
            for path in row[4].split(","):
                name = pathlib.Path(path.replace("{blackhole,wormhole_b0}", "blackhole")).name
                out.setdefault(name, (row[6], row[8], row[9] + "; " + row[10]))
    return out

def inventory():
    bh, wh, seed = headers("bh"), headers("wh"), seed_maps()
    tests, perfs = modules("test_"), modules("perf_")
    rows=[]
    for rel,p in sorted(bh.items()):
        raw,typed,replay,mop=classify(p.read_text(errors="replace")); stem=p.stem.removeprefix("ckernel_sfpu_")
        mapped=seed.get(p.name)
        key=stem.replace("_", "")
        functional=mapped[0] if mapped else ",".join(x for x in tests if key in x.replace("_", ""))
        perf=mapped[1] if mapped else ",".join(x for x in perfs if key in x.replace("_", ""))
        state="mapped" if functional else "unmapped"
        rows.append(dict(version="1",id=rel.removesuffix(".h").replace("/","__"),arches="bh,wh" if rel in wh else "bh",
          header_bh=p.relative_to(ROOT).as_posix(),header_wh=wh[rel].relative_to(ROOT).as_posix() if rel in wh else "",
          raw_tti=str(int(raw)),typed_sfpi=str(int(typed)),replay=str(int(replay)),mop=str(int(mop)),
          functional_modules=functional,perf_modules=perf,mapping_state=state,
          notes=mapped[2] if mapped else "explicitly unmapped: no audited functional module"))
    return rows

def read_manifest():
    with MANIFEST.open() as f: return list(csv.DictReader((x for x in f if not x.startswith("#")), delimiter="\t"))

def write_manifest(rows):
    with MANIFEST.open("w", newline="") as f:
        f.write("# sfpu-corpus-manifest-version\t1\n")
        w=csv.DictWriter(f,FIELDS,delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(rows)

def validate(rows):
    inv=inventory(); errors=[]
    if rows != inv: errors.append("manifest differs from discovered inventory; run --update")
    counts={"bh":len(inv),"wh":sum("wh" in r["arches"].split(",") for r in inv)}
    for key in ("raw","typed","replay","mop"):
        col={"raw":"raw_tti","typed":"typed_sfpi","replay":"replay","mop":"mop"}[key]
        counts[key]=sum(r[col]=="1" for r in inv)
    for k,v in EXPECTED.items():
        if counts[k]!=v: errors.append(f"inventory drift: {k} expected {v}, found {counts[k]}")
    wh=set(headers("wh")); bh=set(headers("bh"))
    if not wh <= bh: errors.append("Wormhole headers are not a subset of Blackhole")
    for r in inv:
        if r["mapping_state"] not in ("mapped","unmapped"): errors.append(f"bad mapping state: {r['id']}")
    return errors,counts

def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def emit_summary(run, records, provenance):
    (run/"results.json").write_text(json.dumps({"provenance":provenance,"results":records},indent=2)+"\n")
    with (run/"results.tsv").open("w",newline="") as f:
        keys=["id","arch","mode","status","reason","artifact"]
        w=csv.DictWriter(f,keys,delimiter="\t",extrasaction="ignore",lineterminator="\n"); w.writeheader(); w.writerows(records)
    lines=["# SFPU corpus run","",f"- mode: `{provenance['mode']}`",f"- revision: `{provenance['tt_metal_head']}`","",
           "| id | arch | status | reason |","|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['arch']} | {r['status']} | {r['reason']} |" for r in records]
    (run/"summary.md").write_text("\n".join(lines)+"\n")

def compare_baseline(records, baseline, threshold):
    old=json.loads(baseline.read_text()).get("results",[])
    index={(r.get("id"),r.get("arch"),r.get("mode")):r for r in old}; compared=[]
    for r in records:
        prior=index.get((r["id"],r["arch"],r["mode"])); now=r.get("cycles"); before=prior and prior.get("cycles")
        if not isinstance(now,(int,float)) or not isinstance(before,(int,float)) or before==0:
            compared.append({"id":r["id"],"status":"SKIP_NO_DEVICE_CYCLES","reason":"both runs need numeric device cycles"}); continue
        delta=100.0*(now-before)/before
        compared.append({"id":r["id"],"status":"REGRESSION" if delta>threshold else "PASS","delta_pct":delta})
    return compared

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--update",action="store_true"); ap.add_argument("--validate",action="store_true")
    ap.add_argument("--list",action="store_true"); ap.add_argument("--mode",choices=("compile","craq","silicon")); ap.add_argument("--arch",choices=("bh","wh"),default="bh")
    ap.add_argument("--run-root",type=pathlib.Path); ap.add_argument("--simulator",type=pathlib.Path); ap.add_argument("--baseline",type=pathlib.Path)
    ap.add_argument("--max-regression-pct",type=float,default=0.0); ap.add_argument("--execute",action="store_true",help="execute the selected mode (otherwise emit a plan)")
    a=ap.parse_args(); rows=inventory()
    if a.update: write_manifest(rows)
    current=read_manifest() if MANIFEST.exists() else []
    errors,counts=validate(current)
    if a.validate or a.update: print(json.dumps({"counts":counts,"errors":errors},sort_keys=True));
    if errors and (a.validate or a.mode): return 1
    selected=[r for r in current if a.arch in r["arches"].split(",")]
    if a.list:
        for r in selected: print("\t".join((r["id"],r["arches"],r["mapping_state"],r["functional_modules"],r["perf_modules"])))
    if not a.mode: return 0
    run=(a.run_root or HERE/"runs"/(time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+f"-{a.arch}-{a.mode}")); run.mkdir(parents=True,exist_ok=False)
    head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    prov={"schema":1,"mode":a.mode,"arch":a.arch,"tt_metal_head":head,"manifest_sha256":sha(MANIFEST),"simulator":str(a.simulator or ""),"threshold_pct":a.max_regression_pct}
    records=[]
    for r in selected:
        mods=r["functional_modules" if a.mode!="silicon" else "perf_modules"]
        if not mods: records.append({"id":r["id"],"arch":a.arch,"mode":a.mode,"status":"SKIP_UNMAPPED","reason":"no audited module mapping","artifact":""}); continue
        if a.mode=="craq" and (not a.simulator or not a.simulator.is_file()): records.append({"id":r["id"],"arch":a.arch,"mode":a.mode,"status":"SKIP_NO_SIMULATOR","reason":"--simulator required","artifact":""}); continue
        if not a.execute:
            status="SKIP_HARDWARE_NOT_AUTHORIZED" if a.mode=="silicon" else "PLAN_ONLY"
            records.append({"id":r["id"],"arch":a.arch,"mode":a.mode,"status":status,"reason":mods,"artifact":""}); continue
        records.append({"id":r["id"],"arch":a.arch,"mode":a.mode,"status":"QUEUED","reason":mods,"artifact":""})
    queued=[r for r in records if r["status"]=="QUEUED"]
    if queued:
        pydir=LLK/"tests/python_tests"; python=pydir/".venv/bin/python"; log=run/f"{a.mode}.log"
        mods=sorted({m for r in queued for m in r["reason"].split(",") if m and " " not in m})
        env=os.environ.copy(); env.update({"TT_METAL_HOME":str(ROOT),"SHORT_ARCH":a.arch,
            "SIM_ARCH":"blackhole" if a.arch=="bh" else "wormhole"})
        if not python.is_file():
            rc=None; why="missing tt-llk .venv"
        elif a.mode=="compile":
            cmd=[str(python),"-m","pytest","-o","addopts=",*mods,"--compile-producer","-q"]; why="compile gate"
            with log.open("w") as f: rc=subprocess.run(cmd,cwd=pydir,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        elif a.mode=="craq":
            runner=pathlib.Path(os.environ.get("CRAQ_SIM_ROOT","/localdev/nkapre/craq-sim"))/"scripts/perf/llk-sim-perf.sh"
            cmd=[str(runner),"--sample","1","--run-root",str(run/"craq")]+sum((["--module",m] for m in mods),[])
            env["SIMULATOR"]=str(a.simulator); why="CRAQ device-cycle gate"
            with log.open("w") as f: rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        else:
            cmd=[str(python),"-m","pytest","-o","addopts=",*mods,"-q"]; why="serialized silicon gate"
            lock=HERE/"sfpu-silicon.lock"
            import fcntl
            with lock.open("w") as lk, log.open("w") as f:
                fcntl.flock(lk,fcntl.LOCK_EX); rc=subprocess.run(cmd,cwd=pydir,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        for rec in queued:
            rec["status"]="SKIP_MISSING_ENV" if rc is None else ("PASS" if rc==0 else "FAIL")
            rec["reason"]=why; rec["artifact"]=str(log) if log.exists() else ""
    if a.baseline:
        comparisons=compare_baseline(records,a.baseline,a.max_regression_pct)
        (run/"comparison.json").write_text(json.dumps(comparisons,indent=2)+"\n")
        prov["baseline"]=str(a.baseline)
    emit_summary(run,records,prov); print(run)
    return 0
if __name__=="__main__": raise SystemExit(main())
