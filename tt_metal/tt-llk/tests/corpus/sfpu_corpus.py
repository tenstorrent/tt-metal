#!/usr/bin/env python3
"""Inventory and execute the complete tt-llk SFPU header corpus."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, pathlib, re, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[4]
HERE = pathlib.Path(__file__).resolve().parent
LLK = ROOT / "tt_metal/tt-llk"
MANIFEST = HERE / "sfpu_corpus_v2.tsv"
DEVICE_BASELINE = HERE / "sfpu_device_baseline_v1.tsv"
EXPECTED = {"logical":164,"bh":152,"wh":138,"qsr":42,"physical_paths":332,"basename_union":143,"legacy_bh":41,"legacy_wh":32,"legacy_qsr":14,"raw":51,"typed":151,"replay":13,"mop":3}
DISCOVERY_FIELDS = ["id","surface","arches","header_bh","header_wh","header_qsr","raw_tti","typed_sfpi","replay","mop",
                    "functional_modules","perf_modules","mapping_state","notes"]
AUDIT_FIELDS = ["semantic_cpp_class","semantic_cpp_blocker","paired_selector_status","test_status","perf_status",
                "correctness_metric","correctness_threshold","correctness_source","silicon_status","silicon_result","silicon_source"]
FIELDS = ["version", *DISCOVERY_FIELDS, *AUDIT_FIELDS]
SEMANTIC_CPP_CLASSES = {"ready", "typed_wrapper_needed", "macro_dependent", "multithread_boundary", "unmapped"}
PAIR_STATUSES = {"absent", "blocked", "implemented"}
GATE_STATUSES = {"not_run", "blocked", "pass", "fail"}
PERF_STATUSES = {"not_run", "blocked", "measured"}
SILICON_STATUSES = {"not_run", "blocked", "win", "parity", "loss"}
CORRECTNESS_METRICS = {"none", "pcc", "exact", "tolerance"}

# Every exception is keyed by the complete stable corpus ID.  There are no
# basename fragments, substring matches, or inferred semantic classifications.
AUDITED_SEEDS = {
    "legacy__ckernel_sfpu_welfords": dict(
        semantic_cpp_class="typed_wrapper_needed", semantic_cpp_blocker="Generated vFloat body exists; raw LREG live-in/live-out ABI remains an explicit typed-boundary requirement.",
        paired_selector_status="implemented", test_status="pass", perf_status="measured", correctness_metric="tolerance",
        correctness_threshold="mean rtol=0.02 atol=0.02; m2 rtol=0.03 atol=0.03", correctness_source="test_sfpu_welford_prefix_snapshot.py:77-78",
        silicon_status="win", silicon_result="BH WELFORD_BODY: generated 323 cycles vs handwritten 326 (-0.92%); scoped body metric.", silicon_source="sfpu_device_baseline_v1.tsv; SFPI_COMPILER_UPGRADE.md section 13.8"),
    "legacy__ckernel_sfpu_reduce_custom": dict(
        semantic_cpp_class="typed_wrapper_needed", semantic_cpp_blocker="Arithmetic is semantic SFPI; destination loads, TTINCRWC barrier, L8 discard load, and replay ownership require typed architectural boundaries.",
        paired_selector_status="implemented", test_status="pass", perf_status="measured", correctness_metric="pcc",
        correctness_threshold="PCC > 0.99 and Float16_b element tolerance rtol=0.05 atol=0.05", correctness_source="test_sfpu_reduce_sdpa.py:135; helpers/utils.py:548-785",
        silicon_status="win", silicon_result="BH REDUCE_SDPA_BODY: D1 generated 834 cycles vs handwritten 840 (-0.714%); three fresh samples.", silicon_source="REDUCE_SDPA_SILICON_AB.md; sfpu_device_baseline_v1.tsv"),
    "legacy__ckernel_sfpu_binary_bcast": dict(
        semantic_cpp_class="typed_wrapper_needed", semantic_cpp_blocker="Arithmetic island is semantic vFloat; broadcast addressing, address modifiers, fixed-LREG endpoints, and replay remain explicit architectural boundaries.",
        paired_selector_status="implemented", test_status="pass", perf_status="measured", correctness_metric="pcc",
        correctness_threshold="PCC > 0.99 and Float16_b element tolerance rtol=0.05 atol=0.05", correctness_source="test_sfpu_binary.py:1748-1749; helpers/utils.py:548-785",
        silicon_status="parity", silicon_result="BH BINARY_BCAST_BODY: generated 608 cycles vs handwritten 608 (exact cycle parity), three fresh samples.", silicon_source="BINARY_BCAST_SILICON_AB.md; sfpu_device_baseline_v1.tsv"),
    "legacy__ckernel_sfpu_where": dict(
        semantic_cpp_class="macro_dependent", semantic_cpp_blocker="Canonical v_if selector is correct, but competitive lowering requires general SFPLOADMACRO formation; generated replay payload is seven slots versus three handwritten slots.",
        paired_selector_status="implemented", test_status="pass", perf_status="measured", correctness_metric="exact",
        correctness_threshold="bit-exact selected Float16_b payload with NaNs equal; no tolerance", correctness_source="test_sfpu_ternary.py:298-321",
        silicon_status="loss", silicon_result="BH TTNN_WHERE_BODY: generated 312.50 cycles vs handwritten 159.25 (+96.23%), three fresh samples.", silicon_source="TTNN_WHERE_COMPILER_AB.md; sfpu_device_baseline_v1.tsv"),
    "legacy__ckernel_sfpu_mul_int": dict(
        semantic_cpp_class="macro_dependent", semantic_cpp_blocker="Fresh integer arithmetic selector is correct, but competitive lowering requires general SFPLOADMACRO formation and typed mul24/shift/saturation scheduling.",
        paired_selector_status="implemented", test_status="pass", perf_status="measured", correctness_metric="exact",
        correctness_threshold="Int32 element tolerance rtol=0 atol=0 plus PCC > 0.99 when signal is nonzero", correctness_source="test_sfpu_binary.py:902-921; helpers/utils.py:548-785",
        silicon_status="loss", silicon_result="BH MATH_ISOLATE: generated 562.625 cycles vs handwritten 283.9296875 (+98.16%).", silicon_source="sfpu_device_baseline_v1.tsv; audited BH device archive"),
    "metal__ckernel_sfpu_mul_int32": dict(
        semantic_cpp_class="macro_dependent", semantic_cpp_blocker="Production Metal implementation owns SFPLOADMACRO scheduling; paired test selector is mapped through the legacy test surface pending a direct row mapping.",
        paired_selector_status="implemented", test_status="pass", perf_status="measured", correctness_metric="exact",
        correctness_threshold="Int32 element tolerance rtol=0 atol=0 plus PCC > 0.99 when signal is nonzero", correctness_source="test_sfpu_binary.py:902-921; helpers/utils.py:548-785",
        silicon_status="loss", silicon_result="BH MATH_ISOLATE: generated 562.625 cycles vs handwritten 283.9296875 (+98.16%).", silicon_source="sfpu_device_baseline_v1.tsv; audited BH device archive"),
    "legacy__ckernel_sfpu_topk": dict(
        semantic_cpp_class="typed_wrapper_needed", semantic_cpp_blocker="Needs sound multi-result indexed SFPSWAP and eight-value SFPTRANSP modeling plus explicit RWC/DST/config/replay boundaries before a full selector is accepted.",
        paired_selector_status="blocked", test_status="blocked", perf_status="blocked", correctness_metric="exact",
        correctness_threshold="exact value/index association; exact stable indices, tie-equivalent indices only for explicitly unstable cases", correctness_source="test_topk.py:134-240; TOPK_TYPED_CONVERSION_BLOCKER.md",
        silicon_status="blocked", silicon_result="No semantically complete paired selector or isolated silicon result.", silicon_source="TOPK_TYPED_CONVERSION_BLOCKER.md; sfpu_device_baseline_v1.tsv"),
}

DEFAULT_AUDIT = dict(
    semantic_cpp_class="unmapped", semantic_cpp_blocker="No row-specific semantic-C++ conversion audit or paired selector has been completed.",
    paired_selector_status="absent", test_status="not_run", perf_status="not_run", correctness_metric="none", correctness_threshold="not established",
    correctness_source="none", silicon_status="not_run", silicon_result="No paired silicon measurement.", silicon_source="none")

def headers(arch):
    chip={"bh":"blackhole","wh":"wormhole_b0","qsr":"quasar"}[arch]
    roots={"legacy":LLK/f"tt_llk_{chip}/common/inc/sfpu", "metal":ROOT/f"tt_metal/hw/ckernels/{chip}"}
    out={f"{surface}:{p.name}":p for surface,base in roots.items() for p in base.rglob("ckernel_sfpu*.h")}
    if arch=="qsr":
        base=LLK/"tt_llk_quasar/common/inc"
        for p in base.rglob("ckernel_sfpu*.h"):
            out[f"legacy:{p.name}"]=p
    return out

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
    bh, wh, qsr, seed = headers("bh"), headers("wh"), headers("qsr"), seed_maps()
    tests, perfs = modules("test_"), modules("perf_")
    rows=[]
    for rel in sorted(set(bh)|set(wh)|set(qsr)):
        p=bh.get(rel) or wh.get(rel) or qsr[rel]
        raw,typed,replay,mop=classify(p.read_text(errors="replace")); stem=p.stem.removeprefix("ckernel_sfpu_")
        surface,shortrel=rel.split(":",1)
        mapped=seed.get(p.name) if surface=="legacy" else None
        # Mapping is evidence, not name similarity.  Only the reviewed override
        # seed may map a header; every other header remains explicitly unmapped.
        functional=mapped[0] if mapped else ""
        perf=mapped[1] if mapped else ""
        state="mapped" if functional else "unmapped"
        arches=",".join(a for a,d in (("bh",bh),("wh",wh),("qsr",qsr)) if rel in d)
        row=dict(version="2",id=(surface+"__"+shortrel.removesuffix(".h").replace("/","__")),surface=surface,arches=arches,
          header_bh=bh[rel].relative_to(ROOT).as_posix() if rel in bh else "",header_wh=wh[rel].relative_to(ROOT).as_posix() if rel in wh else "",header_qsr=qsr[rel].relative_to(ROOT).as_posix() if rel in qsr else "",
          raw_tti=str(int(raw)),typed_sfpi=str(int(typed)),replay=str(int(replay)),mop=str(int(mop)),
          functional_modules=functional,perf_modules=perf,mapping_state=state,
          notes=mapped[2] if mapped else "explicitly unmapped: no audited functional module")
        row.update(DEFAULT_AUDIT); row.update(AUDITED_SEEDS.get(row["id"], {})); rows.append(row)
    return rows

def read_manifest():
    with MANIFEST.open() as f: return list(csv.DictReader((x for x in f if not x.startswith("#")), delimiter="\t"))

def write_manifest(rows):
    with MANIFEST.open("w", newline="") as f:
        f.write("# sfpu-corpus-manifest-version\t2\n")
        w=csv.DictWriter(f,FIELDS,delimiter="\t",lineterminator="\n"); w.writeheader(); w.writerows(rows)

def validate(rows):
    inv=inventory(); errors=[]
    by_id={r.get("id"):r for r in rows}; inv_by_id={r["id"]:r for r in inv}
    if set(by_id) != set(inv_by_id): errors.append("manifest ID set differs from discovered inventory; audit additions/removals before --update")
    for row_id in sorted(set(by_id) & set(inv_by_id)):
        for field in ["version", *DISCOVERY_FIELDS]:
            if by_id[row_id].get(field) != inv_by_id[row_id].get(field):
                errors.append(f"manifest discovery drift: {row_id}.{field}")
        row=by_id[row_id]
        if row.get("semantic_cpp_class") not in SEMANTIC_CPP_CLASSES: errors.append(f"bad semantic_cpp_class: {row_id}")
        if row.get("paired_selector_status") not in PAIR_STATUSES: errors.append(f"bad paired selector status: {row_id}")
        if row.get("test_status") not in GATE_STATUSES: errors.append(f"bad test status: {row_id}")
        if row.get("perf_status") not in PERF_STATUSES: errors.append(f"bad perf status: {row_id}")
        if row.get("correctness_metric") not in CORRECTNESS_METRICS: errors.append(f"bad correctness metric: {row_id}")
        if row.get("silicon_status") not in SILICON_STATUSES: errors.append(f"bad silicon status: {row_id}")
        if not row.get("semantic_cpp_blocker"): errors.append(f"missing exact blocker/readiness statement: {row_id}")
        if row.get("silicon_status") in {"win","parity","loss"}:
            if row.get("test_status") != "pass" or row.get("correctness_metric") == "none":
                errors.append(f"ungated silicon result: {row_id}")
            if row.get("perf_status") != "measured": errors.append(f"silicon result without measured perf: {row_id}")
    counts={"logical":len(inv)}
    for a in ("bh","wh","qsr"): counts[a]=sum(a in r["arches"].split(",") for r in inv)
    for a in ("bh","wh","qsr"): counts[f"legacy_{a}"]=sum(r["surface"]=="legacy" and a in r["arches"].split(",") for r in inv)
    counts["physical_paths"]=counts["bh"]+counts["wh"]+counts["qsr"]
    counts["basename_union"]=len({pathlib.Path(next(r[x] for x in ("header_bh","header_wh","header_qsr") if r[x])).name for r in inv})
    for key in ("raw","typed","replay","mop"):
        col={"raw":"raw_tti","typed":"typed_sfpi","replay":"replay","mop":"mop"}[key]
        counts[key]=sum(r[col]=="1" for r in inv)
    for k,v in EXPECTED.items():
        if counts[k]!=v: errors.append(f"inventory drift: {k} expected {v}, found {counts[k]}")
    wh=set(headers("wh")); bh=set(headers("bh"))
    if not wh <= bh: errors.append("Wormhole headers are not a subset of Blackhole")
    for r in rows:
        if r["mapping_state"] not in ("mapped","unmapped"): errors.append(f"bad mapping state: {r['id']}")
    return errors,counts

def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def emit_summary(run, records, provenance):
    (run/"results.json").write_text(json.dumps({"provenance":provenance,"results":records},indent=2)+"\n")
    with (run/"results.tsv").open("w",newline="") as f:
        keys=["id","arch","mode","status","reason","artifact",*AUDIT_FIELDS]
        w=csv.DictWriter(f,keys,delimiter="\t",extrasaction="ignore",lineterminator="\n"); w.writeheader(); w.writerows(records)
    lines=["# SFPU corpus run","",f"- mode: `{provenance['mode']}`",f"- revision: `{provenance['tt_metal_head']}`","",
           "| id | arch | status | semantic C++ | correctness gate | silicon | reason |","|---|---|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['arch']} | {r['status']} | {r.get('semantic_cpp_class','')} | {r.get('correctness_metric','')}: {r.get('correctness_threshold','')} | {r.get('silicon_status','')}: {r.get('silicon_result','')} | {r['reason']} |" for r in records]
    (run/"summary.md").write_text("\n".join(lines)+"\n")

def load_baseline(path):
    if path.suffix == ".json":
        return json.loads(path.read_text()).get("results", [])
    with path.open() as f:
        return list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))

def compare_baseline(records, baseline, threshold):
    old=load_baseline(baseline)
    key=lambda r:(r.get("id"),r.get("arch"),r.get("metric"),r.get("scope"),r.get("selector"))
    samples={}
    for row in old:
        try: cycles=float(row.get("cycles", ""))
        except (TypeError, ValueError): continue
        samples.setdefault(key(row), []).append(cycles)
    index={k:min(v) for k,v in samples.items()}; compared=[]
    for r in records:
        before=index.get(key(r)); now=r.get("cycles")
        if not isinstance(now,(int,float)) or not isinstance(before,(int,float)) or before==0:
            compared.append({"id":r["id"],"status":"SKIP_NO_DEVICE_CYCLES","reason":"both runs need numeric device cycles"}); continue
        delta=100.0*(now-before)/before
        compared.append({"id":r["id"],"status":"REGRESSION" if delta>threshold else "PASS","delta_pct":delta})
    return compared

def emit_plan(rows, arch, fmt):
    keys=["id","arches","mapping_state","functional_modules","perf_modules",*AUDIT_FIELDS]
    if fmt == "json":
        print(json.dumps({"schema":2,"arch":arch,"rows":[{k:r.get(k,"") for k in keys} for r in rows]},indent=2))
    elif fmt == "markdown":
        print("| id | arches | semantic C++ | selector | test | perf | correctness | silicon |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['id']} | {r['arches']} | {r['semantic_cpp_class']} | {r['paired_selector_status']} | {r['test_status']} | {r['perf_status']} | {r['correctness_metric']}: {r['correctness_threshold']} | {r['silicon_status']}: {r['silicon_result']} |")
    else:
        for r in rows: print("\t".join(r.get(k,"") for k in keys))

def record(row, arch, mode, status, reason, artifact=""):
    return {"id":row["id"],"arch":arch,"mode":mode,"status":status,"reason":reason,"artifact":artifact,
            **{k:row.get(k,"") for k in AUDIT_FIELDS}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--update",action="store_true"); ap.add_argument("--validate",action="store_true")
    ap.add_argument("--list",action="store_true"); ap.add_argument("--mode",choices=("compile","craq","silicon")); ap.add_argument("--arch",choices=("bh","wh","qsr"),default="bh")
    ap.add_argument("--plan-format",choices=("tsv","json","markdown"),default="tsv")
    ap.add_argument("--run-root",type=pathlib.Path); ap.add_argument("--simulator",type=pathlib.Path); ap.add_argument("--baseline",type=pathlib.Path)
    ap.add_argument("--max-regression-pct",type=float,default=0.0); ap.add_argument("--execute",action="store_true",help="execute the selected mode (otherwise emit a plan)")
    ap.add_argument("--require-executed-mapped",action="store_true",help="fail unless at least one mapped row executed and every mapped row passed")
    ap.add_argument("--allow-hardware",action="store_true"); ap.add_argument("--hardware-lock",type=pathlib.Path,default=pathlib.Path("/tmp/tt-llk-sfpu-silicon.lock"))
    ap.add_argument("--measurements",type=pathlib.Path,help="silicon TSV: id,arch,metric,scope,selector,cycles")
    ap.add_argument("--compare-results",type=pathlib.Path,help="compare an existing results.json to --baseline")
    a=ap.parse_args(); rows=inventory()
    if a.compare_results:
        if not a.baseline: ap.error("--compare-results requires --baseline")
        results=json.loads(a.compare_results.read_text()).get("results",[])
        out=compare_baseline(results,a.baseline,a.max_regression_pct); print(json.dumps(out,indent=2))
        return int(any(x["status"]=="REGRESSION" for x in out))
    if a.update:
        # Discovery refreshes paths/features/mappings, but reviewed semantic
        # audit fields are durable data and must survive regeneration.
        if MANIFEST.exists():
            reviewed={r["id"]:r for r in read_manifest()}
            for row in rows:
                if row["id"] in reviewed:
                    row.update({k:reviewed[row["id"]].get(k,"") for k in AUDIT_FIELDS})
        write_manifest(rows)
    current=read_manifest() if MANIFEST.exists() else []
    errors,counts=validate(current)
    if a.validate or a.update: print(json.dumps({"counts":counts,"errors":errors},sort_keys=True));
    if errors and (a.validate or a.mode): return 1
    selected=[r for r in current if a.arch in r["arches"].split(",")]
    if a.list:
        emit_plan(selected,a.arch,a.plan_format)
    if not a.mode: return 0
    run=(a.run_root or HERE/"runs"/(time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+f"-{a.arch}-{a.mode}")); run.mkdir(parents=True,exist_ok=False)
    head=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    prov={"schema":1,"mode":a.mode,"arch":a.arch,"tt_metal_head":head,"manifest_sha256":sha(MANIFEST),"simulator":str(a.simulator or ""),"threshold_pct":a.max_regression_pct,"hardware_lock":str(a.hardware_lock)}
    records=[]
    for r in selected:
        mods=r["functional_modules" if a.mode!="silicon" else "perf_modules"]
        if not mods: records.append(record(r,a.arch,a.mode,"SKIP_UNMAPPED","no audited module mapping")); continue
        if a.mode=="silicon" and (r["paired_selector_status"] != "implemented" or r["test_status"] != "pass" or r["correctness_metric"] == "none"):
            records.append(record(r,a.arch,a.mode,"SKIP_CORRECTNESS_NOT_GATED","silicon requires an implemented selector and an explicit passing correctness metric")); continue
        if a.mode=="craq" and (not a.simulator or not a.simulator.is_file()): records.append(record(r,a.arch,a.mode,"SKIP_NO_SIMULATOR","--simulator required")); continue
        if a.mode=="silicon" and (not a.execute or not a.allow_hardware):
            records.append(record(r,a.arch,a.mode,"SKIP_HARDWARE_NOT_AUTHORIZED","requires --execute --allow-hardware")); continue
        if not a.execute:
            status="SKIP_HARDWARE_NOT_AUTHORIZED" if a.mode=="silicon" else "PLAN_ONLY"
            records.append(record(r,a.arch,a.mode,status,mods)); continue
        records.append(record(r,a.arch,a.mode,"QUEUED",mods))
    queued=[r for r in records if r["status"]=="QUEUED"]
    if queued:
        pydir=LLK/"tests/python_tests"; python=pydir/".venv/bin/python"; log=run/f"{a.mode}.log"
        mods=sorted({m for r in queued for m in r["reason"].split(",") if m and " " not in m})
        if a.mode=="silicon":
            correctness=sorted({m for rec in queued for m in next(x for x in selected if x["id"]==rec["id"])["functional_modules"].split(",") if m and " " not in m})
            mods=sorted(set(mods)|set(correctness))
        env=os.environ.copy(); env.update({"TT_METAL_HOME":str(ROOT),"SHORT_ARCH":a.arch,
            "SIM_ARCH":{"bh":"blackhole","wh":"wormhole","qsr":"quasar"}[a.arch]})
        if not python.is_file():
            rc=None; why="missing tt-llk .venv"
        elif a.mode=="compile":
            cmd=[str(python),"-m","pytest","-o","addopts=",*mods,"--compile-producer","-q"]; why="compile gate"
            with log.open("w") as f: rc=subprocess.run(cmd,cwd=pydir,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        elif a.mode=="craq":
            runner=pathlib.Path(os.environ.get("CRAQ_SIM_ROOT","/localdev/nkapre/craq-sim"))/"scripts/perf/llk-sim-perf.sh"
            cmd=[str(runner),"--sample","1","--run-root",str(run/"craq")]+sum((["--module",m] for m in mods),[])
            env["SIMULATOR"]=str(a.simulator); why="CRAQ modeled-cycle/functional gate"
            with log.open("w") as f: rc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        else:
            cmd=[str(python),"-m","pytest","-o","addopts=",*mods,"-q"]; why="serialized correctness-plus-silicon gate"
            lock=a.hardware_lock
            import fcntl
            with lock.open("w") as lk, log.open("w") as f:
                fcntl.flock(lk,fcntl.LOCK_EX); rc=subprocess.run(cmd,cwd=pydir,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        for rec in queued:
            rec["status"]="SKIP_MISSING_ENV" if rc is None else ("PASS" if rc==0 else "FAIL")
            rec["reason"]=why; rec["artifact"]=str(log) if log.exists() else ""
        if a.mode=="craq" and rc==0:
            metric=run/"craq/llk_sim.tsv"; measured=[]
            if metric.is_file():
                with metric.open() as f: measured=list(csv.DictReader(f,delimiter="\t"))
            for rec in queued:
                names=[pathlib.Path(x).name for x in next(x for x in selected if x["id"]==rec["id"])["functional_modules"].split(",")]
                vals=[float(x["simulated_cycles"]) for x in measured if any(x.get("nodeid","").startswith(n) for n in names) and x.get("simulated_cycles")]
                if vals:
                    rec.update(metric="simulated_cycles",scope="craq_program",selector="default",cycles=max(vals))
                else:
                    rec["status"]="FAIL"; rec["reason"]="mapped CRAQ row produced no modeled-cycle metric"
        if a.mode=="silicon" and rc==0:
            measured=[]
            if a.measurements and a.measurements.is_file():
                with a.measurements.open() as f: measured=list(csv.DictReader(f,delimiter="\t"))
            for rec in queued:
                hits=[x for x in measured if x.get("id")==rec["id"] and x.get("arch")==a.arch and x.get("cycles")]
                if hits:
                    x=hits[-1]; rec.update(metric=x["metric"],scope=x["scope"],selector=x["selector"],cycles=float(x["cycles"]))
                else:
                    rec["status"]="FAIL"; rec["reason"]="mapped silicon row produced no scoped device-cycle metric"
    failed=False
    if a.require_executed_mapped:
        mapped=[r for r in records if next(x for x in selected if x["id"]==r["id"])["functional_modules"]]
        if not mapped or any(r["status"] != "PASS" for r in mapped):
            failed=True
            prov["executed_mapped_gate"]="FAIL"
        else:
            prov["executed_mapped_gate"]="PASS"
    if a.baseline:
        comparisons=compare_baseline(records,a.baseline,a.max_regression_pct)
        (run/"comparison.json").write_text(json.dumps(comparisons,indent=2)+"\n")
        prov["baseline"]=str(a.baseline)
        failed=failed or any(x["status"]=="REGRESSION" for x in comparisons)
    emit_summary(run,records,prov); print(run)
    return int(failed)
if __name__=="__main__": raise SystemExit(main())
