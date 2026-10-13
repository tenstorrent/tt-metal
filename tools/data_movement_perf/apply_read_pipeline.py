"""Set the model's aggregate DRAM efficiency and recalculate Excel formulas locally."""

from pathlib import Path
import functools, json, shutil, zipfile, xml.etree.ElementTree as ET
import numpy as np
import openpyxl
import formulas
from openpyxl.styles import Font, PatternFill, Alignment

root = Path(__file__).resolve().parents[2]
p = root / "Data movement perf calculator (1).xlsx"
backup = p.with_name(p.stem + ".before-read-pipeline.xlsx")
if not backup.exists():
    shutil.copy2(p, backup)
w = openpyxl.load_workbook(p)
old = openpyxl.load_workbook(p, data_only=True)
# Preserve the measured curves and historical comparison sheets.
if "Read pipeline" in w:
    del w["Read pipeline"]
s = w.create_sheet("Read pipeline")
rows = [
    ("DRAM to L1 read pipeline — editable parameters", "Value", "Units / provenance"),
    (
        "Latency source: 0 measured composite, 1 component model",
        0,
        "Default measured; component mode requires populated assumptions",
    ),
    (
        "Measured all-bank first-response proxy",
        "=Hardware!B18",
        "us; composite historical latency, not isolated DRAM service",
    ),
    ("Clock", "=Hardware!B4", "GHz"),
    ("Total NIU boundary cycles for round trip", 20, "Approximate documented 4 x 5 cycles"),
    ("Cycles per router-to-router hop", 9, "Approximate documented value"),
    ("Mean request + response hops", 0, "UNKNOWN: enter actual route-weighted total before enabling component mode"),
    ("DRAM first-response cycles", 0, "UNKNOWN: independently measured/controller dependent"),
    ("Initial command issue cycles", 0, "UNKNOWN"),
    ("Completion observation cycles", 0, "UNKNOWN; barrier polls locally"),
    ("Component startup latency", "=(B5+B6*B7+B8+B9+B10)/(B4*1000)", "us; excludes payload serialization"),
    ("Selected startup latency", "=IF(B2=0,B3,B11)", "us"),
    (
        "Sustained issue cycles/request; 0 disables limit",
        0,
        "UNKNOWN: do not infer from transaction count before barrier",
    ),
    (
        "Effective outstanding requests/core; 0 disables limit",
        0,
        "UNKNOWN hardware/software window; distinct from requests before barrier",
    ),
    ("NoC flit bytes", 64, "BH 512-bit flit"),
    ("Response header flits per packet", 1, "One header plus payload"),
    ("Request header flits per packet", 1, "Separate outbound request stream; no extra permission handshake"),
    ("Available runtime cores", 110, "Observed 11 x 10 logical grid on this device"),
    ("Aggregate efficiency", "=Hardware!B15", "Retained end-to-end cap assumption, not a DRAM hardware constant"),
    (
        "Default policy",
        "Measured latency + empirical sustained-rate envelope",
        "No hop-dependent claim until topology and component latency are supplied",
    ),
    (
        "Prediction",
        "startup + MAX(service limits) + completion/launch extras",
        "Measured per-core service has measured latency removed before re-adding startup",
    ),
    (
        "Shared routes",
        "Not explicitly enumerated",
        "Per-core NoC injection/receive lower bounds only; aggregate efficiency retains shared-path effects",
    ),
    (
        "Latency caution",
        "Historical measured latency is a proxy",
        "Do not add measured composite latency to the individual routing/DRAM terms",
    ),
    (
        "Routing source",
        "https://github.com/tenstorrent/tt-isa-documentation/blob/main/BlackholeA0/NoC/README.md",
        "Approximate boundary/hop timing and flit size",
    ),
    (
        "Scope",
        "DRAM to resident L1 shards, NoC0, one final barrier/core",
        "No DRAM writes, tilize, staging buffers or compute",
    ),
]
for row in rows:
    s.append(row)
s.column_dimensions["A"].width = 60
s.column_dimensions["B"].width = 48
s.column_dimensions["C"].width = 95
for row in s:
    for c in row:
        c.alignment = Alignment(wrap_text=True, vertical="top")
    s.row_dimensions[row[0].row].height = 34
for c in s[1]:
    c.font = Font(bold=True, color="FFFFFF")
    c.fill = PatternFill("solid", fgColor="17365D")
s.freeze_panes = "B2"
w["Hardware"]["B8"] = 110
w["Hardware"]["D8"] = "Runtime device grid 11 x 10 from existing benchmark; logical cores available on this device."
w["Model"]["B24"] = "Pipeline model; measured latency proxy; unknown issue/window limits disabled"
w["Model"]["A20"] = "Legacy latency rule (superseded by Read pipeline)"
search = w["Search"]
heads = [
    "Empirical streaming us",
    "Response NoC service us",
    "Request NoC service us",
    "Issue service us",
    "Outstanding-window service us",
    "Selected startup us",
]
for c, h in enumerate(heads, 27):
    search.cell(3, c, h)
    search.column_dimensions[openpyxl.utils.get_column_letter(c)].width = 24
for r in range(4, 1084):
    search[f"AA{r}"] = f"=MAX(0,M{r}-N{r})"
    search[f"AB{r}"] = f"=F{r}*(ROUNDUP(B{r}/'Read pipeline'!$B$15,0)+'Read pipeline'!$B$16)/(Hardware!$B$4*1000)"
    search[f"AC{r}"] = f"=F{r}*'Read pipeline'!$B$17/(Hardware!$B$4*1000)"
    search[f"AD{r}"] = f"=F{r}*'Read pipeline'!$B$13/(Hardware!$B$4*1000)"
    search[
        f"AE{r}"
    ] = f"=IF('Read pipeline'!$B$14>0,MAX(0,F{r}/MAX(1,'Read pipeline'!$B$14)-1)*(AF{r}+(ROUNDUP(B{r}/'Read pipeline'!$B$15,0)+'Read pipeline'!$B$16)/(Hardware!$B$4*1000)),0)"
    search[f"AF{r}"] = f"=IF('Read pipeline'!$B$2=0,N{r},'Read pipeline'!$B$11)"
    search[f"O{r}"] = f"=AF{r}+MAX(AA{r},AB{r},AC{r},AD{r},AE{r})"
    search[f"T{r}"] = f"=AF{r}+MAX(AA{r},AB{r},AC{r},AD{r},AE{r},R{r},S{r})+P{r}+Model!$B$19"
w["Guide"].append(
    [
        "Read pipeline",
        "Startup + overlapping service bottlenecks; see Read pipeline for editable parameters and uncertainties. Historical comparison sheets are snapshots; Pipeline comparison is current.",
    ]
)
active = {"Hardware", "Model", "Search", "Core summary", "Shard plan", "Read pipeline", "Pipeline comparison"}
names = {s.upper(): s for s in w.sheetnames}


@functools.lru_cache(None)
def value(sn, addr):
    c = w[sn][addr]
    if c.data_type != "f":
        return c.value if c.value is not None else ""
    if sn not in active:
        v = old[sn][addr].value
        return v if v is not None else ""
    fn = formulas.Parser().ast(c.value)[1].compile()
    args = []
    for ref in fn.inputs:
        if "!" in ref:
            sheet, a = ref.rsplit("!", 1)
            sheet = names[sheet.strip("'")]
        else:
            sheet, a = sn, ref
        cells = w[sheet][a.replace("$", "")]
        if isinstance(cells, tuple):
            arg = np.array([[value(sheet, x.coordinate) for x in row] for row in cells], dtype=object)
        else:
            arg = value(sheet, cells.coordinate)
        args.append(arg)
    v = np.asarray(fn(*args)).item()
    if isinstance(v, np.generic):
        v = v.item()
    if not isinstance(v, (str, float, int, bool)):
        raise RuntimeError((sn, addr, v))
    if isinstance(v, str) and v.startswith(("#REF!", "#VALUE!", "#DIV/0!", "#NUM!")):
        raise RuntimeError((sn, addr, v))
    return v


base = root / "ttnn/ttnn/operations/examples/dram_l1_transaction_size"
measured = json.loads((base / "core_sweep_confirmation.json").read_text())
if "Pipeline comparison" in w:
    del w["Pipeline comparison"]
s = w.create_sheet("Pipeline comparison")
s.append(
    [
        "Cores",
        "Measured us (existing 30 trials)",
        "Previous 70% estimate us",
        "Pipeline estimate us",
        "Error %",
        "Measured GB/s",
    ]
)
results = []
for ri, r in enumerate(measured["results"], 2):
    sr = 4 + (r["cores"] - 1) * 9 + 5
    pred = value("Search", f"T{sr}")
    s.append(
        [
            r["cores"],
            r["median_us"],
            old["Search"][f"T{sr}"].value,
            f"=Search!T{sr}",
            f"=(D{ri}/B{ri}-1)*100",
            r["gbps"],
        ]
    )
    results.append(
        dict(
            cores=r["cores"],
            measured_us=r["median_us"],
            predicted_us=pred,
            error_percent=(pred / r["median_us"] - 1) * 100,
        )
    )
for col in "ABCDEF":
    s.column_dimensions[col].width = 28
names = {s.upper(): s for s in w.sheetnames}
# Evaluate real formulas, including updated rank/recommendation and shard assignments.
caches = {}
for sn in w.sheetnames:
    for row in w[sn]:
        for c in row:
            if c.data_type == "f":
                caches[(sn, c.coordinate)] = value(sn, c.coordinate)
# Preserve all prior cached formula results and write new values for the active model.
w.save(p)
ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
with zipfile.ZipFile(p) as z:
    payload = {n: z.read(n) for n in z.namelist()}
for i, ws in enumerate(w, 1):
    name = f"xl/worksheets/sheet{i}.xml"
    tree = ET.fromstring(payload[name])
    changed = False
    for c in tree.findall(".//s:sheetData/s:row/s:c", ns):
        key = (ws.title, c.attrib["r"])
        if key not in caches:
            continue
        val = caches[key]
        v = c.find("s:v", ns)
        if v is None:
            v = ET.SubElement(c, "{" + ns["s"] + "}v")
        if isinstance(val, str):
            c.set("t", "str")
            v.text = val
        else:
            c.attrib.pop("t", None)
            v.text = str(int(val) if isinstance(val, bool) else val)
        changed = True
    if changed:
        payload[name] = ET.tostring(tree, encoding="utf-8")
with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
    for name, data in payload.items():
        z.writestr(name, data)
check = openpyxl.load_workbook(p, data_only=True)
assert check["Measured comparison"]["E15"].value == 27.144
out = dict(
    recommended_cores=value("Model", "B29"),
    transaction_bytes=value("Model", "B30"),
    predicted_us=value("Model", "B33"),
    results=results,
)
(root / "tools/data_movement_perf/pipeline_comparison.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
