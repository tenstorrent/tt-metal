"""Expose actual estimator outputs and coverage limitations beside the measured benchmark."""

from pathlib import Path
import json, shutil, zipfile, xml.etree.ElementTree as ET
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

root = Path(__file__).resolve().parents[2]
p = root / "Data movement perf calculator (1).xlsx"
report = json.loads((root / "tools/data_movement_perf/noc_estimator_comparison.json").read_text())
backup = p.with_name(p.stem + ".before-noc-estimator.xlsx")
if not backup.exists():
    shutil.copy2(p, backup)
w = openpyxl.load_workbook(p)
ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
caches = {}
with zipfile.ZipFile(p) as z:
    for i, ws in enumerate(w, 1):
        for c in ET.fromstring(z.read(f"xl/worksheets/sheet{i}.xml")).findall(".//s:sheetData/s:row/s:c", ns):
            if c.find("s:f", ns) is not None:
                v = c.find("s:v", ns)
                caches[(ws.title, c.attrib["r"])] = (c.attrib.get("t"), v.text if v is not None else None)
if "NoC estimator check" in w:
    del w["NoC estimator check"]
s = w.create_sheet("NoC estimator check")
s.sheet_view.showGridLines = False
s.freeze_panes = "C10"
s.merge_cells("A1:G1")
s["A1"] = "Built-in NoC estimator — raw outputs and coverage limits"
s["A1"].fill = PatternFill("solid", fgColor="17365D")
s["A1"].font = Font(size=16, bold=True, color="FFFFFF")
notes = [
    "Queries: BLACKHOLE, DRAM_INTERLEAVED, UNICAST, NoC0, 2048 B, 1350 MHz; one barrier after all requests for the largest shard.",
    "Comparison workload: BF16 [1,1,4096,1024], 8 MiB; hardware measurements are 30-trial medians with preferred read NoC0.",
    "DRAM calibration supports 1,4,16,64,256 transactions/barrier; num_subordinates is stored only as 0, including the full-grid pattern.",
    "WARNING: the estimator cannot distinguish arbitrary DRAM reader counts. ALL_FROM_ALL represents the full-grid benchmark pattern.",
    "WARNING: transactions/barrier >256 clamp to 256 with no extrapolation. 7- and 11-core outputs below are NOT valid predictions for their full shards.",
    "ONE_FROM_ALL represents one reader cycling across banks; it is not a prediction of simultaneous multicore performance.",
    f"Bundled YAML SHA256: {report['data_sha256']}",
]
for r, text in enumerate(notes, 2):
    s.merge_cells(start_row=r, start_column=1, end_row=r, end_column=7)
    s.cell(r, 1, text).alignment = Alignment(wrap_text=True)
    s.row_dimensions[r].height = 35
headers = [
    "Requested cores",
    "Transactions / largest core",
    "Excel us (unchanged)",
    "Single-reader raw us",
    "Full-grid raw us",
    "Measured us",
    "Coverage / applicability",
]
for c, title in enumerate(headers, 1):
    cell = s.cell(9, c, title)
    cell.fill = PatternFill("solid", fgColor="17365D")
    cell.font = Font(bold=True, color="FFFFFF")
    cell.alignment = Alignment(wrap_text=True)
s.row_dimensions[9].height = 42
for i, r in enumerate(report["rows"], 10):
    status = (
        "INVALID: count clamped to 256; reader count ignored"
        if r["transaction_count_clamped"]
        else "Transaction interpolation; reader count ignored; full-grid template"
    )
    vals = [
        r["cores"],
        r["transactions_per_largest_shard"],
        r["excel_us"],
        r["single_reader"]["result"]["latency_us"],
        r["all_readers"]["result"]["latency_us"],
        r["measured_us"],
        status,
    ]
    for j, v in enumerate(vals, 1):
        s.cell(i, j, v)
    for j in range(3, 7):
        s.cell(i, j).number_format = "0.000"
    s.cell(i, 7).alignment = Alignment(wrap_text=True)
    s.row_dimensions[i].height = 40
extra = [
    "Fixed 64 transactions, ALL_FROM_ALL: requested core counts 1,7,11,44,110 ALL return 35.791852 us. This confirms no usable reader-count axis.",
    "ALL_FROM_ALL requests for 256,373,586,18752 transactions/barrier ALL return 144.959259 us. The latter three are silently clamped.",
    "CLI achieved_GBps uses transactions * transaction_bytes / latency. It does not multiply by reader count; do not interpret it as aggregate chip bandwidth.",
    "The source DRAM benchmark cycles through bank pages into reused L1 scratch. Our kernel reads distinct rows into a resident L1 shard; the access patterns differ.",
    "At 110 cores, 21.353 us estimated vs 22.181 us measured (about 3.7% low). This single agreement does not validate predictions for smaller core counts.",
    "Use the estimator for covered transfer groups, with explicit coverage checks. Extend its DRAM calibration with active core count and larger one-barrier transaction counts before using it to optimize this workload.",
    "Reproduce: tools/data_movement_perf/query_noc_estimator.py; full CLI commands and raw JSON outputs in noc_estimator_comparison.json beside that script.",
    "Existing Excel predictions and measurements are preserved. No model calibration was replaced by these raw outputs.",
]
for i, text in enumerate(extra, 17):
    s.merge_cells(start_row=i, start_column=1, end_row=i, end_column=7)
    s.cell(i, 1, text).alignment = Alignment(wrap_text=True)
    s.row_dimensions[i].height = 40
for c in "ABCDEF":
    s.column_dimensions[c].width = 24
s.column_dimensions["G"].width = 62
# Explicit gating separates audit-only raw outputs from usable estimator results.
from guarded_noc_estimator import GuardedEstimator

estimator = GuardedEstimator()
for c, label in enumerate(["Estimator eligible", "Usable estimator us", "Fallback source", "Fallback us"], 8):
    s.cell(9, c, label).alignment = Alignment(wrap_text=True)
    s.cell(9, c).font = Font(bold=True, color="FFFFFF")
    s.cell(9, c).fill = PatternFill("solid", fgColor="17365D")
    s.column_dimensions[openpyxl.utils.get_column_letter(c)].width = 26
for i, r in enumerate(report["rows"], 10):
    checked = estimator.estimate(r["cores"], r["transactions_per_largest_shard"], 2048)
    eligible = checked["source"] == "bounded_estimator"
    s.cell(i, 8, "YES" if eligible else "NO")
    s.cell(i, 9, checked["latency_us"])
    s.cell(i, 10, "Not required" if eligible else "Analytical model; optimistic")
    s.cell(i, 11, None if eligible else r["excel_us"])
    s.cell(i, 9).number_format = s.cell(i, 11).number_format = "0.000"
s.merge_cells("A27:K27")
s[
    "A27"
] = "Policy: use only bounded, non-relaxed estimator queries; reject missing calibration axes/corners. Blank usable-estimator cells mean no estimate, never zero duration."
s["A27"].alignment = Alignment(wrap_text=True)
s.row_dimensions[27].height = 38
s.merge_cells("A28:K28")
s[
    "A28"
] = "No estimator calls are embedded in Excel. Re-run guarded_noc_estimator.py for new inputs; these are audited snapshots, not automatic estimator recalculation."
s["A28"].alignment = Alignment(wrap_text=True)
s.row_dimensions[28].height = 38
w.save(p)
with zipfile.ZipFile(p) as z:
    data = {n: z.read(n) for n in z.namelist()}
for i, ws in enumerate(w, 1):
    path = f"xl/worksheets/sheet{i}.xml"
    tree = ET.fromstring(data[path])
    changed = False
    for c in tree.findall(".//s:sheetData/s:row/s:c", ns):
        key = (ws.title, c.attrib["r"])
        if key not in caches:
            continue
        typ, value = caches[key]
        v = c.find("s:v", ns)
        if v is None:
            v = ET.SubElement(c, "{" + ns["s"] + "}v")
        v.text = value
        if typ:
            c.set("t", typ)
        else:
            c.attrib.pop("t", None)
        changed = True
    if changed:
        data[path] = ET.tostring(tree, encoding="utf-8")
with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
    for name, contents in data.items():
        z.writestr(name, contents)
check = openpyxl.load_workbook(p, data_only=True)
old = openpyxl.load_workbook(backup, data_only=True)
for sn in old.sheetnames:
    for row in old[sn]:
        for c in row:
            assert c.value == check[sn][c.coordinate].value, (sn, c.coordinate)
print("Added NoC estimator check; all previous model/measurement values preserved.")
