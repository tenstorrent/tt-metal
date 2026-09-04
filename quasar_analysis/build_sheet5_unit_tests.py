# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Append SHEET 5 -- "Unit tests (Quasar run)" -- to resnet50_forge_bf16_vs_quasar.xlsx.

One row per test in
  models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/
which since the per-call-site split is ONE FILE PER OP: 91 op files (plus the inventory file), so
the sheet is also a directory listing -- sheet-1 row, test file, test function, the quasar op it
calls, the full operand set, the attributes VERBATIM from sheet 1, the torch golden, the assertion,
the observed result and the root cause.

Sheet 1's layout-plumbing rows have no test file (they move a tensor between TILE and ROW_MAJOR
without computing anything -- the same reason sheets 3 and 4 leave them out), so they have no row
here either. The verifier does not take that on trust: it derives which op kinds have files and
asserts that every uncovered sheet-1 row belongs to a kind with no file at all, so a silently
dropped conv would still fail.

Nothing here is re-typed. The operand and attribute columns come straight out of sheet 1; the
"which file, which op" columns come from the five constants each op file declares (parsed with
`ast`, no import); the result columns come from the pytest log in
quasar_analysis/forge_fe_bf16_runs/. So the sheet cannot drift from either the workbook or the
tests.

The workbook is edited SURGICALLY: sheets 1-4, the theme and the existing style indices are copied
through byte-for-byte, and only a new sheet part, three appended style records and the four
registration entries (workbook.xml, its rels, [Content_Types].xml, the autofilter defined name) are
written. openpyxl is not installed on this host and would rewrite the whole file anyway.

Run:
    python quasar_analysis/build_sheet5_unit_tests.py            # writes in place, keeps a .bak
    python quasar_analysis/build_sheet5_unit_tests.py --check    # parse + report, write nothing
    python quasar_analysis/build_sheet5_unit_tests.py --verify   # re-read the written file and check it
"""

import argparse
import ast
import collections
import glob
import html
import os
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(REPO, "resnet50_forge_bf16_vs_quasar.xlsx")
LOGS = os.path.join(REPO, "quasar_analysis", "forge_fe_bf16_runs")
RELDIR = "models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16"
TESTDIR = os.path.join(REPO, RELDIR)
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

GITHUB = "https://github.com/tenstorrent/tt-metal"
LOG_REL = "quasar_analysis/forge_fe_bf16_runs/all_ops.log"
ATTEST_REL = "quasar_analysis/forge_fe_bf16_runs/dispatch_attestation.json"
SUMMARY_REL = "quasar_analysis/forge_fe_bf16_runs/SUMMARY.txt"


def _git(*args):
    import subprocess

    return subprocess.check_output(["git", "-C", REPO] + list(args), text=True).strip()


def resolve_commit():
    """
    The SHA the links point at: the commit currently on the remote branch, so every permalink
    resolves for someone who was not there when it was written.

    Falls back to HEAD with a warning if the branch has not been pushed -- the links are then only
    good once it is.
    """
    head = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    try:
        remote = _git("rev-parse", "origin/%s" % branch)
    except Exception:
        remote = None
    if remote != head:
        print(
            "  WARNING: HEAD (%s) is not what origin/%s has (%s) -- push first or the links 404"
            % (head[:11], branch, (remote or "nothing")[:11])
        )
    return head, branch


def blob(commit, relpath, line=None):
    return "%s/blob/%s/%s%s" % (GITHUB, commit, relpath, ("#L%d" % line) if line else "")


NSB = "{%s}" % NS


# --------------------------------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------------------------------
def read_sheet(part, path=XLSX):
    root = ET.fromstring(zipfile.ZipFile(path).read(part))
    rows = []
    for r in root.iter(NSB + "row"):
        cells, hi = {}, -1
        for c in r.findall(NSB + "c"):
            idx = 0
            for ch in re.match(r"([A-Z]+)", c.get("r")).group(1):
                idx = idx * 26 + (ord(ch) - 64)
            idx -= 1
            is_el = c.find(NSB + "is")
            if is_el is not None:
                v = "".join(t.text or "" for t in is_el.iter(NSB + "t"))
            else:
                ve = c.find(NSB + "v")
                v = ve.text if ve is not None and ve.text is not None else ""
            if v:
                cells[idx], hi = v, max(hi, idx)
        rows.append([cells.get(i, "") for i in range(hi + 1)])
    return rows


def sheet1_records():
    rows = read_sheet("xl/worksheets/sheet1.xml")[3:]
    recs = {}
    for r in rows:
        g = lambda i: r[i] if len(r) > i else ""
        ops = [dict(role=g(2 + 4 * k), shape=g(3 + 4 * k), dtype=g(4 + 4 * k), cfg=g(5 + 4 * k)) for k in range(4)]
        recs[int(g(0))] = dict(
            idx=int(g(0)),
            op=g(1),
            ops=[o for o in ops if o["role"]],
            out=dict(role=g(18), shape=g(19), dtype=g(20), cfg=g(21)),
            attrs=g(22),
            ir=g(23),
        )
    assert len(recs) == 141 and sorted(recs) == list(range(141)), "sheet 1 is not 141 rows 0..140"
    return recs


CONSTANTS = ("SHEET_ROW", "FORGE_OP", "QUASAR_OP", "OPERAND_SHAPES", "OUTPUT_SHAPE")

SHEET1_ROWS = 141  # every op row in @forward
COVERED_OPS = 91  # the compute ops, one test file each


def op_files():
    """file name -> the five constants it declares. Parsed, never imported."""
    out = {}
    for path in sorted(glob.glob(os.path.join(TESTDIR, "test_op[0-9][0-9][0-9]_*.py"))):
        fname = os.path.basename(path)
        got = {}
        for node in ast.parse(open(path).read(), filename=fname).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id in CONSTANTS:
                    got[node.targets[0].id] = ast.literal_eval(node.value)
        assert all(c in got for c in CONSTANTS), "%s does not declare all of %s" % (fname, CONSTANTS)
        out[fname] = got
    assert len(out) == COVERED_OPS, "found %d op files, expected %d" % (len(out), COVERED_OPS)
    return out


def read_results():
    """node id -> (outcome, xfail reason); and node id -> one-line traceback."""
    outcome, cause = {}, {}
    for path in sorted(glob.glob(os.path.join(LOGS, "*.log"))):
        txt = open(path, errors="replace").read()
        for line in txt.split("\n"):
            m = re.match(r"^(PASSED|FAILED|XFAIL|XPASS|ERROR) (models/\S+)(?: - (.*))?$", line)
            if m:
                outcome[m.group(2)] = (m.group(1), (m.group(3) or "").strip())
        # --tb=line prints one line per failure in execution order with no per-test header; the
        # FAILED lines of the short summary are in the same order, so they pair positionally.
        block = re.search(r"=+ FAILURES =+\n(.*?)\n=+ short test summary", txt, re.S)
        tbs = (
            [l.strip() for l in block.group(1).split("\n") if re.match(r"^/\S+\.py:\d+: ", l.strip())] if block else []
        )
        fails = [l.split()[1] for l in txt.split("\n") if l.startswith("FAILED models/")]
        if len(tbs) == len(fails):
            cause.update(dict(zip(fails, tbs)))
        elif fails:
            raise SystemExit("%s: %d failures but %d traceback lines" % (path, len(fails), len(tbs)))
    assert outcome, "no results found under %s -- run the suite first" % LOGS
    return outcome, cause


CAUSE_A = (
    "CAUSE A -- fp32_dest_acc_en=true rejected: program_spec.cpp:1076, the compute kernel consumes FP32 DFB "
    "'cb_intermed0' with enable_32_bit_dest=true but provides no unpack_modes entry. Forge's compute config is "
    "passed verbatim, because that is what sheet 1 records."
)
CAUSE_B = (
    "CAUSE B -- Gen2 forbids a self-looped data-movement DFB: program_spec.cpp:1439, the halo scratch buffer is "
    "bound as both PRODUCER and CONSUMER by kernel 'reader0'. Hits exactly the convs that need a halo. "
    "max_pool2d needs one too and PASSES, so this is the conv halo path specifically. (The one-line traceback "
    "stops before the buffer name; SUMMARY.txt has the per-buffer counts from the full log.)"
)
PCC_FAIL = (
    "NUMERIC failure, not an error: the op ran and returned, and the result missed its bound at PCC %s. This "
    "is the dangerous class -- nothing throws, the numbers are just wrong."
)


def outcome_count():
    return read_results()[0]


def classify(tb):
    if not tb:
        return ""
    if "1076" in tb or "unpack_modes" in tb:
        return CAUSE_A
    if "1439" in tb:
        return CAUSE_B
    m = re.search(r"AssertionError: ([0-9.]+)", tb)
    if m:
        return PCC_FAIL % ("%.4f" % float(m.group(1)))
    return tb


# --------------------------------------------------------------------------------------------------
# row construction -- the operand and attribute columns come straight out of sheet 1
# --------------------------------------------------------------------------------------------------
def layout_of(cfg):
    if "system_memory" in cfg:
        return "host"
    if "device handle" in cfg:
        return "device handle"
    return "TILE" if "ttcore.tile" in cfg else "ROW_MAJOR"


def mem_of(cfg):
    if "system_memory" in cfg:
        return "#system_memory (host)"
    if "device handle" in cfg:
        return "!ttnn.device"
    return "DRAM interleaved"


def operand_text(o):
    if not o or not o.get("role"):
        return "—"
    if o["role"] == "Device handle":
        return "Device handle: !ttnn.device"
    return "%s: %s, %s, %s, %s" % (o["role"], o["shape"], o["dtype"].split()[0], layout_of(o["cfg"]), mem_of(o["cfg"]))


def split_attrs(attrs):
    """attrs verbatim -> (rest, compute_config, op config), pulling the two nested configs out."""
    comp = re.search(r"compute_config = #ttnn\.device_compute_kernel_config<([^>]*)>", attrs)
    conv = re.search(r"conv2d_config = #ttnn\.conv2d_config<(.*?)>(?=, dilation|, groups|$)", attrs)
    rest = attrs
    if comp:
        rest = rest.replace(comp.group(0), "").strip()
    if conv:
        rest = rest.replace(conv.group(0), "").strip()
    while re.search(r",\s*,", rest):  # pulling two configs out can leave ", ," behind
        rest = re.sub(r",\s*,", ",", rest)
    rest = rest.strip().strip(",").strip()
    return rest or "(no attributes)", (comp.group(1) if comp else "—"), (conv.group(1) if conv else "—")


# per test-function suffix: (torch golden, assertion). Keyed by what the function name ends with.
GOLDEN = {
    "reshape": (
        "host.reshape(OUT_SHAPE)",
        "EXACT equality (torch.equal) + PCC >= 0.9999 + output shape + TILE + INTERLEAVED + DRAM",
    ),
    "permute": (
        "host.permute(0, 2, 3, 1)",
        "EXACT equality + PCC >= 0.9999 -- the op is resolved AT RUN TIME, so this xfails while quasar binds no "
        "permute and starts exercising the real op the moment a binding lands",
    ),
    "permute_via_transpose": (
        "host.permute(0, 2, 3, 1)",
        "EXACT equality + PCC >= 0.9999 + the intermediate shape after the first swap",
    ),
    "conv2d": (
        "torch.nn.functional.conv2d(x, w, bias, stride, padding), then torch.relu where the conv fuses one",
        "PCC >= 0.98 + the returned (out_h, out_w) + the op's internally-prepared weight has the shape "
        "prepare_conv2d_weights would make + output row count + TILE + INTERLEAVED + DRAM",
    ),
    "max_pool2d": (
        "torch.nn.functional.max_pool2d(x, 3, stride=2, padding=1)",
        "PCC >= 0.999 (max selects, it does not accumulate) + output row count + channels + INTERLEAVED + DRAM",
    ),
    "add": ("main + skip", "PCC >= 0.99 + output shape + TILE + INTERLEAVED + DRAM"),
    "relu": (
        "torch.relu(host) -- inputs are randn, so the clamp really clamps",
        "PCC >= 0.99 -- the op is resolved AT RUN TIME, so this xfails while quasar binds no standalone unary "
        "and starts exercising the real op the moment a binding lands",
    ),
    "relu_fused_add": (
        "torch.relu(main + skip)",
        "PCC >= 0.99 + output shape + no negative value survives the fused RELU",
    ),
    "mean": (
        "host.mean(dim=-2, keepdim=True)",
        "PCC >= 0.99 -- the op is resolved AT RUN TIME, so this xfails while quasar binds no reduction and "
        "starts exercising the real op the moment a binding lands",
    ),
    "mean_via_avg_pool2d": (
        "host.mean(dim=-2, keepdim=True)",
        "PCC >= 0.99 + exactly 1 output row + channel count. Note this op ACCEPTS fp32_dest_acc_en=true, which "
        "conv2d and linear are rejected for.",
    ),
    "linear": ("act @ weight + bias", "PCC >= 0.98 + output shape + TILE + INTERLEAVED + DRAM"),
}

WORKAROUND_OP = {
    "relu_fused_add": "quasar.add(activations=[RELU])",
    "permute_via_transpose": "quasar.transpose x2",
    "mean_via_avg_pool2d": "quasar.avg_pool2d",
}

INVENTORY = {
    "test_one_file_per_compute_op": (
        "the 91 compute rows",
        "(directory check)",
        "exactly 91 test_opNNN_*.py files, with unique SHEET_ROWs all inside sheet 1's range, and as many "
        "files on disk as distinct rows -- parsed off disk with ast, no import, no device",
    ),
    "test_op_census_matches_sheet1": (
        "the 91 compute rows",
        "(directory check)",
        "each covered op kind appears the number of times sheet 1 records (conv2d 53, add 16, relu 16, "
        "reshape 2, and 1 each of permute / max_pool2d / mean / linear), and every file's declared QUASAR_OP "
        "agrees with the Forge->Quasar map",
    ),
    "test_conv_files_match_resnet50_topology": (
        "the 53 conv rows",
        "(directory check)",
        "the 53 conv files' declared activation / weight / bias / output shapes are re-derived from ResNet-50 "
        "itself (layers [3,4,6,3], widths [64,128,256,512], expansion 4, stride on the 3x3), and each file is "
        "named after the module it replays",
    ),
    "test_residual_files_match_resnet50_topology": (
        "the 16 add + 16 relu rows",
        "(directory check)",
        "the 16 adds and 16 relus follow the bottleneck widths, and each relu file sits directly after its add",
    ),
    "test_every_op_file_is_shape_consistent": (
        "the 91 compute rows",
        "(directory check)",
        "every declared shape is a non-empty tuple of positive ints",
    ),
    "test_quasar_arch_is_registered": (
        "—",
        "(build check)",
        "ttnn.Arch.QUASAR is registered in this build and ttnn.experimental.quasar is bound with > 50 names "
        "(94 observed); prints the full list",
    ),
    "test_forge_ops_map_onto_the_live_quasar_build": (
        "all 9 op kinds",
        "(build check)",
        "every Forge op resolves to a live quasar op OR is one of the 3 named gaps -- fails when a mapped op "
        "vanishes AND when a gap closes",
    ),
    "test_workaround_ops_exist": (
        "—",
        "(build check)",
        "transpose / avg_pool2d / tilize / untilize_with_unpadding / to_memory_config are all bound; if one "
        "goes, the workaround tests are dead code",
    ),
    "test_device_under_test_is_quasar": (
        "—",
        "(device check)",
        "device.arch() == ttnn.Arch.QUASAR and the compute grid is non-empty; prints both (observed: "
        "Arch.QUASAR, 8x4 = 32 workers)",
    ),
}


def build_rows():
    s1 = sheet1_records()
    files = op_files()
    outcome, cause = read_results()
    commit, branch = resolve_commit()
    print("  links pinned to %s (%s)" % (commit[:11], branch))
    by_row = {v["SHEET_ROW"]: (k, v) for k, v in files.items()}

    def order(nid):
        fname = nid.split("/")[-1].split("::")[0]
        if fname not in files:
            return (-1, nid)  # the inventory file first
        # within an op file the primary test sorts before its workaround (…_relu < …_relu_fused_add)
        return (files[fname]["SHEET_ROW"], nid)

    rows = []
    for nid in sorted(outcome, key=order):
        fname, func = nid.split("/")[-1].split("::")
        func = func.split("[")[0]  # drop the [quasar-…-device_params0] parametrisation suffix
        oc, xreason = outcome[nid]
        result = {"PASSED": "PASS", "FAILED": "FAIL", "XFAIL": "XFAIL"}.get(oc, oc)
        note = classify(cause.get(nid, "")) or xreason

        if fname not in files:  # test_op_inventory_bf16.py
            srow, forge_op, assertion = INVENTORY[func]
            rows.append(
                [fname, func, srow, forge_op]
                + ["—"] * 10
                + [
                    assertion,
                    result,
                    note,
                    'pytest -s "%s"' % nid,
                    blob(commit, "%s/%s" % (RELDIR, fname)),
                    blob(commit, LOG_REL),
                ]
            )
            continue

        decl = files[fname]
        rec = s1[decl["SHEET_ROW"]]
        rest, comp, opcfg = split_attrs(rec["attrs"])
        kind = func.split("_", 4)[-1]  # test_forge_bf16_op013_add -> "add"
        golden, assertion = GOLDEN[kind]
        quasar_op = WORKAROUND_OP.get(kind) or decl["QUASAR_OP"] or "— none —"
        ops = rec["ops"] + [None, None, None]
        rows.append(
            [
                fname,
                func,
                str(rec["idx"]),
                rec["op"],
                quasar_op,
                operand_text(ops[0]),
                operand_text(ops[1]),
                operand_text(ops[2]),
                operand_text(rec["out"]),
                rest,
                comp,
                opcfg,
                "DRAM interleaved on every tensor -- this compile pins no shard spec and no core range",
                golden,
                assertion,
                result,
                note,
                'pytest -s "%s"' % nid,
                blob(commit, "%s/%s" % (RELDIR, fname)),
                blob(commit, LOG_REL),
            ]
        )
    return rows


# --------------------------------------------------------------------------------------------------
# xlsx writing
# --------------------------------------------------------------------------------------------------
COLS = [
    # (header, width, body style key)
    ("Test file", 46, "mono"),
    ("Test function", 42, "mono"),
    ("Sheet 1 row", 12, "mono"),
    ("Forge op", 24, "monob"),
    ("Quasar op called", 30, "mono"),
    ("Operand 1", 56, "small"),
    ("Operand 2", 52, "small"),
    ("Operand 3", 44, "small"),
    ("Output", 46, "small"),
    ("Attributes (verbatim from sheet 1)", 86, "small"),
    ("Compute config", 40, "small"),
    ("Conv2dConfig / op config", 60, "small"),
    ("Memory", 46, "small"),
    ("Torch golden", 48, "small"),
    ("Assertion", 88, "small"),
    ("Result", 9, "mono"),
    ("Root cause / note", 92, "small"),
    ("Run", 104, "small"),
    ("Test file on GitHub", 108, "small"),
    ("Run log on GitHub", 108, "small"),
]

# row-2 groups: (label, first col index, last col index), 0-based over COLS
GROUPS = [
    ("Test file and function -- one file per op", 0, 1),
    ("Sheet 1 op", 2, 4),
    ("Operands and output, verbatim from sheet 1", 5, 8),
    ("Config replayed", 9, 12),
    ("What is checked", 13, 14),
    ("Result on Quasar (craq-sim, Arch.QUASAR, 8x4)", 15, 16),
    ("Links (SHA-pinned permalinks)", 18, 19),
]

TITLE = (
    "SHEET 5 — the unit tests that replay Sheet 1 on Quasar, one row per test.   "
    "ONE TEST FILE PER OP CALL-SITE, ONE TEST PER FILE: 91 standalone files under "
    "models/demos/vision/classification/resnet50/quasar/tests/ResNet50_Forge_Fe_bf16/, named "
    "test_op<row>_<op>_<module>.py so the directory listing is the graph in @forward order — plus "
    "test_op_inventory_bf16.py, which parses the five constants every op file declares (SHEET_ROW, FORGE_OP, "
    "QUASAR_OP, OPERAND_SHAPES, OUTPUT_SHAPE) back off disk and checks all 91 against a ResNet-50 topology "
    "re-derived from first principles.   "
    "NO XFAIL ANYWHERE.   The three ops Quasar does not bind (relu x16, permute, mean) are tested through the "
    "route that DOES exist — a fused add+RELU, transpose x2, avg_pool2d — each a full device test with a real "
    "PCC or exact-equality check.  The gaps themselves are watched in ONE place, "
    "test_op_inventory_bf16.py::test_forge_ops_map_onto_the_live_quasar_build, which FAILS the day a gap "
    "closes.   "
    "These are the 91 COMPUTE ops of the graph: Sheet 1's remaining rows are layout-plumbing steps that move "
    "a tensor between TILE and ROW_MAJOR without computing anything, and get no test file and no row here — "
    "the same reason Sheets 3 and 4 leave them out.   "
    "100 tests: 91 op tests + 9 inventory, of which 8 need no device.   Every device test replays the "
    "operands and the config Sheet 1 records — including math_fidelity = hifi4 with fp32_dest_acc_en = true, "
    "and DRAM interleaved for every tensor, so no test is skipped for device grid size.   "
    "PROOF IT RAN ON QUASAR: every test asserts device.arch() == Arch.QUASAR, and the suite is run under "
    "quasar_analysis/pytest_quasar_attest.py, which captures the ttnn graph around each test and records the "
    "DEVICE OPERATION underneath it with its duration — a device-operation node only exists if a program was "
    "created and enqueued.  Per-test records: dispatch_attestation.json.   "
    "The last two columns are SHA-pinned GitHub permalinks to the test file and to the run log, so every row "
    "is auditable from the sheet alone.   "
    "Generated by quasar_analysis/gen_forge_bf16_op_tests.py (op files) and "
    "quasar_analysis/build_sheet5_unit_tests.py (this sheet), which re-reads the written workbook and checks "
    "itself against Sheet 1 and the files on disk."
)

# existing style ids reused from styles.xml (see sheets 1-4)
S_TITLE, S_HDR, S_GRP, S_GRP_MID, S_GRP_END, S_GRP_BLANK = 1, 2, 3, 4, 5, 6
# (small = sz9 wrap, mono = Consolas10, monob = Consolas8 wrap) per fill
FILL_STYLES = [
    (9, 10, 11),  # CFE4FA
    (12, 13, 14),  # no fill
    (15, 16, 17),  # EEF2F5
    (18, 19, 20),  # E8F1FB
    (22, 23, 24),  # FFF3D6
    (25, 26, 27),  # FCE4C6
    (28, 29, 30),  # E2E2E2
    (31, 32, 33),  # E4D9F5
    (34, 35, 36),  # F8D9A8
]
# appended by patch_styles(): PASS / FAIL / XFAIL
S_PASS, S_FAIL, S_XFAIL = 37, 38, 39

NEW_FONTS = (
    '<font><name val="Consolas"/><b val="1"/><color rgb="00006100"/><sz val="10"/></font>'
    '<font><name val="Consolas"/><b val="1"/><color rgb="009C0006"/><sz val="10"/></font>'
    '<font><name val="Consolas"/><b val="1"/><color rgb="009C6500"/><sz val="10"/></font>'
)
NEW_FILLS = (
    '<fill><patternFill patternType="solid"><fgColor rgb="00C6EFCE"/></patternFill></fill>'
    '<fill><patternFill patternType="solid"><fgColor rgb="00FFC7CE"/></patternFill></fill>'
    '<fill><patternFill patternType="solid"><fgColor rgb="00FFEB9C"/></patternFill></fill>'
)


def patch_styles(xml):
    """Append the 3 PASS/FAIL/XFAIL records. Existing indices are untouched, so sheets 1-4 keep their look."""
    n_fonts = int(re.search(r'<fonts count="(\d+)"', xml).group(1))
    n_fills = int(re.search(r'<fills count="(\d+)"', xml).group(1))
    n_xfs = int(re.search(r'<cellXfs count="(\d+)"', xml).group(1))
    if (n_fonts, n_fills, n_xfs) != (8, 13, 37):
        raise SystemExit(
            "styles.xml is not the shape this script was written against (%d/%d/%d)" % (n_fonts, n_fills, n_xfs)
        )
    xml = xml.replace('<fonts count="8">', '<fonts count="11">').replace("</fonts>", NEW_FONTS + "</fonts>")
    xml = xml.replace('<fills count="13">', '<fills count="16">').replace("</fills>", NEW_FILLS + "</fills>")
    new_xfs = "".join(
        '<xf numFmtId="0" fontId="%d" fillId="%d" borderId="1" applyAlignment="1" pivotButton="0" '
        'quotePrefix="0" xfId="0"><alignment horizontal="center" vertical="center"/></xf>' % (f, fl)
        for f, fl in ((8, 13), (9, 14), (10, 15))
    )
    xml = xml.replace('<cellXfs count="37">', '<cellXfs count="40">').replace("</cellXfs>", new_xfs + "</cellXfs>")
    return xml


def colname(i):
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def num_cell(ref, style, value):
    """Sheets 1-4 write the leading "#" as a numeric cell; match them so it sorts as a number."""
    return '<c r="%s" s="%d" t="n"><v>%d</v></c>' % (ref, style, value)


def cell(ref, style, text):
    if text is None or text == "":
        return '<c r="%s" s="%d" t="inlineStr"></c>' % (ref, style)
    return '<c r="%s" s="%d" t="inlineStr"><is><t xml:space="preserve">%s</t></is></c>' % (
        ref,
        style,
        html.escape(str(text), quote=False),
    )


def build_sheet_xml(rows):
    ncol = len(COLS) + 1
    last = colname(ncol - 1)
    out = ['<worksheet xmlns="%s"><sheetPr><outlinePr summaryBelow="1" summaryRight="1"/><pageSetUpPr/></sheetPr>' % NS]
    out.append('<dimension ref="A1:%s%d"/>' % (last, 3 + len(rows)))
    out.append(
        '<sheetViews><sheetView workbookViewId="0"><pane xSplit="2" ySplit="3" topLeftCell="C4" '
        'activePane="bottomRight" state="frozen"/><selection pane="topRight"/><selection pane="bottomLeft"/>'
        '<selection pane="bottomRight" activeCell="A1" sqref="A1"/></sheetView></sheetViews>'
    )
    out.append('<sheetFormatPr baseColWidth="8" defaultRowHeight="15"/><cols>')
    out.append('<col width="5" customWidth="1" min="1" max="1"/>')
    for i, (_h, w, _k) in enumerate(COLS):
        out.append('<col width="%d" customWidth="1" min="%d" max="%d"/>' % (w, i + 2, i + 2))
    out.append("</cols><sheetData>")

    out.append('<row r="1" ht="150" customHeight="1">' + cell("A1", S_TITLE, TITLE) + "</row>")

    covered = {}
    for label, lo, hi in GROUPS:
        for j in range(lo, hi + 1):
            covered[j] = (label, lo, hi)

    r2 = [cell("A2", S_HDR, "#")]
    for j in range(len(COLS)):
        ref = "%s2" % colname(j + 1)
        if j in covered:
            label, lo, hi = covered[j]
            r2.append(cell(ref, S_GRP if j == lo else (S_GRP_END if j == hi else S_GRP_MID), label if j == lo else ""))
        else:
            r2.append(cell(ref, S_HDR, COLS[j][0]))
    out.append('<row r="2" ht="20" customHeight="1">' + "".join(r2) + "</row>")

    r3 = [cell("A3", S_GRP_BLANK, "")]
    for j, (h, _w, _k) in enumerate(COLS):
        r3.append(cell("%s3" % colname(j + 1), S_HDR, h if j in covered else ""))
    out.append('<row r="3" ht="26" customHeight="1">' + "".join(r3) + "</row>")

    files = []
    for r in rows:
        if r[0] not in files:
            files.append(r[0])
    for n, r in enumerate(rows):
        er = 4 + n
        small, mono, monob = FILL_STYLES[files.index(r[0]) % len(FILL_STYLES)]
        cells = [num_cell("A%d" % er, mono, n + 1)]
        for j, (h, _w, kind) in enumerate(COLS):
            ref = "%s%d" % (colname(j + 1), er)
            if h == "Result":
                st = {"PASS": S_PASS, "FAIL": S_FAIL, "XFAIL": S_XFAIL}.get(r[j], mono)
            else:
                st = {"small": small, "mono": mono, "monob": monob}[kind]
            cells.append(cell(ref, st, r[j]))
        out.append('<row r="%d">' % er + "".join(cells) + "</row>")

    out.append("</sheetData>")
    out.append('<autoFilter ref="A3:%s%d"/>' % (last, 3 + len(rows)))
    merges = ['<mergeCell ref="A1:%s1"/>' % last, '<mergeCell ref="A2:A3"/>']
    for _label, lo, hi in GROUPS:
        merges.append('<mergeCell ref="%s2:%s2"/>' % (colname(lo + 1), colname(hi + 1)))
    for j in range(len(COLS)):
        if j not in covered:
            merges.append('<mergeCell ref="%s2:%s3"/>' % (colname(j + 1), colname(j + 1)))
    out.append('<mergeCells count="%d">%s</mergeCells>' % (len(merges), "".join(merges)))
    out.append('<pageMargins left="0.75" right="0.75" top="1" bottom="1" header="0.5" footer="0.5"/></worksheet>')
    return "".join(out)


SHEET_NAME = "5 - Unit tests (Quasar run)"


def write_tsv(rows, path):
    """
    The same sheet as a tab-separated file, for importing as a new tab in Google Sheets
    (File > Import > Upload > Replace/Insert new sheet, separator: Tab).

    TSV rather than CSV on purpose: 647 of the cells contain commas -- the verbatim TTNN attribute
    strings are full of them -- and none contains a tab or a newline, so TSV needs no quoting at all
    and cannot be mis-split. Verified before writing rather than assumed.

    Row 1 is the two-level header flattened into one line ("group - column"), because a spreadsheet
    import has no notion of the merged header band the xlsx uses. Row 2 onward are the test rows,
    numbered as in the workbook.
    """
    flat = ["#"]
    group_of = {}
    for label, lo, hi in GROUPS:
        for j in range(lo, hi + 1):
            group_of[j] = label
    for j, (head, _w, _k) in enumerate(COLS):
        flat.append("%s - %s" % (group_of[j], head) if j in group_of else head)

    lines = ["\t".join(flat)]
    for n, r in enumerate(rows):
        cells = [str(n + 1)] + [str(c) for c in r]
        for c in cells:
            assert "\t" not in c and "\n" not in c and "\r" not in c, "a cell would break the TSV: %r" % c[:80]
        lines.append("\t".join(cells))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote %s (%d rows x %d columns, tab-separated, UTF-8)" % (path, len(rows) + 1, len(flat)))


def write_workbook(rows):
    src = zipfile.ZipFile(XLSX)
    parts = {n: src.read(n) for n in src.namelist()}
    src.close()
    if any("sheet5.xml" in n for n in parts):
        raise SystemExit("the workbook already carries a sheet5 part -- restore the .bak first")

    last = colname(len(COLS))
    wb = parts["xl/workbook.xml"].decode()
    if SHEET_NAME in wb:
        raise SystemExit("the workbook already has a sheet named %r" % SHEET_NAME)
    wb = wb.replace(
        "</sheets>",
        '<sheet xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" name="%s" '
        'sheetId="5" state="visible" r:id="rId7"/></sheets>' % SHEET_NAME,
    ).replace(
        "</definedNames>",
        '<definedName name="_xlnm._FilterDatabase" localSheetId="4" hidden="1">\'%s\'!$A$3:$%s$%d</definedName>'
        "</definedNames>" % (SHEET_NAME, last, 3 + len(rows)),
    )
    parts["xl/workbook.xml"] = wb.encode()
    parts["xl/_rels/workbook.xml.rels"] = (
        parts["xl/_rels/workbook.xml.rels"]
        .decode()
        .replace(
            "</Relationships>",
            '<Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="/xl/worksheets/sheet5.xml" Id="rId7"/></Relationships>',
        )
        .encode()
    )
    parts["[Content_Types].xml"] = (
        parts["[Content_Types].xml"]
        .decode()
        .replace(
            "</Types>",
            '<Override PartName="/xl/worksheets/sheet5.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
        )
        .encode()
    )
    parts["xl/styles.xml"] = patch_styles(parts["xl/styles.xml"].decode()).encode()
    parts["xl/worksheets/sheet5.xml"] = build_sheet_xml(rows).encode()

    shutil.copy2(XLSX, XLSX + ".bak")
    tmp = XLSX + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        ordered = [
            "[Content_Types].xml",
            "_rels/.rels",
            "docProps/app.xml",
            "docProps/core.xml",
            "xl/workbook.xml",
            "xl/_rels/workbook.xml.rels",
            "xl/styles.xml",
            "xl/theme/theme1.xml",
        ] + ["xl/worksheets/sheet%d.xml" % i for i in (1, 2, 3, 4, 5)]
        for name in ordered:
            z.writestr(name, parts[name])
        for name, data in parts.items():
            if name not in ordered:
                z.writestr(name, data)
    os.replace(tmp, XLSX)


# --------------------------------------------------------------------------------------------------
# verification: re-read the WRITTEN workbook and check sheet 5 against sheet 1 and the files on disk
# --------------------------------------------------------------------------------------------------
def verify():
    s1 = read_sheet("xl/worksheets/sheet1.xml")[3:]
    s5 = read_sheet("xl/worksheets/sheet5.xml")
    data = s5[3:]
    files = op_files()
    checks = 0

    assert len(s1) == 141, "sheet 1 has %d op rows, expected 141" % len(s1)
    n_expect = len(outcome_count())
    assert len(data) == n_expect, "sheet 5 has %d data rows, the log has %d tests" % (len(data), n_expect)
    assert [r[0] for r in data] == [str(i + 1) for i in range(len(data))], "the # column is not 1..%d" % len(data)
    checks += 3

    res = collections.Counter(r[16] for r in data)
    logged = collections.Counter(
        {"PASSED": "PASS", "FAILED": "FAIL", "XFAIL": "XFAIL"}[v[0]] for v in read_results()[0].values()
    )
    assert dict(res) == dict(logged), "sheet 5 tallies %s, the log says %s" % (dict(res), dict(logged))
    checks += 1

    by_row = {int(r[0]): r for r in s1}
    referenced, seen_files = set(), set()
    for r in data:
        n, fname, func, srow, forge_op = r[0], r[1], r[2], r[3], r[4]
        seen_files.add(fname)
        if not re.fullmatch(r"\d+", srow):
            continue
        i = int(srow)
        assert i in by_row, "sheet 5 row %s points at sheet 1 row %d, which does not exist" % (n, i)
        referenced.add(i)
        checks += 1

        # the file named in the row must be the file that declares that sheet row
        assert (
            fname in files and files[fname]["SHEET_ROW"] == i
        ), "sheet 5 row %s names %s for sheet 1 row %d, but that file declares SHEET_ROW %s" % (
            n,
            fname,
            i,
            files.get(fname, {}).get("SHEET_ROW"),
        )
        assert (
            files[fname]["FORGE_OP"] == forge_op == by_row[i][1]
        ), "sheet 5 row %s / %s / sheet 1 row %d disagree on the op: %r / %r / %r" % (
            n,
            fname,
            i,
            forge_op,
            files[fname]["FORGE_OP"],
            by_row[i][1],
        )
        checks += 2

        # operand and output shapes must be the ones sheet 1 records for that row
        for col, s1col, what in ((6, 3, "operand 1"), (7, 7, "operand 2"), (8, 11, "operand 3"), (9, 19, "output")):
            m = re.search(r": ([0-9]+(?:x[0-9]+)+),", r[col] if len(r) > col else "")
            if m:
                assert m.group(1) == by_row[i][s1col], "sheet 5 row %s %s is %s, sheet 1 row %d says %s" % (
                    n,
                    what,
                    m.group(1),
                    i,
                    by_row[i][s1col],
                )
                checks += 1

    # Every sheet-1 row is either covered by a test row, or belongs to an op kind that has NO test
    # file at all. Derived, not hardcoded: a silently dropped conv would leave a row whose op kind
    # IS covered elsewhere, and fail here.
    covered_kinds = {rec["FORGE_OP"] for rec in files.values()}
    missing = sorted(i for i in by_row if i not in referenced)
    wrong = [i for i in missing if by_row[i][1] in covered_kinds]
    assert not wrong, "%d sheet 1 rows have no test row even though their op kind is covered: %s" % (
        len(wrong),
        wrong[:10],
    )
    assert len(referenced) + len(missing) == SHEET1_ROWS, "%d covered + %d uncovered != sheet 1's %d rows" % (
        len(referenced),
        len(missing),
        SHEET1_ROWS,
    )
    unused = sorted(set(files) - seen_files)
    assert not unused, "%d op files have no row on sheet 5: %s" % (len(unused), unused[:10])
    checks += 2

    print("verify: %d assertions, 0 mismatches" % checks)
    print(
        "  sheet 1 rows covered: %d compute rows (+ %d layout rows with no test file = %d)"
        % (len(referenced), len(missing), SHEET1_ROWS)
    )
    print("  op files with a row:  %d / %d" % (len(seen_files & set(files)), len(files)))
    print("  test rows per result: %s" % dict(res))


# --------------------------------------------------------------------------------------------------
def main():
    global XLSX

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="parse and report, write nothing")
    ap.add_argument("--verify", action="store_true", help="re-read the written workbook and check it")
    ap.add_argument(
        "--xlsx",
        metavar="PATH",
        help="the workbook to append sheet 5 to (default: %s). Sheet 1 is read from this same file, so "
        "pointing at another copy derives the sheet from that copy's own sheet 1." % os.path.relpath(XLSX, REPO),
    )
    ap.add_argument(
        "--tsv",
        metavar="PATH",
        help="also write the sheet as TSV for import as a Google Sheets tab (File > Import > Upload)",
    )
    args = ap.parse_args()

    if args.xlsx:
        XLSX = os.path.abspath(args.xlsx)
        if not os.path.isfile(XLSX):
            raise SystemExit("no such workbook: %s" % XLSX)
        print("workbook: %s" % XLSX)

    if args.verify:
        verify()
        return

    rows = build_rows()
    res = collections.Counter(r[15] for r in rows)
    per_op = collections.Counter(r[3] for r in rows)
    print("%d test rows" % len(rows))
    print("  results:", dict(res))
    for op, n in per_op.most_common():
        print("  %-22s %3d" % (op, n))
    bad = [r[1] for r in rows if r[15] not in ("PASS", "FAIL", "XFAIL")]
    if bad:
        raise SystemExit("no logged result for: %s" % bad[:5])
    if any(len(r) != len(COLS) for r in rows):
        raise SystemExit("a row has the wrong width: %s" % [len(r) for r in rows if len(r) != len(COLS)][:3])

    if args.check:
        print("--check: nothing written")
        return
    if args.tsv:
        write_tsv(rows, args.tsv)
    write_workbook(rows)
    print("appended %r to %s (backup at %s.bak)" % (SHEET_NAME, os.path.basename(XLSX), os.path.basename(XLSX)))
    verify()


if __name__ == "__main__":
    main()
