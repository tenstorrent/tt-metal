import os, sys, re

os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
os.environ["TT_METAL_PROFILER_MID_RUN_DUMP"] = "1"
os.environ["TT_METAL_PROFILER_CPP_POST_PROCESS"] = "1"
os.environ["TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES"] = "1"
repo = os.getcwd()
claude = os.path.join(repo, ".claude")
sys.path.insert(0, claude)
out = os.path.join(repo, "generated", "sdpa_perf")
os.makedirs(out, exist_ok=True)
junit = os.path.join(out, "junit.xml")
import pytest

rc = pytest.main(
    [
        "-p",
        "eval.metrics_plugin",
        "-p",
        "eval.axes_plugin",
        "-p",
        "no:cacheprovider",
        f"--junit-xml={junit}",
        "-q",
        "--no-header",
        "-k",
        "(Q1x1x128x64 or Q1x1x512x64) and self and tile_aligned",
        f"{claude}/eval/golden_tests/scaled_dot_product_attention/test_golden.py",
    ]
)
import xml.etree.ElementTree as ET

root = ET.parse(junit).getroot()
lines = ["RESULTS (dtype/mask/scale @ shape | status | pcc | device_kernel_ns)"]
npass = nperf = 0
for tc in root.iter("testcase"):
    nm = tc.get("name")
    props = {p.get("name"): p.get("value") for p in tc.iter("property")}
    sk = tc.find("skipped")
    fa = tc.find("failure")
    er = tc.find("error")
    status = "xfail/skip" if sk is not None else ("FAIL" if (fa is not None or er is not None) else "passed")
    perf = props.get("metric.device_kernel_ns")
    pcc = props.get("metric.pcc")
    if status == "passed":
        npass += 1
    if perf is not None:
        nperf += 1
    # extract compact label
    sh = re.search(r"Q(\d+x\d+x\d+x\d+)", nm)
    dt = re.search(r"dtype=(\w+)", nm)
    mk = re.search(r"mask_mode=(\w+)", nm)
    sc = re.search(r"scale_mode=(\w+)", nm)
    lab = f"{sh.group(1) if sh else '?':14} {dt.group(1) if dt else '?':9} mask={mk.group(1) if mk else '?':7} scale={sc.group(1) if sc else '?'}"
    if status == "passed" or perf is not None:
        lines.append(f"{lab:52} {status:8} pcc={(pcc or '')[:9]:9} ns={perf}")
lines.append(f"SUMMARY passed={npass} with_perf={nperf} rc={rc}")
open(os.path.join(out, "summary.txt"), "w").write("\n".join(lines))
