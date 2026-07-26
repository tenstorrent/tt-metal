import json, os, sys, tempfile, importlib.util
from pathlib import Path

sys.path.insert(0, "/home/ttuser/tt-metal-llama/models/experimental/perf_automation")
sys.path.insert(0, "/home/ttuser/tt-metal-llama")
from agent import probes

spec = importlib.util.spec_from_file_location(
    "ccrun", "/home/ttuser/tt-metal-llama/models/experimental/perf_automation/cc_optimize/run.py"
)
ccrun = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ccrun)
PERF_DIR = "models/experimental/perf_automation"


def fixture(root, base, timeout=10800):
    r = root / PERF_DIR / "runs" / "2026-01-01T00-00-00"
    (r / "profiles").mkdir(parents=True, exist_ok=True)
    (r / "manifest.json").write_text(json.dumps({"config": {"timeout": timeout}}))
    (r / "events.jsonl").write_text(json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": base}) + "\n")
    return r / "manifest.json"


# OBSERVED durations (seconds) from today's real runs + ACE audit. p95-ish realistic worst case per size.
# base, name, {operation: realistic_duration}
SIZES = [
    (0.8, "micro module", dict(profile=6, pcc=8, build=60, agent_call=40, collect=5, tracy_pp=15, reset=45, cool=60)),
    (
        3.16,
        "tiny (ACE enc_layer)",
        dict(profile=13, pcc=20, build=120, agent_call=60, collect=6, tracy_pp=25, reset=45, cool=60),
    ),
    (
        13.06,
        "mid (ACE dit_layer)",
        dict(profile=52, pcc=70, build=240, agent_call=90, collect=8, tracy_pp=60, reset=45, cool=90),
    ),
    (
        42.61,
        "large (ACE audio_tok)",
        dict(profile=170, pcc=210, build=408, agent_call=150, collect=10, tracy_pp=140, reset=45, cool=120),
    ),
    (
        146.72,
        "8B full pipe (llama)",
        dict(profile=154, pcc=1400, build=872, agent_call=300, collect=6, tracy_pp=420, reset=60, cool=120),
    ),
    (
        900.0,
        "very slow model",
        dict(profile=950, pcc=3000, build=1800, agent_call=600, collect=20, tracy_pp=1200, reset=60, cool=180),
    ),
    (
        3000.0,
        "extreme model",
        dict(profile=3100, pcc=9000, build=3600, agent_call=900, collect=30, tracy_pp=3000, reset=90, cool=240),
    ),
]


def timers(root, base):
    os.environ["PERF_MCP_MANIFEST"] = str(fixture(root, base))
    for k in (
        "PERF_MCP_MEASURE_BACKSTOP",
        "PERF_MCP_ROUND_MAX_SEC",
        "PERF_MCP_ROUND_STALL_SEC",
        "PERF_MCP_MEASURE_STALL_SEC",
        "PERF_MCP_DISCOVER_STALL_SEC",
        "AGENT_CALL_TIMEOUT_S",
        "AGENT_DEVICE_CALL_TIMEOUT_S",
        "PERF_MCP_COMPONENT_RUN_TIMEOUT_S",
    ):
        os.environ.pop(k, None)
    return {
        "round_cap(UNPROD)": (ccrun._round_hard_cap(root, 600), "pcc"),
        "round_stall(FROZEN)": (600, "pcc"),
        "measure_backstop": (probes.adaptive_backstop(3600), "profile"),
        "measure_stall": (600, "profile"),
        "discover_stall": (1200, "build"),
        "discover_backstop": (ccrun._adaptive_cap(root, 3600), "build"),
        "sdk_agent_call": (300, "agent_call"),
        "sdk_device_agent_call": (3600, "agent_call"),
        "component_run": (240, "build"),
        "matmul_sweep_run": (900, "profile"),
        "cc_discovery_subproc": (1200, "build"),
        "pytest_collect": (120, "collect"),
        "tracy_postprocess": (600, "tracy_pp"),
        "git_subprocess": (300, "reset"),
        "thermal_cool_max": (120, "cool"),
    }


LOOSE = 20  # >20x the work = hang goes undetected far too long
rows = []
with tempfile.TemporaryDirectory() as td:
    for base, name, obs in SIZES:
        root = Path(td) / f"m{base}"
        root.mkdir(parents=True, exist_ok=True)
        for tname, (val, opkey) in timers(root, base).items():
            work = obs[opkey]
            if val < work:
                verdict = "FAIL-TIGHT"  # kills legitimate work
            elif val > LOOSE * work:
                verdict = "FAIL-LOOSE"  # hang undetected far too long
            else:
                verdict = "ok"
            rows.append((name, tname, val, work, val / work, verdict))

print(f"{'model':<24}{'timer':<24}{'limit':>7}{'work':>7}{'ratio':>8}  verdict")
print("-" * 88)
for r in rows:
    flag = "" if r[5] == "ok" else "  <<<"
    print(f"{r[0]:<24}{r[1]:<24}{r[2]:>7}{r[3]:>7}{r[4]:>7.1f}x  {r[5]}{flag}")
tight = [r for r in rows if r[5] == "FAIL-TIGHT"]
loose = [r for r in rows if r[5] == "FAIL-LOOSE"]
print("\n" + "=" * 88)
print(
    f"TOTAL {len(rows)} (timer x model) combinations:  ok={len(rows)-len(tight)-len(loose)}  FAIL-TIGHT={len(tight)}  FAIL-LOOSE={len(loose)}"
)
print(f"\nFAIL-TIGHT (kills legitimate work) — {len(tight)}:")
for r in tight:
    print(f"  {r[0]:<24}{r[1]:<24}limit {r[2]}s < work {r[3]}s")
print(f"\nFAIL-LOOSE (hang undetected >{LOOSE}x) — {len(loose)}:")
for r in loose:
    print(f"  {r[0]:<24}{r[1]:<24}limit {r[2]}s = {r[4]:.0f}x work {r[3]}s")
# per-timer summary
print("\nPER-TIMER verdict across all 7 model sizes:")
byt = {}
for r in rows:
    byt.setdefault(r[1], []).append(r[5])
for t, v in byt.items():
    bad = sum(1 for x in v if x != "ok")
    print(f"  {t:<24}{'ALL OK' if bad==0 else f'{bad}/7 sizes BROKEN'}")
