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


def make_fixture(root: Path, base_s: float, timeout: int = 10800):
    run = root / PERF_DIR / "runs" / "2026-01-01T00-00-00"
    (run / "profiles").mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(json.dumps({"config": {"timeout": timeout}}))
    (run / "events.jsonl").write_text(
        json.dumps({"stage": "tracy_baseline", "event": "done", "seconds": base_s}) + "\n"
    )
    return run / "manifest.json"


CASES = [
    ("tiny module (ACE encoder_layer)", 3.16),
    ("small module (ACE di_t_model)", 5.41),
    ("mid module (ACE di_t_layer)", 13.06),
    ("large module (ACE audio_tok)", 42.61),
    ("8B full pipeline (llama)", 146.72),
    ("very slow model", 900.00),
    ("extreme model", 3000.00),
]

print(f"{'model kind':<34}{'base':>9}{'measure_backstop':>18}{'round_cap':>11}{'ratio m/base':>13}  verdict")
print("-" * 104)
fails = []
for name, base in CASES:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mani = make_fixture(root, base)
        os.environ["PERF_MCP_MANIFEST"] = str(mani)
        os.environ.pop("PERF_MCP_MEASURE_BACKSTOP", None)
        os.environ.pop("PERF_MCP_ROUND_MAX_SEC", None)
        mb = probes.adaptive_backstop(3600)
        rc = ccrun._round_hard_cap(root, 600)
        ratio = mb / base
        scaled = "SCALES" if mb != 3600 else "floor"
        bad = []
        if base < 60 and mb > 60 * base:
            bad.append("small: budget >60x work (hang undetected for ages)")
        if base > 120 and rc <= 2400:
            bad.append("big: round cap stuck at floor (premature kill)")
        verdict = "OK" if not bad else "; ".join(bad)
        if bad:
            fails.append((name, verdict))
        print(f"{name:<34}{base:>9.2f}{mb:>18}{rc:>11}{ratio:>12.0f}x  [{scaled}] {verdict}")
print()
print(f"RESULT: {len(fails)} of {len(CASES)} model kinds mis-served")
for n, v in fails:
    print(f"  FAIL  {n}: {v}")
