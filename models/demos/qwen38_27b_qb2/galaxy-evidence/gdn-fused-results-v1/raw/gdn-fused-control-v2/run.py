import hashlib
import json
import signal
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save

p = Path("/home/ttuser/qwen38-artifacts-20261007")
src = p / "gdn-fused-source-v2"
ctl = p / "gdn-fused-control-v2"
out = p / "gdn-fused-v2"
task = Path("/home/ttuser/kimi-prefill.Ubx2wY/runtime/qwen38-27b-20261006")
status = dict(
    state="preflight",
    cleanup_completed=False,
    survives_disconnect=True,
    resumes_after_reboot=False,
    full_model_qualified=False,
    promoted_to_serving=False,
    hardware_lock="/tmp/tt-device.lock",
)


def interrupted(signum, frame):
    raise InterruptedError(f"Signal {signum}")


signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
save(out / "queue.json", status)
try:
    for name, digest in json.loads((ctl / "source-manifest.json").read_text()).items():
        assert hashlib.sha256((src / name).read_bytes()).hexdigest() == digest, name
    env = environment(task, src, p / "checkpoint-pinned-1d4bf0f2")
    for key in (
        "TT_METAL_SIMULATOR",
        "TT_METAL_KERNEL_PATH",
        "TT_METAL_PROFILE_PERF_COUNTERS",
        "TT_METAL_DISABLE_SFPLOADMACRO",
        "TT_VISIBLE_DEVICES",
    ):
        env.pop(key, None)
    env.update(QWEN_GDN_FUSED_BENCHMARK="1", QWEN_GDN_FUSED_SMOKE="1", QWEN_GDN_FUSED_RECEIPT=str(out / "smoke.json"))
    command = [
        "/bin/bash",
        str(src / "scripts/run_safe_pytest.sh"),
        str(src / "models/demos/qwen38_27b_qb2/tests/test_gdn_fused.py"),
        f"--rootdir={src}",
        "-c",
        str(src / "pytest.ini"),
        "-vv",
        "-s",
        "--tb=short",
        "--timeout=900",
        f"--junitxml={out}/hardware.xml",
    ]
    status.update(state="hardware_or_waiting_for_lock", command=command)
    save(out / "queue.json", status)
    run_capture(command, cwd=src, env=env, root=out, timeout=1200)
    receipt = json.loads((out / "smoke.json").read_text())
    assert receipt["state"] == "completed" and receipt["passed"] and receipt["cleanup_completed"]
    assert receipt["device_ids"] == [0, 4, 12, 8] and len(receipt["cases"]) == 6
    assert all(
        c["passed"] and len(c["steps"]) == 4 and all(s["state"] and s["output"] and s["raw"] for s in c["steps"])
        for c in receipt["cases"]
    )
    suites = ET.parse(out / "hardware.xml").getroot().findall(".//testsuite")
    assert sum(int(s.get("tests", 0)) for s in suites) == 1 and all(
        not int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
    )
    assert all(len(c.get("traced_comparison", [])) == 5 for c in receipt["cases"] if c["instance"] == 2)
    status.update(
        state="completed", cleanup_completed=True, exact_component_match=True, traced_comparison_completed=True
    )
except BaseException as error:
    status.update(state="failed", error=type(error).__name__, detail=str(error)[:3000])
    raise
finally:
    save(out / "queue.json", status)
