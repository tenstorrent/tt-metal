#!/usr/bin/env python3
import os, sys, time, json, signal, subprocess, select, codecs, json
import pytest

OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "sdxl_test_results.json"
from models.experimental.stable_diffusion_xl_base.conftest import get_device_name


def kill_all(proc, grace_period=100):
    proc.terminate()  # sends SIGTERM ('kill <pid>')

    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < grace_period:
        time.sleep(1)

    if proc.poll() is None:
        proc.kill()  # sends SIGKILL ('kill -9 <pid>')
    else:
        pass
        # print("process terminated successfully", flush=True)


def test_main():
    hang_timeout = 1000  # seconds

    argv = [
        "pytest",
        "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_accuracy.py",
        "--num-prompts=5000",
        "-k",
        "device_vae and device_encoders and with_trace",
    ]

    env = os.environ.copy()
    env["TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE"] = "7,7"
    env["PYTHONUNBUFFERED"] = "1"
    env["TT_MM_THROTTLE_PERF"] = "0"

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )

    def handle_sigterm(signum, frame):
        """
        This handler ensures that if this script is killed all child processes
        (such as pytest) are also terminated cleanly.
        """
        kill_all(proc)
        sys.exit(1)

    signal.signal(signal.SIGTERM, handle_sigterm)  # signal sent by 'kill <pid>'
    signal.signal(signal.SIGINT, handle_sigterm)  # signal sent by Ctrl+C in the terminal

    start_ts = time.time()
    last_activity = time.time()
    hang_detected = False

    decoder = codecs.getincrementaldecoder("utf-8")()

    while True:
        if proc.poll() is not None:
            break
        r, _, _ = select.select([proc.stdout], [], [], 1)
        if r:
            chunk = os.read(proc.stdout.fileno(), 65536)
            if chunk:
                last_activity = time.time()
                text = decoder.decode(chunk)

                sys.stdout.write(text)
                sys.stdout.flush()
        else:
            # print("total_wait_for now: ", time.time() - last_activity, "sec", flush=True)
            if time.time() - last_activity > hang_timeout:
                hang_detected = True
                # print("Hang detected, calling kill_all...", flush=True)
                kill_all(proc)
                break

    tail = decoder.decode(b"", final=True)
    if tail:
        # print("Flushing tail from decoder:", repr(tail), flush=True)
        sys.stdout.write(tail)
        sys.stdout.flush()

    if proc.returncode != 0:
        pytest.fail(f"Inner pytest failed with exit code {proc.returncode}")

    end_ts = time.time()
    report = {
        "hang_detected": hang_detected,
        "exit_code": proc.returncode,
        "duration_sec": round(end_ts - start_ts, 3),
        "cmd_argv": argv,
        "hang_timeout": hang_timeout,
    }

    # print("wraper report: \n:", json.dumps(report, indent=4))

    if hang_detected:
        data = {
            "model": "sdxl",
            "metadata": {
                "device": get_device_name(),
                "model_name": "sdxl",
            },
            "benchmarks_summary": [
                {
                    "device": get_device_name(),
                    "model": "sdxl",
                    "stability_check": 2,
                }
            ],
        }
    else:
        with open(f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "r") as f:
            data = json.load(f)

        data["benchmarks_summary"][0]["stability_check"] = 3

    with open(f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w") as f:
        json.dump(data, f, indent=4)

    print(f"Test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")
