#!/usr/bin/env python3
import os, sys, time, json, signal, subprocess, select, codecs


def kill_all(proc, grace_period=100):
    print("killing all")
    print(f"proc: {proc}, pid: {proc.pid}, returncode: {proc.returncode}")
    try:
        pgid = os.getpgid(proc.pid)
        print("pgid:", pgid)
    except Exception:
        print("Exception no pgid")
        pgid = None
    try:
        if pgid is not None:
            print("calling os.killpg SIGTERM")
            os.killpg(pgid, signal.SIGTERM)
        else:
            print("calling proc.terminate()")
            proc.terminate()
    except ProcessLookupError:
        print("ProcessLookupError on terminate")

    t0 = time.time()
    while proc.poll() is None and time.time() - t0 < grace_period:
        print("waiting for process to terminate... 1 second")
        time.sleep(1)

    if proc.poll() is None:
        try:
            if pgid is not None:
                print("unsuccessful termination, sending SIGKILL")
                os.killpg(pgid, signal.SIGKILL)
            else:
                print("unsuccessful termination, sending kill")
                proc.kill()

        except ProcessLookupError:
            pass
    else:
        print("process terminated gracefully")


def main():
    idle_timeout_sec = 500
    json_out = "sdxl_watchdog_report.json"

    # argv = [
    #     "python",
    #     "models/experimental/stable_diffusion_xl_base/tests/fake_script.py",
    # ]

    argv = [
        "pytest",
        "models/experimental/stable_diffusion_xl_base/tests/test_sdxl_accuracy.py",
        "--num-prompts=2",
        "-k",
        "device_vae and device_encoders and with_trace",
    ]

    # --- pokretanje procesa ---
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TT_MM_THROTTLE_PERF"] = "5"

    # start_new_session=True == setsid(), lakše za čisto grupno gašenje
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        start_new_session=True,
        env=env,
    )

    import psutil

    parent = psutil.Process(proc.pid)
    children = parent.children(recursive=True)

    start_ts = time.time()
    last_activity = time.time()
    hang = False
    reason = None
    print("hrere")

    decoder = codecs.getincrementaldecoder("utf-8")()

    try:
        while True:
            print()
            if proc.poll() is not None:
                print("process ended, proc poll None")
                break

            r, _, _ = select.select([proc.stdout], [], [], 1)
            print("Subprocesses:", children, flush=True)
            print("r:", r)
            if r:
                chunk = os.read(proc.stdout.fileno(), 65536)
                print("chunk:", chunk)
                if chunk:
                    last_activity = time.time()
                    text = decoder.decode(chunk)

                    print("upisujem chunk")
                    sys.stdout.write(text), sys.stdout.flush()
            else:
                print("u else smo: total_wait_for now: ", time.time() - last_activity, "sec", flush=True)
                if time.time() - last_activity > idle_timeout_sec:
                    hang = True
                    reason = f"idle_{idle_timeout_sec}_seconds__no_logs"
                    kill_all(proc)
                    break
    finally:
        try:
            if proc.poll() is None:
                kill_all(proc)
        except Exception as e:
            print(f"[catch_hang_2] Exception while killing process: {e}")
        # isprazni decoder ako je ostalo nedekodiranih bajtova
        tail = decoder.decode(b"", final=True)
        if tail:
            sys.stdout.write(tail)
            sys.stdout.flush()

    end_ts = time.time()
    report = {
        "hang_detected": hang,
        "reason": reason,
        "exit_code": proc.returncode,
        "duration_sec": round(end_ts - start_ts, 3),
        "cmd_argv": argv,
        "idle_timeout_sec": idle_timeout_sec,
    }

    print(report)
    with open(json_out, "w", encoding="utf-8") as jf:
        json.dump(report, jf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
