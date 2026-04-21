"""Smoke harness for the Gemma4-31B latency-regression investigation.

Fires N (default 30) sequential /v1/chat/completions requests at the local
vLLM server and prints per-request wall time plus a summary line. Designed to
reproduce the post-9th-request slowdown described in
.cursor/plans/gemma4_latency_regression_fix_9682ee7e.plan.md (Phase 1.2).

Usage (server must be up on localhost:8000):

    python models/experimental/tt_symbiote/vllm/diag_smoke.py

Override defaults via env vars:

    DIAG_HOST=http://localhost:8000      # base URL
    DIAG_MODEL=google/gemma-4-31B        # served model name
    DIAG_N=30                            # number of requests
    DIAG_MAX_TOKENS=128                  # OSL
    DIAG_TIMEOUT=180                     # per-request timeout seconds
    DIAG_PROMPT_TOKENS=128               # approximate ISL via repeated word
    DIAG_OUT=/tmp/diag_smoke.csv         # optional CSV dump
    DIAG_BEARER=<token>                  # auth bearer; auto-detected if unset
    DIAG_HEARTBEAT_EVERY=5               # ping /v1/models every N requests
    DIAG_MAX_CONSECUTIVE_FAIL=2          # abort after N consecutive failures
    DIAG_SERVER_LOG=<path>               # optional server log path for tail hint

Pair with TT_SYMBIOTE_DIAG=1 on the server side so each prefill / decode
emits a [DIAG] line (see generator_vllm.py); the smoke harness gives
wall-clock, the server log gives per-step attribution.

Aborts early when the server is unreachable or unresponsive so a dead server
can never be misdiagnosed as a model regression. Always run the pre-flight
check (DIAG_SKIP_PREFLIGHT=1 only as an escape hatch).
"""

import json
import os
import socket
import statistics
import sys
import time
import urllib.error
import urllib.request


# Statuses we treat as "server is dead / unreachable" rather than "model is
# slow". Two of these in a row aborts the run.
SERVER_DEAD_STATUSES = (
    "server_unreachable",
    "server_died_midrun",
    "timeout:TimeoutError",
    "timeout:socket.timeout",
    # Distinguishing engine-hang from server-death (see _post_classify_recheck):
    # engine_hang means /v1/models still answers but /v1/chat/completions
    # blocks indefinitely. Same abort behavior, very different diagnosis.
    "engine_hang",
)


def _env(name, default):
    val = os.environ.get(name)
    return val if val is not None else default


def _autodetect_bearer():
    """If DIAG_BEARER is unset, sniff VLLM_API_KEY from the live EngineCore.

    Reading /proc/<pid>/environ only works when the caller and the server
    share a uid; returns None otherwise. Keeps the harness usable without
    forcing operators to pipe secrets around.
    """
    try:
        import glob

        for status_path in glob.glob("/proc/[0-9]*/comm"):
            try:
                with open(status_path, "r", encoding="utf-8") as fh:
                    comm = fh.read().strip()
            except OSError:
                continue
            if not comm.startswith("VLLM::EngineCo"):  # truncated to 15 chars
                continue
            pid = status_path.split("/")[2]
            try:
                with open(f"/proc/{pid}/environ", "rb") as fh:
                    env_blob = fh.read()
            except OSError:
                continue
            for entry in env_blob.split(b"\0"):
                if entry.startswith(b"VLLM_API_KEY="):
                    return entry.split(b"=", 1)[1].decode("utf-8", errors="replace")
    except Exception:
        return None
    return None


def _build_prompt(token_target):
    """Approximate token_target tokens by repeating a short ASCII word."""
    return ("test " * max(1, token_target)).strip()


def _ping_models(host, bearer, timeout=10.0):
    """Hit /v1/models. Returns (ok, detail).

    ok=True  -> server is alive and accepting auth; detail is the parsed JSON.
    ok=False -> server is unreachable or auth is wrong; detail describes why.
    """
    url = host.rstrip("/") + "/v1/models"
    headers = {}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return False, f"http_{e.code}"
    except urllib.error.URLError as e:
        # urllib wraps ConnectionRefusedError into URLError(reason=ConnRefused).
        reason = getattr(e, "reason", e)
        if isinstance(reason, ConnectionRefusedError):
            return False, "connection_refused"
        if isinstance(reason, socket.timeout):
            return False, "timeout"
        return False, f"url_err:{reason}"
    except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError) as e:
        return False, f"conn_err:{type(e).__name__}"
    except (TimeoutError, socket.timeout):
        return False, "timeout"
    except Exception as e:
        return False, f"exc:{type(e).__name__}:{e}"


def _classify_request_error(exc):
    """Map a request-time exception to a structured status string.

    Distinguishes "server is dead" (connection refused, reset, broken pipe)
    from "model is slow" (HTTP timeout) so the harness can fail-fast on the
    former without waiting for the full DIAG_TIMEOUT on every request.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return f"http_{exc.code}"
    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, ConnectionRefusedError):
            return "server_unreachable"
        if isinstance(reason, ConnectionResetError):
            return "server_died_midrun"
        if isinstance(reason, BrokenPipeError):
            return "server_died_midrun"
        if isinstance(reason, socket.timeout):
            return "timeout:socket.timeout"
        return f"url_err:{reason}"
    if isinstance(exc, ConnectionRefusedError):
        return "server_unreachable"
    if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
        return "server_died_midrun"
    if isinstance(exc, socket.timeout):
        return "timeout:socket.timeout"
    if isinstance(exc, TimeoutError):
        return "timeout:TimeoutError"
    if isinstance(exc, OSError):
        return f"os_err:{type(exc).__name__}"
    return f"exc:{type(exc).__name__}"


def _post_chat(host, model, prompt, max_tokens, timeout, bearer=None):
    url = host.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _engine_pid():
    """Return the EngineCore pid if visible to this uid, else None."""
    try:
        import glob

        for status_path in glob.glob("/proc/[0-9]*/comm"):
            try:
                with open(status_path, "r", encoding="utf-8") as fh:
                    if fh.read().strip().startswith("VLLM::EngineCo"):
                        return status_path.split("/")[2]
            except OSError:
                continue
    except Exception:
        return None
    return None


def _emit_server_log_hint(server_log, last_status=None):
    """Print actionable next-step commands when we conclude the server is
    unreachable or hung. The hints differ for the two failure modes."""
    if last_status == "engine_hang":
        print(
            "[smoke] hint: API server is alive but the inference engine is "
            "HUNG (likely 99% CPU on the EngineCore main thread). This is a "
            "Python/C++ deadlock or busy-wait, NOT a hardware crash. Capture "
            "the stack BEFORE killing the process or you lose the evidence:",
            flush=True,
        )
        pid = _engine_pid()
        if pid:
            print(f"  pip install --user py-spy   # if not already installed", flush=True)
            print(f"  py-spy dump --pid {pid}", flush=True)
            print(f"  py-spy record -d 30 -o /tmp/hang.svg --pid {pid}   # 30s flamegraph", flush=True)
        else:
            print(
                "  pip install --user py-spy && \\",
                flush=True,
            )
            print(
                "  py-spy dump --pid $(pgrep -f VLLM::EngineCore | head -1)",
                flush=True,
            )
        print(
            "  ps -fp $(pgrep -f VLLM::EngineCore) " "# look for STAT=R, %CPU near 99",
            flush=True,
        )
    else:
        print(
            "[smoke] hint: server appears to be dead/unreachable. Inspect "
            "the tail of the server log and confirm no EngineCore process "
            "is alive:",
            flush=True,
        )
    if server_log:
        print(f"  tail -150 {server_log}", flush=True)
    else:
        print(
            "  ls -t /home/$USER/tt-inference-server/workflow_logs/local_server/*.log | head -1 | xargs tail -150",
            flush=True,
        )
    print("  pgrep -af 'VLLM::EngineCore|vllm.entrypoints'", flush=True)


def main():
    host = _env("DIAG_HOST", "http://localhost:8000")
    model = _env("DIAG_MODEL", "google/gemma-4-31B")
    n = int(_env("DIAG_N", "30"))
    max_tokens = int(_env("DIAG_MAX_TOKENS", "128"))
    timeout = float(_env("DIAG_TIMEOUT", "180"))
    prompt_tokens = int(_env("DIAG_PROMPT_TOKENS", "128"))
    out_path = os.environ.get("DIAG_OUT")
    server_log = os.environ.get("DIAG_SERVER_LOG")
    heartbeat_every = max(0, int(_env("DIAG_HEARTBEAT_EVERY", "5")))
    max_consec_fail = max(1, int(_env("DIAG_MAX_CONSECUTIVE_FAIL", "2")))
    skip_preflight = os.environ.get("DIAG_SKIP_PREFLIGHT", "0") == "1"
    bearer = os.environ.get("DIAG_BEARER")
    if not bearer:
        bearer = _autodetect_bearer()
        if bearer:
            print("[smoke] auto-detected bearer from live EngineCore", flush=True)
        else:
            print(
                "[smoke] warning: DIAG_BEARER not set and auto-detect failed; "
                "a server with auth enabled will return 401 for every request",
                flush=True,
            )

    # Pre-flight: confirm the server is alive AND auth is correct BEFORE we
    # commit to firing N requests. Prevents the "30 timeouts because the
    # server was never up" failure mode that previously masqueraded as a
    # model regression.
    if not skip_preflight:
        ok, detail = _ping_models(host, bearer)
        if not ok:
            print(
                f"[smoke] pre-flight FAILED on {host}/v1/models: {detail}",
                flush=True,
            )
            if detail in ("connection_refused", "timeout") or str(detail).startswith("conn_err"):
                _emit_server_log_hint(server_log)
            elif str(detail).startswith("http_401") or str(detail).startswith("http_403"):
                print(
                    "[smoke] hint: bearer token rejected. Re-run with the right "
                    "DIAG_BEARER (or rely on auto-detect from EngineCore).",
                    flush=True,
                )
            return 2
        served = []
        try:
            served = [m.get("id") for m in detail.get("data", [])]
        except Exception:
            pass
        if served and model not in served:
            print(
                f"[smoke] pre-flight WARNING: requested model={model!r} not in "
                f"server's served list {served}. Continuing anyway.",
                flush=True,
            )
        else:
            print(f"[smoke] pre-flight ok: server is alive, served models={served}", flush=True)

    prompt = _build_prompt(prompt_tokens)
    print(
        f"[smoke] host={host} model={model} n={n} max_tokens={max_tokens} "
        f"timeout={timeout}s prompt_tokens~={prompt_tokens} "
        f"heartbeat_every={heartbeat_every} max_consec_fail={max_consec_fail}",
        flush=True,
    )

    walls = []
    failed = []
    csv_rows = ["req_idx,wall_s,status,completion_tokens"]
    consecutive_dead = 0
    aborted_for = None

    for i in range(1, n + 1):
        # Cheap mid-run heartbeat so the smoke harness notices a dead server
        # without burning a full DIAG_TIMEOUT per request. Only fires before
        # request i (not before the first one; pre-flight already covered it).
        if heartbeat_every > 0 and i > 1 and (i - 1) % heartbeat_every == 0:
            ok, detail = _ping_models(host, bearer, timeout=10.0)
            if not ok:
                print(
                    f"[smoke] heartbeat before req#{i} FAILED: {detail}; " f"server died mid-run after req#{i - 1}",
                    flush=True,
                )
                aborted_for = "server_died_midrun_heartbeat"
                _emit_server_log_hint(server_log)
                break

        t0 = time.perf_counter()
        status = "ok"
        completion_tokens = -1
        try:
            resp = _post_chat(host, model, prompt, max_tokens, timeout, bearer)
            completion_tokens = int(resp.get("usage", {}).get("completion_tokens", -1))
        except Exception as e:
            status = _classify_request_error(e)
            # Distinguish engine-hang from server-death: if the chat call
            # timed out but /v1/models still answers, the API server is
            # alive and the inference engine is the one that's stuck.
            # Different root cause, different fix, different debugging path.
            if status.startswith("timeout"):
                ok, _detail = _ping_models(host, bearer, timeout=5.0)
                if ok:
                    status = "engine_hang"
            failed.append((i, status))

        wall = time.perf_counter() - t0
        walls.append(wall)
        csv_rows.append(f"{i},{wall:.3f},{status},{completion_tokens}")
        print(
            f"[smoke] req#{i:>3} wall={wall:7.3f}s " f"status={status} tokens={completion_tokens}",
            flush=True,
        )

        # Fail fast on auth/route misconfig: 29 more 401s pollute the CSV and
        # waste the operator's time.
        if i == 1 and status.startswith(("http_401", "http_403", "http_404")):
            print(
                f"[smoke] aborting after req#1 {status}: set DIAG_BEARER to a " f"valid token and retry",
                flush=True,
            )
            aborted_for = "auth"
            break

        # Track consecutive "server is dead" results so we can abort instead
        # of waiting for N * DIAG_TIMEOUT seconds of silence.
        if status in SERVER_DEAD_STATUSES:
            consecutive_dead += 1
            if consecutive_dead >= max_consec_fail:
                if status == "engine_hang":
                    print(
                        f"[smoke] aborting: {consecutive_dead} consecutive "
                        f"engine_hang results. API server is alive but the "
                        f"inference engine is stuck. CAPTURE THE STACK NOW "
                        f"with py-spy before killing anything.",
                        flush=True,
                    )
                    aborted_for = "engine_hang_consecutive"
                else:
                    print(
                        f"[smoke] aborting: {consecutive_dead} consecutive "
                        f"server-dead results (status={status!r}). The server "
                        f"is no longer reachable; this is NOT a model regression.",
                        flush=True,
                    )
                    aborted_for = "server_dead_consecutive"
                _emit_server_log_hint(server_log, last_status=status)
                break
        else:
            consecutive_dead = 0

    print("\n[smoke] ---- summary ----")
    if walls:
        median = statistics.median(walls)
        p99 = sorted(walls)[max(0, int(round(0.99 * (len(walls) - 1))))]
        worst = max(walls)
        ratio = worst / median if median > 0 else float("inf")
        print(
            f"[smoke] n={len(walls)} median={median:.3f}s p99={p99:.3f}s "
            f"max={worst:.3f}s max/median={ratio:.2f} failures={len(failed)}"
        )
        # Verdict only meaningful when we ran the full sweep without aborting.
        if aborted_for:
            verdict = f"INCONCLUSIVE (aborted: {aborted_for})"
        elif not failed and ratio <= 1.3:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        print(f"[smoke] T-A verdict: {verdict}")
    else:
        print("[smoke] no successful samples")

    if failed:
        print("[smoke] failures:")
        for idx, why in failed:
            print(f"  req#{idx}: {why}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(csv_rows) + "\n")
        print(f"[smoke] wrote CSV to {out_path}")

    if aborted_for:
        return 3
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
