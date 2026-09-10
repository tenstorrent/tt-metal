# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 08: the HTTP server (``server/app.py``) launched the way tt-model's ``tt-dit-server`` kind launches it.

The session fixture starts ``python -m uvicorn --lifespan on ...server.app:app`` in a subprocess with ``HF_MODEL`` set
to the local snapshot and ``MM3_MESH_SHAPE=1x1``, waits for uvicorn's ``Application startup complete`` line and a
``/health`` answer, and at the end sends SIGTERM and checks the process exits cleanly. Gate tests (``-m "not slow"``):

1. ``/health`` reports ``status ok``, ``warm true``, the model id and a 1x1 device;
2. ``/v1/models`` has the OpenAI list shape with the model id;
3. ``POST /v1/audio/speech`` for 10 s (seed 7, 30 steps) returns a 44.1 kHz stereo wav whose duration is within 1 s of
   ``frames / 25`` and whose RMS is above 1e-3; the wav is saved under ``generated/``;
4. bad requests return 4xx (empty lyrics, unknown response_format, too many frames, malformed JSON) and ``stream: true``
   returns 501, and the server is still healthy afterwards;
5. a second real request (2 s, ``audio_duration`` alias) succeeds - the server survives more than one generation;
6. the model card's curl command (``max_new_tokens`` 750 -> 30 s) runs verbatim against the server and its wav is saved.

Evidence: ``doc/server/results.json`` (timings, wav statistics, server load log from ``/health``) and the server log
under ``generated/server_test.log``.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from loguru import logger

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = MODEL_DIR / "doc" / "server"
GENERATED_DIR = MODEL_DIR / "generated"
REPO_ROOT = MODEL_DIR.parents[2]
APP = "models.autoports.minimaxai_minimax_music3.server.app:app"
SAMPLING_RATE = 44100
FRAME_RATE = 25.0
READY_LINE = "Application startup complete"
STARTUP_TIMEOUT_S = float(os.environ.get("MM3_SERVER_STARTUP_TIMEOUT_S", 1500))

GOLDEN_PROMPT = (
    "Genre: acoustic pop. BPM: 96. Key: C major. Warm and intimate, building gently into the chorus. "
    "Vocals: soft female lead, close and breathy, light stacked harmonies in the chorus. "
    "Arrangement: fingerpicked guitar and soft piano; brushed drums and upright bass enter in the chorus."
)
GOLDEN_LYRICS = "[verse]\nMorning light filtering through the pine\nEvery quiet street is yours and mine\n[chorus]\nSoftly the world begins to breathe"

# Verbatim from the model card (README.md "Generate Music"), only the port is substituted.
MODEL_CARD_CURL = """curl http://127.0.0.1:{port}/v1/audio/speech \\
  -H 'Content-Type: application/json' \\
  -d '{{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Verse]\\nMorning light filtering through the pine\\n[Chorus]\\nSoftly the world begins to breathe",
    "instructions": "A warm acoustic pop song with intimate female vocals, fingerpicked guitar, soft piano, and a gradual emotional build into a wide final chorus.",
    "response_format": "wav",
    "seed": 7,
    "max_new_tokens": 750,
    "stream": false
  }}' \\
  --output {output}"""


# ----------------------------------------------------------------------------- helpers
def _run_meta() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=MODEL_DIR, text=True).strip()
    except Exception:  # pragma: no cover
        commit = "unknown"
    return {"recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "commit": commit, "loadavg_1m": os.getloadavg()[0]}


def _record(name: str, **fields):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    path = DOC_DIR / "results.json"
    results = json.loads(path.read_text()) if path.is_file() else {}
    fields["_meta"] = _run_meta()
    results[name] = fields
    path.write_text(json.dumps(results, indent=2, sort_keys=True, default=float) + "\n")


def _device_available() -> bool:
    """Probe the device node only: building ttnn's cluster here (``ttnn.get_num_devices``) takes the ``CHIP_IN_USE``
    lock in the test process and the server subprocess then waits on it forever (seen in the first run). ``conftest.py``
    still imports ttnn, which is harmless - only the cluster construction takes the lock."""
    visible = os.environ.get("TT_METAL_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    return Path(f"/dev/tenstorrent/{visible}").exists()


def _board_name() -> str:
    """``tt-smi -s`` board type of the visible chip (what the launcher passes as ``MESH_DEVICE``), or ``unknown``."""
    try:
        out = subprocess.run(["tt-smi", "-s"], capture_output=True, text=True, timeout=60).stdout
        info = json.loads(out)["device_info"]
        idx = int((os.environ.get("TT_METAL_VISIBLE_DEVICES", "0").split(",")[0] or "0"))
        return str(info[idx]["board_info"]["board_type"])
    except Exception:
        return "unknown"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check_wav(data: bytes, expected_frames: int | None = None) -> dict:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    info = sf.info(io.BytesIO(data))
    assert info.format == "WAV" and info.subtype == "PCM_16", (info.format, info.subtype)
    assert sr == SAMPLING_RATE, sr
    assert audio.shape[1] == 2, audio.shape
    assert np.isfinite(audio).all()
    seconds = audio.shape[0] / sr
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert rms > 1e-3, f"silent audio (RMS {rms})"
    if expected_frames is not None:
        assert abs(seconds - expected_frames / FRAME_RATE) < 1.0, (seconds, expected_frames / FRAME_RATE)
    return {"seconds": round(seconds, 3), "rms": rms, "peak": float(np.abs(audio).max()), "channels": 2, "sr": sr}


def _save_wav(name: str, data: bytes) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_DIR / name
    path.write_bytes(data)
    return path


class Server:
    def __init__(self, port: int, proc: subprocess.Popen, log_path: Path):
        self.port = port
        self.proc = proc
        self.log_path = log_path
        self.base = f"http://127.0.0.1:{port}"
        self.startup_s: float | None = None

    def get(self, path: str, timeout: float = 30.0):
        import httpx

        return httpx.get(self.base + path, timeout=timeout)

    def post(self, path: str, body, timeout: float = 900.0):
        import httpx

        return httpx.post(self.base + path, json=body, timeout=timeout)

    def speech(self, timeout: float = 900.0, **body):
        return self.post("/v1/audio/speech", body, timeout=timeout)

    def log_text(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path.is_file() else ""


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def server():
    if not _device_available():
        pytest.skip("no Tenstorrent device visible")
    weights = os.environ.get("MM3_WEIGHTS") or os.environ.get("HF_MODEL")
    if not weights:
        pytest.skip("MM3_WEIGHTS / HF_MODEL not set")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    log_path = GENERATED_DIR / "server_test.log"
    env = dict(os.environ)
    env.update({"HF_MODEL": weights, "MM3_MESH_SHAPE": "1x1"})
    env.setdefault("MESH_DEVICE", _board_name())
    env.pop("PYTEST_CURRENT_TEST", None)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--lifespan",
        "on",
        APP,
    ]
    logger.info(f"starting server: {' '.join(cmd)} (log {log_path})")
    t0 = time.time()
    with log_path.open("wb") as log_file:
        proc = subprocess.Popen(cmd, cwd=REPO_ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    srv = Server(port, proc, log_path)
    try:
        ready = False
        while time.time() - t0 < STARTUP_TIMEOUT_S:
            if proc.poll() is not None:
                pytest.fail(
                    f"server exited during startup with code {proc.returncode}; log tail:\n{srv.log_text()[-4000:]}"
                )
            if READY_LINE in srv.log_text():
                ready = True
                break
            time.sleep(2)
        if not ready:
            pytest.fail(f"server not ready after {STARTUP_TIMEOUT_S:.0f} s; log tail:\n{srv.log_text()[-4000:]}")
        # The readiness line is printed after the lifespan finished, so the socket answers immediately.
        deadline = time.time() + 60
        while True:
            try:
                r = srv.get("/health", timeout=10)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            assert time.time() < deadline, "readiness line seen but /health does not answer"
            time.sleep(1)
        srv.startup_s = time.time() - t0
        logger.info(f"server ready in {srv.startup_s:.0f} s")
        yield srv
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                code = proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                proc.kill()
                code = proc.wait(timeout=60)
                pytest.fail("server did not exit within 180 s of SIGTERM")
            closed = "device closed" in srv.log_text()
            logger.info(f"server exited with {code} after SIGTERM (device closed: {closed})")
            _record("shutdown", exit_code=code, device_closed=closed)
            # uvicorn >= 0.29 runs the lifespan shutdown and then re-raises the captured SIGTERM (``Server.capture_signals``),
            # so a clean exit is either 0 or -SIGTERM; the lifespan must have logged the device release either way.
            assert code in (
                0,
                -signal.SIGTERM,
            ), f"server exit code {code} after SIGTERM; log tail:\n{srv.log_text()[-3000:]}"
            assert closed, f"lifespan shutdown did not close the device; log tail:\n{srv.log_text()[-3000:]}"


# ----------------------------------------------------------------------------- tests
@pytest.mark.hardware
@pytest.mark.timeout(2400)
def test_health(server):
    r = server.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok" and body["warm"] is True and body["busy"] is False
    assert body["model"] == "MiniMaxAI/MiniMax-Music3"
    # One chip in the mesh; its id is UMD's logical id of the visible chip (3 on this host with TT_METAL_VISIBLE_DEVICES=0).
    assert body["device"]["mesh_shape"] == "1x1" and len(body["device"]["device_ids"]) == 1, body["device"]
    assert body["device"]["arch"] == "blackhole"
    logger.info(f"/health: {json.dumps(body)}")
    _record("startup", startup_s=server.startup_s, health=body)


@pytest.mark.hardware
@pytest.mark.timeout(300)
def test_models(server):
    r = server.get("/v1/models")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "list" and isinstance(body["data"], list) and len(body["data"]) == 1
    model = body["data"][0]
    assert model["id"] == "MiniMaxAI/MiniMax-Music3" and model["object"] == "model"
    assert "created" in model and "owned_by" in model


@pytest.mark.hardware
@pytest.mark.timeout(1200)
def test_speech_10s(server):
    """10 s song (seed 7, 30 steps) while ``/health`` keeps answering (``busy: true``) from the event loop."""
    import threading

    frames = 250
    result = {}

    def _request():
        result["response"] = server.speech(
            model="MiniMaxAI/MiniMax-Music3",
            input=GOLDEN_LYRICS,
            instructions=GOLDEN_PROMPT,
            response_format="wav",
            seed=7,
            max_new_tokens=frames,
            stream=False,
        )

    t0 = time.time()
    worker = threading.Thread(target=_request)
    worker.start()
    health_latencies, saw_busy = [], False
    while worker.is_alive():
        t_h = time.time()
        h = server.get("/health", timeout=10)
        health_latencies.append(time.time() - t_h)
        assert h.status_code == 200
        saw_busy = saw_busy or h.json()["busy"] is True
        time.sleep(0.5)
    worker.join()
    wall = time.time() - t0
    r = result["response"]
    assert saw_busy, "/health never reported busy while the song was generating"
    health_ms = {
        "n": len(health_latencies),
        "max_ms": round(1e3 * max(health_latencies), 1),
        "mean_ms": round(1e3 * sum(health_latencies) / len(health_latencies), 1),
    }
    assert health_ms["max_ms"] < 5000, health_ms
    assert r.status_code == 200, r.text[:2000]
    assert r.headers["content-type"].startswith("audio/wav"), r.headers
    got_frames = int(r.headers["x-mm3-frames"])
    assert got_frames == frames or r.headers["x-mm3-stopped-by"] == "end_token", dict(r.headers)
    stats = _check_wav(r.content, got_frames)
    path = _save_wav("server_speech_seed7_10s.wav", r.content)
    stats.update(
        frames=got_frames,
        seed=int(r.headers["x-mm3-seed"]),
        stopped_by=r.headers["x-mm3-stopped-by"],
        prompt_tokens=int(r.headers["x-mm3-prompt-tokens"]),
        generation_s=float(r.headers["x-mm3-generation-seconds"]),
        wall_s=round(wall, 2),
        bytes=len(r.content),
        wav=str(path.relative_to(MODEL_DIR)),
        health_during_generation=health_ms,
    )
    assert stats["seed"] == 7
    logger.info(f"10 s request: {json.dumps(stats)}")
    _record("speech_10s", **stats)


@pytest.mark.hardware
@pytest.mark.timeout(300)
def test_bad_requests(server):
    cases = {}
    r = server.speech(input="", instructions=GOLDEN_PROMPT, timeout=60)
    cases["empty_lyrics"] = r.status_code
    assert r.status_code in (400, 422), r.text
    r = server.speech(input=GOLDEN_LYRICS, instructions=GOLDEN_PROMPT, response_format="mp3", timeout=60)
    cases["mp3"] = r.status_code
    assert r.status_code in (400, 422), r.text
    r = server.speech(input=GOLDEN_LYRICS, instructions=GOLDEN_PROMPT, max_new_tokens=9001, timeout=60)
    cases["too_many_frames"] = r.status_code
    assert r.status_code in (400, 422), r.text
    assert "9000" in r.text
    r = server.speech(input=GOLDEN_LYRICS, instructions=GOLDEN_PROMPT, audio_duration=0.01, timeout=60)
    cases["too_short"] = r.status_code
    assert r.status_code in (400, 422), r.text
    r = server.speech(input=GOLDEN_LYRICS, instructions=GOLDEN_PROMPT, num_inference_steps=0, timeout=60)
    cases["zero_steps"] = r.status_code
    assert r.status_code in (400, 422), r.text
    r = server.speech(input=GOLDEN_LYRICS, instructions=GOLDEN_PROMPT, stream=True, timeout=60)
    cases["stream"] = r.status_code
    assert r.status_code == 501, r.text
    r = server.speech(input=GOLDEN_LYRICS, instructions=GOLDEN_PROMPT, model="gpt-4o-mini-tts", timeout=60)
    cases["wrong_model"] = r.status_code
    assert r.status_code == 404, r.text
    # A prompt over 5000 tokens (the lyrics limit is part of the checkpoint contract).
    r = server.speech(input="la " * 6000, instructions=GOLDEN_PROMPT, timeout=60)
    cases["prompt_too_long"] = r.status_code
    assert r.status_code == 400 and "5000" in r.text, r.text
    import httpx

    r = httpx.post(
        server.base + "/v1/audio/speech", content=b"{not json", headers={"content-type": "application/json"}, timeout=60
    )
    cases["malformed_json"] = r.status_code
    assert r.status_code == 422, r.text
    for status in cases.values():
        assert 400 <= status < 600
    # Every error, validation errors included, uses the OpenAI shape.
    assert set(r.json()) == {"error"} and {"message", "type"} <= set(r.json()["error"]), r.text
    r = server.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["busy"] is False
    logger.info(f"bad requests -> {cases}")
    _record("bad_requests", status_codes=cases)


@pytest.mark.hardware
@pytest.mark.timeout(600)
def test_second_request_audio_duration_alias(server):
    """The server survives a second generation; ``audio_duration`` is accepted instead of ``max_new_tokens``."""
    r = server.speech(
        input=GOLDEN_LYRICS, instructions=GOLDEN_PROMPT, audio_duration=2.0, seed=11, num_inference_steps=6
    )
    assert r.status_code == 200, r.text[:2000]
    frames = int(r.headers["x-mm3-frames"])
    assert frames == 50 or r.headers["x-mm3-stopped-by"] == "end_token"
    stats = _check_wav(r.content, frames)
    stats.update(frames=frames, generation_s=float(r.headers["x-mm3-generation-seconds"]))
    health = server.get("/health").json()
    assert health["status"] == "ok" and health["requests_served"] >= 2, health
    logger.info(f"2 s request: {json.dumps(stats)}")
    _record("speech_2s_alias", requests_served=health["requests_served"], **stats)


@pytest.mark.hardware
@pytest.mark.timeout(600)
def test_concurrent_requests_serialize(server):
    """Two requests at once: both succeed, and they ran one after the other behind the lock."""
    from concurrent.futures import ThreadPoolExecutor

    def _one(seed):
        t0 = time.time()
        r = server.speech(
            input=GOLDEN_LYRICS, instructions=GOLDEN_PROMPT, audio_duration=2.0, seed=seed, num_inference_steps=6
        )
        return r, time.time() - t0

    t0 = time.time()
    with ThreadPoolExecutor(2) as pool:
        (r1, w1), (r2, w2) = list(pool.map(_one, [21, 22]))
    wall = time.time() - t0
    assert r1.status_code == 200 and r2.status_code == 200, (r1.text[:500], r2.text[:500])
    g1, g2 = (float(r["x-mm3-generation-seconds"]) for r in (r1.headers, r2.headers))
    _check_wav(r1.content, int(r1.headers["x-mm3-frames"]))
    _check_wav(r2.content, int(r2.headers["x-mm3-frames"]))
    assert int(r1.headers["x-mm3-seed"]) == 21 and int(r2.headers["x-mm3-seed"]) == 22
    # Serialized: the pair takes at least the sum of the two generations (a small slack for the HTTP round trips).
    assert wall >= g1 + g2 - 0.5, (wall, g1, g2)
    health = server.get("/health").json()
    assert health["busy"] is False and health["status"] == "ok"
    stats = {
        "wall_s": round(wall, 2),
        "generation_s": [g1, g2],
        "client_wall_s": [round(w1, 2), round(w2, 2)],
        "requests_served": health["requests_served"],
    }
    logger.info(f"concurrent requests: {json.dumps(stats)}")
    _record("concurrent_requests", **stats)


@pytest.mark.hardware
@pytest.mark.timeout(1200)
def test_model_card_curl(server):
    """The README curl (750 frames = 30 s, seed 7) verbatim, output saved under ``generated/``."""
    if shutil.which("curl") is None:
        pytest.skip("curl not installed")
    out = GENERATED_DIR / "minimax_music3_model_card_curl.wav"
    out.unlink(missing_ok=True)
    cmd = MODEL_CARD_CURL.format(port=server.port, output=out)
    t0 = time.time()
    res = subprocess.run(["bash", "-c", cmd + " -sS -w '%{http_code}'"], capture_output=True, text=True, timeout=1100)
    wall = time.time() - t0
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip().endswith("200"), (res.stdout, out.read_bytes()[:500] if out.is_file() else None)
    data = out.read_bytes()
    info = sf.info(io.BytesIO(data))
    stats = _check_wav(data)
    assert stats["seconds"] <= 750 / FRAME_RATE + 1.0
    stats.update(
        wall_s=round(wall, 2),
        bytes=len(data),
        wav=str(out.relative_to(MODEL_DIR)),
        frames_seen=int(round(stats["seconds"] * FRAME_RATE)),
        subtype=info.subtype,
    )
    logger.info(f"model card curl: {json.dumps(stats)}")
    _record("model_card_curl", command=cmd, **stats)
