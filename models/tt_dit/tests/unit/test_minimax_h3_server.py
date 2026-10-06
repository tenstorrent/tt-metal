# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The MiniMax-H3 server's contract, proved without a Tenstorrent device.

Everything here runs the REAL app: the real routes, the real pydantic models, the real worker thread,
the real queue and the real mp4 mux. Only two seams are stubbed -- `H3Engine._load` (which would open
a mesh) and `H3Engine._generate` (which would run the pipeline) -- and they are stubbed at the
narrowest point that removes the device, so the state machine under test is the shipped one.

That matters because the defects this file is here to catch are all in the glue: an off-by-one
between API `steps` and the scheduler's grid points, a phase that goes backwards, a queue that
accepts one job too many, a cancel that leaves a job `queued` forever.
"""

from __future__ import annotations

import importlib
import json
import sys
import time

import numpy as np
import pytest

SERVER = "models.tt_dit.pipelines.minimax_h3.server"


@pytest.fixture
def server_env(tmp_path, monkeypatch):
    """A p150 server configuration whose weights and adapter are placeholders on disk."""
    adapter = tmp_path / "turbo.safetensors"
    adapter.write_bytes(b"")
    weights = tmp_path / "weights"
    weights.mkdir()
    for name, value in {
        "H3_MESH_SHAPE": "1x1",
        "H3_WEIGHTS_DIR": str(weights),
        "H3_TURBO_PATH": str(adapter),
        "H3_OUT_DIR": str(tmp_path / "out"),
        "H3_MAX_QUEUE": "4",
        "H3_WARMUP": "0",
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("MESH_DEVICE", "H3_API_KEY", "H3_DEFAULTS", "H3_FRAMES_MAX", "TT_DIT_CACHE_DIR"):
        monkeypatch.delenv(name, raising=False)
    yield tmp_path


def _fresh_app(monkeypatch, frame_delay: float = 0.0, steps_seen: list | None = None):
    """Import the app fresh under the current environment, with the device seams stubbed out."""
    for module in [m for m in list(sys.modules) if m.startswith(SERVER)]:
        del sys.modules[module]
    app_module = importlib.import_module(f"{SERVER}.app")
    engine = app_module.ENGINE

    def fake_load(self=engine):
        self.boot_s.update(mesh_open_s=0.0, build_s=0.0, total_s=0.0)
        self.build_info = {"coresident": True, "dit_quant_profile": None, "adapter": "turbo.safetensors"}

    def fake_generate(req, job, self=engine):
        """A deterministic stand-in that fires the same events the pipeline does."""
        on_event = self._events(job)
        events = importlib.import_module("models.tt_dit.pipelines.events")
        for section in ("encoder",) + (("vae_encode",) if req.images else ()):
            on_event(events.SectionStart(section))
            on_event(events.SectionEnd(section))
        on_event(events.SectionStart("denoising"))
        if steps_seen is not None:
            steps_seen.append(req.steps)
        for step in range(req.steps):
            time.sleep(frame_delay)
            on_event(events.DenoiseStep(step=step + 1, total=req.steps, sigma=1.0 - step / req.steps))
        on_event(events.SectionEnd("denoising"))
        for section in ("vae", "audio"):
            on_event(events.SectionStart(section))
            on_event(events.SectionEnd(section))
        rng = np.random.default_rng(req.seed)
        frames = rng.integers(0, 255, (req.num_frames, req.height, req.width, 3), dtype=np.uint8)
        samples = int(req.num_frames / 24 * 32000)
        pcm = (rng.standard_normal((samples, 2)) * 0.05).astype(np.float32)
        return {"frames": frames, "pcm": pcm, "sample_rate": 32000}

    monkeypatch.setattr(engine, "_load", fake_load)
    monkeypatch.setattr(engine, "_generate", fake_generate)
    monkeypatch.setattr(engine, "_close_mesh", lambda: None)
    return app_module


@pytest.fixture
def client(server_env, monkeypatch):
    from fastapi.testclient import TestClient

    app_module = _fresh_app(monkeypatch)
    with TestClient(app_module.app) as c:
        c.app_module = app_module
        yield c


# ---------------------------------------------------------------- pure functions


def test_api_steps_are_forwards_and_the_scheduler_gets_one_more_grid_point():
    """NFE 4 is `num_inference_steps=5`. Off by one here is a silent 25 % quality/latency error."""
    from models.tt_dit.pipelines.minimax_h3.server.config import steps_to_grid_points

    assert [steps_to_grid_points(n) for n in (1, 4, 8, 49)] == [2, 5, 9, 50]


def test_the_servers_frame_snap_is_the_models_frame_snap():
    """`config.align_frames` is a copy kept out of the model's import graph; pin the two together."""
    from models.tt_dit.pipelines.minimax_h3.packing import align_num_frames
    from models.tt_dit.pipelines.minimax_h3.server.config import align_frames

    assert [align_frames(n) for n in range(1, 400)] == [align_num_frames(n) for n in range(1, 400)]
    assert align_frames(22) == 22 and align_frames(23) == 39 and align_frames(124) == 124


@pytest.mark.parametrize(
    "aspect, expected",
    [("16:9", (1344, 768)), ("1:1", (768, 768)), ("9:16", (768, 1344)), ("4:3", (1024, 768))],
)
def test_aspect_ratios_resolve_onto_the_grid_and_inside_the_long_edge(aspect, expected):
    from models.tt_dit.pipelines.minimax_h3.server.config import canvas_for_aspect

    width, height = canvas_for_aspect(aspect, 768)
    assert (width, height) == expected
    assert width % 32 == 0 and height % 32 == 0 and max(width, height) <= 1344


# ---------------------------------------------------------------- validation (422)


@pytest.mark.parametrize(
    "body, why",
    [
        ({"prompt": ""}, "empty prompt"),
        ({"prompt": "   "}, "whitespace-only prompt"),
        ({"prompt": "x" * 8001}, "prompt over 8000 characters"),
        ({"prompt": "x", "width": 100, "height": 544}, "width not a multiple of 32"),
        ({"prompt": "x", "width": 4096, "height": 544}, "long edge over 1344"),
        ({"prompt": "x", "width": 128, "height": 128}, "short edge under 256"),
        ({"prompt": "x", "width": 960}, "width without height"),
        ({"prompt": "x", "mode": "fl2va"}, "fl2va with no image"),
        ({"prompt": "x", "bogus_field": 1}, "unknown field"),
        ({"prompt": "x", "steps": 0}, "steps 0"),
        ({"prompt": "x", "steps": 10_000}, "steps over the ceiling"),
        ({"prompt": "x", "num_frames": 10_000}, "frames over the profile maximum"),
        ({"prompt": "x", "num_frames": 4}, "frames below the minimum"),
        ({"prompt": "x", "num_frames": 22, "duration_seconds": 3}, "both frames and duration"),
        ({"prompt": "x", "seed": -1}, "negative seed"),
        ({"prompt": "x", "aspect_ratio": "banana"}, "unparseable aspect ratio"),
        ({"prompt": "x", "image_prompts": [{"image": "!!!", "position": "first"}]}, "unreadable image"),
    ],
)
def test_bad_requests_are_422(client, body, why):
    assert client.post("/v1/videos/generations", json=body).status_code == 422, why


def test_two_image_prompts_must_have_distinct_positions(client):
    pixel = _png_b64(32, 32)
    body = {
        "prompt": "x",
        "image_prompts": [{"image": pixel, "position": "first"}, {"image": pixel, "position": "first"}],
    }
    assert client.post("/v1/videos/generations", json=body).status_code == 422


def test_duration_seconds_snaps_onto_the_frame_grid(client):
    body = {"prompt": "x", "width": 256, "height": 256, "duration_seconds": 2.0, "steps": 1}
    job = client.post("/v1/videos/generations", json=body).json()
    assert job["request"]["num_frames"] == 56  # round(2 * 24) = 48 -> the next 17n + 5


# ---------------------------------------------------------------- discovery


def test_capabilities_reports_the_profiles_defaults_and_limits(client):
    caps = client.get("/v1/capabilities").json()
    assert caps["modes"] == ["t2va", "fl2va"]
    assert caps["defaults"] == {"width": 1344, "height": 768, "num_frames": 73, "steps": 4, "fps": 24, "seed": 0}
    assert caps["limits"]["multiple_of"] == 32 and caps["limits"]["frame_grid"] == "17n+5"
    assert caps["limits"]["short_edge_max"] == 768 and caps["limits"]["long_edge_max"] == 1344
    assert caps["queue_max"] == 4 and caps["sample_rate"] == 32000 and caps["audio_channels"] == 2
    assert client.get("/capabilities").json() == caps


def test_health_and_models_name_the_profile(client):
    health = client.get("/v1/health").json()
    assert health["ready"] is True and health["profile"] == "p150" and health["mesh"] == "1x1"
    assert client.get("/healthz").json() == {"ok": True, "ready": True}
    data = client.get("/v1/models").json()["data"]
    assert data[0]["id"] == "MiniMaxAI/MiniMax-H3" and data[0]["task"] == "text-image-to-audio-video"


def test_an_unknown_job_is_404_everywhere(client):
    for path in (
        "/v1/videos/generations/vid_nope",
        "/v1/videos/generations/vid_nope/download",
        "/jobs/vid_nope",
        "/result/vid_nope",
    ):
        assert client.get(path).status_code == 404, path
    assert client.delete("/v1/videos/generations/vid_nope").status_code == 404
    assert client.post("/cancel/vid_nope").status_code == 404


# ---------------------------------------------------------------- the happy path


def test_a_job_runs_to_a_downloadable_mp4_with_content_hashes(client):
    body = {"prompt": "a fox", "width": 256, "height": 256, "num_frames": 22, "steps": 2, "seed": 7}
    submitted = client.post("/v1/videos/generations", json=body)
    assert submitted.status_code == 202
    job = _await(client, submitted.json()["id"])
    assert job["status"] == "completed" and job["phase"] == "completed" and job["progress"] == 1.0
    assert job["step"] == job["total_steps"] == 2
    result = job["result"]
    assert (result["width"], result["height"], result["num_frames"]) == (256, 256, 22)
    assert result["fps"] == 24 and result["sample_rate"] == 32000 and result["audio_channels"] == 2
    assert len(result["frames_sha256"]) == 64 and len(result["audio_sha256"]) == 64
    clip = client.get(f"/v1/videos/generations/{job['id']}/download")
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4"
    assert clip.content[4:8] == b"ftyp" and len(clip.content) == result["bytes"]
    assert sum(v for k, v in job["timings"].items() if k != "total_s") <= job["timings"]["total_s"] + 0.01


def test_the_same_request_and_seed_give_the_same_bytes(client):
    body = {"prompt": "a fox", "width": 256, "height": 256, "num_frames": 22, "steps": 1, "seed": 11}
    first = _await(client, client.post("/v1/videos/generations", json=body).json()["id"])["result"]
    second = _await(client, client.post("/v1/videos/generations", json=body).json()["id"])["result"]
    other = _await(client, client.post("/v1/videos/generations", json=dict(body, seed=12)).json()["id"])["result"]
    assert first["frames_sha256"] == second["frames_sha256"]
    assert first["audio_sha256"] == second["audio_sha256"]
    assert first["frames_sha256"] != other["frames_sha256"]


def test_a_download_before_the_job_finishes_is_409(server_env, monkeypatch):
    from fastapi.testclient import TestClient

    app_module = _fresh_app(monkeypatch, frame_delay=0.3)
    body = {"prompt": "x", "width": 256, "height": 256, "num_frames": 22, "steps": 20}
    with TestClient(app_module.app) as client:
        job_id = client.post("/v1/videos/generations", json=body).json()["id"]
        _until(lambda: client.get(f"/v1/videos/generations/{job_id}").json()["phase"] == "denoising")
        assert client.get(f"/v1/videos/generations/{job_id}/download").status_code == 409
        client.delete(f"/v1/videos/generations/{job_id}")


def test_the_job_list_is_newest_first(client):
    body = {"prompt": "x", "width": 256, "height": 256, "num_frames": 22, "steps": 1}
    ids = [client.post("/v1/videos/generations", json=dict(body, seed=s)).json()["id"] for s in (1, 2, 3)]
    for job_id in ids:
        _await(client, job_id)
    listed = [j["id"] for j in client.get("/v1/videos/generations").json()["data"]]
    assert listed[:3] == ids[::-1]


# ---------------------------------------------------------------- queue, 429, cancel


def test_the_queue_accepts_exactly_queue_max_and_then_429s(server_env, monkeypatch):
    from fastapi.testclient import TestClient

    app_module = _fresh_app(monkeypatch, frame_delay=0.4)
    body = {"prompt": "x", "width": 256, "height": 256, "num_frames": 22, "steps": 8}
    with TestClient(app_module.app) as client:
        codes = [client.post("/v1/videos/generations", json=dict(body, seed=k)).status_code for k in range(6)]
        assert codes[:4] == [202] * 4, codes
        assert 429 in codes[4:], codes
        for job in app_module.ENGINE.recent(16):
            app_module.ENGINE.cancel(job)


def test_a_queued_job_cancels_at_once_and_a_running_one_between_steps(server_env, monkeypatch):
    from fastapi.testclient import TestClient

    app_module = _fresh_app(monkeypatch, frame_delay=0.3)
    body = {"prompt": "x", "width": 256, "height": 256, "num_frames": 22, "steps": 20}
    with TestClient(app_module.app) as client:
        running = client.post("/v1/videos/generations", json=dict(body, seed=1)).json()["id"]
        queued = client.post("/v1/videos/generations", json=dict(body, seed=2)).json()["id"]
        _until(lambda: client.get(f"/v1/videos/generations/{running}").json()["phase"] == "denoising")

        assert client.delete(f"/v1/videos/generations/{queued}").json()["status"] == "cancelled"
        cancelled_running = client.delete(f"/v1/videos/generations/{running}").json()
        assert cancelled_running["status"] in ("running", "cancelled")
        final = _await(client, running)
        assert final["status"] == "cancelled" and final["step"] < 20
        assert client.get(f"/v1/videos/generations/{running}/download").status_code == 409
        # The mesh is still serving: a cancel must not take the engine down with it.
        assert client.get("/v1/health").json()["ready"] is True


def test_phase_and_progress_only_move_forward(server_env, monkeypatch):
    from fastapi.testclient import TestClient

    from models.tt_dit.pipelines.minimax_h3.server.engine import PHASES

    app_module = _fresh_app(monkeypatch, frame_delay=0.05)
    body = {"prompt": "x", "width": 256, "height": 256, "num_frames": 22, "steps": 12}
    with TestClient(app_module.app) as client:
        job_id = client.post("/v1/videos/generations", json=body).json()["id"]
        seen = []
        while True:
            view = client.get(f"/v1/videos/generations/{job_id}").json()
            seen.append((PHASES.index(view["phase"]), view["progress"], view["step"]))
            if view["status"] in ("completed", "failed", "cancelled"):
                break
            time.sleep(0.02)
    assert [s[0] for s in seen] == sorted(s[0] for s in seen)
    assert [s[1] for s in seen] == sorted(s[1] for s in seen)
    assert [s[2] for s in seen] == sorted(s[2] for s in seen)
    assert all(0.0 <= s[1] <= 1.0 for s in seen)


# ---------------------------------------------------------------- aliases and auth


def test_the_qb2_aliases_drive_the_same_job(client, tmp_path):
    from PIL import Image

    keyframe = tmp_path / "first.png"
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(keyframe)
    posted = client.post(
        "/generate",
        json={
            "prompt": "x",
            "width": 256,
            "height": 256,
            "num_frames": 22,
            "steps": 1,
            "seed": 4,
            "first_frame": str(keyframe),
        },
    )
    assert posted.status_code == 200
    job_id = posted.json()["job_id"]
    while True:
        alias = client.get(f"/jobs/{job_id}").json()
        assert set(alias) == {"job_id", "status", "phase", "step", "total_steps", "elapsed", "error", "result_path"}
        if alias["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)
    assert alias["status"] == "done"
    assert client.get(f"/v1/videos/generations/{job_id}").json()["request"]["mode"] == "fl2va"
    assert client.get(f"/result/{job_id}").status_code == 200
    assert client.post(f"/cancel/{job_id}").json() == {"cancelled": True}


def test_a_server_side_keyframe_that_is_not_there_is_422(client):
    assert client.post("/generate", json={"prompt": "x", "first_frame": "/no/such.png"}).status_code == 422


def test_the_api_key_guards_every_route_that_is_not_a_health_probe(server_env, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("H3_API_KEY", "s3cret")
    app_module = _fresh_app(monkeypatch)
    with TestClient(app_module.app) as client:
        for path in ("/v1/capabilities", "/v1/models", "/v1/videos/generations"):
            assert client.get(path).status_code == 401, path
        assert client.post("/v1/videos/generations", json={"prompt": "x"}).status_code == 401
        # The alias is the same work, so it is guarded too -- an open /generate would be a bypass.
        assert client.post("/generate", json={"prompt": "x"}).status_code == 401
        assert client.get("/v1/health").status_code == 200
        assert client.get("/healthz").status_code == 200
        auth = {"Authorization": "Bearer s3cret"}
        assert client.get("/v1/capabilities", headers=auth).status_code == 200


# ---------------------------------------------------------------- configuration


def test_h3_defaults_moves_the_served_point_and_rejects_unknown_keys(server_env, monkeypatch, expect_error):
    from models.tt_dit.pipelines.minimax_h3.server.config import build_config

    monkeypatch.setenv("H3_DEFAULTS", json.dumps({"width": 960, "height": 544, "num_frames": 56, "steps": 8}))
    cfg = build_config()
    assert (cfg.width, cfg.height, cfg.num_frames, cfg.steps) == (960, 544, 56, 8)
    monkeypatch.setenv("H3_DEFAULTS", json.dumps({"fps": 30}))
    with expect_error(ValueError, "H3_DEFAULTS"):
        build_config()


def test_each_mesh_shape_resolves_to_its_measured_profile(server_env, monkeypatch):
    from models.tt_dit.pipelines.minimax_h3.server.config import build_config

    monkeypatch.setenv("H3_MESH_SHAPE", "1x4")
    cfg = build_config()
    assert cfg.profile == "p300x2" and (cfg.width, cfg.height, cfg.num_frames) == (1344, 768, 124)
    assert cfg.dit_quant_profile is None and cfg.text_encoder_device == "device"
    assert cfg.precomputed_adaln is False

    monkeypatch.delenv("H3_MESH_SHAPE")
    monkeypatch.setenv("MESH_DEVICE", "P150")
    cfg = build_config()
    assert cfg.profile == "p150" and cfg.dit_quant_profile == "bf8_weights_bf8_out_nofp32acc"
    assert cfg.text_encoder_device == "host" and cfg.precomputed_adaln is True
    assert cfg.video_shift == 6.0 and cfg.audio_shift == 3.0


def test_completed_clips_are_pruned_to_the_retention_limit(server_env, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("H3_OUT_KEEP", "2")
    app_module = _fresh_app(monkeypatch)
    body = {"prompt": "x", "width": 256, "height": 256, "num_frames": 22, "steps": 1}
    with TestClient(app_module.app) as client:
        for seed in range(4):
            _await(client, client.post("/v1/videos/generations", json=dict(body, seed=seed)).json()["id"])
    assert len(list((server_env / "out").glob("vid_*.mp4"))) == 2


# ---------------------------------------------------------------- helpers


def _png_b64(width: int, height: int) -> str:
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _await(client, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        view = client.get(f"/v1/videos/generations/{job_id}").json()
        if view["status"] in ("completed", "failed", "cancelled"):
            return view
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout} s")


def _until(predicate, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition never became true")
