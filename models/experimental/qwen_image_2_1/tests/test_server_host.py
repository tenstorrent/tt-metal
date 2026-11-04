# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HTTP validation and server shutdown with a small host-only pipeline double."""

import base64
import importlib
import io
import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

server = importlib.import_module("models.experimental.qwen_image_2_1.server.app")


def test_server_repeated_lifespan_stops_workers_and_releases_device(monkeypatch):
    events = []
    monkeypatch.setenv("QWEN_IMAGE_STEPS", "7")

    class Pipeline:
        te_vl = object()

        def is_resident(self, key):
            return False

        def cache_key_for(self, prompt, images):
            return prompt

        def generate(self, prompt, *, seed, num_steps, images):
            events.append((prompt, seed, num_steps))
            rgb = Image.new("RGB", (8, 8), "red")
            return rgb, rgb.convert("RGBA"), None, SimpleNamespace(as_dict=lambda: {"denoise_s": 0.125})

        def release_traces(self):
            events.append("release")

    def load():
        server.STATE.update(ready=True, pipe=Pipeline(), dev="device", info=server.load_config())

    monkeypatch.setattr(server, "_load_models", load)
    monkeypatch.setitem(
        sys.modules,
        "models.experimental.qwen_image_2_1.common.device",
        SimpleNamespace(close_device=lambda device: events.append(("close", device))),
    )
    for _ in range(2):
        with TestClient(server.app) as client:
            response = client.post("/predict", json={"prompt": "red square", "seed": 7, "num_steps": 3})
            assert response.status_code == 200
            result = response.json()
            assert result["seed"] == 7 and result["num_steps"] == 3
            assert result["timing_ms"]["denoise_s"] == 125.0
            assert Image.open(io.BytesIO(base64.b64decode(result["image"]))).size == (8, 8)
            default = client.post("/predict", json={"prompt": "configured default"})
            assert default.status_code == 200 and default.json()["num_steps"] == 7
            assert events[-1] == ("configured default", 42, 7)
            for size in [(1, 5000), (5000, 1)]:
                encoded = io.BytesIO()
                Image.new("RGB", size).save(encoded, format="PNG")
                before = list(events)
                bad_shape = client.post(
                    "/predict",
                    json={"prompt": "extreme ratio", "images": [base64.b64encode(encoded.getvalue()).decode()]},
                )
                assert bad_shape.status_code == 400
                assert "zero dimension" in bad_shape.json()["detail"]
                assert events == before, "rejected image reached the device queue"
            assert client.post("/predict", json={"prompt": "x", "images": ["bad image"]}).status_code == 400
            for steps in (0, 1):
                before = list(events)
                assert client.post("/predict", json={"prompt": "x", "num_steps": steps}).status_code == 422
                assert events == before, "rejected step count reached the device queue"
        assert not any(thread.name == "predict-queue" for thread in threading.enumerate())
        assert events[-2:] == ["release", ("close", "device")]
        assert server.STATE["ready"] is False


def test_queue_rejects_excess_work_and_drains_admitted_requests(monkeypatch):
    monkeypatch.setattr(server, "QUEUE_MAX_PENDING", 1)
    monkeypatch.setitem(server.STATE, "pipe", SimpleNamespace(is_resident=lambda key: False))
    server.STOP.clear()
    started, release = threading.Event(), threading.Event()
    results = []

    def slow_work():
        started.set()
        assert release.wait(timeout=5)
        return "first"

    def request(key, work):
        results.append(server._run_queued(key, work))

    worker = threading.Thread(target=server._queue_worker)
    first = threading.Thread(target=request, args=("first", slow_work))
    second = threading.Thread(target=request, args=("second", lambda: "second"))
    worker.start()
    first.start()
    try:
        assert started.wait(timeout=5)
        second.start()
        with server.QUEUE_COND:
            assert server.QUEUE_COND.wait_for(lambda: len(server.QUEUE) == 1, timeout=5)
        with pytest.raises(server.HTTPException) as error:  # allow-pytest.raises: pure host queue regression.
            server._run_queued("excess", lambda: pytest.fail("rejected work executed"))
        assert error.value.status_code == 503 and error.value.detail == "request queue full"
        with server.QUEUE_COND:
            server.STOP.set()
            server.QUEUE_COND.notify_all()
    finally:
        release.set()
        first.join(timeout=5)
        if second.ident is not None:
            second.join(timeout=5)
        with server.QUEUE_COND:
            server.STOP.set()
            server.QUEUE_COND.notify_all()
        worker.join(timeout=5)
    assert not any(thread.is_alive() for thread in (first, second, worker))
    assert sorted(results) == ["first", "second"]
    assert not server.QUEUE
