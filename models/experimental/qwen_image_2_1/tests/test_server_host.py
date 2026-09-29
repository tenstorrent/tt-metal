# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""HTTP validation and server shutdown with a small host-only pipeline double."""

import base64
import importlib
import io
import sys
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

server = importlib.import_module("models.experimental.qwen_image_2_1.server.app")


def test_server_repeated_lifespan_stops_workers_and_releases_device(monkeypatch):
    events = []

    class Pipeline:
        te_vl = None

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
        server.STATE.update(ready=True, pipe=Pipeline(), dev="device", info={})

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
            assert client.post("/predict", json={"prompt": "x", "images": ["bad image"]}).status_code == 400
            assert client.post("/predict", json={"prompt": "x", "num_steps": 0}).status_code == 422
        assert not any(thread.name == "predict-queue" for thread in threading.enumerate())
        assert events[-2:] == ["release", ("close", "device")]
        assert server.STATE["ready"] is False
