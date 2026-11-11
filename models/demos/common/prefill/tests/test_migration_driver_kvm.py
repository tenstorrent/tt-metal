# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of the migration driver's tt-d-gen KV Manager mode: one POST /migrate per layer run, against a
stub HTTP server that records the request bodies."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from models.demos.common.prefill.runners.migration_driver import MigrationDriver


def _serve(replies, health=()):
    seen = []
    health = list(health)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, None))
            code = health.pop(0) if health else 200
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            code, reply = replies.pop(0)
            data = json.dumps(reply).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


@pytest.fixture
def kvm_driver(monkeypatch):
    def make(replies, health=(), **env):
        server, seen = _serve(replies, health)
        monkeypatch.setenv("PREFILL_MIGRATION_HANDOFF_TAIL", "1")
        monkeypatch.setenv("PREFILL_MIGRATION_KVM_URL", f"http://127.0.0.1:{server.server_port}/")
        monkeypatch.setenv("PREFILL_MIGRATION_KVM_SESSION_ID", "700")
        monkeypatch.setenv("PREFILL_MIGRATION_TIMEOUT_MS", "10000")
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        driver = MigrationDriver(chunk_size=5120, num_layers=93, default_dst_slot_offset=0)
        driver.attach()
        return driver, seen, server

    yield make


def test_kvm_mode_skips_the_queue_client(kvm_driver):
    driver, _, server = kvm_driver([])
    assert driver.via_kvm and driver.client is None
    server.shutdown()


def test_kvm_migrate_posts_one_layer_run(kvm_driver):
    driver, seen, server = kvm_driver(
        [(200, {"status": "SUCCESSFUL", "duration_ms": 12})], PREFILL_MIGRATION_KVM_RECEIVER="decode-0"
    )
    driver._post_kvm_migrate(
        command_id=3, migration_id=2, src_slot=0, dst_slot=1, layer_begin=3, layer_end=4, pos_begin=0, pos_end=5120
    )
    server.shutdown()
    ((path, body),) = seen
    assert path == "/migrate"
    assert body == {
        "command_session_id": 700,
        "command_id": 3,
        "migration_id": 2,
        "src_slot": 0,
        "dst_slot": 1,
        "layer_begin": 3,
        "layer_end": 4,
        "src_position_begin": 0,
        "src_position_end": 5120,
        "dst_position_begin": 0,
        "dst_position_end": 5120,
        "mode": "full",
        "receiver": "decode-0",
    }


@pytest.mark.parametrize(
    "code, reply",
    [(200, {"status": "FAILED", "reason": "migration_timeout"}), (500, {"error": "boom"}), (200, {})],
)
def test_kvm_migrate_raises_unless_successful(kvm_driver, code, reply):
    driver, _, server = kvm_driver([(code, reply)])
    with pytest.raises(RuntimeError, match="KVM /migrate"):
        driver._post_kvm_migrate(
            command_id=1, migration_id=1, src_slot=0, dst_slot=1, layer_begin=0, layer_end=1, pos_begin=0, pos_end=1
        )
    server.shutdown()


def test_kvm_health_wait_retries_until_200(kvm_driver, monkeypatch):
    monkeypatch.setattr("models.demos.common.prefill.runners.migration_driver.time.sleep", lambda s: None)
    driver, seen, server = kvm_driver([], health=[503, 503, 200])
    driver._wait_kvm_healthy()
    server.shutdown()
    assert [p for p, _ in seen] == ["/health"] * 3


def test_kvm_health_wait_gives_up(kvm_driver, monkeypatch):
    monkeypatch.setattr("models.demos.common.prefill.runners.migration_driver.time.sleep", lambda s: None)
    driver, _, server = kvm_driver([], health=[503] * 50, PREFILL_MIGRATION_KVM_HEALTH_WAIT_S="0")
    with pytest.raises(RuntimeError, match="/health not 200"):
        driver._wait_kvm_healthy()
    server.shutdown()


def test_kvm_migrate_retries_receiver_unready(kvm_driver, monkeypatch):
    monkeypatch.setattr("models.demos.common.prefill.runners.migration_driver.time.sleep", lambda s: None)
    unready = {"status": "FAILED", "reason": "receiver_unready"}
    driver, seen, server = kvm_driver([(500, unready), (500, unready), (200, {"status": "SUCCESSFUL"})])
    driver._post_kvm_migrate(
        command_id=1, migration_id=1, src_slot=0, dst_slot=1, layer_begin=0, layer_end=3, pos_begin=0, pos_end=8
    )
    server.shutdown()
    assert [p for p, _ in seen] == ["/migrate"] * 3
