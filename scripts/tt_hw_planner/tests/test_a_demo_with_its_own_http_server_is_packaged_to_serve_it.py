# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A demo that serves through its own HTTP server is packaged as tt-model's http-server kind, and `tt serve` runs it.

Voxtral-4B-TTS and Voxtral-Mini-3B, 2026-10: speech out and audio in, which the vLLM plugin cannot carry. publish-hf
only knew vLLM packages, so it scaffolded a stub adapter for each; both pulled, and `tt serve` died at model load
(`TTMistralForCausalLM has no attribute 'initialize_vllm_model'`) while the TTS demo shipped a working
/v1/audio/speech server of its own. A demo now declares that server (SERVING_HTTP_COMMAND, SERVING_HTTP_READY_LINE,
SERVING_HTTP_MESH, SERVING_HTTP_PACKAGES) and the package serves it.

No device: the served package is a local fake server.
"""

import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402

_COMMAND = ["python", "-m", "models.demos.tts.serve", "--host", "{host}", "--port", "{port}"]
_HOOKS = "%s = %r\n%s = '[serve] ready'\n%s = '(1, 1)'\n" % (
    ph._SERVING_HTTP_COMMAND,
    _COMMAND,
    ph._SERVING_HTTP_READY,
    ph._SERVING_HTTP_MESH,
)


def _demo(checkout: Path, rel="models/demos/tts", files=None):
    demo = checkout / rel
    (demo / "tt").mkdir(parents=True)
    for name, src in (files if files is not None else {"serve.py": _HOOKS}).items():
        (demo / name).write_text(src)
    return demo


def _yaml(checkout, demo, hooks, packages=()):
    yaml = pytest.importorskip("yaml")
    out = checkout / "tt-model.yaml"
    ph._write_http_server_yaml(
        out,
        state={"env": {"arch": "blackhole"}},
        slug=demo.name,
        repo_id="org/repo",
        checkout=str(checkout),
        demo_rel=demo.relative_to(checkout).as_posix(),
        weights="org/weights",
        box="QB2",
        arch=None,
        hardware=None,
        http=hooks,
        packages=list(packages),
        commit=None,
    )
    return yaml.safe_load(out.read_text())


def test_the_hooks_package_the_demos_own_server_on_the_chips_it_opens(tmp_path):
    demo = _demo(tmp_path)
    hooks = ph._demo_http_server_hooks(demo)
    assert hooks == {
        "module": "models.demos.tts.serve",
        "command": _COMMAND,
        "ready_line": "[serve] ready",
        "mesh": "(1, 1)",
        "packages": [],
    }
    m = _yaml(tmp_path, demo, hooks, ["transformers==4.56.0"])
    assert m["kind"] == ph._HTTP_SERVER_KIND
    assert m["runtime"] == {"command": _COMMAND, "ready_line": "[serve] ready", "packages": ["transformers==4.56.0"]}
    assert (m["serve"]["hardware"], m["serve"]["mesh_device"]) == ("p150", "(1, 1)"), "one chip of the QB2"
    assert not {"block_size", "max_num_seqs", "args", "additional_config"} & set(m["serve"]), "no vLLM settings"
    assert "models/demos/tts" in m["source"]["code"] and m["verify"] == ["import models.demos.tts.serve"]


def test_a_demo_nested_under_models_demos_is_shipped_and_imported_from_where_it_lives(tmp_path):
    demo = _demo(tmp_path, rel="models/demos/hf_eager/voxtral_mini")
    hooks = ph._demo_http_server_hooks(demo)
    assert hooks["module"] == "models.demos.hf_eager.voxtral_mini.serve"
    m = _yaml(tmp_path, demo, hooks)
    assert "models/demos/hf_eager/voxtral_mini" in m["source"]["code"] and "packages" not in m["runtime"]


def test_unpinned_packages_take_the_validated_envs_versions(tmp_path):
    py = tmp_path / "python_env" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
    py.chmod(0o755)
    pinned = ph._pinned_packages(tmp_path, ["pytest", "transformers==4.56.0", "no-such-package-xyz"])
    assert pinned[0] == "pytest==" + pytest.__version__.split("+")[0]
    assert pinned[1:] == ["transformers==4.56.0", "no-such-package-xyz"], "a pin or an unknown package is kept"


@pytest.mark.parametrize(
    "src, problem",
    [
        (_HOOKS.replace("'--port', '{port}'", "'--port', '8000'"), ph._SERVING_HTTP_COMMAND),
        (_HOOKS.replace("'[serve] ready'", "''"), ph._SERVING_HTTP_READY),
        (_HOOKS.replace("'(1, 1)'", "(1, 1)"), ph._SERVING_HTTP_MESH),
        (_HOOKS + "%s = ['ok', 3]\n" % ph._SERVING_HTTP_PACKAGES, ph._SERVING_HTTP_PACKAGES),
    ],
)
def test_incomplete_hooks_are_refused_not_packaged_for_vllm(tmp_path, capsys, src, problem):
    demo = _demo(tmp_path, files={"serve.py": src})
    hooks = ph._demo_http_server_hooks(demo)
    assert problem in hooks["problem"]
    rc = ph._run_container(types.SimpleNamespace(repo="org/repo"), {}, demo.name, demo, None)
    assert rc == 2 and problem in capsys.readouterr().out
    assert not (demo / "vllm_bundle").exists()


def test_two_modules_declaring_a_server_is_ambiguous(tmp_path):
    demo = _demo(tmp_path, files={"serve.py": _HOOKS, "tt/other.py": _HOOKS})
    assert "models.demos.tts.tt.other" in ph._demo_http_server_hooks(demo)["problem"]


def test_a_demo_without_the_hooks_keeps_the_vllm_path(tmp_path):
    demo = _demo(tmp_path, files={"serve.py": "PORT = 8000\n"})
    assert ph._demo_http_server_hooks(demo) is None


def test_the_container_serves_the_hooks_server_with_no_vllm_bundle(tmp_path, monkeypatch):
    demo = _demo(tmp_path)
    seen = {}

    def fake_publish(args, state, slug, checkout, demo_dir, yaml_path, ttm, out, servable, mesh, ready_line=None):
        seen.update(yaml=yaml_path.read_text(), servable=servable, mesh=mesh, ready_line=ready_line, checkout=checkout)
        return 0

    monkeypatch.setattr(ph, "_package_and_publish", fake_publish)
    args = types.SimpleNamespace(repo="org/repo", box="QB2", weights="org/weights")
    assert ph._run_container(args, {"env": {"arch": "blackhole"}}, demo.name, demo, None) == 0
    assert (seen["servable"], seen["mesh"], seen["ready_line"]) == (True, "(1, 1)", "[serve] ready")
    assert "kind: http-server" in seen["yaml"] and seen["checkout"] == tmp_path.resolve()
    assert not (demo / "vllm_bundle").exists(), "no vLLM adapter is scaffolded"


def test_the_gate_passes_a_package_whose_own_server_answers(tmp_path, monkeypatch):
    (tmp_path / "tt_kernel_manifest.json").write_text(
        json.dumps({"name": "tts", "container": {"kind": ph._HTTP_SERVER_KIND, "serve": {}}})
    )
    servers = []

    class Models(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = json.dumps({"data": [{"id": "org/weights"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    def fake_run(cmd, *a, **k):
        if cmd[1] == "serve":
            srv = ThreadingHTTPServer(("127.0.0.1", int(cmd[cmd.index("--port") + 1])), Models)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            servers.append(srv)
        return types.SimpleNamespace(returncode=0, stdout="tt-model-tts-1 Up\n" if "ps" in cmd else "", stderr="")

    monkeypatch.setattr(ph.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(ph.subprocess, "run", fake_run)
    try:
        ok, served = ph._served_speed_gate(types.SimpleNamespace(no_serve_gate=False), {}, tmp_path)
    finally:
        for srv in servers:
            srv.shutdown()
    assert (ok, served) == (True, None), "served, and nothing to time"


def test_the_published_check_waits_for_the_servers_own_ready_line(monkeypatch, capsys):
    log = "[serve] loading\n[serve] ready -- POST /v1/audio/speech\n"

    def fake_run(cmd, *a, **k):
        out = {"ps": "tt-model-tts-1\n", "logs": log}.get(cmd[1], "")
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    monkeypatch.setattr(ph.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(ph.subprocess, "run", fake_run)
    monkeypatch.setattr(ph, "_upload_card_section", lambda *a, **k: None)
    ph._verify_published_package(types.SimpleNamespace(repo="org/tts"), "tts", True, "[serve] ready")
    assert "OK -> PASS (pulled + served: [serve] ready)" in capsys.readouterr().out
