# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""publish-hf must not upload a package that serves slower than the speed the model was optimized to.

kolibri_1, 2026-10-26: its package started cleanly, passed the post-publish check (which only waited
for "Application startup complete") and served ~9.5 tok/s while its card said 39.47. Nothing timed a
served request. And a republish could reintroduce the same gap three more ways: a hand-written adapter
that never replays a trace (nemotron's on its model branch), serve settings the adapter's speed
depends on dropped from the manifest (on-chip sampling, a bigger trace region), and a card sweep that
ran the perf test at its own default depth (nemotron's: 7 of 52 layers, shown as ~80 tok/s/user).

No device: the server is a local fake that streams tokens at a known rate.
"""

import json
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402

KOLIBRI_META = {
    "arch": "Kolibri1ForCausalLM",
    "main_class": "adapter:TTKolibri1ForCausalLM",
    "max_num_seqs": 32,
    "max_model_len": 2048,
    "serve_hf_override_arch": "TTKolibri1ForCausalLM",
    "additional_config": {"tt": {"sample_on_device_mode": "decode_only"}},
}
NEMOTRON_META = {
    "arch": "NemotronHForCausalLM",
    "main_class": "adapter:TTNemotronHForCausalLM",
    "max_num_seqs": 32,
    "max_model_len": 2048,
    "additional_config": {"tt": {"sample_on_device_mode": "decode_only", "trace_region_size": 200000000}},
    "serve_args": ["--max-num-batched-tokens", "4096"],
}


def _serve_block(tmp_path, slug, meta):
    yaml = pytest.importorskip("yaml")
    bundle = tmp_path / "models" / "demos" / slug / "vllm_bundle" / slug
    bundle.mkdir(parents=True)
    (bundle / "vllm_metadata.json").write_text(json.dumps(meta))
    out = tmp_path / "tt-model.yaml"
    ph._write_tt_model_yaml(
        out,
        state={"batch": 32, "env": {"arch": "blackhole"}},
        slug=slug,
        repo_id="org/repo",
        checkout=str(tmp_path),
        weights="org/weights",
        box="QB2",
        arch=None,
        hardware=None,
        mesh_device="(1, 4)",
        kind="vllm-plugin",
        plugin_ref="main",
        vllm_version="0.24.0",
        extra_models_dir=f"models/demos/{slug}/vllm_bundle",
        commit=None,
    )
    return yaml.safe_load(out.read_text())["serve"]


def test_a_republish_keeps_the_serve_settings_kolibri_was_published_with(tmp_path):
    serve = _serve_block(tmp_path, "kolibri_1", KOLIBRI_META)
    assert serve["max_num_seqs"] == 32 and serve["max_model_len"] == 2048
    assert serve["additional_config"] == {"tt": {"sample_on_device_mode": "decode_only"}}
    assert serve["args"] == ["--trust-remote-code", "--hf-overrides", '{"architectures": ["TTKolibri1ForCausalLM"]}']


def test_a_republish_keeps_the_serve_settings_nemotron_was_published_with(tmp_path):
    serve = _serve_block(tmp_path, "nemotron", NEMOTRON_META)
    assert serve["max_model_len"] == 2048
    assert serve["additional_config"]["tt"] == {"sample_on_device_mode": "decode_only", "trace_region_size": 200000000}
    assert serve["args"] == ["--trust-remote-code", "--max-num-batched-tokens", "4096"]


def test_an_adapter_without_settings_gets_no_additional_config(tmp_path):
    serve = _serve_block(tmp_path, "plain", {"arch": "X", "main_class": "adapter:TTX", "max_num_seqs": 1})
    assert "additional_config" not in serve and serve["args"] == ["--trust-remote-code"]


EAGER_HAND_WRITTEN = (
    "class TTNemotronHForCausalLM:\n"
    "    def decode_forward(self, tokens, *, enable_trace=False, **_):\n"
    "        return self.pipe.decode_step(self._state)\n"
)


def test_an_adapter_that_decodes_but_never_replays_a_trace_is_caught(tmp_path):
    (tmp_path / "adapter.py").write_text(EAGER_HAND_WRITTEN)
    assert ph._adapter_never_traces(tmp_path)

    (tmp_path / "adapter.py").write_text(
        ph._pipeline_adapter_src("m", "MForCausalLM", "TTMForCausalLM", "pkg.pipeline", 2048)
    )
    assert not ph._adapter_never_traces(tmp_path), "the tool's own template replays its trace"

    (tmp_path / "adapter.py").write_text("class TTLlamaForCausalLM(_Base):\n    pass\n")
    assert not ph._adapter_never_traces(tmp_path), "a stock generator subclass inherits the stock traced decode"


def test_every_eager_warning_the_template_logs_is_recognised():
    src = ph._pipeline_adapter_src("m", "MForCausalLM", "TTMForCausalLM", "pkg.pipeline", 2048)
    warnings = [ln for ln in src.splitlines() if "much slower" in ln]
    assert len(warnings) >= 2
    for w in warnings:
        assert ph._eager_serve_lines(w), w
    assert not ph._eager_serve_lines("m: decode trace captured; every decode token replays it")
    # kolibri's and nemotron's hand-written adapters word it their own way
    assert ph._eager_serve_lines("kolibri_1: logits decode trace capture failed (oom); those steps decode eagerly")


def test_the_gate_blocks_a_package_serving_at_a_quarter_of_its_optimized_speed():
    one = {"users": 1, "completed": 1, "per_user": 9.5}
    problems = ph._serve_gate_problems(39.47, one, None, [], 0.7)
    assert problems and "9.5" in problems[0] and "39.5" in problems[0]


def test_the_gate_passes_a_package_serving_near_its_optimized_speed():
    one = {"users": 1, "completed": 1, "per_user": 38.5}
    many = {"users": 32, "completed": 32, "per_user": 34.3}
    assert ph._serve_gate_problems(39.47, one, many, [], 0.7) == []
    assert ph._serve_gate_problems(None, one, many, [], 0.7) == [], "no optimized figure: speed is not compared"


def test_the_gate_blocks_eager_decode_dropped_requests_and_a_dead_server():
    one = {"users": 1, "completed": 1, "per_user": 38.5}
    assert ph._serve_gate_problems(39.47, one, None, ["m: decoding eagerly, which is much slower"], 0.7)
    many = {"users": 32, "completed": 20, "errors": ["HTTP Error 500"]}
    assert "20 of 32" in ph._serve_gate_problems(39.47, one, many, [], 0.7)[0]
    assert ph._serve_gate_problems(39.47, {"users": 1, "completed": 0, "error": "refused"}, None, [], 0.7)


def test_the_gate_refuses_rather_than_skips_when_it_cannot_serve(tmp_path, monkeypatch):
    monkeypatch.setattr(ph.shutil, "which", lambda name: None)
    ok, served = ph._served_speed_gate(types.SimpleNamespace(no_serve_gate=False), {}, tmp_path)
    assert (ok, served) == (False, None)
    ok, served = ph._served_speed_gate(types.SimpleNamespace(no_serve_gate=True), {}, tmp_path)
    assert (ok, served) == (True, None)


class _Streamer(BaseHTTPRequestHandler):
    gap = 0.01

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.05)
        for i in range(body["max_tokens"]):
            if i:
                time.sleep(self.gap)
            self.wfile.write(b'data: {"choices": [{"text": "x"}]}\n\n')
            self.wfile.flush()
        usage = {"choices": [], "usage": {"completion_tokens": body["max_tokens"]}}
        self.wfile.write(b"data: " + json.dumps(usage).encode() + b"\n\ndata: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Streamer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_served_decode_is_timed_per_user_from_the_token_stream(server):
    r = ph._time_served_decode(server, "m", 4, osl=32)
    assert r["completed"] == 4
    assert 30 < r["per_user"] < 150, r  # one token every ~10 ms, loosely: localhost TCP batches small writes
    assert r["total"] == pytest.approx(4 * r["per_user"])
    assert r["ttft"] >= 0.05


def test_the_card_sweep_runs_the_full_model_not_the_tests_default(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 52}))
    monkeypatch.setenv("TT_PERF_LAYERS", "7")
    args = types.SimpleNamespace(bench_layers=None, weights=None, token=None)
    assert ph._bench_layers_env(args, tmp_path) == {"TT_PERF_LAYERS": "52"}
    args.bench_layers = 4
    assert ph._bench_layers_env(args, tmp_path) == {"TT_PERF_LAYERS": "4"}


def test_a_multimodal_config_gives_its_text_depth(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 30}}))
    assert ph._full_depth(tmp_path, None, None) == 30


def test_the_card_headline_is_the_served_speed_once_it_was_measured():
    state = {
        "batch": 32,
        "throughput": {"current": 39.47},
        "served": {
            "max_num_seqs": 32,
            "rows": [(1, 128, 128, 0.14, 20.8, 48.1, 48.1), (32, 128, 128, 0.9, 29.2, 34.3, 1094)],
        },
    }
    block = ph._tt_validated_block(state, "Tenstorrent p300x2", None)
    assert "| Served, 1 user (ISL 128 / OSL 128) | 48.1 tok/s/user" in block
    assert "| Served, 32 users at once" in block and "1094 tok/s total" in block
    assert "| Optimized decode (on-device perf test, batch 32) | 39.47 tok/s/user |" in block
    assert "| Decode throughput |" not in block
