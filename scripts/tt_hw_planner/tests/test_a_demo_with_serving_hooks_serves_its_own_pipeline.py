# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A demo that declares row-per-request serving hooks is published serving its own optimized pipeline, the build
checks every demo module the served adapter imports, and the card's speed is what `tt serve` measured.

MiniCPM5-2B, 2026-11: a stock Llama arch, so publish-hf scaffolded the plugin's stock generator; the package served
that, not the optimized pipeline the run measured, and the card still showed the pipeline's 99.3 tok/s/user. Its
tt-model.yaml verify only imported the demo package, so an import the pipeline needed (tt-lang's `ttl`, absent from
the image) passed the build. Later the bundle, re-rendered from the hooks on every publish, dropped the prefill token
budget its speed was measured with (--max-num-batched-tokens 4096: 32 prompts of 128 tokens in one step), and 32
users at once served 84 tok/s/user against 92.

No device: the rendered adapter runs against a stand-in pipeline and ttnn.
"""

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402

_STOCK = "LlamaForCausalLM"
_HOOKS = (
    "import torch\n\n"
    "%s = 4\n%s = 64\n%s = {'l1_small_size': 24576}\n\n\n"
    "def %s(device, capacity=None, layers=None):\n    return None\n"
    % (ph._SERVING_ROWS, ph._SERVING_KV, ph._SERVING_DEVICE_PARAMS, ph._SERVING_BUILDER)
)


def _demo(checkout: Path, slug="m", files=None):
    tt = ph._demo_tt_dir(checkout, slug)
    tt.mkdir(parents=True)
    (tt.parent / "__init__.py").write_text("")
    (tt / "__init__.py").write_text("")
    for name, src in (files if files is not None else {"serving.py": _HOOKS}).items():
        (tt / name).write_text(src)
    return slug


def _scaffold(checkout, arch=_STOCK, slug="m"):
    extra = f"models/demos/{slug}/vllm_bundle"
    created, is_stub, bundle = ph._scaffold_vllm_bundle(checkout, extra, arch, None, "org/weights", slug)
    return extra, is_stub, Path(bundle)


def _yaml(checkout, extra, slug="m"):
    yaml = pytest.importorskip("yaml")
    out = checkout / "tt-model.yaml"
    ph._write_tt_model_yaml(
        out,
        state={"batch": 4, "env": {"arch": "blackhole"}},
        slug=slug,
        repo_id="org/repo",
        checkout=str(checkout),
        weights="org/weights",
        box="QB2",
        arch=None,
        hardware=None,
        mesh_device="(1, 1)",
        kind="vllm-plugin",
        plugin_ref="main",
        vllm_version="0.24.0",
        extra_models_dir=extra,
        commit=None,
    )
    return yaml.safe_load(out.read_text())


def test_a_stock_arch_demo_with_serving_hooks_ships_its_own_pipeline(tmp_path):
    _demo(tmp_path)
    extra, is_stub, bundle = _scaffold(tmp_path)
    assert not is_stub
    src = (bundle / "adapter.py").read_text()
    assert ph._rendered_rows_adapter(bundle) and ph._adapter_is_working(bundle)
    assert not ph._adapter_never_traces(bundle)
    assert "_serving_module.%s" % ph._SERVING_BUILDER in src, "the builder the hooks were found by"
    compile(src, "adapter.py", "exec")
    meta = json.loads((bundle / "vllm_metadata.json").read_text())
    assert meta["main_class"] == "adapter:TT" + _STOCK and "serve_hf_override_arch" not in meta
    assert (meta["max_num_seqs"], meta["max_model_len"]) == (4, 64) and "serve_args" not in meta

    m = _yaml(tmp_path, extra)
    assert (m["serve"]["max_num_seqs"], m["serve"]["max_model_len"]) == (4, 64)
    tt = m["serve"]["additional_config"][ph._PLUGIN_CONFIG_KEY]
    assert tt == {"l1_small_size": 24576, ph._SAMPLE_MODE_KEY: ph._SAMPLE_MODE_DEFAULT}, "the adapter samples on chip"
    assert "--hf-overrides" not in m["serve"]["args"], "a built-in arch needs no override"
    assert m["verify"] == ["import models.demos.m as m; assert m", "import " + ph._demo_tt_module("m", "serving")]


def test_a_novel_arch_with_hooks_is_servable_and_names_its_class(tmp_path):
    _demo(tmp_path)
    extra, is_stub, bundle = _scaffold(tmp_path, arch="NewForCausalLM")
    assert not is_stub
    assert json.loads((bundle / "vllm_metadata.json").read_text())["serve_hf_override_arch"] == "TTNewForCausalLM"
    args = _yaml(tmp_path, extra)["serve"]["args"]
    assert json.loads(args[args.index("--hf-overrides") + 1]) == {"architectures": ["TTNewForCausalLM"]}


def test_the_serve_args_the_hooks_declare_reach_the_served_command(tmp_path):
    flags = ["--max-num-batched-tokens", "4096"]
    _demo(tmp_path, files={"serving.py": _HOOKS + "\n%s = %r\n" % (ph._SERVING_SERVE_ARGS, flags)})
    extra, _stub, bundle = _scaffold(tmp_path)
    meta = json.loads((bundle / "vllm_metadata.json").read_text())
    assert meta["serve_args"] == flags and "--max-model-len 64 --max-num-batched-tokens 4096." in meta["note"]
    assert _yaml(tmp_path, extra)["serve"]["args"][-2:] == flags
    _scaffold(tmp_path)
    assert json.loads((bundle / "vllm_metadata.json").read_text())["serve_args"] == flags, "a republish keeps them"


@pytest.mark.parametrize(
    "files, problem",
    [
        ({"serving.py": _HOOKS.replace("%s = 4" % ph._SERVING_ROWS, "")}, ph._SERVING_ROWS),
        ({"serving.py": _HOOKS.replace("%s = 64" % ph._SERVING_KV, "%s = 2 * 32" % ph._SERVING_KV)}, ph._SERVING_KV),
        ({"serving.py": _HOOKS.replace("{'l1_small_size': 24576}", "dict(a=1)")}, ph._SERVING_DEVICE_PARAMS),
        ({"serving.py": _HOOKS + "%s = ['--x', 1]\n" % ph._SERVING_SERVE_ARGS}, ph._SERVING_SERVE_ARGS),
        ({"serving.py": _HOOKS, "other.py": _HOOKS}, "other"),
    ],
)
def test_incomplete_hooks_keep_the_stock_generator_and_say_why(tmp_path, capsys, files, problem):
    _demo(tmp_path, files=files)
    _extra, is_stub, bundle = _scaffold(tmp_path)
    assert problem in capsys.readouterr().out
    assert not is_stub and not ph._rendered_rows_adapter(bundle)
    assert "class TT%s(_Base)" % _STOCK in (bundle / "adapter.py").read_text()


def test_without_hooks_nothing_changes(tmp_path):
    _demo(tmp_path, files={"pipeline.py": "def build_pipeline(device):\n    pass\n"})
    assert ph._demo_serving_hooks(tmp_path, "m") is None
    _extra, is_stub, bundle = _scaffold(tmp_path)
    assert not ph._rendered_rows_adapter(bundle)
    assert "max_num_seqs" not in json.loads((bundle / "vllm_metadata.json").read_text())


def test_only_the_tools_own_rendering_is_re_rendered(tmp_path):
    _demo(tmp_path)
    _extra, _stub, bundle = _scaffold(tmp_path)
    assert ph._rendered_rows_adapter(bundle)
    (bundle / "adapter.py").write_text((bundle / "adapter.py").read_text().replace(ph._ROWS_ADAPTER_MARK, "Mine"))
    assert not ph._rendered_rows_adapter(bundle), "a hand-written adapter is preserved"
    assert not ph._rendered_rows_adapter(tmp_path / "nowhere")


def test_the_build_imports_every_demo_module_the_adapter_imports(tmp_path):
    _demo(tmp_path, files={"pipeline.py": "", "serving.py": ""})
    adapter = tmp_path / "adapter.py"
    adapter.write_text(
        "import torch\n"
        "try:\n"
        "    from models.demos.m.tt.pipeline import build_pipeline\n"
        "    try:\n"
        "        from models.demos.m.tt.pipeline import OPTIONAL_NAME\n"
        "    except Exception:\n"
        "        pass\n"
        "    from models.demos.m.tt import serving\n"
        "    from models.tt_transformers.tt.generator_vllm import LlamaForCausalLM\n"
        "except Exception:\n"
        "    pass\n"
    )
    got = ph._adapter_demo_modules(tmp_path, adapter, "models.demos.m")
    assert got == ["models.demos.m.tt.pipeline", "models.demos.m.tt.serving"], "modules, never optional names"
    assert ph._adapter_demo_modules(tmp_path, tmp_path / "missing.py", "models.demos.m") == []


def _served():
    return {"rows": [(1, 128, 128, 0.2, 10.7, 93.0, 93.0), (32, 128, 128, 0.9, 11.2, 89.0, 2848.0)]}


def test_the_card_speed_is_what_tt_serve_measured(tmp_path):
    old = "99.3 tok/s/user decode (+7% vs baseline) 10.1 ms/token at batch 32 on 1 chip of a p300x2."
    (tmp_path / "tt_kernel_manifest.json").write_text(json.dumps({"container": {"card": {"performance": old + "\n"}}}))
    (tmp_path / "README.md").write_text("## Expected performance\n\n%s\n\n## Limitations\n" % old)
    line = ph._served_headline(_served(), "1 chip of a p300x2")
    assert line == (
        "89.0 tok/s/user with 32 users at once (2,848 tok/s total) and 93.0 tok/s/user with 1 user, served by "
        "`tt serve` on 1 chip of a p300x2 (~128-token prompts, 128 output tokens each, greedy)."
    )
    assert ph._restate_card_speed(tmp_path, line)
    readme = (tmp_path / "README.md").read_text()
    assert line in readme and "99.3" not in readme and "## Limitations" in readme
    assert (
        json.loads((tmp_path / "tt_kernel_manifest.json").read_text())["container"]["card"]["performance"]
        == line + "\n"
    )


def test_a_package_without_a_card_speed_is_left_alone(tmp_path):
    (tmp_path / "tt_kernel_manifest.json").write_text(json.dumps({"container": {}}))
    assert not ph._restate_card_speed(tmp_path, "x")
    assert not ph._restate_card_speed(tmp_path / "missing", "x")


# ---------------------------------------------------------------- the rendered adapter, served by a stand-in


class _Dev:
    """A device tensor: `.cpu()` gives the host copy."""

    def __init__(self, data):
        self.data = data.clone()

    def cpu(self, blocking=True):
        return _Dev(self.data)


class _Mesh:
    def __init__(self):
        self.programs = 0

    def num_program_cache_entries(self):
        return self.programs


class _Pipe:
    """The serving hooks over ROWS device rows: a decode step's ids are each row's input + 1, its logits peak there."""

    batch, pad_id, top_k, V = 4, 1, 3, 8

    def __init__(self, mesh):
        self.mesh, self.config = mesh, SimpleNamespace(vocab_size=self.V, bos_token_id=0)
        self.tok, self.pos = torch.zeros(self.batch, dtype=torch.int32), torch.zeros(self.batch, dtype=torch.int32)
        self.calls, self.shapes = [], set()

    def prefill_rows(self, prompts, rows):
        self.calls.append(("prefill", [len(p) for p in prompts], list(rows)))
        shape = (len(prompts), max(len(p) for p in prompts))
        if shape not in self.shapes:
            self.shapes.add(shape)
            self.mesh.programs += 1
        out = torch.zeros(len(prompts), self.V + 2)  # padded vocab: the adapter cuts it
        for i, p in enumerate(prompts):
            out[i, len(p) % self.V] = 1.0
        return out

    def set_decode_rows(self, tokens=None, positions=None):
        self.calls.append(
            ("rows", None if tokens is None else tokens.tolist(), None if positions is None else positions.tolist())
        )
        if tokens is not None:
            self.tok = tokens.clone()
        if positions is not None:
            self.pos = positions.clone()

    def set_sampling(self, inv_temp, top_k, top_p, seeds):
        self.calls.append(("sampling", inv_temp, top_k, top_p, seeds))

    def _advance(self):
        self.tok, self.pos = (self.tok + 1) % self.V, self.pos + 1

    def decode_tokens(self):
        self.calls.append(("decode", "tokens"))
        self._advance()
        return _Dev(self.tok.reshape(1, -1).to(torch.uint32))  # a device's uint32 ids

    def decode_logits(self):
        self.calls.append(("decode", "logits"))
        logits = torch.nn.functional.one_hot(((self.tok + 1) % self.V).long(), self.V + 2).float()
        self._advance()
        return _Dev(logits)

    def logits_to_torch(self, t):
        return t.data


def _fake_ttnn(log):
    tt = types.ModuleType("ttnn")
    ids = iter(range(100))
    tt.begin_trace_capture = lambda dev, cq_id=0: (log.append(("capture",)), next(ids))[1]
    tt.end_trace_capture = lambda dev, tid, cq_id=0: None
    tt.execute_trace = lambda dev, tid, cq_id=0, blocking=False: log.append(("replay", tid))
    tt.release_trace = lambda dev, tid: log.append(("release", tid))
    tt.record_event = lambda dev, cq: "event"
    tt.get_device_tensors = lambda t: [t]
    tt.to_torch = lambda t: t.data
    return tt


@pytest.fixture
def served(monkeypatch):
    """(generator class, its pipeline, the ttnn log) for the adapter publish-hf renders over a 4-row pipeline."""
    log, mesh = [], _Mesh()
    pipes = []
    hooks_mod = types.ModuleType("fake_serving")
    hooks_mod.build_serving_pipeline = (
        lambda device, capacity=None, layers=None: pipes.append(_Pipe(device)) or pipes[-1]
    )
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn(log))
    monkeypatch.setitem(sys.modules, "fake_serving", hooks_mod)
    src = ph._rows_adapter_src("m", _STOCK, "TT" + _STOCK, {"module": "fake_serving", "rows": 4, "kv": 64})
    mod = types.ModuleType("rendered_adapter")
    exec(compile(src, "adapter.py", "exec"), mod.__dict__)
    gen = mod.__dict__["TT" + _STOCK].initialize_vllm_model(None, mesh, 4, max_seq_len=64)
    return gen, pipes[0], log, mod


def _sp(n, temperature=0.0, top_k=1, top_p=1.0, seed=None, rep=1.0):
    return SimpleNamespace(
        temperature=[temperature] * n,
        top_k=[top_k] * n,
        top_p=[top_p] * n,
        seed=[seed] * n,
        presence_penalty=[0.0] * n,
        frequency_penalty=[0.0] * n,
        repetition_penalty=[rep] * n,
    )


def test_the_generator_refuses_what_the_pipeline_cannot_hold(served, expect_error):
    gen, _pipe, _log, mod = served
    cls = type(gen)
    assert cls.model_capabilities == {"supports_sample_on_device": True, "supports_async_decode": True}
    with expect_error(ValueError, "--max-model-len 64"):
        cls.initialize_vllm_model(None, _Mesh(), 4, max_seq_len=65)
    with expect_error(ValueError, "--max-num-seqs 4"):
        cls.initialize_vllm_model(None, _Mesh(), 5, max_seq_len=64)


def test_each_request_keeps_its_own_row_from_prefill_through_decode(served):
    gen, pipe, log, _mod = served
    pt = torch.tensor([[5, 0], [9, 0]])
    tokens = torch.tensor([[2, 3, 4, 0, 0, 0], [2, 3, 4, 5, 6, 7]])
    logits = gen.prefill_forward(tokens, pt, prompt_lens=[3, 6], empty_slots=[2, 0])
    assert pipe.calls[-1] == ("prefill", [3, 6], [2, 0]) and logits.shape == (2, 1, 8)
    assert [int(x) for x in logits.reshape(2, 8).argmax(-1)] == [3, 6]

    ids = gen.decode_forward(torch.tensor([[7], [4]]), torch.tensor([3, 6]), pt, sampling_params=_sp(2))
    # row 2 holds request 0, row 0 request 1; idle rows 1 and 3 repeat row 2's token and position
    assert ("rows", [4, 7, 7, 7], [6, 3, 3, 3]) in pipe.calls
    assert ("sampling", [1.0] * 4, [1] * 4, [1.0] * 4, [None] * 4) in pipe.calls
    assert ids.tolist() == [0, 5] and ids.dtype == torch.int32, "vLLM's batch order, not device rows"

    pipe.calls.clear()
    gen.decode_forward(torch.tensor([[0], [5]]), torch.tensor([4, 7]), pt, sampling_params=_sp(2))
    assert pipe.calls[0] == ("rows", [5, 0, 0, 0], None), "steady decode: positions are the device's own advance"
    assert not any(c[0] == "sampling" for c in pipe.calls), "an unchanged rule is not rewritten"


def test_decode_replays_one_trace_per_kind_and_a_new_prefill_shape_releases_them(served):
    gen, pipe, log, _mod = served
    gen.warmup_model_prefill(enable_trace=False)
    assert [c[1:] for c in pipe.calls if c[0] == "prefill"] == [
        ([62], [0]),
        ([62] * 2, [0, 1]),
        ([62] * 4, [0, 1, 2, 3]),
    ]
    gen.warmup_model_decode(enable_trace=False)
    gen.warmup_model_decode(enable_trace=True)
    assert log.count(("capture",)) == 2

    pt = torch.tensor([[5, 0]])
    gen.prefill_forward(torch.tensor([[2] * 62]), pt, prompt_lens=[62], empty_slots=[1])
    assert not any(e[0] == "release" for e in log), "a compiled shape keeps the traces"
    pipe.calls.clear()
    gen.decode_forward(torch.tensor([[3]]), torch.tensor([62]), pt, sampling_params=_sp(1), enable_trace=True)
    gen.decode_forward(torch.tensor([[4]]), torch.tensor([63]), pt, sampling_params=_sp(1), enable_trace=True)
    assert [e for e in log if e[0] == "replay"] == [("replay", 0), ("replay", 0)]
    assert not any(c[0] == "decode" for c in pipe.calls), "replayed, not run"

    gen.prefill_forward(torch.tensor([[2] * 5]), pt, prompt_lens=[5], empty_slots=[1])
    assert sorted(e[1] for e in log if e[0] == "release") == [0, 1]
    pipe.calls.clear()
    gen.decode_forward(torch.tensor([[3]]), torch.tensor([5]), pt, enable_trace=True)
    assert log.count(("capture",)) == 4
    assert [c[1] for c in pipe.calls if c[0] == "decode"] == ["tokens", "logits"], "both kinds, in warm-up order"
    assert log[-1] == ("replay", 3), "the step replays its own kind's new trace"


def test_a_step_that_cannot_be_recorded_runs_with_no_trace_held(served):
    gen, pipe, log, _mod = served
    recorded_logits = pipe.decode_logits

    def logits():
        if log[-1:] == [("capture",)]:
            raise RuntimeError("circular buffers clash with L1 buffers")
        return recorded_logits()

    pipe.decode_logits = logits
    gen.warmup_model_prefill(enable_trace=False)
    gen.warmup_model_decode(enable_trace=False)
    gen.warmup_model_decode(enable_trace=True)
    assert log[-2:] == [("capture",), ("release", 1)], "the failed recording is dropped"

    pt = torch.tensor([[5, 0]])
    gen.prefill_forward(torch.tensor([[2] * 62]), pt, prompt_lens=[62], empty_slots=[1])
    gen.decode_forward(torch.tensor([[3]]), torch.tensor([62]), pt, sampling_params=_sp(1), enable_trace=True)
    assert log[-1] == ("replay", 0)
    out = gen.decode_forward(torch.tensor([[4]]), torch.tensor([63]), pt, enable_trace=True)
    assert out.shape == (1, 1, 8) and log[-1] == ("release", 0), "the eager step ran after the held trace went"
    gen.decode_forward(torch.tensor([[5]]), torch.tensor([64]), pt, sampling_params=_sp(1), enable_trace=True)
    assert log[-2:] == [("capture",), ("replay", 2)], "the recordable kind is recorded again"


def test_a_sampled_request_carries_its_rule_and_seed_to_the_device(served):
    gen, pipe, _log, mod = served
    pt = torch.tensor([[5, 0], [9, 0]])
    gen.prefill_forward(torch.tensor([[2, 3], [2, 3]]), pt, prompt_lens=[2, 2], empty_slots=[1, 3])
    sp = _sp(2, temperature=0.5, top_k=0, top_p=0.9)
    sp.seed = [42, None]
    gen.decode_forward(torch.tensor([[1], [1]]), torch.tensor([2, 2]), pt, sampling_params=sp)
    _s, inv, k, p, seeds = [c for c in pipe.calls if c[0] == "sampling"][-1]
    assert (inv[1], k[1], p[1], seeds[1]) == (2.0, pipe.top_k, 0.9, 42), "top_k off: the device's candidates"
    assert seeds[3] is not None and seeds[3] != 42, "an unseeded request draws its own"
    assert seeds[0] is None and seeds[2] is None, "rows with no request stay greedy"


def test_a_penalty_is_sampled_on_the_host_from_the_logits(served):
    gen, pipe, _log, _mod = served
    pt = torch.tensor([[5, 0]])
    gen.prefill_forward(torch.tensor([[2, 3]]), pt, prompt_lens=[2], empty_slots=[0])
    ids = gen.decode_forward(
        torch.tensor([[4]]), torch.tensor([2]), pt, sampling_params=_sp(1, rep=1.2), output_tokens=torch.tensor([[4]])
    )
    assert ("decode", "logits") in pipe.calls and ids.tolist() == [5]
    out = gen.decode_forward(torch.tensor([[5]]), torch.tensor([3]), pt)
    assert out.shape == (1, 1, 8), "no sampling params: vLLM gets the logits"


def test_async_decode_reads_ids_without_waiting_and_feeds_the_next_step_from_the_device(served):
    gen, pipe, _log, mod = served
    pt = torch.tensor([[5, 0]])
    gen.prefill_forward(torch.tensor([[2, 3]]), pt, prompt_lens=[2], empty_slots=[0])
    out = gen.decode_forward(torch.tensor([[4]]), torch.tensor([2]), pt, sampling_params=_sp(1), read_from_device=False)
    pending, events = gen.read_decode_output(out, async_read=True, sample_rows=[0])
    assert events == ["event"] and pending.host is not None
    pipe.calls.clear()
    gen.decode_forward(torch.tensor([[0]]), torch.tensor([2]), pt, sampling_params=_sp(1), reset_batch=False)
    assert pipe.calls[0] == ("rows", None, None), "placeholder ids and a stale position are not written"
    assert gen.process_decode_output_host(pending).tolist() == [5]
