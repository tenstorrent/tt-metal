# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A published package samples decode tokens on the chip when the class it serves says it can, and the
serve gate holds the concurrent per-user speed to the optimized one.

MiniCPM5-2B, 2026-11: its package rode a stock generator that declares supports_sample_on_device, but
the bundle stated no sample_on_device_mode, so the plugin sampled every decode step on the host. 32
users served 54 tok/s/user against ~89 with decode sampled on the chip. The serve gate passed it: it
compared only the 1-user speed with the optimized number, and checked the 32-user run for completion.

No device: the generator module is a source file the reader parses, never imports.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402

_ARCH = "LlamaForCausalLM"


def _generator_source(supports) -> str:
    """A stand-in for the stock generator module, declaring `supports` for every class."""
    body = "\n".join(
        "class %s(Generator):\n    %s = {%r: %r, 'max_device_top_k': 32}\n"
        % (cls, ph._CAPS_ATTR, ph._CAP_SAMPLE_ON_DEVICE, supports)
        for cls in sorted(set(ph._STOCK_GENERATORS.values()))
    )
    return "from vllm.somewhere import Thing\n\n\nclass Generator:\n    pass\n\n\n" + body


def _generator_module(checkout: Path, supports) -> None:
    f = checkout / Path(*ph._GEN_MOD.split(".")).with_suffix(".py")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(_generator_source(supports))


def _serve(tmp_path, slug="m", supports=True, meta_extra=None, adapter_src=None):
    """The serve block publish-hf writes for a stock-arch bundle the tool scaffolds itself."""
    yaml = pytest.importorskip("yaml")
    _generator_module(tmp_path, supports)
    extra = f"models/demos/{slug}/vllm_bundle"
    ph._scaffold_vllm_bundle(tmp_path, extra, _ARCH, None, "org/weights", slug)
    bundle = tmp_path / extra / slug
    if meta_extra is not None:
        meta = json.loads((bundle / "vllm_metadata.json").read_text())
        meta.update(meta_extra)
        (bundle / "vllm_metadata.json").write_text(json.dumps(meta))
    if adapter_src is not None:
        (bundle / "adapter.py").write_text(adapter_src)
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
        extra_models_dir=extra,
        commit=None,
    )
    return yaml.safe_load(out.read_text())["serve"]


_ON = {ph._PLUGIN_CONFIG_KEY: {ph._SAMPLE_MODE_KEY: ph._SAMPLE_MODE_DEFAULT}}


def test_a_stock_generator_that_can_sample_on_chip_is_served_that_way(tmp_path, capsys):
    assert _serve(tmp_path)["additional_config"] == _ON
    assert ph._SAMPLE_MODE_KEY in capsys.readouterr().out, "the default is announced, not silent"


def test_a_generator_that_cannot_gets_no_sampling_setting(tmp_path):
    assert "additional_config" not in _serve(tmp_path, supports=False)


def test_the_authors_mode_is_kept_even_when_it_is_off(tmp_path):
    off = {ph._PLUGIN_CONFIG_KEY: {ph._SAMPLE_MODE_KEY: None}}
    assert _serve(tmp_path, meta_extra={"additional_config": off})["additional_config"] == off


def test_other_settings_are_merged_not_replaced(tmp_path):
    ac = {ph._PLUGIN_CONFIG_KEY: {"trace_region_size": 200000000}, "other": 1}
    got = _serve(tmp_path, meta_extra={"additional_config": ac})["additional_config"]
    assert got == {
        ph._PLUGIN_CONFIG_KEY: {"trace_region_size": 200000000, ph._SAMPLE_MODE_KEY: ph._SAMPLE_MODE_DEFAULT},
        "other": 1,
    }


def test_the_tools_pipeline_adapter_keeps_host_sampling(tmp_path):
    src = ph._pipeline_adapter_src("m", "MForCausalLM", "TT" + _ARCH, "pkg.pipeline", 2048)
    assert "additional_config" not in _serve(tmp_path, adapter_src=src)


def test_a_capability_the_class_itself_declares_wins_over_its_base(tmp_path):
    src = "from %s import %s as _Base\n\n\nclass TT%s(_Base):\n    %s = {%r: False}\n" % (
        ph._GEN_MOD,
        ph._STOCK_GENERATORS[_ARCH],
        _ARCH,
        ph._CAPS_ATTR,
        ph._CAP_SAMPLE_ON_DEVICE,
    )
    assert "additional_config" not in _serve(tmp_path, adapter_src=src)


def test_an_unreadable_capability_is_unknown_not_on(tmp_path):
    src = "class TT%s:\n    %s = dict(**_somewhere)\n" % (_ARCH, ph._CAPS_ATTR)
    assert "additional_config" not in _serve(tmp_path, adapter_src=src)


def test_a_generator_behind_an_import_only_alias_is_found_where_python_finds_it(tmp_path):
    """A package can move and keep its old dotted name as an import alias with no files under it; the
    generator is then read from the file the import system resolves the old name to."""
    old, _, leaf = ph._GEN_MOD.rpartition(".")
    root, _, rest = old.partition(".")
    moved = "%s.moved_%s" % (root, rest.replace(".", "_"))
    real = tmp_path / Path(*moved.split(".")) / (leaf + ".py")
    real.parent.mkdir(parents=True)
    real.write_text(_generator_source(True))
    alias_pkg = tmp_path / Path(*old.split(".")[:2])
    alias_pkg.mkdir(parents=True, exist_ok=True)
    tail = ".".join(old.split(".")[2:])
    (alias_pkg / "__init__.py").write_text(
        "import importlib, sys\n__path__ = []\nsys.modules[__name__ + %r] = importlib.import_module(%r)\n"
        % ("." + tail if tail else "", moved)
    )
    assert not (tmp_path / Path(*ph._GEN_MOD.split("."))).with_suffix(".py").exists()
    assert ph._module_file(tmp_path, ph._GEN_MOD) == real
    caps = ph._class_capabilities(tmp_path, real, ph._STOCK_GENERATORS[_ARCH])
    assert caps and caps[ph._CAP_SAMPLE_ON_DEVICE] is True


def test_a_module_nothing_resolves_is_none(tmp_path):
    assert ph._module_file(tmp_path, "no.such.module") is None
    assert ph._module_file(tmp_path, "nosuchmodule") is None


def _run(users, per_user):
    return {"users": users, "completed": users, "per_user": per_user, "total": users * per_user}


def test_the_gate_fails_a_concurrent_speed_far_under_the_optimized_batch():
    one, many = _run(1, 96.0), _run(32, 54.0)
    problems = ph._serve_gate_problems(99.3, one, many, [], 0.7, optimized_batch=32)
    assert len(problems) == 1 and "32 users at once decode at 54.0" in problems[0], problems
    assert ph._serve_gate_problems(99.3, one, _run(32, 89.0), [], 0.7, optimized_batch=32) == []


def test_the_concurrent_speed_is_compared_only_at_the_batch_it_was_optimized_at():
    one, many = _run(1, 96.0), _run(32, 54.0)
    assert ph._serve_gate_problems(99.3, one, many, [], 0.7) == [], "no batch stated: the old checks only"
    assert ph._serve_gate_problems(99.3, one, many, [], 0.7, optimized_batch=1) == []
    assert ph._serve_gate_problems(None, one, many, [], 0.7, optimized_batch=32) == []
