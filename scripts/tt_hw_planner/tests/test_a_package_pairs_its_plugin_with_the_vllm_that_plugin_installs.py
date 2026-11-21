# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A container package installs the vLLM release its plugin commit installs, and pins that commit.

MiniCPM5-2B, 2026-11-10: publish-hf wrote `vllm: {version: "0.24.0"}` beside `plugin: {ref: main}`. The
plugin's main had moved to vLLM 0.31.0 and imported vllm.entrypoints.launchers, which 0.24.0 lacks, so the
built image could not start `vllm serve`; the serve gate stopped the upload after the full build. Paired with
0.31.0, the install then resolved vLLM's own torch 2.13.0 and transformers 5.17.0 over the env the model was
validated in (torch 2.11.0, transformers 5.12.1), and tt-model's build check stopped the image.
"""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402


def _git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def _plugin_repo(tmp_path, recipe):
    repo = tmp_path / "plugin"
    (repo / Path(ph._PLUGIN_INSTALL_RECIPE).parent).mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _commit(repo, recipe)
    return repo


def _commit(repo, recipe):
    (repo / ph._PLUGIN_INSTALL_RECIPE).write_text(recipe)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "recipe")
    return _git(repo, "rev-parse", "HEAD")


def _recipe(version):
    return (
        f"curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v{version}/requirements/common.txt\n"
        f"VLLM_TARGET_DEVICE=empty uv pip install --no-deps --no-binary vllm vllm=={version}\n"
        "uv pip install -e .\n"
    )


def test_the_vllm_release_comes_from_the_commit_the_ref_names(tmp_path):
    repo = _plugin_repo(tmp_path, _recipe("0.30.0"))
    first = _git(repo, "rev-parse", "HEAD")
    assert ph._plugin_serving_pair(repo.as_uri(), "main") == (first, "0.30.0")

    moved = _commit(repo, _recipe("0.31.0"))
    assert moved != first
    assert ph._plugin_serving_pair(repo.as_uri(), "main") == (moved, "0.31.0")


def test_a_recipe_names_one_release_or_none_is_chosen():
    assert ph._recipe_vllm_version(_recipe("0.31.0")) == "0.31.0"
    assert ph._recipe_vllm_version(_recipe("0.31.0") + "uv pip install vllm==0.31.0\n") == "0.31.0"
    assert ph._recipe_vllm_version(_recipe("0.30.0") + "uv pip install vllm==0.31.0\n") is None
    assert ph._recipe_vllm_version("uv pip install vllm-tt-plugin==0.1.0\n") is None
    assert ph._recipe_vllm_version("") is None


def test_an_unreadable_plugin_gives_no_pair(tmp_path):
    assert ph._plugin_serving_pair((tmp_path / "missing").as_uri(), "main") is None

    repo = _plugin_repo(tmp_path, "uv pip install -e .\n")
    assert ph._plugin_serving_pair(repo.as_uri(), "main") is None
    assert ph._plugin_serving_pair(repo.as_uri(), "no-such-ref") is None


def _installed(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return None


def test_the_held_versions_are_the_checkout_envs(tmp_path):
    env_bin = tmp_path / "python_env" / "bin"
    env_bin.mkdir(parents=True)
    # A symlink would start the interpreter outside its venv: Python finds pyvenv.cfg beside the path it was run as.
    (env_bin / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (env_bin / "python").chmod(0o755)
    want = [f"{p}=={_installed(p).split('+')[0]}" for p in ph._VALIDATED_PACKAGES if _installed(p)]
    assert ph._validated_pins(tmp_path) == want


def _runtime(tmp_path, overrides):
    yaml = pytest.importorskip("yaml")
    out = tmp_path / "tt-model.yaml"
    ph._write_tt_model_yaml(
        out,
        state={"batch": 32, "env": {"arch": "blackhole"}},
        slug="m",
        repo_id="org/repo",
        checkout=str(tmp_path),
        weights="org/weights",
        box="QB2",
        arch=None,
        hardware=None,
        mesh_device="(1, 1)",
        kind="vllm-plugin",
        plugin_ref="0123abc",
        vllm_version="0.31.0",
        extra_models_dir="models/demos/m/vllm_bundle",
        commit=None,
        overrides=overrides,
    )
    return yaml.safe_load(out.read_text())["runtime"]


def test_the_manifest_carries_the_held_versions(tmp_path):
    pins = ["torch==2.11.0", "transformers==5.12.1"]
    rt = _runtime(tmp_path, pins)
    assert rt["overrides"] == pins
    assert rt["vllm"] == {"version": "0.31.0"}
    assert rt["plugin"] == {"repo": ph._PLUGIN_REPO, "ref": "0123abc"}
    assert "overrides" not in _runtime(tmp_path, None)
