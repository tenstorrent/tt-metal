# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The models.tt_transformers import alias for models.ttt_compat.

CPU only. Most cases drive the alias finder against a throwaway package, so they need
neither ttnn nor a device; the last one checks the real package when ttnn is available.
"""

import importlib
import importlib.util
import os
import pickle
import subprocess
import sys
import uuid
from pathlib import Path
from unittest import mock

import pytest

import models.tt_transformers as alias_package

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def aliased_packages(tmp_path, monkeypatch):
    """A namespace package ``<new>.sub.mod`` (like models.ttt_compat) and an alias ``<old>`` for it."""
    suffix = uuid.uuid4().hex[:8]
    old, new = f"alias_old_{suffix}", f"alias_new_{suffix}"
    module_dir = tmp_path / new / "sub"
    module_dir.mkdir(parents=True)
    (module_dir / "mod.py").write_text("class Thing:\n    VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    old_package = type(sys)(old)
    old_package.__path__ = []
    monkeypatch.setitem(sys.modules, old, old_package)
    finder = alias_package._install(old, new)

    yield old, new

    sys.meta_path.remove(finder)
    for name in [n for n in sys.modules if n.startswith((old + ".", new))]:
        del sys.modules[name]


def test_old_and_new_paths_give_the_same_module(aliased_packages):
    old, new = aliased_packages
    assert importlib.import_module(f"{old}.sub.mod") is importlib.import_module(f"{new}.sub.mod")
    assert importlib.import_module(f"{old}.sub") is importlib.import_module(f"{new}.sub")


def test_from_import_and_module_class_strings_resolve_to_the_same_class(aliased_packages):
    old, new = aliased_packages
    namespace = {}
    exec(f"from {old}.sub.mod import Thing", namespace)
    real_thing = importlib.import_module(f"{new}.sub.mod").Thing
    assert namespace["Thing"] is real_thing

    # vLLM registers models as "module:Class" strings and imports them lazily.
    module_name, class_name = f"{old}.sub.mod:Thing".split(":")
    assert getattr(importlib.import_module(module_name), class_name) is real_thing


def test_old_path_import_keeps_the_real_module_names(aliased_packages):
    old, new = aliased_packages
    module = importlib.import_module(f"{old}.sub.mod")  # imported through the alias first
    assert module.__name__ == f"{new}.sub.mod"
    assert module.__spec__.name == f"{new}.sub.mod"
    assert module.Thing.__module__ == f"{new}.sub.mod"
    assert importlib.reload(module) is module


def test_patching_through_the_old_path_patches_the_real_module(aliased_packages):
    old, new = aliased_packages
    real_module = importlib.import_module(f"{new}.sub.mod")
    with mock.patch(f"{old}.sub.mod.Thing.VALUE", 2):
        assert real_module.Thing.VALUE == 2
    assert real_module.Thing.VALUE == 1


def test_pickles_load_under_both_names(aliased_packages):
    old, new = aliased_packages
    thing_class = importlib.import_module(f"{new}.sub.mod").Thing
    assert type(pickle.loads(pickle.dumps(thing_class()))) is thing_class
    # A pickle written before the move names the class by its old module path.
    old_pickle = f"c{old}.sub.mod\nThing\n.".encode()
    assert pickle.loads(old_pickle) is thing_class


def test_missing_modules_stay_missing(aliased_packages, expect_error):
    old, _ = aliased_packages
    assert importlib.util.find_spec(f"{old}.sub.missing") is None
    with expect_error(ModuleNotFoundError, "missing"):
        importlib.import_module(f"{old}.sub.missing")


def test_alias_is_installed_once_and_warns_once():
    finders = [f for f in sys.meta_path if getattr(f, "old", None) == "models.tt_transformers"]
    assert len(finders) == 1

    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "always::DeprecationWarning",
            "-c",
            "import importlib, models.tt_transformers; importlib.import_module('models.tt_transformers')",
        ],
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY_ROOT)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stderr.count("DeprecationWarning: models.tt_transformers is deprecated") == 1


def test_real_package_alias():
    ttnn = pytest.importorskip("ttnn")
    if not hasattr(ttnn, "Tensor"):  # e.g. the source tree's ttnn/ directory without a build
        pytest.skip("ttnn is not built")
    old = importlib.import_module("models.tt_transformers.demo.trace_region_config")
    new = importlib.import_module("models.ttt_compat.demo.trace_region_config")
    assert old is new
    assert old.__spec__.name == "models.ttt_compat.demo.trace_region_config"

    from models.tt_transformers.tt.model_config import ModelArgs as OldModelArgs
    from models.ttt_compat.tt.model_config import ModelArgs

    assert OldModelArgs is ModelArgs
