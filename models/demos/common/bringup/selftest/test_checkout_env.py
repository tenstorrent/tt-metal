# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F68: the framework pins TT_METAL_HOME / PYTHONPATH / TT_METAL_CACHE / PYTHON_ENV_DIR to its own checkout."""

from models.demos.common.bringup.core import checkout_env
from models.demos.common.bringup.core.spec import CODE_ROOT


def test_pin_overrides_another_checkout(tmp_path):
    env = {"TT_METAL_HOME": "/other/tt-metal", "TT_METAL_CACHE": "/other/tt-metal/built", "HOME": "/h"}
    said = []
    changed = checkout_env.pin(tmp_path, env=env, echo=said.append)
    assert env["TT_METAL_HOME"] == env["PYTHONPATH"] == str(tmp_path.resolve())
    assert env["TT_METAL_CACHE"] == str(tmp_path.resolve() / "built") and env["HOME"] == "/h"
    assert env["PYTHON_ENV_DIR"] == str(tmp_path.resolve() / "python_env")
    assert changed["TT_METAL_HOME"] == "/other/tt-metal" and "was /other/tt-metal" in said[0]
    assert checkout_env.pin(tmp_path, env=env, echo=said.append) == {} and len(said) == 1


def test_default_root_is_this_checkout():
    assert checkout_env.values()["TT_METAL_HOME"] == str(CODE_ROOT.resolve())
    assert (CODE_ROOT / "tt_metal").is_dir()
