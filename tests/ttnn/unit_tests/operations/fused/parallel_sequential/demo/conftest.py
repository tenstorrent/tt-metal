# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pathlib

_THIS_DIR = pathlib.Path(__file__).parent


def pytest_ignore_collect(collection_path, config):
    """Skip this directory unless explicitly targeted from the command line."""
    if not collection_path.is_relative_to(_THIS_DIR):
        return None
    args = config.invocation_params.args
    return not any(
        _THIS_DIR == pathlib.Path(a).resolve() or pathlib.Path(a).resolve().is_relative_to(_THIS_DIR) for a in args
    )
