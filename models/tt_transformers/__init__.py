# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Deprecated import path: models.tt_transformers moved to models.ttt_compat.

Importing ``models.tt_transformers.<x>`` returns the module object of
``models.ttt_compat.<x>`` itself, so every class keeps a single identity under both
names (isinstance checks, vLLM ``"module:Class"`` registrations, pickles).

Only imports are aliased. This directory holds no other files, so filesystem paths
must use models/ttt_compat/.
"""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

_NEW = "models.ttt_compat"

# Nothing is importable from this directory itself; every submodule goes through the finder.
__path__ = []

warnings.warn(f"{__name__} is deprecated; import {_NEW} instead.", DeprecationWarning, stacklevel=2)


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves ``<old>.<x>`` to the already-importable module ``<new>.<x>``."""

    def __init__(self, old, new):
        self.old = old
        self.new = new
        self._real_specs = {}

    def _target(self, fullname):
        return self.new + fullname[len(self.old) :]

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.old + "."):
            return None
        try:
            if importlib.util.find_spec(self._target(fullname)) is None:
                return None
        except ModuleNotFoundError:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        module = importlib.import_module(self._target(spec.name))
        self._real_specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module):
        # The import system has just overwritten module.__spec__ with the alias spec.
        # Put the real one back so reload() and importlib.resources keep working.
        module.__spec__ = self._real_specs.pop(module.__spec__.name)


def _install(old, new):
    for finder in sys.meta_path:
        if isinstance(finder, _AliasFinder) and (finder.old, finder.new) == (old, new):
            return finder
    finder = _AliasFinder(old, new)
    sys.meta_path.insert(0, finder)
    return finder


_install(__name__, _NEW)
