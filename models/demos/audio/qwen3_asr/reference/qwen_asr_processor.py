# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Load ``Qwen3ASRProcessor`` from the pinned ``qwen-asr`` PyPI package.

The TT port needs exactly one thing from ``qwen-asr``: the processor that builds the
chat prompt and the log-mel features
(``qwen_asr.core.transformers_backend.processing_qwen3_asr``). That module imports only
``numpy`` plus four long-stable ``transformers`` utility modules — it does not touch the
CPU modeling stack.

Importing it the obvious way (``from qwen_asr.core.transformers_backend import
Qwen3ASRProcessor``) executes two package ``__init__`` files that *do* import the CPU
modeling stack (``Qwen3ASRForConditionalGeneration``), which only loads under
``qwen-asr``'s own ``transformers==4.57.6`` pin. tt-metal pins a newer ``transformers``,
so that import fails in the tt-metal environment.

Earlier revisions worked around this by copying the package into the container image and
truncating its ``__init__.py`` files (``server/setup_container.sh``), which meant the
image could not be reproduced from a clean checkout. Instead, load the processing module
directly from its file, registering stub parent packages so the heavy ``__init__`` files
never run. ``qwen-asr`` is then an ordinary pinned dependency, installed with
``--no-deps`` so it cannot disturb tt-metal's own pins:

    pip install --no-deps -r models/demos/audio/qwen3_asr/requirements-processor.txt

The CPU reference/golden tooling (``dump_reference.py``, ``extract_text_decoder.py``,
``prep_wav.py``) *does* need the full package and therefore its own virtualenv — see
``requirements-reference.txt``.
"""
import importlib.util
import os
import sys
import types

PKG = "qwen_asr"
PROCESSING_MOD = "qwen_asr.core.transformers_backend.processing_qwen3_asr"
_SUBPATH = ("core", "transformers_backend", "processing_qwen3_asr.py")

_INSTALL_HINT = (
    "the pinned Qwen3-ASR processor is not installed. Install it with\n"
    "  pip install --no-deps -r models/demos/audio/qwen3_asr/requirements-processor.txt"
)


def _load_isolated():
    """Import the processing module without executing any ``qwen_asr`` ``__init__``."""
    spec = importlib.util.find_spec(PKG)  # locates the package; does not execute it
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(_INSTALL_HINT)
    root = list(spec.submodule_search_locations)[0]
    path = os.path.join(root, *_SUBPATH)
    if not os.path.isfile(path):
        raise ImportError(f"{PKG} is installed at {root} but {os.path.join(*_SUBPATH)} is missing. {_INSTALL_HINT}")

    # Stub the parent packages so ``exec_module`` below resolves the module's relative
    # position without running the real ``__init__`` files (which pull in the CPU modeling
    # stack and its conflicting transformers pin).
    for name, dirpath in (
        (PKG, root),
        (f"{PKG}.core", os.path.join(root, "core")),
        (f"{PKG}.core.transformers_backend", os.path.join(root, "core", "transformers_backend")),
    ):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [dirpath]
            sys.modules[name] = stub

    mod_spec = importlib.util.spec_from_file_location(PROCESSING_MOD, path)
    mod = importlib.util.module_from_spec(mod_spec)
    sys.modules[PROCESSING_MOD] = mod
    mod_spec.loader.exec_module(mod)
    return mod


def load_processor_cls():
    """Return the ``Qwen3ASRProcessor`` class.

    Uses the plain import when the environment can satisfy it (e.g. the reference venv,
    where the full package loads), and falls back to the isolated file import used in the
    tt-metal environment. Raises ``ImportError`` with an install hint if neither works.
    """
    try:
        from qwen_asr.core.transformers_backend import Qwen3ASRProcessor

        return Qwen3ASRProcessor
    except Exception:  # noqa: BLE001
        # Expected in the tt-metal env. The failure is not always an ImportError: with a
        # newer transformers the modeling module fails while *executing* its class bodies
        # (e.g. TypeError from a changed decorator signature), so catch broadly here and
        # let the isolated loader below raise if it genuinely cannot find the package.
        pass

    mod = _load_isolated()
    return mod.Qwen3ASRProcessor
