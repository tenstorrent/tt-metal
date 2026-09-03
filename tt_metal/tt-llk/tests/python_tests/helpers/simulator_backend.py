# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Which simulator backend the test session is talking to, and how it differs.

Three backends reach the tests through tt-exalens, and they do not offer the
same device surface:

- ttsim   -- a ``libttsim_*.so``. Models a whole chip, registers included.
- VCS     -- an RTL simulator directory. Real RTL, so it behaves like silicon.
- Versim  -- a build directory holding a ``versim-<arch>`` binary. A functional
             Tensix model reached over NNG. Host reads and writes resolve
             straight into L1, so accesses to the Tensix register aperture
             (0xFFB.....) silently alias instead of reaching the register bus:
             a read returns whatever L1 holds at the aliased offset and a write
             lands somewhere in L1. Anything driven from inside the Tensix (the
             RISCs' own ``reg_read``/``reg_write``) works normally, because that
             never leaves the model.

Only Versim needs the tests to behave differently, so that is the distinction
this module draws. Kept free of ttexalens and helpers imports: conftest has to
consult it before the tt-exalens context exists.
"""

import glob
import os

# TT_METAL_SIMULATOR is the canonical env var; TT_UMD_SIMULATOR_PATH is the
# alias the RTL-simulator workflow already used.
_ENV_VARS = ("TT_METAL_SIMULATOR", "TT_UMD_SIMULATOR_PATH")


def get_simulator_path() -> str | None:
    """Path of the simulator to run against, or None if no env var is set."""
    for name in _ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def is_versim_path(path: str | None) -> bool:
    """Whether ``path`` is a Versim build directory.

    A Versim build directory is what tt-umd-simulators produces: the ``run.sh``
    that UMD spawns, plus the ``versim-<arch>`` executable it launches. Both are
    checked so a VCS directory (which also has ``run.sh``) is not mistaken for
    one.
    """
    if not path or not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "run.sh")):
        return False
    return any(
        os.access(candidate, os.X_OK)
        for candidate in glob.glob(os.path.join(path, "versim-*"))
    )


def is_versim() -> bool:
    """Whether this session's simulator is Versim."""
    return is_versim_path(get_simulator_path())


def runs_in_process(path: str | None) -> bool:
    """Whether tt-exalens should drive ``path`` in-process instead of via a server.

    ttsim always has. Versim joins it for a concrete reason: UMD's simulator
    reset API takes a ``tt_umd.RiscType``, which tt-exalens cannot serialize
    over its Pyro5 server (RiscType is absent from
    ``ttexalens.server.UMD_SERIALIZABLE_TYPES``, so the call fails with
    "unsupported serialized class"). In-process there is no serialization step,
    and UMD spawns the Versim process itself, so no separate server is needed.
    """
    if not path:
        return False
    return path.endswith(".so") or is_versim_path(path)
