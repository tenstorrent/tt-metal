# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Discovery of compiled compute-kernel ELFs from a tt-metal run.

A TTNN/metal run writes JIT-compiled kernels under
``<cache_root>/<build_key>/kernels/<kernel_name>/<compile_hash>/``. A compute
kernel compiles into three TRISC ELFs (``trisc0/1/2/<trisc>.elf``). This module
locates those triples beneath an arbitrary root, so the analyzer works whether
the user points it at the whole cache, a single build key, or one kernel.

Data-movement kernels (``brisc``/``ncrisc``/``erisc``) and the device firmware
under ``<build_key>/firmware/`` are ignored: only user compute kernels are of
interest, and they are the only ones with ``trisc*`` binaries under ``kernels/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_TRISC_DIRS = ("trisc0", "trisc1", "trisc2")


@dataclass
class KernelArtifacts:
    """Locations of the compiled artifacts for one compute kernel."""

    name: str
    directory: Path
    trisc_elfs: dict[int, Path]  # trisc_id -> elf path
    descriptors_header: Path | None


def discover_compute_kernels(root: str | Path) -> list[KernelArtifacts]:
    """Find all compute-kernel build directories beneath ``root``.

    ``root`` may be the cache root, a ``<build_key>`` directory, a ``kernels``
    directory, a single kernel directory, or a single compile-hash directory.
    """
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    build_dirs = sorted(_find_build_dirs(root))
    return [_build_artifacts(d) for d in build_dirs]


def _find_build_dirs(root: Path) -> set[Path]:
    """Return every directory that directly contains ``triscN/triscN.elf``.

    A build directory is identified by holding at least one ``trisc*`` subdir
    with a matching ELF, and not living under a ``firmware`` path.
    """
    build_dirs: set[Path] = set()
    # If root itself is a build dir, include it directly.
    if _trisc_elfs_in(root):
        build_dirs.add(root)

    for trisc_elf in root.rglob("trisc*/trisc*.elf"):
        if "firmware" in trisc_elf.parts:
            continue
        # build dir is the parent of the trisc dir
        build_dir = trisc_elf.parent.parent
        if _trisc_elfs_in(build_dir):
            build_dirs.add(build_dir)
    return build_dirs


def _trisc_elfs_in(directory: Path) -> dict[int, Path]:
    """Map ``trisc_id -> elf path`` for the trisc ELFs directly in ``directory``."""
    found: dict[int, Path] = {}
    for trisc_id, trisc_dir in enumerate(_TRISC_DIRS):
        elf = directory / trisc_dir / f"{trisc_dir}.elf"
        if elf.is_file():
            found[trisc_id] = elf
    return found


def _build_artifacts(build_dir: Path) -> KernelArtifacts:
    descriptors = build_dir / "chlkc_descriptors.h"
    return KernelArtifacts(
        name=_kernel_name(build_dir),
        directory=build_dir,
        trisc_elfs=_trisc_elfs_in(build_dir),
        descriptors_header=descriptors if descriptors.is_file() else None,
    )


def _kernel_name(build_dir: Path) -> str:
    """Derive a readable kernel name from the build path.

    The canonical layout is ``.../kernels/<kernel_name>/<compile_hash>/``; the
    kernel name is the parent of the (numeric) compile-hash directory.
    """
    parent = build_dir.parent
    if parent.name == "kernels":
        return build_dir.name
    return parent.name
