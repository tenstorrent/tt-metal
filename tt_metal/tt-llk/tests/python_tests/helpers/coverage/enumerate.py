# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Coverage source paths; C++ definitions and domains come from the AST collector."""

from pathlib import Path

ARCH_TREES = {
    "wormhole": "tt_llk_wormhole_b0",
    "blackhole": "tt_llk_blackhole",
    "quasar": "tt_llk_quasar",
}


def canonical_source(source: str, root: Path) -> str:
    path = Path(source)
    if path.parts and path.parts[0] in {*ARCH_TREES.values(), "common"}:
        return path.as_posix()
    try:
        return (
            (path if path.is_absolute() else root / "tests" / path)
            .resolve()
            .relative_to(root.resolve())
            .as_posix()
        )
    except ValueError:
        return path.as_posix()


def headers(root: Path, arch: str) -> set[str]:
    tree = root / ARCH_TREES[arch]
    if not tree.is_dir():
        raise ValueError(f"Architecture source tree does not exist: {tree}")
    return {
        canonical_source(str(path.resolve()), root)
        for directory in (
            tree / "llk_lib",
            tree / "common/inc",
            root / "common",
            root.parent
            / "hw/ckernels"
            / ("wormhole_b0" if arch == "wormhole" else arch)
            / "metal/llk_api/llk_sfpu",
        )
        for path in directory.rglob("*.h")
    }
