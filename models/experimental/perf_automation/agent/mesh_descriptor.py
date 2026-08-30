"""Find the fabric mesh-graph descriptor that matches a requested chip count.

WHY THIS EXISTS. Restricting TT_VISIBLE_DEVICES to a subset makes fabric auto-discovery classify the
board as a CUSTOM cluster, which then refuses to initialise without a descriptor:

    Using CUSTOM cluster type for P300 board with 1 chips
    TT_FATAL: Custom fabric mesh graph descriptor path must be specified for CUSTOM cluster type

So visibility was left unrestricted, on the reasoning that the descriptor was something the tool did
not have. tt-metal ships 52 of them. Supplying the matching one alongside the visibility setting is
what makes a subset request actually work, and it matters far more than it sounds:

tt-metal decides a model's chip count with

    {<label table>}.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))

and that table (models/tt_transformers/conftest.py, upstream) lists no Blackhole shape but the
8-chip one -- so on this box every label MISSED and fell through to the number of VISIBLE chips.
Measured 2026-08-29: a demo declaring {"chips": 1, "tp": 1, "mesh": [1, 1]} ran on FOUR chips at 85W
each with the ethernet fabric up, reached 99-103C, and two chips stopped answering. Making
visibility honest fixes that through the SAME fallback, without touching the upstream table -- which
is not ours to patch and is re-merged from upstream regularly anyway.

Descriptors are matched by READING what they declare (arch and device_topology dims), never by
filename, so a new board's descriptor is picked up without editing this file.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

_DESCRIPTOR_REL = ("tt_metal", "fabric", "mesh_graph_descriptors")
_ARCH_RE = re.compile(r"arch:\s*([A-Z_]+)")
_DIMS_RE = re.compile(r"device_topology\s*\{\s*dims:\s*\[([^\]]*)\]")
_MESH_RE = re.compile(r"mesh_descriptors\s*\{")


def _declared(path: Path) -> Optional[Tuple[str, Tuple[int, ...], int]]:
    """(arch, dims, mesh_count) a descriptor declares, or None if it declares none of it."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    arch = _ARCH_RE.search(text)
    dims = _DIMS_RE.search(text)
    if not arch or not dims:
        return None
    try:
        shape = tuple(int(n) for n in re.findall(r"\d+", dims.group(1)))
    except ValueError:
        return None
    if not shape:
        return None
    return arch.group(1).upper(), shape, len(_MESH_RE.findall(text))


def find_descriptor(repo_root, arch: str, chips: int) -> Optional[Path]:
    """A single-mesh descriptor for `chips` chips on `arch`, or None if the repo ships none.

    None is a normal answer, not a failure: the caller must then leave visibility alone rather than
    set it and crash on the CUSTOM-cluster fatal above.
    """
    want = str(arch or "").strip().upper()
    if not want or chips <= 0:
        return None
    root = Path(repo_root).joinpath(*_DESCRIPTOR_REL)
    if not root.is_dir():
        return None
    matches = []
    for path in sorted(root.glob("*.textproto")):
        got = _declared(path)
        if not got:
            continue
        got_arch, shape, meshes = got
        total = 1
        for n in shape:
            total *= n
        # ONE mesh only: a multi-mesh descriptor describes several boards wired together, which is
        # not what "run on N chips of this host" means.
        if meshes == 1 and got_arch == want and total == chips:
            matches.append(path)
    return matches[0] if matches else None
