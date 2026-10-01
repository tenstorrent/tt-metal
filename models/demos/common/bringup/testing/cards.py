# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Which cards of THIS box to show a test that needs a given mesh: a TT_VISIBLE_DEVICES value, found, not hardcoded.

A fork test (or a source test a fork carries) needs a mesh of a given shape: 1x1 for a single-device test, 1x4 / 2x2 /
4x2 for the model cases. A box may have more cards than the test needs, and then a test that wants an exact device
count skips, a submesh of the whole box may fail its fabric handshake, and on some boxes even the single-device
fixture fails (an 8 x P150 box has no ClusterType, so ``ttnn.cluster.get_cluster_type()`` throws). So the test is run
with only the cards it needs visible. Which cards form a 2x2 or a ring of 4 depends on how the box is cabled, so it is
probed, not written down:

    from models.demos.common.bringup.testing.cards import visible_devices
    env["TT_VISIBLE_DEVICES"] = visible_devices((2, 2))   # None: show every card

The cards are enumerated from /dev/tenstorrent. For a shape that needs every card, or more cards than there are, the
answer is None (run as before; the test itself decides whether it can run). Otherwise candidate subsets are tried in a
fixed order (one card: the first that opens; k cards: consecutive runs first, then every combination) and each one is
probed under scripts/run_safe_pytest.sh (device lock, reset): the probe reads the system mesh shape the subset forms.
A subset fits when its system mesh is the requested shape or its transpose (open_mesh_device rotates). The answer is
cached per box (host name and card count) in generated/bringup_cards.json, so a box is probed once per shape.

    python -m models.demos.common.bringup.testing.cards 2x2 1x4 1x1    # print (and cache) the subsets
"""

from __future__ import annotations

import itertools
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
CACHE = REPO / "generated" / "bringup_cards.json"
PROBE = Path(__file__).resolve().parent / "card_probe.py"
DEV_DIR = Path("/dev/tenstorrent")
MAX_PROBES = 40  # per shape; consecutive runs come first, so a regular box answers in a few


def num_cards(dev_dir: Path = DEV_DIR) -> int:
    return sum(1 for p in dev_dir.iterdir() if p.name.isdigit()) if dev_dir.is_dir() else 0


def candidates(n: int, k: int):
    """Card subsets of size k out of n, in probe order: consecutive runs, then the rest lexicographically."""
    seen = set()
    for start in range(n - k + 1):
        c = tuple(range(start, start + k))
        seen.add(c)
        yield c
    for c in itertools.combinations(range(n), k):
        if c not in seen:
            yield c


def fits(probed: tuple[int, int] | None, shape: tuple[int, int]) -> bool:
    return probed is not None and tuple(probed) in (tuple(shape), tuple(reversed(shape)))


def _key() -> str:
    return f"{socket.gethostname()}:{num_cards()}"


def _load() -> dict:
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _save(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1) + "\n")


def probe(cards: tuple[int, ...]) -> tuple[int, int] | None:
    """The system mesh shape `cards` form (under the device lock), or None when they do not open."""
    out = REPO / "generated" / f"bringup_card_probe_{os.getpid()}.json"
    out.unlink(missing_ok=True)
    env = dict(os.environ, TT_VISIBLE_DEVICES=",".join(map(str, cards)), BRINGUP_CARD_PROBE_OUT=str(out))
    subprocess.run(
        ["scripts/run_safe_pytest.sh", "--no-precompile", str(PROBE)],
        cwd=REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        got = json.loads(out.read_text())
    except (OSError, ValueError):
        return None
    finally:
        out.unlink(missing_ok=True)
    return tuple(got["shape"]) if got.get("ok") else None


def visible_devices(shape: tuple[int, int], probe_fn=probe) -> str | None:
    """A TT_VISIBLE_DEVICES value showing a subset of this box's cards that forms `shape`, or None for every card."""
    rows, cols = int(shape[0]), int(shape[1])
    n, k = num_cards(), rows * cols
    if n == 0 or k >= n:
        return None
    cache = _load()
    box = cache.setdefault(_key(), {})
    name = f"{rows}x{cols}"
    if name in box:
        return box[name]
    found = None
    for i, c in enumerate(candidates(n, k)):
        if i >= MAX_PROBES:
            break
        got = probe_fn(c)  # one card: any card that opens is a 1x1
        if got is not None and (k == 1 or fits(got, (rows, cols))):
            found = ",".join(map(str, c))
            break
    if found is not None:  # a miss is not cached: it may be a transient failure
        box[name] = found
        _save(cache)
    return found


def main(argv=None) -> int:
    shapes = [tuple(int(x) for x in s.lower().split("x")) for s in (argv if argv is not None else sys.argv[1:])]
    for s in shapes:
        print(f"{s[0]}x{s[1]}: TT_VISIBLE_DEVICES={visible_devices(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
