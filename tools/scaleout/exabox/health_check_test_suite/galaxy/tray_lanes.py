#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""tray_lanes.py: chunk_load.py --trays N[,N..], load only the chosen trays of a
4x8 galaxy while all 32 chips stay enumerated. chunk_load.py opens the full mesh as always, then one submesh per chosen
tray (a "lane": the tray's own 2x4 block, read from the mesh map, never assumed), and runs its load on the lanes only:
the heavy matmul on every chip of a chosen tray, the CCL mix (all_gather / reduce_scatter) inside each tray's own
submesh. Chips of the other trays run no kernels. Standard library only, so it is unit-tested without a device.

Lanes(ttnn, subs) is a stand-in for the ttnn module: an op whose arguments hold a Lane (a submesh, or a tensor, mapper
or event made on the lanes) runs once per lane on that lane's own objects and returns a Lane. get_device_tensors and a
mesh's get_device_ids / get_devices join the lanes in lane order, so chunk_load.py's per-chip index i is the i-th chip
of the chosen trays (lane order = tray order). Names trays only by tray number 1-4."""
import contextlib
import types

TRAYS = (1, 2, 3, 4)
# tray-drop shapes: with only(lanes) a part runs on some lanes. An op on an off lane runs nothing and
# keeps its first Lane argument's object (acc = add(acc, d) keeps that lane's counter); these return None there
# instead, and an op whose Lane argument is None on a lane (an event recorded while it was off) skips that lane.
OFF_NONE = {"record_event", "deallocate", "synchronize_device", "event_synchronize", "wait_for_event"}
ACTIVE = None  # None: every lane; else the lane indices ops run on


@contextlib.contextmanager
def only(lanes):
    """Ops inside run on these lane indices only (None: every lane)."""
    global ACTIVE
    old, ACTIVE = ACTIVE, (None if lanes is None else frozenset(lanes))
    try:
        yield
    finally:
        ACTIVE = old


JOIN = {"get_device_tensors"}  # ttnn functions whose per-lane lists are joined
MESH_JOIN = {"get_device_ids", "get_devices"}  # mesh methods whose per-lane lists are joined
MESH_FIRST = {"compute_with_storage_grid_size", "arch", "num_dram_channels"}  # the same on every lane: lane 0's


def parse_trays(s):
    """'1,2,3' -> [1, 2, 3] (sorted, unique); None or '' -> None (the default: every chip loaded). ValueError otherwise."""
    if s is None or not str(s).strip():
        return None
    out = []
    for x in str(s).replace(" ", "").split(","):
        if not x:
            continue
        if not x.isdigit() or int(x) not in TRAYS:
            raise ValueError(f"--trays takes trays 1-4, comma-separated (got '{s}')")
        out.append(int(x))
    if not out:
        raise ValueError(f"--trays takes trays 1-4, comma-separated (got '{s}')")
    return sorted(set(out))


def tray_boxes(coord_tray, trays):
    """{(r, c): tray or None} for every mesh coordinate -> {tray: (r0, c0, h, w)}, each chosen tray's block. Raises
    ValueError when a chosen tray has no chip in the map or its chips do not fill one rectangle (a submesh needs one).
    """
    out = {}
    for t in trays:
        cs = [rc for rc, tt in coord_tray.items() if tt == t]
        if not cs:
            raise ValueError(
                f"tray {t}: no chip of it in the mesh map (its BDFs unknown: no UMD map, or the tray is not enumerated)"
            )
        r0, r1 = min(r for r, _ in cs), max(r for r, _ in cs)
        c0, c1 = min(c for _, c in cs), max(c for _, c in cs)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h * w != len(cs):
            raise ValueError(
                f"tray {t}: its {len(cs)} chips do not fill one rectangle of the mesh (rows {r0}-{r1}, "
                f"columns {c0}-{c1}): no submesh for it"
            )
        out[t] = (r0, c0, h, w)
    shapes = {(h, w) for _, _, h, w in out.values()}
    if len(shapes) > 1:
        raise ValueError(f"the chosen trays' blocks differ in shape ({sorted(shapes)}): one load shape is needed")
    return out


def describe(trays, chips, all_trays=TRAYS):
    """The log line: the loaded trays with their chips' BDFs, and the trays left enumerated but unloaded."""

    def bdfs(t):
        return (
            " ".join(
                sorted(str(c["pci"]).replace("0000:", "", 1) for c in chips if c.get("tray") == t and c.get("pci"))
            )
            or "?"
        )

    left = [t for t in all_trays if t not in trays]
    s = "chunk_load: loading tray " + "; tray ".join(f"{t} ({bdfs(t)})" for t in trays)
    return s + (f"; tray {','.join(map(str, left))} enumerated, not loaded (no kernels)" if left else "")


class Lane:
    """One object per lane (a submesh, or a tensor, mapper, semaphore or event made on the lanes)."""

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = list(v)

    def __getattr__(self, name):
        xs = [getattr(x, name) for x in self.v]
        if not all(callable(x) for x in xs):
            return xs[0]  # an attribute (shape, dtype): the same on every lane
        if name in MESH_JOIN:
            return lambda *a, **k: [y for f in xs for y in f(*a, **k)]
        if name in MESH_FIRST:
            return lambda *a, **k: xs[0](*a, **k)
        return lambda *a, **k: _fan(xs, a, k, name, self)


def _lanes(a, k):
    n = {len(x.v) for x in list(a) + list(k.values()) if isinstance(x, Lane)}
    for x in list(a) + list(k.values()):
        if isinstance(x, (list, tuple)):
            n |= {len(y.v) for y in x if isinstance(y, Lane)}
    if len(n) > 1:
        raise ValueError(f"lane counts differ: {sorted(n)}")
    return n.pop() if n else 0


def _at(x, i):
    if isinstance(x, Lane):
        return x.v[i]
    if isinstance(x, (list, tuple)) and any(isinstance(y, Lane) for y in x):
        return type(x)(_at(y, i) for y in x)
    return x


def _off(i, a, k):
    """Lane i runs nothing: only() leaves it out, or a Lane argument holds None there."""
    if ACTIVE is not None and i not in ACTIVE:
        return True
    return any(isinstance(x, Lane) and x.v[i] is None for x in list(a) + list(k.values()))


def _kept(i, a, name, obj=None):
    """An off lane's result: None for OFF_NONE and mesh methods, else its first Lane argument's object."""
    if name in OFF_NONE or obj is not None:
        return None
    return next((x.v[i] for x in a if isinstance(x, Lane)), None)


def _fan(fs, a, k, name="", obj=None):
    """fs: one callable per lane (a bound method) or one for all; runs it per lane on that lane's arguments."""
    n = len(fs) if isinstance(fs, list) else _lanes(a, k)
    return Lane(
        _kept(i, a, name, obj)
        if _off(i, a, k)
        else (fs[i] if isinstance(fs, list) else fs)(*[_at(x, i) for x in a], **{kk: _at(v, i) for kk, v in k.items()})
        for i in range(n)
    )


class Fn:
    """A ttnn callable (function, op or class): with a Lane among its arguments it runs per lane, else as is."""

    def __init__(self, f, name):
        self._f, self._name = f, name

    def __getattr__(self, name):
        return wrap(getattr(self._f, name), name)

    def __call__(self, *a, **k):
        if not _lanes(a, k):
            return self._f(*a, **k)
        r = _fan(self._f, a, k, self._name)
        if self._name in JOIN:
            return [y for x in r.v for y in x]
        return r


class Lanes:
    """Stand-in for the ttnn module over the lanes: attributes as ttnn's, callables through Fn."""

    def __init__(self, mod):
        self._m = mod

    def __getattr__(self, name):
        return wrap(getattr(self._m, name), name)


def wrap(v, name):
    if isinstance(v, (types.ModuleType, types.SimpleNamespace)):  # ttnn.experimental
        return Lanes(v)
    if callable(v):
        return Fn(v, name)
    return v
