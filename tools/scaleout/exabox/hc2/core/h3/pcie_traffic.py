# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""h3/pcie_traffic.py: LTX-like PCIe traffic inside the synthetic H3 steps, and the AER reads around them.

Marginal PCIe links (a box that dropped chips mostly under LTX serving, which moves 100+ GB of weights host to
device at every worker start, plus data every generation) barely show errors under compute-bound H3. So each H3 step
also moves that traffic:
- a weight reload at step start, before the pipeline is built: reload_mb_per_chip on every chip, written in chunk_mb
  chunks and each chunk read back and checked (3328 MB x 32 chips = 112 GB per direction per box);
- per gen: gen_mb_per_chip written to every chip before the gen and read back and checked after it (the data sits in
  the chip's DRAM through the gen's load).
Nothing is stored: every byte comes from a seeded generator on the host at run time. The step's row of h3/pcie-traffic.tsv
('mode seed reload_mb_per_chip gen_mb_per_chip chunk_mb') fixes volume and seed, so every loop and every box moves the
same bytes. A chip's chunk is one seeded random block XOR a key made from (seed, chip, kind, index), so no two chunks or
chips carry the same data and a misrouted or stale transfer shows.
A readback that differs from what was written is a finding: 'H3 PCIE FAIL <bdf> tray <t> ASIC <a> direction=<H2D|D2H|
H2D/D2H> <what> first bad offset=0x<byte>'. The chunk is read once more: the same bad data again means it landed bad in
DRAM (H2D), a clean reread means the readback was bad (D2H), other bad data is H2D/D2H.
'H3 PCIE RELOAD' and 'H3 PCIE GEN <n>' lines give GB per direction per box and per chip, and GB/s per direction.
H3_PCIE_TRAFFIC=0 (h3.sh --no-pcie-traffic) turns it off, with 'H3 PCIE WARNING' lines.

CLI (h3.sh, system python3; reads only): before and after each H3 step, the correctable AER counts (sysfs
aer_dev_correctable) of each chip and of its root port:
  pcie_traffic.py aer-snap > before.tsv
  pcie_traffic.py aer-diff before.tsv after.tsv   'H3 AER ...' lines; a rise during the step is 'H3 AER FINDING' (exit 0)
H3_PCIE_SYSFS (test hook) reads another tree than /sys and says so in a WARNING line.
"""
from __future__ import annotations

import os
import sys
import time

GB = 1e9
MB = 1 << 20
WORDS_PER_ROW = 4096  # uint32 words per row-major row (16 KB pages)
TRAYS = {0x00: 1, 0x40: 2, 0xC0: 3, 0x80: 4}
# The step-start reload stops this long before stop-at, so the pipeline build (185-301 s measured) and a gen start still
# fit in the step (smoke job 193913's 600 s H3 window went 585.8 s to the reload, 11 of 52 chunks, and ran
# no gen). The first chunk always runs, so a short window still moves and checks some reload traffic.
RELOAD_RESERVE_S = 420
OFF_WARNING = (
    "WARNING: H3 PCIe traffic (weight reloads, per-gen writes and checked readbacks) DISABLED by --no-pcie-traffic"
)


# ---- table and generator ----
def read_table(path: str, mode: str) -> dict:
    """The step's row of h3/pcie-traffic.tsv: {'seed', 'reload_mb', 'gen_mb', 'chunk_mb'}."""
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p or p[0].startswith("#") or p[0] != mode:
                continue
            seed, reload_mb, gen_mb, chunk_mb = (int(x) for x in p[1:5])
            if chunk_mb <= 0 or reload_mb < 0 or gen_mb < 0 or reload_mb % chunk_mb:
                raise ValueError(
                    f"{path}: mode {mode}: reload_mb {reload_mb} must be whole chunks of {chunk_mb} MB, gen_mb {gen_mb} >= 0"
                )
            return {"seed": seed, "reload_mb": reload_mb, "gen_mb": gen_mb, "chunk_mb": chunk_mb}
    raise ValueError(f"{path}: no row for mode {mode}")


def _mix(*parts: int) -> int:
    """splitmix64 over the parts, folded to 32 bits (a chunk's XOR key)."""
    x = 0x9E3779B97F4A7C15
    for p in parts:
        x = (x ^ (p & 0xFFFFFFFFFFFFFFFF)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 31
        x = x * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        x ^= x >> 29
    return (x ^ (x >> 32)) & 0xFFFFFFFF or 1


KIND = {"reload": 1, "gen": 2}


class Generator:
    """Seeded payloads: chunk(chip, kind, index) = base(seed) XOR key(seed, chip, kind, index), uint32, words long."""

    def __init__(self, seed: int, words: int):
        import numpy as np

        self.np, self.seed, self.words = np, seed, words
        self.base = np.random.default_rng(seed).integers(0, 1 << 32, size=words, dtype=np.uint32)

    def key(self, chip: int, kind: str, index: int) -> int:
        return _mix(self.seed, chip, KIND[kind], index)

    def chunk(self, chip: int, kind: str, index: int, out=None):
        return self.np.bitwise_xor(self.base, self.np.uint32(self.key(chip, kind, index)), out=out)

    def block(self, n_chips: int, kind: str, index: int):
        """All chips' chunks, chip-major: shape (n_chips * rows, WORDS_PER_ROW)."""
        rows = self.words // WORDS_PER_ROW
        out = self.np.empty((n_chips, self.words), dtype=self.np.uint32)
        for c in range(n_chips):
            self.chunk(c, kind, index, out=out[c])
        return out.reshape(n_chips * rows, WORDS_PER_ROW)


# ---- chips by BDF and tray ----
def bdf_tray_asic(bdf: str) -> tuple[int, int]:
    try:
        bus = int(bdf.split(":")[-2], 16)
    except (ValueError, IndexError):
        return 0, 0
    return TRAYS.get(bus & 0xF0, 0), bus & 0x0F


def name(bdf: str) -> str:
    t, a = bdf_tray_asic(bdf)
    return f"{bdf} tray {t} ASIC {a}" if t else f"{bdf} tray ? ASIC ?"


def sysfs() -> str:
    return os.environ.get("H3_PCIE_SYSFS", "/sys")


def chips_by_dev(root: str | None = None) -> dict[int, str]:
    """/dev/tenstorrent/<n> -> BDF, from /sys/class/tenstorrent."""
    root = root or sysfs()
    out = {}
    d = os.path.join(root, "class/tenstorrent")
    for e in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        n = e.rsplit("!", 1)[-1].rsplit("tenstorrent", 1)[-1]
        if n.isdigit():
            out[int(n)] = os.path.basename(os.path.realpath(os.path.join(d, e, "device")))
    return out


def chip_bdfs(device_ids: list[int], descriptor: str | None, root: str | None = None) -> list[str]:
    """BDF per mesh position. The cluster descriptor maps chip id to its PCIe device number (chips_with_mmio) or bus
    (chip_to_bus_id); a chip neither names is 'chip<id>' (its line then says BDF unknown)."""
    mmio, bus = {}, {}
    if descriptor:
        try:
            import yaml

            with open(descriptor) as f:
                y = yaml.safe_load(f) or {}
            for e in y.get("chips_with_mmio") or []:
                if isinstance(e, dict):
                    mmio.update({int(k): int(v) for k, v in e.items()})
            for k, v in (y.get("chip_to_bus_id") or {}).items():
                bus[int(k)] = int(v, 16) if isinstance(v, str) else int(v)
        except Exception as error:  # noqa: BLE001 - the chips then go by id, and the line says why
            print(
                f"H3 PCIE INFO cluster descriptor {descriptor} unreadable ({type(error).__name__}: {error}); chips by id"
            )
    by_dev = chips_by_dev(root)
    by_bus = {int(b.split(":")[-2], 16): b for b in by_dev.values() if b.count(":") == 2}
    out = []
    for i in device_ids:
        if i in bus and bus[i] in by_bus:
            out.append(by_bus[bus[i]])
        elif i in mmio and mmio[i] in by_dev:
            out.append(by_dev[mmio[i]])
        else:
            out.append(f"chip{i}")
    return out


# ---- the traffic ----
class Traffic:
    """Moves and checks the traffic through a transport: write(block) -> handle, shards(handle) -> one shard per chip
    in chip order, read_shard(shard) -> that chip's words, free(handle). emit gets each line; clock is time.time (tests pass a fake).
    """

    def __init__(self, transport, bdfs: list[str], row: dict, emit, clock=time.time):
        self.t, self.bdfs, self.row, self.emit, self.clock = transport, bdfs, row, emit, clock
        self.n = len(bdfs)
        self.gen = Generator(row["seed"], row["chunk_mb"] * MB // 4)  # reload chunks
        self.gwords = row["gen_mb"] * MB // 4
        self.ggen = Generator(row["seed"] + 1, self.gwords) if self.gwords else None  # per-gen data
        self.fails = []
        self.total = {"H2D": 0.0, "D2H": 0.0}
        self.gen_handle = None

    def _read_check(self, h, expect, what: str) -> tuple[float, bool]:
        """Read h back one chip shard at a time and check it against expect(c), chip c's words (no
        whole-box readback or kept block on the host); a bad shard is read again for the direction. Returns
        (D2H s, ok)."""
        np = self.gen.np
        shards = list(self.t.shards(h))
        if len(shards) != self.n:
            raise RuntimeError(f"{len(shards)} device shards for {self.n} chips")
        d2h, ok = 0.0, True
        for c, shard in enumerate(shards):
            t0 = self.clock()
            g = self.t.read_shard(shard).reshape(-1)
            d2h += self.clock() - t0
            s = expect(c)
            if np.array_equal(s, g):
                continue
            ok = False
            bad = np.flatnonzero(s != g)
            w = int(bad[0])
            a = self.t.read_shard(shard).reshape(-1)
            if np.array_equal(a, s):
                d = "D2H"
            elif np.array_equal(a, g):
                d = "H2D"
            else:
                d = "H2D/D2H"
            line = (
                f"H3 PCIE FAIL {name(self.bdfs[c])} direction={d} {what} first bad offset=0x{w * 4:x}"
                f" (wrote 0x{int(s[w]):08x}, read 0x{int(g[w]):08x}; {bad.size} bad words of {s.size})"
            )
            self.fails.append(line)
            self.emit(line)
        return d2h, ok

    def _round(self, kind: str, index: int, gen: Generator):
        """Write one chunk to every chip, read it back, check it; returns (h2d s, d2h s, ok)."""
        block = gen.block(self.n, kind, index)
        t0 = self.clock()
        h = self.t.write(block)
        t1 = self.clock()
        del block
        d2h, ok = self._read_check(h, lambda c: gen.chunk(c, kind, index), f"{kind} chunk {index}")
        self.t.free(h)
        return t1 - t0, d2h, ok

    def _line(self, head: str, per_chip_bytes: float, h2d_s: float, d2h_s: float, extra: str = "") -> str:
        box = per_chip_bytes * self.n / GB
        self.total["H2D"] += box
        self.total["D2H"] += box
        rate = lambda s: f"{box / s:.2f}" if s > 0 else "-"  # noqa: E731
        return (
            f"{head} H2D {box:.2f} GB box {per_chip_bytes / GB:.3f} GB/chip {rate(h2d_s)} GB/s"
            f" D2H {box:.2f} GB box {per_chip_bytes / GB:.3f} GB/chip {rate(d2h_s)} GB/s chips={self.n}{extra}"
        )

    def reload(self, stop_at: float, reserve: float = 0.0, min_chunks: int = 0) -> bool:
        """The step-start weight reload; stops between chunks at stop_at - reserve, after at least min_chunks chunks.
        False on a readback mismatch."""
        n_chunks = self.row["reload_mb"] // self.row["chunk_mb"]
        start, h2d, d2h, done, ok = self.clock(), 0.0, 0.0, 0, True
        for k in range(n_chunks):
            if k >= min_chunks and self.clock() >= stop_at - reserve:
                break
            a, b, good = self._round("reload", k, self.gen)
            h2d, d2h, done, ok = h2d + a, d2h + b, done + 1, ok and good
        cut = (
            ""
            if done == n_chunks
            else (
                f" cut at stop-at{f' minus {reserve:.0f} s for setup and gens' if reserve else ''}"
                f" after {done} of {n_chunks} chunks"
            )
        )
        self.emit(
            self._line(
                f"H3 PCIE RELOAD {start:.0f} {self.clock() - start:.1f}s seed={self.row['seed']}",
                done * self.gen.words * 4,
                h2d,
                d2h,
                f" chunks={done}/{n_chunks} ok={int(ok)}{cut}",
            )
        )
        for c, b in enumerate(self.bdfs):
            self.emit(
                f"H3 PCIE CHIP {name(b)} reload H2D {done * self.gen.words * 4 / GB:.3f} GB D2H {done * self.gen.words * 4 / GB:.3f} GB"
            )
        return ok

    def before_gen(self, n: int) -> None:
        """Write gen n's input data to every chip; it stays in DRAM through the gen."""
        if not self.gwords:
            return
        block = self.ggen.block(self.n, "gen", n)
        t0 = self.clock()
        self.gen_handle = self.t.write(block)
        self.h2d_s = self.clock() - t0
        del block  # not kept through the gen; after_gen makes each chip's words again

    def after_gen(self, n: int) -> bool:
        """Read gen n's data back and check it. False on a mismatch."""
        if self.gen_handle is None:
            return True
        h = self.gen_handle
        d2h_s, ok = self._read_check(h, lambda c: self.ggen.chunk(c, "gen", n), f"gen {n} data")
        self.t.free(h)
        self.gen_handle = None
        self.emit(
            self._line(
                f"H3 PCIE GEN {n} seed={self.row['seed'] + 1}", self.gwords * 4, self.h2d_s, d2h_s, f" ok={int(ok)}"
            )
        )
        return ok

    def summary(self) -> str:
        return (
            f"H3 PCIE TOTAL H2D {self.total['H2D']:.1f} GB D2H {self.total['D2H']:.1f} GB box, "
            f"{self.total['H2D'] / max(self.n, 1):.2f} GB/chip each way, chips={self.n} fails={len(self.fails)}"
        )


class TtnnTransport:
    """uint32 row-major DRAM tensors, chip-major rows sharded over the mesh in its device order."""

    def __init__(self, ttnn, torch, mesh_device):
        self.ttnn, self.torch, self.mesh = ttnn, torch, mesh_device

    def write(self, block):
        ttnn = self.ttnn
        return ttnn.from_torch(
            self.torch.from_numpy(block.view("int32")),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
        )

    def shards(self, handle):
        return self.ttnn.get_device_tensors(handle)

    def read_shard(self, shard):
        return self.ttnn.to_torch(shard).contiguous().numpy().view("uint32")

    def free(self, handle):
        self.ttnn.deallocate(handle)


def for_mesh(ttnn, torch, mesh_device, mode: str, emit):
    """Traffic for this step (H3_PCIE_TABLE row of mode), or None with a WARNING line when H3_PCIE_TRAFFIC=0."""
    if os.environ.get("H3_PCIE_TRAFFIC", "1") == "0":
        emit(f"H3 PCIE {OFF_WARNING}")
        return None
    row = read_table(os.environ["H3_PCIE_TABLE"], mode)
    ids = list(mesh_device.get_device_ids())
    try:
        desc = ttnn.cluster.serialize_cluster_descriptor()
    except Exception as error:  # noqa: BLE001
        emit(f"H3 PCIE INFO no cluster descriptor ({type(error).__name__}: {error}); chips by id")
        desc = None
    bdfs = chip_bdfs(ids, desc)
    emit(
        f"H3 PCIE PLAN seed={row['seed']} reload {row['reload_mb']} MB/chip in {row['chunk_mb']} MB chunks, gen "
        f"{row['gen_mb']} MB/chip each way, chips={len(ids)}: " + " ".join(f"{i}={b}" for i, b in zip(ids, bdfs))
    )
    return Traffic(TtnnTransport(ttnn, torch, mesh_device), bdfs, row, emit)


# ---- AER (reads only) ----
def aer_file(path: str) -> dict[str, int]:
    out = {}
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) == 2 and p[1].isdigit():
                    out[p[0]] = int(p[1])
    except OSError:
        pass
    return out


def aer_snap(root: str | None = None) -> list[str]:
    """'chip|port <bdf> <port bdf> KEY=N ...' per chip and per root port; 'unread' where the file is missing."""
    root = root or sysfs()
    lines, ports = [], {}
    for _, bdf in sorted(chips_by_dev(root).items()):
        dev = os.path.realpath(os.path.join(root, "bus/pci/devices", bdf))
        port = os.path.basename(os.path.dirname(dev))
        if port.count(":") == 2:
            ports.setdefault(port, os.path.dirname(dev))
        lines.append(_snap_line("chip", bdf, port, os.path.join(dev, "aer_dev_correctable")))
    for port, path in sorted(ports.items()):
        lines.append(_snap_line("port", port, port, os.path.join(path, "aer_dev_correctable")))
    return lines


def _snap_line(kind: str, bdf: str, port: str, path: str) -> str:
    c = aer_file(path)
    return f"{kind}\t{bdf}\t{port}\t" + (" ".join(f"{k}={v}" for k, v in c.items()) if c else "unread")


def _parse_snap(path: str) -> dict[tuple[str, str], tuple[str, dict | None]]:
    out = {}
    with open(path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) != 4:
                continue
            c = None if p[3] == "unread" else {k: int(v) for k, v in (x.split("=", 1) for x in p[3].split())}
            out[(p[0], p[1])] = (p[2], c)
    return out


def aer_diff(before: str, after: str) -> list[str]:
    """'H3 AER' lines: per chip and root port the TOTAL_ERR_COR change by BDF and tray; a rise is 'H3 AER FINDING'."""
    b, a = _parse_snap(before), _parse_snap(after)
    lines, rises, unread = [], 0, 0
    for key in sorted(set(b) | set(a), key=lambda k: (k[0] != "chip", k[1])):
        kind, bdf = key
        port, cb = b.get(key, ("?", None))
        port, ca = a.get(key, (port, None))
        who = f"{kind} {name(bdf)}" if kind == "chip" else f"root port {bdf}"
        if cb is None or ca is None:
            unread += 1
            lines.append(f"H3 AER {who}: correctable count not read {'before' if cb is None else 'after'} the step")
            continue
        d = ca.get("TOTAL_ERR_COR", 0) - cb.get("TOTAL_ERR_COR", 0)
        keys = ",".join(f"{k}+{ca[k] - cb.get(k, 0)}" for k in ca if k != "TOTAL_ERR_COR" and ca[k] > cb.get(k, 0))
        via = f" (root port {port})" if kind == "chip" else ""
        if d > 0:
            rises += 1
            lines.append(
                f"H3 AER FINDING {who}{via}: correctable AER +{d} during the step"
                f" ({cb.get('TOTAL_ERR_COR', 0)} -> {ca.get('TOTAL_ERR_COR', 0)}{'; ' + keys if keys else ''})"
            )
        else:
            lines.append(f"H3 AER {who}{via}: correctable {ca.get('TOTAL_ERR_COR', 0)}, +0 during the step")
    n_chip = sum(1 for k in set(b) | set(a) if k[0] == "chip")
    lines.append(f"H3 AER SUMMARY chips={n_chip} ports={len(set(b) | set(a)) - n_chip} rises={rises} unread={unread}")
    return lines


def main(argv: list[str]) -> int:
    if sysfs() != "/sys":
        print(f"WARNING: AER reads from {sysfs()}, not /sys, DISABLED by H3_PCIE_SYSFS (test hook)")
    if argv[:1] == ["aer-snap"] and len(argv) == 1:
        print("\n".join(aer_snap()))
        return 0
    if argv[:1] == ["aer-diff"] and len(argv) == 3:
        print("\n".join(aer_diff(argv[1], argv[2])))
        return 0
    print("usage: pcie_traffic.py aer-snap | aer-diff <before> <after>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
