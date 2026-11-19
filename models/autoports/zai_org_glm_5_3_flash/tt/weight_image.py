# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DRAM-image weight load for GLM-5.3-Flash (Stage 12 fast start).

A normal (tensor) load leaves every chip's weights in DRAM at addresses that depend only on the sequence of DRAM
allocations of the load (per-submesh allocators are deterministic: tt_metal FreeListOpt, first fit, DRAM bottom-up;
program binaries are allocated top-down). This module turns that into a bulk load:

* record: the normal load runs with the two weight-load primitives wrapped (`FdTileCacheSource._load(device=)` for
  every tile-cache tensor, `fused_decoder.load_fused_experts` for the 288 x 2 routed-expert tensors per MoE layer);
  every loaded tensor's key, spec and DRAM address is logged per chip, in order.
* image: after the load, every device tensor the model holds (object-graph walk) is mapped to its per-bank extent
  [address, address + aligned_size_per_bank) (interleaved and sharded DRAM buffers both reserve that range in every
  bank); extents are merged into segments and `ttnn._ttnn.dram_image.read_image` dumps them bank-major to
  `<root>/<code id>-<weights id>/chip<N>.img` with per-GiB checksums, plus `manifest.json` (identities, segments,
  checksums) and `chip<N>.map.json` (load log + tensor map).
* replay (GLM53_WEIGHT_SOURCE=image): the same constructor code runs with the primitives replaced by
  `ttnn.allocate_tensor_on_device` (no file read, no upload); every allocation is checked against the recorded address
  as it happens; device ops of the constructor (pads, reshards, typecasts, slices) run as usual on uninitialised
  inputs, so the allocator history -- and hence every address -- is identical. Meanwhile one `ImagePuller` per chip
  pins 1 GiB hugepage slots and reads the image file ahead; after the constructor, all chips pull in parallel (one
  thread per chip, GIL released): Tensix cores read the slots over PCIe and write the DRAM segments, overwriting the
  constructor's uninitialised outputs with the recorded bytes. The final tensor map and the checksums are compared
  with the manifest; any mismatch (code / weights identity, allocation address, tensor map, checksum, missing file,
  hugepages) tears the replay down and falls back to the tensor loader.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import subprocess
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable, Optional

import ttnn

FORMAT = "glm53-dram-image/1"
DEFAULT_ROOT = "/mnt/fab/mpi-shfs/tenstorrent/glm53-flash-bringup-data/dram_images"
CODE_PATHS = (
    "models/autoports/zai_org_glm_5_3_flash/tt",
    "ttnn",
    "tt_metal",
    "models/autoports/zai_org_glm_5_3_flash/tests/common.py",
    "models/autoports/zai_org_glm_5_3_flash/tests/common_optimized.py",
)
HUGEPAGE_FREE = "/sys/kernel/mm/hugepages/hugepages-1048576kB/free_hugepages"


class ImageMismatch(RuntimeError):
    """The image cannot be used for this process (identity, allocation, map, checksum or resource mismatch)."""


def log(msg: str) -> None:
    print(f"[fast-start] {time.strftime('%H:%M:%S', time.gmtime())}Z {msg}", flush=True)


def _di():
    return ttnn._ttnn.dram_image


# ================================================================================================ identity
def _git(root: str, *args: str) -> bytes:
    return subprocess.run(["git", "-C", root, *args], check=True, capture_output=True).stdout


def code_identity(root: Optional[str] = None) -> dict:
    """Key of an image = the git TREE hashes of what defines the weight layout (CODE_PATHS: the model's tt/ package, ttnn,
    tt_metal, plus the two test modules on the load path) + a hash of their uncommitted changes (tracked diff and
    untracked files under tt/). Not the HEAD sha: docs / work-log / unrelated commits must not invalidate an image set
    (Stage 9 requirement 2026-10-19 07:02Z). HEAD and the build fingerprint (sizes / mtimes of the compiled libraries)
    are recorded for diagnosis only."""
    root = (
        root or os.environ.get("TT_METAL_HOME") or os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
    )
    head = _git(root, "rev-parse", "HEAD").decode().strip()
    trees = _git(root, "rev-parse", *[f"HEAD:{p}" for p in CODE_PATHS]).decode().split()
    h = hashlib.sha256()
    diff = _git(root, "diff", "HEAD", "--binary", "--", *CODE_PATHS)
    h.update(diff)
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z", "--", CODE_PATHS[0]).split(b"\0")
    for p in sorted(u for u in untracked if u and not u.endswith(b".pyc")):
        h.update(p)
        with open(os.path.join(root, p.decode()), "rb") as f:
            h.update(f.read())
    dirty = h.hexdigest()[:10] if (diff or any(untracked)) else ""
    key = hashlib.sha256(" ".join(trees).encode()).hexdigest()[:12]
    build = {}
    for rel in ("ttnn/ttnn/_ttnn.so", "build/lib/libtt_metal.so"):
        try:
            st = os.stat(os.path.join(root, rel))
            build[rel] = [st.st_size, int(st.st_mtime)]
        except OSError:
            build[rel] = None
    return {
        "head": head,
        "trees": dict(zip(CODE_PATHS, trees)),
        "dirty": dirty,
        "id": f"t{key}" + (f"-d{dirty}" if dirty else ""),
        "build": build,
    }


def weights_identity(manifest_path: Optional[str] = None) -> dict:
    """sha256 of the weight-cache manifest (every tile-cache file is listed there with its own sha256; the fused-experts
    permuted cache is a byte permutation keyed by those hashes)."""
    if manifest_path is None:
        from models.autoports.zai_org_glm_5_3_flash.tests.common import MANIFEST as manifest_path
    with open(manifest_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"manifest": manifest_path, "sha256": sha, "id": sha[:12]}


DEFAULT_LAYOUT = "mk-ds-v1"  # Stage 7a contract 2026-10-19 06:59Z: the production layout is the DS width-sharded one (op-chain-v0 = same bytes)


def layout_version(manifest_path: Optional[str] = None) -> str:
    """Weight layout the loader places on device: $GLM53_LAYOUT_VERSION, else the weight-cache manifest's
    "layout_version", else the model code's WEIGHT_LAYOUT_VERSION, else 'op-chain-v0'. Part of the image key: an image
    is never replayed against a build that places another layout."""
    v = os.environ.get("GLM53_LAYOUT_VERSION")
    if v:
        return v
    try:
        if manifest_path is None:
            from models.autoports.zai_org_glm_5_3_flash.tests.common import MANIFEST as manifest_path
        with open(manifest_path) as f:
            head = f.read(4096)
        if '"layout_version"' in head:
            return json.loads(open(manifest_path).read()).get("layout_version", DEFAULT_LAYOUT)
    except (OSError, ValueError):
        pass
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt import optimized_decoder as od

        return getattr(od, "WEIGHT_LAYOUT_VERSION", DEFAULT_LAYOUT)
    except ImportError:
        return DEFAULT_LAYOUT


def image_dir(code: dict, weights: dict, root: Optional[str] = None, layout: Optional[str] = None) -> str:
    layout = layout or layout_version(weights.get("manifest"))
    return os.path.join(
        root or os.environ.get("GLM53_IMAGE_ROOT", DEFAULT_ROOT), f"{code['id']}-{weights['id']}-{layout}"
    )


# ================================================================================================ tensor map
def chip_of(mesh) -> int:
    ids = list(mesh.get_device_ids())
    if len(ids) != 1:
        raise ImageMismatch(f"weight tensors must live on 1x1 submeshes (got devices {ids})")
    return int(ids[0])


_SKIP_TYPES = (str, bytes, int, float, bool, type(None), type)


def device_tensors(root, prefix: str = "be") -> list:
    """(path, tensor) for every allocated device tensor reachable from `root` through model objects (classes from
    models.*), SimpleNamespace / dataclass instances, lists, tuples and dicts; deterministic order (insertion order of
    attribute dicts and containers); each object visited once."""
    out, seen = [], set()

    def visit(obj, path):
        if isinstance(obj, _SKIP_TYPES) or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, ttnn.Tensor):
            if obj.storage_type() == ttnn.StorageType.DEVICE and obj.is_allocated():
                out.append((path, obj))
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                visit(v, f"{path}[{k!r}]")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]")
        elif isinstance(obj, SimpleNamespace) or type(obj).__module__.startswith("models."):
            d = getattr(obj, "__dict__", None)
            if d is not None:
                for k, v in list(d.items()):
                    visit(v, f"{path}.{k}")

    visit(root, prefix)
    return out


def tensor_map(root) -> dict:
    """chip -> {"tensors": [...], "l1": [...], "mesh": submesh} of the model's device tensors."""
    di = _di()
    chips: dict = {}
    for path, t in device_tensors(root):
        info = di.buffer_info(t)
        if info["num_devices"] != 1:
            raise ImageMismatch(f"{path}: tensor on a {info['num_devices']}-device mesh (images are per chip)")
        c = chips.setdefault(info["device_id"], {"tensors": [], "l1": [], "mesh": t.device()})
        row = {
            "path": path,
            "address": int(info["address"]),
            "extent": int(info["aligned_size_per_bank"]),
            "page": int(info["aligned_page_size"]),
            "layout": info["layout"],
            "shape": list(t.shape),
            "dtype": str(t.dtype),
        }
        (c["tensors"] if info["dram"] else c["l1"]).append(row)
    return chips


def merge_segments(rows: list) -> list:
    """Per-bank extents [address, address + extent) of the DRAM tensors -> sorted, merged [lo, hi) segments."""
    ext = sorted({(r["address"], r["address"] + r["extent"]) for r in rows})
    segs: list = []
    for lo, hi in ext:
        if segs and lo <= segs[-1][1]:
            segs[-1][1] = max(segs[-1][1], hi)
        else:
            segs.append([lo, hi])
    return segs


def comparable(rows: list) -> list:
    return [(r["path"], r["address"], r["extent"], r["shape"], r["dtype"]) for r in rows]


# ================================================================================================ record / replay
class WeightIO:
    """Context manager wrapping the two weight-load primitives.

    mode "record": loads run normally; every device load is logged per chip: key, address, spec.
    mode "replay": loads allocate an uninitialised tensor with the recorded spec instead; the address must equal the
    recorded one (else ImageMismatch at that call). `expected` = {chip: [load rows]} from chip<N>.map.json."""

    def __init__(self, mode: str, expected: Optional[dict] = None, host_tensors: Optional[dict] = None):
        if mode not in ("record", "replay"):
            raise ValueError(mode)
        self.mode = mode
        self.expected = {int(k): v for k, v in (expected or {}).items()}
        # small host-side weights (FdTileCacheSource.host: mHC bases / scales, dt_bias, A_log, router bias): recorded
        # into the image (host_tensors.pt) so a replay opens no weight-cache file at all
        self.host_tensors: dict = dict(host_tensors or {})
        self.host_misses = 0
        self.loads: dict = {}
        self._chip: dict = {}
        self._patched: list = []
        self.alloc_s = 0.0
        self.calls = 0

    # ---- bookkeeping
    def _chip_id(self, mesh) -> int:
        k = id(mesh)
        if k not in self._chip:
            self._chip[k] = chip_of(mesh)
        return self._chip[k]

    def _note(self, mesh, key: str, t) -> None:
        chip = self._chip_id(mesh)
        seq = self.loads.setdefault(chip, [])
        row = {"key": key, "address": int(t.buffer_address()), "spec": _spec_str(t)}
        if self.mode == "replay":
            exp = self.expected.get(chip)
            i = len(seq)
            if exp is None or i >= len(exp):
                raise ImageMismatch(f"chip {chip}: load #{i} {key} has no recorded counterpart")
            e = exp[i]
            if (e["key"], e["address"], e["spec"]) != (row["key"], row["address"], row["spec"]):
                raise ImageMismatch(f"chip {chip}: load #{i} {key} @ {row['address']:#x} {row['spec']} != recorded {e}")
        seq.append(row)
        self.calls += 1

    def _alloc(self, mesh, shape, dtype, layout, memory_config):
        t0 = time.perf_counter()
        t = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, mesh, memory_config)
        self.alloc_s += time.perf_counter() - t0
        return t

    # ---- patches
    def __enter__(self):
        from models.autoports.zai_org_glm_5_3_flash.tt import fused_decoder as fd
        from models.autoports.zai_org_glm_5_3_flash.tt import optimized_decoder as od

        io = self
        orig_load = od.FdTileCacheSource._load
        orig_host = od.FdTileCacheSource.host
        orig_experts = fd.load_fused_experts

        def host(src, layer_idx, name):
            key = f"layers.{layer_idx}.{name}"
            if io.mode == "replay" and key in io.host_tensors:
                return io.host_tensors[key].clone()
            t = orig_host(src, layer_idx, name)
            if io.mode == "record":
                io.host_tensors[key] = t.clone()
            else:
                io.host_misses += 1
            return t

        def _load(src, e, device=None):
            if device is None:
                return orig_load(src, e)
            if io.mode == "record":
                t = orig_load(src, e, device)
            else:
                t = io._alloc(
                    device,
                    e["shape"],
                    getattr(ttnn, e["dtype"]),
                    getattr(ttnn, f"{e['layout']}_LAYOUT"),
                    ttnn.DRAM_MEMORY_CONFIG,
                )
            io._note(device, e["path"], t)
            return t

        def load_fused_experts(source, layer_idx, num_experts, mesh_device, cache_dir):
            if io.mode == "record" or not isinstance(source, fd.TileCacheSource):
                gate_up, down = orig_experts(source, layer_idx, num_experts, mesh_device, cache_dir)
            else:
                gate_up, down = [], []
                gu_cfg = dn_cfg = None
                for e in range(num_experts):
                    ge = source._entry(layer_idx, f"mlp.experts.{e}.gate_up_proj")
                    de = source._entry(layer_idx, f"mlp.experts.{e}.down_proj")
                    assert ge["dtype"] == "bfloat4_b" and de["dtype"] == "bfloat4_b", (ge["dtype"], de["dtype"])
                    if gu_cfg is None:
                        gu_cfg = fd._nd_dram(ge["shape"][0], ge["shape"][1] // fd.FE_CORES)
                        dn_cfg = fd._nd_dram(de["shape"][0], de["shape"][1] // fd.FE_CORES)
                    gate_up.append(io._alloc(mesh_device, ge["shape"], ttnn.bfloat4_b, ttnn.TILE_LAYOUT, gu_cfg))
                    down.append(io._alloc(mesh_device, de["shape"], ttnn.bfloat4_b, ttnn.TILE_LAYOUT, dn_cfg))
            # log in allocation order (gate_up e, down e, gate_up e + 1, ...)
            for e, (gu, dn) in enumerate(zip(gate_up, down)):
                io._note(mesh_device, f"L{layer_idx}.e{e}.gate_up", gu)
                io._note(mesh_device, f"L{layer_idx}.e{e}.down", dn)
            return gate_up, down

        od.FdTileCacheSource._load = _load
        od.FdTileCacheSource.host = host
        fd.load_fused_experts = load_fused_experts
        self._patched = [
            (od.FdTileCacheSource, "_load", orig_load),
            (od.FdTileCacheSource, "host", orig_host),
            (fd, "load_fused_experts", orig_experts),
        ]
        return self

    def __exit__(self, *exc):
        for obj, name, orig in self._patched:
            setattr(obj, name, orig)
        self._patched = []
        return False

    def check_complete(self) -> None:
        if self.mode != "replay":
            return
        for chip, exp in self.expected.items():
            got = len(self.loads.get(chip, []))
            if got != len(exp):
                raise ImageMismatch(f"chip {chip}: replay made {got} loads, image recorded {len(exp)}")


def _spec_str(t) -> str:
    return f"{list(t.shape)}|{t.dtype}|{t.layout}|{t.memory_config()}"


# ================================================================================================ build
def build_image(backend, io: WeightIO, out_dir: str, *, code: dict, weights: dict, signature: dict, threads: int = 8):
    """Dump every chip's model tensors to out_dir (chips in parallel). `io` is the WeightIO("record") the backend was
    built under. Returns the manifest dict (also written to out_dir/manifest.json, last, atomically)."""
    di = _di()
    t0 = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    chips = tensor_map(backend)
    man = {
        "format": FORMAT,
        "layout_version": layout_version(weights.get("manifest")),
        "code": code,
        "weights": weights,
        "signature": signature,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chips": {},
    }
    results: dict = {}
    errors: list = []

    def dump(chip, c):
        try:
            segs = merge_segments(c["tensors"])
            path = os.path.join(out_dir, f"chip{chip}.img")
            r = di.read_image(c["mesh"], [tuple(s) for s in segs], path + ".tmp", threads=threads)
            os.replace(path + ".tmp", path)
            results[chip] = (segs, r)
        except BaseException as e:  # noqa: BLE001
            errors.append((chip, e))

    th = [threading.Thread(target=dump, args=(chip, c)) for chip, c in sorted(chips.items())]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise RuntimeError(f"image dump failed: {errors}")
    for chip, c in sorted(chips.items()):
        segs, r = results[chip]
        mp = {"chip": chip, "loads": io.loads.get(chip, []), "tensors": c["tensors"], "l1": c["l1"]}
        blob = json.dumps(mp, separators=(",", ":")).encode()
        with open(os.path.join(out_dir, f"chip{chip}.map.json"), "wb") as f:
            f.write(blob)
        man["chips"][str(chip)] = {
            "image": f"chip{chip}.img",
            "map": f"chip{chip}.map.json",
            "map_sha256": hashlib.sha256(blob).hexdigest(),
            "segments": segs,
            "bytes": int(r["bytes"]),
            "checksums": [int(x) for x in r["checksums"]],
            "block_bytes": int(r["block_bytes"]),
            "block_hashes": [int(x) for x in r["block_hashes"]],
            "loads": len(mp["loads"]),
            "tensors": len(c["tensors"]),
            "l1_tensors": len(c["l1"]),
            "dump_s": round(float(r["total_s"]), 3),
            "dram_read_s": round(float(r["dram_read_s"]), 3),
        }
        log(
            f"image chip {chip}: {len(segs)} segments, {r['bytes'] / 1e9:.2f} GB, {len(c['tensors'])} tensors, "
            f"{len(mp['loads'])} loads, {len(c['l1'])} L1 tensors, dump {r['total_s']:.1f} s"
        )
    import torch

    hp = os.path.join(out_dir, "host_tensors.pt")
    torch.save(io.host_tensors, hp + ".tmp")
    os.replace(hp + ".tmp", hp)
    with open(hp, "rb") as f:
        man["host_tensors"] = {
            "file": "host_tensors.pt",
            "count": len(io.host_tensors),
            "sha256": hashlib.sha256(f.read()).hexdigest(),
        }
    man["build_s"] = round(time.perf_counter() - t0, 3)
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))
    return man


# ================================================================================================ load
def find_image(code: dict, weights: dict, root: Optional[str] = None) -> Optional[tuple]:
    layout = layout_version(weights.get("manifest"))
    d = image_dir(code, weights, root, layout)
    p = os.path.join(d, "manifest.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        man = json.load(f)
    if man.get("format") != FORMAT:
        raise ImageMismatch(f"{p}: format {man.get('format')} != {FORMAT}")
    if man["code"]["id"] != code["id"] or man["weights"]["sha256"] != weights["sha256"]:
        raise ImageMismatch(f"{p}: identity {man['code']['id']}/{man['weights']['id']} != {code['id']}/{weights['id']}")
    if man.get("layout_version") != layout:
        raise ImageMismatch(f"{p}: layout_version {man.get('layout_version')} != {layout} (refusing to replay)")
    return d, man


def load_maps(d: str, man: dict) -> dict:
    maps = {}
    for chip, c in man["chips"].items():
        with open(os.path.join(d, c["map"]), "rb") as f:
            blob = f.read()
        if hashlib.sha256(blob).hexdigest() != c["map_sha256"]:
            raise ImageMismatch(f"chip {chip}: map file hash mismatch")
        maps[int(chip)] = json.loads(blob)
    return maps


def load_host_tensors(d: str, man: dict) -> dict:
    h = man.get("host_tensors")
    if not h:
        return {}
    import io as _io

    import torch

    with open(os.path.join(d, h["file"]), "rb") as f:
        blob = f.read()
    if hashlib.sha256(blob).hexdigest() != h["sha256"]:
        raise ImageMismatch("host_tensors.pt hash mismatch")
    return torch.load(_io.BytesIO(blob))


def free_hugepages() -> int:
    try:
        with open(HUGEPAGE_FREE) as f:
            return int(f.read())
    except OSError:
        return 0


def device_hashes(man: dict, submeshes: dict) -> dict:
    """chip -> on-device block hashes of the chip's image segments (all chips in parallel threads; ~0.1 s / GB)."""
    di = _di()
    out, errors = {}, []

    def run(chip, c):
        try:
            r = di.device_block_hashes(submeshes[chip], [tuple(x) for x in c["segments"]], block_bytes=c["block_bytes"])
            out[chip] = [int(x) for x in r["block_hashes"]]
        except BaseException as e:  # noqa: BLE001
            errors.append((chip, repr(e)))

    th = [threading.Thread(target=run, args=(int(k), c)) for k, c in man["chips"].items()]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise ImageMismatch(f"device block hashes failed: {errors}")
    return out


def stale_blocks(man: dict, hashes: dict) -> dict:
    """chip -> indices of the blocks whose device hash differs from the image's."""
    return {
        chip: [i for i, (a, b) in enumerate(zip(h, man["chips"][str(chip)]["block_hashes"])) if a != b]
        for chip, h in hashes.items()
    }


def start_pullers(
    d: str,
    man: dict,
    submeshes: dict,
    *,
    slots=None,
    cores=None,
    threads=None,
    direct=None,
    checksum=None,
    only_blocks: Optional[dict] = None,
):
    """One started ImagePuller per chip of the manifest (`submeshes`: chip id -> 1x1 submesh). only_blocks: chip -> block
    indices to pull (None / missing chip = the whole image; [] = nothing for that chip)."""
    di = _di()
    n = len(man["chips"])
    if slots is None:
        # leave 2 free 1 GiB pages (a chip open takes one for its sysmem): 3 slots per chip on one chip, 2 on eight
        slots = int(os.environ.get("GLM53_IMAGE_SLOTS", "0")) or max(1, min(3, (free_hugepages() - 2) // max(1, n)))
    if slots * n > free_hugepages():
        raise ImageMismatch(f"need {slots * n} free 1 GiB hugepages, {free_hugepages()} free")
    cores = cores or int(os.environ.get("GLM53_IMAGE_CORES", "32"))
    threads = threads or int(os.environ.get("GLM53_IMAGE_READ_THREADS", "16"))
    direct = bool(int(os.environ.get("GLM53_IMAGE_DIRECT", "0"))) if direct is None else direct
    checksum = bool(int(os.environ.get("GLM53_IMAGE_CHECKSUM", "1"))) if checksum is None else checksum
    # page-cache budget: GLM53_IMAGE_CACHE_GB (all chips) -> the first budget / n bytes of each chip image are read
    # buffered (they stay resident for the next start), the rest with O_DIRECT; unset = everything buffered
    budget = os.environ.get("GLM53_IMAGE_CACHE_GB")
    direct_from = (
        (1 << 64) - 1 if budget is None or direct else int(float(budget) * (1 << 30) / max(1, n)) // 4096 * 4096
    )
    pullers = {}
    for chip, c in sorted(man["chips"].items(), key=lambda kv: int(kv[0])):
        chip = int(chip)
        if chip not in submeshes:
            raise ImageMismatch(f"image chip {chip} is not a submesh of this mesh ({sorted(submeshes)})")
        blocks = None if only_blocks is None else only_blocks.get(chip)
        if blocks is not None and len(blocks) == 0:
            continue  # device-resident: nothing to pull
        if blocks is not None and len(blocks) == len(c["block_hashes"]):
            blocks = None  # everything stale: the whole-image stream (sequential reads, file checksums)
        p = di.ImagePuller(
            submeshes[chip],
            [tuple(s) for s in c["segments"]],
            os.path.join(d, c["image"]),
            slots=slots,
            num_cores=cores,
            read_threads=threads,
            direct=direct,
            checksum=checksum,
            direct_from=direct_from,
            only_blocks=blocks or [],
            block_bytes=c.get("block_bytes", 4 << 20),
        )
        if blocks is None and p.total_bytes != c["bytes"]:
            raise ImageMismatch(f"chip {chip}: segments give {p.total_bytes} B, manifest {c['bytes']} B")
        p.start()
        pullers[chip] = (p, blocks is not None)
    log(
        f"pullers started: {n} chips, {slots} x 1 GiB slots, {cores} cores, {threads} read threads, direct={direct}, "
        f"page-cache budget per chip={'all' if budget is None else f'{direct_from / 2**30:.1f} GiB'}"
    )
    return pullers, checksum


def finish_pullers(pullers: dict, man: dict, checksum: bool) -> dict:
    """Pull every chip in parallel (one thread per chip; finish() releases the GIL)."""
    errors, stats = [], {}

    def run(chip, p):
        try:
            p.finish()
        except BaseException as e:  # noqa: BLE001
            errors.append((chip, repr(e)))

    t0 = time.perf_counter()
    th = [threading.Thread(target=run, args=(chip, pp[0])) for chip, pp in pullers.items()]
    for t in th:
        t.start()
    for t in th:
        t.join()
    if errors:
        raise ImageMismatch(f"pull failed: {errors}")
    tot = 0
    for chip, (p, partial) in pullers.items():
        r = dict(p.result())
        if checksum and not partial and [int(x) for x in r["checksums"]] != man["chips"][str(chip)]["checksums"]:
            raise ImageMismatch(f"chip {chip}: image checksum mismatch")
        r.pop("checksums")
        r["partial"] = partial
        stats[chip] = r
        tot += int(r["bytes"])
    wall = time.perf_counter() - t0
    log(f"pulled {tot / 1e9:.2f} GB into {len(pullers)} chips in {wall:.2f} s ({tot / max(wall, 1e-9) / 1e9:.2f} GB/s)")
    stats["_bytes"] = tot
    stats["_wall_s"] = wall
    return stats


def replay_load(
    build: Callable[[], Any],
    d: str,
    man: dict,
    submeshes: dict,
    *,
    teardown: Callable[[Any], None],
    timeline: Optional[dict] = None,
):
    """Replay `build()` (the normal constructor; must use `submeshes`) under WeightIO("replay") while the pullers read
    ahead, verify the tensor map, pull, verify checksums. Raises ImageMismatch (after tearing down) on any mismatch."""
    tl = timeline if timeline is not None else {}
    t0 = time.perf_counter()
    maps = load_maps(d, man)
    host_tensors = load_host_tensors(d, man)
    # device-resident restart (GLM53_IMAGE_RESIDENT=1, default): DRAM survives process exit without a reset, so hash
    # the image regions on device first (views over still-unallocated DRAM) and pull only the stale blocks; after the
    # replay (whose constructor ops rewrite derived tensors from inputs that may no longer be resident) hash again and
    # pull what the constructor clobbered; a final device hash of everything must match the image
    resident = os.environ.get("GLM53_IMAGE_RESIDENT", "1") == "1" and all(
        "block_hashes" in c for c in man["chips"].values()
    )
    stale = None
    if resident:
        t = time.perf_counter()
        stale = stale_blocks(man, device_hashes(man, submeshes))
        tl["resident_pre_hash_s"] = time.perf_counter() - t
        tl["resident_stale_blocks"] = {str(c): len(v) for c, v in stale.items()}
        tl["resident_blocks"] = {str(c): len(man["chips"][str(c)]["block_hashes"]) for c in stale}
        log(f"resident check: stale blocks per chip {tl['resident_stale_blocks']} of {tl['resident_blocks']}")
    pullers, checksum = start_pullers(d, man, submeshes, only_blocks=stale)
    tl["pullers_start_s"] = time.perf_counter() - t0
    obj = None
    try:
        t1 = time.perf_counter()
        with WeightIO("replay", {c: m["loads"] for c, m in maps.items()}, host_tensors) as io:
            obj = build()
        io.check_complete()
        tl["replay_host_misses"] = io.host_misses
        tl["replay_s"] = time.perf_counter() - t1
        tl["replay_alloc_s"] = io.alloc_s
        tl["replay_loads"] = io.calls
        t2 = time.perf_counter()
        got = tensor_map(obj)
        for chip, m in maps.items():
            g = got.get(chip)
            if g is None or comparable(g["tensors"]) != comparable(m["tensors"]):
                diff = _first_diff(comparable(g["tensors"]) if g else [], comparable(m["tensors"]))
                raise ImageMismatch(f"chip {chip}: tensor map differs from the image ({diff})")
            if g["l1"]:
                raise ImageMismatch(f"chip {chip}: {len(g['l1'])} L1 tensors (not covered by DRAM images)")
        extra = sorted(set(got) - set(maps))
        if extra:
            raise ImageMismatch(f"chips {extra} hold tensors but have no image")
        tl["map_check_s"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        tl["pull"] = finish_pullers(pullers, man, checksum)
        tl["pull_s"] = time.perf_counter() - t3
        pullers.clear()
        if resident:
            t4 = time.perf_counter()
            clobbered = {c: v for c, v in stale_blocks(man, device_hashes(man, submeshes)).items()}
            tl["resident_post_hash_s"] = time.perf_counter() - t4
            tl["resident_clobbered_blocks"] = {str(c): len(v) for c, v in clobbered.items()}
            if any(clobbered.values()):
                t5 = time.perf_counter()
                p2, _ = start_pullers(d, man, submeshes, only_blocks=clobbered)
                tl["pull_clobbered"] = finish_pullers(p2, man, False)
                p2.clear()
                tl["pull_clobbered_s"] = time.perf_counter() - t5
                t6 = time.perf_counter()
                left = {c: v for c, v in stale_blocks(man, device_hashes(man, submeshes)).items() if v}
                tl["resident_final_hash_s"] = time.perf_counter() - t6
                if left:
                    raise ImageMismatch(
                        f"blocks still differ from the image after the pull: { {c: len(v) for c, v in left.items()} }"
                    )
            log(f"resident: clobbered by the replay {tl['resident_clobbered_blocks']}; DRAM == image (device hashes)")
        return obj
    except BaseException:
        pullers.clear()
        gc.collect()
        if obj is not None:
            with contextlib.suppress(Exception):
                teardown(obj)
        raise


def _first_diff(a: list, b: list) -> str:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"#{i}: got {x} expected {y}"
    return f"lengths {len(a)} vs {len(b)}"
