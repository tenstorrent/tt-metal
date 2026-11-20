# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Golden trace format, reader, and shared helpers for the reference-side scripts.

One directory per ladder rung, ``<golden_root>/s{seq}_c{chunk}/`` (the prefill server's golden-trace layout):
    manifest.json                         run description, layers, dumped chunks, content_hash
    metadata.json                         token_ids, num_layers, state tensor names, seq_len
    kv_cache/layer_{i}.safetensors        final state per layer, key f"{name}_cache_layer_{i}" (key/value for a KV cache)
    kv_cache/layer_{i}_at_{p}.safetensors the fixed-size state tensors (spec state.fixed) as of position p
    kv_cache/handoff_layer_{i}_at_{p}.safetensors, handoff.json, handoff-drift-<rung>.json   F264: the state at a
                                          serving handoff point (reference/handoff.py; derived, own sha256 index)
    chunk_{c:02d}/layer_{i:02d}.safetensors   boundary tensors of layer i in chunk c, keyed by boundary name
    chunk_{c:02d}/model.safetensors       embed, final_norm, top32 ids/values per position, logits of the last 32 rows

``content_hash`` is the sha256 over the sorted (path, file sha256) list of every file except manifest.json and the
files derived from the golden afterwards (drift-<rung>.json, F128 / F134; official_floor.json, F80).
F128: the golden is the full-precision reference (fp32, no rounding); the reference at the shipped code's precision is
used only for its drift (reference/drift.py), the end-to-end margin (``<rung dir>-official``: F136's hf_official).
Tests pin the manifest's hash in their frozen block; a regenerated or edited golden then fails the gate.

F260 (owner 2026-11-06): the goldens are read from a node-local mirror, never from NFS on every test. The shared copy
(``spec.golden_root``, paths.golden: generated there, survives a change of box) is the source of truth; the mirror
(``spec.golden_mirror_root``, paths.golden_local) is a copy of the ladder's rung dirs (and their ``-official``
siblings). ``mirror.json`` in each dir lists its files (relative path: [size, mtime_ns]; sha256 where files.json
already has it); generate_golden writes it beside the shared copy, ``sync`` refreshes it and rsyncs it with the files.
``read_dir`` (Golden.for_rung) gives the mirror's rung dir when, for the dir and its ``-official`` sibling, the
mirror's mirror.json lists the shared files as they are now and every mirror file matches its line; else the shared
dir with a one-line warning. Writers use ``rung_dir`` / ``official_dir`` / ``Golden.shared_dir`` (the shared copy).
Frozen files (tasks.yaml ``frozen.files``, keyed by the shared path) stay checked on the shared copy: the keys are
the same on every box, the frozen golden files are small JSON (manifest, drift), and the mirror is only read when it
matches the shared files (size + mtime after rsync -a), so the bytes checked are the bytes read.

    python -m models.demos.common.bringup.reference.golden sync --spec S [--rung R]     # shared -> mirror, verified
    python -m models.demos.common.bringup.reference.golden status --spec S [--rung R]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file

from models.demos.common.bringup.core import mirror as MR
from models.demos.common.bringup.core.freeze import sha256_file
from models.demos.common.bringup.core.spec import GOLDEN_MIRROR_FILE, Spec

TOPK = 32
LOGITS_TAIL = 32


def rung_dirname(seq: int, chunk: int) -> str:
    return f"s{seq}_c{chunk}"


def rung_dir(spec: Spec, rung: dict) -> Path:
    """The rung's golden in the shared copy (where it is written, pinned and frozen; readers: read_dir, F260). A rung
    with ``golden: <other rung>`` reads the other rung's golden (e.g. last chunk after a golden prefix)."""
    src = spec.rung(rung["golden"]) if rung.get("golden") else rung
    return spec.golden_root / rung_dirname(src["seq"], src["chunk"])


# written after the golden; not in content_hash. files.json (F186): every file's sha256, so a partial copy verifies
DERIVED = ("manifest.json", "drift.json", "official_floor.json", "files.json", GOLDEN_MIRROR_FILE)  # F260: mirror.json
INDEX = "files.json"


def derived(name: str) -> bool:
    # F134: drift-<rung>.json; F264: the handoff snapshots and their drift (reference/handoff.py, own sha256 index)
    return name in DERIVED or (name.startswith("drift-") and name.endswith(".json")) or name.startswith("handoff")


OFFICIAL_SUFFIX = "-official"  # F128: the same rung at the reference's own shipped precision (the margin's source)


def official_dir(spec: Spec, rung: dict) -> Path:
    """F128: the rung's golden at the reference's own precision, next to the full-precision one; only its drift from
    the full-precision golden is used (reference/drift.py), never as a target."""
    d = rung_dir(spec, rung)
    return d.with_name(d.name + OFFICIAL_SUFFIX)


# ---------------------------------------------------------------- F260: the node-local mirror
MIRROR = GOLDEN_MIRROR_FILE
_warned: set[str] = set()


def listing(d: Path) -> dict[str, list[int]]:
    """{relative path: [size, mtime_ns]} of every file under d but mirror.json (a stat walk: ~0.05 s for a rung on NFS;
    core/mirror.py, shared with the weight cache's mirror, F261)."""
    return MR.listing(d)


def manifest_of(d: Path) -> dict:
    """The mirror manifest of d as it is now: its listing, and the sha256 files.json (F186) already has (never
    computed here: a rung is tens of GB)."""
    d = Path(d)
    files = listing(d)
    index = json.loads((d / INDEX).read_text()) if (d / INDEX).exists() else {}
    return {"files": files, "sha256": {k: v for k, v in index.items() if k in files}}


def write_manifest(d: Path) -> dict:
    """mirror.json beside a shared golden dir (generate_golden, sync). Rewritten only when the files changed, so an
    in-sync mirror stays in sync. A read-only shared copy (another user's) keeps no manifest: sync writes the mirror's.
    """
    m = manifest_of(d)
    MR.write_manifest(d, m)
    return m


_read = MR.read_json


def golden_units(spec: Spec, rungs: list[str] | None = None) -> list[str]:
    """The shared golden dir names the ladder (or the given rungs) reads: each rung's dir and its -official sibling."""
    names = []
    for r in spec.data.get("ladder") or []:
        if rungs and r["name"] not in rungs:
            continue
        src = spec.rung(r["golden"]) if r.get("golden") else r
        d = rung_dirname(src["seq"], src["chunk"])
        names += [n for n in (d, d + OFFICIAL_SUFFIX) if n not in names]
    return names


def mirror_problem(shared: Path, mirror: Path) -> str | None:
    """Why the mirror of one shared golden dir cannot be read in its place (None: it is complete). The shared copy
    as it is now is the reference (a drift file rewritten there after the sync makes the mirror stale)."""
    return MR.mirror_problem(shared, mirror)


def unit_problem(spec: Spec, name: str) -> str | None:
    """mirror_problem of a rung dir and its -official sibling together (tests open ``<dir>-official`` beside the
    rung dir they were given); a sibling missing from the shared copy is not a problem."""
    shared, mirror = spec.golden_root, spec.golden_mirror_root
    if not (shared / name).exists() and not (mirror / name).exists():
        return None  # not generated: nothing to mirror
    for n in (name, name + OFFICIAL_SUFFIX) if not name.endswith(OFFICIAL_SUFFIX) else (name,):
        if n != name and not (shared / n).exists():
            continue
        why = mirror_problem(shared / n, mirror / n)
        if why:
            return f"{n}: {why}"
    return None


def read_dir(spec: Spec, rung: dict) -> Path:
    """F260: the dir a test reads the rung's golden from: the node-local mirror when complete, else the shared copy
    with a one-line warning (a slow path is never silent, and never a failure)."""
    d = rung_dir(spec, rung)
    mroot = spec.golden_mirror_root
    if mroot is None or not d.exists() and not (mroot / d.name).exists():
        return d
    why = unit_problem(spec, d.name)
    if why is None:
        return mroot / d.name
    if str(d) not in _warned:
        _warned.add(str(d))
        print(
            f"WARNING (F260): golden {d.name} read from the shared copy {d} (slow), its node-local mirror "
            f"{mroot / d.name} is not complete ({why}); python -m models.demos.common.bringup.reference.golden sync "
            f"--spec {spec.path or '<spec>'}",
            file=sys.stderr,
            flush=True,
        )
    return d


def _rsync(src: Path, dst: Path) -> None:
    MR.rsync(src, dst)


def sync(spec: Spec, rungs: list[str] | None = None, *, echo=print) -> dict[str, str]:
    """F260: rsync -a the shared golden dirs the ladder reads into the mirror, then verify each against its
    mirror.json. Idempotent: a complete dir costs one stat walk of each side. Returns {dir: result}; never raises
    for a dir (a failed copy leaves the tests on the shared copy, with the warning)."""
    shared, mroot = spec.golden_root, spec.golden_mirror_root
    if mroot is None:
        echo(f"goldens: no mirror ({shared} is node-local or goldens.mirror is off)")
        return {}
    out = {}
    for name in golden_units(spec, rungs):
        s, m = shared / name, mroot / name
        if not s.is_dir():
            if not name.endswith(OFFICIAL_SUFFIX):
                out[name] = "not generated"
            continue
        t0 = time.time()
        try:
            man = write_manifest(s)
            if mirror_problem(s, m) is None:
                out[name] = "in sync"
            else:
                _rsync(s, m)
                (m / MIRROR).write_text(json.dumps(man, indent=0, sort_keys=True))  # a read-only shared copy has none
                why = mirror_problem(s, m)
                out[name] = f"FAILED: {why}" if why else f"synced in {time.time() - t0:.0f}s"
            extra = sorted(set(listing(m)) - set(man["files"]))
            if extra:  # written into the mirror instead of the shared copy: lost on another box
                out[name] += f"; {len(extra)} files only in the mirror (write them to the shared copy): {extra[:3]}"
        except (OSError, subprocess.CalledProcessError) as e:
            out[name] = f"FAILED: {e}"
        echo(f"goldens mirror: {name}: {out[name]}")
    return out


TARGET_NUMERICS = "native"  # F128 / F132 (owner): the target of every check is the full-precision reference; no switch
NUMERICS = ("native", "official")


def reference_at(spec: Spec, numerics: str, *, layers=None, dtype=torch.float32):
    """The only way into hooks.reference (F132): the CPU reference at ``numerics`` ('native' = full precision, the
    target; 'official' = the shipped code's arithmetic, only for the margin). The hook must name a ``numerics``
    parameter (a ``**kw``-only hook would take it and silently ignore it). A hook without one has one arithmetic:
    allowed only when the spec declares it free of parity roundings (``reference.no_rounding: true``; F136 then checks
    a margin exists); never a silent fallback (F128)."""
    import inspect

    if numerics not in NUMERICS:
        raise ValueError(f"numerics {numerics!r}, want one of {NUMERICS}")
    fn = spec.hooks().reference
    if "numerics" in inspect.signature(fn).parameters:
        return fn(spec, layers=layers, dtype=dtype, numerics=numerics)
    if not spec.get("reference.no_rounding"):
        raise SystemExit(
            f"hooks.reference has no `numerics` parameter, so it cannot run at {numerics!r}; add it (the reference "
            "role: native = fp32 with every quantize-dequantize off, official = the shipped code's arithmetic), or "
            "set reference.no_rounding: true if its arithmetic rounds nothing (F128, F132)"
        )
    return fn(spec, layers=layers, dtype=dtype)


def file_hashes(d: Path) -> dict[str, str]:
    """{relative path: sha256} of every golden file (not the derived ones), sorted by path."""
    return {
        str(f.relative_to(d)): sha256_file(f)
        for f in sorted(p for p in d.rglob("*") if p.is_file() and not derived(p.name))
    }


def hash_of(files: dict[str, str]) -> str:
    h = hashlib.sha256()
    for rel, sha in sorted(files.items()):
        h.update(f"{rel}\0{sha}\n".encode())
    return h.hexdigest()


def content_hash(d: Path) -> str:
    return hash_of(file_hashes(d))


def write_index(d: Path) -> dict:
    """F186: files.json beside the manifest, written where the golden is complete, so a copy that leaves files out
    (only what a run's tests read) still verifies: the index reproduces the content_hash, each copied file its line."""
    d = Path(d)
    files = file_hashes(d)
    want = json.loads((d / "manifest.json").read_text()).get("content_hash")
    if hash_of(files) != want:
        raise ValueError(f"{d}: the files do not give the manifest's content_hash; no index for a broken golden")
    (d / INDEX).write_text(json.dumps(files, indent=0, sort_keys=True))
    return files


def verify_copy(d: Path) -> tuple[bool, list[str]]:
    """F186: (intact, missing files) of a golden that may be a partial copy: the index gives the manifest's
    content_hash and every file present matches its index line; files left out are listed, never guessed."""
    d = Path(d)
    if not (d / INDEX).exists():
        return Golden(d).verify(), []
    index = json.loads((d / INDEX).read_text())
    have = file_hashes(d)
    ok = hash_of(index) == json.loads((d / "manifest.json").read_text()).get("content_hash")
    ok = ok and all(index.get(rel) == sha for rel, sha in have.items())
    return ok, sorted(set(index) - set(have))


class Golden:
    def __init__(self, d: Path):
        self.dir = Path(d)
        self.shared_dir = self.dir  # F260: for_rung sets the shared copy when it reads the mirror
        if not (self.dir / "manifest.json").exists():
            raise FileNotFoundError(f"golden not generated: {self.dir}")
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        self.seq, self.chunk = self.manifest["seq"], self.manifest["chunk"]
        self.n_chunks = self.seq // self.chunk
        self._layer = lru_cache(maxsize=8)(self._load_layer)

    @classmethod
    def for_rung(cls, spec: Spec, name: str) -> "Golden":
        """F260: read from the node-local mirror when complete (``dir``); ``shared_dir`` is where to write."""
        rung = spec.rung(name)
        g = cls(read_dir(spec, rung))
        g.shared_dir = rung_dir(spec, rung)
        return g

    @property
    def layers(self) -> list[int]:
        return list(self.manifest["layers"])

    @property
    def dumped_chunks(self) -> list[int]:
        return list(self.manifest.get("dumped_chunks", [self.n_chunks - 1]))

    @property
    def state_names(self) -> list[str]:
        return list(self.meta.get("state_tensors", ["key", "value"]))

    def tokens(self) -> torch.Tensor:
        return torch.tensor(self.meta["token_ids"], dtype=torch.int64)

    def _load_layer(self, c: int, i: int) -> dict:
        return load_file(str(self.dir / f"chunk_{c:02d}" / f"layer_{i:02d}.safetensors"))

    def layer(self, c: int, i: int) -> dict[str, torch.Tensor]:
        return self._layer(c, i)

    def has_layer(self, c: int, i: int) -> bool:
        return (self.dir / f"chunk_{c:02d}" / f"layer_{i:02d}.safetensors").exists()

    def model(self, c: int) -> dict[str, torch.Tensor]:
        return load_file(str(self.dir / f"chunk_{c:02d}" / "model.safetensors"))

    def layer_state_names(self, i: int) -> list[str]:
        return list(self.meta.get("state_by_layer", {}).get(str(i), self.state_names))

    @property
    def state_fixed(self) -> list[str]:
        return list(self.meta.get("state_fixed", []))

    def state(self, i: int, at: int | None = None) -> dict[str, torch.Tensor]:
        """Layer i's state. Tensors that grow along the sequence are the final ones (the consumer slices [0, at)).
        Fixed-size tensors are the snapshot as of position ``at`` (None or seq: the final state)."""
        d = load_file(str(self.dir / "kv_cache" / f"layer_{i}.safetensors"))
        out = {n: d[f"{n}_cache_layer_{i}"] for n in self.layer_state_names(i)}
        fixed = [n for n in out if n in self.state_fixed]
        if fixed and at is not None and at != self.seq:
            p = self.dir / "kv_cache" / f"layer_{i}_at_{at}.safetensors"
            if not p.exists():
                from models.demos.common.bringup.reference.handoff import load_snapshot

                snap = load_snapshot(self, i, at)  # F264: a serving handoff point (reference/handoff.py)
                if snap is not None:
                    out.update({n: snap[f"{n}_cache_layer_{i}"] for n in fixed})
                    return out
                raise FileNotFoundError(
                    f"{p}: no snapshot of the fixed-size state {fixed} at position {at} "
                    f"(snapshots at {self.meta.get('state_snapshots', [])})"
                )
            snap = load_file(str(p))
            out.update({n: snap[f"{n}_cache_layer_{i}"] for n in fixed})
        return out

    def pinned_hash(self) -> str:
        """The hash recorded at generation (or, for goldens made before hashes, the manifest file's own hash)."""
        return self.manifest.get("content_hash") or sha256_file(self.dir / "manifest.json")

    def verify(self) -> bool:
        return "content_hash" in self.manifest and content_hash(self.dir) == self.manifest["content_hash"]


# ---------------------------------------------------------------- shared helpers for the scripts
def load_spec(path: str | None = None) -> Spec:
    path = path or os.environ.get("BRINGUP_SPEC")
    if not path:
        raise SystemExit("no spec: pass --spec or set BRINGUP_SPEC")
    return Spec.load(path)


def hf_path(spec: Spec) -> str:
    """Local checkpoint dir: paths.hf if set, else $ART/<model>/hf, else the HF hub cache (no download)."""
    if spec.get("paths.hf") or (spec.hf_dir / "config.json").exists():
        return str(spec.hf_dir)
    from huggingface_hub import snapshot_download

    return snapshot_download(spec.data["hf_id"], local_files_only=True)


def tokenizer(spec: Spec):
    hooks = spec.hooks()
    if hasattr(hooks, "tokenizer"):
        return hooks.tokenizer(spec)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_path(spec), trust_remote_code=bool(spec.get("hf.trust_remote_code")))


def text_tokens(spec: Spec, n: int, tok=None) -> torch.Tensor:
    """The first n tokens of the model's canonical prompt (reference/prompt.py). Every step uses this."""
    from models.demos.common.bringup.reference import prompt

    return prompt.tokens(spec, n, tok)


def store_dtype(t: torch.Tensor, keep_fp32: tuple[str, ...] | bool, name: str) -> torch.Tensor:
    """keep_fp32: True keeps every float tensor fp32 (F128: the full-precision golden), else the names kept fp32."""
    if not t.is_floating_point():
        return t.contiguous()
    if keep_fp32 is True or name.endswith(keep_fp32):
        return t.float().contiguous()
    return t.to(torch.bfloat16).contiguous()


def main(argv=None) -> int:
    """F260: ``sync`` copies the shared goldens the ladder reads into the node-local mirror (low I/O priority) and
    verifies them; ``status`` says which dir each rung is read from."""
    ap = argparse.ArgumentParser(prog="python -m models.demos.common.bringup.reference.golden")
    ap.add_argument("command", choices=["sync", "status"])
    ap.add_argument("--spec")
    ap.add_argument("--rung", action="append", help="only this rung's golden (repeatable); default: the whole ladder")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    if a.command == "sync":
        t0 = time.time()
        res = sync(spec, a.rung)
        print(f"goldens mirror: {spec.golden_mirror_root}: done in {time.time() - t0:.1f}s")
        return 1 if any(v.startswith("FAILED") for v in res.values()) else 0
    for r in spec.data.get("ladder") or []:
        if not a.rung or r["name"] in a.rung:
            print(f"{r['name']}: {read_dir(spec, r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
