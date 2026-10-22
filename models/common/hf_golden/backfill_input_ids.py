#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Write input_ids.safetensors into traces generated before engine.py wrote it.

    python backfill_input_ids.py [ROOT] [--only <hf-repo>[/isl<N>]] [--write] [--check-tokenizer]

The ids are rebuilt as the engine built them: metadata prefix_ids, then the first N - len(prefix_ids) ids of the
token cache. Before anything is written they are checked against the trace itself:
  * the token cache was built for this repo, by this tokenizer (when the trace recorded its fingerprint);
  * hidden/embed.safetensors: every position holding the same id has the same embedding row, and no two ids share
    a row. The embedding is a per-id lookup, so a prefix shift or a wrong cache fails here with no weights loaded;
  * --check-tokenizer: the model's tokenizer turns metadata prefix into exactly prefix_ids.

Dry run unless --write. Only adds input_ids.safetensors, readable by all (via a temp file, then os.replace); every
other file of the trace, metadata.json and SHA256SUMS included, is left as it is, and an existing input_ids.safetensors
is never overwritten. Traces without prefix_ids (the first engine, the whole-model layout) had no prefix. The
whole-model layout stores no embed file, so its ids (token_cache[:N], as its read_trace_tokens defines them) are
checked against the token cache's metadata only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import INPUT_IDS, hub_kwargs  # noqa: E402

DEFAULT_ROOT = os.environ.get("HF_GOLDEN_ROOT", "/mnt/models/agentic-prefill-goldens")
WRITTEN_BY = "backfill_input_ids.py"


class Fail(Exception):
    pass


class Skip(Exception):
    pass


def find_traces(root: Path, only: str | None) -> list[Path]:
    """Every isl<N> directory holding a metadata.json, not descending into one (trimmed copies are traces too)."""
    start = root / only if only else root
    if start.name.startswith("isl") and (start / "metadata.json").is_file():
        return [start]
    found = []
    for dirpath, dirnames, _ in os.walk(start):
        for d in list(dirnames):
            if d.startswith("isl") and d[3:].isdigit():
                dirnames.remove(d)
                if (Path(dirpath) / d / "metadata.json").is_file():
                    found.append(Path(dirpath) / d)
    return sorted(found)


def rebuild(trace: Path, meta: dict) -> tuple[torch.Tensor, dict]:
    if "hf_repo" not in meta or "token_cache" not in meta:
        raise Skip("no hf_repo / token_cache in metadata.json")
    # No prefix_ids: written before the engine had --prefix (or by the whole-model generator), so nothing came before
    # the token cache.
    n, prefix_ids = meta["n_tokens"], meta.get("prefix_ids", [])
    cache_path = (trace / meta["token_cache"]).resolve()
    if not cache_path.is_file():
        raise Fail(f"token cache {cache_path} missing")
    with safe_open(str(cache_path), "pt") as f:
        cache_meta = f.metadata()
        cache = f.get_tensor("token_ids").flatten()
    if cache_meta.get("hf_repo") != meta["hf_repo"]:
        raise Fail(f"token cache built for {cache_meta.get('hf_repo')}, trace is {meta['hf_repo']}")
    if meta.get("tokenizer_sha256") and cache_meta.get("tokenizer_sha256") not in (None, meta["tokenizer_sha256"]):
        raise Fail("token cache built by a different tokenizer than the trace")
    if cache_meta.get("source_sha256") != meta["source_sha256"]:
        raise Fail("token cache built from a different text than the trace")
    if n - len(prefix_ids) > cache.numel():
        raise Fail(f"token cache holds {cache.numel()} ids, trace needs {n - len(prefix_ids)}")
    ids = torch.cat([torch.tensor(prefix_ids, dtype=torch.int32), cache[: n - len(prefix_ids)].to(torch.int32)])
    assert ids.numel() == n
    file_meta = {
        "hf_repo": meta["hf_repo"],
        "n_tokens": str(n),
        "n_prefix_tokens": str(len(prefix_ids)),
        "prefix": meta.get("prefix") or "",
        "token_cache": meta["token_cache"],
        "tokenizer_sha256": meta.get("tokenizer_sha256") or cache_meta.get("tokenizer_sha256") or "",
        "source": meta["source"],
        "source_sha256": meta["source_sha256"],
        "written_by": WRITTEN_BY,
    }
    return ids, file_meta


def check_embeddings(trace: Path, ids: torch.Tensor) -> str:
    """Same id -> same row everywhere, different ids -> different rows. Returns a short summary."""
    embed = load_file(str(trace / "hidden/embed.safetensors"))
    if len(embed) != 1:
        raise Fail(f"hidden/embed.safetensors holds {sorted(embed)}, expected one tensor")
    (rows,) = embed.values()
    rows = rows.reshape(ids.numel(), -1)
    uniq, inverse = torch.unique(ids, return_inverse=True)
    first = torch.full((uniq.numel(),), ids.numel(), dtype=torch.long)
    first.scatter_reduce_(0, inverse, torch.arange(ids.numel()), reduce="amin")
    same = (rows == rows[first[inverse]]).all(dim=1)
    if not same.all():
        bad = (~same).nonzero().flatten()
        raise Fail(f"{bad.numel()} positions differ from their id's first embedding row (first at {bad[0].item()})")
    seen: dict[bytes, int] = {}
    for u, i in zip(uniq.tolist(), first.tolist()):
        key = hashlib.sha256(rows[i].contiguous().view(torch.uint8).numpy().tobytes()).digest()
        if key in seen:
            raise Fail(f"ids {seen[key]} and {u} share one embedding row")
        seen[key] = u
    return f"{uniq.numel()} distinct ids consistent"


def check_tokenizer(meta: dict) -> None:
    if not meta.get("prefix"):
        return
    from transformers import AutoTokenizer

    repo = meta["hf_repo"]
    tok = AutoTokenizer.from_pretrained(
        repo, trust_remote_code=meta.get("trust_remote_code", False), **hub_kwargs(repo, meta.get("revision"))
    )
    got = tok(meta["prefix"], add_special_tokens=False)["input_ids"]
    if got != meta["prefix_ids"]:
        raise Fail(f"tokenizer gives {got[:8]}... for the prefix, metadata has {meta['prefix_ids'][:8]}...")


def write_input_ids(trace: Path, ids: torch.Tensor, file_meta: dict) -> None:
    tmp = trace / f".{INPUT_IDS}.tmp"
    save_file({"input_ids": ids}, str(tmp), metadata=file_meta)
    os.chmod(tmp, 0o664)  # readable by all, whatever the umask
    os.replace(tmp, trace / INPUT_IDS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root", type=Path, nargs="?", default=Path(DEFAULT_ROOT), help=f"default {DEFAULT_ROOT}")
    ap.add_argument("--only", help="<hf-repo> or <hf-repo>/isl<N> under ROOT")
    ap.add_argument("--write", action="store_true", help="write the files (default: dry run)")
    ap.add_argument("--check-tokenizer", action="store_true", help="also re-tokenize the prefix (needs transformers)")
    args = ap.parse_args()

    traces = find_traces(args.root, args.only)
    if not traces:
        raise SystemExit(f"no trace (isl<N>/metadata.json) under {args.root / (args.only or '')}")
    rows, failed = [], False
    for trace in traces:
        name = str(trace.relative_to(args.root)) if trace.is_relative_to(args.root) else str(trace)
        n = prefix = first = "-"
        try:
            meta = json.loads((trace / "metadata.json").read_text())
            n = meta.get("n_tokens", "-")
            ids, file_meta = rebuild(trace, meta)
            prefix, first = int(file_meta["n_prefix_tokens"]), " ".join(map(str, ids[:8].tolist()))
            if (trace / "hidden/embed.safetensors").is_file():
                note = check_embeddings(trace, ids)
            else:
                note = "no embed file: token cache metadata checked only"
            if args.check_tokenizer:
                check_tokenizer(meta)
            exists = (trace / INPUT_IDS).is_file()
            if exists:
                if not torch.equal(load_file(str(trace / INPUT_IDS))["input_ids"], ids):
                    raise Fail(f"{INPUT_IDS} exists and differs")
                note = f"exists, identical; {note}"
            if exists:
                result = f"OK ({note})"
            elif args.write:
                write_input_ids(trace, ids, file_meta)
                result = f"WROTE ({note})"
            else:
                result = f"WOULD WRITE ({note})"
        except Skip as e:
            result = f"SKIP ({e})"
        except Fail as e:
            result, failed = f"FAIL ({e})", True
        except PermissionError as e:
            result, failed = f"FAIL (permission: {e.filename})", True
        rows.append((name, n, prefix, result, first))
        print(f"[{name}] {result}", flush=True)

    print()
    widths = [max(len(str(r[c])) for r in rows) for c in range(3)]
    for r in rows:
        print(f"{r[0]:{widths[0]}}  N={str(r[1]):{widths[1]}}  prefix={str(r[2]):{widths[2]}}  {r[3]}  [{r[4]}]")
    if not args.write:
        print("\ndry run: pass --write to write")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
