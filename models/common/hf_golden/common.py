# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pieces every golden generator shares: the store layout, the per-model token cache, checkpoint shards streamed from
the hub, loading them into meta modules, and the inventory of stored tensors written to metadata.json.

Needs only torch, transformers, safetensors and huggingface_hub: this folder is meant to work copied out on its own.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import sys
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(os.environ.get("HF_GOLDEN_ROOT", "generated/agentic-prefill-goldens"))
TMP = Path(os.environ.get("HF_GOLDEN_TMP", "generated/hf_golden_tmp"))
TEXT = "wikitext-2-raw-v1-train.txt"
BLOCK_BYTES = 1 << 20
# Bumped when the trace layout or key names change, so a reader can fail loudly on a format it does not know.
GOLDEN_SCHEMA = "1"
# ISL has to suit the device even though nothing here talks to one: the device shards the sequence over sp ranks in
# 32-row tiles, so a trace at any other ISL can never be graded. Literals, not a spec import: no device stack here.
ISL_ALIGNMENT = 32 * 8  # ttnn.TILE_SIZE * sp
ISL_CHUNK_SIZE = 5120  # spec chunk_size; a chunked grade additionally needs this
# Checked after this user's own HF cache, so checkpoints already on the cluster are never downloaded again.
SHARED_HF_CACHE = os.environ.get("HF_GOLDEN_SHARED_CACHE", "/mnt/models/huggingface/hub")

COMMON_NOTES = {
    "model.embed_tokens.output": "Token embeddings of the first n_tokens ids of the token cache (no BOS added).",
    "model.layers.{i}.output": "Decoder layer output, residual included: the next layer's input.",
    "model.last_hidden_state": "Final norm applied to the last layer's output.",
    "logits": "lm_head over the last position only (plus final_logit_softcapping where the model has it), fp32.",
}


def raise_cpu_time_limit() -> None:
    """RLIMIT_CPU counts CPU-seconds over every thread, so a 64-thread matmul drains it ~64x faster than wall time
    and the run dies with SIGXCPU mid-forward. Raise the soft limit to the hard one (allowed unprivileged)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    if soft != resource.RLIM_INFINITY and (hard == resource.RLIM_INFINITY or soft < hard):
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (hard, hard))
        except (ValueError, OSError) as e:
            print(f"[limit] WARNING: could not raise RLIMIT_CPU (soft={soft}s): {e}", file=sys.stderr)


def token_cache(repo: str, root: Path, text: Path, tokenizer_path: str | None = None, **hub) -> Path:
    """Tokenize the whole text once per model. Blocks of ~1 MiB, cut on a line boundary so no token straddles two
    blocks (wikitext is line-structured; the result equals tokenizing the text in one call), no special tokens."""
    out = root / repo / f"{text.stem}.tokens.safetensors"
    if out.exists():
        return out
    if not text.exists():
        raise SystemExit(f"{text} not found: pass --text, or put {TEXT} in --root")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path or repo, **hub)
    raw = text.read_bytes()
    ids: list[int] = []
    i = 0
    while i < len(raw):
        j = min(i + BLOCK_BYTES, len(raw))
        if j < len(raw):
            nl = raw.rfind(b"\n", i, j)
            j = nl + 1 if nl >= i else j
        ids += tok(raw[i:j].decode("utf-8"), add_special_tokens=False)["input_ids"]
        i = j
    meta = {
        "hf_repo": repo,
        "source": text.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_bytes": str(len(raw)),
        "n_tokens": str(len(ids)),
        "tokenizer_class": type(tok).__name__,
        "add_special_tokens": "false",
        "block_bytes": str(BLOCK_BYTES),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file({"token_ids": torch.tensor(ids, dtype=torch.int32)}, str(out), metadata=meta)
    print(f"[tokens] {len(ids)} tokens -> {out}", flush=True)
    return out


def check_isl(isl: int, strict: bool) -> None:
    """Reject (``strict``) or flag an ISL no device run could grade. See ISL_ALIGNMENT."""
    if isl % ISL_ALIGNMENT:
        lo, hi = isl // ISL_ALIGNMENT * ISL_ALIGNMENT, (isl // ISL_ALIGNMENT + 1) * ISL_ALIGNMENT
        msg = f"isl {isl} is not a multiple of {ISL_ALIGNMENT} (TILE_SIZE*sp): no device run can grade it ({lo}, {hi} can)"
        if strict:
            raise SystemExit(msg)
        print(f"[isl] note: {msg}", flush=True)
    elif isl % ISL_CHUNK_SIZE:
        print(
            f"[isl] note: {isl} is not a multiple of chunk_size {ISL_CHUNK_SIZE}: grades one-shot runs only", flush=True
        )


def share(trace: Path, root: Path) -> None:
    """``a+rX`` on everything in ``trace`` and each directory from ``root`` down to it: the store is shared, but a 0077
    umask writes files nobody else can read. OR-ed onto the mode (keeps a setgid bit); warns rather than raises, since
    a path owned by someone else is no reason to lose a correct trace."""
    try:
        parents = list(trace.relative_to(root).parents)[:-1]
    except ValueError:
        parents = []
    for path in [*(root / p for p in reversed(parents)), trace, *trace.rglob("*")]:
        try:
            extra = 0o055 if path.is_dir() else 0o044
            mode = path.stat().st_mode
            if mode & extra != extra:
                path.chmod((mode & 0o7777) | extra)
        except OSError as e:
            print(f"[perm] WARNING: could not chmod {path}: {e}", file=sys.stderr, flush=True)


def resolve_revision(repo: str, revision: str) -> str:
    """Pin a branch or tag to its commit, so a re-upload mid-run cannot mix two checkpoints (or two versions of remote
    code) in one trace. A branch already in an HF cache pins to the commit stored there, so a checkpoint on disk is
    used as is rather than replaced by a newer upload; pass a commit to override. A local directory has no revision."""
    if Path(repo).is_dir():
        return "local"
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    if (snapshot := local_snapshot(repo, revision)) is not None:
        return snapshot.name
    return Path(hf_hub_download(repo, "config.json", revision=revision)).parent.name  # .../snapshots/<commit>/


def hub_kwargs(repo: str, revision: str | None) -> dict:
    """from_pretrained kwargs reading ``repo`` at ``revision`` from the copy on disk when there is one, offline, so
    config, tokenizer and remote code need no hub access. By repo id plus cache_dir rather than the snapshot path:
    remote code loaded from a snapshot path resolves its own imports next to the cache's blobs, where they are not."""
    if Path(repo).is_dir():
        return {}
    snapshot = local_snapshot(repo, revision)
    if snapshot is None:
        return {"revision": revision}
    return {"revision": revision, "cache_dir": str(snapshot.parents[2]), "local_files_only": True}


def local_snapshot(repo: str, revision: str | None) -> Path | None:
    """The directory holding ``repo`` at ``revision`` (config, tokenizer, remote code, maybe shards) if one is on
    disk: the repo itself when it is a local directory, else its snapshot in an HF cache."""
    path = cached_file(repo, "config.json", revision)
    return path.parent if path is not None and path.exists() else None


def cached_file(repo: str, filename: str, revision: str) -> Path | None:
    """``filename`` of ``repo`` at ``revision`` if it is already on disk: a local checkpoint dir, this user's HF
    cache, or the shared one. None when it has to be downloaded."""
    if Path(repo).is_dir():
        return Path(repo) / filename
    for cache_dir in (None, SHARED_HF_CACHE):
        try:
            return Path(hf_hub_download(repo, filename, revision=revision, cache_dir=cache_dir, local_files_only=True))
        except (LocalEntryNotFoundError, EntryNotFoundError, OSError):
            continue
    return None


class Shards:
    """Checkpoint shards read on demand. A shard already on disk (local dir, HF cache) is read in place; any other is
    downloaded into ``tmp`` and deleted after the last stage that reads it, so peak disk is a few shards rather than
    the checkpoint."""

    def __init__(self, repo: str, revision: str, tmp: Path, stages: list[str]):
        self.repo, self.revision, self.tmp = repo, revision, tmp
        index = self.fetch("model.safetensors.index.json", must_exist=False)
        if index is not None:
            self.weight_map: dict[str, str] = json.loads(index.read_text())["weight_map"]
        else:  # a single-file checkpoint
            single = self.fetch("model.safetensors")
            with safe_open(str(single), "pt") as h:
                self.weight_map = dict.fromkeys(h.keys(), "model.safetensors")
        self.last_use = {f: s for s, prefix in enumerate(stages) for f in self.files(prefix)}

    def fetch(self, filename: str, must_exist: bool = True) -> Path | None:
        path = cached_file(self.repo, filename, self.revision)
        if path is not None and path.exists():
            return path
        try:
            return Path(hf_hub_download(self.repo, filename, revision=self.revision, local_dir=self.tmp))
        except EntryNotFoundError:
            if must_exist:
                raise
            return None

    def has(self, prefix: str) -> bool:
        return any(k.startswith(prefix) for k in self.weight_map)

    def files(self, prefix: str) -> list[str]:
        return sorted({f for k, f in self.weight_map.items() if k.startswith(prefix)})

    def load(self, stage: int, prefix: str) -> dict[str, torch.Tensor]:
        """Every checkpoint tensor under ``prefix``, keys relative to it, in the checkpoint's own dtype."""
        sd = {}
        for f in self.files(prefix):
            path = self.fetch(f)
            with safe_open(str(path), "pt") as h:
                for k in h.keys():
                    if k.startswith(prefix):
                        sd[k[len(prefix) :]] = h.get_tensor(k)
            if self.last_use[f] <= stage and path.is_relative_to(self.tmp):  # never delete a cached copy
                path.unlink()
        return sd


def fuse_experts(sd: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """The checkpoint stores routed experts one by one (``experts.<e>.{gate,up,down}_proj.weight``); the HF MoE holds
    them fused as ``gate_up_proj`` [E, 2I, H] and ``down_proj`` [E, H, I] (from_pretrained's qwen2_moe conversion
    mapping). Filled in place so peak memory is one extra copy of a single expert, not of all of them."""
    groups: dict[str, dict[str, dict[int, str]]] = defaultdict(lambda: defaultdict(dict))
    for k in list(sd):
        m = re.fullmatch(r"(.*\.experts)\.(\d+)\.(gate|up|down)_proj\.weight", k)
        if m:
            groups[m[1]][m[3]][int(m[2])] = k
    for pre, g in groups.items():
        n = len(g["down"])
        inter, hidden = sd[g["gate"][0]].shape
        gate_up = torch.empty(n, 2 * inter, hidden, dtype=dtype)
        down = torch.empty(n, hidden, inter, dtype=dtype)
        for e in range(n):
            gate_up[e, :inter] = sd.pop(g["gate"][e])
            gate_up[e, inter:] = sd.pop(g["up"][e])
            down[e] = sd.pop(g["down"][e])
        sd[f"{pre}.gate_up_proj"], sd[f"{pre}.down_proj"] = gate_up, down
    return sd


class Loader:
    """Swaps a meta module's parameters for the real ones, at the compute dtype except where the model pins a tensor
    to fp32 (``_get_dtype_plan``: the same rule from_pretrained applies, e.g. GLM's router e_score_correction_bias
    under bf16). Experts are fused on the host, then the module moves to the device as a whole.

    Non-persistent buffers (RoPE inv_freq, embedding scales, ...) are not in any checkpoint. from_pretrained leaves
    them on meta and then rebuilds them with the model's own ``_init_weights``; so does this, skipping every tensor
    that came from the checkpoint."""

    def __init__(self, model, dtype: torch.dtype, device: torch.device):
        self.model, self.dtype, self.device = model, dtype, device
        self.fp32 = [re.compile(rf"(^|\.){re.escape(m)}(\.|$)") for m in model._get_dtype_plan(dtype)]

    def __call__(self, module: torch.nn.Module, sd: dict[str, torch.Tensor], prefix: str) -> None:
        if any(k.endswith("experts.gate_up_proj") for k in module.state_dict()):
            sd = fuse_experts(sd, self.dtype)
        sd = {k: v.to(self.device, self.dtype_of(prefix + k, v)) for k, v in sd.items()}
        module.load_state_dict(sd, strict=True, assign=True)  # strict: any key HF and the checkpoint disagree on fails
        for t in module.state_dict(keep_vars=True).values():
            t._is_hf_initialized = True
        for sub in module.modules():
            for name, buf in sub.named_buffers(recurse=False):
                if buf is not None and buf.is_meta:
                    sub._buffers[name] = torch.empty_like(buf, device=self.device)
            if any(b is not None and not getattr(b, "_is_hf_initialized", False) for b in sub.buffers(recurse=False)):
                sub._is_hf_initialized = False
                self.model._initialize_weights(sub, self.model.is_remote_code())

    def dtype_of(self, name: str, t: torch.Tensor) -> torch.dtype:
        if not t.is_floating_point():  # token -> expert tables and other integer buffers keep their type
            return t.dtype
        return torch.float32 if any(p.search(name) for p in self.fp32) else self.dtype

    @staticmethod
    def release(module: torch.nn.Module) -> None:
        """Back to meta: nothing reads this stage again."""
        module.to_empty(device="meta")


def layer_pattern(name: str) -> str:
    return re.sub(r"\blayers\.\d+\.", "layers.{i}.", name)


class Inventory:
    """Every stored tensor by name pattern: file, shape, dtype, the layers it exists on, and what it is. When a
    per-layer tensor's shape depends on the layer kind (sliding vs full attention, ...), ``shape`` maps kind -> shape.
    """

    def __init__(self, notes: dict[str, str]):
        self.notes, self.entries = notes, {}

    def add(
        self, tensors: dict[str, torch.Tensor], file: str, layer: int | None = None, kind: str | None = None
    ) -> None:
        """``file`` relative to the trace dir, as a template for per-layer files: cache/layer_{i:02d}.safetensors."""
        for name, t in tensors.items():
            pattern = layer_pattern(name)
            e = self.entries.setdefault(
                pattern,
                {
                    "file": file,
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).removeprefix("torch."),
                    **({"layers": []} if layer is not None else {}),
                    **({"meaning": self.notes[pattern]} if pattern in self.notes else {}),
                },
            )
            if layer is not None:
                e["layers"].append(layer)
                e.setdefault("_by_kind", {}).setdefault(kind or "", list(t.shape))

    def finish(self) -> dict:
        for e in self.entries.values():
            by_kind = e.pop("_by_kind", {})
            if len({tuple(s) for s in by_kind.values()}) > 1:
                e["shape"] = by_kind
        return self.entries


def store(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (t.to("cpu", dtype) if t.is_floating_point() else t.cpu()).contiguous()
