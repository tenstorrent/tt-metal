# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Dump the Q / K / V of a diffusion model's attention calls during one SGLang run, in bench_tt.py's input format.

Runs any `sglang ...` command line with a hook on the attention backends of `sglang.multimodal_gen` (every
`*Impl` class under the backends package that defines `forward(query, key, value, attn_metadata)` on NHD
[B, T, H, D] tensors). The hook is installed through a generated `sitecustomize.py`, so it reaches the worker
processes SGLang spawns. Each selected call is written to `<out>/qkv_l{layer}_s{step}_r{rank}.safetensors`.

  python capture_qkv_sglang.py --out /data/qkv --layers 2,20 --steps 12,30 --text-len 512 -- \
      sglang generate --model-path <MiniMax-H3 checkpoint> --prompt "..." --attention-backend fa ...

Assumptions (checked where possible, recorded in the file metadata otherwise):
- Each attention layer owns one backend impl instance; layer = order in which instances are first called,
  step = number of earlier calls on the same instance (divided by --calls-per-step, 2 for unbatched CFG).
- Only calls with --heads heads of --head-dim and at least --min-tokens tokens are counted (skips the text encoder).
- Run on one GPU (no sequence / Ulysses / tensor parallelism), so a call sees the whole sequence and all heads.
  On more ranks every rank writes its own local tensors and `world_size` is recorded; bench_tt.py rejects
  tensors without the full head count.
- Text length, segment lengths and the video token grid are not part of the backend interface: pass --text-len /
  --segments (packed order, e.g. text=512,condition=0,audio=760,video=36488) / --video-grid (latent frames,
  height, width in tokens, e.g. 37,24,42); attributes named text_len / num_text_tokens on the impl or the metadata
  are recorded when present. bench_tt.py's video_sparse_attn / cube_sparse_attn need --segments and --video-grid.
"""

from __future__ import annotations

import argparse
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading

PKG = "sglang.multimodal_gen.runtime.layers.attention.backends"
ENV = "SPARSE_RING_QKV_CAPTURE"  # JSON settings for the hook in the SGLang processes

_settings = None
_layers: dict[int, int] = {}  # id(impl) -> layer index
_calls: dict[int, int] = {}  # id(impl) -> counted calls
_saved = 0
_local = threading.local()
_lock = threading.Lock()


def _probe(objs, names):
    for o in objs:
        for n in names:
            v = getattr(o, n, None)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return v
    return None


def _maybe_dump(impl, query, key, value, attn_metadata):
    global _saved
    s = _settings
    if query.dim() != 4 or query.shape[2] != s["heads"] or query.shape[3] != s["head_dim"]:
        return
    if query.shape[1] < s["min_tokens"]:
        return
    with _lock:
        layer = _layers.setdefault(id(impl), len(_layers))
        call = _calls.get(id(impl), 0)
        _calls[id(impl)] = call + 1
        step = call // s["calls_per_step"]
        if layer not in s["layers"] or step not in s["steps"] or call % s["calls_per_step"] or _saved >= s["max_files"]:
            return
        _saved += 1
    import torch
    from safetensors.torch import save_file

    rank, world = 0, 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    meta = {
        "layer": layer,
        "step": step,
        "world_size": world,
        "rank": rank,
        "backend": type(impl).__name__,
        "layout": "BTHD",
        "source": "sglang",
    }
    text_len = s.get("text_len") or _probe([impl, attn_metadata], ("text_len", "num_text_tokens", "text_seq_len"))
    if text_len is not None:
        meta["text_len"] = int(text_len)
    if s.get("segments"):
        meta["segments"] = json.dumps(s["segments"])
    if s.get("video_grid"):
        meta["video_grid"] = json.dumps(s["video_grid"])
    if s.get("seconds"):
        meta["seconds"] = s["seconds"]
    timestep = _probe([attn_metadata], ("current_timestep", "timestep"))
    if timestep is not None:
        meta["timestep"] = timestep
    tensors = {n: t.detach().to(torch.bfloat16).contiguous().cpu() for n, t in (("q", query), ("k", key), ("v", value))}
    os.makedirs(s["out"], exist_ok=True)
    path = os.path.join(s["out"], f"qkv_l{layer}_s{step}_r{rank}.safetensors")
    save_file(tensors, path, metadata={k: str(v) for k, v in meta.items()})
    print(f"[capture_qkv] {path} {tuple(query.shape)}", file=sys.stderr, flush=True)


def _wrap(cls):
    orig = cls.forward

    def forward(self, query, key, value, *args, **kwargs):
        if not getattr(_local, "busy", 0):  # a sparse backend that falls back to a dense one: capture once
            attn_metadata = args[0] if args else kwargs.get("attn_metadata")
            _maybe_dump(self, query, key, value, attn_metadata)
        _local.busy = getattr(_local, "busy", 0) + 1
        try:
            return orig(self, query, key, value, *args, **kwargs)
        finally:
            _local.busy -= 1

    cls.forward = forward


def _patch_module(module):
    for name, obj in list(vars(module).items()):
        if isinstance(obj, type) and obj.__module__ == module.__name__ and name.endswith("Impl"):
            if "forward" in obj.__dict__ and not getattr(obj, "_qkv_capture", False):
                _wrap(obj)
                obj._qkv_capture = True


class _Finder(importlib.abc.MetaPathFinder):
    """Patches each backend module right after it is first imported."""

    def __init__(self):
        self.seen = set()

    def find_spec(self, name, path, target=None):
        if not name.startswith(PKG + ".") or name in self.seen:
            return None
        self.seen.add(name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        exec_module = spec.loader.exec_module

        def patched(module):
            exec_module(module)
            _patch_module(module)

        spec.loader.exec_module = patched
        return spec


def install():
    """Called from the generated sitecustomize in every SGLang process."""
    global _settings
    raw = os.environ.get(ENV)
    if not raw or _settings is not None:
        return
    _settings = json.loads(raw)
    _settings["layers"], _settings["steps"] = set(_settings["layers"]), set(_settings["steps"])
    for name, module in list(sys.modules.items()):
        if name.startswith(PKG + ".") and module is not None:
            _patch_module(module)
    sys.meta_path.insert(0, _Finder())


def _ints(s):
    return [int(x) for x in s.split(",") if x]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--layers", type=_ints, required=True, help="comma list of attention layer indices")
    p.add_argument("--steps", type=_ints, required=True, help="comma list of denoise step indices")
    p.add_argument("--text-len", type=int, default=None)
    p.add_argument("--segments", default=None, help="packed segment lengths, e.g. text=512,audio=760,video=36488")
    p.add_argument("--video-grid", type=_ints, default=None, help="video token grid t,h,w, e.g. 37,24,42 (5 s 768p)")
    p.add_argument("--seconds", type=int, default=None, help="recorded only (5 / 10 / 15)")
    p.add_argument("--heads", type=int, default=56)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--min-tokens", type=int, default=4096)
    p.add_argument("--calls-per-step", type=int, default=1)
    p.add_argument("--max-files", type=int, default=16)
    p.add_argument("cmd", nargs=argparse.REMAINDER, help="-- sglang ...")
    a = p.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        p.error("give the SGLang command after --")
    segments = None
    if a.segments:
        segments = {k: int(v) for k, v in (kv.split("=") for kv in a.segments.split(","))}
    settings = dict(
        out=os.path.abspath(a.out),
        layers=a.layers,
        steps=a.steps,
        text_len=a.text_len if a.text_len is not None else (segments or {}).get("text"),
        segments=segments,
        video_grid=a.video_grid,
        seconds=a.seconds,
        heads=a.heads,
        head_dim=a.head_dim,
        min_tokens=a.min_tokens,
        calls_per_step=a.calls_per_step,
        max_files=a.max_files,
    )
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as hook_dir:
        with open(os.path.join(hook_dir, "sitecustomize.py"), "w") as f:
            f.write(
                f"import sys\nsys.path.insert(0, {here!r})\nimport capture_qkv_sglang\ncapture_qkv_sglang.install()\n"
            )
        env = dict(os.environ)
        env[ENV] = json.dumps(settings)
        env["PYTHONPATH"] = os.pathsep.join(x for x in (hook_dir, env.get("PYTHONPATH")) if x)
        sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
