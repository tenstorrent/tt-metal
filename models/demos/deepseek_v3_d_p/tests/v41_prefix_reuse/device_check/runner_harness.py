# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runner side of the V4.1 prefix-reuse H1 device check: the REAL common prefill runner (``prefill_runner.main``: H2D socket,
layer-ack channel, KV table exported to file) with the V4.1 runtime and ``V41_PREFILL_PREFIX_REUSE=1``, plus TEST-ONLY
patches, so one model build measures every arm on the same programs. Started by ``run_device_check.sh``.

``PREFIX_CHECK_CTL`` holds the arm of the NEXT prompt (read when a chunk at start 0 arrives): ``<mode> <tag> [digest]``
    off    prefix reuse bypassed (today's runner) and its retention emptied
    cold   retention emptied before the prompt: every chunk computed (checkpoints still saved)
    reuse  retention as served
An EMPTY chunk at 0 is a FLUSH: if the previous prompt's arm asked for ``digest``, the harness writes a
``[prefix-check] digest`` line -- crc32 of every export tensor of slot 0 (the 7 tensors whole, rows past the prompt
included: what the KV Manager could read; chip 0, window rings also on the last chip) and of the model's own rows
[0, S / ratio) of every KV source and layer 20 (entries + index keys) -- then hands the chunk to the runtime, whose EMPTY-chunk
path (tt-d-gen prefill_wait_for_arm_us) acknowledges every layer and computes nothing: every prompt after a flush is also a
"declined request, then a prompt" check. SIGUSR1 dumps every Python thread's stack.
"""

from __future__ import annotations

import faulthandler
import os
import runpy
import signal
import time
import zlib

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.v41 import runtime as R

_STACKS = open(os.environ.get("PREFIX_CHECK_STACKS", "/tmp/v41_prefix_check_stacks.txt"), "a")
faulthandler.register(signal.SIGUSR1, file=_STACKS, all_threads=True, chain=False)
_CTL = os.environ["PREFIX_CHECK_CTL"]


def _ctl() -> list:
    try:
        words = open(_CTL).read().split()
    except FileNotFoundError:
        words = []
    return words or ["reuse", "-"]


def _crc(t: torch.Tensor, acc: int = 0) -> int:
    return zlib.crc32(t.contiguous().view(torch.uint8).numpy().tobytes(), acc) & 0xFFFFFFFF


def _read(t, k: int = 0) -> torch.Tensor:
    """Chip ``k``'s copy of a replicated mesh tensor (kv_export.read_back's read)."""
    return ttnn.to_torch(ttnn.get_device_tensors(t)[k])


def _digest(rt, kv, S: int) -> dict:
    t0 = time.perf_counter()
    notes, exp = [], {}
    tensors = kv.tensors()
    for name, t in tensors.items():
        exp[name] = _crc(_read(t, 0))
    last = len(ttnn.get_device_tensors(kv.swa)) - 1
    for name in ("swa", "win"):
        if _crc(_read(tensors[name], last)) != exp[name]:
            notes.append(f"{name}: chip {last} != chip 0")
    acc = 0
    for name in sorted(exp):
        acc = zlib.crc32(f"{name}:{exp[name]:08x}".encode(), acc)
    pf, rows, racc = rt.pf, {}, 0
    mods = [(b.layer, b.state) for b in pf.blocks if pf.cfg.role(b.layer).mode == "full"]
    if pf.kv_attn is not None:
        mods.append((pf.kv_only_layer, pf.kv_state))
    for L, st in mods:
        n = -(-S // pf.cfg.role(L).compress_ratio)
        n = -(-n // 32) * 32
        c = 0
        for src in (st.compressed_kv, st.index_k):
            part = ttnn.slice(src, [0, 0, 0, 0], [1, 1, n, int(src.shape[3])])
            c = _crc(_read(part, 0), c)
            ttnn.deallocate(part)
        rows[L] = c
        racc = zlib.crc32(f"{L}:{c:08x}".encode(), racc)
    return {
        "S": S,
        "export": acc,
        "tensors": exp,
        "rows": racc,
        "notes": notes,
        "digest_s": round(time.perf_counter() - t0, 2),
    }


_prefill_chunk = R.V41PrefillRuntime.prefill_chunk


def prefill_chunk(self, input_tensor, kv_caches, *, metadata_msg=None, **kw):
    st = self.__dict__.setdefault("_chk", {"pending": None, "tag": "-", "digest": False})
    start, end, request_id = int(kw["actual_start"]), int(kw["actual_end"]), int(kw.get("request_id", 0))
    # a flush: digest the previous prompt, then the runtime's own EMPTY-chunk path (acks, nothing else)
    if end <= start:
        if st["pending"] is not None:
            tag, S = st["pending"]
            st["pending"] = None
            try:
                d = _digest(self, kv_caches, S)
                if d["notes"]:
                    logger.warning(f"[prefix-check] digest {tag} notes: {d['notes']}")
                logger.info(
                    f"[prefix-check] digest {tag} S={S} export={d['export']:08x} rows={d['rows']:08x} "
                    f"({d['digest_s']} s) tensors=" + ";".join(f"{k}:{v:08x}" for k, v in d["tensors"].items())
                )
            except Exception as e:  # noqa: BLE001 -- a failed digest must not take the runner down
                logger.exception(f"[prefix-check] digest {tag} FAILED: {e}")
        # every flush is also the tt-d-gen prefill_wait_for_arm_us EMPTY chunk: each next prompt checks the runtime left the
        # slot as a cold prompt (or, with reuse, a retained prefix) expects
        return _prefill_chunk(self, input_tensor, kv_caches, metadata_msg=metadata_msg, **kw)
    if start == 0:
        words = _ctl()
        mode, tag = words[0], (words[1] if len(words) > 1 else "-")
        pr = self._prefix
        assert pr is not None, "the check needs V41_PREFILL_PREFIX_REUSE=1"
        if mode == "off":
            pr.reset_all("check: arm off")
            pr.on = False
        elif mode == "cold":
            pr.reset_all("check: arm cold")
            pr.on = True
        else:
            pr.on = True
        st["tag"], st["digest"] = tag, "digest" in words
        logger.info(f"[prefix-check] chunk {request_id}: prompt {tag} arm {mode}{' digest' if st['digest'] else ''}")
    out = _prefill_chunk(self, input_tensor, kv_caches, metadata_msg=metadata_msg, **kw)
    if st["digest"]:
        st["pending"] = (st["tag"], end)
    return out


R.V41PrefillRuntime.prefill_chunk = prefill_chunk
logger.info(f"[prefix-check] harness up: pid {os.getpid()}, control file {_CTL}")
runpy.run_module("models.demos.common.prefill.runners.prefill_runner", run_name="__main__", alter_sys=True)
