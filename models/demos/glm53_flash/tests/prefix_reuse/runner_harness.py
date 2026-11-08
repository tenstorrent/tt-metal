# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Runner side of the prefix-reuse device check (tt-blaze G53F-0311): the REAL common prefill runner
(``prefill_runner.main``: H2D socket, layer-ack router, KV export to file) with the GLM runtime and
``GLM53F_PREFIX_REUSE=1``, plus TEST-ONLY patches, so one model build measures every arm on the same programs.

PREFIX_CHECK_CTL holds the arm of the NEXT prompt (read when a chunk at start 0 arrives): ``<mode> <tag> [digest]``
    off    retention bypassed (today's runner) and emptied
    cold   retention emptied before the prompt: every chunk computed (checkpoints still saved)
    reuse  retention as served
An EMPTY chunk at 0 is a flush: the harness then digests what the KV Manager would ship for the previous prompt, if its
arm asked for ``digest``: crc32 of every DSA layer's export-slab latent pages [0, ceil(S / 32)) and pool rows
[0, ceil(S / 128)) (read back from the chip the table names for that layer), and of the KDA recurrent + conv slabs of
the four TP columns (row 0) -- a ``[prefix-check] digest`` line. SIGUSR1 dumps every Python thread's stack.
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
from models.demos.glm53_flash.tt.runners import runtime as R
from models.demos.glm53_flash.tt.runners.kv_export import LATENT_PAGE_BYTES, POOL_CHUNK_TOKENS, POOL_ROW_BYTES, TPP

_STACKS = open(os.environ.get("PREFIX_CHECK_STACKS", "/tmp/prefix_check_stacks.txt"), "a")
faulthandler.register(signal.SIGUSR1, file=_STACKS, all_threads=True, chain=False)
_CTL = os.environ["PREFIX_CHECK_CTL"]


def _ctl() -> list:
    try:
        words = open(_CTL).read().split()
    except FileNotFoundError:
        words = []
    return words or ["reuse", "-"]


def _crc(t: torch.Tensor) -> int:
    return zlib.crc32(t.contiguous().view(torch.uint8).numpy().tobytes()) & 0xFFFFFFFF


class _Reader:
    """Device ``k``'s shard of mesh tensor ``t``: one chip's command-queue read, or -- if the per-device path refuses
    the layout (e.g. ND-sharded) -- one whole-mesh read shared by every shard asked for in this digest."""

    def __init__(self, t):
        self.t, self.host = t, None

    def __call__(self, k: int) -> torch.Tensor:
        if self.host is None:
            try:
                return ttnn.to_torch(ttnn.get_device_tensors(self.t)[k])
            except Exception:  # noqa: BLE001
                self.host = ttnn.get_device_tensors(ttnn.from_device(self.t))
        return ttnn.to_torch(self.host[k])


def _digest(rt, kv, S: int) -> dict:
    t0 = time.perf_counter()
    npg, nrow = -(-S // TPP), -(-S // POOL_CHUNK_TOKENS)
    per_layer, crc_all, notes = {}, 0, []
    read_lat, read_pool = _Reader(kv.latent), _Reader(kv.pool)
    for i, layer in enumerate(kv.dsa_layers):
        lat = read_lat(i).reshape(-1, LATENT_PAGE_BYTES)[:npg]
        pool = read_pool(i).reshape(-1, POOL_ROW_BYTES)[:nrow]
        if i in (0, len(kv.dsa_layers) - 1):  # the command-queue read must be the chip the export raw-writes
            raw = kv._banks.read(kv._dev_ids[i], 0, kv.latent.buffer_address(), LATENT_PAGE_BYTES)
            if bytes(lat[0].contiguous().numpy().tobytes()) != bytes(raw):
                notes.append(f"layer {layer}: command-queue page 0 != raw page 0 of chip {kv._dev_ids[i]}")
        c = (_crc(lat), _crc(pool))
        per_layer[layer] = c
        crc_all = zlib.crc32(f"{layer}:{c}".encode(), crc_all)
    kda = []
    rows, cols = tuple(rt.mesh_device.shape)
    for t in (kv.kda_rec, kv.kda_conv):
        read = _Reader(t)
        kda.append([_crc(read(c)) for c in range(cols)])
        # SP replicas must agree (the KV Manager may read any row): the last row's column 0
        if _crc(read((rows - 1) * cols)) != kda[-1][0]:
            notes.append("KDA slab replicas differ across SP rows")
    return {
        "notes": notes,
        "S": S,
        "dsa": crc_all,
        "kda_rec": kda[0],
        "kda_conv": kda[1],
        "layers": per_layer,
        "digest_s": round(time.perf_counter() - t0, 2),
    }


_prefill_chunk = R.GLM53FlashPrefillRuntime.prefill_chunk


def prefill_chunk(self, input_tensor, kv_caches, *, slot_id, actual_start, actual_end, request_id=0, **kw):
    st = self.__dict__.setdefault("_chk", {"pending": None, "tag": "-", "digest": False, "end": 0})
    if actual_start == 0 and getattr(self, "_prefix", None) is not None and self._prefix_on is not None:
        if actual_end <= actual_start:  # a flush: digest the previous prompt
            if st["pending"] is not None:
                tag, S = st["pending"]
                st["pending"] = None
                try:
                    d = _digest(self, kv_caches, S)
                    if d["notes"]:
                        logger.warning(f"[prefix-check] digest {tag} notes: {d['notes']}")
                    logger.info(
                        f"[prefix-check] digest {tag} S={S} dsa={d['dsa']:08x} "
                        f"kda_rec={','.join(f'{x:08x}' for x in d['kda_rec'])} "
                        f"kda_conv={','.join(f'{x:08x}' for x in d['kda_conv'])} ({d['digest_s']} s) layers="
                        + ";".join(f"{L}:{a:08x}/{b:08x}" for L, (a, b) in d["layers"].items())
                    )
                except Exception as e:  # noqa: BLE001 -- a failed digest must not take the runner down
                    logger.exception(f"[prefix-check] digest {tag} FAILED: {e}")
        else:
            words = _ctl()
            mode, tag = words[0], (words[1] if len(words) > 1 else "-")
            if mode == "off":
                self._prefix.reset("check: arm off")
                self._prefix_on = False
            elif mode == "cold":
                self._prefix.reset("check: arm cold")
                self._prefix_on = True
            else:
                self._prefix_on = True
            st["tag"], st["digest"] = tag, "digest" in words
            logger.info(
                f"[prefix-check] chunk {request_id}: prompt {tag} arm {mode}{' digest' if st['digest'] else ''}"
            )
    out = _prefill_chunk(
        self,
        input_tensor,
        kv_caches,
        slot_id=slot_id,
        actual_start=actual_start,
        actual_end=actual_end,
        request_id=request_id,
        **kw,
    )
    if actual_end > actual_start and st["digest"]:
        st["pending"] = (st["tag"], actual_end)
    return out


R.GLM53FlashPrefillRuntime.prefill_chunk = prefill_chunk
logger.info(f"[prefix-check] harness up: pid {os.getpid()}, control file {_CTL}")
runpy.run_module("models.demos.common.prefill.runners.prefill_runner", run_name="__main__", alter_sys=True)
