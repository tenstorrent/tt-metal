# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Runner side of the fixed-shapes device check (tt-blaze G53F-0313): the REAL common prefill runner
(``prefill_runner.main``: H2D socket, layer acks, KV export to file) with the GLM runtime, plus TEST-ONLY patches:

* ``FS_CHECK_CTL`` holds the arm of the NEXT prompt, read when its chunk at start 0 arrives: ``<0|1> <tag> <S>``;
  the harness sets ``GLM53_PREFILL_FIXED_SHAPES`` to the arm (the model and the runtime read it per call);
* digest of what the KV Manager would ship, without a device read of the slabs: crc32 per DSA layer chained over the
  rows every chunk hands ``GlmRingExport.export_dsa`` (latent rows + pool rows, host tensors: the bytes it packs and
  writes), and crc32 of every KDA layer's final recurrent + conv state on the four row-0 chips (buffer reads, no
  program) -- one ``[fs-check] digest`` line at the prompt's last chunk (actual_end == S);
* program attribution: the program-cache entries each component added during the chunk (``[fs-check] chunk`` line):
  index maintenance, the sparse rows (score / top-k / expansion / attention), the rest of the DSA layer, KDA, the
  decoder layers in all (MoE + mHC included), the export.
"""

from __future__ import annotations

import os
import runpy
import time
import zlib

import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt import dsa as DSA
from models.demos.glm53_flash.tt import functional_decoder as FD
from models.demos.glm53_flash.tt import indexer as IX
from models.demos.glm53_flash.tt import kda as KDA
from models.demos.glm53_flash.tt.runners import kv_export as KX
from models.demos.glm53_flash.tt.runners import runtime as R

_CTL = os.environ["FS_CHECK_CTL"]
_ST = {"mesh": None, "attr": {}, "prompt": None}


def _ctl() -> list:
    try:
        return open(_CTL).read().split()
    except FileNotFoundError:
        return []


def _crc(t: torch.Tensor, crc: int = 0) -> int:
    return zlib.crc32(t.contiguous().view(torch.uint8).numpy().tobytes(), crc) & 0xFFFFFFFF


def _attr(name: str, fn):
    def wrapped(*a, **k):
        mesh = _ST["mesh"]
        if mesh is None:
            return fn(*a, **k)
        n0 = mesh.num_program_cache_entries()
        try:
            return fn(*a, **k)
        finally:
            _ST["attr"][name] = _ST["attr"].get(name, 0) + mesh.num_program_cache_entries() - n0

    return wrapped


IX.TtDSAIndexer.maintain_prefill = _attr("maintain", IX.TtDSAIndexer.maintain_prefill)
DSA.TtDSA._sparse_rows = _attr("sparse", DSA.TtDSA._sparse_rows)
DSA.TtDSA.prefill_forward = _attr("dsa", DSA.TtDSA.prefill_forward)
KDA.TtKDA.prefill_forward = _attr("kda", KDA.TtKDA.prefill_forward)
FD.FunctionalDecoder.prefill_forward = _attr("layers", FD.FunctionalDecoder.prefill_forward)
R.GLM53FlashPrefillRuntime._export = _attr("export", R.GLM53FlashPrefillRuntime._export)

_export_dsa = KX.GlmRingExport.export_dsa


def export_dsa(self, layer, latent, latent_first, pools, pool_first):
    p = _ST["prompt"]
    if p is not None:
        c = p["dsa"].get(layer, 0)
        c = _crc(torch.tensor([latent_first, pool_first]), c)
        c = _crc(latent.to(torch.bfloat16), c)
        if pools is not None:
            c = _crc(pools.to(torch.bfloat16), c)
        p["dsa"][layer] = c
    return _export_dsa(self, layer, latent, latent_first, pools, pool_first)


KX.GlmRingExport.export_dsa = export_dsa

_prefill_chunk = R.GLM53FlashPrefillRuntime.prefill_chunk


def _kda_digest(rt) -> tuple[int, int]:
    rows, cols = tuple(rt.mesh_device.shape)
    rec = conv = 0
    for layer in KX.KDA_LAYERS:
        st = rt.decs[layer].kda_batched
        for name, t in (("rec", st.recurrent), ("conv", st.convolution)):
            devs = ttnn.get_device_tensors(t)
            host = None
            for c in range(cols):
                try:
                    h = ttnn.to_torch(devs[c]) if host is None else ttnn.to_torch(host[c])
                except Exception:  # noqa: BLE001 -- a layout the per-chip read refuses: one whole-mesh read
                    host = ttnn.get_device_tensors(ttnn.from_device(t))
                    h = ttnn.to_torch(host[c])
                if name == "rec":
                    rec = _crc(h, rec)
                else:
                    conv = _crc(h, conv)
    return rec, conv


def prefill_chunk(self, input_tensor, kv_caches, *, slot_id, actual_start, actual_end, request_id=0, **kw):
    _ST["mesh"] = self.mesh_device
    if actual_start == 0 and actual_end > actual_start:
        w = _ctl()
        arm, tag, S = (w + ["0", "-", "0"])[:3] if w else ("0", "-", "0")
        os.environ["GLM53_PREFILL_FIXED_SHAPES"] = "1" if arm == "1" else "0"
        _ST["prompt"] = {
            "tag": tag,
            "S": int(S),
            "fixed": arm == "1",
            "dsa": {},
            "p0": self.mesh_device.num_program_cache_entries(),
            "t": [],
        }
    _ST["attr"] = {}
    n0 = self.mesh_device.num_program_cache_entries()
    t0 = time.perf_counter()
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
    wall = time.perf_counter() - t0
    p = _ST["prompt"]
    if actual_end > actual_start:
        n1 = self.mesh_device.num_program_cache_entries()
        a = _ST["attr"]
        logger.info(
            f"[fs-check] chunk [{actual_start}, {actual_end}) fixed {os.environ.get('GLM53_PREFILL_FIXED_SHAPES')} "
            f"{p['tag'] if p else '-'}: programs +{n1 - n0} (maintain +{a.get('maintain', 0)}, sparse +{a.get('sparse', 0)}, "
            f"dsa +{a.get('dsa', 0)}, kda +{a.get('kda', 0)}, layers +{a.get('layers', 0)}, export +{a.get('export', 0)}) "
            f"total {n1}, wall {wall:.3f} s"
        )
        if p is not None:
            p["t"].append(round(wall, 3))
            if actual_end == p["S"]:
                try:
                    rec, conv = _kda_digest(self)
                except Exception as e:  # noqa: BLE001 -- a failed digest must not take the runner down
                    logger.exception(f"[fs-check] KDA digest failed: {e}")
                    rec = conv = 0
                dsa = 0
                for layer in sorted(p["dsa"]):
                    dsa = zlib.crc32(f"{layer}:{p['dsa'][layer]}".encode(), dsa) & 0xFFFFFFFF
                logger.info(
                    f"[fs-check] digest {p['tag']} S={p['S']} fixed={int(p['fixed'])} dsa={dsa:08x} kda_rec={rec:08x} "
                    f"kda_conv={conv:08x} programs_added={n1 - p['p0']} chunk_wall={p['t']} layers="
                    + ";".join(f"{L}:{c:08x}" for L, c in sorted(p["dsa"].items()))
                )
                _ST["prompt"] = None
    return out


R.GLM53FlashPrefillRuntime.prefill_chunk = prefill_chunk
logger.info(f"[fs-check] harness up: pid {os.getpid()}, control file {_CTL}")
runpy.run_module("models.demos.common.prefill.runners.prefill_runner", run_name="__main__", alter_sys=True)
