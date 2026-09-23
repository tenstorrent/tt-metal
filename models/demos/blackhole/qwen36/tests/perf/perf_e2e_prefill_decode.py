# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TRUE end-to-end: SP prefill (ISL 4096) -> prefill->decode handoff -> 8 decode tokens.

Every number reported elsewhere is one phase in isolation. This runs the whole flow on one
device session and times each phase on the same clock, so the handoff -- which goes through
HOST torch tensors (~70 MB at ISL 4096) and has never been measured -- lands in the total
instead of being an estimate.

  prefill  : sp.prefill_traced() -> true TTFT (device wavefront + on-device-argmax token read)
  handoff  : sp.export_state_host() (device->host) + inject_into_tp_model() (host->device)
  decode   : 8 x model.decode_tp(), greedy, seeded from prefill's own first token

The decode model is the LAST SP submesh's model: it already holds the weights and the whole
4096-token KV, so this measures the handoff's real cost (reshard + round-trip) without paying
for a fifth model's weights. decode_tp reads attn.k_caches, a different buffer from prefill's
attn.paged_k, which is exactly why the handoff exists.

    export HF_MODEL=Qwen/Qwen3.5-2B TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0
    SP_DIES=4 SP_TP=8 QWEN36_NO_AGMM=1 pytest \
      models/demos/blackhole/qwen36/tests/perf/perf_e2e_prefill_decode.py -sv
"""
import os
import time

import torch
from loguru import logger

from models.demos.blackhole.qwen36.tests.test_factory import model_path
from models.demos.blackhole.qwen36.tests.test_sp_prefill import (
    SP_DIES,
    SP_TP,
    _close_sp_mesh,
    _e2e_layer_indices,
    _open_sp_mesh,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.sp_handoff import (
    gdn_conv_tp_host_to_full,
    gdn_rec_tp_host_to_full,
    inject_into_tp_model,
    kv_tp_host_to_full,
)
from models.demos.blackhole.qwen36.tt.sp_prefill import SPPrefill

N_DECODE = int(os.environ.get("E2E_DECODE_STEPS", "8"))


def test_e2e_prefill_handoff_decode():
    T = int(os.environ.get("SP_ISL", "4096"))
    span_len = T // SP_DIES
    torch.manual_seed(0)
    tokens = torch.randint(1000, 100000, (1, T), dtype=torch.long)

    mesh, parent = _open_sp_mesh(trace_region_size=200_000_000)  # (mesh, owner) -- owner is what close takes
    sp = None
    try:
        sp = SPPrefill(
            mesh,
            n_spans=SP_DIES,
            tp=SP_TP,
            span_len=span_len,
            max_seq_len=T,
            hf_model=model_path(),
            layer_indices=_e2e_layer_indices(),
        )
        sp.capture(tokens)  # warm-up + trace capture, OUTSIDE the timed region
        sp.prefill_traced(tokens)  # one discarded replay so caches/programs are hot

        # Dedicated decode model on the LAST span's submesh. It cannot be sp.models[-1]: those
        # are built sequence_parallel=True, and under the replicated residual their norms hold a
        # full-width replicated gamma, while decode_tp feeds hidden-FRACTURED activations ->
        # "Gamma's last padded dim needs to equal tile width". A separate TP model is what
        # sp_handoff targets by design. Built and warmed OUTSIDE the timed region; only the
        # state transfer and the decode steps are timed.
        dec = Qwen36Model.from_pretrained(
            sp.subs[-1],
            max_batch_size=1,
            max_seq_len=T + 128,  # decode runs positions T..T+N-1; must be a multiple of the 128 SDPA chunk
            layer_indices=_e2e_layer_indices(),
            sequence_parallel=False,
        )
        dec.reset_tp()
        dec.decode_tp(1, 0)  # compile the decode programs
        dec.reset_tp()

        # ---------------- timed e2e ----------------
        t_start = time.perf_counter()

        _, wave_s, ttft_s = sp.prefill_traced(tokens)
        first_token = sp.last_first_token
        t_after_prefill = time.perf_counter()

        kv_snap, gdn_snap, export_s = sp.export_state_host()

        # export_state_host's docstring says it returns the FULL layout, but ttnn.to_torch with
        # ConcatMeshToTensor(dim=0) actually yields the per-device STACKED form
        # ([nd,1,S,HD] for KV), which is what inject_into_tp_model re-shards FROM full. Convert
        # with the inverse helpers. This host-side reshape is part of the handoff, so it stays
        # inside the timed region.
        nd = dec.device.get_num_devices()
        a = dec.args
        k0 = next(iter(kv_snap.values()))[0]
        logger.info(f"[e2e] exported KV shape {tuple(k0.shape)} (nd={nd}, n_kv_heads={a.n_kv_heads})")
        if k0.shape[0] == nd and k0.shape[1] == 1:
            kv_snap = {
                li: kv_tp_host_to_full(K, V, num_devices=nd, n_kv_heads=a.n_kv_heads) for li, (K, V) in kv_snap.items()
            }
        r0, c0 = next(iter(gdn_snap.values()))
        logger.info(f"[e2e] exported GDN rec {tuple(r0.shape)} conv {tuple(c0.shape)}")
        if r0.shape[0] == nd:
            kd, vd = a.gdn_key_dim, a.gdn_value_dim
            fixed = {}
            for li, (rec, conv) in gdn_snap.items():
                rec_f = gdn_rec_tp_host_to_full(rec, num_devices=nd)
                cl = [torch.zeros(nd, 1, conv.shape[-1], dtype=conv.dtype)] + [
                    conv[:, m : m + 1, :].reshape(nd, 1, -1) for m in range(conv.shape[1])
                ]
                fixed[li] = (rec_f, gdn_conv_tp_host_to_full(cl, num_devices=nd, key_dim=kd, value_dim=vd))
            gdn_snap = fixed
        t_after_export = time.perf_counter()

        dec.reset_tp()
        inject_into_tp_model(dec, kv_snap, gdn_snap)
        t_after_inject = time.perf_counter()

        tok, pos, toks, step_ms = first_token, T, [first_token], []
        for _ in range(N_DECODE):
            s = time.perf_counter()
            lg = dec.decode_tp(tok, pos)
            tok = int(torch.argmax(lg))
            step_ms.append((time.perf_counter() - s) * 1e3)
            toks.append(tok)
            pos += 1
        t_end = time.perf_counter()
        # -------------------------------------------

        ms = lambda a, b: (b - a) * 1e3
        prefill_wall_ms = ms(t_start, t_after_prefill)
        # prefill_traced stamps ttft_s after the ON-DEVICE-argmax token read, then keeps going to
        # pull the full [1,1,32,vocab] logits back for PCC/debug. Production reads only the
        # token, so true TTFT is ttft_s; the remainder is harness-only and reported separately.
        prefill_ms = ttft_s * 1e3
        debug_readback_ms = prefill_wall_ms - prefill_ms
        export_ms = ms(t_after_prefill, t_after_export)
        inject_ms = ms(t_after_export, t_after_inject)
        decode_ms = ms(t_after_inject, t_end)
        total_ms = prefill_ms + export_ms + inject_ms + decode_ms  # excludes the debug readback

        logger.info("=" * 68)
        logger.info(f"TRUE E2E: ISL {T}, SP={SP_DIES} x TP={SP_TP}, {N_DECODE} decode tokens")
        logger.info("=" * 68)
        logger.info(f"  prefill (true TTFT)        {prefill_ms:8.2f} ms   [wavefront {wave_s * 1e3:.2f}]")
        logger.info(f"    (+ debug-only full-logits readback, NOT counted: {debug_readback_ms:.2f} ms)")
        logger.info(f"  handoff: export dev->host  {export_ms:8.2f} ms")
        logger.info(f"  handoff: inject host->dev  {inject_ms:8.2f} ms")
        logger.info(f"  handoff TOTAL              {export_ms + inject_ms:8.2f} ms")
        logger.info(
            f"  {N_DECODE} decode tokens            {decode_ms:8.2f} ms   "
            f"({decode_ms / N_DECODE:.2f} ms/token, EAGER -- decode_tp is untraced)"
        )
        logger.info(f"  {'-' * 44}")
        logger.info(f"  E2E TOTAL                  {total_ms:8.2f} ms")
        logger.info(
            f"  share: prefill {100 * prefill_ms / total_ms:.0f}%  "
            f"handoff {100 * (export_ms + inject_ms) / total_ms:.0f}%  "
            f"decode {100 * decode_ms / total_ms:.0f}%"
        )
        logger.info(f"  per-step decode (ms): {['%.2f' % x for x in step_ms]}")
        logger.info(f"  tokens: {toks}")
        assert len(set(toks[1:])) >= 1
    finally:
        if sp is not None:
            sp.close()
        _close_sp_mesh(parent)
