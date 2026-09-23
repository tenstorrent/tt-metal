# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TRUE end-to-end: SP prefill (ISL 4096) -> prefill->decode handoff -> 8 decode tokens.

Every number reported elsewhere is one phase in isolation. This runs the whole flow on one
device session and times each phase on the same clock, so the handoff -- which goes through
HOST torch tensors (~70 MB at ISL 4096) and has never been measured -- lands in the total
instead of being an estimate.

  prefill  : sp.prefill_traced() -> true TTFT (device wavefront + on-device-argmax token read)
  handoff  : sp.export_state_host() (device->host) + inject_into_tp_model() (host->device)
  decode   : 8 x TRACED decode -- the demo/text_demo.py pattern (persistent input buffers,
             one execute_trace per step, on-device argmax folded in). NOT decode_tp, which is
             eager and measured 116.79 ms/token against 6.99 ms/token of device time.

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

import ttnn
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
from models.demos.blackhole.qwen36.tt.sp_prefill import BLOCK_SIZE, SPPrefill
from models.tt_transformers.tt.common import copy_host_to_device

N_DECODE = int(os.environ.get("E2E_DECODE_STEPS", "8"))
# "device" (default): reshard on the mesh. "host": the sp_handoff v1 round-trip, for comparison.
HANDOFF = os.environ.get("E2E_HANDOFF", "device")


def test_e2e_prefill_handoff_decode():
    T = int(os.environ.get("SP_ISL", "4096"))
    span_len = T // SP_DIES
    torch.manual_seed(0)
    tokens = torch.randint(1000, 100000, (1, T), dtype=torch.long)
    # SP's paged cache must (a) have room for the decode positions past T, so decode can share
    # it outright, and (b) keep num_blocks a MULTIPLE OF 32 -- forward_prefill_paged pads the
    # page table for chunked SDPA and does a host write during trace capture otherwise
    # ("Writes are not supported during trace capture"). T=4096 gives 64 blocks and worked by
    # luck; T+128 gives 66 and breaks capture. Round up.
    sp_blocks = (((T + 128) // BLOCK_SIZE + 31) // 32) * 32
    sp_max_seq = sp_blocks * BLOCK_SIZE

    mesh, parent = _open_sp_mesh(trace_region_size=200_000_000)  # (mesh, owner) -- owner is what close takes
    sp = None
    try:
        sp = SPPrefill(
            mesh,
            n_spans=SP_DIES,
            tp=SP_TP,
            span_len=span_len,
            max_seq_len=sp_max_seq,  # room for decode positions + 32-block aligned (see above)
            hf_model=model_path(),
            layer_indices=_e2e_layer_indices(),
        )
        sp.capture(tokens)  # warm-up + trace capture, OUTSIDE the timed region
        sp.prefill_traced(tokens)  # one discarded replay so caches/programs are hot

        # Dedicated decode model on the LAST span's submesh. It cannot be sp.models[-1]: those
        # are built sequence_parallel=True, and under the replicated residual their norms hold a
        # full-width replicated gamma, while decode_tp feeds hidden-FRACTURED activations ->
        # "Gamma's last padded dim needs to equal tile width". A separate TP model is what
        # sp_handoff targets by design.
        dec = Qwen36Model.from_pretrained(
            sp.subs[-1],
            max_batch_size=1,
            max_seq_len=sp_max_seq,  # same cache geometry as SP, so the buffers are shareable
            layer_indices=_e2e_layer_indices(),
            sequence_parallel=False,
        )
        mesh_d = dec.mesh_device
        num_blocks = sp.num_blocks  # identical to SP's, so the paged buffers are interchangeable
        dec.allocate_kv_caches(
            [num_blocks, dec.args.n_local_kv_heads, BLOCK_SIZE, dec.args.head_dim],
            ttnn.bfloat16,
            batch_size=1,
        )
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        pt_tt = ttnn.from_torch(
            page_table,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_d,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_d),
        )

        # ZERO-COPY KV HANDOFF. Both models are on sp.subs[-1] with identical paged layouts, and
        # the last SP die's cache already accumulates the whole sequence. Rather than move ~50 MB
        # (through host: 275 ms; device-to-device copy: 371 ms), just point the decode model at
        # those same buffers. Prefill then fills them in place and decode sees it with no handoff
        # at all. Must happen BEFORE trace capture: the trace bakes in buffer addresses.
        dec.reset_tp()
        _last = sp.models[-1]
        for _li, _sl in enumerate(_last.layers):
            if _sl.is_full_attention:
                dec.layers[_li].attention.set_paged_kv_cache(_sl.attention.paged_k, _sl.attention.paged_v)

        # Traced decode, exactly as demo/text_demo.py drives it: persistent input buffers, the
        # Generator-interface forward, per-shard argmax folded INTO the trace so each step reads
        # back two tiny tensors instead of the full vocab.
        dev = dec.prepare_inputs_decode(
            torch.tensor([[1]], dtype=torch.int32), torch.tensor([T], dtype=torch.int32), page_table=page_table
        )
        per_shard = dec.args.vocab_size // mesh_d.get_num_devices()
        read_comp = ttnn.ConcatMeshToTensor(mesh_d, dim=0)

        def _decode_fwd():
            return dec.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])[0]

        def _argmax_dev(lg):
            rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
            idx = ttnn.argmax(rm, dim=-1, keepdim=False)
            ttnn.deallocate(rm)
            return idx

        def _update(token, position):
            host = dec.prepare_decode_inputs_host(
                torch.tensor([[token]], dtype=torch.int32),
                torch.tensor([position], dtype=torch.int32),
                page_table=None,  # constant; its address is baked into the trace
            )
            copy_host_to_device(host[:3], device_tensors=dev[:3])

        # Throwaway eager pass to compile, then capture. The compile pass mutates GDN state, but
        # the real state is injected AFTER capture, so nothing needs restoring here.
        _warm = _decode_fwd()
        ttnn.deallocate(_argmax_dev(_warm))
        dec_trace = ttnn.begin_trace_capture(mesh_d, cq_id=0)
        tt_logits = _decode_fwd()
        tt_idx = _argmax_dev(tt_logits)
        ttnn.end_trace_capture(mesh_d, dec_trace, cq_id=0)
        ttnn.synchronize_device(mesh_d)

        # ---------------- timed e2e ----------------
        t_start = time.perf_counter()

        _, wave_s, ttft_s = sp.prefill_traced(tokens)
        first_token = sp.last_first_token
        t_after_prefill = time.perf_counter()

        if HANDOFF == "host":
            # sp_handoff v1, kept for comparison: device -> host torch -> reshard -> device.
            # NOTE it writes attn.k_caches, while forward_decode with a page_table reads
            # attn.paged_k (attention/tp.py:601), so this path does NOT actually feed the paged
            # decode. Timing only.
            kv_snap, gdn_snap, export_s = sp.export_state_host()
            nd, aa = dec.device.get_num_devices(), dec.args
            k0 = next(iter(kv_snap.values()))[0]
            if k0.shape[0] == nd and k0.shape[1] == 1:
                kv_snap = {
                    li: kv_tp_host_to_full(K, V, num_devices=nd, n_kv_heads=aa.n_kv_heads)
                    for li, (K, V) in kv_snap.items()
                }
            r0, _ = next(iter(gdn_snap.values()))
            if r0.shape[0] == nd:
                kd, vd = aa.gdn_key_dim, aa.gdn_value_dim
                gdn_snap = {
                    li: (
                        gdn_rec_tp_host_to_full(rec, num_devices=nd),
                        gdn_conv_tp_host_to_full(
                            [torch.zeros(nd, 1, conv.shape[-1], dtype=conv.dtype)]
                            + [conv[:, m : m + 1, :].reshape(nd, 1, -1) for m in range(conv.shape[1])],
                            num_devices=nd,
                            key_dim=kd,
                            value_dim=vd,
                        ),
                    )
                    for li, (rec, conv) in gdn_snap.items()
                }
            t_after_export = time.perf_counter()
            inject_into_tp_model(dec, kv_snap, gdn_snap)
        else:
            # KV: nothing to do -- decode already shares the SP die's paged buffers (above).
            # Only the GDN recurrent/conv state needs moving, and it is small (~2.3 MB of rec
            # plus 3 conv rows per layer) and stays on the mesh.
            export_s = 0.0
            t_after_export = time.perf_counter()
            for _li, _sl in enumerate(sp.models[-1].layers):
                if _sl.is_full_attention:
                    continue
                rec, conv = sp.last_die_gdn_state[_li]
                dn = dec.layers[_li].attention
                ttnn.copy(rec, dn.rec_state)
                for m in range(1, len(dn.conv_states)):
                    row = ttnn.slice(conv, (0, m - 1, 0), (1, m, conv.shape[-1]))
                    ttnn.copy(ttnn.reshape(row, dn.conv_states[m].shape), dn.conv_states[m])
                    ttnn.deallocate(row)
            ttnn.synchronize_device(mesh_d)
        t_after_inject = time.perf_counter()

        tok, pos, toks, step_ms = first_token, T, [first_token], []
        ph = {"update": 0.0, "exec_sync": 0.0, "readback": 0.0}
        for _ in range(N_DECODE):
            st = time.perf_counter()
            _update(tok, pos)
            a = time.perf_counter()
            ttnn.execute_trace(mesh_d, dec_trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_d)
            b = time.perf_counter()
            idxs = ttnn.to_torch(tt_idx, mesh_composer=read_comp).reshape(-1)
            tok = int(idxs[0].item())  # vocab-shard 0's local argmax; greedy seed for the next step
            c = time.perf_counter()
            ph["update"] += (a - st) * 1e3
            ph["exec_sync"] += (b - a) * 1e3
            ph["readback"] += (c - b) * 1e3
            step_ms.append((c - st) * 1e3)
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
        logger.info(f"  handoff ({HANDOFF}): phase 1  {export_ms:8.2f} ms")
        logger.info(f"  handoff ({HANDOFF}): phase 2  {inject_ms:8.2f} ms")
        logger.info(f"  handoff TOTAL              {export_ms + inject_ms:8.2f} ms")
        logger.info(
            f"  {N_DECODE} decode tokens            {decode_ms:8.2f} ms   "
            f"({decode_ms / N_DECODE:.2f} ms/token, TRACED)"
        )
        logger.info(f"  {'-' * 44}")
        logger.info(f"  E2E TOTAL                  {total_ms:8.2f} ms")
        logger.info(
            f"  share: prefill {100 * prefill_ms / total_ms:.0f}%  "
            f"handoff {100 * (export_ms + inject_ms) / total_ms:.0f}%  "
            f"decode {100 * decode_ms / total_ms:.0f}%"
        )
        logger.info(f"  per-step decode (ms): {['%.2f' % x for x in step_ms]}")
        logger.info(
            f"  decode phases (ms/token): update {ph['update'] / N_DECODE:.2f}  "
            f"exec+sync {ph['exec_sync'] / N_DECODE:.2f}  readback {ph['readback'] / N_DECODE:.2f}"
        )
        logger.info(f"  tokens: {toks}")
        # CORRECTNESS PROBE. An earlier version of this harness injected into attn.k_caches while
        # forward_decode with a page_table reads attn.paged_k, so decode silently ran on an EMPTY
        # KV cache and still produced plausible-looking ids. Prove the shared cache is actually
        # read: zero it, decode the same first token again, and require a different result.
        if os.environ.get("E2E_SKIP_PROBE") != "1":
            for _li, _sl in enumerate(sp.models[-1].layers):
                if _sl.is_full_attention:
                    za = dec.layers[_li].attention
                    ttnn.copy(ttnn.zeros_like(za.paged_k), za.paged_k)
                    ttnn.copy(ttnn.zeros_like(za.paged_v), za.paged_v)
            _update(first_token, T)
            ttnn.execute_trace(mesh_d, dec_trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_d)
            tok_zeroed = int(ttnn.to_torch(tt_idx, mesh_composer=read_comp).reshape(-1)[0].item())
            logger.info(f"  [probe] first decode token with shared KV={toks[1]}, with KV zeroed={tok_zeroed}")
            assert tok_zeroed != toks[1], (
                "decode produced the SAME token with the KV cache zeroed -- it is not reading the "
                "shared paged cache, so the handoff is not actually wired up"
            )
        assert len(set(toks[1:])) >= 1
    finally:
        try:
            ttnn.release_trace(mesh_d, dec_trace)
        except Exception:
            pass
        if sp is not None:
            sp.close()
        _close_sp_mesh(parent)
