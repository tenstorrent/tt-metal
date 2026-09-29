# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ONE-process end-to-end: SP prefill (SPPrefillSC, 4 unit submeshes, ISL 4096, traced) -> device-side handoff
(tt/sp_tp_handoff.py, traced per submesh) -> traced TP=4 decode on the parent 1x4 mesh (7 steps).

Needs QWEN36_SP_IMPL=sc and the SC env (demo/sp_sc_flags.env). Real chat prompt (QWEN36_E2E_REAL_PROMPT
tokens of test_sp_prefill._real_prompt_tokens); expected greedy tokens [760, 1414, 33241, 279, 3712, 314, 19820, 10903].

Phase order (see sp_tp_handoff.py for why): TP4 model build + eager decode warm-up (parent) -> reserve subs<-parent
-> SP build -> handoff build + eager (compile) -> eager SP prefill + eager handoff (+ state check) -> SP capture ->
handoff capture -> reserve parent<-subs -> prep (conv_hist refresh) + decode trace capture -> reserve subs<-parent ->
timed repetitions.

Env: E2E_REPS (5), E2E_OVERLAP (1: dies 0-2 start their handoff trace right behind their SP trace; 0: after TTFT),
E2E_CHECK_STATE (1: compare the TP state after the eager handoff against SPPrefillSC.export_state_host),
E2E_TRACE_MB (384), E2E_TP_KV_BF16 (1: TP4 paged KV bf16, cast on the source die -- matches the expected tokens; 0: bf8 like SP -- 2 near-tie tokens flip).
"""
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path
from models.demos.blackhole.qwen36.tests.test_sp_prefill import _e2e_tokenizer, _real_prompt_tokens, _sp_set_fabric
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.model_config import GDN_CONV1D_L1_SMALL_SIZE
from models.demos.blackhole.qwen36.tt.sp_handoff import gdn_conv_full_to_tp_host
from models.demos.blackhole.qwen36.tt.sp_prefill import BLOCK_SIZE
from models.demos.blackhole.qwen36.tt.sp_tp_handoff import (
    MeshSwitch,
    SPTPHandoff,
    reserve_all,
    sp_mesh_proxy,
    tp_bind_contiguous_gdn_state,
    tp_conv_landing,
    tp_page_table,
)

EXPECTED = [760, 1414, 33241, 279, 3712, 314, 19820, 10903]
T = 4096
N_DEC = 7
W = 96  # logical page-table width (6144 positions)
N_SLOTS = 192  # TP cache physical blocks per device


def _export_die3(sp):
    """Host copy of die 3's post-prefill state in export_state_host's format, read with plain to_torch (no device
    ops, so nothing compiles on die 3 before its trace is captured)."""
    m = sp.models[-1]
    comp = ttnn.ConcatMeshToTensor(m.device, dim=0)
    kv, gdn = {}, {}
    S = sp.total_len
    for li, layer in enumerate(m.layers):
        a = layer.attention
        if layer.is_full_attention:
            out = []
            for c in (a.paged_kv_cache_key, a.paged_kv_cache_value):
                h = ttnn.to_torch(c, mesh_composer=comp).to(torch.bfloat16)  # [nb, nkv, 64, hd]
                nb, nkv, bs, hd = h.shape
                out.append(h[: S // bs].permute(1, 0, 2, 3).reshape(1, nkv, S, hd))
            kv[li] = tuple(out)
        else:
            gdn[li] = (
                ttnn.to_torch(a.recurrent_state, mesh_composer=comp).to(torch.float32),
                ttnn.to_torch(a.fused_conv_state, mesh_composer=comp).to(torch.bfloat16),
            )
    return kv, gdn, 0.0


def _check_state(sp, model, ho, mesh):
    """After a handoff (subs synchronized, parent side active): TP caches / rec / conv vs the SP export (host)."""
    kv_snap, gdn_snap, _ = ho._export  # taken on the sub side before the switch
    comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
    pt = tp_page_table(4, ho.p_blocks, ho.heads, 64, n_slots=N_SLOTS)  # logical blocks 0..63
    bad = []
    for j, li in enumerate(ho.fa_idx):
        K, V = kv_snap[li]
        for kv, full in ((0, K), (1, V)):
            c = 2 * j + kv
            got = ttnn.to_torch(ho.tp_caches[c], mesh_composer=comp).float().reshape(4, N_SLOTS, 1, BLOCK_SIZE, -1)
            for d in range(4):
                h = ho.heads[d]
                ref = full[0, h].float().reshape(64, BLOCK_SIZE, -1)
                g = got[d, pt[d].long(), 0]
                if not torch.equal(g, ref):
                    nbad = int((g != ref).any(-1).any(-1).sum())
                    bad.append(f"kv layer {li} {'KV'[kv]} die {d}: {nbad}/64 blocks differ")
    rec = ttnn.to_torch(ho.big_rec, mesh_composer=comp).float()  # [4 * n_gdn, 4, 128, 128]
    rec = rec.reshape(4, ho.n_gdn, ho.nv_tp, *rec.shape[-2:])
    for j, li in enumerate(ho.gdn_idx):
        r_full, c_full = gdn_snap[li]
        for d in range(4):
            if not torch.equal(rec[d, j], r_full[0, d * ho.nv_tp : (d + 1) * ho.nv_tp].float()):
                bad.append(f"rec layer {li} die {d}: max err {(rec[d, j] - r_full[0, d*4:(d+1)*4]).abs().max():.3e}")
        exp = gdn_conv_full_to_tp_host(c_full, num_devices=4, key_dim=ho.kd, value_dim=ho.vd)
        dn = model.layers[li].attention
        for m in range(1, len(dn.conv_states)):
            g = ttnn.to_torch(dn.conv_states[m], mesh_composer=comp).float().reshape(4, 1, -1)
            if not torch.equal(g, exp[m].float()):
                bad.append(f"conv layer {li} m {m}: max err {(g - exp[m].float()).abs().max():.3e}")
    for b in bad[:20]:
        logger.error(f"[e2e_tp4] state check: {b}")
    logger.info(f"[e2e_tp4] state check: {len(bad)} mismatches")
    return len(bad) == 0


def test_sp_e2e_tp4():
    assert os.environ.get("QWEN36_SP_IMPL", "").strip().lower() == "sc", "set QWEN36_SP_IMPL=sc"
    from models.demos.blackhole.qwen36.tt.sp_prefill_sc import SPPrefillSC

    reps = int(os.environ.get("E2E_REPS", "5"))
    overlap = os.environ.get("E2E_OVERLAP", "1") == "1"
    check_state = os.environ.get("E2E_CHECK_STATE", "1") == "1"
    tok = _e2e_tokenizer()
    tokens = _real_prompt_tokens(T, tok)

    _sp_set_fabric()
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        trace_region_size=int(os.environ.get("E2E_TRACE_MB", "384")) << 20,
        l1_small_size=GDN_CONV1D_L1_SMALL_SIZE,
    )
    subs = mesh.create_submeshes(ttnn.MeshShape(1, 1))
    sw = MeshSwitch(mesh, subs, side="parent")
    sp = ho = None
    prep_trace = None
    drv = None
    try:
        # ---------------- 1. TP4 decode model (parent) ----------------
        t_b = time.perf_counter()
        model = Qwen36Model.from_pretrained(mesh, max_batch_size=1, max_seq_len=W * BLOCK_SIZE, hf_model=model_path())
        vocab = model.args.vocab_size
        model.allocate_kv_caches(
            [N_SLOTS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim],
            ttnn.bfloat16 if os.environ.get("E2E_TP_KV_BF16", "1") == "1" else ttnn.bfloat8_b,
            batch_size=1,
        )
        tp_caches = [t for kv in model._paged_kv_caches for t in kv]
        big_rec, big_conv = tp_bind_contiguous_gdn_state(model)
        conv_land = tp_conv_landing(model)
        pt_host = tp_page_table(4, [16, 32, 48, 64], [0, 0, 1, 1], W, n_slots=N_SLOTS)
        pt_tt = ttnn.from_torch(
            pt_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        from models.demos.blackhole.qwen36.tt.tp_decode_driver import TPTracedDecoder

        drv = TPTracedDecoder(model, pt_host[:1])
        model._ondev_argmax = True
        drv.dev = model.prepare_inputs_decode(
            torch.tensor([[EXPECTED[0]]], dtype=torch.int32),
            torch.tensor([T], dtype=torch.int32),
            page_table=pt_host[:1],
        )
        drv.dev = (drv.dev[0], drv.dev[1], drv.dev[2], pt_tt)  # per-device (sharded) page table
        # eager warm-up: every parent program (decode step + greedy tail, conv_hist refresh) compiles here
        model.refresh_tp_gdn_conv_hist()
        for t_ in drv._step_ops():
            ttnn.deallocate(t_)
        model.refresh_tp_gdn_conv_hist()
        ttnn.synchronize_device(mesh)
        logger.info(f"[e2e_tp4] TP4 model + warm-up in {time.perf_counter() - t_b:.1f}s")

        # ---------------- 2. SP (subs) ----------------
        logger.info(f"[e2e_tp4] reserved subs<-parent: {reserve_all(mesh, subs, 'subs')} B")
        sw.to("subs")
        t_b = time.perf_counter()
        sp = SPPrefillSC(sp_mesh_proxy(mesh, subs), n_spans=4, span_len=T // 4, max_seq_len=T, hf_model=model_path())
        ho = SPTPHandoff(
            mesh, subs, sp, model, big_rec=big_rec, big_conv=big_conv, conv_land=conv_land, tp_caches=tp_caches
        )
        logger.info(f"[e2e_tp4] SP + handoff build in {time.perf_counter() - t_b:.1f}s")
        # eager prefill (real state) + eager handoff (compiles the handoff programs, before any sub trace is parked)
        logits = sp.prefill(tokens, return_logits=True).float()
        first_eager = int(torch.argmax(logits))
        logger.info(f"[e2e_tp4] eager SP prefill first token {first_eager}")
        ho.run_eager()
        if check_state:
            ho._export = _export_die3(sp)
        # ---------------- 3. sub traces ----------------
        sp.capture(tokens)
        ho.capture()
        # ---------------- 4. parent traces ----------------
        logger.info(f"[e2e_tp4] reserved parent<-subs: {reserve_all(mesh, subs, 'parent')} B")
        sw.to("parent")
        if check_state:
            # the handoff state of the eager pass is still in the TP buffers (capture replays nothing)
            assert _check_state(sp, model, ho, mesh), "handoff state mismatch"
        n_par = mesh.num_program_cache_entries()
        # decode trace first: its outputs are allocated during its own capture, before the prep trace is resident
        drv.trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        drv.out = drv._step_ops()
        ttnn.end_trace_capture(mesh, drv.trace_id, cq_id=0)
        prep_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        model.refresh_tp_gdn_conv_hist()
        ttnn.end_trace_capture(mesh, prep_trace, cq_id=0)
        ttnn.synchronize_device(mesh)
        assert mesh.num_program_cache_entries() == n_par, "parent trace capture compiled"
        logger.info(f"[e2e_tp4] reserved subs<-parent: {reserve_all(mesh, subs, 'subs')} B")

        # ---------------- 5. timed repetitions ----------------
        results = []
        sub_pc = [s.num_program_cache_entries() for s in subs]
        for rep in range(reps + 1):  # rep 0 = warm replay (reported, excluded from the means)
            t_back = sw.to("subs", eager=False)  # first sub op: the SP trace
            for d, (s, m) in enumerate(zip(subs, sp.models)):
                th = ttnn.from_torch(sp._span_tokens(tokens, d), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                ttnn.copy_host_to_device_tensor(th, m._chunk_token_buf)
            assert [s.num_program_cache_entries() for s in subs] == sub_pc, "sub-side compile after park"
            ho.check_no_compile()
            t0 = time.perf_counter()
            for d, s in enumerate(subs):
                ttnn.execute_trace(s, sp._trace_ids[d], cq_id=0, blocking=False)
            if overlap:
                ho.execute([0, 1, 2])
            first = int(
                ttnn.to_torch(sp._traced_tok, mesh_composer=ttnn.ConcatMeshToTensor(subs[-1], dim=0)).reshape(-1)[0]
            )
            t_ttft = time.perf_counter()
            ho.execute([3] if overlap else [0, 1, 2, 3])
            for s in subs:
                ttnn.synchronize_device(s)
            t_ho = time.perf_counter()
            sw.to("parent", eager=False)  # first parent op: the prep trace
            t_sw = time.perf_counter()
            ttnn.execute_trace(mesh, prep_trace, cq_id=0, blocking=False)
            toks, info = drv.decode(first, T, N_DEC)
            steps = info.get("step_s", [])
            t_end = time.perf_counter()
            r = {
                "ttft": t_ttft - t0,
                "handoff": t_ho - t_ttft,
                "switch": t_sw - t_ho,
                "decode": t_end - t_sw,
                "e2e": t_end - t0,
                "switch_back": t_back,
                "steps": steps,
                "tokens": toks,
            }
            results.append(r)
            logger.info(
                f"[e2e_tp4] rep {rep}: TTFT {r['ttft']*1e3:.2f} handoff {r['handoff']*1e3:.2f} switch "
                f"{r['switch']*1e3:.2f} decode(7) {r['decode']*1e3:.2f} E2E {r['e2e']*1e3:.2f} ms "
                f"(switch back {t_back*1e3:.2f}; steps {['%.2f' % (x*1e3) for x in steps]}) tokens {toks}"
            )
        timed = results[1:]
        mean = lambda k: sum(x[k] for x in timed) / len(timed) * 1e3
        mn = lambda k: min(x[k] for x in timed) * 1e3
        logger.info(
            "[e2e_tp4] MEAN over %d reps: TTFT %.2f (min %.2f) handoff %.2f (min %.2f) switch %.2f decode(7) %.2f "
            "(min %.2f) E2E %.2f (min %.2f) ms; switch back %.2f ms"
            % (
                len(timed),
                mean("ttft"),
                mn("ttft"),
                mean("handoff"),
                mn("handoff"),
                mean("switch"),
                mean("decode"),
                mn("decode"),
                mean("e2e"),
                mn("e2e"),
                mean("switch_back"),
            )
        )
        for x in results:
            logger.info(
                f"[e2e_tp4] tokens {x['tokens']} text {tok.decode(x['tokens'])!r} match={x['tokens'] == EXPECTED}"
            )
        assert all(x["tokens"] == EXPECTED for x in results), "tokens differ from the expected continuation"
    finally:
        try:
            sw.to("parent")
            if drv is not None:
                drv.release()
            if prep_trace is not None:
                ttnn.release_trace(mesh, prep_trace)
            sw.to("subs")
            if ho is not None:
                ho.release()
            if sp is not None:
                sp.close()
            sw.to("parent")
        except Exception as ex:
            logger.warning(f"[e2e_tp4] teardown: {ex!r}")
        for s in subs:
            s.quiesce_devices()
        mesh.quiesce_devices()
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
