# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device PREFILL scenarios on ONE built model (the 40-layer build takes 15-50 min, so several prompt lengths / chunkings / checks run per process).

Env:
  DSV41_U        users per mesh row (batch 4*U), default 4
  DSV41_LAYERS   default 0-39
  DSV41_SCEN     ';'-separated scenarios  S[:chunk[:steps[:flags]]]  (chunk 0 = whole prompt, steps = teacher-forced decode steps after the prefill),
                 flags: c = per-layer hidden PCC vs the dump (single chunk), s = decode-state PCC of every layer vs the dump,
                        w = also run the whole prompt in one chunk and compare the logits with the chunked run, t = capture/replay a prefill trace,
                        z = no reference dump for this S (random tokens)
  DSV41_REPS     prefill repetitions per scenario (default 2; run 0 includes the cold host Engram table reads)
Reference dumps: /mnt/tt-data/ssinghal/dsv4-prefill-s{S} (B=16) or ...-s{S}b{B} (reference/ref_prefill_dump.py)."""

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.attention import WINDOW
from models.demos.blackhole.deepseek_v41_flash.tt.decoder import DSV41Decoder
from models.demos.blackhole.deepseek_v41_flash.tt.device_head import DSV41DeviceEmbedding, DSV41DeviceHead
from models.demos.blackhole.deepseek_v41_flash.tt.engram import DSV41DeviceEngram, HostEngramRows
from models.demos.blackhole.deepseek_v41_flash.tt.loader import load_layer
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_attention import DSV41PrefillAttention, pad_len
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_layer import DSV41PrefillLayer, DSV41PrefillMoE
from models.demos.blackhole.deepseek_v41_flash.tt.prefill_model import DSV41PrefillModel, T, from_chunks
from models.demos.blackhole.deepseek_v41_flash.tt.step_state import DSV41StepState

U = int(os.environ.get("DSV41_U", "4"))
UPR = int(
    os.environ.get("DSV41_UPR", os.environ.get("DSV41_U", "4"))
)  # users per mesh row per prefill iteration (GPT-OSS: upr); U // UPR iterations
LAYERS = os.environ.get("DSV41_LAYERS", "0-39")
SPARSE = os.environ.get("DSV41_PF_SPARSE") == "1"
REPS = int(os.environ.get("DSV41_REPS", "2"))
BASE = "/mnt/tt-data/ssinghal/dsv4-prefill-s128"  # state seed / gate cutoff source


def parse(spec):
    f = spec.split(":")
    S = int(f[0])
    return {
        "S": S,
        "chunk": int(f[1]) if len(f) > 1 and f[1] else 0,
        "steps": int(f[2]) if len(f) > 2 and f[2] else 0,
        "flags": f[3] if len(f) > 3 else "",
    }


SCEN = [parse(x) for x in os.environ.get("DSV41_SCEN", "128").split(";")]
MAX_S = max(s["S"] + s["steps"] for s in SCEN)


MiB = 1 << 20


def mem(md, tag):
    ttnn.synchronize_device(md)
    mv = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    print(
        f"DRAMMEM {tag:40s} allocated {mv.total_bytes_allocated_per_bank / MiB:8.1f}  free {mv.total_bytes_free_per_bank / MiB:8.1f}  "
        f"largest_free_block {mv.largest_contiguous_bytes_free_per_bank / MiB:8.1f}  (MiB/bank, x8 banks per chip)",
        flush=True,
    )


def dump_dir(S):
    return os.environ.get(
        f"DSV41_DIR_{S}", f"/mnt/tt-data/ssinghal/dsv4-prefill-s{S}" + ("" if U == 4 else f"b{4 * U}")
    )


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_TRACE_REGION", "700000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(14400)
@torch.no_grad()
def test_prefill_scenarios(mesh_device):
    md = mesh_device
    rows, cols = tuple(md.shape)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    log = lambda m: print(m, flush=True)
    B = 4 * U
    need_steps = any(s["steps"] for s in SCEN)
    chain = DSV41DecodeChain(md, users_per_row=U, max_comp=256, log=log)
    sh = _Shards()

    pool = ThreadPoolExecutor(max_workers=2)
    futs = {}
    max_rope = max(256, MAX_S + 64)
    submit = lambda L: (
        futs.setdefault(L, pool.submit(load_layer, L, True, max_rope, SPARSE)) if L in layer_ids else None
    )
    for L in layer_ids[:2]:
        submit(L)
    pls, built, groups, first_moe = [], [], {}, None
    idx_w, sinks = {}, {}
    t0 = time.time()
    for L in layer_ids:
        ref = torch.load(
            os.path.join(BASE, f"layer_{L}.pt"), mmap=True
        )  # decode-state seed (zeroed below) + router cutoff
        # (the S=128 dump has 16 users: tile its state for B > 16)
        meta = {
            "state": {k: torch.cat([v] * -(-B // v.shape[0]))[:B] for k, v in ref["state"].items()},
            "S": 1,
            "gate_cutoff": ref["gate_cutoff"],
        }
        submit(L + 1), submit(L + 2)
        w = futs.pop(L).result()
        if SPARSE and "indexer" in w:
            idx_w[L] = w["indexer"]
        sinks[L] = w["attn"]["attn_sink"]
        layer, attn = chain.build_layer(L, meta, w)
        del w, ref
        ttnn.copy(ttnn.zeros_like(attn.cache), attn.cache)  # decode state: the prefill must write it
        if getattr(attn, "prev_cs", None) is not None:
            ttnn.copy(ttnn.zeros_like(attn.prev_cs), attn.prev_cs)
        attn.prefill = DSV41PrefillAttention(attn, sh.get(f"layers.{L}.attn.attn_sink").float())
        pmoe = DSV41PrefillMoE(layer.moe, T=T, buffers=None if first_moe is None else first_moe.decode.buffers)
        first_moe = first_moe or pmoe
        pls.append((L, DSV41PrefillLayer(layer, attn.prefill, pmoe, T=T)))
        from models.demos.blackhole.deepseek_v41_flash.tt import uni_policy

        if uni_policy.decide(md, U, log):
            from models.demos.blackhole.deepseek_v41_flash.tt.dsv41_model import UNI_LAYERS
            from models.demos.blackhole.deepseek_v41_flash.tt.prefill_unified_moe import DSV41UnifiedMoE

            if L in UNI_LAYERS(layer_ids):
                if uni_policy.ring_requested():  # one weight copy: read the decode ring weights in place
                    es = layer.moe.decode.expert_state
                    pls[-1][1].umoe = DSV41UnifiedMoE(md, L, log=log, ring=(es.tt_w0_w1, es.tt_w2))
                else:
                    pls[-1][1].umoe = DSV41UnifiedMoE(md, L, log=log)
                from models.demos.blackhole.deepseek_v41_flash.tt import moe_overlap

                if moe_overlap.prep_enabled():
                    moe_overlap.SDOverlap.get(md).split_weights(layer.shared)
        key = getattr(attn, "ratio", 0)
        if need_steps and key not in groups:
            groups[key] = DSV41StepState(attn, max_pos=257)
        built.append((L, layer, key))
        gc.collect()
    log(f"built {len(layer_ids)} layers in {time.time() - t0:.0f}s")

    engram_ids = [l for l in (1, 14) if l in layer_ids]
    host_rows = (
        HostEngramRows(tuple(engram_ids), max_batch_size=B, max_seq_len=max(256, MAX_S + 32)) if engram_ids else None
    )
    if (
        host_rows is not None and os.environ.get("DSV41_ENGRAM_RAM") == "1"
    ):  # ~200 GB RSS, ~4 min once: Engram row gathers then cost ~0.3 us/row
        t1 = time.time()
        host_rows.load_ram()
        log(f"Engram tables in RAM {time.time() - t1:.0f}s")
    dev_engram = {l: DSV41DeviceEngram(md, l, sh, mesh_config=chain.mesh_config, ccl=chain.ccl) for l in engram_ids}
    embedding = DSV41DeviceEmbedding(md, sh.get("embed.weight"), users_per_row=U)
    head = DSV41DeviceHead(md, sh.get("norm.weight").float(), sh.get("head.weight"), norm_eps=R.model_args().norm_eps)
    if SPARSE:  # indexer top-512 + sparse_sdpa once more than 512 compressed entries are visible (tt/prefill_sparse.py)
        from models.demos.blackhole.deepseek_v41_flash.tt.prefill_sparse import attach_prefill_sparse

        max_chunk = max([sc["chunk"] for sc in SCEN] + [128])
        attach_prefill_sparse(
            {L: pl.pa for L, pl in pls}, idx_w, UPR, -(-MAX_S // 128) * 128, max_chunk, sinks, force=False, enable=True
        )
    model = DSV41PrefillModel(md, pls, embedding, head, dev_engram, host_rows, users_per_row=UPR)
    for _, pl in pls:
        pl.pa.U = UPR
    rd = lambda t: torch.cat([ttnn.to_torch(ttnn.get_device_tensors(t)[r * cols]).float() for r in range(rows)])
    ids = {id(pl.pa.a): lid for lid, pl in pls}
    mem(md, "after build")

    def run_decode(sc, tag, toks, fin):
        S, STEPS = sc["S"], sc["steps"]
        dec_tok = toks["decode_tokens"]
        N = min(dec_tok.shape[1], STEPS)
        dec = DSV41Decoder(md, built, embedding, head, dev_engram, step_states=groups)
        dec.enable_sampling(chain.mesh_config, chain.ccl)
        pos_at = lambda i: torch.full((B,), S + i)
        hashes_dec = [host_rows.hashes(dec_tok[:, i : i + 1], S + i) for i in range(N)] if host_rows is not None else []
        rows_at = lambda i: host_rows.rows_all(hashes_dec[i], engram_ids) if engram_ids else {}
        set_in = lambda i, r: dec.set_packed_inputs(dec_tok[:, i], r, pos_at(i))
        snaps = dec.snapshot_states()
        set_in(0, rows_at(0))
        dec.forward()
        ttnn.synchronize_device(md)
        dec.restore_states(snaps)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        dl = dec.forward()
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        dec.restore_states(snaps)
        for i in range(N):
            t1 = time.perf_counter()
            set_in(i, rows_at(i))
            ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(md)
            am = head.combine(dec.sampled)[:B]
            wall = (time.perf_counter() - t1) * 1e3
            if fin is not None and "logits_steps" in fin:
                got = head.gather_logits(dl)[:B]
                log(
                    f"DECODE STEP {tag} {i} pos {S + i}: logits PCC {R.pcc(got, fin['logits_steps'][i]):.5f}  tokens match {int((am == fin['argmax_steps'][i]).sum())}/{B}  wall {wall:.1f} ms"
                )
            else:
                log(f"DECODE STEP {tag} {i}: tokens {am.tolist()} wall {wall:.1f} ms")
        ttnn.release_trace(md, tid)

    for sc in SCEN:
        S, CH, STEPS, flags = sc["S"], sc["chunk"], sc["steps"], sc["flags"]
        DIR = dump_dir(S)
        has_dump = "z" not in flags and os.path.exists(os.path.join(DIR, "tokens.pt"))
        tag = f"[S={S} chunk={CH or 'whole'} U={U}]"
        if has_dump:
            toks = torch.load(os.path.join(DIR, "tokens.pt"))
            fin = torch.load(os.path.join(DIR, "final.pt")) if os.path.exists(os.path.join(DIR, "final.pt")) else None
        else:
            g = torch.Generator().manual_seed(0)
            toks = {
                "prefill_tokens": torch.randint(1000, 100000, (B, S), generator=g),
                "decode_tokens": torch.zeros(B, 0, dtype=torch.long),
            }
            fin = None
        prompt = toks["prefill_tokens"]
        if (
            prompt.shape[0] == 1 and B > 1
        ):  # single-prompt dump: every user runs the same prompt, compared with the one reference
            prompt = prompt.expand(B, -1).contiguous()
            if fin is not None:
                fin = dict(fin)
                for k in ("prefill_logits", "prefill_argmax"):
                    if k in fin:
                        fin[k] = fin[k].expand(B, *fin[k].shape[1:]).contiguous()
        assert prompt.shape == (B, S), (prompt.shape, B, S)
        Sp = pad_len(S)

        def hook(lid, xs, pres):
            ref = torch.load(os.path.join(DIR, f"layer_{lid}.pt"), mmap=True)["prefill"]
            got = from_chunks([rd(x) for x in xs], rows, U, Sp, S, (4, 5120))
            log(
                f"CHAIN {tag} layer {lid:2d}: hidden PCC {R.pcc(got, ref['h_out'].float()):.5f}  last-token {R.pcc(got[:, -1], ref['h_out'][:, -1].float()):.5f}"
            )

        mode = ""
        if set(flags) & set(
            "mPREQ"
        ):  # prefill optimisation mode of this scenario: m = baseline, P = packed mHC, R = + own-chunk router, E = + own-chunk Engram
            fl = set(flags)
            os.environ["DSV41_PF_MHC"] = "packed" if fl & set("PREQ") else "0"
            os.environ["DSV41_PF_ROUTE_OWN"] = "1" if fl & set("RE") else "0"
            os.environ["DSV41_PF_ENGRAM_OWN"] = "1" if fl & set("EQ") else "0"
            mode = "_" + "".join(ch for ch in "mPREQ" if ch in fl)
            if getattr(model, "dyn", None) is not None:
                model.teardown_dyn()  # a new trace capture with the new mode
            log(
                f"MODE {tag}: DSV41_PF_MHC={os.environ['DSV41_PF_MHC']} ROUTE_OWN={os.environ['DSV41_PF_ROUTE_OWN']} ENGRAM_OWN={os.environ['DSV41_PF_ENGRAM_OWN']}"
            )
        model.fake_rows = S > int(os.environ.get("DSV41_ENGRAM_FAKE_ABOVE", "1000000"))
        if "D" in flags:  # traced chunks only (no eager run): performance + accuracy vs the dump where one exists
            assert CH, "D needs a chunk size"
            n_ch = -(-S // CH)
            iters = U // UPR
            for rep in range(REPS):
                t1 = time.perf_counter()
                outs, comp, per_it = [], 0.0, []
                tm = {}
                for it in range(iters):
                    sub = prompt.reshape(4, U, S)[:, it * UPR : (it + 1) * UPR].reshape(4 * UPR, S)
                    t2 = time.perf_counter()
                    outs.append(model.run_traced_chunks(sub, CH))
                    per_it.append(time.perf_counter() - t2)
                    comp += model.timing.get("compile_and_capture", 0.0)
                    for k, v in model.timing.items():
                        tm[k] = tm.get(k, 0.0) + v
                lg_d = torch.stack([o.reshape(4, UPR, -1) for o in outs], dim=1).reshape(
                    B, -1
                )  # [4, iters, UPR] -> row-major users
                ttft = time.perf_counter() - t1
                n_all = n_ch * iters
                log(
                    f"TRACED{mode} {tag} upr={UPR} x{iters} iters run {rep}: TTFT(last user) {ttft:.2f} s ({B * S / ttft:.0f} tok/s) [compile+capture {comp:.1f} s, "
                    f"without compile {ttft - comp:.2f} s = {B * S / (ttft - comp):.0f} tok/s; first-iteration users get their token after {per_it[0] - comp:.2f} s]; "
                    f"{n_ch} chunks of {CH} x {UPR} users/row per iteration: replay/chunk {tm['replay_per_chunk'] / n_all:.3f} s, host/chunk "
                    f"{tm['host_per_chunk'] / n_all:.3f} s, head+readback {tm['head_readback']:.2f} s"
                )
            if os.environ.get("DSV41_SAVE_LOGITS"):
                torch.save(lg_d, f"{os.environ['DSV41_SAVE_LOGITS']}_{S}_{CH}{mode}.pt")
            if fin is not None and "prefill_logits" in fin:
                log(
                    f"TRACED FIRST TOKEN{mode} {tag}: logits PCC {R.pcc(lg_d, fin['prefill_logits']):.5f}, argmax match {int((lg_d.argmax(-1) == fin['prefill_argmax']).sum())}/{B}"
                )
            else:
                log(f"TRACED first tokens {tag}: {lg_d.argmax(-1).tolist()} finite={bool(torch.isfinite(lg_d).all())}")
            mem(md, f"{tag} after traced run")
            continue
        logits = None
        for rep in range(REPS):
            t1 = time.perf_counter()
            try:
                logits = model.run_eager(
                    prompt, chunk=CH, hook=hook if ("c" in flags and has_dump and rep == 0) else None
                )
            except RuntimeError as e:
                log(f"PREFILL {tag} FAILED: {str(e)[:300]}")
                mem(md, f"{tag} after failure")
                logits = None
                gc.collect()
                break
            ttft = time.perf_counter() - t1
            log(
                f"PREFILL {tag} run {rep}: TTFT {ttft:.2f} s for {B} users x {S} tokens ({B * S / ttft:.0f} tok/s) plan {model.plan if len(model.plan) < 9 else len(model.plan)} "
                f"breakdown {{{', '.join(f'{k}: {v:.2f}' for k, v in model.timing.items())}}}"
            )
        if logits is None:
            continue  # this scenario failed (e.g. DRAM OOM): go on with the next one
        mem(md, f"{tag} after prefill")
        if os.environ.get("DSV41_SAVE_LOGITS"):
            torch.save(logits, f"{os.environ['DSV41_SAVE_LOGITS']}_eager_{S}_{CH}.pt")
        if fin is not None and "prefill_logits" in fin:
            p = R.pcc(logits, fin["prefill_logits"])
            log(
                f"FIRST TOKEN {tag}: logits PCC {p:.5f}, argmax match {int((logits.argmax(-1) == fin['prefill_argmax']).sum())}/{B}"
            )
        else:
            log(f"first tokens {tag}: {logits.argmax(-1).tolist()}  finite={bool(torch.isfinite(logits).all())}")
        if "w" in flags and CH:
            lw = model.run_eager(prompt, chunk=0)
            log(
                f"CONSISTENCY {tag}: chunked vs whole-prompt logits PCC {R.pcc(logits, lw):.5f}, argmax match {int((logits.argmax(-1) == lw.argmax(-1)).sum())}/{B}"
            )
        if "d" in flags and CH:  # ONE traced chunk forward replayed for every chunk
            for rep in range(2):
                t1 = time.perf_counter()
                lg_d = model.run_traced_chunks(prompt, CH)
                ttft = time.perf_counter() - t1
                n_ch = len(model.plan)
                tm = model.timing
                log(
                    f"TRACED-CHUNKS {tag} run {rep}: TTFT {ttft:.2f} s ({B * S / ttft:.0f} tok/s), {n_ch} chunks of {CH}: compile+capture "
                    f"{tm.get('compile_and_capture', 0):.1f} s, replay/chunk {tm['replay_per_chunk'] / n_ch:.3f} s, host/chunk "
                    f"{tm['host_per_chunk'] / n_ch:.3f} s, head+readback {tm['head_readback']:.2f} s"
                )
            log(
                f"TRACED-CHUNKS {tag}: logits PCC vs eager {R.pcc(lg_d, logits):.5f}, argmax match {int((lg_d.argmax(-1) == logits.argmax(-1)).sum())}/{B}"
                + (
                    f"; vs dump PCC {R.pcc(lg_d, fin['prefill_logits']):.5f}"
                    if (fin is not None and "prefill_logits" in fin)
                    else ""
                )
            )
            mem(md, f"{tag} after traced chunks")
        if "s" in flags and has_dump:
            for (lid, pl), (_, layer, key) in zip(pls, built):
                at = pl.pa.a
                st = torch.load(os.path.join(DIR, f"layer_{lid}.pt"), mmap=True)["state"]
                cache = torch.cat(
                    [
                        ttnn.to_torch(ttnn.get_device_tensors(at.cache)[r * cols]).float().reshape(U, -1, 512)
                        for r in range(rows)
                    ]
                )
                nw = min(S, WINDOW)
                msg = f"STATE {tag} layer {lid:2d}: ring PCC {R.pcc(cache[:, :nw], st['window'][:, :nw].float()):.5f}"
                ratio = getattr(at, "ratio", 0)
                if ratio:
                    sstate = (
                        st
                        if at.source is None
                        else torch.load(os.path.join(DIR, f"layer_{ids[id(at.source)]}.pt"), mmap=True)["state"]
                    )
                    if sstate is not None and "comp" in sstate:
                        nc = min(S // ratio, at.max_comp)
                        msg += f" comp PCC {R.pcc(cache[:, WINDOW:WINDOW + nc], sstate['comp'][:, :nc].float()):.5f}"
                    if ratio > 1 and at.source is None and S % ratio and "kv_state" in st:
                        refcs = torch.cat(
                            [st["kv_state"][:, (S - 1) % ratio], st["score_state"][:, (S - 1) % ratio]], dim=-1
                        ).float()
                        gotcs = torch.cat(
                            [
                                ttnn.to_torch(ttnn.get_device_tensors(at.prev_cs)[r * cols]).float().reshape(U, -1)
                                for r in range(rows)
                            ]
                        )
                        msg += f" prev_cs PCC {R.pcc(gotcs, refcs):.6f}"
                log(msg)
        if "t" in flags and not CH:
            model.capture_trace(S)
            log(f"prefill trace captured {tag}")
            for rep in range(3):
                t1 = time.perf_counter()
                lg_t = model.run_traced(prompt)
                ttft = time.perf_counter() - t1
                log(
                    f"PREFILL TRACED {tag} run {rep}: TTFT {ttft:.3f} s for {B} users x {S} tokens ({B * S / ttft:.0f} tok/s) breakdown "
                    f"{{{', '.join(f'{k}: {v:.3f}' for k, v in model.timing.items())}}}  logits PCC vs eager {R.pcc(lg_t, logits):.5f}, "
                    f"argmax match {int((lg_t.argmax(-1) == logits.argmax(-1)).sum())}/{B}"
                )
            ttnn.release_trace(md, model.trace_id)
        if STEPS:
            if U == 1:  # the T=4k packed mHC mixes kernels of the decode path need 4 users per row
                os.environ["DSV41_MHC_MIXES_V2"] = "0"
            try:
                run_decode(sc, tag, toks, fin)
            except Exception as e:
                log(f"DECODE {tag} FAILED: {type(e).__name__}: {str(e)[:300]}")

    log("SCENARIOS DONE")
