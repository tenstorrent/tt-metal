# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Trace-timed sibling of kob.py for the Qwen3.5 gated-delta-net (GDN) prefill→decode flow.

Same setup as kob.py — one HF Qwen3_5GatedDeltaNet with random weights, those SAME
weights loaded into our TT module, PCC-checked against the HF golden — but the DECODE
timing is gathered differently. kob.py times a bare synchronized call, which on every
step re-issues all of decode's many small ops from the host (dispatch-bound). Here we
instead CAPTURE decode as a ttnn trace and time it by REPLAYING the trace, so each step
costs a single execute_trace dispatch instead of re-issuing every op — the same trick the
9B text demo uses to lift decode throughput.

Prefill is now timed BOTH ways. It used to be eager-only: forward_prefill minted constant
tensors INSIDE the forward — the ttnn.zeros conv pad in causal_conv1d_silu and the
ttnn.pad(value=0) calls in the chunk math — and trace capture forbids those host-side writes
("Writes are not supported during trace capture"). That blocker is gone: those constants are
now pre-built once in __init__ (a persistent conv pad, plus persistent zero tails the chunk
math slices + concats), the same pattern the production GDN uses, so the whole prefill graph
is pure device work and captures cleanly. We keep the warm-eager measurement AND add a traced
replay so the summary shows the trace win directly. The old comment assumed trace wouldn't
help prefill (one big compute-bound op sequence), but with chunk_size=128 the chunk loop
unrolls into 64 chunks × ~10 ops each, so host dispatch is NOT negligible — hence worth
measuring rather than assuming.

The run has two phases, and the order matters:
  1. EAGER (correctness + compile): run forward_prefill once and PCC its output plus the
     two hand-off states (conv window + recurrent KV) decode reads back, then a few eager
     decode_forward steps PCC'd against HF. This validates the math AND compiles every
     kernel — trace capture cannot compile, so decode's program must already be warm.
  2. TIMING: time prefill BOTH eager (warm synchronized calls) and traced (capture once,
     replay) to compare TTFT, then capture the decode trace and replay it many times (TPOT).
     All PCC checks finish in phase 1, before any capture, so we needn't snapshot/restore state
     around the throwaway capture runs the way the demo must — a replay only feeds the timer,
     never an HF PCC. We release the prefill trace before capturing decode so only one trace
     occupies the region at a time; the last prefill replay leaves valid post-prefill state for
     the decode trace to continue from, exactly as a real decode loop would.

NOTE: the HF Qwen3_5GatedDeltaNet.forward in site-packages has a breakpoint() right before
its return, so each hf_gdn() call drops into pdb — type `c`, or run with PYTHONBREAKPOINT=0.

Run from the repo root:  PYTHONBREAKPOINT=0 python kob2.py
"""
import os
import time

import torch
from loguru import logger

import ttnn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tt.gdn import Qwen35GatedDeltaNet
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

# HF_MODEL is the single source of truth for dims (Qwen35ModelArgs parses it). Default to the
# bare hub id only as a fallback; override with `export HF_MODEL=/path/to/Qwen3.5-27B-FP8` to
# point at the local snapshot on this host and skip a multi-GB snapshot_download.
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

torch.manual_seed(0)

SEQ_LEN = 8192 * 4
PCC_TARGET = 0.99  # bf16 GDN may land lower (the old TP tests used ~0.95); loosen as needed.
# Mesh shape: (1, 4) runs the GDN tensor-parallel across all 4 P150s; (1, 1) is the original
# single-device path. The head counts in HF_MODEL must divide by the device count (9B: 16 K /
# 32 V, 27B: 16 K / 48 V both divide by 4) and hidden/devices must stay tile-aligned.
MESH_SHAPE = (1, 4)

# How many decode-trace replays to average TPOT over (mirrors kob.py's 100 steps; a replay is
# one cheap device dispatch, so a healthy sample is affordable). Prefill is timed with a few warm
# eager calls — it's already compiled by phase 1, so each is steady-state, no compile folded in.
N_DECODE_REPLAYS = 100
N_PREFILL_ITERS = 5
# Eager decode steps run for correctness only — enough to exercise the multi-step recurrence
# (step k's PCC only holds if step k-1 wrote correct state) and to compile the decode kernels
# before capture. The 100-step loop kob.py ran was purely for timing, which trace now supplies.
DECODE_PCC_STEPS = 4

# A trace needs a reserved DRAM region for its baked command stream; ttnn's default is 0, which
# would make begin_trace_capture fail. We now trace BOTH the prefill and the decode forward (one at
# a time — prefill is released before decode is captured), so the region must hold the LARGER of the
# two: prefill's 64-chunk unrolled command stream dwarfs decode's single step. The demo reserves
# 256 MiB for its chunk-prefill trace; our hand-rolled loop emits more ops, so we give it 512 MiB of
# headroom. If capture errors, ttnn prints the exact size required — bump this to match.
TRACE_REGION_SIZE = 512 * 1024 * 1024

# CCL ops (the out_proj reduce-scatter, TP>1 only) ride the ethernet fabric, which must be
# enabled BEFORE the mesh opens — the FABRIC_1D the 9B demo sets for its multi-device mesh.
# Skip it on a single device, where no collective runs.
if MESH_SHAPE != (1, 1):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
# trace_region_size is the one addition over kob.py's open_mesh_device: without it the device
# has no room to record a trace, and begin_trace_capture below would raise immediately.
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), trace_region_size=TRACE_REGION_SIZE)

# The trace ids are created mid-run; default None so the finally can release whichever exists if an
# exception fires between capture and the in-flow release (close_mesh_device also frees them).
prefill_tid = None
decode_tid = None
try:
    args = Qwen35ModelArgs(mesh_device, max_seq_len=SEQ_LEN)

    # Build the HF reference from the SAME parsed text config so its qkv/z/a/b/conv/norm/out
    # dims match what args (and our loaded weights) expect. layer_idx=0 matters now that we pass
    # a cache: cfg.layer_types[0] == "linear_attention", so cache.layers[0] is the conv/recurrent
    # state layer the GDN forward indexes by layer_idx. float32 + eval for a clean golden; on this
    # CPU host the torch fallbacks run (no flash-linear-attention).
    cfg = args.hf_config.get_text_config()
    # Re-seed immediately before weight init: the global torch RNG is one mutable stream, and
    # Qwen35ModelArgs (above) may draw from it, which would shift the weights run-to-run. Pinning
    # here makes the HF layer's random weights identical every run, independent of upstream code.
    torch.manual_seed(0)
    hf_gdn = Qwen3_5GatedDeltaNet(config=cfg, layer_idx=0).to(torch.float32).eval()
    state_dict = hf_gdn.state_dict()

    # load_gdn_weights reads exactly these state_dict keys (in_proj_{qkv,z,a,b}, out_proj,
    # conv1d, A_log, dt_bias, norm), so the HF layer's weights drop straight into the TT module.
    # TP>1: the row-parallel out_proj needs a reduce-scatter driven by a TT_CCL. tensor_cache_path
    # stays None — these are RANDOM weights, and caching them would corrupt a later re-seeded run.
    tp = mesh_device.get_num_devices()
    tt_ccl = TT_CCL(mesh_device) if tp > 1 else None
    gdn = Qwen35GatedDeltaNet(args=args, state_dict=state_dict, mesh_device=mesh_device, tt_ccl=tt_ccl)

    torch.manual_seed(0)
    x = torch.randn(1, SEQ_LEN, args.dim, dtype=torch.float32)

    # HF golden: pass a real cache (the same DynamicCache(config=...) Qwen3_5Model builds) so the
    # prefill path also writes its end-of-prompt states into cache.layers[0]. With seq_len > 1 the
    # cache is write-only here — use_precomputed_states stays False, so core_attn_out (hence ref) is
    # bit-for-bit what cache_params=None produced; the cache just additionally captures conv_state
    # (left-pad of the projected qkv) and the final recurrent_state. forward() returns [1, S, dim]
    # (post out_proj), still our output PCC target; the two state tensors are the goldens for the TT
    # module's conv_state / recurrent_state once forward_prefill grows past the input projections.
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        ref = hf_gdn(x, cache_params=cache, attention_mask=None)
    ref = ref[0]  # [S, dim]
    conv_state_ref = cache.layers[0].conv_states  # rolling conv window: last conv_kernel_size cols of qkv
    recurrent_state_ref = cache.layers[0].recurrent_states  # final gated-delta (KV) recurrent state
    logger.info(f"HF cache states: conv={tuple(conv_state_ref.shape)} recurrent={tuple(recurrent_state_ref.shape)}")

    # TT prefill expects hidden_states as [B=1, 1, S, dim], replicated across the mesh. This is a
    # PERSISTENT DRAM buffer: the prefill trace bakes its address in, so it must stay allocated for
    # every replay (we never deallocate it). The replays reuse this same prompt — fine, since trace
    # timing measures on-device compute, which is input-value-independent.
    x_tt = ttnn.from_torch(
        x.to(torch.bfloat16).reshape(1, 1, SEQ_LEN, args.dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # ──────────────── Phase 1: eager correctness (also compiles every kernel) ────────────────
    # One eager prefill: PCC the output AND the two states decode reads back. This is the same
    # validation kob.py does; it doubles as the compile run so the prefill trace below replays a
    # warm program (trace capture itself cannot compile). No synchronize timer here — Phase 2 owns
    # timing — but we still PCC the result.
    out = gdn.forward_prefill(x_tt)

    # forward_prefill returns rank-4 [1, 1, S, dim] in both cases (the out_proj reduce no longer
    # branches on device count). On TP>1 it's fractured along hidden (dim 3), so gather it back; on a
    # single device the gather is a no-op. Either way flatten to [S, dim] for the PCC.
    if tp > 1:
        out_torch = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        out_torch = out_torch.reshape(SEQ_LEN, args.dim).float()  # [S, dim]
    else:
        out_torch = ttnn.to_torch(out).reshape(SEQ_LEN, args.dim).float()  # [S, dim]
    passing, pcc = comp_pcc(ref, out_torch, PCC_TARGET)
    logger.info(f"GDN forward_prefill PCC (S={SEQ_LEN}) = {pcc}  passing={passing}")

    # Validate the prefill→decode hand-off that the bare output PCC can't see: the
    # conv window and the recurrent (KV) state that decode_forward reads back must
    # match HF's cached states, or decode diverges from the very first token.
    # conv_state_ref is HF's channels-first [1, conv_dim, K]; ours is channels-last
    # [B, 1, K, conv_dim], so transpose the ref to line them up.
    if tp > 1:
        # conv_state is sharded over channels; each device holds [q|k|v] for ITS heads, so the
        # dim-3 gather interleaves devices — regroup back to HF's [q_all|k_all|v_all] order.
        conv_g = ttnn.to_torch(gdn.conv_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))[:, 0].float()
        kd_tp, vd_tp = args.gdn_key_dim // tp, args.gdn_value_dim // tp
        per = 2 * kd_tp + vd_tp
        qs = [conv_g[..., d * per : d * per + kd_tp] for d in range(tp)]
        ks = [conv_g[..., d * per + kd_tp : d * per + 2 * kd_tp] for d in range(tp)]
        vs = [conv_g[..., d * per + 2 * kd_tp : (d + 1) * per] for d in range(tp)]
        conv_state_tt = torch.cat(qs + ks + vs, dim=-1)  # [B, K, conv_dim]
    else:
        conv_state_tt = ttnn.to_torch(gdn.conv_state)[:, 0].float()  # [B, K, conv_dim]
    conv_ref = conv_state_ref.transpose(1, 2).float()  # [1, K, conv_dim]
    passing_c, pcc_c = comp_pcc(conv_ref, conv_state_tt, PCC_TARGET)
    logger.info(f"prefill conv_state PCC = {pcc_c}  passing={passing_c}")

    # Recurrent state is sharded over value heads (dim 1); the reorder keeps each device's V
    # heads contiguous & in original order, so a dim-1 gather rebuilds the HF head order.
    if tp > 1:
        rec_state_tt = ttnn.to_torch(
            gdn.last_recurrent_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)
        ).float()  # [B, Hv, Dk, Dv]
    else:
        rec_state_tt = ttnn.to_torch(gdn.last_recurrent_state).float()  # [B, Hv, Dk, Dv]
    passing_r, pcc_r = comp_pcc(recurrent_state_ref.float(), rec_state_tt, PCC_TARGET)
    logger.info(f"prefill recurrent_state PCC = {pcc_r}  passing={passing_r}")

    # ── A few eager decode steps continuing from the prefill-filled states ──
    # HF reads & updates the SAME DynamicCache on each call: with seq_len==1 and a populated cache,
    # use_precomputed_states flips True so the recurrent path (not the chunk path) runs and rolls
    # conv_states / recurrent_states forward. TT's decode_forward likewise reads & rewrites the
    # self.conv_state / self.last_recurrent_state that forward_prefill just wrote. So this exercises
    # recurrent_gated_delta_rule + the conv-window roll across steps, and step k only passes if step
    # k-1 wrote correct state — a genuine multi-step recurrence check, and it compiles the decode
    # kernels for the trace.
    B = args.max_batch_size
    # The prefill golden above ran batch-1 (x is [1, SEQ_LEN, dim]), so HF lazily sized its cached
    # conv/recurrent states with batch 1. Decode must match; assert loudly rather than let a future
    # max_batch_size > 1 surface as a confusing HF broadcast error deep in the recurrent path.
    assert B == 1, f"kob2.py decode assumes max_batch_size==1 (prefill golden is batch-1); got {B}"
    for step in range(DECODE_PCC_STEPS):
        pos = SEQ_LEN + step
        # Re-seed per step so the decode tokens differ but stay reproducible.
        torch.manual_seed(1000 + step)
        x_dec = torch.randn(B, 1, args.dim, dtype=torch.float32)

        with torch.no_grad():
            ref_dec = hf_gdn(x_dec, cache_params=cache, attention_mask=None)
        ref_dec = ref_dec[:, 0]  # [B, dim]

        # TT decode layout: token replicated as [1, 1, B, dim] (batch on dim 2, the convention
        # decode_forward decodes via hidden_states.shape[2]).
        x_dec_tt = ttnn.from_torch(
            x_dec.to(torch.bfloat16).reshape(1, 1, B, args.dim),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        out_dec = gdn.decode_forward(x_dec_tt)

        if tp > 1:
            out_dec_torch = (
                ttnn.to_torch(out_dec, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
                .reshape(B, args.dim)
                .float()
            )  # [B, dim]
        else:
            out_dec_torch = ttnn.to_torch(out_dec).reshape(B, args.dim).float()  # [B, dim]
        passing_d, pcc_d = comp_pcc(ref_dec, out_dec_torch, PCC_TARGET)
        logger.info(f"GDN decode step {step} PCC (pos={pos}, B={B}) = {pcc_d}  passing={passing_d}")

    # ──────────────── Phase 2: timing (prefill eager + traced, decode traced) ────────────────

    def _bench_trace(tid, n):
        """Return (per_step_mean_s, amortized_s) for n replays of trace `tid`.

        Shared by the prefill and decode timings — both reduce to "replay one captured trace".
        * per_step  — one execute_trace + one synchronize each. This is the realistic per-call
          latency: you must drain the device to read a result before issuing the next call (decode
          samples a token between steps; prefill hands off to the sampler). The first replay can
          carry one-time costs, so we run n+1 and drop step 0 as warm-up.
        * amortized — n back-to-back replays, then ONE drain. Removes per-call host/sync overhead so
          the device pipelines replays end to end: the pure on-device ceiling.
        """
        per_step = []
        for _ in range(n + 1):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            per_step.append(time.perf_counter() - t0)
        per_step = per_step[1:]  # drop the warm-up replay

        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        amortized = (time.perf_counter() - t0) / n
        return sum(per_step) / len(per_step), amortizeds

    # ── Prefill (TTFT): warm eager calls. ──
    # Kernels are warm from phase 1, so each call is steady-state — no compile folded in (kob.py's
    # single cold call included it). ttnn dispatches ops async, so synchronize on BOTH sides to clock
    # real on-device compute, not just host dispatch. forward_prefill resets its recurrent state and
    # overwrites conv_state internally, so re-running on the same x_tt is idempotent (and conveniently
    # leaves the post-prefill state in place for the decode trace below to continue from).
    prefill_times = []
    for _ in range(N_PREFILL_ITERS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        gdn.forward_prefill(x_tt)
        ttnn.synchronize_device(mesh_device)
        prefill_times.append(time.perf_counter() - t0)
    prefill_s = sum(prefill_times) / len(prefill_times)
    logger.info(
        f"prefill (TTFT, warm eager) = {prefill_s * 1e3:.1f} ms for {SEQ_LEN} tokens "
        f"→ {SEQ_LEN / prefill_s:,.0f} tok/s  (mean of {N_PREFILL_ITERS} warm calls)"
    )

    # ── Prefill (TTFT): traced. Capture once, replay. ──
    # Newly possible: forward_prefill no longer mints constants inline (the conv pad and the chunk
    # zero-tails are pre-built in __init__ and slice+concat'd), so its graph is pure device work and
    # captures like decode does. The chunk loop unrolls into many small ops, so collapsing them into
    # a single execute_trace dispatch is exactly where trace can shave TTFT over the eager call above.
    # We capture on the SAME persistent x_tt (its address is baked into the trace), then release this
    # trace before capturing decode so only one occupies the region at a time. The replays re-run
    # forward_prefill, leaving valid post-prefill state for the decode trace below to continue from.
    logger.info("capturing prefill trace…")
    prefill_tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    gdn.forward_prefill(x_tt)
    ttnn.end_trace_capture(mesh_device, prefill_tid, cq_id=0)
    prefill_trace_step_s, prefill_trace_amort_s = _bench_trace(prefill_tid, N_PREFILL_ITERS)
    ttnn.release_trace(mesh_device, prefill_tid)
    prefill_tid = None  # released; don't double-free in finally
    logger.info(
        f"prefill (TTFT, traced) = per-step {prefill_trace_step_s * 1e3:.1f} ms | amortized "
        f"{prefill_trace_amort_s * 1e3:.1f} ms  ({SEQ_LEN / prefill_trace_amort_s:,.0f} tok/s)"
    )

    # ── Decode (TPOT): capture the trace once, replay it N times. ──
    # A trace records decode's exact command stream against fixed buffer addresses; execute_trace
    # re-runs it with ONE host dispatch instead of re-issuing every op — the win for a per-token path
    # that's dispatch-bound. decode_forward allocates no constants either, so it captures just like
    # prefill above. The capture run's output is unreliable in this tt-metal version and it advances
    # the conv/recurrent state; both are harmless, since every PCC check ran in phase 1 and replays
    # only feed the timer. Replaying keeps rolling the state forward exactly as a real decode loop
    # would — which is the per-token cost we want to clock.

    # Persistent single-token input the decode trace bakes its address into; the token value is
    # irrelevant to timing, so any random vector does.
    torch.manual_seed(9999)
    x_dec_trace = ttnn.from_torch(
        torch.randn(B, 1, args.dim, dtype=torch.float32).to(torch.bfloat16).reshape(1, 1, B, args.dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    logger.info("capturing decode trace…")
    decode_tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    gdn.decode_forward(x_dec_trace)
    ttnn.end_trace_capture(mesh_device, decode_tid, cq_id=0)
    decode_step_s, decode_amort_s = _bench_trace(decode_tid, N_DECODE_REPLAYS)
    ttnn.release_trace(mesh_device, decode_tid)
    decode_tid = None  # released; don't double-free in finally
    logger.info(
        f"decode trace replay (TPOT): per-step {decode_step_s * 1e3:.2f} ms | amortized "
        f"{decode_amort_s * 1e3:.2f} ms  ({B / decode_amort_s:,.1f} tok/s)"
    )

    # ──────────────── Timing summary: TTFT / TPOT / throughput ────────────────
    # Both numbers exclude one-time kernel compilation (warm from phase 1). They differ in HOW host
    # dispatch is handled: prefill is one warm synchronized eager call (dispatch is a tiny fraction of
    # its compute), while decode's many tiny ops are collapsed into a single execute_trace replay —
    # the dispatch saving trace exists for. For decode, per-step is the realistic per-token latency;
    # amortized is the device-bound floor with back-to-back replays.
    logger.info("──────── timing summary ────────")
    logger.info(
        f"prefill (TTFT, warm eager) : {prefill_s * 1e3:8.1f} ms"
        f"  |  {SEQ_LEN / prefill_s:>10,.0f} tok/s  (prompt={SEQ_LEN} tokens)"
    )
    logger.info(
        f"prefill (TTFT, traced)     : per-step {prefill_trace_step_s * 1e3:7.1f} ms | amortized "
        f"{prefill_trace_amort_s * 1e3:7.1f} ms  |  {SEQ_LEN / prefill_trace_amort_s:>10,.0f} tok/s"
    )
    logger.info(
        f"decode  (TPOT, traced)     : per-step {decode_step_s * 1e3:7.2f} ms | amortized "
        f"{decode_amort_s * 1e3:7.2f} ms  |  {B / decode_amort_s:>8,.1f} tok/s  (B={B})"
    )
finally:
    if prefill_tid is not None:
        ttnn.release_trace(mesh_device, prefill_tid)
    if decode_tid is not None:
        ttnn.release_trace(mesh_device, decode_tid)
    ttnn.close_mesh_device(mesh_device)
    if MESH_SHAPE != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
