# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Prefill-trace harness for the Qwen3.5 gated-delta-net (GDN), sibling of kob2.py.

kob2.py captures DECODE as a ttnn trace but times PREFILL the eager way, because
my_gdn's forward_prefill used to allocate constants INSIDE the forward (the
ttnn.zeros conv left-pad in causal_conv1d_silu and the ttnn.pad(value=0) chunk
seq-pad), and trace capture forbids those host-side writes. Those two have now been
reworked to concat against persistent zero buffers pre-built in __init__
(initialize_params_gated_delta_rule), so prefill should be trace-capturable. This
harness proves it: it CAPTURES forward_prefill as a trace, REPLAYS it, and PCCs the
replayed output against the HF golden — not just timing, but a correctness check
that the traced program produces the right answer.

Same correctness scaffolding as kob2.py: one HF Qwen3_5GatedDeltaNet with random
weights, those SAME weights loaded into our TT module, PCC-checked against the HF
golden (output + the conv-window and recurrent-KV hand-off states).

Two phases, order matters:
  1. EAGER (correctness + compile): run forward_prefill once eagerly, PCC its output
     plus the two hand-off states. This validates the concat-based padding math AND
     compiles every kernel (trace capture cannot compile) AND forces the first-call
     ttnn.zeros in reset_recurrent_state so capture only sees the device->device copy.
  2. TRACE: capture forward_prefill, replay it once, read the replayed output back and
     PCC vs HF (proves the trace is correct), then time eager vs trace replay for TTFT.

SEQ_LEN is env-configurable (GDN_SEQ_LEN). 8192 is chunk-aligned (chunk_size=128) so
pad_size==0 and the chunk-pad concat branch is skipped (only the conv pad is exercised
under trace). Set e.g. GDN_SEQ_LEN=8160 (a multiple of 32 but not 128 -> pad_size=32)
to exercise the new chunk-pad slice+concat branch under trace as well.

NOTE: the HF Qwen3_5GatedDeltaNet.forward in site-packages has a breakpoint() right
before its return, so each hf_gdn() call drops into pdb — run with PYTHONBREAKPOINT=0.

Run from the repo root:  PYTHONBREAKPOINT=0 python pob.py
                         PYTHONBREAKPOINT=0 GDN_SEQ_LEN=8160 python pob.py
"""
import os
import time

import torch
from loguru import logger

import ttnn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tt.my_gdn import Qwen35GatedDeltaNet
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

torch.manual_seed(0)

# SEQ_LEN drives both the trace shape and the pad_size. 8192 % 128 == 0 -> pad_size 0
# (chunk-pad concat skipped); a multiple of 32 that is NOT a multiple of 128 -> pad_size
# in {32,64,96} (chunk-pad concat exercised). Must stay tile-aligned (multiple of 32).
SEQ_LEN = int(os.environ.get("GDN_SEQ_LEN", "8192"))
assert SEQ_LEN % 32 == 0, f"SEQ_LEN must be tile-aligned (multiple of 32); got {SEQ_LEN}"

PCC_TARGET = 0.99  # bf16 GDN may land lower; loosen if needed.
MESH_SHAPE = (1, 4)  # (1,4) runs TP across 4 P150s; (1,1) is single-device.
N_PREFILL_ITERS = 5  # warm eager prefill calls to average TTFT over.
N_TRACE_REPLAYS = 5  # trace replays to average over (prefill replay is heavy, few suffice).

# A prefill trace records a MUCH longer command stream than decode (64 chunks unrolled),
# so reserve a larger trace DRAM region than kob2's 64 MiB decode region. The 9B text
# demo's chunk-prefill trace uses ~256 MiB; if capture errors on size, ttnn prints the
# required amount and we bump this.
TRACE_REGION_SIZE = 256 * 1024 * 1024

if MESH_SHAPE != (1, 1):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE), trace_region_size=TRACE_REGION_SIZE)

# prefill trace id is created mid-run; default None so the finally can release it on an
# exception between capture and the in-flow release.
prefill_tid = None
try:
    args = Qwen35ModelArgs(mesh_device, max_seq_len=SEQ_LEN)
    cfg = args.hf_config.get_text_config()

    # Re-seed immediately before weight init so the HF layer's random weights are
    # identical every run, independent of any RNG draws Qwen35ModelArgs made above.
    torch.manual_seed(0)
    hf_gdn = Qwen3_5GatedDeltaNet(config=cfg, layer_idx=0).to(torch.float32).eval()
    state_dict = hf_gdn.state_dict()

    tp = mesh_device.get_num_devices()
    tt_ccl = TT_CCL(mesh_device) if tp > 1 else None
    gdn = Qwen35GatedDeltaNet(args=args, state_dict=state_dict, mesh_device=mesh_device, tt_ccl=tt_ccl)

    torch.manual_seed(0)
    x = torch.randn(1, SEQ_LEN, args.dim, dtype=torch.float32)

    # HF golden + the cache it writes its end-of-prompt states into.
    cache = DynamicCache(config=cfg)
    with torch.no_grad():
        ref = hf_gdn(x, cache_params=cache, attention_mask=None)
    ref = ref[0]  # [S, dim]
    conv_state_ref = cache.layers[0].conv_states
    recurrent_state_ref = cache.layers[0].recurrent_states
    logger.info(f"HF cache states: conv={tuple(conv_state_ref.shape)} recurrent={tuple(recurrent_state_ref.shape)}")

    # Persistent prefill input [B=1, 1, S, dim], replicated across the mesh. The prefill
    # trace bakes this buffer's address in, so it must stay allocated for every replay.
    x_tt = ttnn.from_torch(
        x.to(torch.bfloat16).reshape(1, 1, SEQ_LEN, args.dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def _gather_out(out_tt):
        """forward_prefill returns [1,1,S,dim]; on TP>1 it's fractured along hidden (dim 3)."""
        if tp > 1:
            t = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
        else:
            t = ttnn.to_torch(out_tt)
        return t.reshape(SEQ_LEN, args.dim).float()

    # ──────────────── Phase 1: eager correctness (also compiles every kernel) ────────────────
    out = gdn.forward_prefill(x_tt)
    out_torch = _gather_out(out)
    passing, pcc = comp_pcc(ref, out_torch, PCC_TARGET)
    logger.info(
        f"[eager] forward_prefill output PCC (S={SEQ_LEN}, pad_size={(128 - SEQ_LEN % 128) % 128}) = {pcc}  passing={passing}"
    )

    # conv_state hand-off (channels-last [B,1,K,conv_dim] here vs HF channels-first [1,conv_dim,K]).
    if tp > 1:
        conv_g = ttnn.to_torch(gdn.conv_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))[:, 0].float()
        kd_tp, vd_tp = args.gdn_key_dim // tp, args.gdn_value_dim // tp
        per = 2 * kd_tp + vd_tp
        qs = [conv_g[..., d * per : d * per + kd_tp] for d in range(tp)]
        ks = [conv_g[..., d * per + kd_tp : d * per + 2 * kd_tp] for d in range(tp)]
        vs = [conv_g[..., d * per + 2 * kd_tp : (d + 1) * per] for d in range(tp)]
        conv_state_tt = torch.cat(qs + ks + vs, dim=-1)
    else:
        conv_state_tt = ttnn.to_torch(gdn.conv_state)[:, 0].float()
    conv_ref = conv_state_ref.transpose(1, 2).float()
    passing_c, pcc_c = comp_pcc(conv_ref, conv_state_tt, PCC_TARGET)
    logger.info(f"[eager] prefill conv_state PCC = {pcc_c}  passing={passing_c}")

    # recurrent_state hand-off (sharded over value heads, dim 1).
    if tp > 1:
        rec_state_tt = ttnn.to_torch(
            gdn.last_recurrent_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)
        ).float()
    else:
        rec_state_tt = ttnn.to_torch(gdn.last_recurrent_state).float()
    passing_r, pcc_r = comp_pcc(recurrent_state_ref.float(), rec_state_tt, PCC_TARGET)
    logger.info(f"[eager] prefill recurrent_state PCC = {pcc_r}  passing={passing_r}")

    eager_ok = passing and passing_c and passing_r
    logger.info(f"[eager] all PCC passing = {eager_ok}")

    # ──────────────── Phase 2: capture forward_prefill as a trace + replay ────────────────
    # Capture records the command stream against fixed buffers; replay re-runs it with one
    # host dispatch. With the conv/chunk pads now pre-built in __init__, the forward contains
    # only device-side ops, so capture should no longer hit "Writes not supported during
    # trace capture". The output tensor returned during capture is written into by
    # execute_trace, so after a replay we can read it back and PCC it vs HF.
    logger.info("capturing forward_prefill trace…")
    prefill_tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_trace = gdn.forward_prefill(x_tt)  # output buffer the replay writes into
    ttnn.end_trace_capture(mesh_device, prefill_tid, cq_id=0)
    logger.info("✓ trace captured (no host-write error) — replaying…")

    # One real replay, then read the output back. The capture-run output can be unreliable,
    # but execute_trace genuinely runs the program into out_trace's buffer.
    ttnn.execute_trace(mesh_device, prefill_tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    out_trace_torch = _gather_out(out_trace)
    passing_t, pcc_t = comp_pcc(ref, out_trace_torch, PCC_TARGET)
    logger.info(f"[TRACE] replayed forward_prefill output PCC (S={SEQ_LEN}) = {pcc_t}  passing={passing_t}")

    # Sanity: traced replay should match the eager output, not just HF (catches any
    # capture-vs-eager divergence even if both happen to clear the HF bar).
    passing_te, pcc_te = comp_pcc(out_torch, out_trace_torch, 0.999)
    logger.info(f"[TRACE] replay-vs-eager output PCC = {pcc_te}  passing={passing_te}")

    # ── Timing: warm eager prefill vs trace replay (TTFT). ──
    prefill_times = []
    for _ in range(N_PREFILL_ITERS):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        gdn.forward_prefill(x_tt)
        ttnn.synchronize_device(mesh_device)
        prefill_times.append(time.perf_counter() - t0)
    eager_s = sum(prefill_times) / len(prefill_times)

    replay_times = []
    for _ in range(N_TRACE_REPLAYS + 1):  # drop warm-up replay
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, prefill_tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        replay_times.append(time.perf_counter() - t0)
    trace_s = sum(replay_times[1:]) / len(replay_times[1:])

    ttnn.release_trace(mesh_device, prefill_tid)
    prefill_tid = None  # released; don't double-free in finally

    # ──────────────── summary ────────────────
    logger.info("──────── prefill trace summary ────────")
    logger.info(f"SEQ_LEN={SEQ_LEN}  pad_size={(128 - SEQ_LEN % 128) % 128}  mesh={MESH_SHAPE}  tp={tp}")
    logger.info(f"eager PCC (out/conv/rec) : {pcc:.5f} / {pcc_c:.5f} / {pcc_r:.5f}  passing={eager_ok}")
    logger.info(f"TRACE  PCC (out vs HF)   : {pcc_t:.5f}  passing={passing_t}")
    logger.info(f"TRACE  PCC (out vs eager): {pcc_te:.5f}  passing={passing_te}")
    logger.info(
        f"TTFT eager : {eager_s * 1e3:8.1f} ms  |  {SEQ_LEN / eager_s:>10,.0f} tok/s  (mean of {N_PREFILL_ITERS})"
    )
    logger.info(
        f"TTFT trace : {trace_s * 1e3:8.1f} ms  |  {SEQ_LEN / trace_s:>10,.0f} tok/s  (mean of {N_TRACE_REPLAYS})"
    )
    logger.info(f"RESULT: trace capture {'SUCCEEDED' if passing_t and passing_te else 'CAPTURED but PCC FAILED'}")
finally:
    if prefill_tid is not None:
        ttnn.release_trace(mesh_device, prefill_tid)
    ttnn.close_mesh_device(mesh_device)
    if MESH_SHAPE != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
