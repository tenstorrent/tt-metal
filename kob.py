# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Quick standalone harness for the Qwen3.5 gated-delta-net (GDN) prefill→decode flow.

Builds one HF Qwen3_5GatedDeltaNet layer with random weights, loads those SAME
weights into our TT Qwen35GatedDeltaNet module, then runs the full prefill→decode
path on a P150 mesh, PCC-checking each stage against the HF golden:
  * prefill: forward_prefill on a 2026-token prompt; PCC the output AND the two
    hand-off states (conv window + recurrent KV state) it leaves behind, since
    those — not the output — are what decode reads back.
  * decode: two further single-token steps via decode_forward, each continuing
    from the prefill-filled states. HF advances the same DynamicCache, so this
    exercises recurrent_gated_delta_rule and the conv-state roll, and step 1 only
    passes if step 0 wrote correct states.
It's the GDN sibling of rob.py (the gated full-attention prefill→decode harness),
in script form so the conv / delta-rule ops can be poked at a breakpoint.

NOTE: the HF Qwen3_5GatedDeltaNet.forward in site-packages has a breakpoint()
right before its return, so each hf_gdn() call (1 prefill + 2 decode) will drop
into pdb — type `c` to continue, or run with PYTHONBREAKPOINT=0 for a clean pass.

Run from the repo root:  python kob.py
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

# HF_MODEL is the single source of truth for dims (Qwen35ModelArgs parses it). Default to the
# bare hub id only as a fallback; override with `export HF_MODEL=/path/to/Qwen3.5-27B-FP8` to
# point at the local snapshot on this host and skip a multi-GB snapshot_download.
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-27B")

torch.manual_seed(0)

SEQ_LEN = 8192
PCC_TARGET = 0.99  # bf16 GDN may land lower (the old TP tests used ~0.95); loosen as needed.
# Mesh shape: (1, 4) runs the GDN tensor-parallel across all 4 P150s; (1, 1) is the original
# single-device path. The head counts in HF_MODEL must divide by the device count (9B: 16 K /
# 32 V, 27B: 16 K / 48 V both divide by 4) and hidden/devices must stay tile-aligned.
MESH_SHAPE = (1, 4)

# CCL ops (the out_proj reduce-scatter, TP>1 only) ride the ethernet fabric, which must be
# enabled BEFORE the mesh opens — the FABRIC_1D the 9B demo sets for its multi-device mesh.
# Skip it on a single device, where no collective runs.
if MESH_SHAPE != (1, 1):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))

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

    # TT prefill expects hidden_states as [B=1, 1, S, dim], replicated across the mesh.
    x_tt = ttnn.from_torch(
        x.to(torch.bfloat16).reshape(1, 1, SEQ_LEN, args.dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Time prefill = TTFT (time to first token): the cost of consuming the whole prompt before
    # decode can emit anything. ttnn dispatches ops asynchronously, so a bare timer around the
    # call would only clock host-side dispatch; the synchronize_device on both sides forces the
    # device to drain so we measure real on-device compute. NOTE: this is a cold, single-shot run,
    # so prefill_s folds in one-time kernel compilation — it's a ceiling, not steady-state latency.
    ttnn.synchronize_device(mesh_device)
    t_prefill_start = time.perf_counter()
    out = gdn.forward_prefill(x_tt)
    ttnn.synchronize_device(mesh_device)
    prefill_s = time.perf_counter() - t_prefill_start
    logger.info(
        f"prefill latency (TTFT) = {prefill_s * 1e3:.1f} ms for {SEQ_LEN} tokens "
        f"→ prefill throughput {SEQ_LEN / prefill_s:,.0f} tok/s"
    )

    # forward_prefill now returns rank-4 [1, 1, S, dim] in both cases (the out_proj reduce no longer
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
    # match HF's cached states, or decode diverges from the very first token. These
    # are the goldens this file's header promised once the kernel grew past the
    # projections. conv_state_ref is HF's channels-first [1, conv_dim, K]; ours is
    # channels-last [B, 1, K, conv_dim], so transpose the ref to line them up.
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

    # ── Decode: two single-token steps continuing from the prefill-filled states ──
    # HF reads & updates the SAME DynamicCache on each call: with seq_len==1 and a
    # populated cache, use_precomputed_states flips True so the recurrent path (not
    # the chunk path) runs and rolls conv_states / recurrent_states forward. TT's
    # decode_forward likewise reads & rewrites the self.conv_state and
    # self.last_recurrent_state that forward_prefill just wrote. So this exercises
    # recurrent_gated_delta_rule + the conv-window roll across two steps, not just
    # the input projections — and the step-1 PCC only holds if step-0 wrote correct
    # states, making it a genuine multi-step recurrence check.
    B = args.max_batch_size
    # The prefill golden above ran batch-1 (x is [1, SEQ_LEN, dim]), so HF lazily
    # sized its cached conv/recurrent states with batch 1. Decode must match that
    # batch; assert loudly rather than let a future max_batch_size > 1 surface as a
    # confusing HF broadcast error deep in the recurrent path.
    assert B == 1, f"kob.py decode assumes max_batch_size==1 (prefill golden is batch-1); got {B}"
    # Per-step decode latencies feed TPOT (time per output token) and decode throughput below.
    # Kept as a list so we can report the steady-state mean separately from step 0, which eats
    # the one-time decode-kernel compile and would otherwise inflate the average.
    decode_times = []
    for step in range(100):
        pos = SEQ_LEN + step
        # Re-seed per step so the two decode tokens differ but stay reproducible.
        torch.manual_seed(1000 + step)
        x_dec = torch.randn(B, 1, args.dim, dtype=torch.float32)

        with torch.no_grad():
            ref_dec = hf_gdn(x_dec, cache_params=cache, attention_mask=None)
        ref_dec = ref_dec[:, 0]  # [B, dim]

        # TT decode layout: token replicated as [1, 1, B, dim] (batch on dim 2, the
        # convention decode_forward decodes via hidden_states.shape[2]).
        x_dec_tt = ttnn.from_torch(
            x_dec.to(torch.bfloat16).reshape(1, 1, B, args.dim),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        # Same async caveat as prefill: synchronize on both sides so the timer captures the
        # token's on-device recurrence (delta-rule + conv roll), not just op dispatch.
        ttnn.synchronize_device(mesh_device)
        t_step_start = time.perf_counter()
        out_dec = gdn.decode_forward(x_dec_tt)
        ttnn.synchronize_device(mesh_device)
        decode_times.append(time.perf_counter() - t_step_start)

        if tp > 1:
            out_dec_torch = (
                ttnn.to_torch(out_dec, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
                .reshape(B, args.dim)
                .float()
            )  # [B, dim]
        else:
            out_dec_torch = ttnn.to_torch(out_dec).reshape(B, args.dim).float()  # [B, dim]
        passing_d, pcc_d = comp_pcc(ref_dec, out_dec_torch, PCC_TARGET)
        logger.info(
            f"GDN decode step {step} PCC (pos={pos}, B={B}) = {pcc_d}  passing={passing_d} "
            f"({decode_times[-1] * 1e3:.2f} ms)"
        )

    # ── Timing summary: TTFT / TPOT / throughput ──
    # TPOT (time per output token) is the mean decode-step latency. Step 0 carries the one-time
    # decode-kernel compile, so we also report a steady-state mean over steps 1+; decode throughput
    # is just B / TPOT (B==1 here). Prefill numbers come from prefill_s captured above.
    tpot_all = sum(decode_times) / len(decode_times)
    steady = decode_times[1:] or decode_times  # drop the compile-laden first step when we can
    tpot_steady = sum(steady) / len(steady)
    logger.info("──────── timing summary ────────")
    logger.info(
        f"prefill (TTFT)    : {prefill_s * 1e3:8.1f} ms  |  {SEQ_LEN / prefill_s:>10,.0f} tok/s  "
        f"(prompt={SEQ_LEN} tokens, cold incl. compile)"
    )
    logger.info(
        f"decode TPOT (all) : {tpot_all * 1e3:8.2f} ms  |  {B / tpot_all:>10,.1f} tok/s  "
        f"(mean over {len(decode_times)} steps)"
    )
    logger.info(
        f"decode TPOT (warm): {tpot_steady * 1e3:8.2f} ms  |  {B / tpot_steady:>10,.1f} tok/s  "
        f"(mean over steps 1..{len(decode_times) - 1}, excl. compile)"
    )
finally:
    ttnn.close_mesh_device(mesh_device)
    if MESH_SHAPE != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
