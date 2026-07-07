# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-op device-time profile of TPGatedDeltaNet.forward_prefill (the FULL TP GDN layer).

Unlike the §0b harness (which profiled only the GDN chunk pipeline in isolation),
this drives the whole `forward_prefill` — the two fused projections, the causal
conv1d+SiLU, the GDN core (chunk_gated_delta_rule_seq_adapter), the gated
RMSNorm + Z-gate, the row-parallel out-proj, and the final all-reduce — so the
Tracy device profiler can attribute DEVICE KERNEL DURATION per op and let you
see the GDN-core share RELATIVE to projections / conv / out-proj / comms.

With GDN_PROFILE=1, `gdn/tp.py` emits Tracy signposts at each data-flow stage:
    PF_proj  PF_conv  PF_gdn_core  PF_state_carry  PF_gate_norm  PF_out_proj  PF_all_reduce
Ops between two signposts belong to that region; sum DEVICE KERNEL DURATION [ns]
per region to get the stage breakdown.

Run (ONE T per capture for clean op attribution — pick with -k):

    GDN_PROFILE=1 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      python -m tracy -r -p -m pytest \
      models/demos/blackhole/qwen36/tests/test_forward_prefill_perf.py -k T2048 -s

Reading results:
  * Merged, op-named report: generated/profiler/reports/<ts>/ops_perf_results*.csv
    (column DEVICE KERNEL DURATION [ns]); filter rows between the MEASURE_T*
    signposts, then bucket by the PF_* signpost markers.
  * If the host<->device merge in tools/tracy/process_ops_logs.py crashes on
    host-only ops (see handoff §0b), the raw per-op device report is still at
    generated/profiler/.logs/cpp_device_perf_report.csv (durations only, no names).

Device-kernel time is trace-independent, so these eager numbers are the real
on-device floor for the whole GDN layer at each prefill window.

No-checkpoint mode (GDN_PERF_RANDOM_WEIGHTS=1):
  Skips the FP8 GDN-layer read and synthesizes a random state dict with the exact
  keys/shapes load_gdn_weights_tp expects. Device timing is weight-VALUE
  independent, so this changes nothing measured. Qwen36ModelArgs still needs the
  model's config.json to derive dims — but it only snapshot_downloads when
  config.json is absent locally (model_config.py:50), so point HF_MODEL at a dir
  that already has config.json (or a cached snapshot) to avoid the shard fetch:

    GDN_PERF_RANDOM_WEIGHTS=1 HF_MODEL=/path/to/qwen36-config-only-dir \
    MESH_DEVICE=P150x4 python -m tracy -r -p -m pytest \
      models/demos/blackhole/qwen36/tests/test_forward_prefill_perf.py -k T2048 -s
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import (
    _resolve_mesh_shape,
    load_gdn_layer,
    model_path,
    parametrize_mesh_only,
    parametrize_mesh_tp,
    replicate_to_device,
    tp_composer,
)

# Single device (1x1) needs NO fabric — tt_all_reduce short-circuits at mesh_shape==[1,1],
# and initializing FABRIC_1D on a multi-card box times out on ethernet-router sync. Use the
# fabric-less parametrization for 1x1, the fabric one for real TP. Resolved from MESH_DEVICE.
_MESH = parametrize_mesh_only() if _resolve_mesh_shape() == (1, 1) else parametrize_mesh_tp()
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs

try:
    from tracy import signpost
except Exception:  # tracy not built / not importable

    def signpost(*_args, **_kwargs):
        pass


def _synth_gdn_sd(args):
    """Random GDN state dict with the exact keys/shapes load_gdn_weights_tp expects.

    Values are irrelevant to device-kernel timing (data-independent ops), so this
    lets the harness profile without reading the FP8 checkpoint shards. Keys carry
    the ``linear_attn.`` prefix so load_gdn_weights_tp's prefix detection matches.
    """
    tp = args.num_devices
    dim = args.dim
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    nv, dv = args.gdn_nv, args.gdn_dv
    K = args.gdn_conv_kernel_size
    z_total = args.gdn_z_dim_tp * tp  # z is column-sharded per device (z_per * tp)
    qkv_ch = 2 * key_dim + value_dim  # fused Q|K|V projection / conv channels
    gen = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    return {
        "linear_attn.in_proj_qkv.weight": r(qkv_ch, dim),
        "linear_attn.in_proj_z.weight": r(z_total, dim),
        "linear_attn.in_proj_a.weight": r(nv, dim),
        "linear_attn.in_proj_b.weight": r(nv, dim),
        "linear_attn.out_proj.weight": r(dim, value_dim),  # row-parallel: [out=dim, in=value_dim]
        "linear_attn.conv1d.weight": r(qkv_ch, 1, K),
        "linear_attn.A_log": r(nv),
        "linear_attn.dt_bias": r(nv),
        "linear_attn.norm.weight": r(dv),  # gated RMSNorm over head_v dim
    }


# (ISL, outer prefill window). window == ISL -> single forward_prefill call.
# window < ISL -> chunk-outer prefill: the ISL is processed in `window`-token
# windows with GDN recurrent + conv state carried across them (_stable_state).
# NOTE: `window` is the OUTER prefill chunk (the issue's "chunk_size 2048"); the
# delta-rule kernel chunk is always 128 (TT_FATAL-fixed), passed as chunk_size=128.
_CASES = [
    (512, 512),
    (1024, 1024),
    (2048, 2048),
    (4096, 2048),  # 2 windows of 2k
    (8192, 2048),  # 4 windows of 2k
]
_CASE_IDS = ["T512", "T1024", "T2048", "T4096w2048", "T8192w2048"]


@torch.no_grad()
@pytest.mark.parametrize("T,window", _CASES, ids=_CASE_IDS)
@_MESH
def test_forward_prefill_perf(mesh_device, T, window, reset_seeds, ensure_gc, request):
    """Profile forward_prefill at ISL=T, processed in `window`-token outer windows.

    Weights loaded (or synthesized); values don't affect device-kernel timing.
    The MEASURE_T* window sums all outer windows -> total device time for the ISL.
    """
    os.environ.setdefault("HF_MODEL", model_path())
    windowed = window < T
    nwin = (T + window - 1) // window
    # max_seq_len only needs to cover ONE outer window (each forward_prefill sees `window` tokens;
    # GDN state is O(1), so the ISL never appears as a single tensor dim).
    args = Qwen36ModelArgs(mesh_device, max_batch_size=1, max_seq_len=max(256, window + 128))
    nd = mesh_device.get_num_devices()
    # Qwen36ModelArgs gates the TP config (gdn_*_tp dims) on nd>1. For single-device profiling
    # (chosen because the 4-card Tracy path has a device->host clock-sync bug that corrupts
    # multi-core op durations), force it so TPGatedDeltaNet builds with tp=1 (each "shard" = full
    # tensor; all_reduce short-circuits at mesh_shape==[1,1]). Clean per-stage COMPUTE timing;
    # the only stage this misses is the all_reduce (a no-op at 1x1).
    if nd == 1 and not hasattr(args, "gdn_nk_tp"):
        args._init_tp_config(mesh_device)
    li = next(i for i, t in enumerate(args.attention_type_list) if t == "linear_attention")
    logger.info(
        f"forward_prefill perf: devices={nd} gdn_layer={li} ISL={T} window={window} nwin={nwin} "
        f"Nk_tp={args.gdn_nk_tp} Nv_tp={args.gdn_nv_tp} dim={args.dim}"
    )

    # Weights: real (FP8 checkpoint) or synthesized. Timing is data-independent for
    # these ops, so weight *values* don't affect the profile — GDN_PERF_RANDOM_WEIGHTS
    # skips the shard read entirely (see module docstring).
    if os.environ.get("GDN_PERF_RANDOM_WEIGHTS") == "1":
        logger.info("GDN_PERF_RANDOM_WEIGHTS=1 -> synthesizing random GDN weights (no checkpoint shard read)")
        sd = _synth_gdn_sd(args)
    else:
        sd = load_gdn_layer(args.CKPT_DIR, li)
    from models.tt_transformers.tt.ccl import TT_CCL

    tt_ccl = TT_CCL(mesh_device) if nd > 1 else None
    tw = load_gdn_weights_tp(mesh_device, sd, args)
    gdn = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)

    x = torch.randn(1, 1, T, args.dim, dtype=torch.bfloat16)  # host; sliced+replicated per window

    def _win_tt(w):  # replicate outer window w -> [1,1,win,dim] across the mesh
        return replicate_to_device(mesh_device, x[:, :, w * window : min((w + 1) * window, T), :])

    # Chunk-outer prefill carries GDN recurrent + conv state across windows in-place.
    if windowed:
        gdn._stable_state = True

    # ---- Warmup: compiles kernels + primes allocators (excluded from MEASURE region). ----
    # One window compiles the (single) per-window program reused by every window.
    gdn.reset_state()
    out = gdn.forward_prefill(_win_tt(0), chunk_size=128)
    ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(out)

    # ---- Measured pass: MEASURE_T* brackets the whole ISL (all outer windows). ----
    gdn.reset_state()  # zero the carried state -> from-scratch sequence
    signpost(f"MEASURE_T{T}")
    out = None
    for w in range(nwin):
        if out is not None:
            ttnn.deallocate(out)  # keep only the last window's output for the shape check
        out = gdn.forward_prefill(_win_tt(w), chunk_size=128)
        ttnn.synchronize_device(mesh_device)
    signpost(f"MEASURE_T{T}_END")

    last_win = T - (nwin - 1) * window  # size of the final window
    o = ttnn.to_torch(out, mesh_composer=tp_composer(mesh_device))[0, 0].float()  # [last_win, dim]
    assert o.shape == (last_win, args.dim), f"unexpected output shape {tuple(o.shape)}"
    assert not torch.isnan(o).any() and o.abs().max() > 0
    logger.info(
        f"forward_prefill ISL={T} window={window} nwin={nwin} OK: "
        f"last-window out {tuple(o.shape)} max|.|={o.abs().max().item():.4g}"
    )
