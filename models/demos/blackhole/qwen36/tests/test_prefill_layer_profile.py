# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer prefill (TTFT) profiler: GDN vs full-attention, across ISLs.

Answers "is TTFT dominated by the chunked GDN or by the full attention layers?"
by timing ONE representative GDN layer (checkpoint layer 0) and ONE representative
full-attention layer (checkpoint layer 3) through a *full eager* chunked prefill at
several input sequence lengths, then scaling each by its real layer count.

Why eager + full-ISL (not the demo's traced replay): the traced chunk-outer prefill
only exposes an eager, per-layer-attributable pass for a *single* 2048-token chunk at
chunk_start=0 (cheapest full-attention chunk). That hides the fact that full attention
grows ~O(L^2) with context while chunked GDN is ~O(L). Here we drive the eager chunked
path (`prefill_traced_chunked` with no trace captured -> `_prefill_chunked_eager_tp`),
which processes every chunk with the correct growing chunk_start, and we sum each
layer's device-synced wall time across all chunks.

Metric: device-synced wall clock around each sublayer call (includes dispatch, so it is
slightly inflated vs. a fully pipelined run, but the GDN-vs-attention comparison is fair
because both are measured identically). This is the quantity that actually makes up TTFT.

Run (P150x4, the benchmarked TP config):
    MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen36/tests/test_prefill_layer_profile.py -v -s

Tunables (env):
    QWEN36_PROFILE_ISLS=4096,16384,32768,65536   # ISLs to sweep (multiples of 2048)
    QWEN36_PROFILE_REPS=3                         # timed repeats per ISL (median reported)
"""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

_MESH_SHAPE = {"P150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 4))
_MULTI = _MESH_SHAPE != (1, 1)
_TP_TRACE_REGION_SIZE = 1024 * 1024 * 1024
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **(
            {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": _TP_TRACE_REGION_SIZE} if _MULTI else {}
        ),
    }
]

BLOCK_SIZE = 64
CHUNK = 2048  # GDN long_prefill_chunk_size drives chunk-outer prefill
# Representative layers: 0 is linear_attention (GDN), 3 is full_attention in the [l,l,l,f]*N pattern.
GDN_PROBE_LAYER = 0
ATTN_PROBE_LAYER = 3
RESULTS_PATH = Path("models/demos/blackhole/qwen36") / "prefill_layer_profile_results.json"


def _isls():
    env = os.environ.get("QWEN36_PROFILE_ISLS")
    if env:
        return [int(x) for x in env.split(",") if x.strip()]
    return [4096, 16384, 32768, 65536]


def _num_blocks_for(seqlen):
    """Paged-KV block budget for seqlen, rounded up to a multiple of 32 (TP SDPA alignment)."""
    blocks = (seqlen + BLOCK_SIZE - 1) // BLOCK_SIZE
    return ((blocks + 31) // 32) * 32


def _fresh_acc():
    keys = ["gdn_layer", "attn_layer", "gdn_attn", "attn_attn", "gdn_mlp", "attn_mlp"]
    return {k: {"t": 0.0, "n": 0} for k in keys}


def _reset(acc):
    for k in acc:
        acc[k]["t"] = 0.0
        acc[k]["n"] = 0


def _timed(fn, key, acc, device):
    """Wrap a bound method with device-synced wall-clock accumulation into acc[key]."""

    def wrapped(*args, **kwargs):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        ttnn.synchronize_device(device)
        acc[key]["t"] += time.perf_counter() - t0
        acc[key]["n"] += 1
        return out

    return wrapped


def _instrument(model, acc, device):
    """Wrap layer/attention/MLP forwards on the two probe layers; return restore list."""
    orig = []
    for layer in model.layers:
        typ = "attn" if layer.is_full_attention else "gdn"

        f = layer.forward
        orig.append((layer, "forward", f))
        layer.forward = _timed(f, f"{typ}_layer", acc, device)

        mf = layer.feed_forward.forward
        orig.append((layer.feed_forward, "forward", mf))
        layer.feed_forward.forward = _timed(mf, f"{typ}_mlp", acc, device)

        attn = layer.attention
        candidates = (
            ["forward_prefill_paged", "forward_prefill", "forward"]
            if layer.is_full_attention
            else ["forward_prefill", "forward"]
        )
        for name in candidates:
            if hasattr(attn, name):
                af = getattr(attn, name)
                orig.append((attn, name, af))
                setattr(attn, name, _timed(af, f"{typ}_attn", acc, device))
                break
    return orig


def _restore(orig):
    for obj, name, f in orig:
        setattr(obj, name, f)


def _run_eager_prefill(model, token_ids, page_table, T):
    """Drive a full eager prefill of the whole ISL (no trace -> per-chunk eager path)."""
    if model.num_devices > 1:
        # No trace captured -> prefill_traced_chunked falls back to _prefill_chunked_eager_tp,
        # which walks every chunk with the correct growing chunk_start (real attention growth).
        assert model._chunked_trace_id is None, "trace must not be captured for the eager path"
        return model.prefill_traced_chunked(token_ids, page_table, actual_len=T)
    # Single device: prefill_paged -> prefill_layer_chunked (eager, layer-outer/chunk-inner).
    return model.prefill_paged(token_ids, page_table)


def _dealloc(out):
    try:
        if out is not None and hasattr(out, "layout"):  # ttnn device tensor
            ttnn.deallocate(out)
    except Exception:
        pass


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_prefill_layer_profile(mesh_device):
    device = mesh_device
    device.enable_program_cache()

    isls = _isls()
    for T in isls:
        assert T % CHUNK == 0, f"ISL {T} must be a multiple of chunk size {CHUNK}"
    max_seq_len = _num_blocks_for(max(isls)) * BLOCK_SIZE

    t0 = time.time()
    model = Qwen36Model.from_pretrained(
        device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        layer_indices=[GDN_PROBE_LAYER, ATTN_PROBE_LAYER],  # one GDN + one full-attn layer
    )
    logger.info(f"Model load: {time.time() - t0:.1f}s")

    # Real full-model layer counts (attention_type_list is NOT truncated by layer_indices).
    types = list(model.args.attention_type_list)
    n_gdn = sum(1 for t in types if t == "linear_attention")
    n_full = sum(1 for t in types if t == "full_attention")
    logger.info(f"Full model: {len(types)} layers = {n_gdn} GDN + {n_full} full-attention")
    assert n_gdn > 0 and n_full > 0, f"unexpected layer types: {set(types)}"

    reps = int(os.environ.get("QWEN36_PROFILE_REPS", "3"))
    acc = _fresh_acc()

    n_local_kv = getattr(model.args, "n_local_kv_heads", model.args.n_kv_heads)
    kv_heads = n_local_kv if model.num_devices > 1 else model.args.n_kv_heads

    results = []
    for T in isls:
        num_blocks = _num_blocks_for(T)
        num_chunks = T // CHUNK
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        # Token content is irrelevant to timing; avoid id 0 (corrupts GDN state per model note).
        token_ids = (torch.arange(T, dtype=torch.long) % 2000 + 1).reshape(1, T)

        kv_shape = [num_blocks, kv_heads, BLOCK_SIZE, model.args.head_dim]
        model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

        orig = _instrument(model, acc, device)
        try:
            # Warmup (compile programs); timings discarded.
            _reset(acc)
            _dealloc(_run_eager_prefill(model, token_ids, page_table, T))
            ttnn.synchronize_device(device)

            samples = []
            for _ in range(reps):
                _reset(acc)
                _dealloc(_run_eager_prefill(model, token_ids, page_table, T))
                ttnn.synchronize_device(device)
                samples.append({k: acc[k]["t"] for k in acc})
        finally:
            _restore(orig)
            model.free_kv_caches()

        def med(key):
            vals = sorted(s[key] for s in samples)
            return vals[len(vals) // 2]

        # One representative layer, summed over the whole ISL prefill (all chunks).
        gdn_layer = med("gdn_layer")
        attn_layer = med("attn_layer")
        gdn_attn = med("gdn_attn")
        attn_attn = med("attn_attn")
        gdn_mlp = med("gdn_mlp")
        attn_mlp = med("attn_mlp")

        # Scale to the full model.
        gdn_attn_total = n_gdn * gdn_attn
        attn_attn_total = n_full * attn_attn
        gdn_layer_total = n_gdn * gdn_layer
        attn_layer_total = n_full * attn_layer
        mech = gdn_attn_total + attn_attn_total

        row = {
            "isl": T,
            "chunks": num_chunks,
            "per_layer_ms": {"gdn": gdn_layer * 1e3, "attn": attn_layer * 1e3},
            "attn_mech_ms": {"gdn": gdn_attn * 1e3, "attn": attn_attn * 1e3},
            "mlp_ms": {"gdn": gdn_mlp * 1e3, "attn": attn_mlp * 1e3},
            "full_model_attn_mech_ms": {"gdn": gdn_attn_total * 1e3, "attn": attn_attn_total * 1e3},
            "full_model_layer_ms": {"gdn": gdn_layer_total * 1e3, "attn": attn_layer_total * 1e3},
            "gdn_share_of_mechanism_pct": (100.0 * gdn_attn_total / mech) if mech > 0 else 0.0,
            "gdn_over_attn_ratio": (gdn_attn_total / attn_attn_total) if attn_attn_total > 0 else float("inf"),
        }
        results.append(row)
        logger.info(
            f"[ISL {T:>6} / {num_chunks:>2} chunks] "
            f"1x GDN layer={gdn_layer*1e3:8.1f}ms (mech {gdn_attn*1e3:7.1f}) | "
            f"1x full-attn layer={attn_layer*1e3:8.1f}ms (mech {attn_attn*1e3:7.1f}) || "
            f"model GDN-mech={gdn_attn_total*1e3:8.1f}ms x{n_gdn} vs attn-mech={attn_attn_total*1e3:8.1f}ms x{n_full} "
            f"=> GDN {row['gdn_share_of_mechanism_pct']:5.1f}% (x{row['gdn_over_attn_ratio']:.2f})"
        )

    _print_report(results, n_gdn, n_full, reps)
    RESULTS_PATH.write_text(json.dumps({"n_gdn": n_gdn, "n_full": n_full, "reps": reps, "rows": results}, indent=2))
    logger.info(f"Saved: {RESULTS_PATH}")


def _print_report(results, n_gdn, n_full, reps):
    lines = []
    lines.append("")
    lines.append("=" * 108)
    lines.append(
        f"PER-LAYER PREFILL PROFILE (device-synced wall clock, median of {reps} reps) — "
        f"full model = {n_gdn} GDN + {n_full} full-attn layers"
    )
    lines.append("=" * 108)
    lines.append("")
    lines.append("Attention/GDN mechanism only (isolates the sublayer under debate, excludes MLP):")
    lines.append(
        f"  {'ISL':>7} {'chunks':>6} | {'1x GDN':>10} {'1x attn':>10} | "
        f"{'model GDN':>12} {'model attn':>12} | {'GDN share':>10} {'GDN/attn':>9}"
    )
    lines.append("  " + "-" * 96)
    for r in results:
        lines.append(
            f"  {r['isl']:>7} {r['chunks']:>6} | "
            f"{r['attn_mech_ms']['gdn']:>9.1f}m {r['attn_mech_ms']['attn']:>9.1f}m | "
            f"{r['full_model_attn_mech_ms']['gdn']:>11.1f}m {r['full_model_attn_mech_ms']['attn']:>11.1f}m | "
            f"{r['gdn_share_of_mechanism_pct']:>9.1f}% {r['gdn_over_attn_ratio']:>8.2f}x"
        )
    lines.append("")
    lines.append("Whole decoder layer (norm + mechanism + MLP + residuals), for reference:")
    lines.append(f"  {'ISL':>7} | {'1x GDN layer':>13} {'1x attn layer':>13} | {'model GDN':>12} {'model attn':>12}")
    lines.append("  " + "-" * 72)
    for r in results:
        lines.append(
            f"  {r['isl']:>7} | "
            f"{r['per_layer_ms']['gdn']:>12.1f}m {r['per_layer_ms']['attn']:>12.1f}m | "
            f"{r['full_model_layer_ms']['gdn']:>11.1f}m {r['full_model_layer_ms']['attn']:>11.1f}m"
        )
    lines.append("=" * 108)
    report = "\n".join(lines)
    logger.info(report)
    print(report)
