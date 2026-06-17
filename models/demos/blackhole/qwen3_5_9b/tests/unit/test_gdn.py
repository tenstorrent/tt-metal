# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unit test for the Qwen3.5 Gated DeltaNet (linear-attention) kernel vs the HF golden.

Run:
    pytest models/demos/blackhole/qwen3_5_9b/tests/unit/test_gdn.py -v
"""
import os

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen3_5_9b.tt.gdn import Qwen35GatedDeltaNet
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.tt_transformers.tt.ccl import TT_CCL

### Test Parameters & Fixtures ─────────────────────────────────────────────────────────
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

BATCHES = [1, 32]
PREFILL_SEQLEN = [512]
DECODE_STEPS = 5
PCC = 0.99
TRACE_REGION_SIZE = 268435456


@pytest.fixture
def setup(mesh_device, layer_idx=0) -> tuple[Qwen3_5GatedDeltaNet, dict, Qwen35ModelArgs]:
    """
    Returns HF Reference GDN, its state_dict, and Qwen35ModelArgs.
    """
    args = Qwen35ModelArgs(mesh_device, max_seq_len=max(PREFILL_SEQLEN))
    cfg = args.hf_config.get_text_config()
    hf_gdn = Qwen3_5GatedDeltaNet(config=cfg, layer_idx=layer_idx).to(torch.float32).eval()
    cache = DynamicCache(config=cfg)
    return hf_gdn, hf_gdn.state_dict(), args, cache


# ── Helpers (host torch ⇄ possibly-sharded TT tensors) ───────────────────────────────
def _build_tt_gdn(mesh_device, state_dict, args, batch) -> Qwen35GatedDeltaNet:
    """The TT GDN under test. GDN bakes the batch size into its persistent conv / recurrent state
    buffers (sized at args.max_batch_size in __init__), so pin it to this case's batch before
    constructing. TT_CCL only on a real mesh (it drives the out_proj reduce-scatter); the cache path
    stays None — caching random weights would corrupt a later re-run."""
    args.max_batch_size = batch
    tt_ccl = TT_CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    return Qwen35GatedDeltaNet(args=args, state_dict=state_dict, mesh_device=mesh_device, tt_ccl=tt_ccl)


def _tt_prefill_input(x, mesh_device) -> ttnn.Tensor:
    """Prefill activation layout: [B, seq, dim] host → [B, 1, seq, dim] bf16 DRAM."""
    batch_size, seq_len, hidden_dim = x.shape
    return ttnn.from_torch(
        x.to(torch.bfloat16).reshape(batch_size, 1, seq_len, hidden_dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_decode_input(x, mesh_device) -> ttnn.Tensor:
    """Decode activation layout: [B, 1, dim] host → [1, 1, B, dim] bf16 DRAM"""
    batch_size, _, hidden_dim = x.shape
    return ttnn.from_torch(
        x.to(torch.bfloat16).reshape(1, 1, batch_size, hidden_dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _read_prefill_out(out, mesh_device, batch, seq, dim):
    """forward_prefill returns [B, 1, seq, dim]. On TP the out_proj reduce-scatter leaves it
    fractured along the hidden dim (and flattens batch into seq internally), so gather dim=3 and
    reshape; on a single device the gather is a no-op. Either way → [B, seq, dim]."""
    if mesh_device.get_num_devices() > 1:
        t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    else:
        t = ttnn.to_torch(out)
    return t.reshape(batch, seq, dim).float()


def _read_decode_out(out, mesh_device, batch, dim):
    """forward_decode returns [1, 1, B, dim]; same hidden-dim gather as prefill → [B, dim]."""
    if mesh_device.get_num_devices() > 1:
        t = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))
    else:
        t = ttnn.to_torch(out)
    return t.reshape(batch, dim).float()


def _read_conv_state(gdn, mesh_device, args):
    """TT conv_state is [B, 1, K, conv_dim] channels-last → [B, K, conv_dim]. On TP it is sharded
    over conv channels and each device holds [q|k|v] for ITS heads, so a dim-3 gather interleaves
    devices — regroup back to HF's [q_all|k_all|v_all] order."""
    tp = mesh_device.get_num_devices()
    if tp > 1:
        conv_g = ttnn.to_torch(gdn.conv_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))[:, 0].float()
        kd_tp, vd_tp = args.gdn_key_dim // tp, args.gdn_value_dim // tp
        per = 2 * kd_tp + vd_tp
        qs = [conv_g[..., d * per : d * per + kd_tp] for d in range(tp)]
        ks = [conv_g[..., d * per + kd_tp : d * per + 2 * kd_tp] for d in range(tp)]
        vs = [conv_g[..., d * per + 2 * kd_tp : (d + 1) * per] for d in range(tp)]
        return torch.cat(qs + ks + vs, dim=-1)  # [B, K, conv_dim]
    return ttnn.to_torch(gdn.conv_state)[:, 0].float()  # [B, K, conv_dim]


def _read_recurrent_state(gdn, mesh_device):
    """TT recurrent state is [B, Hv, Dk, Dv], sharded over value heads (dim 1); the head reorder keeps
    each device's V heads contiguous and in original order, so a dim-1 gather rebuilds HF head order."""
    if mesh_device.get_num_devices() > 1:
        return ttnn.to_torch(
            gdn.last_recurrent_state, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)
        ).float()
    return ttnn.to_torch(gdn.last_recurrent_state).float()


def _inject_recurrent_state(gdn, rec_rand, mesh_device):
    """Overwrite the TT recurrent (KV) state with rec_rand [B, Hv, Dk, Dv] (global heads, fp32). Shard
    over value heads (dim 1) — the same per-head layout load_gdn_weights uses and _read_recurrent_state
    inverts — so it stays consistent with the HF copy injected in lock-step. fp32 + TILE matches the
    persistent buffer; ttnn.copy keeps its address (the decode trace bakes that in)."""
    rec_tt = ttnn.from_torch(
        rec_rand,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.float32,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    ttnn.copy(rec_tt, gdn.last_recurrent_state)


def _inject_conv_state(gdn, conv_window, mesh_device, args):
    """Overwrite the TT conv window with conv_window [B, K, conv_dim] (global [q_all|k_all|v_all]
    channel order, bf16-valued fp32) — the exact inverse of _read_conv_state. The persistent buffer is
    [B, 1, K, conv_dim] bf16/TILE, so unsqueeze the channels-last window onto dim 1. On TP it is sharded
    over conv channels (dim 3) with each device holding [q|k|v] for ITS heads, so first reorder the
    global channels into that per-device interleave (the regroup _read_conv_state undoes) before the
    dim-3 shard; ttnn.copy keeps the buffer address the decode trace bakes in."""
    tp = mesh_device.get_num_devices()
    if tp > 1:
        kd, vd = args.gdn_key_dim, args.gdn_value_dim
        kd_tp, vd_tp = kd // tp, vd // tp
        q_all, k_all, v_all = conv_window[..., :kd], conv_window[..., kd : 2 * kd], conv_window[..., 2 * kd :]
        blocks = []
        for d in range(tp):  # rebuild [q0|k0|v0 | q1|k1|v1 | ...] so the dim-3 shard hands device d its [q|k|v]
            blocks.append(q_all[..., d * kd_tp : (d + 1) * kd_tp])
            blocks.append(k_all[..., d * kd_tp : (d + 1) * kd_tp])
            blocks.append(v_all[..., d * vd_tp : (d + 1) * vd_tp])
        conv_window = torch.cat(blocks, dim=-1)
        mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
    else:
        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    conv_tt = ttnn.from_torch(
        conv_window.unsqueeze(1),  # [B, K, conv_dim] → [B, 1, K, conv_dim]
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=mapper,
    )
    ttnn.copy(conv_tt, gdn.conv_state)


# Mesh × device-params matrix shared by every test: (1, 1) single device and the (1, 4) TP mesh with
# the 1D fabric the out_proj reduce-scatter rides. trace_region_size is reserved for the trace tests
# (harmless for the eager ones).
_MESH_PARAMS = [
    ((1, 1), {"trace_region_size": TRACE_REGION_SIZE}),
    ((1, 4), {"trace_region_size": TRACE_REGION_SIZE, "fabric_config": ttnn.FabricConfig.FABRIC_1D}),
]


# ── Tests ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
@pytest.mark.parametrize("seq_len", PREFILL_SEQLEN, ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_gdn_prefill(mesh_device, batch, setup, seq_len, reset_seeds, ensure_gc):
    """Eager forward_prefill PCC vs HF across batch on (1, 1) and the (1, 4) TP mesh"""

    # 1. Build reference GDN and TT GDN
    hf_gdn, state_dict, args, hf_cache = setup
    tt_gdn = _build_tt_gdn(mesh_device, state_dict, args, batch)

    # 2. Instantiate random input prefill tensor.
    x = torch.randn([batch, seq_len, args.dim], dtype=torch.float32)  # [B, seq, dim]
    x_tt = _tt_prefill_input(x, mesh_device)  # [B, 1, seq, dim]

    # 3. HF reference output
    out_ref = hf_gdn(x, cache_params=hf_cache, attention_mask=None)  # [B, seq, dim]

    # 4. TT GDN output
    out_tt = tt_gdn.forward_prefill(x_tt)
    out_tt = _read_prefill_out(out_tt, mesh_device, batch, seq_len, args.dim)

    # 5. Compare reference vs TT output PCC
    passing, pcc = comp_pcc(out_ref, out_tt, PCC)
    logger.info(f"GDN prefill OUTPUT PCC (mesh={tuple(mesh_device.shape)}, b={batch}, S={seq_len}) = {pcc}")
    assert passing, f"prefill output PCC too low (b={batch}): {pcc}"

    # 6. Compare reference conv state vs TT
    conv_ref = hf_cache.layers[0].conv_states.transpose(1, 2).float()  # HF [B, conv_dim, K] → [B, K, conv_dim]
    conv_tt = _read_conv_state(tt_gdn, mesh_device, args)
    passing, pcc = comp_pcc(conv_ref, conv_tt, PCC)
    logger.info(f"GDN prefill CONV_STATE PCC (b={batch}) = {pcc}")
    assert passing, f"prefill conv_state PCC too low (b={batch}): {pcc}"

    # 7. Compare reference recurrent state vs TT
    rec_ref = hf_cache.layers[0].recurrent_states.float()  # [B, Hv, Dk, Dv]
    rec_tt = _read_recurrent_state(tt_gdn, mesh_device)
    passing, pcc = comp_pcc(rec_ref, rec_tt, PCC)
    logger.info(f"GDN prefill RECURRENT_STATE PCC (b={batch}) = {pcc}")
    assert passing, f"prefill recurrent_state PCC too low (b={batch}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_gdn_decode(mesh_device, batch, setup, reset_seeds, ensure_gc):
    """Eager forward_decode PCC vs HF across batch on (1, 1) and the (1, 4) TP mesh."""
    # 1. Build reference GDN and TT GDN
    hf_gdn, state_dict, args, hf_cache = setup
    tt_gdn = _build_tt_gdn(mesh_device, state_dict, args, batch)

    # 2. Check HF vs TT GDN config matches.
    tp = mesh_device.get_num_devices()
    assert tt_gdn.conv_dim * tp == hf_gdn.conv_dim, "HF conv_dim must match TT conv_dim (per-device × TP)"
    assert tt_gdn.conv_kernel_size == hf_gdn.conv_kernel_size, "HF conv_kernel_size must match TT conv_kernel_size"
    assert tt_gdn.head_v_dim == hf_gdn.head_v_dim, "HF head_v_dim must match TT head_v_dim"
    assert tt_gdn.head_k_dim == hf_gdn.head_k_dim, "HF head_k_dim must match TT head_k_dim"

    # 3. Create random conv + recurrent states, simulating the result of prefill
    conv_state = torch.randn([batch, hf_gdn.conv_kernel_size, hf_gdn.conv_dim])
    recurrent_state = torch.randn(batch, hf_gdn.num_v_heads, hf_gdn.head_k_dim, hf_gdn.head_v_dim, dtype=torch.float32)

    # 4. Inject the random states in HF cache
    hf_cache.update_conv_state(conv_state.transpose(1, 2).contiguous(), layer_idx=0)
    hf_cache.update_recurrent_state(recurrent_state, layer_idx=0)

    # 5. Inject the random states in TT GDN's persistent caches
    _inject_conv_state(tt_gdn, conv_state, mesh_device, args)
    _inject_recurrent_state(tt_gdn, recurrent_state, mesh_device)

    # 6. Run multiple decode steps, comparing the HF vs TT output PCC each step. This confirms forward_decode
    for step in range(DECODE_STEPS):
        x = torch.randn(batch, 1, args.dim, dtype=torch.float32)
        x_tt = _tt_decode_input(x, mesh_device)

        # Reference decode step, updates the hf_cache states every step
        ref = hf_gdn(x, cache_params=hf_cache, attention_mask=None)[:, 0]  # [B, dim]

        out_tt = tt_gdn.forward_decode(x_tt)
        out_tt = _read_decode_out(out_tt, mesh_device, batch, args.dim)

        passing, pcc = comp_pcc(ref, out_tt, PCC)
        logger.info(f"GDN decode step {step} PCC (mesh={tuple(mesh_device.shape)}, b={batch}) = {pcc}")
        assert passing, f"decode PCC too low (b={batch}, step={step}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("seq_len", PREFILL_SEQLEN, ids=lambda s: f"seq{s}")
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_gdn_prefill_trace(mesh_device, batch, setup, seq_len, reset_seeds, ensure_gc):
    """Same prefill PCC as test_gdn_prefill, but forward_prefill runs as a captured ttnn trace — the
    path the demo replays to collapse the unrolled chunk loop, the conv, and the projections into one
    device dispatch. forward_prefill is idempotent (it resets the recurrent state and recomputes the
    conv window each call), so the replay reproduces the eager golden with no state juggling."""

    # 1. Build reference GDN and TT GDN
    hf_gdn, state_dict, args, hf_cache = setup
    tt_gdn = _build_tt_gdn(mesh_device, state_dict, args, batch)

    # 2. Instantiate random input prefill tensor
    x = torch.randn([batch, seq_len, args.dim], dtype=torch.float32)  # [B, seq, dim]
    x_tt = _tt_prefill_input(x, mesh_device)  # [B, 1, seq, dim]

    # 3. HF reference output
    out_ref = hf_gdn(x, cache_params=hf_cache, attention_mask=None)  # [B, seq, dim]

    # 4. Compile the kernels (trace capture cannot compile), then capture once. `out_tt` is the
    #    persistent buffer the trace writes into.
    tt_gdn.forward_prefill(x_tt)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_tt = tt_gdn.forward_prefill(x_tt)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    # 5. Replay recomputes forward_prefill(x) into `out_tt` with a single dispatch, then read it back
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    out_tt = _read_prefill_out(out_tt, mesh_device, batch, seq_len, args.dim)
    ttnn.release_trace(mesh_device, tid)

    # 6. Compare reference vs TT replayed output PCC
    passing, pcc = comp_pcc(out_ref, out_tt, PCC)
    logger.info(f"GDN prefill TRACE PCC (mesh={tuple(mesh_device.shape)}, b={batch}, S={seq_len}) = {pcc}")
    assert passing, f"prefill trace PCC too low (b={batch}): {pcc}"


@torch.no_grad()
@pytest.mark.parametrize("mesh_device, device_params", _MESH_PARAMS, indirect=True)
@pytest.mark.parametrize("batch", BATCHES, ids=lambda b: f"b{b}")
def test_gdn_decode_trace(mesh_device, batch, setup, reset_seeds, ensure_gc):
    """Same decode PCC as test_gdn_decode, but forward_decode runs as a captured ttnn trace — the path
    the demo replays each token to collapse the conv update, the recurrent read/write, and the
    projections into one device dispatch. forward_decode mutates the persistent conv/recurrent buffers
    in place and the trace bakes in the buffer addresses (not the contents), so after capture we
    re-inject the start state S0 into those same buffers before replay — the single replayed step then
    lines up with the HF golden."""

    # 1. Build reference GDN and TT GDN
    hf_gdn, state_dict, args, hf_cache = setup
    tt_gdn = _build_tt_gdn(mesh_device, state_dict, args, batch)

    # 2. Create random conv + recurrent states, simulating the result of prefill
    conv_state = torch.randn([batch, hf_gdn.conv_kernel_size, hf_gdn.conv_dim])
    recurrent_state = torch.randn(batch, hf_gdn.num_v_heads, hf_gdn.head_k_dim, hf_gdn.head_v_dim, dtype=torch.float32)

    # 3. Inject S0 into the HF cache and the TT GDN's persistent caches (lock-step, as in eager decode)
    hf_cache.update_conv_state(conv_state.transpose(1, 2).contiguous(), layer_idx=0)
    hf_cache.update_recurrent_state(recurrent_state, layer_idx=0)
    _inject_conv_state(tt_gdn, conv_state, mesh_device, args)
    _inject_recurrent_state(tt_gdn, recurrent_state, mesh_device)

    # 4. Fixed decode token — its buffer address is baked into the trace, so the exact tensor is reused
    x = torch.randn(batch, 1, args.dim, dtype=torch.float32)
    x_tt = _tt_decode_input(x, mesh_device)

    # 5. HF golden: one decode step from S0 (advances the HF cache, which we don't reuse)
    out_ref = hf_gdn(x, cache_params=hf_cache, attention_mask=None)[:, 0]  # [B, dim]

    # 6. Compile the decode kernels (trace capture cannot compile; this advances TT state — fine),
    #    then capture once. `out_tt` is the persistent buffer the trace writes into.
    tt_gdn.forward_decode(x_tt)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_tt = tt_gdn.forward_decode(x_tt)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)

    # 7. Restore S0 by re-injecting into the buffers the trace reads (inject copies into the existing
    #    persistent buffers, preserving the baked-in addresses), then replay one decode step from S0
    _inject_conv_state(tt_gdn, conv_state, mesh_device, args)
    _inject_recurrent_state(tt_gdn, recurrent_state, mesh_device)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    out_tt = _read_decode_out(out_tt, mesh_device, batch, args.dim)
    ttnn.release_trace(mesh_device, tid)

    # 8. Compare reference vs TT replayed output PCC
    passing, pcc = comp_pcc(out_ref, out_tt, PCC)
    logger.info(f"GDN decode TRACE PCC (mesh={tuple(mesh_device.shape)}, b={batch}) = {pcc}")
    assert passing, f"decode trace PCC too low (b={batch}): {pcc}"
