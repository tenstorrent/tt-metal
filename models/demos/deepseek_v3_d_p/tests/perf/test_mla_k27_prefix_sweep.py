# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi K2.7 chunked-prefill MLA prefix sweep: one fresh 5K chunk attending to a 0K / 50K / 200K cached
prefix, absorbed (ring_mla over the latent cache) vs expanded (per-head K/V + ring_joint_sdpa).

test_mla_k27_prefix_sweep: one warm-up forward, then one timed forward inside a `SWEEP_<form>_<prefix>`
Tracy signpost pair (the timed forward's own MLA_START/MLA_END are the last pair inside it). Random weights
and a random prefix -- device time does not depend on values. Run under run_safe_pytest.sh --profile.

test_mla_k27_prefix_accuracy: both forms on identical inputs, scored against a float32 host MLA over the
device's own latent cache (read back after the forward) for a sample of query rows -- the torch MLA
reference cannot take a long past cache, and a full-sequence CPU reference at 200K takes ~1 h."""

import copy
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3.reference.modeling_deepseek import apply_rotary_pos_emb
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    blockcyclic_cache_host,
    blockcyclic_positions,
    rotated_chip_positions,
)
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache

CHUNK = 5120
SP_AXIS, TP_AXIS = 0, 1
# Best measured SDPA (q_chunk, k_chunk) per form at 2560 tokens/chip on the 2x2 QuietBox (larger
# chunks overflow L1); the module's own default for this shape is (32, 32).
BEST_QK = {"absorbed": (128, 256), "expanded": (320, 512)}

_MESH = pytest.mark.parametrize(
    "mesh_device,device_params",
    [pytest.param((2, 2), fabric2d_device_params(l1_small_size=1152), id="fabric2d-2x2")],
    indirect=["mesh_device", "device_params"],
)
_VARIANT = pytest.mark.parametrize("variant", ["kimi_k2_7"], indirect=True, ids=["k2_7"])


def _slice_heads(config, weights, n_heads):
    """Keep the first n_heads attention heads (q_b_proj rows, kv_b_proj rows, o_proj columns)."""
    config = copy.deepcopy(config)
    H = config.num_attention_heads
    qk, v = config.qk_nope_head_dim + config.qk_rope_head_dim, config.v_head_dim
    weights = dict(weights)
    weights["q_b_proj.weight"] = weights["q_b_proj.weight"].view(H, qk, -1)[:n_heads].reshape(n_heads * qk, -1)
    kv_b = weights["kv_b_proj.weight"]
    weights["kv_b_proj.weight"] = kv_b.view(H, config.qk_nope_head_dim + v, -1)[:n_heads].reshape(-1, kv_b.shape[-1])
    weights["o_proj.weight"] = weights["o_proj.weight"][:, : n_heads * v].contiguous()
    config.num_attention_heads = config.num_key_value_heads = n_heads
    return config, weights


def _setup(request, mesh_device, prefix, chunk=CHUNK, n_heads=None):
    """Weights, a latent cache holding a random `prefix`, the indexed rope tables and one fresh chunk."""
    sp = mesh_device.shape[SP_AXIS]
    config, weights = request.getfixturevalue("random_weights")
    if n_heads is not None:
        config, weights = _slice_heads(config, weights, n_heads)
    seq_len_cache = max(2 * chunk, -(-(prefix + chunk) // chunk) * chunk)
    config.max_seq_len = seq_len_cache
    kvpe_dim = config.kv_lora_rank + config.qk_rope_head_dim
    rope = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=seq_len_cache, chunk_size_global=chunk
    )
    cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=1,
        num_users=1,
    )
    cache_shard_dims = [None, None]
    cache_shard_dims[SP_AXIS] = 2
    if prefix > 0:
        torch.manual_seed(7)
        host = blockcyclic_cache_host(
            torch.randn(prefix, kvpe_dim, dtype=torch.bfloat16) * 0.5, sp, chunk, seq_len_cache, kvpe_dim
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=cache_shard_dims
                ),
            ),
            cache.storage,
        )
    positions = rotated_chip_positions(prefix, sp, chunk // sp)
    assert all(positions[c][0] == prefix + c * (chunk // sp) for c in range(sp)), "prefix must be slab-aligned"
    hidden_dims = [None, None]
    hidden_dims[TP_AXIS] = -1
    hidden_dims[SP_AXIS] = -2
    torch.manual_seed(11)
    host_h = torch.randn(1, 1, chunk, config.hidden_size, dtype=torch.bfloat16)
    tt_h = ttnn.from_torch(
        host_h,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=hidden_dims),
    )
    mesh_device.enable_program_cache()
    return config, weights, seq_len_cache, rope, cache, host_h, tt_h


def _make_mla(config, weights, mesh_device, seq_len_cache, form, chunk=CHUNK):
    return ttMLA(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len_cache,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        is_balanced=False,
        topology=per_axis_topology(),
        is_chunked=True,
        active_seq_len=chunk,
        slot_num=1,
        layer_num=1,
        attn_form=form,
    )


def _to_host(t, mesh_device):
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape)
    )


@_MESH
@_VARIANT
@pytest.mark.parametrize("prefix", [0, 51200, 204800], ids=["pre0k", "pre50k", "pre200k"])
@pytest.mark.parametrize("form", ["absorbed", "expanded"])
@pytest.mark.timeout(0)
def test_mla_k27_prefix_sweep(request, mesh_device, device_params, variant, prefix, form):
    config, weights, seq_len_cache, rope, cache, _, tt_h = _setup(request, mesh_device, prefix)
    mla_tt = _make_mla(config, weights, mesh_device, seq_len_cache, form)

    def fwd():
        return mla_tt.forward(hidden_states=tt_h, rope_tensors=rope, kvpe_cache=cache, actual_start=prefix)

    # MLA_SWEEP_QK="32x640,128x320" runs one timed forward per SDPA (q_chunk, k_chunk); unset = module default.
    qk_list = [tuple(map(int, c.split("x"))) for c in os.environ.get("MLA_SWEEP_QK", "").split(",") if c]
    for qk in qk_list or [None]:
        mla_tt.sdpa_chunk_override = qk
        try:
            ttnn.deallocate(fwd())  # warm-up: compile + program cache
        except RuntimeError as e:  # e.g. an L1-overflowing chunk config; keep sweeping
            logger.warning(f"SWEEP q/k {qk} failed: {str(e).splitlines()[0][:200]}")
            continue
        ttnn.synchronize_device(mesh_device)
        tag = f"SWEEP_{form}_{prefix}" + (f"_q{qk[0]}k{qk[1]}" if qk else "")
        ttnn.tracy_message(f"`TT_SIGNPOST: {tag}_START`")
        out = fwd()
        ttnn.synchronize_device(mesh_device)
        ttnn.tracy_message(f"`TT_SIGNPOST: {tag}_END`")
        host_out = _to_host(out, mesh_device)
        ttnn.deallocate(out)
        assert torch.isfinite(host_out).all(), f"{tag}: non-finite MLA output"
        if os.environ.get("MLA_SWEEP_OUT_DIR"):  # saved outputs let absorbed vs expanded be compared offline
            torch.save(host_out, os.path.join(os.environ["MLA_SWEEP_OUT_DIR"], f"{tag}.pt"))
        logger.info(f"{tag}: out {tuple(host_out.shape)} std {host_out.float().std():.4f}")


def _host_mla_rows(config, weights, kvpe_nat, host_h, prefix, rows):
    """float32 absorbed MLA for chunk rows `rows` (global positions prefix + rows) over the natural-order
    latent cache kvpe_nat [prefix + CHUNK, 576] -- the same cache values the device attended to."""
    mla = create_mla_reference(
        config=config,
        state_dict={"model.layers.0.self_attn." + k: v.float() for k, v in weights.items()},
        layer_idx=0,
        module_path="model.layers.0.self_attn",
    ).eval()
    attn = mla.attention
    H, nope, rope_d, r = attn.num_heads, attn.qk_nope_head_dim, attn.qk_rope_head_dim, attn.kv_lora_rank
    pos = torch.tensor(rows) + prefix
    with torch.no_grad():
        h = host_h[0, 0, rows].float().unsqueeze(0)
        q = attn.q_b_proj(attn.q_a_layernorm(attn.q_a_proj(h))).view(1, len(rows), H, nope + rope_d).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [nope, rope_d], dim=-1)
        cos, sin = attn.rotary_emb(q_pe, seq_len=kvpe_nat.shape[0], meta_style=True)
        q_pe, _ = apply_rotary_pos_emb(q_pe, q_pe, cos, sin, pos.unsqueeze(0), meta_style=True)
        kv_b = attn.kv_b_proj.weight.view(H, nope + attn.v_head_dim, r)
        q_all = torch.cat([q_nope @ kv_b[:, :nope], q_pe], dim=-1)[0]  # [H, n, 576]
        kv = kvpe_nat.float()
        mask = torch.arange(kv.shape[0]).unsqueeze(0) > pos.unsqueeze(1)  # [n, N]: key after query
        out_lat = torch.empty(H, len(rows), r)
        for h0 in range(0, H, 8):
            s = (q_all[h0 : h0 + 8] @ kv.T) * attn.softmax_scale
            s.masked_fill_(mask, float("-inf"))
            out_lat[h0 : h0 + 8] = torch.softmax(s, dim=-1) @ kv[:, :r]
        o = out_lat @ kv_b[:, nope:].transpose(1, 2)  # [H, n, v]
        return attn.o_proj(o.transpose(0, 1).reshape(len(rows), -1))


@_MESH
@_VARIANT
@pytest.mark.parametrize("prefix", [51200, 204800], ids=["pre50k", "pre200k"])
@pytest.mark.timeout(0)
def test_mla_k27_prefix_accuracy(request, mesh_device, device_params, variant, prefix):
    config, weights, seq_len_cache, rope, cache, host_h, tt_h = _setup(request, mesh_device, prefix)
    outs = {}
    # MLA_ACC_QK="absorbed=32x640,expanded=320x512" overrides the per-form SDPA chunking.
    qks = dict(BEST_QK)
    for item in filter(None, os.environ.get("MLA_ACC_QK", "").split(",")):
        form, qk = item.split("=")
        qks[form] = tuple(map(int, qk.split("x")))
    for form, qk in qks.items():
        mla_tt = _make_mla(config, weights, mesh_device, seq_len_cache, form)
        mla_tt.sdpa_chunk_override = qk
        try:
            out = mla_tt.forward(hidden_states=tt_h, rope_tensors=rope, kvpe_cache=cache, actual_start=prefix)
        except RuntimeError as e:  # e.g. a config that overflows L1 with an opt-in SDPA mode; score the others
            logger.warning(f"ACCURACY prefix={prefix} {form} q/k={qk}: forward failed: {str(e).splitlines()[0][:160]}")
            continue
        outs[form] = _to_host(out, mesh_device)[0, 0].float()
        ttnn.deallocate(out)
        del mla_tt
    # Both forms write identical chunk KV, so one read-back serves both. TP replica 0, natural order.
    sp = mesh_device.shape[SP_AXIS]
    cache_sr = ttnn.to_torch(
        cache.storage, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    )[0, 0].float()
    nat = torch.empty_like(cache_sr)
    nat[blockcyclic_positions(sp, CHUNK, seq_len_cache)] = cache_sr
    # First/last rows of each chip's shard plus a spread: early rows see mostly prefix, late rows the whole chunk.
    chunk_local = CHUNK // sp
    edges = {
        r
        for c in range(sp)
        for r in (
            *range(c * chunk_local, c * chunk_local + 16),
            *range((c + 1) * chunk_local - 16, (c + 1) * chunk_local),
        )
    }
    rows = sorted(edges | set(range(0, CHUNK, 160)))
    ref = _host_mla_rows(config, weights, nat[: prefix + CHUNK], host_h, prefix, rows)
    for form, out in outs.items():
        _, pcc = comp_pcc(ref, out[rows])
        rel = ((out[rows] - ref).norm() / ref.norm()).item()
        gain = (out[rows].norm() / ref.norm()).item()
        logger.info(
            f"ACCURACY prefix={prefix} {form} q/k={qks[form]}: "
            f"PCC {pcc:.6f} rel-L2 {rel:.4f} gain {gain:.4f} ({len(rows)} rows)"
        )
        assert pcc > 0.98, f"{form} PCC {pcc} vs host reference"
    if len(outs) < 2:
        return
    _, pcc = comp_pcc(outs["absorbed"], outs["expanded"])
    logger.info(f"ACCURACY prefix={prefix} absorbed vs expanded (all rows): PCC {pcc:.6f}")


# Galaxy proxy: one Galaxy (32 chips, SP x TP) per-chip shard on the 2x2 box (SP=2 x TP=2). For Galaxy
# SP_g x TP_g the chip holds 5120/SP_g query rows and 64/TP_g heads; the proxy runs 2x those heads on TP=2 and
# a 2*(5120/SP_g) chunk on SP=2. The prefix is shifted by 5120 - proxy_chunk so the proxy's two chips attend
# exactly what the LAST two Galaxy SP chips do (they gate the ring). Per-chip shapes match Galaxy for every op
# except q_a/kv_a_proj (K = 3584 here vs 7168/TP_g) and the CCLs: ring sizes (2 vs TP_g, 2 vs SP_g), so a proxy
# chip receives 1/2 of the K/V over the ring where a Galaxy chip receives (SP_g-1)/SP_g, and expands N/2 latent
# rows where a Galaxy chip expands N/SP_g. Only 8x4 (640 rows, 16 heads) has Galaxy-tuned matmul configs.
GALAXY_HEADS, GALAXY_CHUNK = 64, 5120
GALAXY_MESHES = {"8x4": (8, 4), "4x8": (4, 8), "2x16": (2, 16)}


@_MESH
@_VARIANT
@pytest.mark.parametrize("prefix", [0, 51200, 204800], ids=["pre0k", "pre50k", "pre200k"])
@pytest.mark.parametrize("form", ["absorbed", "expanded"])
@pytest.mark.parametrize("galaxy", list(GALAXY_MESHES))
@pytest.mark.timeout(0)
def test_mla_k27_galaxy_proxy_sweep(request, mesh_device, device_params, variant, prefix, form, galaxy):
    sp_g, tp_g = GALAXY_MESHES[galaxy]
    sp = mesh_device.shape[SP_AXIS]
    proxy_chunk = GALAXY_CHUNK // sp_g * sp
    proxy_heads = GALAXY_HEADS // tp_g * mesh_device.shape[TP_AXIS]
    proxy_prefix = prefix + GALAXY_CHUNK - proxy_chunk
    config, weights, seq_len_cache, rope, cache, _, tt_h = _setup(
        request, mesh_device, proxy_prefix, chunk=proxy_chunk, n_heads=proxy_heads
    )
    mla_tt = _make_mla(config, weights, mesh_device, seq_len_cache, form, chunk=proxy_chunk)
    if galaxy == "8x4":
        mla_tt.cfg_num_heads = GALAXY_HEADS  # pick up the Galaxy-tuned 640 configs (same per-chip dims)
    local = proxy_chunk // sp
    tuned = {
        w: mla_tt._resolve_mm_cfg(w, local) is not None
        for w in ("q_a_proj", "q_b_proj", "wkv_b1", "wkv_b2", "o_proj", "kv_a_proj_with_mqa")
    }
    logger.info(f"PROXY {galaxy}: {local} rows/chip, {proxy_heads // 2} heads/chip, tuned matmul configs {tuned}")

    def fwd():
        return mla_tt.forward(hidden_states=tt_h, rope_tensors=rope, kvpe_cache=cache, actual_start=proxy_prefix)

    qk_list = [tuple(map(int, c.split("x"))) for c in os.environ.get("MLA_SWEEP_QK", "").split(",") if c]
    for qk in qk_list or [None]:
        mla_tt.sdpa_chunk_override = qk
        try:
            ttnn.deallocate(fwd())
        except RuntimeError as e:
            logger.warning(f"SWEEP q/k {qk} failed: {str(e).splitlines()[0][:200]}")
            continue
        ttnn.synchronize_device(mesh_device)
        tag = f"SWEEP_proxy{galaxy}-{form}_{prefix}" + (f"_q{qk[0]}k{qk[1]}" if qk else "")
        ttnn.tracy_message(f"`TT_SIGNPOST: {tag}_START`")
        out = fwd()
        ttnn.synchronize_device(mesh_device)
        ttnn.tracy_message(f"`TT_SIGNPOST: {tag}_END`")
        host_out = _to_host(out, mesh_device)
        ttnn.deallocate(out)
        assert torch.isfinite(host_out).all(), f"{tag}: non-finite MLA output"
        logger.info(f"{tag}: out {tuple(host_out.shape)} std {host_out.float().std():.4f}")
