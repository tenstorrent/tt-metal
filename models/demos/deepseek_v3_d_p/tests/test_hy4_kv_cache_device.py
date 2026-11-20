# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4 context-parallel KV cache on device (#65886), read back de-striped with ``tt/hy4/kv_cache.py`` only.

``test_hy4_kv_stripe_golden_round_trip`` (op level, both mappings of the 32-chip stripe on the physical 8x4
Galaxy mesh: TP32's SP 8 on axis 0 x TP 4 on axis 1, which is also the dev layout's, and SP 4 on axis 1 x TP 8 on
axis 0, which the stripe module still supports):

  * writes layer 0 of the HF golden (latent, rope, index keys, in the device frame) into the real Hy4 cache pair
    (scaled-FP8 KVPE + BFP8 index keys, allocated as ``GLM53Adapter.allocate_kv_cache`` does) with
    ``update_padded_kv_cache``, the prefill write op, three ways: one shot, chunked with a mid-block resume, and a
    prefix-cache hit followed by resumes at a block boundary and mid-period;
  * reads each slot back per chip by physical coordinate and de-stripes it with ``KvStripe``: PCC >= 0.99 against
    the golden (latent nope, rope, index keys), every chip's unwritten rows zero, and the three histories
    byte-identical (the layout is a function of the position only);
  * measures the allocated bytes per token-row of each cache and the free DRAM per chip after mesh open.

``test_hy4_prefill_layer0_cache_destriped`` (dev layout, the prefill path): runs layer 0 of the 4K golden through
``TtPrefillTransformer`` (the chunked driver the device smoke uses), captures the caches it wrote and scores them
after de-striping with ``KvStripe`` instead of the driver's own un-rotation. The dev layout runs TP32's mesh and axes,
so the stripe it writes is TP32's.

Env: ``PREFILL_TRACE_DIR`` (the 4K golden), and for the prefill test the smoke's ``HY4_HF_MODEL`` /
``PREFILL_TTNN_CACHE`` and ``PREFILL_CHUNK_SIZE=4096``. ``HY4_KV_REPORT``: optional JSON path for the measurements.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.hy4.kv_cache import (
    INDEX_K_BFP8_ROW_BYTES,
    KVPE_SCALED_FP8_DRAM_ROW_BYTES,
    KvBudget,
    KvStripe,
    device_topk_merge,
    read_chip_slabs,
    read_slot,
    stripe_key_axis,
)
from models.demos.deepseek_v3_d_p.tt.mla.indexer import normalized_hadamard_matrix
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache

GOLDEN = Path(os.environ.get("PREFILL_TRACE_DIR", "/mnt/tt-data/dwadhwani/hy4-golden/hy4-bf16-longbench-4k-ropeint"))
S = 4096
PCC_FLOOR = 0.99
KV_LORA, ROPE, INDEX_DIM = 512, 64, 128
# tp32 (and dev): sequence over SP 8 on axis 0, hidden over TP 4 on axis 1. sp4-axis1: SP on the 4 columns.
STRIPES = {"tp32-sp8tp4": KvStripe(sp=8, tp=4, sp_axis=0), "sp4-axis1": KvStripe(sp=4, tp=8, sp_axis=1)}
# Per user slot, the chunk starts its prefill is written with (tokens [0, S) in every case).
HISTORIES = {
    "one_shot": [0],
    "resume_mid_block": [0, 1312],  # a chunk ends at 1312 (mid-block), the next resumes there
    "prefix_hit": [0, 256, 2048 + 640],  # the first 256 tokens come from a prefix cache hit
}

# #65884's per-chip weight estimates (TP32 cannot load its weights yet) and its activation/trace/CCL reserve.
WEIGHTS_PER_CHIP_GB = {"experts_bfp4_rest_bfp8": 14.1, "experts_bfp8_rest_bfp8": 25.9, "experts_bfp8_attn_bf16": 26.7}
RESERVE_PER_CHIP_GB = 2.5

pytestmark = pytest.mark.skipif(
    not (GOLDEN / "kv_cache" / "layer_0").is_dir(), reason=f"no Hy4 golden at {GOLDEN} (PREFILL_TRACE_DIR)"
)


def _rows(group: str, key: str, n: int = S) -> torch.Tensor:
    shards = sorted((GOLDEN / group).glob("rows_*.safetensors"), key=lambda p: int(p.stem.split("_")[1]))
    parts = []
    for shard in shards:
        with safe_open(str(shard), "pt") as f:
            parts.append(f.get_tensor(key))
    return torch.cat(parts)[:n].float()


def _golden_layer0():
    """Layer 0 in the device frames: KVPE ``[latent 512 | rope 64 interleaved]``, index keys ``[rope | nope]``. The
    golden is already in them (RoPE paired interleaved, the checkpoint's pairing); a trace written before that fix
    (``pe_rope: half_split``) has its pe re-based."""
    from models.demos.deepseek_v3_d_p.reference.hy4.golden import (
        golden_pe_rope,
        half_split_to_interleaved_perm,
        read_metadata,
    )

    kv = _rows("kv_cache/layer_0", "kv_post_transform_layer_0")
    if golden_pe_rope(read_metadata(GOLDEN)) == "half_split":
        kv = torch.cat([kv[:, :KV_LORA], kv[:, KV_LORA:][:, half_split_to_interleaved_perm(ROPE)]], dim=-1)
    ik = _rows("dsa/indexer_k_layer_0", "indexer_k_layer_0")
    return kv, ik


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def _dram_view(mesh_device):
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return {
        "banks": v.num_banks,
        "total": v.total_bytes_per_bank * v.num_banks,
        "allocated": v.total_bytes_allocated_per_bank * v.num_banks,
        "free": v.total_bytes_free_per_bank * v.num_banks,
        "largest_free_per_bank": v.largest_contiguous_bytes_free_per_bank,
    }


def _chunk_input(rows: torch.Tensor, start: int, valid_end: int, stripe: KvStripe) -> torch.Tensor:
    """The producer's chunk ``[start, start + period)``, rotated for a mid-slab start; pad rows (>= valid_end) zero."""
    c = stripe.period
    pos = torch.tensor(rotate_chunk_tokens(list(range(start, start + c)), start, stripe.sp))
    out = torch.zeros(c, rows.shape[-1], dtype=rows.dtype)
    real = pos < valid_end
    out[real] = rows[pos[real]]
    return out.reshape(1, 1, c, -1)


def _write_history(kvpe_cache, index_cache, kv, ik, starts, slot, stripe, mesh_device, num_layers):
    mapper_dims = [None, None]
    mapper_dims[stripe.sp_axis] = 2  # the chunk split over SP, replicated over TP, as the KV stem emits it
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=mapper_dims)

    def dev(t, dtype):
        return ttnn.from_torch(
            t.contiguous(), device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
        )

    bounds = list(starts) + [S]
    for a, z in zip(bounds[:-1], bounds[1:]):
        while a < z:
            valid = min(z, a + stripe.period)
            common = dict(
                slot_idx=slot,
                layer_idx=0,
                num_layers=num_layers,
                kv_actual_global=a,
                cluster_axis=stripe.sp_axis,
                valid_global=valid,
                tp_axis=stripe.tp_axis,
            )
            chunk = _chunk_input(kv, a, valid, stripe)
            latent, rope = dev(chunk[..., :KV_LORA], ttnn.bfloat16), dev(chunk[..., KV_LORA:], ttnn.bfloat16)
            latent_fp8, scales, rope_t = kvpe_cache.prepare_scaled_fp8_inputs(latent, rope, keep_rope_tiled=True)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                kvpe_cache.storage, latent_fp8, rope=rope_t, scales=scales, **common
            )
            for t in {id(x): x for x in (latent, rope, latent_fp8, scales, rope_t)}.values():
                ttnn.deallocate(t)
            idx = dev(_chunk_input(ik, a, valid, stripe), ttnn.bfloat8_b)
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(index_cache, idx, **common)
            ttnn.deallocate(idx)
            a = valid


def _measure_row_bytes(mesh_device, stripe, alloc) -> float:
    """Allocated DRAM bytes per token-row of one cache (one user): the allocator delta between two sizes over the
    row difference, so the allocation's fixed overhead cancels."""
    deltas = []
    for periods in (32, 64):
        before = _dram_view(mesh_device)["allocated"]
        cache = alloc(periods * stripe.period, 4)
        deltas.append((_dram_view(mesh_device)["allocated"] - before, 4 * periods * stripe.block))
        ttnn.deallocate(cache)
    (b0, r0), (b1, r1) = deltas
    return (b1 - b0) / (r1 - r0)


def _capacity(free_after_open: int, budget: KvBudget) -> dict:
    """KV room per precision policy: measured free DRAM per chip after mesh open, minus #65884's weight estimate
    and reserve, over the measured bytes per token (each chip holds 1/32 of every token)."""
    out = {}
    for policy, weights_gb in WEIGHTS_PER_CHIP_GB.items():
        free_kv = free_after_open - int((weights_gb + RESERVE_PER_CHIP_GB) * 1e9)
        out[policy] = {
            "free_kv_per_chip_gb": free_kv / 1e9,
            "free_kv_galaxy_gb": free_kv * budget.num_chips / 1e9,
            "tokens": budget.tokens(free_kv),
            "users_1M": budget.users(free_kv, 1 << 20),
            "users_256K": budget.users(free_kv, 256 << 10),
            "users_128K": budget.users(free_kv, 128 << 10),
        }
    return out


@pytest.mark.parametrize(
    "mesh_device, device_params, stripe_id",
    [
        pytest.param((8, 4), torus_xy_device_params(), sid, id=sid)
        for sid in STRIPES  # both stripes on the physical 8x4 mesh: no logical 4x8 view (it hung, #65885)
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_hy4_kv_stripe_golden_round_trip(mesh_device, device_params, stripe_id):
    stripe = STRIPES[stripe_id]
    assert tuple(mesh_device.shape) == stripe.mesh_shape
    dram_open = _dram_view(mesh_device)
    kv, ik = _golden_layer0()
    hf = type("Cfg", (), {"kv_lora_rank": KV_LORA, "qk_rope_head_dim": ROPE})
    users, num_layers, seq = len(HISTORIES), 1, 2 * stripe.period
    common = dict(
        mesh_device=mesh_device,
        mesh_shape=list(mesh_device.shape),
        sp_axis=stripe.sp_axis,
        num_users=users,
        tp_axis=stripe.tp_axis,
        bank_aligned=True,
    )
    # The pair GLM53Adapter.allocate_kv_cache allocates for Hy4, at test size.
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.SCALED_FP8, hf_config=hf, seq_len=seq, num_kvpe_cache_layers=num_layers, **common
    )
    index_cache = init_kvpe_cache(
        kvpe_cache_head_dim=INDEX_DIM, seq_len=seq, num_kvpe_cache_layers=num_layers, dtype=ttnn.bfloat8_b, **common
    )
    for slot, starts in enumerate(HISTORIES.values()):
        _write_history(kvpe_cache, index_cache, kv, ik, starts, slot, stripe, mesh_device, num_layers)
    ttnn.synchronize_device(mesh_device)

    results = {}
    raw = {}
    for slot, name in enumerate(HISTORIES):
        phys = read_slot(kvpe_cache.storage, mesh_device, stripe, slot)
        raw[name] = phys[:S].contiguous().view(torch.uint8)
        got = kvpe_cache.unpack_host(phys).float()
        got_ik = read_slot(index_cache, mesh_device, stripe, slot).float()
        results[name] = {
            "kvpe_nope": pcc(got[:S, :KV_LORA], kv[:, :KV_LORA]),
            "kvpe_pe": pcc(got[:S, KV_LORA:], kv[:, KV_LORA:]),
            "index_k_rope": pcc(got_ik[:S, : INDEX_DIM // 2], ik[:, : INDEX_DIM // 2]),
            "index_k_nope": pcc(got_ik[:S, INDEX_DIM // 2 :], ik[:, INDEX_DIM // 2 :]),
            # nothing written past the prompt
            "tail_zero": bool(torch.count_nonzero(got[S:]) == 0 and torch.count_nonzero(got_ik[S:]) == 0),
        }
    # each chip holds exactly rows_per_chip(S) = 128 tokens of the prompt, at the positions the stripe names
    slabs = read_chip_slabs(index_cache, mesh_device, stripe)[:, 0].float()  # [chips, rows, 128]
    held = stripe.populated_rows(0, S)
    for chip in (0, 5, 17, 31):
        pos = stripe.local_positions(chip, held)
        assert pcc(slabs[chip, :held], ik[pos]) > 0.999, f"chip {chip} does not hold positions {pos[:3].tolist()}..."

    one_user = {**common, "num_users": 1}
    kvpe_row = _measure_row_bytes(
        mesh_device,
        stripe,
        lambda s, n: init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.SCALED_FP8, hf_config=hf, seq_len=s, num_kvpe_cache_layers=n, **one_user
        ).storage,
    )
    index_row = _measure_row_bytes(
        mesh_device,
        stripe,
        lambda s, n: init_kvpe_cache(
            kvpe_cache_head_dim=INDEX_DIM, seq_len=s, num_kvpe_cache_layers=n, dtype=ttnn.bfloat8_b, **one_user
        ),
    )
    measured = KvBudget(kvpe_row_bytes=round(kvpe_row), index_row_bytes=round(index_row))
    report = {
        "stripe": stripe.descriptor(),
        "pcc": results,
        "kvpe_row_bytes_measured": kvpe_row,
        "index_row_bytes_measured": index_row,
        "bytes_per_token": measured.bytes_per_token,
        "bytes_per_token_mtp": KvBudget(
            num_layers=79, num_index_layers=22, kvpe_row_bytes=round(kvpe_row), index_row_bytes=round(index_row)
        ).bytes_per_token,
        "dram_per_chip_after_open": dram_open,
        "capacity": _capacity(dram_open["free"], measured),
    }
    logger.info(f"Hy4 KV stripe {stripe_id}: {json.dumps(report, indent=1)}")
    if os.environ.get("HY4_KV_REPORT"):
        out = Path(os.environ["HY4_KV_REPORT"])
        prior = json.loads(out.read_text()) if out.is_file() else {}
        prior[stripe_id] = report
        out.write_text(json.dumps(prior, indent=1))

    assert (kvpe_row, index_row) == (KVPE_SCALED_FP8_DRAM_ROW_BYTES, INDEX_K_BFP8_ROW_BYTES), (kvpe_row, index_row)
    for name, r in results.items():
        assert r["tail_zero"], f"{name}: rows past the prompt were written"
        for key in ("kvpe_nope", "kvpe_pe", "index_k_rope", "index_k_nope"):
            assert r[key] >= PCC_FLOOR, f"{name} {key} PCC {r[key]:.6f} < {PCC_FLOOR}"
    for name in HISTORIES:
        assert torch.equal(raw[name], raw["one_shot"]), f"{name}: KVPE bytes differ from the one-shot prefill"


# ----------------------------------------------------------------------------------------------------------------
# Distributed top-k on golden scores
# ----------------------------------------------------------------------------------------------------------------
# Layer-0 indexer scores [S, S] (fp32, causal -inf) from the fp32 reference with the golden's index keys and the
# reference's queries / head weights. Produced offline (the reference lives on the golden-refs branch, #65936).
GOLDEN_SCORES = Path(
    os.environ.get("HY4_LAYER0_INDEX_SCORES", "/mnt/tt-data/dwadhwani/hy4-test/kvcp/layer0_index_scores.pt")
)


@pytest.mark.skipif(not GOLDEN_SCORES.is_file(), reason=f"no layer-0 golden index scores at {GOLDEN_SCORES}")
@pytest.mark.parametrize(
    "mesh_device, device_params, stripe_id",
    [pytest.param((8, 4), torus_xy_device_params(), sid, id=sid) for sid in STRIPES],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("k", [64, 2048])
@pytest.mark.timeout(0)
def test_hy4_local_topk_merge_on_golden_scores(mesh_device, device_params, stripe_id, k):
    """Each chip scores only its own stripe of keys, keeps a local top-k, the 32 lists are gathered and merged: the
    result must be the global top-k of the same (bf16) scores. k = 64 makes the local selection real (64 of each
    chip's 128 keys at 4K); k = 2048 is Hy4's index_topk."""
    stripe = STRIPES[stripe_id]
    scores = torch.load(GOLDEN_SCORES, weights_only=True)[:S, :S].to(torch.bfloat16)
    per_chip = stripe_key_axis(scores, stripe)  # [8, 4, Q, 128]
    rows = per_chip.shape[-1]
    labels = torch.stack([stripe.local_positions(c, rows) for c in range(stripe.num_chips)])  # [chips, rows]
    lab = torch.empty(*stripe.mesh_shape, S, rows, dtype=torch.int32)
    for chip in range(stripe.num_chips):
        r, c = stripe.coord(chip)
        lab[r, c] = labels[chip].to(torch.int32).expand(S, rows)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, 1))
    tt_scores = ttnn.from_torch(
        per_chip, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
    )
    tt_labels = ttnn.from_torch(lab, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
    k_local = min(k, rows)
    mv, mp = device_topk_merge(tt_scores, tt_labels, k, k_local, stripe, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)
    # every chip holds the merged result; read two of them
    dev_v = [ttnn.to_torch(t).float().reshape(S, k) for t in ttnn.get_device_tensors(mv)]
    dev_p = [ttnn.to_torch(t).long().reshape(S, k) for t in ttnn.get_device_tensors(mp)]
    assert all(torch.equal(dev_v[0], x) for x in dev_v[1:]) and all(torch.equal(dev_p[0], x) for x in dev_p[1:])
    dv, dp = dev_v[0], dev_p[0]

    gv, gp = scores.float().topk(k, dim=-1)
    finite = ~torch.isneginf(gv)
    # 1) the merged values are the global top-k values, row by row
    assert torch.equal(dv[finite], gv[finite]), "merged values differ from the global top-k"
    assert torch.all(torch.isneginf(dv[~finite]))
    # 2) every merged position is a real key with that score
    assert torch.equal(scores.float().gather(1, dp.clamp(0, S - 1))[finite], dv[finite])
    # 3) the selected sets match wherever the k-th score is not tied with the next one (bf16 ties are common)
    nxt = scores.float().topk(min(k + 1, S), dim=-1).values[:, -1]
    untied = torch.isneginf(nxt) | (gv[:, -1] > nxt)
    same = torch.tensor(
        [set(dp[i][finite[i]].tolist()) == set(gp[i][finite[i]].tolist()) for i in range(S)], dtype=torch.bool
    )
    logger.info(
        f"{stripe_id} k={k}: {int(untied.sum())} of {S} rows have no tie at the k-th score; "
        f"sets equal on {int(same[untied].sum())} of them, on {int(same.sum())} rows overall"
    )
    assert torch.all(same[untied]), "merged set differs from the global top-k on a row without a k-th-place tie"
    if os.environ.get("HY4_KV_REPORT"):
        out = Path(os.environ["HY4_KV_REPORT"])
        prior = json.loads(out.read_text()) if out.is_file() else {}
        prior[f"topk_merge_{stripe_id}_k{k}"] = {
            "rows": S,
            "untied_rows": int(untied.sum()),
            "sets_equal_untied": int(same[untied].sum()),
            "sets_equal_all": int(same.sum()),
        }
        out.write_text(json.dumps(prior, indent=1))


# ----------------------------------------------------------------------------------------------------------------
# The prefill path (dev layout)
# ----------------------------------------------------------------------------------------------------------------
def _smoke():
    import models.demos.deepseek_v3_d_p.tests.test_hy4_prefill_device_smoke as smoke

    return smoke


try:
    _SMOKE = _smoke()
except Exception as e:  # noqa: BLE001 -- the smoke's imports need the full model stack
    logger.warning(f"Hy4 prefill smoke module unavailable: {e}")
    _SMOKE = None


@pytest.mark.skipif(_SMOKE is None, reason="Hy4 prefill smoke module unavailable")
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, sp_axis",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=_SMOKE.chunked.moe_fabric_payload(_SMOKE.GLM53Config),
                l1_small_size=_SMOKE.chunked.GLM_L1_SMALL_SIZE,
                trace_region_size=_SMOKE.chunked.GLM_TRACE_REGION_SIZE,
            )
            if _SMOKE is not None
            else {},
            2,
            0,
            id="torus-xy-8x4",
            marks=list(_SMOKE.pytestmark) if _SMOKE is not None else [],
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["hy4_preview"], indirect=True, ids=["hy4"])
@pytest.mark.timeout(0)
def test_hy4_prefill_layer0_cache_destriped(
    variant, config_only, mesh_device, device_params, num_links, sp_axis, monkeypatch
):
    """Layer 0 of the 4K golden through ``TtPrefillTransformer`` (dev layout, one 4096-token chunk), its caches
    de-striped with ``KvStripe`` (not the driver's un-rotation) and scored against the golden."""
    from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config
    from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology

    smoke, chunked = _SMOKE, _SMOKE.chunked
    monkeypatch.setenv(hy4_layout.LAYOUT_ENV, hy4_layout.DEV_LAYOUT)
    hy4_layout.resolve_layout(8, 4, 0, Hy4Config.NUM_ATTENTION_HEADS)
    cache = smoke._mesh_cache_dir(mesh_device.get_num_devices())
    if not cache.is_dir() or not hc.mla_cache_stamp(cache, 0).exists():
        pytest.skip(f"no stamped Hy4 weight cache for layer 0 at {cache}")
    stripe = KvStripe.from_mesh(tuple(mesh_device.shape), sp_axis, smoke.CHUNK_SIZE)
    assert stripe == STRIPES["tp32-sp8tp4"]

    captured = {}
    monkeypatch.setattr(chunked, "_record_kv_cache_pcc", lambda trace, layout, c, *a, **kw: captured.update(kvpe=c))
    monkeypatch.setattr(
        chunked, "_record_indexer_k_cache_pcc", lambda trace, layout, c, *a, **kw: captured.update(index=c)
    )
    chunked.run_chunked_transformer_updated(
        variant,
        config_only,
        mesh_device,
        cache.parent,
        1,  # layer 0
        1,  # one 4096-token chunk
        GateComputeMode.DEVICE_FP32,
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        1,
        routing_use_l1_small_for_semaphores=True,
        preload_isl=0,
        check_pcc=True,
        check_layer_pcc=False,
        use_trace=False,
        seq_cache=2 * smoke.CHUNK_SIZE,
        kv_cache_format=chunked.GLM_KV_CACHE_FORMAT,
        sp_axis=sp_axis,
        chunk_size=smoke.CHUNK_SIZE,
    )
    assert "kvpe" in captured and "index" in captured, f"the driver did not hand over both caches: {list(captured)}"

    kv, ik = _golden_layer0()
    got = captured["kvpe"].unpack_host(read_slot(captured["kvpe"].storage, mesh_device, stripe, 0, S)).float()
    # The device keeps GLM's Hadamard-rotated index keys (H is symmetric and orthonormal; q.k is unchanged).
    got_ik = (
        read_slot(captured["index"], mesh_device, stripe, 0, S).float() @ normalized_hadamard_matrix(INDEX_DIM).float()
    )
    r = {
        "kvpe_nope": pcc(got[:, :KV_LORA], kv[:, :KV_LORA]),
        "kvpe_pe": pcc(got[:, KV_LORA:], kv[:, KV_LORA:]),
        "index_k_rope": pcc(got_ik[:, : INDEX_DIM // 2], ik[:, : INDEX_DIM // 2]),
        "index_k_nope": pcc(got_ik[:, INDEX_DIM // 2 :], ik[:, INDEX_DIM // 2 :]),
    }
    logger.info(f"Hy4 prefill layer 0, de-striped with KvStripe: {r}")
    if os.environ.get("HY4_KV_REPORT"):
        out = Path(os.environ["HY4_KV_REPORT"])
        prior = json.loads(out.read_text()) if out.is_file() else {}
        prior["prefill_layer0_dev"] = r
        out.write_text(json.dumps(prior, indent=1))
    for key, v in r.items():
        assert v >= PCC_FLOOR, f"layer 0 {key} PCC {v:.6f} < {PCC_FLOOR}"
