# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""1 M-position cache test (one user): the DSA layer's paged latent cache, pooled-key index and sparse
decode at the advertised 1 048 576 positions on one chip, without prefilling 1 M tokens.

Layer 3 is built with ``max_seq_len = N_max + 64`` (the full 1 M-position allocation: 1 GiB of latent pages
+ 64 MiB of pooled keys next to the layer weights). The context is seeded synthetically: random attention
inputs x (per-feature mean / std matched to the real R4097 attention inputs) are streamed through
``TtDSA.prefill_state`` in 32 K-token chunks (latent -> paged cache, indexer stems -> pooled-key index, no
attention output), while the host computes the HF module's own latent and packed indexer rows for the same x
(``kv_a_proj_with_mqa`` / ``kv_a_layernorm``, ``indexer.wk`` / ``k_norm`` / compress gate). At every curve
point N the device cache rows of the chunk are checked against the host rows, then decode steps at positions
N, N + 1, ... with real attention inputs (the R4097 decode rows) are compared with the HF attention module
(``Glm5NextTextAttention``) given a ``DynamicCache`` whose layer entries are the host latent + packed indexer
rows, in two forms: the genuine HF forward (masked attention over all N keys: ~130 GiB of host K/V at 1 M,
``GLM53_CACHE1M_HF_FULL_MAX`` bounds where it runs) and the restricted form (the same HF ops over the
HF-selected keys only, identical up to summation order; ``test_host_reference_matches_hf`` proves both the
cache population and the restricted form against a genuine HF prefill). Also reported: the TT vs HF
selected-id overlap, the per-stage wall time of the eager sparse decode step (``StageTimer``: index scoring,
top-k, the whole-prefix untilize, sparse_sdpa, rest), the dense flash-MLA bound (``dense_positions``) A/B and
the DRAM view.

    TT_VISIBLE_DEVICES=5 pytest models/demos/glm53_flash/tests/test_cache_1m.py -s --timeout 7000

Knobs: GLM53_CACHE1M_N (curve points, increasing multiples of 128; default 16384,131072,1048576),
GLM53_CACHE1M_CHUNK (seed chunk, default 32768), GLM53_CACHE1M_HF_FULL_MAX (largest N at which the full HF
forward runs, default 1048576; it runs at every decode step for N <= max/8 and at the first step otherwise),
GLM53_CACHE1M_ATTN_ONLY=1 (TtDSA + cache only, no MoE / mHC weights: faster iteration, smaller footprint).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import load_config
from models.demos.glm53_flash.tests.test_dsa import _hf_attention
from models.demos.glm53_flash.tests.utils import attention_input, hf_pool, log_dram, pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import StageTimer, TtDSA, TtLatentCache, decode_position_tensors

LAYER = 3
R4097 = Path("bringup/references/glm53_flash/R4097")
SEED = 1_000_000
CURVE = [int(v) for v in os.environ.get("GLM53_CACHE1M_N", "16384,131072,1048576").split(",")]
CHUNK = int(os.environ.get("GLM53_CACHE1M_CHUNK", "32768"))
HF_FULL_MAX = int(os.environ.get("GLM53_CACHE1M_HF_FULL_MAX", str(1 << 20)))
ATTN_ONLY = os.environ.get("GLM53_CACHE1M_ATTN_ONLY", "0") == "1"
STEPS = 4  # decode steps per curve point: N, N+1 (unbounded dense op), N+2, N+3 (dense op bounded to 2050)
OVERLAP_FLOOR = float(os.environ.get("GLM53_CACHE1M_OVERLAP_FLOOR", "0.8"))

# ------------------------------------------------------------------------------------------ host side


def _attn_keys(name: str) -> bool:
    return name.startswith("self_attn.") or name in (
        "hc_attn_fn",
        "hc_attn_base",
        "hc_attn_scale",
        "input_layernorm.weight",
    )


class HostContext:
    """Host twin of the seeded device context: the HF attention module's latent rows and packed indexer
    rows ``[k | gate | valid]`` for the positions seeded so far, extended chunk by chunk with the module's
    own ops; ``make_cache(N)`` builds the HF ``DynamicCache`` holding the first N positions."""

    def __init__(self, hf, text_config, layer_idx: int, n_max: int):
        self.hf, self.text, self.layer_idx = hf, text_config, layer_idx
        self.L, self.D = hf.kv_lora_rank, hf.indexer.head_dim
        self.latent = torch.empty(n_max, self.L, dtype=torch.bfloat16)
        self.packed = torch.empty(n_max, 2 * self.D + 1, dtype=torch.bfloat16)
        self.n = 0

    @torch.no_grad()
    def extend(self, x: torch.Tensor):
        """x [T, 4096] bf16 -> (latent [T, L], k [T, D], gate [T, D]) as HF computes them, appended."""
        hf, T = self.hf, x.shape[0]
        c = hf.kv_a_layernorm(hf.kv_a_proj_with_mqa(x)[:, : self.L])
        k = hf.indexer.k_norm(hf.indexer.wk(x))
        g = F.linear(x, hf.indexer.index_kpool_compress_gate)
        self.latent[self.n : self.n + T] = c
        self.packed[self.n : self.n + T, : self.D] = k
        self.packed[self.n : self.n + T, self.D : 2 * self.D] = g
        self.packed[self.n : self.n + T, 2 * self.D] = 1.0
        self.n += T
        return c, k, g

    def make_cache(self, N: int):
        from transformers import DynamicCache

        assert N <= self.n, (N, self.n)
        cache = DynamicCache(config=self.text)
        layer = cache.layers[self.layer_idx]
        assert hasattr(layer, "update_indexer"), type(layer)
        cpu = torch.device("cpu")
        layer.dtype, layer.device = torch.bfloat16, cpu
        layer.keys = self.latent[:N].reshape(1, 1, N, self.L)
        layer.values = torch.empty(1, 1, N, 0, dtype=torch.bfloat16)
        layer.is_initialized = True
        layer.indexer_dtype, layer.indexer_device = torch.bfloat16, cpu
        layer.indexer_keys = self.packed[:N].reshape(1, N, 2 * self.D + 1)
        layer.is_indexer_initialized = True
        return cache


def _valid_ids(ids: torch.Tensor) -> set[int]:
    return set(int(i) for i in ids.flatten().tolist() if i >= 0)


@torch.no_grad()
def hf_decode_step(hf, cache, x_row: torch.Tensor, *, full: bool, layer_idx: int = LAYER):
    """One HF attention decode step of ``x_row`` [4096] bf16 on ``cache`` (advanced by one position).
    ``full``: the genuine module forward (masked attention over every cached key). Otherwise the same
    module ops restricted to the HF-selected keys (the masked keys contribute exactly zero, so this
    differs from the full forward only by the summation order of the bf16 matmuls).
    Returns (y [4096] fp32, the selected token ids)."""
    from transformers.models.glm5_next.modeling_glm5_next import eager_attention_forward

    x = x_row.reshape(1, 1, -1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    if full:
        hf.next_skip_topk = True  # the forward then returns its top-k ids
        y, _, ids = hf(x, attention_mask=mask, past_key_values=cache)
        return y.reshape(-1).float(), _valid_ids(ids)
    q_resid = hf.q_a_layernorm(hf.q_a_proj(x))
    q = hf.q_b_proj(q_resid).view(1, 1, -1, hf.qk_head_dim).transpose(1, 2)  # [1,H,1,256]
    ckv = hf.kv_a_proj_with_mqa(x)
    kv_pass, k_rot = torch.split(ckv, [hf.kv_lora_rank, hf.qk_rope_head_dim], dim=-1)
    k_pass = hf.kv_a_layernorm(kv_pass).view(1, 1, 1, hf.kv_lora_rank)
    keys, rots = cache.update(k_pass, k_rot.view(1, 1, 1, hf.qk_rope_head_dim), layer_idx)
    ids = hf.indexer(hidden_states=x, q_resid=q_resid, attention_mask=mask, past_key_values=cache)
    sel = ids[0, 0]
    sel = sel[sel >= 0].long()
    k_states, v_states = hf.expand_kv(keys[:, :, sel], rots[:, :, sel])
    attn, _ = eager_attention_forward(hf, q, k_states, v_states, None, scaling=hf.scaling)
    y = hf.o_proj(attn.reshape(1, 1, -1))
    return y.reshape(-1).float(), set(sel.tolist())


def _x_stats(raw, cfg):
    """Per-feature mean / std of the real layer-3 attention inputs (R4097 prompt) for the synthetic rows."""
    x_real = attention_input(raw, cfg, torch.load(R4097 / f"layer{LAYER}_prefill.pt")["x_in"]).float()
    return x_real.mean(0), x_real.std(0)


def synthetic_rows(mu: torch.Tensor, sigma: torch.Tensor, chunk_index: int, T: int) -> torch.Tensor:
    """Deterministic realistic-statistics attention inputs [T, 4096] bf16 for chunk ``chunk_index``."""
    g = torch.Generator().manual_seed(SEED + chunk_index)
    return (mu + sigma * torch.randn(T, mu.numel(), generator=g)).to(torch.bfloat16)


def _decode_rows(raw, cfg, n: int) -> list[torch.Tensor]:
    """Real attention-input rows [4096] bf16 of the R4097 decode steps (position differs, content is real)."""
    return [
        attention_input(raw, cfg, torch.load(R4097 / f"layer{LAYER}_decode{j}.pt")["x_in"]).reshape(-1)
        for j in range(n)
    ]


def _overlap(a: set[int], b: set[int]) -> float:
    return len(a & b) / max(1, len(a))


# -------------------------------------------------------------------------------- host-only validation


def test_host_reference_matches_hf():
    """The host twin + restricted step reproduce the genuine HF module: a 2100-token prompt (sparse
    selection active) prefilled through the HF attention forward vs the host-populated DynamicCache, then
    one decode step: identical caches, identical full-forward outputs, restricted == full (PCC + ids)."""
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=_attn_keys)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf = _hf_attention(text, {k: v for k, v in raw.items() if k.startswith("self_attn.")})
    from transformers import DynamicCache

    T = 2100
    mu, sigma = _x_stats(raw, cfg)
    x = synthetic_rows(mu, sigma, 0, T)
    cache_hf = DynamicCache(config=text)
    with torch.no_grad():
        hf(x[None], attention_mask=torch.ones(1, T, dtype=torch.bool), past_key_values=cache_hf)
    host = HostContext(hf, text, LAYER, T + 8)
    host.extend(x)
    cache_host = host.make_cache(T)
    lh, lp = cache_hf.layers[LAYER], cache_host.layers[LAYER]
    assert torch.equal(lh.keys, lp.keys), "host latent rows differ from the HF prefill cache"
    assert torch.equal(lh.indexer_keys, lp.indexer_keys), "host packed indexer rows differ from the HF prefill cache"
    row = _decode_rows(raw, cfg, 1)[0]
    y_hf, ids_hf = hf_decode_step(hf, cache_hf, row, full=True)
    y_full, ids_full = hf_decode_step(hf, cache_host, row, full=True)
    y_r, ids_r = hf_decode_step(hf, host.make_cache(T), row, full=False)
    d_full = (y_hf - y_full).abs().max().item()
    p_r = pcc(y_full, y_r)
    logger.info(
        f"host reference at T={T}: HF-prefill vs host-populated cache max|diff| {d_full:.3e} (ids equal {ids_hf == ids_full}); "
        f"restricted vs full pcc {p_r:.7f}, max|diff| {(y_full - y_r).abs().max().item():.3e}, ids equal {ids_r == ids_full}, "
        f"{len(ids_full)} ids"
    )
    assert d_full == 0.0 and ids_hf == ids_full
    assert ids_r == ids_full and p_r > 0.9999


# --------------------------------------------------------------------------------------- device test


def _build(mesh_device, raw, cfg, n_max: int):
    """(attn: TtDSA, cache: TtLatentCache, dec: FunctionalDecoder | None) at max_seq_len n_max + 64."""
    if ATTN_ONLY:
        attn = TtDSA(
            mesh_device,
            cfg,
            {k[len("self_attn.") :]: v for k, v in raw.items() if k.startswith("self_attn.")},
            layer_idx=LAYER,
        )
        cache = TtLatentCache(mesh_device, cfg, max_batch=1, max_seq_len=n_max + 64)
        return attn, cache, None
    from models.demos.glm53_flash.tt.functional_decoder import FunctionalDecoder

    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=1, max_seq_len=n_max + 64
    )
    return dec.layer.attn, dec.cache, dec


def _check_chunk(cache: TtLatentCache, hf, cfg, start: int, c_h, k_h, g_h) -> tuple[float, float]:
    """Device latent rows [start, start+T) and the pooled keys of the chunk vs the host rows."""
    T = c_h.shape[0]
    b0, nb = start // cache.block_size, T // cache.block_size
    blocks = ttnn.slice(cache.cache, [b0, 0, 0, 0], [b0 + nb, 1, cache.block_size, cache.latent_dim])
    lat_dev = ttnn.to_torch(blocks).reshape(T, cache.latent_dim).float()
    ttnn.deallocate(blocks)
    p_lat = pcc(c_h.float(), lat_dev)
    P = cfg.index_kpool
    p0, n_pools = start // P, T // P
    view = cache.index.pools_batched()
    pools_t = ttnn.slice(view, [0, 0, p0, 0], [1, 1, p0 + n_pools, cache.index.D])
    pools_dev = ttnn.to_torch(pools_t).reshape(n_pools, -1).float()
    ttnn.deallocate(pools_t)
    pools_ref = hf_pool(k_h.float(), g_h.float(), hf.indexer.index_kpool_compress_ape.detach().float(), P)
    return p_lat, pcc(pools_ref, pools_dev)


def _aggregate(timer: StageTimer) -> str:
    groups = {
        "index_scoring": ("index_queries", "index_score"),
        "topk": ("score_untilize", "score_expand", "topk", "ids_concat"),
        "kv_untilize": ("q_rm", "kv_slice", "kv_untilize"),
        "sparse_sdpa": ("sparse_sdpa",),
        "dense_mla_decode": ("dense_mla_decode",),
        "rest": ("stems+append+index_maintain", "absorb_q", "sparse_out", "out_proj"),
    }
    parts = [f"{g} {sum(timer.times.get(k, 0.0) for k in ks) * 1e3:.2f}ms" for g, ks in groups.items()]
    return "  ".join(parts) + f"  | total {timer.total * 1e3:.2f}ms"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_cache_curve(mesh_device):
    assert all(n % 128 == 0 for n in CURVE) and CURVE == sorted(CURVE) and CHUNK % 128 == 0, (CURVE, CHUNK)
    n_max = CURVE[-1]
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=_attn_keys if ATTN_ONLY else None)
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    P = cfg.index_kpool
    hf = _hf_attention(text, {k: v for k, v in raw.items() if k.startswith("self_attn.")})
    mu, sigma = _x_stats(raw, cfg)
    rows = _decode_rows(raw, cfg, STEPS + 1)

    log_dram(mesh_device, "before build")
    t0 = time.perf_counter()
    attn, cache, dec = _build(mesh_device, raw, cfg, n_max)
    kind = "TtDSA + cache (attention only)" if ATTN_ONLY else "FunctionalDecoder (full layer)"
    alloc, free = log_dram(
        mesh_device, f"layer {LAYER} {kind} built, max_seq_len={n_max + 64} ({time.perf_counter() - t0:.0f} s)"
    )
    gib = 2**30
    lat_bytes = cache.max_blocks * cache.block_size * cache.latent_dim * 2
    pool_bytes = cache.max_batch * cache.index.pool_blocks * 32 * cache.index.D * 2
    logger.info(
        f"cache allocation: latent {cache.max_blocks} blocks x 32 x {cache.latent_dim} bf16 = {lat_bytes / gib:.3f} GiB, "
        f"pooled keys {cache.index.pools_per_user} pools x {cache.index.D} bf16 = {pool_bytes / 2**20:.1f} MiB "
        f"(x 11 DSA layers = {11 * (lat_bytes + pool_bytes) / gib:.2f} GiB of the whole model)"
    )
    host = HostContext(hf, text, LAYER, n_max + 8)
    timer = StageTimer(mesh_device)
    attn.set_stage_timer(timer)

    results = []
    filled, chunk_index = 0, 0
    for N in CURVE:
        # ---- seed [filled, N) synthetically on both sides, chunk by chunk
        while filled < N:
            T = min(CHUNK, N - filled)
            x = synthetic_rows(mu, sigma, chunk_index, T)
            c_h, k_h, g_h = host.extend(x)
            x_tt = ttnn.from_torch(
                x.reshape(1, 1, T, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
            )
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            attn.prefill_state(x_tt, cache, 0, T, start=filled)
            ttnn.synchronize_device(mesh_device)
            dt = time.perf_counter() - t0
            ttnn.deallocate(x_tt)
            p_lat, p_pool = _check_chunk(cache, hf, cfg, filled, c_h, k_h, g_h)
            logger.info(
                f"seed chunk {chunk_index} [{filled}, {filled + T}): state prefill {dt * 1e3:.1f} ms ({T / dt:.0f} tok/s); "
                f"device vs host latent pcc {p_lat:.6f}, pooled keys pcc {p_pool:.6f}"
            )
            assert p_lat > 0.995 and p_pool > 0.995, (chunk_index, p_lat, p_pool)
            filled += T
            chunk_index += 1
        alloc, free = log_dram(mesh_device, f"N={N} seeded")

        # ---- decode steps at N, N+1 (dense op unbounded), N+2, N+3 (dense op bounded to 2050) vs HF
        cache_r = host.make_cache(N)
        full_steps = [s for s in range(STEPS) if N <= HF_FULL_MAX and (s == 0 or N <= HF_FULL_MAX // 8)]
        cache_f = host.make_cache(N) if full_steps else None
        point = {"N": N, "steps": []}
        for s in range(STEPS):
            a = N + s
            row = rows[s]
            x_tt = ttnn.from_torch(
                row.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
            )
            pos, pool_ids, ring_slots = decode_position_tensors(mesh_device, [a], P)
            bounded = s >= STEPS // 2
            dense_pos = (
                decode_position_tensors(mesh_device, [min(a, attn.dense_exact_max_pos)], P)[0] if bounded else None
            )
            timer.reset()
            t0 = time.perf_counter()
            y_tt = attn.decode_forward(
                x_tt, cache, pos, pool_ids, ring_slots, host_positions=[a], dense_positions=dense_pos
            )
            ttnn.synchronize_device(mesh_device)
            wall = time.perf_counter() - t0
            y_t = ttnn.to_torch(y_tt).reshape(-1).float()
            ttnn.deallocate(y_tt)
            tt_ids = attn.selected_ids(x_tt, cache, 0, a)
            first_sentinel = next((i for i, v in enumerate(tt_ids) if v == 0xFFFFFFFF), len(tt_ids))
            assert all(v == 0xFFFFFFFF for v in tt_ids[first_sentinel:]), "sentinel inside the valid prefix"
            tt_set = set(tt_ids[:first_sentinel])
            assert all(0 <= v <= a for v in tt_set), "selected id out of range"
            t0 = time.perf_counter()
            y_r, ids_r = hf_decode_step(hf, cache_r, row, full=False)
            t_r = time.perf_counter() - t0
            p_r, ov_r = pcc(y_r, y_t), _overlap(ids_r, tt_set)
            rec = {
                "pos": a,
                "bounded": bounded,
                "wall_ms": wall * 1e3,
                "pcc_restricted": p_r,
                "overlap": ov_r,
                "stages": dict(timer.times),
            }
            msg = (
                f"N={N} step {s} pos {a} ({'bounded' if bounded else 'unbounded'} dense op): TT vs HF-restricted pcc {p_r:.6f}, "
                f"ids HF {len(ids_r)} TT {len(tt_set)} overlap {ov_r:.4f}; eager wall {wall * 1e3:.1f} ms (host ref {t_r:.1f} s)"
            )
            if s in full_steps:
                t0 = time.perf_counter()
                y_f, ids_f = hf_decode_step(hf, cache_f, row, full=True)
                t_f = time.perf_counter() - t0
                p_f = pcc(y_f, y_t)
                rec.update(pcc_full=p_f, full_vs_restricted=pcc(y_f, y_r), ids_full_eq=ids_f == ids_r)
                msg += (
                    f"; TT vs HF-full pcc {p_f:.6f} (full vs restricted {rec['full_vs_restricted']:.7f}, "
                    f"max|diff| {(y_f - y_r).abs().max().item():.2e}, ids equal {rec['ids_full_eq']}, {t_f:.0f} s)"
                )
            elif cache_f is not None:
                # keep the full cache in step with the restricted one for the next full step
                hf_decode_step(hf, cache_f, row, full=False)
            logger.info(msg)
            logger.info(f"    stages: {timer.report()}")
            logger.info(f"    aggregated: {_aggregate(timer)}")
            point["steps"].append(rec)
        results.append(point)
        peak = max(timer.dram.values()) / gib if timer.dram else float("nan")
        logger.info(f"N={N}: DRAM at the last step's stage boundaries peaks at {peak:.2f} GiB allocated")
        if dec is not None:
            # the whole layer (mHC + attention + MoE) for one eager sparse decode step: compile + warm
            from models.demos.glm53_flash.tt.functional_decoder import host_streams_to_device

            d = torch.load(R4097 / f"layer{LAYER}_decode{STEPS}.pt")
            streams = host_streams_to_device(d["x_in"], mesh_device, pad_to=1)
            attn.set_stage_timer(None)
            walls = {}
            for s, variant in enumerate(("unbounded (compile)", "unbounded", "bounded")):
                dp = dec.make_decode_positions([N + STEPS + s])
                if not variant.startswith("bounded"):
                    dp.dense_positions = None  # the dense flash-MLA reads the whole prefix
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                y = dec.decode_forward(streams, positions=dp)
                ttnn.synchronize_device(mesh_device)
                walls[variant] = time.perf_counter() - t0
                ttnn.deallocate(y)
            attn.set_stage_timer(timer)
            ttnn.deallocate(streams)
            logger.info(
                f"N={N}: full layer {LAYER} eager sparse decode step (mHC + attention + MoE): "
                + ", ".join(f"{k} {v * 1e3:.1f} ms" for k, v in walls.items())
            )
            point["layer_step_ms"] = walls

    # ---- summary + gates
    logger.info("summary (one user, layer 3, eager sparse decode):")
    for point in results:
        N = point["N"]
        for rec in point["steps"]:
            full = f" full {rec['pcc_full']:.6f}" if "pcc_full" in rec else ""
            st = rec["stages"]
            logger.info(
                f"  N={N} pos {rec['pos']} {'bounded' if rec['bounded'] else 'unbounded':9s} pcc restricted {rec['pcc_restricted']:.6f}{full} "
                f"overlap {rec['overlap']:.4f} wall {rec['wall_ms']:.1f} ms: index_score {st.get('index_score', 0) * 1e3:.2f} "
                f"score_untilize {st.get('score_untilize', 0) * 1e3:.2f} score_expand {st.get('score_expand', 0) * 1e3:.2f} "
                f"topk {st.get('topk', 0) * 1e3:.2f} kv_slice {st.get('kv_slice', 0) * 1e3:.2f} "
                f"kv_untilize {st.get('kv_untilize', 0) * 1e3:.2f} sparse_sdpa {st.get('sparse_sdpa', 0) * 1e3:.2f} "
                f"dense_mla {st.get('dense_mla_decode', 0) * 1e3:.2f} ms"
            )
    for point in results:
        for rec in point["steps"]:
            assert rec["pcc_restricted"] > 0.995, (point["N"], rec)
            if "pcc_full" in rec:
                assert rec["pcc_full"] > 0.995 and rec["ids_full_eq"], (point["N"], rec)
            # the synthetic Gaussian context has no structure: its top-512 pool boundary is dense with near-ties
            # that the bf16 device scoring orders differently from the fp32 HF ranking (92-94 % at 16 K vs the
            # 98-99 % of a real 4 K prompt in test_dsa_long); the attention PCC above is the semantic gate
            assert rec["overlap"] > OVERLAP_FLOOR, (point["N"], rec)
