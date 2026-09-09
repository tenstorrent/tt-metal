# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DFlash drafter context-KV PCC gate against the golden GPU trace.
Device cache is bfloat8_b, while the golden is fp32.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.rope import interleaved_to_halfsplit_perm
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions

# Its own env var rather than riding PREFILL_TRACE_DIR: the drafter golden is a separate artifact with its
# own layout (whole-tensor safetensors, not row-sharded), and a CI leg may point PREFILL_TRACE_DIR at a
# DeepSeek trace that has no drafter golden beside it at all.
GOLDEN_KV_ENV = "PREFILL_DFLASH_GOLDEN_KV_DIR"
GOLDEN_KV_DEFAULT = "/mnt/models/deepseek-prefill-cache/golden/dflash_context_kv_55k_v3"
PCC_ENV = "PREFILL_DFLASH_PCC"
DEFAULT_PCC = 0.88


def read_slot_dflash_kv(
    mesh_device, cache, *, sp: int, chunk_size_global: int, num_layers: int, slot_id: int, out_len: int
) -> torch.Tensor:
    """Read one slot's drafter K or V cache off the device and return it as
    ``[num_layers, num_kv_heads, out_len, head_dim]`` fp32 in natural token order,
    undoing the user-major slot layout and the block-cyclic sharding it was written in."""
    s = list(cache.shape)
    assert slot_id * num_layers + num_layers <= s[0], (
        f"slot {slot_id} x {num_layers} layers exceeds drafter cache dim0 {s[0]} "
        f"(user-major slots: dim0 must be num_users * num_layers)"
    )
    sl = ttnn.slice(
        cache,
        [slot_id * num_layers, 0, 0, 0],
        [(slot_id + 1) * num_layers, s[1], s[2], s[3]],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # SP shards seq (dim 2), TP shards kv-head (dim 1) — concat both back to the global shape.
    rotated = ttnn.to_torch(
        sl, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=tuple(mesh_device.shape))
    ).float()
    ttnn.deallocate(sl)

    cache_len = rotated.shape[2]
    assert cache_len % chunk_size_global == 0, (
        f"drafter cache_len {cache_len} is not a whole number of {chunk_size_global}-token bands; the "
        f"un-rotation period MUST equal the period the cache was written at"
    )
    assert out_len <= cache_len, f"out_len {out_len} exceeds drafter cache_len {cache_len}"
    positions = blockcyclic_positions(sp, chunk_size_global, cache_len)
    natural = torch.zeros_like(rotated)
    natural[:, :, positions, :] = rotated
    return natural[:, :, :out_len, :]


def _load_golden_kv(
    golden_dir: Path,
    *,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    out_len: int,
    clamp: bool = False,
    rope_convention: str = "interleaved",
):
    """Load the golden reference context K and V (fp32) from the safetensors artifact in ``golden_dir``,
    trimmed to ``out_len`` (or the golden's own length when ``clamp`` is set).

    The golden stores **half-split** roped K (it was traced from the HF drafter). When the device runs the
    Meta ``"interleaved"`` convention (the meta-rope branch default), its stored K is the half-split K with
    its head_dim ``src``-permuted, so reindex the golden **K only** by the same ``src`` here to compare like
    with like: ``interleaved[j] == halfsplit[src[j]]`` ⇒ ``golden_k[..., src]`` equals the device K. V never
    touches rope, so it is untouched. Exact (a pure permutation), reversible via the config flag."""
    from safetensors import safe_open

    out = []
    for name in ("k_cache", "v_cache"):
        path = golden_dir / f"{name}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"drafter golden {path} not found (set {GOLDEN_KV_ENV} or unset it to skip)")
        with safe_open(path, framework="pt") as f:
            sl = f.get_slice(name)
            shape = list(sl.get_shape())
            assert shape[0] == num_layers and shape[1] == num_kv_heads and shape[3] == head_dim, (
                f"{name} shape {shape} does not match (draft_layer={num_layers}, kv_head={num_kv_heads}, "
                f"seq, head_dim={head_dim}) — the golden's axes must be (layer, head, seq, head_dim)"
            )
            n = min(out_len, shape[2]) if clamp else out_len
            assert shape[2] >= n, f"{name} has only {shape[2]} positions, need {n}"
            out.append(sl[:, :, :n, :].to(torch.float32))
    if out[0].shape[2] != out[1].shape[2]:
        raise RuntimeError(f"drafter golden K/V seq differ: {out[0].shape[2]} vs {out[1].shape[2]}")
    if rope_convention == "interleaved":
        src = torch.argsort(interleaved_to_halfsplit_perm(head_dim))
        out[0] = out[0][..., src].contiguous()  # K only; V never roped
    elif rope_convention != "half_split":
        raise ValueError(f"unknown rope_convention {rope_convention!r} (expected 'interleaved' or 'half_split')")
    logger.info(f"[dflash-pcc] golden K/V {tuple(out[0].shape)} from {golden_dir} (rope_convention={rope_convention})")
    return out[0], out[1]


def _pcc_per_head(tag: str, expected: torch.Tensor, actual: torch.Tensor, threshold: float, failures: list) -> float:
    """Compare expected vs actual K/V per (layer, head), logging each layer's PCC and recording any slice
    at or below ``threshold`` in ``failures``, and return the worst PCC over all (layer, head) slices."""
    from tests.ttnn.utils_for_testing import comp_pcc

    worst, worst_at = 1.0, None
    for i in range(expected.shape[0]):
        heads = []
        for h in range(expected.shape[1]):
            _, pcc = comp_pcc(expected[i, h], actual[i, h])
            heads.append(pcc)
            if pcc < worst:
                worst, worst_at = pcc, (i, h)
            if pcc <= threshold:
                failures.append(f"{tag} layer {i} head {h}: PCC {pcc:.6f} <= {threshold}")
        logger.info(
            f"[dflash-pcc]   {tag} layer {i}: per-head min={min(heads):.6f} (head {heads.index(min(heads))}) "
            f"max={max(heads):.6f}"
        )
    logger.info(f"[dflash-pcc]   --> {tag} min over all (layer, head) = {worst:.6f} at {worst_at}")
    return worst


def dflash_kv_cache_pcc_check(
    mesh_device,
    k_cache,
    v_cache,
    *,
    sp: int,
    chunk_size_global: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    slot_id: int,
    out_len: int,
    golden_dir=None,
    threshold: float = None,
    record_only: bool = False,
    rope_convention: str = "interleaved",
) -> float:
    """Validate the drafter's populated context K/V for one slot against the golden trace by reading the
    device caches directly, returning the minimum per-(layer, head) PCC. Raises on failure unless
    ``record_only``, and returns 1.0 (unmeasured) when no golden is available. Host-side counterpart of
    ``dflash_kv_table_pcc_check`` below. ``rope_convention`` reindexes the golden K to the device's stored-K
    convention (see :func:`_load_golden_kv`)."""
    golden_dir = Path(golden_dir or os.environ.get(GOLDEN_KV_ENV) or GOLDEN_KV_DEFAULT)
    if not golden_dir.is_dir():
        logger.info(f"[dflash-pcc] SKIPPED: golden dir {golden_dir} not present (set {GOLDEN_KV_ENV})")
        return 1.0
    if threshold is None:
        threshold = float(os.environ.get(PCC_ENV, DEFAULT_PCC))

    golden_k, golden_v = _load_golden_kv(
        golden_dir,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        out_len=out_len,
        rope_convention=rope_convention,
    )
    logger.info(
        f"[dflash-pcc] drafter context-KV vs golden: slot={slot_id} out_len={out_len} layers={num_layers} "
        f"kv_heads={num_kv_heads} head_dim={head_dim} sp={sp} chunk_global={chunk_size_global} "
        f"threshold={threshold}"
    )

    failures: list = []
    mins = {}
    # V FIRST, and keep it first: V never touches RoPE, so V-passes-while-K-fails localizes the fault to the
    # rope table rather than to the taps, the tap order, the FC, or the D2D transport of the partial. Both
    # earlier disagreements with this golden failed exactly that way.
    for kind, cache, golden in (("V", v_cache, golden_v), ("K", k_cache, golden_k)):
        actual = read_slot_dflash_kv(
            mesh_device,
            cache,
            sp=sp,
            chunk_size_global=chunk_size_global,
            num_layers=num_layers,
            slot_id=slot_id,
            out_len=out_len,
        )
        mins[kind] = _pcc_per_head(f"drafter-{kind}", golden, actual, threshold, failures)

    min_pcc = min(mins.values())
    if failures:
        msg = f"[dflash-pcc] drafter context-KV below {threshold} ({len(failures)} slice(s)): " + "; ".join(
            failures[:8]
        )
        if record_only:
            logger.warning(f"{msg} (record-only, not asserted)")
        else:
            raise AssertionError(msg)
    else:
        logger.success(
            f"[dflash-pcc] drafter context-KV PASSED vs golden: min PCC {min_pcc:.6f} >= {threshold} "
            f"(V {mins['V']:.6f}, K {mins['K']:.6f})"
        )
    return min_pcc


# ---------------------------------------------------------------------------
# Table-readback sibling of dflash_kv_cache_pcc_check: read the drafter KV back THROUGH the published KV
# chunk address table over UMD -- the exact path migration runs on -- instead of from the device cache
# handle. Called from the producer's resident-slot validator when the published table carries dflash_*
# configs (a real DFlash run); the KVPE/indexer e2e publishes none, so it does not exercise this path.
# ---------------------------------------------------------------------------

_BFP8_TILE_BYTES = 1088  # one [32, 32] bfloat8_b tile: 64 exponent + 1024 mantissa bytes
_KV_BLOCK_TOKENS = 32  # NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK: table reads go by whole 32-token blocks


def _dflash_config_ids(table):
    """Return the drafter caches' ``(k_ids, v_ids)`` config ids (indexed by kv-head) from a published KV
    table, or None when the table carries no drafter configs; raises on a partial/malformed set."""
    from ..runners.kv_chunk_table import dflash_config_name

    names = [table.config_name(i) for i in range(table.num_configs())]
    present = {n for n in names if n.startswith("dflash_")}
    if not present:
        return None
    num_kv_heads = len(present) // 2
    expected = {dflash_config_name(kind, h) for kind in ("k", "v") for h in range(num_kv_heads)}
    if present != expected:
        raise RuntimeError(
            f"drafter table configs are not one K and one V per kv-head: got {sorted(present)}, "
            f"expected {sorted(expected)} for {num_kv_heads} kv-heads"
        )
    return (
        [table.config_id_of(dflash_config_name("k", h)) for h in range(num_kv_heads)],
        [table.config_id_of(dflash_config_name("v", h)) for h in range(num_kv_heads)],
    )


def dflash_kv_table_pcc_check(
    table,
    slot_id: int,
    real_len: int,
    *,
    read_config_slice,
    threshold: float,
    golden_dir: str = None,
    rope_convention: str = "interleaved",
) -> float | None:
    """Validate the drafter's context K/V for one slot against the golden by reading it back through the
    published KV chunk table (the same path migration uses), returning the minimum per-(layer, head) PCC
    or None when there is nothing to check. Table-based counterpart of ``dflash_kv_cache_pcc_check`` that
    never raises. ``rope_convention`` reindexes the golden K to the device's stored-K convention (see
    :func:`_load_golden_kv`)."""
    ids = _dflash_config_ids(table)
    if ids is None:
        return None
    cfg_ids = {"k": ids[0], "v": ids[1]}
    num_kv_heads = len(ids[0])

    golden_dir = golden_dir or os.environ.get(GOLDEN_KV_ENV, "").strip()
    if not golden_dir:
        logger.info(
            f"[dflash-pcc] table carries the drafter caches ({num_kv_heads} kv-heads x K/V) but "
            f"{GOLDEN_KV_ENV} is unset (prompt-specific golden), so the drafter half is NOT checked."
        )
        return None

    cfg = table.config(cfg_ids["k"][0])
    n_layers = cfg.num_layers  # the DRAFTER's layer count, not the verifier's NUM_LAYERS
    # head_dim inferred from the physical chunk size; _load_golden_kv then cross-checks it against the
    # golden's own head_dim (shape[3]), so a mismatch fails on shape rather than as a bad PCC.
    head_dim = {(d // 32) * _BFP8_TILE_BYTES: d for d in (64, 128, 256)}.get(cfg.chunk_size_bytes)
    if head_dim is None:
        raise RuntimeError(
            f"drafter config chunk_size_bytes {cfg.chunk_size_bytes} is not a known head_dim (64/128/256)"
        )

    # Context K/V at position p depends only on tokens <= p, so a shorter golden still validates its own
    # prefix; clamp to what the golden holds and read cmp_len back off its shape.
    golden_k, golden_v = _load_golden_kv(
        Path(golden_dir),
        num_layers=n_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        out_len=real_len,
        clamp=True,
        rope_convention=rope_convention,
    )
    cmp_len = golden_k.shape[2]
    if cmp_len < real_len:
        logger.warning(
            f"[dflash-pcc] slot {slot_id} holds {real_len} tok but golden covers {cmp_len}; checking [0,{cmp_len})."
        )
    # Reads go by whole 32-token blocks, so the rounded length is what the table must cover.
    read_len = ((cmp_len + _KV_BLOCK_TOKENS - 1) // _KV_BLOCK_TOKENS) * _KV_BLOCK_TOKENS
    if read_len > cfg.max_sequence_length:
        raise RuntimeError(f"drafter config max_sequence_length {cfg.max_sequence_length} < {read_len} to read")

    # V before K: V never touches RoPE, so V-passing while K-fails localizes a fault to the rope table.
    failures: list = []
    mins = {}
    for kind, golden in (("v", golden_v), ("k", golden_k)):
        layers = []
        for layer in range(n_layers):
            heads = [
                read_config_slice(cfg_ids[kind][h], layer, slot_id, read_len, head_dim)[:cmp_len]
                for h in range(num_kv_heads)
            ]
            layers.append(torch.stack(heads, dim=0))  # [heads, cmp_len, head_dim]
        actual = torch.stack(layers, dim=0).float()  # [layers, heads, cmp_len, head_dim]
        mins[kind] = _pcc_per_head(f"drafter-{kind.upper()} (slot {slot_id})", golden, actual, threshold, failures)

    min_pcc = min(mins.values())
    if failures:
        logger.error(
            f"[dflash-pcc] slot {slot_id}: drafter KV over [0,{cmp_len}) below {threshold} "
            f"({len(failures)} slice(s)): " + "; ".join(failures[:8])
        )
    else:
        logger.success(
            f"[dflash-pcc] slot {slot_id} drafter KV over [0,{cmp_len}) PASSED: min PCC {min_pcc:.6f} "
            f">= {threshold} (V {mins['v']:.6f}, K {mins['k']:.6f})"
        )
    return min_pcc
