# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DFlash drafter context-KV PCC gate against the golden GPU trace.

The drafter analog of ``tt/runners/prefill_kv_validation.py``: that module gates the verifier's ``kvpe``
cache, this one gates the drafter's separate K/V context caches. Both are optional bring-up hooks the
runtime forwards into — never called in production serving.

Why this exists as its own gate: ``kv_cache_pcc_check`` covers only ``kv_caches.kvpe``, so with DFlash on,
the drafter cache the last rank writes was validated by NOTHING in the runner. On a multi-rank (D2D)
pipeline that is the one cache whose correctness depends on the inter-galaxy transport — rank 0 sends its
reduce_scattered FC partial packed alongside the hidden, and the drafter's K/V is the only observable that
the partial arrived intact and was accumulated in the right place.

Reference is the golden ``{k,v}_cache.safetensors`` pair, fp32 ``[draft_layer, kv_head, seq, head_dim]``
(``$PREFILL_DFLASH_GOLDEN_KV_DIR``, default ``.../golden/dflash_context_kv_55k_v3``). Those axes are the
device cache's global axes with the slot axis folded out, so the golden indexes straight against the
readback. Comparing a prefix is sound: context K/V at position p depends only on tokens <= p (causal), so
the first ``out_len`` rows of a full-length golden equal an ``out_len``-token run's cache.

The device cache is ``bfloat8_b`` while the golden is fp32, which costs ~1e-4 of PCC — hence a threshold
near 0.999 rather than 0.9999. Measured 0.9998 on the single-galaxy Stage-1 run.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions

# Its own env var rather than riding PREFILL_TRACE_DIR: the drafter golden is a separate artifact with its
# own layout (whole-tensor safetensors, not row-sharded), and a CI leg may point PREFILL_TRACE_DIR at a
# DeepSeek trace that has no drafter golden beside it at all.
GOLDEN_KV_ENV = "PREFILL_DFLASH_GOLDEN_KV_DIR"
GOLDEN_KV_DEFAULT = "/mnt/models/deepseek-prefill-cache/golden/dflash_context_kv_55k_v3"
PCC_ENV = "PREFILL_DFLASH_PCC"
DEFAULT_PCC = 0.99


def read_slot_dflash_kv(
    mesh_device, cache, *, sp: int, chunk_size_global: int, num_layers: int, slot_id: int, out_len: int
) -> torch.Tensor:
    """One slot's drafter K or V cache as ``[num_layers, num_kv_heads, out_len, head_dim]`` fp32, in
    NATURAL token order.

    Two layout inverses, in this order:

    * dim0 is the writer's user-major linearization ``slot = user_id * num_layers + layer_idx``
      (``allocate_dflash_kv_cache``), so this slot is rows ``[slot*L, (slot+1)*L)``. Slice it ON DEVICE
      with an explicit ``DRAM_MEMORY_CONFIG``: the cache is ND-sharded ROUND_ROBIN_1D and slicing into
      another ND-shard miscomputes the DRAM core on read-back (same requirement as ``read_slot_kv``).
    * dim 2 is block-cyclic across SP chips. ``blockcyclic_positions`` maps shard row -> global natural
      position, so the inverse is a SCATTER (``natural[p] = rotated``), never a gather — a gather is the
      whole-cache PCC 0.354 = sqrt(1/8) signature.

    Un-rotate the FULL cache and slice to ``out_len`` afterwards. Slicing first would take shard rows
    rather than natural positions, which is a no-op only in the degenerate ``cache_len == chunk_size_global``
    case (one aligned chunk) that hides every layout mistake.
    """
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


def _load_golden_kv(golden_dir: Path, *, num_layers: int, num_kv_heads: int, head_dim: int, out_len: int):
    """Golden context K and V as ``[num_layers, num_kv_heads, out_len, head_dim]`` fp32.

    The shape assert is the axis-mapping check: if the artifact ever changes axis order this fails here
    instead of producing a uniformly bad PCC that reads like a device bug. ``get_slice`` keeps the read
    partial — at ``out_len`` 5120 that is 126 MiB rather than the full 1.38 GiB.
    """
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
            assert shape[2] >= out_len, f"{name} has only {shape[2]} positions, need {out_len}"
            out.append(sl[:, :, :out_len, :].to(torch.float32))
    logger.info(f"[dflash-pcc] golden K/V {tuple(out[0].shape)} from {golden_dir}")
    return out[0], out[1]


def _pcc_per_head(tag: str, expected: torch.Tensor, actual: torch.Tensor, threshold: float, failures: list) -> float:
    """PCC every (layer, head) slice separately, log each layer, and return the minimum.

    Per-head, not whole-tensor: PCC dilutes as 1 - (fraction wrong), so one wrong head out of 48 lands at
    ~0.99 on the whole tensor and hides under any threshold worth setting. Per-head also localizes the
    fault — all heads of one TP column bad means the column/device-group mapping, one head per chip bad
    means the ``head_idx % heads_per_chip`` term, one layer bad means the head/layer shard stride.
    """
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
) -> float:
    """PCC the drafter's populated context K/V for ``slot_id`` against the golden trace; returns the min
    per-(layer, head) PCC and raises on failure unless ``record_only``.

    Returns 1.0 unmeasured when the golden dir is absent, so a bring-up run on a host without the golden
    artifact is not turned into a failure by enabling the gate.

    Reads the caches host-side and un-rotates with ``blockcyclic_positions`` — the same shape as the
    verifier's ``kv_cache_pcc_check``. That the migration KV chunk address table addresses these same
    caches correctly is a separate concern, covered by test_dflash_trace_table.py (table readback vs
    host-math readback, bit for bit) and the producer<->runner mock-migration e2e.
    """
    golden_dir = Path(golden_dir or os.environ.get(GOLDEN_KV_ENV) or GOLDEN_KV_DEFAULT)
    if not golden_dir.is_dir():
        logger.info(f"[dflash-pcc] SKIPPED: golden dir {golden_dir} not present (set {GOLDEN_KV_ENV})")
        return 1.0
    if threshold is None:
        threshold = float(os.environ.get(PCC_ENV, DEFAULT_PCC))

    golden_k, golden_v = _load_golden_kv(
        golden_dir, num_layers=num_layers, num_kv_heads=num_kv_heads, head_dim=head_dim, out_len=out_len
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
