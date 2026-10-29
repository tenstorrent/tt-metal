# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked prefill (``prefill_forward(..., start=...)``) vs the HF dumps and vs the single-shot
prefill, real weights, one chip.

* ``test_r4097_chunked_vs_dump_and_single_shot``: layers 0 (KDA + dense) and 3 (DSA + MoE) on the
  random-prompt dump ``R4097`` in chunks 1024/1024/1024/1025 (last chunk non-aligned; layer 3 also
  with 1536/1024/1024/513 so one chunk straddles the dense/sparse row split at 2048): every chunk
  and the concatenation vs the dump and vs the single-shot output, DSA per-row classes, then the
  dump's decode steps after the chunked prefill.
* ``test_512_chunks_then_decode``: layer 3, dense-only prompts ``R2040`` / ``R2042`` in 512-token
  chunks plus a non-aligned last chunk (tails 0 and 2), 12 decode steps across the 2048/2050 switch
  and the pooled-key cache re-derived on the host.
* ``test_chunk_equals_single_shot``: both layers at T=2040 in chunks 1024 + 1016; the contract
  checks (alignment, contiguity) on the same decoder.
* ``test_indexer_maintain_prefill_offsets``: the index maintenance alone at arbitrary chunk
  boundaries (starts not multiples of the pool size, a chunk ending inside a straddling pool) vs
  the host pooling; cheap, no attention.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_chunked_prefill.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import attention_input as _attention_input
from models.demos.glm53_flash.tests.utils import hf_index_stems as _hf_index_stems
from models.demos.glm53_flash.tests.utils import hf_pool as _hf_pool
from models.demos.glm53_flash.tests.utils import pcc, per_row_stats
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.functional_decoder import (
    FunctionalDecoder,
    device_streams_to_host,
    host_streams_to_device,
)
from models.demos.glm53_flash.tt.indexer import STATIC_PREFILL_ENV, TtDSAIndexer, TtIndexCache

REF = Path("bringup/references/glm53_flash")
PCC_BAR = 0.995
DENSE_ROWS = 2048  # index_topk: rows at absolute positions below it attend densely


def _load(layer_idx: int):
    snapshot = default_snapshot_dir()
    raw = load_layer_raw(snapshot, layer_idx, shard_headers(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    return raw, cfg


def _decoder(mesh_device, raw, cfg, layer_idx: int, max_seq_len: int, max_batch: int = 1) -> FunctionalDecoder:
    return FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=layer_idx, mesh_device=mesh_device, max_batch=max_batch, max_seq_len=max_seq_len
    )


def _kind(dec: FunctionalDecoder) -> str:
    return f"{'KDA' if dec.layer.is_kda else 'DSA'}+{'MoE' if dec.layer.is_moe else 'dense'}"


def prefill_single(dec: FunctionalDecoder, x_in: torch.Tensor, device, user: int = 0) -> torch.Tensor:
    S = x_in.shape[1]
    y = dec.prefill_forward(host_streams_to_device(x_in, device), user=user, seq_len=S)
    out = device_streams_to_host(y, 4, S)
    ttnn.deallocate(y)
    return out


def prefill_chunked(dec: FunctionalDecoder, x_in: torch.Tensor, chunks, device, user: int = 0):
    """Prefill ``x_in`` [1,S,4,4096] in consecutive chunks of the given logical lengths; returns the
    concatenated host output [1,S,4,4096] and the per-chunk outputs."""
    assert sum(chunks) == x_in.shape[1], (chunks, x_in.shape)
    outs, start = [], 0
    for n in chunks:
        y = dec.prefill_forward(
            host_streams_to_device(x_in[:, start : start + n], device), user=user, seq_len=n, start=start
        )
        outs.append(device_streams_to_host(y, 4, n))
        ttnn.deallocate(y)
        start += n
    return torch.cat(outs, dim=1), outs


def _row_classes(ref: torch.Tensor, out: torch.Tensor, start: int, P: int) -> dict[str, float]:
    """PCC per row class of a [1,n,4,4096] chunk whose row i is absolute position start + i."""
    pos = torch.arange(ref.shape[1]) + start
    classes = {"rows<2048": pos < DENSE_ROWS, "rows>=2048": pos >= DENSE_ROWS}
    classes.update({f"pos%{P}=={j}": (pos % P) == j for j in range(P)})
    return {k: pcc(ref[:, m], out[:, m]) for k, m in classes.items() if m.any()}


def _fmt(d: dict[str, float]) -> str:
    return " ".join(f"{k} {v:.5f}" for k, v in d.items())


def _equality(a: torch.Tensor, b: torch.Tensor) -> str:
    a, b = a.float(), b.float()
    return f"max|diff| {(a - b).abs().max():.3g}, equal elements {(a == b).float().mean() * 100:.2f}%"


def _decode_steps(dec: FunctionalDecoder, dump: Path, layer_idx: int, steps, device, label: str):
    for step in steps:
        d = torch.load(dump / f"layer{layer_idx}_decode{step}.pt")
        dp = dec.make_decode_positions([d["position"]])
        yd = dec.decode_forward(host_streams_to_device(d["x_in"], device, pad_to=1), positions=dp)
        p = pcc(d["x_out"], device_streams_to_host(yd, 4, 1))
        logger.info(f"  {label} decode step {step} (pos {d['position']}): pcc {p:.6f}")
        assert p > PCC_BAR


# ------------------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx, chunks",
    [(0, (1024, 1024, 1024, 1025)), (3, (1024, 1024, 1024, 1025)), (3, (1536, 1024, 1024, 513))],
    ids=["layer0-1024x3+1025", "layer3-1024x3+1025", "layer3-1536+1024x2+513"],
)
def test_r4097_chunked_vs_dump_and_single_shot(mesh_device, layer_idx, chunks):
    raw, cfg = _load(layer_idx)
    P = cfg.index_kpool
    dump = REF / "R4097"
    pre = torch.load(dump / f"layer{layer_idx}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    dec = _decoder(mesh_device, raw, cfg, layer_idx, S + 64)
    kda = dec.layer.is_kda

    single = prefill_single(dec, x_in, mesh_device)
    p_single = pcc(x_out, single)
    logger.info(f"layer {layer_idx} ({_kind(dec)}) single-shot S={S}: pcc vs HF {p_single:.6f}")
    assert p_single > PCC_BAR

    chunked, parts = prefill_chunked(dec, x_in, chunks, mesh_device)
    start = 0
    for i, (n, part) in enumerate(zip(chunks, parts)):
        ref, sgl = x_out[:, start : start + n], single[:, start : start + n]
        p_hf, p_ss = pcc(ref, part), pcc(sgl, part)
        nd = min(n, max(0, DENSE_ROWS - start))
        split = "" if kda else f" dense {nd} + sparse {n - nd} rows;"
        logger.info(
            f"  chunk {i} [{start},{start + n}){split} pcc vs HF {p_hf:.6f}, vs single-shot {p_ss:.6f} "
            f"({_equality(sgl, part)})"
        )
        assert p_hf > PCC_BAR and p_ss > PCC_BAR
        if not kda:
            classes = _row_classes(ref, part, start, P)
            logger.info(f"    row classes vs HF: {_fmt(classes)}")
            for name, v in classes.items():
                assert v > PCC_BAR, f"chunk {i} row class {name}: pcc {v}"
        start += n
    p_all, p_vs = pcc(x_out, chunked), pcc(single, chunked)
    logger.info(
        f"layer {layer_idx} chunked {list(chunks)} S={S}: pcc vs HF {p_all:.6f}, vs single-shot {p_vs:.6f} "
        f"({_equality(single, chunked)})  {per_row_stats(x_out, chunked)}"
    )
    assert p_all > PCC_BAR and p_vs > PCC_BAR
    if not kda:
        classes = _row_classes(x_out, chunked, 0, P)
        logger.info(f"  row classes vs HF: {_fmt(classes)}")
        for name, v in classes.items():
            assert v > PCC_BAR, f"row class {name}: pcc {v}"

    _decode_steps(dec, dump, layer_idx, range(4), mesh_device, f"layer {layer_idx} after chunked prefill:")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "chunks",
    [(1024, 1024, 1024, 1025), (320, 2048, 1729), (320, 704, 1024, 2049)],
    ids=["aligned-1024x3+1025", "unaligned-320+2048mixed+1729", "unaligned-320+704+1024+2049"],
)
def test_dense_prefix_static_matches_dynamic(mesh_device, chunks):
    """``GLM53_DENSE_PREFIX_STATIC=1`` (the shape-static dense prefix of ``TtDSA``: the keys are the user's first
    2048 cached rows whatever the start, the dense query rows a superset masked by position) against the exact-prefix
    path on the same DSA layer (3) and chunk plan, the plans including chunks that start at a 32-aligned position
    inside the dense range (320: a prefix-cache restore at a decode crossing) and a mixed chunk from there
    (320 + 2048 > 2048) and an all-sparse chunk (from 2048 / 2368). Both paths run in the same decoder on their own
    user slot (0: dynamic, 1: static). Gate: PCC between the paths > 0.9999 (the key set only grows by rows the
    causal mask excludes) and PCC vs HF within 1e-4 of the dynamic path's."""
    layer_idx = 3
    raw, cfg = _load(layer_idx)
    dump = REF / "R4097"
    pre = torch.load(dump / f"layer{layer_idx}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    dec = _decoder(mesh_device, raw, cfg, layer_idx, S + 64, max_batch=2)
    assert not dec.layer.is_kda, "layer 3 is a DSA layer"
    attn = dec.layer.attn
    attn.dense_prefix_static = False
    dyn, _ = prefill_chunked(dec, x_in, chunks, mesh_device, user=0)
    attn.dense_prefix_static = True
    static, _ = prefill_chunked(dec, x_in, chunks, mesh_device, user=1)
    attn.dense_prefix_static = False
    p_dyn, p_static, p_vs = pcc(x_out, dyn), pcc(x_out, static), pcc(dyn, static)
    logger.info(
        f"layer {layer_idx} chunks {list(chunks)}: dynamic pcc vs HF {p_dyn:.6f}, static {p_static:.6f}, static vs dynamic "
        f"{p_vs:.6f} ({_equality(dyn, static)})"
    )
    assert p_vs > 0.9999, f"static vs dynamic pcc {p_vs}"
    assert p_static > p_dyn - 1e-4, f"static pcc vs HF {p_static} below the dynamic path's {p_dyn}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("dump_name, chunks", [("R2042", (512, 512, 512, 506)), ("R2040", (512, 512, 512, 504))])
def test_512_chunks_then_decode(mesh_device, dump_name, chunks):
    """Layer 3, dense-only region: 512-token chunks + a non-aligned last chunk, 12 decode steps through
    the completion of the last pool and the 2048/2050 dense->sparse switch, pooled-key cache vs host."""
    layer_idx = 3
    raw, cfg = _load(layer_idx)
    P = cfg.index_kpool
    dump = REF / dump_name
    pre = torch.load(dump / f"layer{layer_idx}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    assert S == sum(chunks) and S < DENSE_ROWS
    dec = _decoder(mesh_device, raw, cfg, layer_idx, S + 64)

    chunked, parts = prefill_chunked(dec, x_in, chunks, mesh_device)
    start = 0
    for i, (n, part) in enumerate(zip(chunks, parts)):
        p = pcc(x_out[:, start : start + n], part)
        logger.info(f"  {dump_name} chunk {i} [{start},{start + n}): pcc vs HF {p:.6f}")
        assert p > PCC_BAR
        start += n
    p_all = pcc(x_out, chunked)
    logger.info(
        f"layer 3 {dump_name} chunked {list(chunks)} (tail {S % P}): pcc vs HF {p_all:.6f}  "
        f"{_fmt(_row_classes(x_out, chunked, 0, P))}  {per_row_stats(x_out, chunked)}"
    )
    assert p_all > PCC_BAR

    attn_inputs = [_attention_input(raw, cfg, x_in)]
    for step in range(12):
        d = torch.load(dump / f"layer{layer_idx}_decode{step}.pt")
        dp = dec.make_decode_positions([d["position"]])
        yd = dec.decode_forward(host_streams_to_device(d["x_in"], mesh_device, pad_to=1), positions=dp)
        p_d = pcc(d["x_out"], device_streams_to_host(yd, 4, 1))
        attn_inputs.append(_attention_input(raw, cfg, d["x_in"]))
        mode = "sparse" if d["position"] > dec.layer.attn.dense_exact_max_pos else "dense"
        logger.info(f"  decode pos {d['position']} ({mode}, tail {(d['position'] + 1) % P}): pcc {p_d:.6f}")
        assert p_d > PCC_BAR
    k, g, ape = _hf_index_stems(raw, cfg, torch.cat(attn_inputs, dim=0))
    pools_ref = _hf_pool(k, g, ape, P)
    n_complete = pools_ref.shape[0]
    cache_rows = ttnn.to_torch(dec.cache.index.pools_batched())[0, 0, :n_complete].float()
    worst = min(pcc(pools_ref[i], cache_rows[i]) for i in range(n_complete))
    logger.info(
        f"  pooled-key cache: {n_complete} complete pools vs host, pcc {pcc(pools_ref, cache_rows):.6f}, worst row {worst:.5f}"
    )
    assert worst > 0.98


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [0, 3])
def test_chunk_equals_single_shot(mesh_device, layer_idx, expect_error):
    """T=2040 (dump R2040) as 1024 + 1016 (non-aligned last chunk) vs the single-shot prefill."""
    raw, cfg = _load(layer_idx)
    dump = REF / "R2040"
    pre = torch.load(dump / f"layer{layer_idx}_prefill.pt")
    x_in, x_out = pre["x_in"], pre["x_out"]
    S = x_in.shape[1]
    chunks = (1024, S - 1024)
    dec = _decoder(mesh_device, raw, cfg, layer_idx, S + 64)

    single = prefill_single(dec, x_in, mesh_device)
    chunked, parts = prefill_chunked(dec, x_in, chunks, mesh_device)
    p_hf_s, p_hf_c, p_vs = pcc(x_out, single), pcc(x_out, chunked), pcc(single, chunked)
    p_c1 = pcc(single[:, 1024:], parts[1])
    logger.info(
        f"layer {layer_idx} ({_kind(dec)}) S={S} chunks {list(chunks)}: single-shot vs HF {p_hf_s:.6f}, chunked vs HF "
        f"{p_hf_c:.6f}, chunked vs single-shot {p_vs:.6f} (second chunk {p_c1:.6f}; {_equality(single, chunked)})"
    )
    assert p_hf_c > PCC_BAR and p_vs > PCC_BAR and p_c1 > PCC_BAR

    # contract checks on the same decoder (host-side validation, no device work)
    x32 = host_streams_to_device(x_in[:, :32], mesh_device)
    assert dec.next_start[0] == S
    with expect_error(ValueError, "contiguous"):
        dec.prefill_forward(x32, user=0, seq_len=32, start=2048)  # the prompt ended at 2040
    with expect_error(ValueError, "multiple of"):
        dec.prefill_forward(x32, user=0, seq_len=32, start=16)
    with expect_error(ValueError, "max_seq_len"):
        dec.prefill_forward(x32, user=0, seq_len=32, start=-(-(S + 64) // 32) * 32)  # aligned, past max_seq_len
    logger.info(
        "  contract checks: non-contiguous start, unaligned start and start + seq_len > max_seq_len are refused"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_indexer_maintain_prefill_offsets(mesh_device):
    """Index maintenance at arbitrary chunk boundaries (indexer only, layer-3 weights): starts that are
    not pool multiples, a chunk ending inside a straddling pool, a 2-row chunk; after every chunk the
    complete pools and the ring rows of the user match the host, and the other user is untouched."""
    layer_idx = 3
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    P, D = cfg.index_kpool, cfg.index_head_dim
    keep = lambda n: n.startswith("self_attn.indexer.") or n.startswith("hc_attn") or n == "input_layernorm.weight"
    raw = load_layer_raw(snapshot, layer_idx, shard_headers(snapshot), name_filter=keep)
    idx = TtDSAIndexer(mesh_device, cfg, {k[len("self_attn.indexer.") :]: v for k, v in raw.items() if "indexer." in k})
    cache = TtIndexCache(mesh_device, cfg, max_batch=2, max_seq_len=512)
    x_streams = torch.load(REF / "R2040" / f"layer{layer_idx}_prefill.pt")["x_in"]
    S = 300
    x_all = _attention_input(raw, cfg, x_streams[:, :S])  # [S, 4096] bf16 normalized attention inputs
    k_ref, g_ref, ape = _hf_index_stems(raw, cfg, x_all)
    pools_ref = _hf_pool(k_ref, g_ref, ape, P)  # [75, D]

    def chunk_tensor(start, n):
        T = -(-n // 32) * 32
        xp = torch.zeros(1, 1, T, cfg.hidden_size, dtype=torch.bfloat16)
        xp[0, 0, :n] = x_all[start : start + n]
        return ttnn.from_torch(xp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    # user 0: one plain chunk (must stay untouched by user 1's writes)
    idx.maintain_prefill(chunk_tensor(0, 70), cache, user=0, seq_len=70)
    user0_before = ttnn.to_torch(cache.pools_batched())[0, 0, :17].float().clone()
    plan = [(0, 37), (37, 1), (38, 89), (127, 5), (132, 2), (134, 166)]  # starts % 4 = 0,1,2,3,0,2
    assert all(a + n == b for (a, n), (b, _) in zip(plan, plan[1:])) and plan[-1][0] + plan[-1][1] == S
    for start, n in plan:
        idx.maintain_prefill(chunk_tensor(start, n), cache, user=1, seq_len=n, start=start)
        end = start + n
        n_complete, tail = end // P, end % P
        rows = ttnn.to_torch(cache.pools_batched())[1, 0, :n_complete].float()
        worst = min(pcc(pools_ref[i], rows[i]) for i in range(n_complete)) if n_complete else float("nan")
        ring_k = ttnn.to_torch(cache.ring_k)[1, 0, :tail].float()
        ring_g = ttnn.to_torch(cache.ring_g)[1, 0, :tail].float()
        p_ring = (
            min(pcc(k_ref[n_complete * P : end], ring_k), pcc(g_ref[n_complete * P : end], ring_g))
            if tail
            else float("nan")
        )
        logger.info(
            f"  chunk [{start},{end}) (start%P={start % P}): {n_complete} complete pools, worst row pcc {worst:.5f}; "
            f"ring tail {tail} rows pcc {p_ring:.5f}"
        )
        assert not n_complete or worst > 0.98
        assert not tail or p_ring > 0.99
    user0_after = ttnn.to_torch(cache.pools_batched())[0, 0, :17].float()
    assert torch.equal(user0_before, user0_after), "user 0's pools changed"
    logger.info("  user 0's pools untouched by user 1's chunked maintenance")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_indexer_static_maintenance_matches_offset_path(mesh_device, monkeypatch):
    """The shape-static eager maintenance (an aligned chunk padded to a multiple of 128 rows: every op shape
    depends on the padded length only, like the traced path) writes the same complete pools and ring rows as
    the offset path for a prompt whose length is not a multiple of the pool (286 = 71 pools + 2 tail rows, the
    served image prompt that hung the mesh on 2026-10-28), and matches the host reference."""
    layer_idx = 3
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    P, D = cfg.index_kpool, cfg.index_head_dim
    keep = lambda n: n.startswith("self_attn.indexer.") or n.startswith("hc_attn") or n == "input_layernorm.weight"
    raw = load_layer_raw(snapshot, layer_idx, shard_headers(snapshot), name_filter=keep)
    idx = TtDSAIndexer(mesh_device, cfg, {k[len("self_attn.indexer.") :]: v for k, v in raw.items() if "indexer." in k})
    cache = TtIndexCache(mesh_device, cfg, max_batch=2, max_seq_len=512)
    x_streams = torch.load(REF / "R2040" / f"layer{layer_idx}_prefill.pt")["x_in"]
    S, T_pad = 286, 512
    x_all = _attention_input(raw, cfg, x_streams[:, :S])
    k_ref, g_ref, ape = _hf_index_stems(raw, cfg, x_all)
    pools_ref = _hf_pool(k_ref, g_ref, ape, P)  # [71, D]
    n_complete, tail = S // P, S % P
    xp = torch.zeros(1, 1, T_pad, cfg.hidden_size, dtype=torch.bfloat16)
    xp[0, 0, :S] = x_all
    chunk = lambda: ttnn.from_torch(xp, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)

    monkeypatch.setenv(STATIC_PREFILL_ENV, "0")
    idx.maintain_prefill(chunk(), cache, user=0, seq_len=S)  # the offset path into user 0
    assert idx.static_maintenance_calls == 0
    monkeypatch.setenv(STATIC_PREFILL_ENV, "1")
    idx.maintain_prefill(chunk(), cache, user=1, seq_len=S)  # the static path into user 1
    assert idx.static_maintenance_calls == 1

    pools = ttnn.to_torch(cache.pools_batched()).float()
    rk, rg = ttnn.to_torch(cache.ring_k).float(), ttnn.to_torch(cache.ring_g).float()
    worst_ref = min(pcc(pools_ref[i], pools[1, 0, i]) for i in range(n_complete))
    same_pools = torch.equal(pools[0, 0, :n_complete], pools[1, 0, :n_complete])
    same_ring = torch.equal(rk[0, 0, :tail], rk[1, 0, :tail]) and torch.equal(rg[0, 0, :tail], rg[1, 0, :tail])
    p_ring = min(pcc(k_ref[n_complete * P : S], rk[1, 0, :tail]), pcc(g_ref[n_complete * P : S], rg[1, 0, :tail]))
    logger.info(
        f"  static vs offset maintenance of {S} tokens: {n_complete} complete pools "
        f"{'bit-identical' if same_pools else 'differ'} (worst row pcc vs HF {worst_ref:.5f}); ring tail {tail} rows "
        f"{'bit-identical' if same_ring else 'differ'} (pcc vs HF {p_ring:.5f})"
    )
    assert worst_ref > 0.98 and p_ring > 0.99
    assert same_pools and same_ring, "the static maintenance must reproduce the offset path's writes"


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "aligned, split",
    [((16384, 16384, 16384, 16362), (16384, 16384, 3968, 16256, 12522))],
    ids=["65514-continued-at-36736-and-52992"],
)
def test_sparse_continuation_at_unaligned_chunk_ends(mesh_device, aligned, split):
    """A chunk plan whose chunks end at 32-aligned but not device-chunk-aligned positions deep in the sparse
    range must give the same layer output as the device-chunk-aligned plan of the same prompt. That is what
    vLLM's chunked-prefill budget produces under prefix caching when two prompts prefill in one step (the
    2026-10-29 APC smoke: a request's chunk ended at 36736, the next continued from there; another continued
    from 52992), so the adapter continues a prefill from 36736 / 52992 instead of 32768 / 49152. Layer 3
    (DSA + MoE), random activations at the dump's scale (no HF dump at this length), both plans in the same
    decoder on their own user slot. Gate: PCC between the plans > 0.999 overall and on the rows after the
    first unaligned continuation (bf16 noise only; a wrong prefix read / split shows as a collapse there)."""
    layer_idx = 3
    raw, cfg = _load(layer_idx)
    assert sum(aligned) == sum(split), (aligned, split)
    S = sum(aligned)
    pre = torch.load(REF / "R4097" / f"layer{layer_idx}_prefill.pt")
    scale = pre["x_in"].float().std().item()
    torch.manual_seed(0)
    x_in = (torch.randn(1, S, 4, cfg.hidden_size) * scale).to(pre["x_in"].dtype)
    dec = _decoder(mesh_device, raw, cfg, layer_idx, S + 64, max_batch=2)
    assert not dec.layer.is_kda, "layer 3 is a DSA layer"
    ref, _ = prefill_chunked(dec, x_in, aligned, mesh_device, user=0)
    out, _ = prefill_chunked(dec, x_in, split, mesh_device, user=1)
    first_unaligned_end = next(e for e in torch.tensor(split).cumsum(0).tolist() if e % 16384)
    p_all = pcc(ref, out)
    p_tail = pcc(ref[:, first_unaligned_end:], out[:, first_unaligned_end:])
    p_head = pcc(ref[:, :first_unaligned_end], out[:, :first_unaligned_end])
    logger.info(
        f"layer {layer_idx} S={S}: split plan {list(split)} vs aligned {list(aligned)}: pcc {p_all:.6f} "
        f"(rows < {first_unaligned_end}: {p_head:.6f}, rows from the first unaligned continuation: {p_tail:.6f})"
    )
    assert p_head > 0.999, f"the rows before the first unaligned chunk end differ: pcc {p_head}"
    assert p_tail > 0.999 and p_all > 0.999, f"continuation at an unaligned chunk end: pcc {p_tail} (all {p_all})"
