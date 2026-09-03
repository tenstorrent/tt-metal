# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Gate for the in-trace CP device sampler and the fused CP frame's building blocks.

The CodePredictor's 15 sampling steps used to round-trip through the host once per
step. Closing that loop on device needs three things to hold, and each of them broke
in a way that per-op unit tests would not have caught:

  1. ``ttnn.sampling``'s ``seed`` is a COMPILE-TIME kernel argument, so a traced
     sampling op replays one fixed uniform draw forever. Randomness has to come from
     data instead — Gumbel noise the host refreshes per frame. These tests pin the
     equivalence ``argmax(logits + T*g) ~ softmax(logits/T)`` restricted to top-k,
     and that ``ttnn.sampling(k=1)`` is exactly an argmax-gather.
  2. On TP>1 every chip samples from its OWN copy of the logits, and the model's
     tensor-parallel path does not produce bit-identical logits on every device.
     Sampling per chip therefore picks different tokens (measured: 3.6% of tokens on
     N300), and once the token is embedded ON device the TP halves diverge. The
     sampler all-gathers the id and keeps device 0's; ``test_token_agrees_across_chips``
     is the regression gate for that.
  3. ``ttnn.embedding`` needs BFLOAT16 weights but a FLOAT32 output, or the 16-way
     accumulate that builds the next Talker input drifts 1.2% from the host sum.

    pytest -s models/demos/qwen3_tts/tests/test_qwen3_tts_cp_device_sampling.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.demos.qwen3_tts.tt.mesh_utils import to_torch as _mesh_to_torch
from models.demos.qwen3_tts.tt.server import (
    _NOISE_SLOTS,
    _SAMPLING_NEG,
    _SAMPLING_TOPK,
    _DeviceSampler,
    append_device_embedding,
    upload_embed_tables,
)

VOCAB = 2048
HIDDEN = 256
TEMP = 0.9
TOP_K = 50


def _open_device():
    mesh_shape = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"))
    if mesh_shape is None:
        return ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=32768000), None
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    return (
        ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), l1_small_size=32768, trace_region_size=32768000),
        mesh_shape,
    )


@pytest.fixture(scope="module")
def device():
    d, mesh_shape = _open_device()
    d.enable_program_cache()
    yield d
    if mesh_shape is None:
        ttnn.close_device(d)
    else:
        ttnn.close_mesh_device(d)
        if mesh_shape != (1, 1):
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _num_devices(device) -> int:
    try:
        return device.get_num_devices() if device.__class__.__name__ == "MeshDevice" else 1
    except Exception:
        return 1


def _all_chips(device, t) -> torch.Tensor:
    if _num_devices(device) > 1:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    return ttnn.to_torch(t)


def _mapper(device):
    return ttnn.ReplicateTensorToMesh(device) if _num_devices(device) > 1 else None


def _logits_to_device(device, logits: torch.Tensor):
    return ttnn.from_torch(
        logits.reshape(1, 1, 1, -1).bfloat16(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(device),
    )


def _reference_topk_dist(logits: torch.Tensor) -> torch.Tensor:
    """What :func:`server.sample_token` samples from: softmax(logits/T) over top-k."""
    x = logits.bfloat16().float() / TEMP
    kth = torch.topk(x, TOP_K)[0][..., -1, None]
    return F.softmax(x.masked_fill(x < kth, float("-inf")), dim=-1)


# --------------------------------------------------------------------------- #
# Host-side properties of the noise tile
# --------------------------------------------------------------------------- #


def test_noise_tile_is_fresh_per_frame_and_per_slot(device):
    """Every sampling call in a frame must get its OWN Gumbel row.

    A single shared row would correlate all 15 codes of a frame; reusing a tile
    across frames would make the sampler deterministic, which is the exact failure
    mode of ttnn.sampling's own baked seed.
    """
    sampler = _DeviceSampler(device, top_k=TOP_K, temperature=TEMP)
    torch.manual_seed(0)
    sampler.refresh_noise()
    first = _all_chips(device, sampler.noise_tt)[0].float().clone()
    sampler.refresh_noise()
    second = _all_chips(device, sampler.noise_tt)[0].float()

    assert not torch.equal(first, second), "noise tile did not change between frames"
    assert not torch.equal(first[0, 0], first[0, 1]), "slots 0 and 1 share one Gumbel row"
    # Columns at/after top_k carry the mask that applies the top-k truncation.
    assert torch.all(first[0, :, TOP_K:] < _SAMPLING_NEG / 2), "top-k truncation mask missing"
    assert torch.all(first[0, :, :TOP_K] > _SAMPLING_NEG / 2), "live columns were masked"


def test_noise_tile_follows_torch_seed(device):
    """--seed must still make a run reproducible: the tile comes from the global RNG."""
    sampler = _DeviceSampler(device, top_k=TOP_K, temperature=TEMP)
    torch.manual_seed(1234)
    sampler.refresh_noise()
    a = _all_chips(device, sampler.noise_tt)[0].float().clone()
    torch.manual_seed(1234)
    sampler.refresh_noise()
    b = _all_chips(device, sampler.noise_tt)[0].float()
    assert torch.equal(a, b), "same torch seed produced a different noise tile"


def test_noise_tile_is_identical_on_every_chip(device):
    """A per-chip noise tile would make the chips sample differently by construction."""
    if _num_devices(device) < 2:
        pytest.skip("single chip")
    sampler = _DeviceSampler(device, top_k=TOP_K, temperature=TEMP)
    torch.manual_seed(0)
    sampler.refresh_noise()
    tiles = _all_chips(device, sampler.noise_tt).float()
    for c in range(1, tiles.shape[0]):
        assert torch.equal(tiles[0], tiles[c]), f"chip {c} has a different noise tile"


# --------------------------------------------------------------------------- #
# The sampling chain itself
# --------------------------------------------------------------------------- #


def test_chain_is_argmax_gather_of_perturbed_topk(device):
    """The device chain must compute exactly ``topk_indices[argmax(values + noise)]``.

    This is the property Gumbel-max needs; ``ttnn.sampling(k=1)`` is used only for
    the index gather, which no plain ttnn op provides.

    The logits are built so the live candidates are DISTINCT multiples of 0.5 (exactly
    representable in bfloat16) and everything else is far below, which makes ttnn.topk
    and torch.topk agree on the ordering so positions correspond and the noise, which
    is applied BY POSITION, lines up. Real bf16 logits have 27-49 exact ties inside the
    top-64, where the two topk implementations order candidates differently; the
    tie-tolerant behaviour there is covered by
    :func:`test_sampled_token_stays_inside_the_host_top_k` and
    :func:`test_distribution_matches_host_top_k_sampling`.

    Ties in the PERTURBED values are still possible (bf16 has 8 mantissa bits), so the
    invariant asserted is "the chosen candidate attains the maximum", not "it equals
    torch's argmax position" — any maximiser is a correct answer.
    """
    sampler = _DeviceSampler(device, top_k=TOP_K, temperature=TEMP)
    tok = sampler.alloc_token_buf()
    gen = torch.Generator().manual_seed(7)
    mismatches = []
    for i in range(24):
        logits = torch.full((VOCAB,), -100.0)
        winners = torch.randperm(VOCAB, generator=gen)[:TOP_K]
        logits[winners] = torch.arange(TOP_K, dtype=torch.float32) * 0.5
        logits_tt = _logits_to_device(device, logits)
        torch.manual_seed(100 + i)
        sampler.refresh_noise()
        noise = _all_chips(device, sampler.noise_tt)[0, 0, 0].float()

        sampler.append_sampling(logits_tt, 0, tok)
        got = int(_all_chips(device, ttnn.reshape(tok, [1, 1, 1, 1])).flatten()[0])

        vals, idx = torch.topk(logits.bfloat16().float(), _SAMPLING_TOPK)
        pert = (vals.bfloat16() + noise.bfloat16()).bfloat16()
        pos = (idx == got).nonzero().flatten()
        if pos.numel() == 0:
            mismatches.append((i, got, "token not in topk set"))
        elif pert[int(pos[0])] != pert.max():
            mismatches.append((i, got, f"pert={float(pert[int(pos[0])]):.4f} < max={float(pert.max()):.4f}"))
        ttnn.deallocate(logits_tt)
    assert not mismatches, f"device chain != argmax-gather on {len(mismatches)}/24 vectors: {mismatches[:4]}"


def test_sampled_token_stays_inside_the_host_top_k(device):
    """On REAL-shaped (tie-heavy) logits the token must always be a top-k candidate.

    Weaker than exact agreement, but it is the property that survives bf16 ties, and
    it catches a broken index gather (which would return arbitrary vocab ids).
    """
    sampler = _DeviceSampler(device, top_k=TOP_K, temperature=TEMP)
    tok = sampler.alloc_token_buf()
    gen = torch.Generator().manual_seed(21)
    outside = 0
    total = 0
    for i in range(16):
        logits = torch.randn(VOCAB, generator=gen) * 6.0
        logits_tt = _logits_to_device(device, logits)
        allowed = set(torch.topk(logits.bfloat16().float(), _SAMPLING_TOPK)[1].tolist())
        torch.manual_seed(300 + i)
        sampler.refresh_noise()
        for slot in range(8):
            sampler.append_sampling(logits_tt, slot, tok)
            got = int(_all_chips(device, ttnn.reshape(tok, [1, 1, 1, 1])).flatten()[0])
            total += 1
            if got not in allowed:
                outside += 1
        ttnn.deallocate(logits_tt)
    assert outside == 0, f"{outside}/{total} sampled tokens fell outside the top-{_SAMPLING_TOPK} set"


def test_token_agrees_across_chips(device):
    """Regression gate for the TP divergence.

    Real per-chip logits differ in the last bits, so without the all_gather the two
    chips pick different tokens and then embed different rows on device. Feeding
    deliberately DIFFERENT logits per chip is the strongest form of this check: the
    sampler must still return device 0's token everywhere.
    """
    if _num_devices(device) < 2:
        pytest.skip("single chip")
    nd = _num_devices(device)
    sampler = _DeviceSampler(device, top_k=TOP_K, temperature=TEMP)
    tok = sampler.alloc_token_buf()
    torch.manual_seed(0)
    sampler.refresh_noise()

    # Per-chip logits that peak on a different token on each chip.
    per_chip = torch.full((nd, 1, 1, VOCAB), -20.0)
    for c in range(nd):
        per_chip[c, 0, 0, 100 + c * 37] = 50.0
    logits_tt = ttnn.from_torch(
        per_chip.bfloat16(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
    )
    sampler.append_sampling(logits_tt, 0, tok)
    toks = _all_chips(device, ttnn.reshape(tok, [1, 1, 1, 1])).flatten().tolist()
    assert len(set(toks)) == 1, f"chips disagree on the sampled token: {toks}"
    assert toks[0] == 100, f"expected device 0's token (100), got {toks}"


def test_distribution_matches_host_top_k_sampling(device):
    """The sampler must be as close to the exact top-k distribution as the host is.

    A per-step "is the token plausible" check passes even for a badly biased sampler
    (a baked seed picks the rank-8 candidate every time and still stays inside
    top-k), so gate on the empirical distribution instead.
    """
    sampler = _DeviceSampler(device, top_k=TOP_K, temperature=TEMP)
    tok = sampler.alloc_token_buf()
    torch.manual_seed(3)
    logits = torch.randn(VOCAB) * 4.0
    logits_tt = _logits_to_device(device, logits)
    ref = _reference_topk_dist(logits)

    draws = []
    n_frames = 24
    for f in range(n_frames):
        torch.manual_seed(500 + f)
        sampler.refresh_noise()
        for slot in range(_NOISE_SLOTS):
            sampler.append_sampling(logits_tt, slot, tok)
            draws.append(int(_all_chips(device, ttnn.reshape(tok, [1, 1, 1, 1])).flatten()[0]))

    emp = torch.bincount(torch.tensor(draws), minlength=VOCAB).float() / len(draws)
    tv_device = 0.5 * float((emp - ref).abs().sum())

    # Host baseline drawn the same number of times: the Monte-Carlo floor at this
    # sample count. The device sampler must not be materially worse than it.
    host = torch.multinomial(ref, len(draws), replacement=True)
    emp_host = torch.bincount(host, minlength=VOCAB).float() / len(draws)
    tv_host = 0.5 * float((emp_host - ref).abs().sum())

    assert len(set(draws)) > TOP_K // 3, f"sampler collapsed onto {len(set(draws))} tokens"
    assert tv_device < tv_host + 0.05, f"device TV {tv_device:.4f} vs host floor {tv_host:.4f}"
    ttnn.deallocate(logits_tt)


def test_temperature_is_folded_into_the_noise_not_the_kernel(device):
    """``ttnn.sampling``'s ``temp`` is 1/T and is bypassed here (k=1 collapses the
    softmax), so the temperature must come from the noise scale. A hotter sampler has
    to produce a broader spread of ranks."""
    tok_lo = _DeviceSampler(device, top_k=TOP_K, temperature=0.05)
    tok_hi = _DeviceSampler(device, top_k=TOP_K, temperature=2.0)
    torch.manual_seed(11)
    logits = torch.randn(VOCAB) * 4.0
    logits_tt = _logits_to_device(device, logits)
    order = torch.topk(logits.bfloat16().float(), _SAMPLING_TOPK)[1].tolist()
    rank_of = {t: r for r, t in enumerate(order)}

    def mean_rank(sampler):
        buf = sampler.alloc_token_buf()
        ranks = []
        for f in range(6):
            torch.manual_seed(900 + f)
            sampler.refresh_noise()
            for slot in range(_NOISE_SLOTS):
                sampler.append_sampling(logits_tt, slot, buf)
                t = int(_all_chips(device, ttnn.reshape(buf, [1, 1, 1, 1])).flatten()[0])
                ranks.append(rank_of.get(t, _SAMPLING_TOPK))
        return sum(ranks) / len(ranks)

    cold, hot = mean_rank(tok_lo), mean_rank(tok_hi)
    assert cold < hot, f"temperature has no effect: mean rank cold={cold:.2f} hot={hot:.2f}"
    assert cold < 1.0, f"T=0.05 should be near-greedy, got mean rank {cold:.2f}"
    ttnn.deallocate(logits_tt)


# --------------------------------------------------------------------------- #
# Device embedding tables (the other half of closing the loop)
# --------------------------------------------------------------------------- #


def test_device_embedding_matches_f_embedding(device):
    """bf16 tables + a bf16 output reproduce ``F.embedding(...).bfloat16()`` exactly."""
    torch.manual_seed(0)
    codec = torch.randn(VOCAB + 1024, HIDDEN).bfloat16()
    cp = [torch.randn(VOCAB, HIDDEN).bfloat16() for _ in range(3)]
    codec_tt, cp_tts = upload_embed_tables(device, codec, cp)

    for tok, table_tt, table in ((1234, codec_tt, codec), (77, cp_tts[1], cp[1])):
        idx = ttnn.from_torch(
            torch.tensor([[tok]], dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_mapper(device),
        )
        out = append_device_embedding(idx, table_tt, HIDDEN, dtype=ttnn.bfloat16)
        got = _mesh_to_torch(out).reshape(-1).float()
        assert torch.equal(got, table[tok].float()), f"embedding row {tok} is wrong"


def test_float32_accumulate_is_bit_exact_with_the_host_sum(device):
    """The Talker's next input is a 16-way sum of codec rows.

    ``ttnn.embedding`` rejects float32 WEIGHTS but accepts a float32 OUTPUT. Summing
    the bf16 rows in bf16 drifts 1.2% from the host's float32 sum; requesting a
    float32 output makes it bit-exact, which matters because the sum is the Talker's
    input embedding for every frame.
    """
    torch.manual_seed(0)
    n = 16
    codec = torch.randn(VOCAB, HIDDEN).bfloat16()
    tables = [torch.randn(VOCAB, HIDDEN).bfloat16() for _ in range(n - 1)]
    codec_tt, cp_tts = upload_embed_tables(device, codec, tables)
    toks = [int(torch.randint(0, VOCAB, (1,))) for _ in range(n)]

    def _idx(t):
        return ttnn.from_torch(
            torch.tensor([[t]], dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_mapper(device),
        )

    acc = append_device_embedding(_idx(toks[0]), codec_tt, HIDDEN, dtype=ttnn.float32)
    for i in range(1, n):
        row = append_device_embedding(_idx(toks[i]), cp_tts[i - 1], HIDDEN, dtype=ttnn.float32)
        acc = ttnn.add(acc, row)
    got = _mesh_to_torch(ttnn.typecast(acc, ttnn.bfloat16)).reshape(-1).float()

    ref = codec[toks[0]].float().clone()
    for i in range(1, n):
        ref += tables[i - 1][toks[i]].float()
    assert torch.equal(got, ref.bfloat16().float()), "device float32 accumulate drifted from the host sum"
