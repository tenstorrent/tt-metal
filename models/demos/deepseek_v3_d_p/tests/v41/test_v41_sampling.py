# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The batched decode's device sampling tail (``TtV41BatchDecoder.sample``) on the mesh, against torch: argmax rows,
top-k rows (the sample lies in the row's top-k) and the gathered slot order; and its draws
against the CPU model of the served sampler (``spec_sample.py``): the histogram, the match rule's acceptance,
the seeded stream."""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import ROWS, DeviceSampler
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.pool import V41DecodePool


@pytest.mark.timeout(int(os.environ.get("V41_SAMPLING_TIMEOUT", 1800)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=0), indirect=True)
def test_v41_device_sampling(mesh_device, device_params):
    sp, tp = tuple(mesh_device.shape)
    vocab = C.VOCAB_SIZE
    per_chip = vocab // tp
    torch.manual_seed(0)
    # every rank its own logits [ROWS, vocab]; chip (r, j) holds columns [j * per_chip, (j + 1) * per_chip)
    logits = torch.randn(sp, ROWS, vocab) * 4
    # a clear maximum per row (the candidates are compared in bf16: two of 129,280 Gaussians can tie there)
    peak = torch.randint(0, vocab, (sp, ROWS))
    logits.view(sp * ROWS, vocab)[torch.arange(sp * ROWS), peak.flatten()] = 40.0
    host = logits.view(sp, ROWS, tp, per_chip).permute(0, 2, 1, 3).reshape(sp, tp, 1, 1, ROWS, per_chip)
    t = ttnn.from_torch(
        host.contiguous(),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
    )
    t = ttnn.reshape(t, (1, 1, ROWS, per_chip))
    ccl = V41Collectives(mesh_device)
    pool_like = V41DecodePool.__new__(V41DecodePool)  # only gather_sp is needed
    V41DecodePool._init_ccl(pool_like, mesh_device)
    sampler = DeviceSampler(mesh_device, C, ccl, pool_like.gather_sp)
    # argmax on every row
    k = torch.ones(sp * ROWS, dtype=torch.int32)
    p = torch.zeros(sp * ROWS)
    temp = torch.ones(sp * ROWS)
    inputs = sampler.host_inputs(k, p, temp)
    buffers = {name: ttnn.to_device(v, mesh_device) for name, v in inputs.items()}
    ids = sampler(ttnn.clone(t), buffers)  # the sampler consumes its logits
    got = DeviceSampler.to_host(ids)
    want = logits.view(sp * ROWS, vocab).argmax(dim=-1)
    bad = (got != want).nonzero().flatten().tolist()
    if bad:
        # the candidates of the first bad rows: is the peak among them, what does the op see
        values, indices = sampler.candidates(ttnn.clone(t))
        for slot in bad[:4]:
            r, u = divmod(slot, ROWS)
            v = ttnn.to_torch(ttnn.get_device_tensors(values)[r * tp])[0, 0, u].float()
            i = ttnn.to_torch(ttnn.get_device_tensors(indices)[r * tp])[0, 0, u].to(torch.int64)
            where = (i == int(want[slot])).nonzero().flatten().tolist()
            print(
                f"slot {slot}: peak {int(want[slot])} among the candidates at {where}; candidate max "
                f"{v.max().item():.2f} at {int(i[v.argmax()])}; got {int(got[slot])} (candidate "
                f"{(i == int(got[slot])).nonzero().flatten().tolist()}); first 8 {i[:8].tolist()} {v[:8].tolist()}"
            )
    assert not bad, f"argmax rows {bad[:8]}: got {got[bad[:8]].tolist()} want {want[bad[:8]].tolist()}"
    # top-8 sampling at temperature 1 on every row of flat logits (the top-8 within a few tenths of each other),
    # 16 draws: every draw within the row's top-8 (ties in bf16 allowed), and not always the same one
    k[:] = 8
    p[:] = 1.0  # no nucleus truncation (p 0 keeps the top candidate only: the argmax convention)
    inputs = sampler.host_inputs(k, p, temp)
    for name, v in inputs.items():
        ttnn.copy_host_to_device_tensor(v, buffers[name])
    flat = torch.randn(sp, ROWS, vocab)
    host = flat.view(sp, ROWS, tp, per_chip).permute(0, 2, 1, 3).reshape(sp, tp, 1, 1, ROWS, per_chip)
    t2 = ttnn.from_torch(
        host.contiguous(),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
    )
    t2 = ttnn.reshape(t2, (1, 1, ROWS, per_chip))
    # the row's top-12 (the candidates are compared in bf16: the 8th and 9th of 129,280 normals are within it)
    top8 = flat.view(sp * ROWS, vocab).topk(12, dim=-1).indices
    draws = []
    for _ in range(16):
        ids = sampler(ttnn.clone(t2), buffers)
        draws.append(DeviceSampler.to_host(ids))
    draws = torch.stack(draws)  # [16, slots]
    inside = (draws.unsqueeze(-1) == top8.unsqueeze(0)).any(-1)
    assert inside.all(), f"draws outside the top-12: {(~inside).sum().item()} of {inside.numel()}"
    assert (draws != draws[0]).any(), "16 draws at temperature 1 over 8 candidates never differed"


def _chi_square(counts: dict, law: dict, total: int) -> tuple[float, int, float]:
    """Pearson's statistic over the law's cells (expectation under 5 pooled), the degrees of freedom and the
    Wilson-Hilferty p-value."""
    import math

    stat, df, pooled_exp, pooled_obs = 0.0, 0, 0.0, 0
    for cell, q in law.items():
        exp, obs = q * total, counts.get(cell, 0)
        if exp < 5:
            pooled_exp += exp
            pooled_obs += obs
            continue
        stat += (obs - exp) ** 2 / exp
        df += 1
    if pooled_exp > 0:
        stat += (pooled_obs - pooled_exp) ** 2 / pooled_exp
        df += 1
    if sum(c for cell, c in counts.items() if cell not in law):
        return math.inf, df - 1, 0.0
    df -= 1
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return stat, df, 0.5 * math.erfc(z / math.sqrt(2))


def _rows_tensor(mesh_device, logits):
    """``[sp, ROWS, vocab]`` fp32 -> the per-chip logits shard ``[1, 1, ROWS, vocab / tp]`` of the sampler."""
    sp, tp = tuple(mesh_device.shape)
    vocab = logits.shape[-1]
    per_chip = vocab // tp
    host = logits.view(sp, ROWS, tp, per_chip).permute(0, 2, 1, 3).reshape(sp, tp, 1, 1, ROWS, per_chip)
    t = ttnn.from_torch(
        host.contiguous(),
        device=mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(0, 1)),
    )
    return ttnn.reshape(t, (1, 1, ROWS, per_chip))


# the sampling settings: (top_k as the plugin sends it: 0 is none, p, temperature)
SETTINGS = {"t0.6-p0.95": (0, 0.95, 0.6), "t1.0-p1.0": (0, 1.0, 1.0), "t1.0-k4-p0.9": (4, 0.9, 1.0)}
CALLS = int(os.environ.get("V41_SAMPLING_CALLS", 400))


@pytest.mark.timeout(int(os.environ.get("V41_SAMPLING_TIMEOUT", 1800)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=0), indirect=True)
def test_v41_spec_match_sampler(mesh_device, device_params):
    """The device sampler's draws against the CPU model of it (``spec_sample.served_probs``) on
    rows of known logits: the histogram over the candidates at each setting (chi-square, p > 1e-3), the match
    rule's acceptance frequency of a draft equal to the mode and to the runner-up (``P(d)`` to 4 sigma), the
    rows of the other ranks drawing the same uniform as rank 0's (the replicated seeds: recorded), and the
    seeded reproducibility (``V41_SAMPLE_SEED``: a second sampler draws the same stream)."""
    from collections import Counter

    from models.demos.deepseek_v3_d_p.tests.v41.spec_sample import served_probs

    sp, tp = tuple(mesh_device.shape)
    vocab = C.VOCAB_SIZE
    torch.manual_seed(0)
    # eight live candidates a row (bf16-exact logits), the rest far below (the sampler's top-32 holds them plus 24
    # negligible fillers); each of rank 0's rows its own candidate ids, the same layout on every rank
    live = [4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]
    ids = torch.stack([torch.randperm(vocab)[: len(live)] for _ in range(ROWS)])  # [ROWS, 8]
    logits = torch.full((sp, ROWS, vocab), -24.0)
    for u in range(ROWS):
        logits[:, u, ids[u]] = torch.tensor(live)
    t = _rows_tensor(mesh_device, logits)
    ccl = V41Collectives(mesh_device)
    pool_like = V41DecodePool.__new__(V41DecodePool)
    V41DecodePool._init_ccl(pool_like, mesh_device)
    sampler = DeviceSampler(mesh_device, C, ccl, pool_like.gather_sp, seed=12345)
    results = {}
    for name, (k, p, temperature) in SETTINGS.items():
        kk = torch.full((sp * ROWS,), max(1, min(k, 32)) if k > 0 else 32, dtype=torch.int32)
        pp = torch.full((sp * ROWS,), p)
        tt = torch.full((sp * ROWS,), 1.0 / temperature)
        inputs = sampler.host_inputs(kk, pp, tt)
        buffers = {name_: ttnn.to_device(v, mesh_device) for name_, v in inputs.items()}
        law = {i: q for i, q in enumerate(q for _, q in served_probs(live + [-24.0] * 24, k, p, temperature))}
        counts, same_as_rank0, total = Counter(), 0, 0
        for _ in range(CALLS):
            got = DeviceSampler.to_host(sampler(ttnn.clone(t), buffers)).view(sp, ROWS)
            for u in range(ROWS):
                where = (ids[u] == int(got[0, u])).nonzero().flatten().tolist()
                counts[where[0] if where else -1] += 1
            same_as_rank0 += int((got[1:] == got[0]).sum())
            total += (sp - 1) * ROWS
        n = CALLS * ROWS
        stat, df, pval = _chi_square(counts, law, n)
        # the match rule: a draft equal to the mode (candidate 0) is accepted with frequency P(mode); the runner-up
        # (candidate 1) with P(runner-up): the "accept" of spec_verify is "the draw equals the draft"
        acc = {d: counts[d] / n for d in (0, 1)}
        sigma = {d: (law.get(d, 0.0) * (1 - law.get(d, 0.0)) / n) ** 0.5 for d in (0, 1)}
        results[name] = {
            "draws": n,
            "chi2": round(stat, 1),
            "df": df,
            "p": pval,
            "hist": {i: counts[i] / n for i in sorted(counts)},
            "law": {i: round(q, 4) for i, q in law.items()},
            "accept_mode": acc[0],
            "accept_runner_up": acc[1],
            "P_mode": law.get(0, 0.0),
            "P_runner_up": law.get(1, 0.0),
            "other_ranks_same_draw": same_as_rank0 / total,
        }
        logger.info(f"spec match sampler {name}: {results[name]}")
        assert counts[-1] == 0, f"{name}: {counts[-1]} draws outside the eight candidates"
        assert pval > 1e-3, f"{name}: histogram off the served law: chi2 {stat:.1f} df {df} p {pval:.3g}"
        for d in (0, 1):
            assert (
                abs(acc[d] - law.get(d, 0.0)) < 4 * sigma[d] + 1e-9
            ), f"{name}: accept of candidate {d} {acc[d]:.4f} vs P {law.get(d, 0.0):.4f}"
    # seeded reproducibility: a second sampler seeded the same draws the same stream
    k, p, temperature = SETTINGS["t1.0-p1.0"]
    inputs = sampler.host_inputs(
        torch.full((sp * ROWS,), 32, dtype=torch.int32), torch.full((sp * ROWS,), p), torch.full((sp * ROWS,), 1.0)
    )
    buffers = {name_: ttnn.to_device(v, mesh_device) for name_, v in inputs.items()}
    streams = []
    for _ in range(2):
        s2 = DeviceSampler(mesh_device, C, ccl, pool_like.gather_sp, seed=777)
        streams.append(torch.stack([DeviceSampler.to_host(s2(ttnn.clone(t), buffers)) for _ in range(8)]))
    assert torch.equal(streams[0], streams[1]), "two samplers seeded 777 drew different streams"
    s3 = DeviceSampler(mesh_device, C, ccl, pool_like.gather_sp, seed=778)
    other = torch.stack([DeviceSampler.to_host(s3(ttnn.clone(t), buffers)) for _ in range(8)])
    results["seeded"] = {"repeat": True, "seed_778_differs": bool(not torch.equal(streams[0], other))}
    logger.info("V41_SAMPLING_RESULT " + json.dumps(results))
    out_path = os.environ.get("V41_SAMPLING_OUT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=1)
