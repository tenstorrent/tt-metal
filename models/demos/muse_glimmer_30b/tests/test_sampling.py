# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device test: the on-device sampler draws from the distribution that tt/sampling.py filtered_probs describes."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingParams, format_sampling_params
from models.demos.muse_glimmer_30b.tt.ccl import default_num_links
from models.demos.muse_glimmer_30b.tt.mesh_utils import dp_submeshes, mesh_test_params
from models.demos.muse_glimmer_30b.tt.model import make_sampling_generator
from models.demos.muse_glimmer_30b.tt.model_config import DEFAULT_CKPT_DIR, MuseGlimmerModelArgs
from models.demos.muse_glimmer_30b.tt.sampling import filtered_probs, lane_seeds

CKPT = os.environ.get("HF_MODEL") or DEFAULT_CKPT_DIR

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(CKPT, "config.json")), reason="checkpoint not available"
)

_mesh_params = mesh_test_params()

LANES = 32
DRAWS = 1500
NUM_HEAD = 48
TV_MAX = 0.05  # ttnn.sampling uses bf16 probabilities and approximate exp; measured TV about 0.03 at N=96000
COINC_TOL = 0.03


def _make_row(args):
    g = torch.Generator().manual_seed(0)
    row = torch.randn(args.padded_vocab_size, generator=g) * 1.0
    vl = args.vocab_local
    ids = []
    for d in range(4):  # spread over the four shards; the last shard also gets 4 ids near vocab_size
        base = d * vl
        n = 12 if d < 3 else 8
        step = max(vl // 16, 1)
        ids += [base + 100 + step * i for i in range(n if d < 3 else 4)]
    ids += [args.vocab_size - 1 - 40 * i for i in range(4)]
    ids = list(dict.fromkeys(ids))
    cand = 0
    while len(ids) < NUM_HEAD:  # top up with ids spread over shards
        c = (cand * 7919 + 13) % args.vocab_size
        if c not in ids:
            ids.append(c)
        cand += 1
    ids = ids[:NUM_HEAD]
    assert len({i // vl for i in ids}) == 4 and sum(i >= args.vocab_size - 200 for i in ids) >= 4
    for j, i in enumerate(ids):
        row[i] = 6.0 + 0.15 * j
    row[args.vocab_size :] = 0.0
    return row


def _params(T, k, p, base=None):
    seed = lane_seeds(base) if base is not None else None
    return format_sampling_params(SamplingParams(temperature=[T] * LANES, top_k=k, top_p=p, seed=seed), LANES)


def _load(sampler, params):
    sampler.reset_sampling_params(params)
    if params.seed is not None:
        sampler.seed_manager.reset_seed(params.seed, list(range(LANES)))


def _draw(sampler, logits_tt):
    sampler.seed_manager.get_new_values()
    out = sampler.sample(logits=logits_tt, enable_trace=False)
    out = out[0] if isinstance(out, tuple) else out
    t = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[:LANES].to(torch.int64)
    ttnn.deallocate(out)
    return t


def _run(sampler, logits_tt, n):
    return torch.stack([_draw(sampler, logits_tt) for _ in range(n)])  # [n, LANES]


@pytest.mark.timeout(1800)
@_mesh_params
def test_device_sampling_matches_filtered_probs(mesh_device, tp):
    if tp != 4:
        pytest.skip("test uses a 1x4 submesh")
    sm = dp_submeshes(mesh_device, 1, 4)[0]
    args = MuseGlimmerModelArgs.from_checkpoint(mesh_device=sm, max_batch_size=1)
    sampler = make_sampling_generator(sm, args, default_num_links(sm))
    assert sampler is not None

    row = _make_row(args)
    vs = args.vocab_size
    logits_tt = ttnn.from_torch(
        row.reshape(1, 1, 1, -1).repeat(1, 1, LANES, 1),
        device=sm,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(sm, dim=-1),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    cases = {
        "A": (1.0, 0.95, 32),
        "B": (0.7, 1.0, 10),
        "C": (1.0, 0.5, 32),
        "D": (2.0, 0.9, 32),
    }
    failures = []
    for name, (T, p, k) in cases.items():
        _load(sampler, _params(T, k, p, base=0))
        tok = _run(sampler, logits_tt, DRAWS)  # [DRAWS, LANES]
        toks = tok.reshape(-1)
        probs, ids = filtered_probs(row[None, :vs], torch.arange(vs)[None], T, p, k)
        ref = torch.zeros(vs)
        ref[ids[0]] = probs[0]
        n_kept = int((probs[0] > 0).sum())
        # host rank (1-based) over the top-k order before the top-p cut; ranks beyond k count as k + 1
        rank_of = torch.full((args.padded_vocab_size,), k + 1)
        for r, i in enumerate(ids[0].tolist()):
            rank_of[i] = r + 1
        max_rank = int(rank_of[toks].max())
        hist = torch.bincount(toks, minlength=args.padded_vocab_size)[:vs].float()
        tv = 0.5 * (hist / hist.sum() - ref).abs().sum().item()
        distinct = int((hist > 0).sum())
        logger.info(
            f"case {name} T={T} top_p={p} k={k}: TV={tv:.4f} n_kept={n_kept} max_rank={max_rank} distinct={distinct}"
        )
        ok = tv < TV_MAX
        # the device top-p cut can keep one more token at the boundary
        ok &= max_rank <= n_kept + 1
        ok &= not (n_kept > 1 and distinct <= 1)
        if name == "A":
            sp2 = float((ref**2).sum())
            pairs = [(tok[:, i] == tok[:, j]).float().mean().item() for i in range(LANES) for j in range(i + 1, LANES)]
            same = sum(pairs) / len(pairs)
            cross = (tok[:-1, :-1] == tok[1:, 1:]).float().mean().item()
            logger.info(f"case A sum(p^2)={sp2:.4f} same-step coincidence={same:.4f} lag1 cross-lane={cross:.4f}")
            ok &= abs(same - sp2) <= COINC_TOL and abs(cross - sp2) <= COINC_TOL
        if not ok:
            failures.append(name)

    # greedy: every lane is the argmax
    _load(sampler, format_sampling_params(SamplingParams(temperature=0.0, top_k=1, top_p=1.0), LANES))
    greedy = _draw(sampler, logits_tt)
    assert (greedy == int(row[:vs].argmax())).all(), f"greedy lanes {greedy.tolist()} vs {int(row[:vs].argmax())}"

    # seeds
    runs = {}
    for label, base in (("a", 0), ("b", 0), ("c", 1000)):
        _load(sampler, _params(1.0, 32, 0.95, base=base))
        runs[label] = _run(sampler, logits_tt, 20)
    assert torch.equal(runs["a"], runs["b"]), "same seed gave different tokens"
    assert not torch.equal(runs["a"], runs["c"]), "different seed gave identical tokens"

    ttnn.deallocate(logits_tt)
    assert not failures, f"distribution mismatch in cases {failures}"
