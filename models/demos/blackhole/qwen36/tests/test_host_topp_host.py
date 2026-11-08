# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of QWEN36_JOIN_TOPP_CANDIDATES (tt/host_topp.py) against vLLM's own apply_top_k_top_p and the vLLM TT plugin's
host sampler (both must be importable: the serving python env).

* top_p >= 1 with top-k disabled: the candidate path is the whole vocabulary: the sampled token equals the full path's
  for the same RNG state, bit for bit.
* 0 < top_p < 1 with top-k disabled: whenever the candidates are used, they keep exactly vLLM's kept set, with the same
  probabilities (to fp32 rounding); otherwise the full path runs.
* Timing of the plugin sampler with / without the hook on a [1, 248320] row (printed).

  R6_TOPP_LOGITS=<dir of greedy_L*.npy prefill logits> adds real model logits to the random ones.
  PYTHONPATH=<tree> python -m pytest models/demos/blackhole/qwen36/tests/test_host_topp_host.py -q -s
"""
import glob
import os
import time

import numpy as np
import pytest
import torch

try:
    from models.demos.blackhole.qwen36.tt import host_topp
except ImportError:  # pod run from a scratch dir next to host_topp.py
    import host_topp

vllm_ops = pytest.importorskip("vllm.v1.sample.ops.topk_topp_sampler")
hs = pytest.importorskip("vllm_tt_plugin.host_sampler")

V = 248320


def _logit_rows():
    g = torch.Generator().manual_seed(1)
    rows = []
    for scale in (1.0, 3.0, 8.0, 15.0):  # flat .. peaked
        rows.append(torch.randn(V, generator=g) * scale)
    zipf = -torch.log(torch.arange(1, V + 1, dtype=torch.float64)) * 1.2
    rows.append(zipf[torch.randperm(V, generator=g)].float() * 4.0)
    d = os.environ.get("R6_TOPP_LOGITS")
    if d:
        for f in sorted(glob.glob(os.path.join(d, "**", "greedy_L*.npy"), recursive=True))[:12]:
            x = torch.from_numpy(np.load(f).astype(np.float32)).reshape(-1)[:V]
            for t in (0.6, 1.0):
                rows.append(x / t)
    return rows


def _vllm_kept(logits, p):
    out = vllm_ops.apply_top_k_top_p_pytorch(
        logits.clone(), torch.full((logits.shape[0],), V), torch.tensor(p, dtype=torch.float32), allow_cpu_sync=True
    )
    return torch.isfinite(out)


@pytest.mark.parametrize("p", [0.5, 0.8, 0.9, 0.95, 0.99])
def test_topp_kept_set_matches_vllm(p):
    used = fell = 0
    for row in _logit_rows():
        logits = row.reshape(1, -1).float()
        k = torch.full((1,), V)
        pt = torch.tensor([p], dtype=torch.float32)
        cand = host_topp.topp_candidates(logits, k, pt)
        if cand is None:
            fell += 1
            continue
        used += 1
        vals, ids = cand
        kept_c = set(ids[torch.isfinite(vals)].tolist())
        kept_v = set(torch.nonzero(_vllm_kept(logits, [p])[0]).reshape(-1).tolist())
        assert kept_c == kept_v, (p, len(kept_c), len(kept_v))
        # probabilities over the kept set: candidate softmax == full-path softmax (fp32 rounding)
        pc = vals.softmax(-1, dtype=torch.float32)[torch.isfinite(vals)]
        full = logits.clone().masked_fill(~_vllm_kept(logits, [p]), float("-inf")).softmax(-1, dtype=torch.float32)
        pv = full[0, ids[torch.isfinite(vals)]]
        assert torch.allclose(pc, pv, rtol=1e-5, atol=1e-9)
    print(f"[topp] p={p}: candidates used {used}, full path {fell}")
    assert used > 0


def test_noop_filter_identical_tokens():
    hook = host_topp.topp_candidates
    for row in _logit_rows()[:4]:
        logits = row.reshape(1, -1).float()
        k, p = torch.full((1,), V), torch.ones(1)
        for seed in range(5):
            torch.manual_seed(seed)
            vals, ids = hook(logits, k, p)
            probs = vals.softmax(dim=-1, dtype=torch.float32)
            got = ids.gather(1, vllm_ops.random_sample(probs, {}).unsqueeze(1)).squeeze(1)
            torch.manual_seed(seed)
            ref_logits = vllm_ops.apply_top_k_top_p_pytorch(logits.clone(), k, p, allow_cpu_sync=True)
            ref = vllm_ops.random_sample(ref_logits.softmax(dim=-1, dtype=torch.float32), {})
            assert int(got) == int(ref)


def test_declines():
    logits = torch.randn(2, V)
    f = host_topp.topp_candidates
    assert f(logits, torch.tensor([20, V]), torch.tensor([0.9, 0.9])) is None  # a row with top-k
    assert f(logits, None, None) is None
    assert f(logits, torch.full((2,), V), torch.tensor([1.0, 0.9])) is None  # mixed no-op / top-p rows
    assert f(logits, torch.full((2,), V), torch.tensor([0.0, 0.9])) is None
    assert f(logits.to(torch.float64)[:0], None, torch.ones(0)) is None
    flat = torch.zeros(1, V)  # all ties: the cut always falls on a tie
    assert f(flat, torch.full((1,), V), torch.tensor([0.5])) is None


def test_plugin_hook_and_timing():
    assert host_topp.install_plugin_topp_candidates()
    s = hs.TTTopKTopPSampler()
    orig = hs.TTTopKTopPSampler._top_k_candidates._qwen36_orig
    k, p = torch.full((1,), V), torch.tensor([0.95], dtype=torch.float32)
    # a row the candidates serve (real prefill logits at temperature 1 when R6_TOPP_LOGITS is set): the timing of the
    # candidate path itself, not of a decline + full sort
    rows = [r.reshape(1, -1).float() for r in _logit_rows()[5:]] + [(torch.randn(1, V) * 15.0).float()]
    logits = next(r for r in rows if host_topp.topp_candidates(r, k, p) is not None)

    def run(hooked, reps=15):
        ts = []
        for _ in range(reps):
            x = logits.clone()
            t0 = time.perf_counter()
            if hooked:
                s.forward_native(x, {}, k, p)
            else:
                cand = orig(s, x, k, p)
                assert cand is None
                hs.TopKTopPSampler.forward_native(s, x, {}, k, p)
            ts.append(1e3 * (time.perf_counter() - t0))
        return sorted(ts)[len(ts) // 2]

    full_ms, hook_ms = run(False), run(True)
    one = torch.ones(1)
    t0 = time.perf_counter()
    for _ in range(15):
        s.forward_native(logits.clone(), {}, k, one)
    noop_ms = 1e3 * (time.perf_counter() - t0) / 15
    print(f"[topp] plugin sampler [1, {V}] top_k off: top_p=0.95 full sort {full_ms:.2f} ms -> candidates {hook_ms:.2f} ms; "
          f"top_p=1.0 {noop_ms:.2f} ms (threads {torch.get_num_threads()})")
