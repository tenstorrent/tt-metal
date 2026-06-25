# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the discrete-diffusion sampling reference (#47463/#47468).

Pure torch — no checkpoint / ttnn / hardware. These pin the exact semantics the
device path must match, especially the entropy-budget acceptance scatter-back.
"""

import pytest
import torch

from models.experimental.diffusion_gemma.reference import sampling as S


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_temperature_schedule_endpoints_and_monotone():
    # HF reversed-step formula: step 0 == t_max (0.8); last step == t_min + (t_max-t_min)/N,
    # NOT exactly t_min (cur_step bottoms out at 1, not 0).
    assert S.temperature_at_step(0, 48, 0.8, 0.4) == pytest.approx(0.8)
    assert S.temperature_at_step(47, 48, 0.8, 0.4) == pytest.approx(0.4 + 0.4 * (1 / 48))
    assert S.temperature_at_step(24, 48, 0.8, 0.4) == pytest.approx(0.6)  # cur_step=24 -> midpoint
    ts = [S.temperature_at_step(i, 48, 0.8, 0.4) for i in range(48)]
    assert all(ts[i] >= ts[i + 1] for i in range(47))  # monotonically decreasing
    assert 0.4 < ts[24] < 0.8


def test_token_entropy_uniform_and_peaked():
    vocab = 64
    uniform = torch.zeros(1, 4, vocab)  # equal logits -> uniform softmax
    h = S.token_entropy(uniform)
    expected = torch.log(torch.tensor(float(vocab)))
    assert torch.allclose(h, torch.full_like(h, expected), atol=1e-5)

    peaked = torch.full((1, 4, vocab), -1e4)
    peaked[..., 0] = 1e4
    assert torch.all(S.token_entropy(peaked) < 1e-3)


def test_gumbel_max_zero_noise_is_argmax():
    logits = torch.randn(2, 8, 50, generator=_gen(1))
    out = S.gumbel_max_sample(logits, temperature=0.7, noise=torch.zeros_like(logits))
    assert torch.equal(out, logits.argmax(dim=-1))


def test_entropy_budget_accept_extremes_and_monotone():
    entropy = torch.tensor([[0.5, 0.1, 0.9, 0.2, 0.7]])
    assert S.entropy_budget_accept(entropy, budget=1e9).all()  # huge budget -> all

    acc = S.entropy_budget_accept(entropy, budget=0.0, min_accept=1)
    assert acc.sum().item() == 1 and bool(acc[0, 1])  # only the lowest-entropy pos (idx 1)

    prev = torch.zeros_like(entropy, dtype=torch.bool)
    for b in [0.0, 0.15, 0.35, 0.8, 1.5, 3.0]:  # increasing budget never un-accepts
        cur = S.entropy_budget_accept(entropy, budget=b, min_accept=1)
        assert torch.equal(cur | prev, cur)
        prev = cur


def test_entropy_budget_accept_exclusive_prefix_cutoff():
    # ascending order by value: idx 1(0.1), 3(0.2), 0(0.5), 4(0.7), 2(0.9)
    # EXCLUSIVE prefix (sum of strictly-more-confident) per sorted pos:
    #   idx1: 0.0 | idx3: 0.1 | idx0: 0.3 | idx4: 0.8 | idx2: 1.5
    entropy = torch.tensor([[0.5, 0.1, 0.9, 0.2, 0.7]])
    # budget 0.35: accept idx1(0.0), idx3(0.1), idx0(0.3) (<=0.35); reject idx4(0.8), idx2(1.5)
    acc = S.entropy_budget_accept(entropy, budget=0.35, min_accept=1)
    assert torch.equal(acc, torch.tensor([[True, True, False, True, False]]))


def test_acceptance_scatter_back_inverse_permutation():
    # The scatter-back the device path must replicate (#47463): accept decisions
    # taken in sorted-by-confidence order map to ORIGINAL canvas positions. Mirror
    # HF EntropyBoundSampler.accept_canvas exactly (exclusive-prefix cutoff).
    entropy = torch.rand(3, 17, generator=_gen(4))
    budget = 0.9
    acc = S.entropy_budget_accept(entropy, budget=budget, min_accept=1)

    sorted_e, idx = torch.sort(entropy, dim=-1)
    cum = torch.cumsum(sorted_e, dim=-1)
    accept_sorted = (cum - sorted_e) <= budget  # exclusive prefix
    ref = torch.zeros_like(entropy, dtype=torch.bool)
    for r in range(entropy.shape[0]):
        for c in range(entropy.shape[1]):
            ref[r, idx[r, c]] = accept_sorted[r, c]
    assert torch.equal(acc, ref)


def test_sample_canvas_multinomial_matches_argmax_when_peaked():
    # multinomial(softmax) of near-one-hot logits returns the peak token id.
    vocab = 50
    peaked = torch.full((2, 8, vocab), -1e4)
    peaked[..., 13] = 1e4
    out = S.sample_canvas(peaked, temperature=0.7, generator=_gen(11))
    assert out.shape == (2, 8)
    assert torch.equal(out, torch.full((2, 8), 13))
    # in-range for arbitrary logits
    rand = S.sample_canvas(torch.randn(1, 5, vocab, generator=_gen(12)), generator=_gen(13))
    assert int(rand.min()) >= 0 and int(rand.max()) < vocab


def test_renoise_keeps_accepted_replaces_rejected():
    vocab = 100
    tokens = torch.arange(5).view(1, 5)
    accept = torch.tensor([[True, False, True, False, True]])
    out = S.renoise(tokens, accept, vocab, noise_tokens=torch.full((1, 5), 99))
    assert torch.equal(out, torch.tensor([[0, 99, 2, 99, 4]]))

    out2 = S.renoise(tokens, accept, vocab, generator=_gen(5))  # random renoise in range
    assert int(out2.min()) >= 0 and int(out2.max()) < vocab
    assert out2[0, 0] == 0 and out2[0, 2] == 2 and out2[0, 4] == 4


def test_decision_dtype_red_lines_bf16_safe():
    """#47468 acceptance red lines for the diffusion DECISIONS under dtype.

    bf16 is the floor for the decision-critical ops: the entropy-budget accept mask
    and the clean-argmax commit must barely move at bf16, while bfp8 is NOT safe for
    them (entropy PCC ~0.74 measured on device -> accept flips; ttnn.argmax rejects
    bfp8). These bars are the harness's value: a regression that pushes decisions
    into bfp8, or otherwise perturbs them, trips here.
    """

    def bf16(x):
        return x.to(torch.bfloat16).float()

    # accept-mask flip rate at bf16 vs fp32, across budgets + varied entropy
    acc_flips = acc_tot = 0
    for seed in range(8):
        logits = torch.randn(1, 256, 2048, generator=_gen(seed)) * torch.linspace(0.3, 5, 256).view(1, 256, 1)
        for bound in [0.05, 0.1, 0.5]:
            ref = S.entropy_budget_accept(S.token_entropy(logits), bound, min_accept=0)
            got = S.entropy_budget_accept(S.token_entropy(bf16(logits)), bound, min_accept=0)
            acc_flips += int((ref != got).sum())
            acc_tot += ref.numel()
    acc_rate = acc_flips / acc_tot
    assert acc_rate <= 0.005, f"bf16 accept-mask flip rate {acc_rate:.4%} > 0.5% red line (measured ~0.13%)"

    # clean-argmax commit flip rate at bf16 vs fp32 (near-max ties may flip; bound generously)
    arg_flips = arg_tot = 0
    for seed in range(8):
        logits = torch.randn(1, 256, 2048, generator=_gen(100 + seed))
        arg_flips += int((logits.argmax(-1) != bf16(logits).argmax(-1)).sum())
        arg_tot += logits.shape[1]
    arg_rate = arg_flips / arg_tot
    assert arg_rate <= 0.03, f"bf16 commit-argmax flip rate {arg_rate:.4%} > 3% red line (measured ~1.3%)"


def test_random_canvas_in_range():
    canvas = S.random_canvas((2, 256), 262144, generator=_gen(7))
    assert canvas.shape == (2, 256)
    assert int(canvas.min()) >= 0 and int(canvas.max()) < 262144


def test_denoise_step_shapes_commit_argmax():
    batch, length, vocab = 2, 16, 64
    logits = torch.randn(batch, length, vocab, generator=_gen(6))
    res = S.denoise_step(
        logits,
        temperature=0.6,
        entropy_budget=0.5,
        vocab_size=vocab,
        gumbel_noise=torch.zeros_like(logits),  # -> sampled == argmax
    )
    assert res.canvas.shape == (batch, length)
    assert res.accept_mask.shape == (batch, length)
    assert res.entropy.shape == (batch, length)
    assert torch.equal(res.argmax, logits.argmax(dim=-1))  # commit value = clean argmax
    assert torch.equal(res.sampled, logits.argmax(dim=-1))  # zero noise
    # accepted canvas positions hold the (clean argmax) sample
    assert torch.equal(res.canvas[res.accept_mask], res.argmax[res.accept_mask])


def test_is_converged():
    toks = torch.zeros(1, 8, dtype=torch.long)
    assert S.is_converged(toks, toks.clone(), torch.full((1, 8), 0.01), entropy_threshold=0.1)
    assert not S.is_converged(toks, toks.clone(), torch.full((1, 8), 5.0), entropy_threshold=0.1)
    assert not S.is_converged(toks, toks + 1, torch.full((1, 8), 0.01), entropy_threshold=0.1)
