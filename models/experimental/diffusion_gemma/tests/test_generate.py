# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the block-autoregressive multi-canvas loop (#47464)."""

import torch

from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.reference.generate import generate_blocks


def _gen(seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cfg(canvas_length):
    return DiffusionConfig(
        canvas_length=canvas_length, max_denoise_steps=6, entropy_stop_threshold=0.1, stable_steps_to_halt=1
    )


class _PrefixDependentModel:
    """Peaked logits whose target token depends on how long the prefix is.

    target = prefix_len // canvas_len, so each committed block is predictable and
    we can verify the prefix actually grows (commit-append) between blocks.
    """

    def __init__(self, batch, canvas_len, vocab):
        self.batch, self.canvas_len, self.vocab = batch, canvas_len, vocab

    def __call__(self, prefix, canvas, step):
        target = (prefix.shape[1] // self.canvas_len) % self.vocab
        logits = torch.full((self.batch, self.canvas_len, self.vocab), -1e4)
        logits[..., target] = 1e4
        return logits


def test_generates_num_blocks_times_canvas_tokens():
    batch, canvas_len, vocab, blocks = 1, 4, 16, 3
    prompt = torch.zeros(batch, 2 * canvas_len, dtype=torch.long)  # prompt_len = 2*canvas_len
    model = _PrefixDependentModel(batch, canvas_len, vocab)

    out = generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(1))

    assert out.generated.shape == (batch, blocks * canvas_len)
    assert out.prompt_len == 2 * canvas_len
    assert len(out.trajectories) == blocks


def test_prefix_grows_so_committed_targets_advance():
    batch, canvas_len, vocab, blocks = 1, 4, 16, 3
    prompt = torch.zeros(batch, 2 * canvas_len, dtype=torch.long)  # prompt_len//canvas_len == 2
    model = _PrefixDependentModel(batch, canvas_len, vocab)

    out = generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(2))

    # block b sees prefix_len = (2 + b) * canvas_len -> target token = 2 + b
    for b in range(blocks):
        block_tokens = out.generated[:, b * canvas_len : (b + 1) * canvas_len]
        assert torch.all(block_tokens == 2 + b)


def test_generation_is_deterministic_under_fixed_generator():
    batch, canvas_len, vocab, blocks = 2, 4, 16, 2
    prompt = torch.zeros(batch, canvas_len, dtype=torch.long)
    model = _PrefixDependentModel(batch, canvas_len, vocab)

    a = generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(3))
    b = generate_blocks(model, prompt, blocks, _cfg(canvas_len), vocab, generator=_gen(3))
    assert torch.equal(a.generated, b.generated)
