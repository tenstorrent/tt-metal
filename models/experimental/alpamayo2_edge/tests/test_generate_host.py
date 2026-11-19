# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side CoT generation loop against a scripted language model: stop logic, anchor caching, sampling."""

import pytest
import torch

import ttnn
from models.experimental.alpamayo2_edge.reference.inputs import ModelInputs, prepare_sample_inputs, synthetic_sample
from models.experimental.alpamayo2_edge.reference.model import process_logits
from models.experimental.alpamayo2_edge.tt.generate import CotResult, generate_cot, sample

PROMPT = 10
ROPE_DELTA = -3
# Plain text token ids; none of them is a trajectory, EOS or anchor token.
TEXT = [1000, 2001, 3002, 4003, 5004, 6005, 7006, 8007]
# The prompt has no image tokens; the fake LM never reads the embeddings.
NO_IMAGES = torch.zeros(0, 2048)


def _inputs(prompt: int = PROMPT, rope_delta: int = ROPE_DELTA) -> ModelInputs:
    positions = torch.arange(prompt) + rope_delta
    return ModelInputs(
        input_ids=torch.full((1, prompt), 7, dtype=torch.int64),
        attention_mask=torch.ones(1, prompt, dtype=torch.int64),
        mm_token_type_ids=torch.zeros(1, prompt, dtype=torch.int64),
        patches=torch.zeros(0, 768),
        image_grid_thw=torch.zeros(0, 3, dtype=torch.int64),
        position_ids=positions.expand(3, 1, prompt).clone(),
        rope_deltas=torch.tensor([[rope_delta]]),
        ego_history_xyz=torch.zeros(1, 20, 3),
        ego_history_rot=torch.zeros(1, 20, 3, 3),
    )


class FakeCache:
    def __init__(self, *, size: int = 512, batch_size: int = 1) -> None:
        self.size = size
        self.batch_size = batch_size
        self.position = 0
        self.deallocated = False

    def repeat_to(self, batch: int) -> "FakeCache":
        out = FakeCache(size=self.size, batch_size=batch)
        out.position = self.position
        return out

    def deallocate(self) -> None:
        self.deallocated = True


class FakeLM:
    """Returns scripted logits: call i (the prefill is call 0) peaks at `script[i]`, a token per row."""

    device = None

    def __init__(self, vocab: int, script: list[list[int]]) -> None:
        self.vocab = vocab
        self.script = script
        self.decoded: list[list[int]] = []
        self.rope_offsets: list[list[int]] = []
        self.returned: list[torch.Tensor] = []

    def _logits(self, rows: int) -> ttnn.Tensor:
        call = len(self.returned)
        peaks = self.script[min(call, len(self.script) - 1)]
        peaks = peaks * rows if len(peaks) == 1 else peaks
        logits = torch.randn(len(peaks), self.vocab, generator=torch.Generator().manual_seed(call))
        logits[torch.arange(len(peaks)), torch.tensor(peaks)] += 1e4
        logits = logits.bfloat16()
        self.returned.append(logits.float())
        return ttnn.from_torch(logits, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def prefill(self, tokens, *, positions, vision_embeds, vision_mask, cache):
        assert tuple(tokens.shape) == (1, PROMPT) and cache.batch_size == 1
        cache.position = tokens.shape[1]
        return self._logits(1)

    def decode_step(self, tokens, *, cache, rope_offset):
        ids = ttnn.to_torch(tokens).reshape(-1)[: cache.batch_size]
        self.decoded.append(ids.tolist())
        self.rope_offsets.append(ttnn.to_torch(rope_offset).reshape(-1).tolist())
        cache.position += 1
        return self._logits(cache.batch_size)


def _generate(cfg, lm, **kwargs) -> CotResult:
    return generate_cot(lm, _inputs(), NO_IMAGES, tokens=cfg.tokens, cache=FakeCache(), **kwargs)


def test_stop_one_step_after_anchor(cfg):
    anchor = cfg.tokens.future_start
    # The prefill's logits draw TEXT[0]; the anchor is drawn on step 5.
    lm = FakeLM(cfg.text.vocab, [[t] for t in TEXT[:5]] + [[anchor], [TEXT[6]]])
    cot = _generate(cfg, lm, seed=0, return_logits=True)

    assert cot.tokens.tolist() == [TEXT[:5] + [anchor]]
    assert len(lm.decoded) == 6, "one decode step per token, the anchor included"
    assert lm.decoded == [[t] for t in cot.tokens[0].tolist()]
    assert lm.rope_offsets == [[ROPE_DELTA]] * 6
    assert cot.offsets.tolist() == [PROMPT + 6] and cot.prefix_length == PROMPT + 6
    assert cot.cache.position == PROMPT + 6
    assert cot.logits.shape == (1, 6, cfg.text.vocab)
    assert torch.equal(cot.logits[0], torch.cat(lm.returned[:6]))


def test_anchor_on_last_step_is_cached(cfg):
    anchor = cfg.tokens.future_start
    lm = FakeLM(cfg.text.vocab, [[t] for t in TEXT[:3]] + [[anchor]])
    cot = _generate(cfg, lm, seed=0, max_new_tokens=4)

    assert cot.tokens.tolist() == [TEXT[:3] + [anchor]]
    # Three steps inside the token budget, then the anchor drawn on the last allowed step is forwarded.
    assert lm.decoded == [[TEXT[0]], [TEXT[1]], [TEXT[2]], [anchor]]
    assert cot.offsets.tolist() == [PROMPT + 4] and cot.prefix_length == PROMPT + 4
    assert cot.cache.position == PROMPT + 4
    assert cot.logits is None


def test_rows_wait_for_the_latest_anchor(cfg):
    anchor = cfg.tokens.future_start
    forced = torch.tensor([[TEXT[0], anchor, TEXT[2], TEXT[3]], [TEXT[4], TEXT[5], TEXT[6], anchor]])
    lm = FakeLM(cfg.text.vocab, [[TEXT[7]]])
    prefill_cache = FakeCache()
    cot = generate_cot(
        lm, _inputs(), NO_IMAGES, tokens=cfg.tokens, cache=prefill_cache, num_samples=2, forced=forced, max_new_tokens=4
    )
    assert prefill_cache.deallocated and not cot.cache.deallocated

    assert torch.equal(cot.tokens, forced)
    assert cot.offsets.tolist() == [PROMPT + 2, PROMPT + 4] and cot.prefix_length == PROMPT + 4
    assert lm.decoded == forced.T.tolist()
    assert lm.rope_offsets == [[ROPE_DELTA, ROPE_DELTA]] * 4
    assert cot.cache.batch_size == 2 and cot.cache.position == PROMPT + 4


def test_no_anchor_raises(cfg, expect_error):
    lm = FakeLM(cfg.text.vocab, [[t] for t in TEXT])
    with expect_error(ValueError, r"No <traj_future_start> token found in generated rows \[0\]"):
        _generate(cfg, lm, seed=0, max_new_tokens=4)


def test_forced_cot_without_anchor_raises(cfg, expect_error):
    lm = FakeLM(cfg.text.vocab, [[TEXT[0]]])
    with expect_error(ValueError, r"No <traj_future_start> token found in generated rows \[0\]"):
        _generate(cfg, lm, forced=torch.tensor([TEXT[:4]]), max_new_tokens=256)


@pytest.mark.parametrize("num_samples", [0, 9])
def test_unsupported_sample_counts_raise(cfg, num_samples, expect_error):
    lm = FakeLM(cfg.text.vocab, [[TEXT[0]]])
    with expect_error(ValueError, "num_samples"):
        _generate(cfg, lm, num_samples=num_samples)


def test_cache_too_small_raises(cfg, expect_error):
    lm = FakeLM(cfg.text.vocab, [[TEXT[0]]])
    with expect_error(ValueError, "cannot hold"):
        generate_cot(
            lm, _inputs(), NO_IMAGES, tokens=cfg.tokens, cache=FakeCache(size=PROMPT + 100), max_new_tokens=256
        )


def test_image_token_mismatch_raises(cfg, expect_error):
    lm = FakeLM(cfg.text.vocab, [[TEXT[0]]])
    with expect_error(ValueError, "0 image tokens for 4 vision embeddings"):
        generate_cot(lm, _inputs(), torch.zeros(4, 2048), tokens=cfg.tokens, cache=FakeCache())
    assert not lm.returned, "the check runs before the prefill"


def test_process_logits_bans_without_warping(cfg):
    tok = cfg.tokens
    logits = torch.randn(2, cfg.text.vocab, generator=torch.Generator().manual_seed(0))
    raw = logits.clone()
    masked = process_logits(logits, tok, temperature=1.0, top_p=1.0)

    assert torch.equal(logits, raw), "the raw logits must not be modified"
    banned = torch.zeros(cfg.text.vocab, dtype=torch.bool)
    banned[tok.history_id0 : tok.history_id0 + 1000] = True
    banned[tok.future_id0 : tok.future_id0 + 3000] = True
    banned[tok.eos] = tok.eos != tok.future_start
    assert tok.history_vocab == 1000 and tok.future_vocab == 3000
    assert torch.isneginf(masked[:, banned]).all()
    assert torch.equal(masked[:, ~banned], logits[:, ~banned])
    assert masked[0, tok.future_start] == logits[0, tok.future_start]
    assert torch.equal(masked[:, TEXT], logits[:, TEXT])


def test_sample_draws_like_the_reference(cfg):
    logits = torch.randn(3, cfg.text.vocab, generator=torch.Generator().manual_seed(1)) * 4
    ours = sample(logits, tokens=cfg.tokens, generator=torch.Generator().manual_seed(5))
    probs = torch.softmax(process_logits(logits, cfg.tokens), dim=-1)
    theirs = torch.multinomial(probs, 1, generator=torch.Generator().manual_seed(5)).squeeze(1)
    assert ours.dtype == torch.int64 and torch.equal(ours, theirs)


def test_decode_rope_axes_equal(cfg):
    """The decode's 1D rope at cache position + rope_deltas continues every M-RoPE axis of the prompt."""
    inputs = prepare_sample_inputs(cfg, synthetic_sample(0))
    seq = inputs.input_ids.shape[1]
    last = inputs.position_ids[:, :, -1]
    assert torch.equal(last + 1, (seq + inputs.rope_deltas[:, 0]).expand(3, -1))
