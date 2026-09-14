# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Speculative-decoding accept rate of the DSpark drafter against a target greedy.

Two checks:

* A weight-matched clone of the drafter (same hiddens, same anchor) must accept the
  full drafted prefix — this is the DSpark sampler vs a target that *is* DSpark.
* After a short distill of DSpark onto a tiny causal LM (shared embed / LM head,
  last-3-layer hiddens as context), first-token accept on held-out steps must be
  at least 90%.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from models.experimental.deepseek_v4_flash.dspark import (
    DSparkConfig,
    DSparkModel,
    speculative_accept_rate,
)


class _TinyCausalLM(nn.Module):
    """3-layer causal LM whose last-layer stack is the DSpark fusion input."""

    def __init__(self, hidden: int, vocab: int, n_layers: int = 3, n_heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=n_heads,
                    dim_feedforward=hidden * 2,
                    dropout=0.0,
                    batch_first=True,
                    activation="gelu",
                )
                for _ in range(n_layers)
            ]
        )
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward_hiddens(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq = ids.shape[1]
        causal = torch.triu(torch.full((seq, seq), float("-inf"), device=ids.device), diagonal=1)
        x = self.embed(ids)
        hs = []
        for layer in self.layers:
            x = layer(x, src_mask=causal)
            hs.append(x)
        stacked = torch.stack(hs[-3:], dim=2)
        return self.lm_head(x), stacked


@torch.no_grad()
def test_dspark_accept_rate_matched_target_exceeds_90_percent():
    """Drafter and target are the same DSpark weights: every drafted token is accepted."""
    torch.manual_seed(0)
    teacher = DSparkModel(DSparkConfig.tiny()).eval()
    drafter = DSparkModel(DSparkConfig.tiny()).eval()
    drafter.load_state_dict(teacher.state_dict())

    drafts, gold = [], []
    cfg = teacher.config
    for _ in range(32):
        hiddens = torch.randn(2, 6, cfg.num_target_layers, cfg.hidden_size)
        anchors = torch.randint(0, cfg.vocab_size - 1, (2,))
        drafts.append(drafter(hiddens, anchors).draft_ids)
        gold.append(teacher(hiddens, anchors).draft_ids)
    stats = speculative_accept_rate(torch.cat(drafts), torch.cat(gold))
    logger.info(
        f"matched-target accept: first={stats['first_token']:.3f} "
        f"prefix={stats['mean_prefix_frac']:.3f} mean_len={stats['mean_accept_len']:.2f}/{stats['gamma']:.0f}"
    )
    assert stats["first_token"] >= 0.90
    assert stats["mean_prefix_frac"] >= 0.90


def test_dspark_first_token_accept_rate_distilled_exceeds_90_percent():
    """DSpark distilled on a tiny causal LM: first-token accept ≥ 90% on held-out steps."""
    torch.manual_seed(1)
    hidden, vocab, seq = 32, 64, 8
    cfg = DSparkConfig.tiny(
        hidden_size=hidden,
        vocab_size=vocab,
        dspark_noise_token_id=vocab - 1,
        dspark_block_size=1,
        num_attention_heads=4,
        head_dim=8,
        intermediate_size=64,
        dspark_markov_rank=8,
    )
    target = _TinyCausalLM(hidden, vocab).eval()
    drafter = DSparkModel(cfg)
    drafter.share_from_target(target.embed, target.lm_head)
    opt = torch.optim.Adam([p for p in drafter.parameters() if p.requires_grad], lr=3e-3)

    def _batch(n: int):
        ids = torch.randint(0, vocab - 1, (n, seq))
        with torch.no_grad():
            logits, stacked = target.forward_hiddens(ids)
        gold = logits[:, -1].argmax(dim=-1)
        hiddens = stacked[:, -1:, :, :].contiguous()
        anchors = ids[:, -1].contiguous()
        return hiddens, anchors, gold

    drafter.train()
    for _ in range(250):
        hiddens, anchors, gold = _batch(16)
        out = drafter(hiddens, anchors, greedy=True)
        loss = F.cross_entropy(out.logits[:, 0], gold)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    drafter.eval()
    hits = 0
    n = 0
    with torch.no_grad():
        for _ in range(8):
            hiddens, anchors, gold = _batch(16)
            pred = drafter(hiddens, anchors, greedy=True).draft_ids[:, 0]
            hits += int((pred == gold).sum().item())
            n += gold.numel()
    rate = hits / n
    logger.info(f"distilled first-token accept: {hits}/{n} = {rate:.3f}")
    assert rate >= 0.90, f"first-token accept {rate:.3f} < 0.90"


@torch.no_grad()
def test_speculative_accept_lengths_stops_at_first_mismatch():
    draft = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    target = torch.tensor([[1, 2, 9, 4], [0, 6, 7, 8]])
    from models.experimental.deepseek_v4_flash.dspark import speculative_accept_lengths

    assert speculative_accept_lengths(draft, target).tolist() == [2, 0]
