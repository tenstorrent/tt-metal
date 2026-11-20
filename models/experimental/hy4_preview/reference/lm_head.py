# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch ground-truth reference for the two ends of Hy4-preview: the token embedding and the
final norm + LM head (tenstorrent/tt-metal#65892).

Pure torch, runs on CPU. Mirrors HF transformers ``models/hy_v4/modeling_hy_v4.py``:

  - HYV4Model.embed_tokens                      -> Hy4Embedding
  - HYV4Model.norm (HYV4RMSNorm)                -> Hy4LMHead.norm
  - HYV4ForCausalLM.lm_head                     -> Hy4LMHead.logits

The order HF runs them in, with the pieces that live in reference/ihc.py:

    h      = embed_tokens(input_ids)                       Hy4Embedding
    x      = h expanded to the 4 residual streams          ihc.ihc_expand
    x      = the 78 decoder layers
    h      = hc_head(x)                                    ihc.IHCHead, the stream readout
    h      = norm(h)                                       Hy4LMHead.norm
    logits = lm_head(h.to(lm_head.weight.dtype))           Hy4LMHead.logits

Precision, as HF runs it with the BF16 checkpoint:

* The embedding table is BF16 and the lookup returns BF16.
* HYV4RMSNorm computes the normalisation in fp32, casts the result back to its input's dtype and
  only then multiplies by the weight, in that dtype. With a BF16 model the hidden state that
  reaches the LM head is therefore rounded to BF16 twice.
* ``lm_head`` is in ``HYV4ForCausalLM._keep_in_fp32_modules_strict``: its weight is held in fp32
  (the checkpoint's BF16 values, widened) and the hidden state is cast to that dtype, so the
  6144-term dot product and the logits are fp32. config.json's ``enable_lm_head_fp32: true`` is
  the name of this behaviour; the HF code never reads the flag, the fp32 head is unconditional.

Both behaviours fall out of this module from the dtypes of its parameters and input: build it in
fp32 for the exact arithmetic, or cast the embedding and norm weight to BF16 for what HF does.

Checkpoint names (tencent/Hy4-preview):
    model.embed_tokens.weight   [120832, 6144]
    model.norm.weight           [6144]
    lm_head.weight              [120832, 6144]   (not tied to the embedding)

``VocabShards`` is the contract of the vocab-sharded device modules (tt/embedding.py,
tt/lm_head.py): which chip owns a token id and which row of the chip's table it is.
``sampling_parameters`` is the contract of what the on-device sampler may be handed.

The last three functions are the yardstick of the LM-head precision question (does the head need
fp32): stand-in readouts, the agreement metrics, and the values a BFP8 tensor holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LMHeadConfig:
    """Defaults are the tencent/Hy4-preview config.json values."""

    hidden_size: int = 6144
    vocab_size: int = 120832
    norm_eps: float = 1e-5  # "rms_norm_eps"


class Hy4Embedding(nn.Module):
    """Token ids [...] -> hidden states [..., hidden] in the table's dtype."""

    def __init__(self, cfg: LMHeadConfig, init: bool = True):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size))
        if init:  # skipped when the caller overwrites the 742M-entry table anyway
            nn.init.normal_(self.weight, std=0.006)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.weight)


class Hy4LMHead(nn.Module):
    """The final norm and the LM head: the iHC head's readout [..., hidden] -> logits [..., vocab].

    Parameters: norm_weight [hidden], weight [vocab, hidden] (no bias).
    """

    PARAMETER_NAMES = ("norm_weight", "weight")
    CHECKPOINT_NAMES = {"norm_weight": "model.norm.weight", "weight": "lm_head.weight"}

    def __init__(self, cfg: LMHeadConfig, init: bool = True):
        super().__init__()
        self.cfg = cfg
        self.norm_weight = nn.Parameter(torch.ones(cfg.hidden_size))
        self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size))
        if init:
            nn.init.normal_(self.weight, std=0.006)

    def as_hf_bf16(self) -> "Hy4LMHead":
        """This head in the dtypes HF holds it in for the BF16 checkpoint: a BF16 norm weight and
        the fp32 head weight (shared with ``self``, not copied). Feed it a BF16 readout."""
        twin = Hy4LMHead.__new__(Hy4LMHead)
        nn.Module.__init__(twin)
        twin.cfg = self.cfg
        twin.norm_weight = nn.Parameter(self.norm_weight.detach().to(torch.bfloat16))
        twin.weight = self.weight
        return twin.eval()

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        """HYV4RMSNorm: normalise in fp32, cast back to x's dtype, then apply the weight."""
        xf = x.to(torch.float32)
        variance = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(variance + self.cfg.norm_eps)
        return self.norm_weight * xf.to(x.dtype)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """The LM head on a normed hidden state, in the head weight's dtype (fp32 in HF)."""
        return F.linear(h.to(self.weight.dtype), self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(self.norm(x))

    def weights(self) -> dict:
        return {name: getattr(self, name).detach() for name in self.PARAMETER_NAMES}


@dataclass(frozen=True)
class VocabShards:
    """The vocab cut into ``num_chips`` equal contiguous runs of token ids.

        chip = token_id // rows_per_chip      chips counted row-major over the mesh, the order
                                              ttnn.get_device_tensors returns them in
        row  = token_id %  rows_per_chip      the row inside that chip's slice

    The embedding table a chip holds is its slice with one zero row before and one after it, so a
    token id maps on EVERY chip to a valid row of that chip's table:

        table_row(chip, token_id) = clamp(token_id - chip * rows_per_chip + 1, 0, rows_per_chip + 1)

    which is the token's row on the chip that owns it and a zero row on the other chips.
    """

    vocab_size: int = 120832
    num_chips: int = 32

    def __post_init__(self):
        assert self.vocab_size % self.num_chips == 0, f"{self.vocab_size} token ids do not split {self.num_chips} ways"

    @property
    def rows_per_chip(self) -> int:
        return self.vocab_size // self.num_chips

    @property
    def table_rows(self) -> int:
        """Rows of a chip's embedding table: its slice plus the two zero rows."""
        return self.rows_per_chip + 2

    def chip(self, token_id):
        return token_id // self.rows_per_chip

    def row(self, token_id):
        return token_id % self.rows_per_chip

    def token_id(self, chip, row):
        return chip * self.rows_per_chip + row

    def offsets(self) -> torch.Tensor:
        """[num_chips]: the first token id of every chip."""
        return torch.arange(self.num_chips) * self.rows_per_chip

    def table_rows_for(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Token ids [...] -> [num_chips, ...]: the row each chip looks up in its padded table.

        Ids outside the vocab are refused, as torch's embedding refuses them. The formula itself
        sends such an id to a zero row on every chip, which is what the on-device arithmetic
        (tt/embedding.py ``table_rows``) does with one: a zero embedding, not an error.
        """
        in_vocab = (token_ids >= 0) & (token_ids < self.vocab_size)
        assert (
            in_vocab.all()
        ), f"token ids outside [0, {self.vocab_size}): {token_ids[~in_vocab].flatten()[:8].tolist()}"
        shape = (self.num_chips,) + (1,) * token_ids.dim()
        return (token_ids.unsqueeze(0) - self.offsets().reshape(shape) + 1).clamp(0, self.rows_per_chip + 1)

    def split_table(self, weight: torch.Tensor) -> torch.Tensor:
        """Embedding table [vocab, hidden] -> [num_chips, rows_per_chip + 2, hidden], zero rows first and last."""
        out = weight.new_zeros(self.num_chips, self.table_rows, weight.shape[-1])
        out[:, 1:-1] = weight.reshape(self.num_chips, self.rows_per_chip, weight.shape[-1])
        return out

    def split_head(self, weight: torch.Tensor) -> torch.Tensor:
        """LM head [vocab, hidden] -> [num_chips, hidden, rows_per_chip]: ``x @ out[chip]`` is that chip's logits."""
        return weight.reshape(self.num_chips, self.rows_per_chip, weight.shape[-1]).transpose(1, 2)


def sampling_parameters(k, p, temperature, rows: int, max_top_k: int = 32):
    """The per-row parameters ttnn.sampling takes, checked: (k int32 [rows], p fp32 [rows],
    1 / temperature fp32 [rows]). Each argument is one number or one per row.

    The op does not check them and its kernel trusts them: it walks ``k`` entries of a candidate
    row that holds ``max_top_k`` of them, so a k above that reads past the row, and it multiplies
    the logits by the reciprocal of the temperature, which a temperature of 0 makes infinite.
    Greedy is k = 1, not temperature = 0.

        k            whole numbers in [1, max_top_k]
        p            [0, 1]; 0 keeps the leading candidate only, 1 keeps all k
        temperature  finite and > 0
    """

    def per_row(name, value):
        t = torch.as_tensor(value, dtype=torch.float32).flatten()
        assert t.numel() in (1, rows), f"{name}: one value or one per row ({rows}), got {t.numel()}"
        assert torch.isfinite(t).all(), f"{name} must be finite, got {t.tolist()}"
        return t.expand(rows).contiguous()

    k_t, p_t, temp_t = per_row("k", k), per_row("p", p), per_row("temperature", temperature)
    assert (
        (k_t >= 1) & (k_t <= max_top_k) & (k_t == k_t.round())
    ).all(), f"k must be whole numbers in [1, {max_top_k}]"
    assert ((p_t >= 0) & (p_t <= 1)).all(), "p must be in [0, 1]"
    assert (temp_t > 0).all(), "temperature must be > 0 (greedy is k = 1)"
    return k_t.to(torch.int32), p_t, 1.0 / temp_t


def stand_in_readouts(embedding: torch.Tensor, head: torch.Tensor, tokens: int, seed: int) -> dict:
    """Readouts [tokens, hidden] standing in for the model's final hidden states, BF16-valued.

    The real ones need the 78 layers. ``embedding`` and ``head`` are the two [vocab, hidden] tables.

    * ``embedding rows``: rows of the input embedding table. Real token geometry, but not where
      the model's last layer lands: the head sees them as near-random directions.
    * ``peaked``: a unit-RMS Gaussian vector plus five LM-head row directions weighted 1, 0.97,
      0.9, 0.7 and 0.5, scaled so the leading token's logit is about 25 over a background of
      standard deviation 2. Shaped like a confident prediction with close runners-up.
    * ``random``: Gaussian vectors, a flat distribution with no structure and small margins.
    """
    vocab, hidden = head.shape
    g = torch.Generator().manual_seed(seed)
    rows = embedding[torch.randint(0, vocab, (tokens,), generator=g)].float()
    picks = torch.randint(0, vocab, (tokens, 5), generator=g)
    directions = F.normalize(head[picks].float(), dim=-1) * hidden**0.5
    mix = (directions * torch.tensor([1.0, 0.97, 0.9, 0.7, 0.5])[None, :, None]).sum(1)
    peaked = torch.randn(tokens, hidden, generator=g) + 0.16 * mix
    rand = torch.randn(tokens, hidden, generator=g)
    return {name: x.bfloat16().float() for name, x in (("embedding rows", rows), ("peaked", peaked), ("random", rand))}


def logit_agreement(ref_logits: torch.Tensor, logits: torch.Tensor) -> dict:
    """How well ``logits`` [T, vocab] reproduce the reference's ranking.

    ``top1``: rows with the same argmax. ``top5_set`` / ``top5_order``: rows with the same top 5,
    as a set and in order. ``flip_margin1`` / ``flip_margin5``: the widest reference margin (top-1
    minus top-2, 5th minus 6th) at which a row disagrees, 0 when none does: how close the
    reference's own call was where the two differ.
    """
    a, b = ref_logits.flatten().double(), logits.flatten().double()
    r_top = torch.topk(ref_logits, 6, dim=-1)
    d_top = torch.topk(logits, 5, dim=-1)
    top1 = d_top.indices[:, 0] == r_top.indices[:, 0]
    top5_set = (d_top.indices.sort(-1).values == r_top.indices[:, :5].sort(-1).values).all(-1)
    top5_order = (d_top.indices == r_top.indices[:, :5]).all(-1)
    margin1 = r_top.values[:, 0] - r_top.values[:, 1]
    margin5 = r_top.values[:, 4] - r_top.values[:, 5]
    return {
        "pcc": torch.corrcoef(torch.stack([a, b]))[0, 1].item(),
        "max_abs": (a - b).abs().max().item(),
        "top1": top1.float().mean().item(),
        "top5_set": top5_set.float().mean().item(),
        "top5_order": top5_order.float().mean().item(),
        "flip_margin1": margin1[~top1].max().item() if (~top1).any() else 0.0,
        "flip_margin5": margin5[~top5_set].max().item() if (~top5_set).any() else 0.0,
    }


def format_agreement(a: dict) -> str:
    return (
        f"logits_pcc={a['pcc']:.7f} max|Δ|={a['max_abs']:.2e} top1={a['top1']:.4f} top5_set={a['top5_set']:.4f} "
        f"top5_order={a['top5_order']:.4f} flip_margin1={a['flip_margin1']:.4f} flip_margin5={a['flip_margin5']:.4f}"
    )


def bfp8_values(t: torch.Tensor) -> torch.Tensor:
    """The values ``t`` takes when stored as BFP8 (ttnn.bfloat8_b), computed in torch.

    Follows tt_metal/impl/data_format/blockfloat_common.cpp: every run of 16 consecutive elements
    of the last dim (a row of a 16 x 16 tile face) shares the exponent of its largest magnitude,
    and each element keeps a sign and 7 mantissa bits against that exponent, rounded to nearest
    with ties to even. The last dim must be a multiple of 16 and start on a face boundary.
    """
    assert t.shape[-1] % 16 == 0, t.shape
    runs = t.float().reshape(*t.shape[:-1], -1, 16)
    _, exponent = torch.frexp(runs.abs().amax(-1, keepdim=True))  # largest = m * 2^exponent, m in [0.5, 1)
    step = torch.exp2((exponent - 7).float())
    return (torch.round(runs / step).clamp(-127, 127) * step).reshape(t.shape)
