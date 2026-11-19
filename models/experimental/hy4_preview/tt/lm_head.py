# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview final norm and LM head on the SC1 layout (tenstorrent/tt-metal#65892): the
120832 x 6144 head sharded by vocab over all 32 chips, 3776 logits to a chip, with an on-device
top-k / greedy / sampled next token over the sharded logits.

Mirrors ../reference/lm_head.py Hy4LMHead method for method; ``VocabShards`` there says which
chip owns a token id (chips counted row-major, the order ttnn.get_device_tensors returns).

    stage        per chip                    cross-chip traffic
    whole        [1, 1, T, 6144]             all_gather(hidden axis) of T x 1536
    norm         [1, 1, T, 6144]             none: the state is whole, the norm weight replicated
    logits       [1, 1, T, 3776]             none: each chip holds its 6144 x 3776 slice of the head
    candidates   [1, 1, T, 32 x 32]          2 x (all_gather(head) + all_gather(hidden)) of T x 32
    top_k        [1, 1, T, k]                none

* The input is the readout tt/ihc.py TtIHCHead produces, [1, 1, T, 6144/4] split over the hidden
  axis. A chip needs the whole hidden state to compute its logits, so it is gathered once and the
  final norm then runs on the whole state, chip-local. Normalising before the gather would cost
  an all-reduce of the sum of squares on top.
* The norm follows HYV4RMSNorm: the normalisation in fp32, then the weight. With
  ``hidden_dtype=bfloat16`` the normalised state is cast to BF16 before the weight is applied,
  which is what HF does with the BF16 checkpoint; with float32 it stays fp32 throughout.
* The head matmul runs at HiFi4 with fp32 accumulation whatever the weight dtype: it decides a
  discrete choice. ``logits_dtype`` is the dtype the 6144-term sums are packed to. BF16 logits
  have 8 bits of mantissa, a step of 1/16 between 8 and 16 and 1/8 between 16 and 32, so nearby
  tokens tie; fp32 logits cost no weight memory. The checkpoint's head is BF16, so a BF16 head
  on device holds the same numbers as HF's fp32 one: what HF's "fp32 LM head" buys is the fp32
  sum, and the defaults here (BF16 head, BF16 state, fp32 logits) keep exactly that. The formats
  are compared in exact arithmetic in tests/test_lm_head_reference_host.py
  ``test_lm_head_precision_formats`` and on device in tests/test_embedding_lm_head.py
  ``test_lm_head_precision``. Measured there on the 8x4 Galaxy with the checkpoint's norm and head
  on 3 x 256 stand-in readouts, against HF's recipe: a BF16 head and an fp32 head give the same
  logits (max |diff| 6.7e-3, the HiFi4 matmul; top-1 the same on 768 of 768 rows); BF16 logits
  move them by up to 0.12 and lose 0.4 to 2.3 % of top-1; a BFP8 head moves them by 0.09 and
  loses up to 0.8 %. On BF16-valued readouts the BF16 normed state is HF's bit for bit.
* The readout arrives in fp32 (the device's streams are fp32) and is normalised as it is; HF's
  readout is BF16 by then. On a readout that is not BF16-valued this alone moves the logits by
  0.04 to 0.06 against HF's recipe (PCC 0.999995), which is the size of HF's own rounding.
* Top-k without reading the logits back: every chip takes the top 32 of its own 3776 logits with
  the global token ids riding along as ttnn.topk's ``indices_tensor``, the 32 x 32 candidates are
  gathered over both mesh axes, and a second top-k (again carrying the ids) picks the global top
  k on every chip. A token in the global top 32 is in its own chip's top 32, so the result is
  exact for k <= 32. ``greedy`` is that with k = 1. ``sample`` hands the gathered candidates to
  ttnn.sampling (top-k, top-p, temperature); that op takes BF16 values only, so sampling
  probabilities see BF16-rounded logits while greedy and top-k keep the logits' own dtype.

Both ``greedy`` and ``sample`` return token ids as uint32 [1, 1, 1, T] on every chip: the form
tt/embedding.py ``table_rows`` takes, so the next token never leaves the device.
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.hy4_preview.reference.lm_head import LMHeadConfig, VocabShards, sampling_parameters
from models.experimental.hy4_preview.tt.common import MeshPlan, compute_config
from models.experimental.hy4_preview.tt.embedding import per_chip_mapper, per_chip_shape

# The widest candidate list ttnn.sampling takes per user and chip, and so the largest k of top_k.
MAX_TOP_K = 32


class TtHy4LMHead(LightweightModule):
    """Readout [1, 1, T, hidden/tp] split over the hidden axis -> logits [1, 1, T, vocab/chips] per chip.

    ``weights``: ``norm_weight`` [hidden] and ``weight`` [vocab, hidden] (reference names).
    ``weight_dtype``: the head's dtype on device. ``hidden_dtype``: the dtype of the normed state
    going into the head (bfloat16 is HF's behaviour, see above). ``logits_dtype``: the logits.
    """

    def __init__(
        self,
        plan: MeshPlan,
        cfg: LMHeadConfig,
        weights: dict,
        weight_dtype=ttnn.bfloat16,
        hidden_dtype=ttnn.bfloat16,
        logits_dtype=ttnn.float32,
        fidelity=ttnn.MathFidelity.HiFi4,
    ):
        super().__init__()
        assert hidden_dtype in (ttnn.bfloat16, ttnn.float32), hidden_dtype
        self.plan, self.cfg = plan, cfg
        self.hidden_dtype, self.logits_dtype = hidden_dtype, logits_dtype
        self.chip_shape = per_chip_shape(plan)
        self.num_chips = self.chip_shape[0] * self.chip_shape[1]
        self.shards = shards = VocabShards(cfg.vocab_size, self.num_chips)
        self.ckc = compute_config(fidelity, fp32_acc=True)
        self._labels = {}

        norm_weight, weight = weights["norm_weight"], weights["weight"]
        assert tuple(norm_weight.shape) == (cfg.hidden_size,), tuple(norm_weight.shape)
        assert tuple(weight.shape) == (cfg.vocab_size, cfg.hidden_size), tuple(weight.shape)
        mesh_device = plan.mesh_device
        self.norm_weight = ttnn.from_torch(
            norm_weight.float().reshape(1, 1, 1, cfg.hidden_size),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=hidden_dtype,
        )
        # Per chip [1, 1, hidden, vocab/chips]: x @ W is the chip's own run of the vocab.
        self.weight = ttnn.from_torch(
            shards.split_head(weight.float())
            .reshape(*self.chip_shape, cfg.hidden_size, shards.rows_per_chip)
            .contiguous(),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=weight_dtype,
            mesh_mapper=per_chip_mapper(plan),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------------------------------
    # Readout -> logits
    # ------------------------------------------------------------------------------------------
    def whole(self, x):
        """[1, 1, T, hidden/tp] split over the hidden axis -> [1, 1, T, hidden] on every chip."""
        return self.plan.all_gather(x, dim=3, axis=self.plan.hidden_axis)

    def norm(self, x_whole):
        """HYV4RMSNorm on the whole state: [1, 1, T, hidden] -> [1, 1, T, hidden] in ``hidden_dtype``.

        Composite on purpose: ttnn.rms_norm does not keep fp32 precision on an fp32 input.
        """
        cfg = self.cfg
        xf = x_whole if x_whole.dtype == ttnn.float32 else ttnn.typecast(x_whole, ttnn.float32)
        ss = ttnn.sum(ttnn.multiply(xf, xf), dim=-1, keepdim=True)
        inv = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / cfg.hidden_size), cfg.norm_eps))
        h = ttnn.multiply(xf, inv)
        if self.hidden_dtype != ttnn.float32:
            h = ttnn.typecast(h, self.hidden_dtype)
        return ttnn.multiply(h, self.norm_weight)

    def logits(self, h):
        """Normed state [1, 1, T, hidden] -> this chip's logits [1, 1, T, vocab/chips]."""
        return ttnn.matmul(h, self.weight, compute_kernel_config=self.ckc, dtype=self.logits_dtype)

    def forward(self, x):
        return self.logits(self.norm(self.whole(x)))

    def to_torch(self, logits) -> torch.Tensor:
        """Host readback of the sharded logits: [T, vocab], fp32, token ids in order."""
        shards = [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(logits)[: self.num_chips]]
        return torch.cat(shards, dim=-1)[0, 0]

    # ------------------------------------------------------------------------------------------
    # Next token on device
    # ------------------------------------------------------------------------------------------
    def _token_ids(self, T):
        """uint32 [1, 1, T, vocab/chips] per chip: the global token id of every logit the chip holds."""
        if T not in self._labels:
            rows = self.shards.rows_per_chip
            ids = self.shards.offsets().reshape(-1, 1, 1) + torch.arange(rows).reshape(1, 1, -1)
            self._labels[T] = ttnn.from_torch(
                ids.expand(self.num_chips, T, rows).to(torch.int32).reshape(*self.chip_shape, T, rows).contiguous(),
                layout=ttnn.TILE_LAYOUT,
                device=self.plan.mesh_device,
                dtype=ttnn.uint32,
                mesh_mapper=per_chip_mapper(self.plan),
            )
        return self._labels[T]

    def _gather_all(self, x):
        """Concatenate a per-chip [1, 1, T, w] over all chips on the last dim, on every chip."""
        plan = self.plan
        if plan.hidden_axis is None:
            return x
        return plan.all_gather(plan.all_gather(x, dim=3, axis=plan.head_axis), dim=3, axis=plan.hidden_axis)

    def candidates(self, logits):
        """Every chip's top 32 with their token ids, gathered: (values, ids), both
        [1, 1, T, 32 * chips] and the same on every chip. Values keep the logits' dtype; ids are uint32."""
        T = logits.shape[-2]
        assert (
            self.shards.rows_per_chip >= MAX_TOP_K
        ), f"a chip holds {self.shards.rows_per_chip} logits, fewer than {MAX_TOP_K}"
        values, ids = ttnn.topk(
            logits, k=MAX_TOP_K, dim=-1, largest=True, sorted=True, indices_tensor=self._token_ids(T)
        )
        return self._gather_all(values), self._gather_all(ids)

    def top_k(self, logits, k: int):
        """The k largest logits per token over the whole vocab, sorted: (values [1, 1, T, k],
        token ids uint32 [1, 1, T, k]), the same on every chip."""
        assert 1 <= k <= MAX_TOP_K, f"k={k}: a chip contributes its top {MAX_TOP_K}"
        values, ids = self.candidates(logits)
        return ttnn.topk(values, k=k, dim=-1, largest=True, sorted=True, indices_tensor=ids)

    def greedy(self, logits):
        """The argmax token per row: uint32 [1, 1, 1, T], the same on every chip."""
        _, ids = self.top_k(logits, 1)
        return ttnn.reshape(ids, [1, 1, 1, ids.shape[-2]])

    def sample(self, logits, k, p, temperature, seed=None):
        """One sampled token per row with ttnn.sampling: uint32 [1, 1, 1, 32], ROW_MAJOR, on every chip.

        ``k`` (1..32), ``p`` (0..1) and ``temperature`` (> 0) are one number each or one per row,
        checked by reference/lm_head.py ``sampling_parameters``. The candidates are every chip's top 32.

        Only the shape that has run on device is accepted: 32 rows (the op runs one core per row)
        over the candidates of more than one chip. ttnn.sampling can deadlock on a candidate list
        one tile wide (tenstorrent/tt-metal#53781), and a one-chip call with one row was the last
        thing the bring-up test ran before chip 0 stopped responding. A batch with fewer live rows
        is padded to 32 by the caller, with copies of a live row and not with zeros: a zero readout
        gives logits that are all 0, so its candidates all tie, which is the condition of that
        deadlock and has not been tried at this width.

        What a draw is, measured on the Galaxy (tests/test_embedding_lm_head.py
        ``test_sampling_distribution_sc1``): the softmax of the top ``k`` at the given temperature,
        cut after the token that takes the cumulative mass above ``p``, renormalised.

        ``seed`` is for reproducing a draw in a test, not for serving. The op seeds every row's
        core with it, so all 32 rows of a seeded call are drawn with ONE random number, and two
        calls with the same seed with the same one. The cores then stay in step: every later call
        without a seed still gives all rows one number (measured in the same test; without a seed
        on a freshly opened device the rows draw independently). Seeds per user need
        ttnn.manual_seed(seeds, user_ids) on the op's core grid, as models/common/sampling does;
        that is not built here.
        """
        # Everything is checked before the first device op: the op checks none of it, and a k past
        # the candidate row or a zero temperature is read by its kernel as it stands.
        T = logits.shape[-2]
        assert T == 32, f"sampling is verified for 32 rows only, got {T}"
        assert self.num_chips > 1, "sampling needs the candidates of more than one chip"
        k_h, p_h, inv_temp_h = sampling_parameters(k, p, temperature, T, MAX_TOP_K)
        mesh_device = self.plan.mesh_device

        def per_row(t, dtype):
            return ttnn.from_torch(t, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=dtype)

        k_t = per_row(k_h, ttnn.uint32)
        p_t = per_row(p_h, ttnn.bfloat16)
        temp_t = per_row(inv_temp_h, ttnn.bfloat16)  # ttnn.sampling takes the reciprocal of the temperature
        values, ids = self.candidates(logits)
        values = values if values.dtype == ttnn.bfloat16 else ttnn.typecast(values, ttnn.bfloat16)
        return ttnn.sampling(values, ttnn.untilize(ids), k=k_t, p=p_t, temp=temp_t, seed=seed)
