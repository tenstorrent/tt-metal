# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Centroid Masked Embedding (CME) output head for the Gemma4 it-assistant drafter.

The E2B assistant (``google/gemma-4-E2B-it-assistant``) sets
``use_ordered_embeddings: true``, which replaces the drafter's dense
``lm_head`` with a sparse two-stage head. Reference:
``transformers.models.gemma4_assistant.modeling_gemma4_assistant.Gemma4AssistantMaskedEmbedder``.

HF's forward, in our notation (E2B: H=256, V=262144, C=2048 centroids,
top_k=32, P=V/C=128, so N=top_k*P=4096 candidates):

    centroid_logits = h @ centroids^T                     # [.., C]
    top_k_indices   = topk(centroid_logits, top_k)        # [.., top_k]
    canon           = token_ordering.view(C, P)           # [C, P] token ids
    selected_canon  = canon[top_k_indices]                # [.., top_k, P]
    selected_emb    = lm_head_weight[selected_canon]      # [.., N, H]
    selected_logits = h . selected_emb^T                  # [.., N]
    mask_value      = selected_logits.min() - 1
    out             = full((.., V), mask_value).scatter_(-1, selected_canon, selected_logits)

Two properties drive this implementation:

1. **The full-vocab tensor is never needed for greedy.** Every selected entry
   strictly exceeds ``mask_value``, and ``token_ordering`` is a permutation (so
   the scatter indices are unique). Therefore

       argmax_over_vocab(out) == selected_canon[argmax(selected_logits)]

   and greedy drafting only needs the compact ``[.., N]`` logits plus their token
   ids. We keep both on device as a :class:`CmeLogits` pair.

2. **``mask_value`` matters for sampling.** ~258k of the 262144 entries sit at
   ``min-1``, which is a non-negligible share of the softmax denominator, so it
   cannot be approximated. :meth:`Gemma4TTMaskedEmbedder.to_host_full_vocab`
   reconstructs the exact HF tensor on host from the compact pair — cheap, and it
   keeps the sampling path bit-comparable with HF.

Beyond correctness this is a large drafter speedup: it replaces a 256x262144
lm_head matmul (~67M MAC) with 256x2048 + 4096x256 (~1.6M MAC).

Every step is a device op (no host round-trip), so the head captures inside the
existing fused Metal trace in ``spec_decode.py``.
"""

import math
from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4.utils.general_utils import get_cache_file_name


@dataclass
class CmeLogits:
    """Compact drafter logits: the ``N`` selected scores and their token ids.

    ``values``: [1,1,rows,N] TILE float32 — the selected dot products.
    ``ids``:    [1,1,rows,N] ROW_MAJOR uint32 — the vocab id each score belongs
                to (HF's ``selected_canonical``, flattened in ``top_k`` x ``P`` C order).

    ``spec_decode`` dispatches on this type in ``_logits_to_host`` and
    ``_argmax_last``, so the ~10 drafter-logits call sites stay untouched.
    """

    values: "ttnn.Tensor"
    ids: "ttnn.Tensor"
    rows: int

    def deallocate(self, force=True):
        self.values.deallocate(force)
        self.ids.deallocate(force)


def _same_buffer(a, b):
    """Return whether tensors alias, conservatively preserving unknown storage."""
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:
        # Failure to inspect storage must never authorize a forced deallocation.
        return True


def _encode_base_digits(values, base, num_digits):
    """Encode integer IDs least-significant digit first."""
    return [(values // (base**digit)) % base for digit in range(num_digits)]


def _reconstruct_base_digits(digits, base):
    """CPU reference for the device FP32 Horner reconstruction."""
    result = torch.zeros_like(digits[0], dtype=torch.float32)
    for digit in reversed(digits):
        result = result * float(base) + digit.to(torch.float32)
    return result.to(torch.int64)


class Gemma4TTMaskedEmbedder:
    """TT implementation of ``Gemma4AssistantMaskedEmbedder``.

    All weights are replicated across TP. The drafter's hidden is only 256 wide,
    so sharding it buys nothing, and replication removes the full-vocab logits
    all-gather that the dense ``lm_head`` path needs (``ccl_allgather`` over
    262144 columns) — CME mode does no CCL at all in the head.
    """

    def __init__(
        self,
        mesh_device,
        assistant_args,
        state_dict,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=None,
    ):
        self.mesh_device = mesh_device
        self.args = assistant_args
        text_args = assistant_args.text_args
        self.hidden_size = text_args.hidden_size
        self.vocab_size = text_args.vocab_size
        self.num_centroids = assistant_args.num_centroids
        self.top_k = assistant_args.centroid_intermediate_top_k
        if self.vocab_size % self.num_centroids != 0:
            raise ValueError(
                f"vocab_size ({self.vocab_size}) must be divisible by num_centroids ({self.num_centroids})"
            )
        self.vocab_per_centroid = self.vocab_size // self.num_centroids
        self.num_candidates = self.top_k * self.vocab_per_centroid

        is_mesh = hasattr(mesh_device, "shape")
        replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
        self._mapper = replicate if is_mesh else None
        self.mesh_config = mesh_config

        # Both head matmuls run at HiFi4 with fp32 accumulation. This is not
        # gold-plating: the checkpoint's weights and the drafter hidden are both
        # bf16, so the fp32-accumulated dot IS the exact answer, and HF (which
        # upcasts to float) computes exactly that. With the default lower-fidelity
        # bf16 accumulation the ~4096 candidate scores carry ~0.03-0.06 of error at
        # a ~5.0 scale, which is enough to reorder the top two candidates — the
        # observed top-2 gap can be as small as ~0.09, i.e. ~3 bf16 ULPs. A
        # reordered winner is a wasted draft (pure acceptance-rate loss), and these
        # matmuls are tiny (H x C and N x H), so the extra fidelity is ~free.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        centroids = state_dict.get("masked_embedding.centroids.weight")
        ordering = state_dict.get("masked_embedding.token_ordering")
        embed = state_dict.get("lm_head.weight")
        if embed is None:
            # lm_head is tied to the assistant's own embed_tokens (E2B stores no
            # separate lm_head.weight).
            embed = state_dict.get("model.embed_tokens.weight")
        missing = [
            n
            for n, t in (
                ("masked_embedding.centroids.weight", centroids),
                ("masked_embedding.token_ordering", ordering),
                ("lm_head.weight / model.embed_tokens.weight", embed),
            )
            if t is None
        ]
        if missing:
            raise ValueError(f"Assistant checkpoint has use_ordered_embeddings but is missing: {', '.join(missing)}")

        # Stage 1 weight: [C, H] -> [1,1,H,C] so `ttnn.linear(normed, .)` gives [.., C].
        self.centroids = ttnn.as_tensor(
            centroids.transpose(-2, -1).unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "cme_centroids"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # HF's `canonical_positions_per_cluster`: token_ordering.view(C, P).
        ordering_2d = ordering.to(torch.int64).reshape(1, 1, self.num_centroids, self.vocab_per_centroid)

        def load_ordering(name, weight, tensor_dtype, layout):
            return ttnn.as_tensor(
                weight,
                device=mesh_device,
                dtype=tensor_dtype,
                layout=layout,
                mesh_mapper=self._mapper,
                cache_file_name=get_cache_file_name(tensor_cache_path, name),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Embedding is much faster than the generic integer gather, but its BF16
        # weight cannot represent full token IDs. Store exact base-64 digits,
        # gather each digit row, and reconstruct IDs in FP32. Integers above
        # 2**24 are not exactly representable in FP32, so retain the uint32 gather
        # for larger vocabularies.
        self.digit_base = 64
        self.num_digits = max(1, math.ceil(math.log(max(self.vocab_size, 2)) / math.log(self.digit_base)))
        if self.digit_base**self.num_digits < self.vocab_size:
            self.num_digits += 1
        self.ordering_digits = []
        use_digit_lookup = self.vocab_size <= 2**24 and int(ordering_2d.max()) < 2**24
        if use_digit_lookup:
            for d, digit in enumerate(_encode_base_digits(ordering_2d, self.digit_base, self.num_digits)):
                self.ordering_digits.append(
                    load_ordering(
                        f"cme_token_ordering_b{self.digit_base}_d{d}",
                        digit.to(torch.bfloat16),
                        ttnn.bfloat16,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                )
        # Preserve the original integer gather outside the FP32-exact range.
        self.ordering = (
            None
            if use_digit_lookup
            else load_ordering("cme_token_ordering", ordering_2d, ttnn.uint32, ttnn.TILE_LAYOUT)
        )

        # Row-gatherable copy of the output embedding: `ttnn.embedding` requires a
        # ROW_MAJOR bf16 weight whose leading two dims are 1. This REPLACES the
        # dense [H, V] lm_head in CME mode, so DRAM is net-neutral.
        self.embed_table = ttnn.as_tensor(
            embed.unsqueeze(0).unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=self._mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "cme_embed_table"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # ── forward ───────────────────────────────────────────────────────────────
    def forward(self, normed):
        """``normed`` [1,1,rows,H] TILE -> :class:`CmeLogits` with rows x N entries.

        Rows are processed independently in a fixed-count Python loop and
        concatenated. A loop keeps every tensor op on its native shape (the
        alternative — folding rows into the gather's row dim — needs a
        [rows,top_k] -> [rows*top_k,1] TILE reshape, which is a physical relayout,
        not a view). The count is static, so the loop captures in a trace, and at
        the only enabled batch (rows=1) it runs exactly once.
        """
        rows = int(normed.shape[-2])
        val_rows, id_rows = [], []
        for r in range(rows):
            v, i = self._forward_row(normed if rows == 1 else self._row(normed, r))
            val_rows.append(v)
            id_rows.append(i)
        if rows == 1:
            return CmeLogits(values=val_rows[0], ids=id_rows[0], rows=1)
        values = ttnn.concat(val_rows, dim=2)
        ids = ttnn.concat(id_rows, dim=2)
        for t in val_rows + id_rows:
            t.deallocate(True)
        return CmeLogits(values=values, ids=ids, rows=rows)

    def _row(self, normed, r):
        return ttnn.slice(normed, [0, 0, r, 0], [1, 1, r + 1, self.hidden_size])

    def _forward_row(self, h):
        """One row: h [1,1,1,H] TILE -> (values [1,1,1,N] TILE, ids [1,1,1,N] TILE uint32)."""
        C, P, K, N = self.num_centroids, self.vocab_per_centroid, self.top_k, self.num_candidates

        # 1. centroid logits [1,1,1,C]
        centroid_logits = ttnn.linear(h, self.centroids, compute_kernel_config=self.compute_kernel_config)

        # 2. top-k centroids. k=32 is tile-aligned, so topk takes no round-up/slice
        #    path. Indices come back TILE uint16 (width C <= 65535); typecast to
        #    uint32 so the downstream repeat/gather run on a 4-byte dtype.
        _, top_idx = ttnn.topk(centroid_logits, k=K, dim=-1)  # [1,1,1,K]
        centroid_logits.deallocate(True)
        if top_idx.dtype != ttnn.uint32:
            top_u32 = ttnn.typecast(top_idx, ttnn.uint32)
            top_idx.deallocate(True)
            top_idx = top_u32

        # 3-4. selected_canonical = canon[top_k_indices] -> the candidate token ids,
        #      i.e. the row gather out[s,:] = ordering[top_idx[s],:], done as
        #      num_digits base-64 embeddings + an fp32 recombine. See the
        #      ordering_digits comment in __init__ for why not ttnn.gather.
        if self.ordering is not None:
            idx_col = ttnn.transpose(top_idx, -2, -1)  # [1,1,K,1]
            top_idx.deallocate(True)
            gather_idx = ttnn.repeat(idx_col, ttnn.Shape([1, 1, 1, P]))  # [1,1,K,P]
            idx_col.deallocate(True)
            sel_ids = ttnn.gather(self.ordering, dim=2, index=gather_idx)  # [1,1,K,P] uint32
            gather_idx.deallocate(True)
        else:
            # ttnn.embedding wants a [1, K] ROW_MAJOR index.
            top_rm = ttnn.to_layout(top_idx, ttnn.ROW_MAJOR_LAYOUT)
            top_idx.deallocate(True)
            top_rm = ttnn.reshape(top_rm, (1, K))

            # Recombine most-significant digit first: acc = acc*base + digit.
            # fp32 throughout — the result runs to 262143, which needs 18 bits
            # and so is exact in fp32 but NOT in bf16.
            acc = None
            for d in reversed(range(self.num_digits)):
                dig = ttnn.embedding(top_rm, self.ordering_digits[d], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
                if len(dig.shape) == 3:
                    dig = ttnn.unsqueeze_to_4D(dig)  # [1,1,K,P]
                dig_f32 = ttnn.typecast(dig, ttnn.float32)
                dig.deallocate(True)
                if acc is None:
                    acc = dig_f32
                else:
                    scaled = ttnn.multiply(acc, float(self.digit_base), dtype=ttnn.float32)
                    acc.deallocate(True)
                    acc = ttnn.add(scaled, dig_f32, dtype=ttnn.float32)
                    scaled.deallocate(True)
                    dig_f32.deallocate(True)
            top_rm.deallocate(True)
            sel_ids = ttnn.typecast(acc, ttnn.uint32)  # [1,1,K,P] uint32 TILE
            acc.deallocate(True)

        # Flatten K x P -> N in C order (matches HF's `.view(batch, seq, -1)`), so a
        # local index f corresponds to centroid slot f // P, offset f % P. The
        # reshape is done in ROW_MAJOR where it is a free view; `ttnn.embedding`
        # wants a ROW_MAJOR index whose leading dims are 1 anyway.
        sel_ids_rm = ttnn.to_layout(sel_ids, ttnn.ROW_MAJOR_LAYOUT)
        sel_ids.deallocate(True)
        sel_ids_rm = ttnn.reshape(sel_ids_rm, (1, N))  # [1,N] uint32 RM

        # 5. Gather the N candidate output embeddings: [1,N,H] -> [1,1,N,H] TILE.
        sel_emb = ttnn.embedding(sel_ids_rm, self.embed_table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if len(sel_emb.shape) == 3:
            sel_emb = ttnn.unsqueeze_to_4D(sel_emb)

        # 6. selected_logits = h . sel_emb^T, as a matvec against h^T so no
        #    [N,H] -> [H,N] transpose of the 2 MB gathered block is needed.
        h_col = ttnn.transpose(h, -2, -1)  # [1,1,H,1]
        # fp32 OUTPUT, not just fp32 accumulation: at a ~5.0 logit scale one bf16
        # ULP is 0.031, while the measured gap between the top two candidates is
        # frequently smaller than that (median 0.33, but down to 0.003), so packing
        # the scores to bf16 is itself enough to reorder the winner. Keeping them
        # fp32 costs 16 KB and makes both the on-device argmax and the host
        # sampling reconstruction exact w.r.t. the fp32 reference.
        sel_logits_col = ttnn.matmul(
            sel_emb, h_col, compute_kernel_config=self.compute_kernel_config, dtype=ttnn.float32
        )  # [1,1,N,1]
        sel_emb.deallocate(True)
        h_col.deallocate(True)
        values = ttnn.transpose(sel_logits_col, -2, -1)  # [1,1,1,N]
        sel_logits_col.deallocate(True)

        # Keep candidate IDs row-major for the second gather. The reshape is a
        # view, so ids owns the storage through sel_ids_rm; do not force-free it.
        ids = ttnn.reshape(sel_ids_rm, (1, 1, 1, N))
        return values, ids

    # ── greedy: argmax straight to a token id, on device ──────────────────────
    def argmax_token_id(self, pack, rows):
        """``CmeLogits`` -> [1,1,rows] uint32 ROW_MAJOR **vocab** token ids.

        Same output contract as ``spec_decode._argmax_last``, so the fused-trace
        drafter recurrence and ``_ids_to_host`` consume it unchanged. Exact,
        because argmax over the selected set equals argmax over the full masked
        vocab (see the module docstring).
        """
        local = self._argmax_rows(pack.values, rows)  # [1,1,rows] uint32 RM, in [0,N)
        # Map local index -> token id entirely in row-major layout. Convert tiled
        # packs supplied by existing callers before gathering.
        ids = pack.ids
        detiled = None
        if ids.layout != ttnn.ROW_MAJOR_LAYOUT:
            ids = detiled = ttnn.to_layout(ids, ttnn.ROW_MAJOR_LAYOUT)
        local_col = ttnn.reshape(local, (1, 1, rows, 1))
        ids_col = ttnn.gather(ids, dim=-1, index=local_col)  # [1,1,rows,1] uint32 RM
        if detiled is not None:
            detiled.deallocate(True)
        return ttnn.reshape(ids_col, (1, 1, rows))

    @staticmethod
    def _argmax_rows(values, rows):
        """argmax over the last dim of [1,1,rows,N] -> [1,1,rows] uint32 RM.

        Mirrors ``spec_decode._argmax_last``: ``ttnn.argmax`` needs ROW_MAJOR
        input, and its fast multicore path is row-parallel and only correct when
        the row dim is exactly one tile — so pad to 32 rows and slice back. The
        width here is N=4096 rather than the 262144 vocab, which is what makes
        this ~2 orders of magnitude cheaper than the dense head's argmax.
        """
        R32 = 32
        if rows > R32:
            n_cols = values.shape[-1]
            chunks = []
            off = 0
            while off < rows:
                n = min(R32, rows - off)
                part = ttnn.slice(values, [0, 0, off, 0], [1, 1, off + n, n_cols])
                chunks.append(Gemma4TTMaskedEmbedder._argmax_rows(part, n))
                part.deallocate(True)
                off += n
            out = ttnn.concat(chunks, dim=2)
            for c in chunks:
                c.deallocate(True)
            return out
        # The drafter's single logical row can be untilized and reduced directly;
        # retain the established tile-padded path for rows 2 through 31.
        if rows == 1:
            untilized = ttnn.untilize(values, use_multicore=True)
            idx = ttnn.argmax(untilized, dim=-1, keepdim=False)
            untilized.deallocate(True)
            return idx
        src = values
        padded = None
        if rows < R32:
            padded = ttnn.pad(values, [(0, 0), (0, 0), (0, R32 - rows), (0, 0)], value=0.0)
            src = padded
        u = ttnn.untilize(src, use_multicore=True)
        # Padding a tiled partial row may return a view of values. Keep the owner
        # when aliasing is confirmed or cannot be disproved.
        if padded is not None and not _same_buffer(padded, values):
            padded.deallocate(True)
        idx = ttnn.argmax(u, dim=-1, keepdim=False)  # [1,1,32] uint32 RM
        u.deallocate(True)
        if rows < R32:
            sliced = ttnn.slice(idx, [0, 0, 0], [1, 1, rows])
            idx.deallocate(True)
            idx = sliced
        return idx

    # ── sampling: exact HF full-vocab reconstruction on host ──────────────────
    def to_host_full_vocab(self, pack, read_replica):
        """``CmeLogits`` -> torch [rows, vocab] identical to HF's masked output.

        ``read_replica`` is a callable that pulls one device tensor to host (the
        caller owns the TP replica choice). Reconstruction is exact, including
        ``mask_value = selected_logits.min() - 1`` computed over the whole row set
        the same way HF does it (a single scalar over the full tensor, not
        per-row).
        """
        vals = read_replica(pack.values).reshape(pack.rows, -1).float()
        ids = read_replica(pack.ids).reshape(pack.rows, -1).to(torch.int64)
        mask_value = vals.min().item() - 1.0
        full = torch.full((pack.rows, self.vocab_size), mask_value, dtype=vals.dtype)
        full.scatter_(-1, ids, vals)
        return full
