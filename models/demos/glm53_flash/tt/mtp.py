# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MTP (multi-token prediction) module of GLM-5.3-Flash on device: checkpoint layer 45, the speculative drafter.

transformers has no MTP wrapper for ``glm5_next`` (``layers.45.*`` is dropped on load); the conventions are
those of vLLM's GLM-4.5 MTP / ``deepseek_v3_d_p/reference/glm_5_3/mtp.py`` and the CPU reference
``_agents/mtp_head/mtp_reference.py`` (checked against the HF classes' own forward of the block)::

    e      = enorm(embed_tokens[t_{i+1}])       # the NEXT token's embedding (first in the concat)
    h      = hnorm(H_i)                         # the main model's final hidden at row i: norm(mean(streams)),
                                                #   i.e. the LM-head input (``TtModelHead.norm`` output)
    x      = eh_proj(cat[e, h])                 # [8192] -> [4096]
    x     += DSA(input_layernorm(x))            # layer 45's own MLA + k-pool indexer on its OWN latent cache / index
    x     += MoE(post_attention_layernorm(x))   # 288 routed top-8 + 1 shared, the main model's router rules
    logits = lm_head(shared_head.norm(x))       # the main model's LM head; row i predicts t_{i+2}
    level k >= 2: H <- shared_head.norm(x) of level k-1, t <- draft_{k-1}, MTP position + 1

MTP positions: the level-1 row of main position ``p`` (hidden ``H_p``, token ``t_{p+1}`` = the greedy id of the
step at ``p``) sits at MTP position ``p``; level ``k`` of the chain sits at ``p + k - 1``. The MTP cache therefore
holds exactly one row per main position (the prompt rows from ``prefill_forward``, then one per decode step).

Layer 45 has no ``hc_*`` tensors: the block is a plain pre-norm residual block (no mHC wrap) and its residual is
one ``[1, 1, T, 4096]`` bf16 tensor, HF's own dtype for the block (the norms accumulate in fp32 inside the op).

Pieces reused from the main model: ``TtDSA`` (own ``TtLatentCache`` + ``TtIndexCache`` allocated here, MTP
positions), ``TtMoE`` (layer 45's experts, the same weight-cache conventions as ``tt/model.py``:
``<weight_cache_root>/layer45``), the shared ``TtEmbedding`` table (a plain gather, no stream expansion) and the
shared ``TtModelHead`` (``logits`` + ``greedy`` on the MTP's own ``shared_head.norm`` output). ``eh_proj`` is a
replicated ``DecodeLinear`` (v1 of the plan: 34 MB per chip, no collective).

Geometry: ``prefill_forward`` runs rows ``[start, start + T_pad)`` of one user (level 1 on a prompt or on every
verified row, the cache rows written at the MTP positions; on a multi-row mesh the rows hold slabs of the chunk,
the ``TtDecoderLayer`` v1 sequence-parallel form); ``decode_forward`` runs one row per user slot at the persistent
``DecodePositions`` (the draft step: MTP position = the main model's position + level - 1). Everything in
``decode_forward`` is a device op on persistent inputs, the trace contract of ``TtDSA.decode_forward``.
``block_rows`` runs the residual block with a caller-supplied attention (the speculative verify / chain rows of
``tt/spec_decode.py``, which address the cache with their own page tables and skip the index maintenance).
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import (
    SparseDecodeMeta,
    TtDSA,
    TtLatentCache,
    _meta_mapper,
    decode_position_tensors,
    sparse_decode_enabled,
)
from models.demos.glm53_flash.tt.embedding import TtEmbedding
from models.demos.glm53_flash.tt.functional_decoder import DecodePositions
from models.demos.glm53_flash.tt.matmul_configs import DecodeLinear
from models.demos.glm53_flash.tt.mesh import GLMMesh, single_chip
from models.demos.glm53_flash.tt.model_head import TtModelHead
from models.demos.glm53_flash.tt.moe import TtMoE

TILE = 32
MTP_LAYER = 45  # checkpoint layer index of the (single) MTP module: num_nextn_predict_layers = 1
_EXPERT_RE = re.compile(r"^mlp\.experts\.\d+\.")
# the layer-45 tensors the block needs besides ``self_attn.*`` / ``mlp.*``
GLUE_KEYS = ("enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight")
NORM_KEYS = ("input_layernorm.weight", "post_attention_layernorm.weight")


def _strip(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


class TtMTPBlock(LightweightModule):
    """Layer 45 as the drafter: glue (enorm / hnorm / eh_proj) -> DSA -> MoE -> shared_head.norm -> LM head."""

    def __init__(
        self,
        device,
        cfg: GLM53FlashConfig,
        state_dict: dict[str, torch.Tensor],
        *,
        embedding: TtEmbedding,
        head: TtModelHead,
        max_batch: int,
        max_seq_len: int,
        block_size: int = TILE,
        weight_cache_path: Path | str | None = None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        mesh: GLMMesh | None = None,
        partition_users: bool = False,
    ):
        """``state_dict``: the raw checkpoint tensors of ``model.language_model.layers.45.`` (prefix stripped, fp8
        groups dequantized: ``reference/checkpoint.load_layer_raw(snapshot, 45)``); the ``mlp.experts.N.*`` tensors
        may be absent when the expert cache under ``weight_cache_path`` is complete (``TtMoE.expert_cache_complete``).
        ``embedding`` / ``head``: the main model's (shared) table and head. ``max_batch`` user slots per mesh row and
        ``max_seq_len`` MTP positions per slot size the block's own latent cache + pooled-key index (allocated here,
        before any trace capture). ``partition_users``: the user-partitioned cache of ``FunctionalDecoder`` (decode
        through the traced sparse path only)."""
        self.device = device
        self.mesh = mesh if mesh is not None else single_chip(device)
        self.cfg = cfg
        self.layer_idx = MTP_LAYER
        self.hidden = cfg.hidden_size
        self.embedding = embedding
        self.head = head
        self.max_batch = max_batch
        self.total_users = max_batch * self.mesh.rows
        self.max_seq_len = max_seq_len
        self.partition = bool(partition_users) and self.mesh.cols > 1
        sd = state_dict
        missing = [k for k in GLUE_KEYS + NORM_KEYS if k not in sd]
        if missing:
            raise KeyError(f"layer {MTP_LAYER} state dict lacks {missing}")
        H = self.hidden
        assert tuple(sd["eh_proj.weight"].shape) == (H, 2 * H), sd["eh_proj.weight"].shape
        up = self.mesh.upload
        norm_w = lambda key: up(sd[key].float().reshape(1, 1, 1, -1), ttnn.bfloat16)  # bf16 weights, as the layers
        self.enorm_w = norm_w("enorm.weight")
        self.hnorm_w = norm_w("hnorm.weight")
        self.input_norm_w = norm_w("input_layernorm.weight")
        self.post_norm_w = norm_w("post_attention_layernorm.weight")
        self.shared_norm_w = norm_w("shared_head.norm.weight")
        # eh_proj: nn.Linear [4096, 8192] -> device [8192, 4096], replicated (plan 2a v1: 34 MB/chip, no CCL);
        # the concat order is [enorm(e) | hnorm(h)], i.e. the weight's input columns [0, H) act on the embedding
        self.w_eh = up(sd["eh_proj.weight"].t(), weight_dtype)
        self.lin_eh = DecodeLinear(device, self.w_eh)
        self.attn = TtDSA(
            device, cfg, _strip(sd, "self_attn."), layer_idx=MTP_LAYER, weight_dtype=weight_dtype, mesh=self.mesh
        )
        self.mlp = TtMoE(
            device,
            cfg,
            _strip(sd, "mlp."),
            weight_dtype=weight_dtype,
            expert_dtype=expert_dtype,
            expert_down_dtype=expert_down_dtype,
            mesh=self.mesh,
            weight_cache_path=weight_cache_path,
        )
        # the MTP's own autoregressive state: latent cache + pooled-key index (``TtLatentCache.index``)
        self.cache = TtLatentCache(
            device,
            cfg,
            max_batch=max_batch,
            max_seq_len=max_seq_len,
            block_size=block_size,
            mesh=self.mesh,
            partition_users=partition_users,
        )
        self.next_start: list[int] = [0] * max_batch
        # norms: HiFi4 with fp32 accumulation (``TtDecoderLayer.norm_compute``); the glue matmul: the bfp8 policy
        self.norm_compute = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.hifi2 = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_checkpoint(
        cls,
        device,
        cfg: GLM53FlashConfig,
        snapshot: Path | str | None = None,
        headers=None,
        *,
        raw: dict[str, torch.Tensor] | None = None,
        embedding: TtEmbedding | None = None,
        head: TtModelHead | None = None,
        weight_cache_root: Path | str | None = None,
        mesh: GLMMesh | None = None,
        expert_dtype=ttnn.bfloat8_b,
        expert_down_dtype=ttnn.bfloat8_b,
        **kw,
    ) -> "TtMTPBlock":
        """Build from the checkpoint: layer 45's raw tensors (``raw`` when the caller already holds them, e.g. a
        test that feeds the same dict to the CPU reference), the embedding and head built here unless given (the
        resident model passes its own). ``weight_cache_root``: the model's tensorbin root (``tt/model.py`` /
        ``mesh_model.py`` convention: ``<root>/layer45`` holds the routed experts; with a complete expert cache
        the experts are neither read nor dequantized)."""
        snapshot = Path(snapshot) if snapshot else default_snapshot_dir()
        headers = headers or shard_headers(snapshot)
        mesh = mesh if mesh is not None else GLMMesh.from_device(device)
        cache_dir = Path(weight_cache_root) / f"layer{MTP_LAYER}" if weight_cache_root else None
        if raw is None:
            hit = TtMoE.expert_cache_complete(cache_dir, mesh, expert_dtype, expert_down_dtype)
            name_filter = (lambda n: not _EXPERT_RE.match(n)) if hit else None
            raw = load_layer_raw(snapshot, MTP_LAYER, headers, name_filter=name_filter)
        if embedding is None:
            embedding = TtEmbedding.from_checkpoint(device, cfg, snapshot, headers)
        if head is None:
            head = TtModelHead.from_checkpoint(device, cfg, snapshot, headers, mesh=mesh)
        return cls(
            device,
            cfg,
            raw,
            embedding=embedding,
            head=head,
            weight_cache_path=cache_dir,
            mesh=mesh,
            expert_dtype=expert_dtype,
            expert_down_dtype=expert_down_dtype,
            **kw,
        )

    # ------------------------------------------------------------------ host <-> device helpers
    def tokens_to_device(self, ids: torch.Tensor, pad_to: int | None = None) -> ttnn.Tensor:
        """Next-token ids (any shape) -> the ``[1, 1, 1, T]`` uint32 ROW_MAJOR tensor ``embed`` consumes (single
        mesh row; a multi-row model shards its own ids as ``GLM53MeshModel.ids_to_device`` does)."""
        return self.embedding.tokens_to_device(ids, pad_to)

    def hidden_to_device(self, hidden: torch.Tensor, pad_to: int | None = None) -> ttnn.Tensor:
        """Host hidden rows ``[T, H]`` (the main model's normed final hidden) -> ``[1, 1, T_pad, H]`` bf16 TILE."""
        T, H = hidden.shape
        T_pad = T if pad_to is None else pad_to
        assert T_pad >= T, (T_pad, T)
        h = torch.zeros(1, 1, T_pad, H, dtype=torch.bfloat16)
        h[0, 0, :T] = hidden.to(torch.bfloat16)
        return ttnn.from_torch(
            h,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=self.mesh.replicate() if not self.mesh.is_single else None,
        )

    # ------------------------------------------------------------------ positions (the MTP's own: p + level)
    def make_decode_positions(self, host_positions: list[int]) -> DecodePositions:
        """Persistent ``[B]`` int32 position tensors for the MTP cache (mirrors
        ``FunctionalDecoder.make_decode_positions``; one entry per user slot, ``rows * max_batch`` on a mesh)."""
        if len(host_positions) != self.total_users:
            raise ValueError(f"{len(host_positions)} positions for {self.total_users} user slots")
        P = self.cfg.index_kpool
        pos, pool_ids, ring_slots = decode_position_tensors(
            self.device, host_positions, P, mesh=self.mesh, partition=self.partition
        )
        dense = None
        if not self.partition:
            dense = decode_position_tensors(self.device, self._dense_bound(host_positions), P, mesh=self.mesh)[0]
        sparse = SparseDecodeMeta.from_positions(
            self.device, host_positions, self.cfg, mesh=self.mesh, partition=self.partition
        )
        return DecodePositions(pos, pool_ids, ring_slots, list(host_positions), dense, sparse)

    def _dense_bound(self, host_positions: list[int]) -> list[int]:
        return [min(p, self.attn.dense_exact_max_pos) for p in host_positions]

    def update_decode_positions(self, dp: DecodePositions, host_positions: list[int]):
        """Refresh the persistent position tensors in place (trace-replay safe)."""
        if len(host_positions) != self.total_users:
            raise ValueError(f"{len(host_positions)} positions for {self.total_users} user slots")
        P = self.cfg.index_kpool
        mapper = _meta_mapper(self.mesh, self.partition)
        updates = [
            (dp.positions, host_positions),
            (dp.pool_ids, [p // P for p in host_positions]),
            (dp.ring_slots, [p % P for p in host_positions]),
        ]
        if dp.dense_positions is not None:
            updates.append((dp.dense_positions, self._dense_bound(host_positions)))
        if dp.sparse is not None:
            dp.sparse.update(host_positions, self.cfg, mesh=self.mesh, partition=self.partition)
        for t, vals in updates:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(
                    torch.tensor(vals, dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=mapper,
                ),
                t,
            )
        dp.host = list(host_positions)

    # ------------------------------------------------------------------ pieces
    def _norm(self, h: ttnn.Tensor, w: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(h, epsilon=self.cfg.rms_norm_eps, weight=w, compute_kernel_config=self.norm_compute)

    def embed(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, 1, T]`` uint32 ids -> ``[1, 1, T, H]`` bf16 TILE embedding rows (the gather of
        ``TtEmbedding.forward`` without the 4-stream expansion; bf16 like HF's ``embed_tokens``)."""
        T = int(ids.shape[-1])
        e = ttnn.embedding(ids, self.embedding.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        e = ttnn.unsqueeze_to_4D(e)
        if list(e.shape) != [1, 1, T, self.hidden]:
            e = ttnn.reshape(e, [1, 1, T, self.hidden])
        return e

    def project(self, ids: ttnn.Tensor, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """The glue: ``eh_proj(cat[enorm(embed(ids)), hnorm(hidden)])``. ``ids`` ``[1, 1, 1, T]`` uint32 (the NEXT
        token of every row), ``hidden`` ``[1, 1, T, H]`` bf16 or fp32 (row i = the main model's normed final hidden
        at position i, or the previous level's ``shared_head.norm`` output). Returns ``[1, 1, T, H]`` bf16."""
        T = int(ids.shape[-1])
        assert list(hidden.shape) == [1, 1, T, self.hidden], (list(hidden.shape), T)
        e = self.embed(ids)
        en = self._norm(e, self.enorm_w)
        ttnn.deallocate(e)
        hn = self._norm(hidden, self.hnorm_w)
        if hn.dtype != ttnn.bfloat16:
            hb = ttnn.typecast(hn, ttnn.bfloat16)
            ttnn.deallocate(hn)
            hn = hb
        eh = ttnn.concat([en, hn], dim=-1)  # [1, 1, T, 2H]: embedding first
        ttnn.deallocate(en)
        ttnn.deallocate(hn)
        x = self.lin_eh(eh, dtype=ttnn.bfloat16, compute_kernel_config=self.hifi2)
        ttnn.deallocate(eh)
        return x

    def _residual_block(self, x: ttnn.Tensor, attention, *, prefill: bool) -> ttnn.Tensor:
        """``x += attention(input_layernorm(x)); x += MoE(post_attention_layernorm(x))`` on ``[1, 1, T, H]`` bf16."""
        h = self._norm(x, self.input_norm_w)
        a = attention(h)
        ttnn.deallocate(h)
        x1 = ttnn.add(x, a)
        ttnn.deallocate(a)
        ttnn.deallocate(x)
        h = self._norm(x1, self.post_norm_w)
        m = self.mlp(h, prefill=True) if prefill else self.mlp(h)
        ttnn.deallocate(h)
        out = ttnn.add(x1, m)
        ttnn.deallocate(m)
        ttnn.deallocate(x1)
        return out

    def block_rows(self, x: ttnn.Tensor, attention) -> ttnn.Tensor:
        """The residual block on decode-shaped rows ``x`` ``[1, 1, R, H]`` bf16 (the glue output; consumed) with a
        caller-supplied ``attention(h)`` (``h`` = ``input_layernorm(x)`` bf16, returns ``[1, 1, R, H]``). No batch
        or position bookkeeping: the speculative verify (``B (K+1)`` candidate rows) and the draft chain of
        ``tt/spec_decode.py`` bring their own page tables / positions."""
        return self._residual_block(x, attention, prefill=False)

    def prefill_rows(self, x: ttnn.Tensor, *, user: int, seq_len: int, start: int = 0) -> ttnn.Tensor:
        """The block on prefill-shaped rows ``x`` bf16 (the glue output; consumed): row i is MTP position
        ``start + i`` of ``user``; the latent rows / index of ``[start, start + seq_len)`` are written into the
        block's own cache (``TtDSA.prefill_forward`` contract: ``T_pad % 32 == 0``, ``start % 32 == 0``, chunks of a
        user in order). Rows past ``seq_len`` come back unspecified. On a multi-row mesh ``x`` is THIS row's slab
        ``[1, 1, T_pad / rows, H]`` of the chunk (row r holds rows ``[r T/rows, (r+1) T/rows)``, the layout the
        main model's prefill leaves its streams in): the attention gathers the slabs over the rows, every row
        fills the replicated cache / index identically and keeps its own slab of the output through the
        reduce-scatter of ``rows`` equal copies (``TtDecoderLayer.prefill_forward`` v1, exact: a power-of-two sum
        of equal bf16 values scaled back by ``1 / rows``); the glue and the MoE run on the slab."""
        rows = self.mesh.rows
        T = int(x.shape[-2]) * rows
        if T % TILE or seq_len > T or seq_len <= 0:
            raise ValueError(f"T_pad={T} must be a multiple of {TILE} and 0 < seq_len={seq_len} <= T_pad")
        if user >= self.max_batch:
            raise ValueError(f"user {user} >= max_batch {self.max_batch}")
        if start < 0 or start % self.cache.block_size:
            raise ValueError(f"chunk start {start} must be a non-negative multiple of {self.cache.block_size}")
        if start + seq_len > self.max_seq_len:
            raise ValueError(f"start {start} + seq_len {seq_len} > max_seq_len {self.max_seq_len}")
        if start and start != self.next_start[user]:
            raise ValueError(f"user {user}: chunk start {start} is not contiguous (expected {self.next_start[user]})")
        with self.mesh.prefill_scope():
            return self._prefill_rows_impl(x, user=user, seq_len=seq_len, start=start, rows=rows)

    def _prefill_rows_impl(self, x: ttnn.Tensor, *, user: int, seq_len: int, start: int, rows: int) -> ttnn.Tensor:
        if rows == 1:
            attention = lambda h: self.attn.prefill_forward(h, self.cache, user, seq_len, start=start)
        else:

            def attention(h):  # h: this row's slab [1,1,T/rows,H] bf16 (input_layernorm applied)
                h_full = self.mesh.all_gather_rows(h, dim=2)  # [1,1,T,H], rows in mesh-row order
                y_full = self.attn.prefill_forward(h_full, self.cache, user, seq_len, start=start)
                ttnn.deallocate(h_full)
                y = self.mesh.reduce_scatter_rows(y_full, dim=2)  # rows x this row's slab
                ttnn.deallocate(y_full)
                return ttnn.multiply(y, 1.0 / rows, output_tensor=y)

        out = self._residual_block(x, attention, prefill=True)
        self.next_start[user] = start + seq_len
        return out

    def prefill_cache(
        self, ids: ttnn.Tensor, hidden: ttnn.Tensor, *, user: int, seq_len: int, start: int = 0
    ) -> ttnn.Tensor:
        """``prefill_forward`` without the head: fills the MTP cache / index of ``user`` for the prompt rows (the
        speculative decoder's prefill; the drafts of the prompt rows are not needed). ``ids`` / ``hidden`` are this
        mesh row's slab (``[1, 1, 1, T_loc]`` / ``[1, 1, T_loc, H]``) on a multi-row mesh. Returns the block output
        (the caller deallocates it)."""
        x = self.project(ids, hidden)
        return self.prefill_rows(x, user=user, seq_len=seq_len, start=start)

    @staticmethod
    def ids_row(ids: ttnn.Tensor) -> ttnn.Tensor:
        """Greedy ids ``[1, 1, T, 1]`` uint32 ROW_MAJOR (``TtModelHead.greedy``) -> the ``[1, 1, 1, T]`` id row
        ``embed`` consumes (same element order; a device reshape, trace-safe). The result may alias ``ids``: never
        deallocate it, deallocate ``ids`` after the embedding consumed the row."""
        T = int(ids.shape[-2])
        return ttnn.reshape(ids, [1, 1, 1, T])

    def decode_rows(self, x: ttnn.Tensor, positions: DecodePositions) -> ttnn.Tensor:
        """The block on one row per user slot ``x`` ``[1, 1, B, H]`` bf16 (the glue output; consumed) at the MTP
        positions ``positions`` (``make_decode_positions``; the latent row is appended and the index maintained)."""
        B = int(x.shape[-2])
        if B != self.max_batch:
            raise ValueError(f"decode batch {B} must equal max_batch {self.max_batch} (fixed user slots)")
        if any(p >= self.max_seq_len for p in positions.host):
            raise ValueError(f"positions {positions.host} exceed max_seq_len {self.max_seq_len}")

        def attention(h):
            return self.attn.decode_forward(
                h,
                self.cache,
                positions.positions,
                positions.pool_ids,
                positions.ring_slots,
                positions.host,
                dense_positions=positions.dense_positions,
                sparse_meta=positions.sparse if sparse_decode_enabled() else None,
            )

        out = self._residual_block(x, attention, prefill=False)
        self.next_start = [p + 1 for p in positions.host]
        return out

    def head_forward(self, out: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Block output ``[1, 1, T, H]`` bf16 -> ``(normed, logits, draft ids)``: ``shared_head.norm`` (bf16, the
        next level's hidden input), the shared LM head's logits ``[1, 1, T, V]`` bf16 and its greedy ids
        ``[1, 1, T, 1]`` uint32."""
        y = self._norm(out, self.shared_norm_w)
        logits = self.head.logits(y)
        ids = self.head.greedy(logits)
        return y, logits, ids

    # ------------------------------------------------------------------ entry points
    def prefill_forward(
        self, ids: ttnn.Tensor, hidden: ttnn.Tensor, *, user: int, seq_len: int, start: int = 0
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Level 1 on ``T_pad`` rows of one user: ``ids`` ``[1, 1, 1, T_pad]`` uint32 = the next token of every
        row (``t_{i+1}``), ``hidden`` ``[1, 1, T_pad, H]`` = the main model's normed hidden at row i. Returns
        ``(block out, normed, logits, draft ids)``, row i's draft predicting ``t_{i+2}``."""
        x = self.project(ids, hidden)
        out = self.prefill_rows(x, user=user, seq_len=seq_len, start=start)
        return (out,) + self.head_forward(out)

    def decode_forward(
        self, ids: ttnn.Tensor, hidden: ttnn.Tensor, positions: DecodePositions
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """One draft level for ``B`` user slots: ``ids`` ``[1, 1, 1, B]`` uint32 (level 1: the main model's new
        greedy token; level k: the level-(k-1) draft), ``hidden`` ``[1, 1, B, H]`` (level 1: the head's normed
        hidden; level k: the previous level's ``normed``), ``positions``: the MTP positions (main position +
        level - 1). Returns ``(block out, normed, logits, draft ids)``; ``normed`` and the ids feed the next level."""
        x = self.project(ids, hidden)
        out = self.decode_rows(x, positions)
        return (out,) + self.head_forward(out)
