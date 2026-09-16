# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B GQA attention for chunked prefill (tt-blaze#4144).

    qkv       = x @ wqkv                          column-parallel, 4096 -> 768/chip at TP=8
    q, k, v   = split_heads(qkv)                   4 Q-heads + 1 KV-head + 1 KV-head per chip
    q, k      = rope(q), rope(k)                   Meta-interleaved, llama3-scaled
    write_kv_chunk(k, v)                           post-RoPE K, raw V, into the block-cyclic cache
    attn      = sdpa(q, cache_k, cache_v)          ring-joint over the SP axis, or plain causal
    out_full  = concat_heads(attn) @ o_proj        row-parallel, 512/chip -> 4096 partial
    out       = reduce_scatter(out_full, dim=-1)   -> 4096/tp per chip

**Layout contract**, matching ``tt/mlp.py`` exactly: ``forward`` consumes a *replicated*
``[1, 1, seq, emb_dim]`` activation (what the distributed ``attn_norm`` emits) and returns a
``reduce_scatter``-ed ``[1, 1, seq, emb_dim / tp]``, the layout the residual stream is already in.
Attention and the MLP having the same contract is what lets the decoder layer add both into the
residual without a collective at either boundary.

**Why TP=8 is the interesting case.** 32 Q-heads / 8 KV-heads over 8 columns puts 4 Q-heads and
exactly *one* KV head on each chip, so a KV chunk for a given layer lives on exactly one chip (see
``tt/config.py``). Every width divides cleanly — 4096/8 = 512 = 16 tiles — so none of the o_proj
tile-alignment padding gpt-oss carries (2880/8 = 360) is needed here.

**What Llama does not have.** No attention sinks, no sliding window, no QK-norm, and no q/k/v/o
biases. That is most of the difference from ``gpt_oss_d_p/tt/attention/``, which this follows
structurally: that package alternates sliding and full layers and therefore carries a second
circular KV cache, a compact-halo gather-buffer calculation, a per-layer ``layer_view`` remap, and
a learned per-head sink pre-divided by the softmax scale. All 32 Llama layers are identical
full-causal GQA, so all of it collapses — and it collapses into one flat module rather than a
multi-file package, matching ``tt/mlp.py`` / ``tt/rope.py`` / ``tt/kv_cache.py``.

**RoPE frame.** The device op is interleaved-only, so q/k projections are un-permuted out of the HF
half-split layout at load time (:func:`hf_to_meta_head_frame`) and rotated Meta-interleaved with the
adjacently-duplicated tables ``tt/rope.py`` builds. Attention *output* is unchanged by this choice —
the same permutation applies to q and k, and q·k is invariant under it — but the *stored* K is not,
and blaze decode reads K in the Meta frame. See ``tt/rope.py``'s module docstring for why getting
this wrong passes every byte-level migration gate.

Kept free of reference-model and safetensors imports so this module stays cheap to import; the
adapter's import-light contract is asserted by ``tests/unit/test_scaffold.py``.
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import Llama31KVCache, write_kv_chunk

# HiFi4 for the attention core. fp32_dest_acc_en MUST stay False: the ring-joint op's
# streaming-softmax compute requires it, and enabling it there is a silent wrong-answer rather than
# an error. Kept the same on the single-device path so the two paths are graded at one fidelity.
COMPUTE_KERNEL_CONFIG_HIFI4 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=False,
)

# The projections (QKV in, O out) are plain matmuls and carry none of that constraint, so they get
# fp32 accumulation. This is not a free knob -- it is what fixes V.
#
# Without it a 4096-long dot product accumulates in bf16. That is invisible in K, whose outputs run
# |max| 10-20, and severe in V, whose outputs run 0.5-2.5: a small result built from large operands
# means cancellation, and cancellation turns accumulator rounding into large *relative* error.
# Measured against the golden trace, V bottomed out at 0.9666 around layer 12 while the bf8 cache's
# own quantisation ceiling is 0.99996, so the cache dtype never was the limit -- this was. The error
# was uniform across all 8 KV heads and all token blocks, which is the signature of accumulator
# precision rather than a wiring fault.
#
# packer_l1_acc likewise, and this pairing is exactly what llama3_70b_galaxy uses for its HiFi4
# matmuls (``model_config.py``), so the production Llama on this hardware was already doing it.
COMPUTE_KERNEL_CONFIG_PROJECTIONS = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")

# SDPA chunking. Short sequences cannot fill a 256-row chunk, and over-chunking a short prefill
# wastes more in padding than it saves in scheduling.
_SDPA_CHUNK_THRESHOLD = 2048
_SDPA_CHUNK_SMALL = 32
_SDPA_CHUNK_LARGE = 256


def hf_to_meta_head_frame(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Un-permute a ``[n_heads * head_dim, emb_dim]`` q/k projection from HF to Meta head frame.

    HF ships q_proj/k_proj with each head's rows in half-split order ``[r0..r63, i0..i63]``,
    precisely so its ``rotate_half`` reproduces Meta's adjacent-pair rotation. The device op rotates
    adjacent pairs directly, so the rows have to be interleaved back to ``[r0, i0, r1, i1, ...]``.

    Equivalent to ``tt_transformers.load_checkpoints.reverse_permute``, and to
    ``reference/model.py:to_meta_frame`` applied along the head axis rather than the last axis.
    Restated here rather than imported: ``reference`` is off-limits to this module (import-light),
    and a copy graded against HF by ``test_attention_vs_ref`` is held in place by an external
    oracle. **v_proj and o_proj are NOT permuted** — RoPE never touches V, and permuting it (or
    o_proj's rows, which are indexed by the same head layout) would corrupt the output.
    """
    if w.shape[0] != n_heads * head_dim:
        raise ValueError(f"expected leading dim {n_heads * head_dim} (n_heads*head_dim), got {tuple(w.shape)}")
    emb_dim = w.shape[1]
    # [n_heads*head_dim, emb] -> [n_heads, emb, head_dim] so the interleave lands on head_dim, then back.
    per_head = w.reshape(n_heads, head_dim, emb_dim).transpose(-1, -2)
    half = head_dim // 2
    interleaved = torch.stack((per_head[..., :half], per_head[..., half:]), dim=-1).flatten(-2)
    return interleaved.transpose(-1, -2).reshape(n_heads * head_dim, emb_dim).contiguous()


def sdpa_program_config(mesh_device, seq_len: int, carve_ccl_column: bool = False) -> ttnn.SDPAProgramConfig:
    """SDPA program config for a prefill of ``seq_len`` rows.

    ``carve_ccl_column`` drops the last compute column, which the ring-joint path requires: its CCL
    workers live there (``CCLManager.ring_attention_ccl_core_grid_offset``) and the op asserts the
    compute and CCL core sets do not overlap.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    chunk = _SDPA_CHUNK_LARGE if seq_len >= _SDPA_CHUNK_THRESHOLD else _SDPA_CHUNK_SMALL
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid.x - 1 if carve_ccl_column else grid.x, grid.y),
        exp_approx_mode=False,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
    )


class TtLlamaAttention(LightweightModule):
    """Full-causal GQA attention, TP-sharded over ``mesh_config.tp_axis``."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        mesh_config,
        torch_weights: Optional[dict] = None,
        emb_dim: int = Llama31_8BConfig.EMB_SIZE,
        n_heads: int = Llama31_8BConfig.NUM_ATTENTION_HEADS,
        n_kv_heads: int = Llama31_8BConfig.NUM_KEY_VALUE_HEADS,
        head_dim: int = Llama31_8BConfig.HEAD_DIM,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        activations_dtype: ttnn.DataType = ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI4,
        projection_compute_kernel_config=COMPUTE_KERNEL_CONFIG_PROJECTIONS,
        hf_frame_weights: bool = True,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
    ):
        """
        Args:
            mesh_config: ``tt/config.py`` ``MeshConfig``. Supplies the TP degree and the axes TP and
                SP live on. Passed in rather than rebuilt so attention cannot end up sharded on a
                different axis than the MLP next to it.
            torch_weights: ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` in HF ``(out, in)``
                orientation, no biases. Random weights when omitted — shape bring-up only, never PCC.
            hf_frame_weights: ``q_proj`` / ``k_proj`` arrive in HF half-split head order and must be
                un-permuted for the interleaved device RoPE (see :func:`hf_to_meta_head_frame`). Set
                False only if the caller has already converted them.
            weights_dtype: ``bfloat16``. #4144's PCC floor is 0.99; ``bfloat8_b`` projections do not
                reliably clear it once the softmax amplifies the QK error.
        """
        super().__init__()

        tp = mesh_config.tp
        if mesh_device.shape[mesh_config.tp_axis] != tp:
            raise ValueError(
                f"mesh_config.tp({tp}) != mesh_device.shape[{mesh_config.tp_axis}]"
                f"({mesh_device.shape[mesh_config.tp_axis]}); the weight mappers shard the whole TP "
                f"axis, so a mismatch places data on devices the config does not know about."
            )
        if n_heads % n_kv_heads:
            raise ValueError(f"{n_heads} query heads do not group evenly over {n_kv_heads} KV heads")
        # Heads are the unit of TP here, not features: nlp_create_qkv_heads splits the per-chip fused
        # width into whole heads, so a TP degree that does not divide the KV head count cannot be
        # expressed at all. TP > n_kv_heads would need KV-head replication, which this model does not
        # do (and which would break the one-KV-head-per-chip property the migration layer relies on).
        for name, count in (("n_heads", n_heads), ("n_kv_heads", n_kv_heads)):
            if count % tp:
                raise ValueError(
                    f"{name}({count}) is not divisible by tp({tp}); TP shards whole attention heads, "
                    f"and replicating KV heads across chips is not supported"
                )
        if emb_dim % tp:
            raise ValueError(f"emb_dim({emb_dim}) is not divisible by tp({tp})")
        # 4096/8 = 512 = 16 whole tiles. Asserted rather than assumed: an off-tile o_proj output
        # shard does not fail, it silently pads, and gpt-oss needs a whole padding/slice path for
        # exactly this reason (2880/8 = 360). Llama should never reach that code.
        if mesh_config.shard_size(emb_dim) % ttnn.TILE_SIZE:
            raise ValueError(
                f"emb_dim({emb_dim}) / tp({tp}) = {mesh_config.shard_size(emb_dim)} is not a multiple "
                f"of the {ttnn.TILE_SIZE}-wide tile; this shape would be silently padded"
            )
        if n_heads * head_dim != emb_dim:
            # True for Llama-3.1-8B (32*128 == 4096) and assumed by o_proj's row-parallel sharding,
            # which shards the same 4096 as the Q-head projection.
            raise ValueError(f"n_heads({n_heads}) * head_dim({head_dim}) != emb_dim({emb_dim})")

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_local_heads = mesh_config.shard_size(n_heads)
        self.n_local_kv_heads = mesh_config.shard_size(n_kv_heads)
        self.emb_dim_per_chip = mesh_config.shard_size(emb_dim)
        self.scale = head_dim**-0.5
        self.num_links = num_links
        self.topology = topology
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config
        # Separate from the above on purpose: the SDPA/ring ops must keep fp32_dest_acc_en False.
        self.projection_compute_kernel_config = projection_compute_kernel_config
        self.weight_cache_path = weight_cache_path
        self.cache_name_prefix = cache_name_prefix

        if torch_weights is not None:
            missing = [p for p in PROJECTIONS if p not in torch_weights]
            if missing:
                raise ValueError(f"torch_weights is missing {missing}; expected all of {list(PROJECTIONS)}")
            q = torch_weights["q_proj"]
            k = torch_weights["k_proj"]
            v = torch_weights["v_proj"]
            o = torch_weights["o_proj"]
            for name, w, rows in (
                ("q_proj", q, n_heads * head_dim),
                ("k_proj", k, n_kv_heads * head_dim),
                ("v_proj", v, n_kv_heads * head_dim),
                ("o_proj", o, emb_dim),
            ):
                if tuple(w.shape) != (rows, emb_dim):
                    raise ValueError(f"{name} has shape {tuple(w.shape)}, expected {(rows, emb_dim)}")
            if hf_frame_weights:
                # Q and K only — see hf_to_meta_head_frame.
                q = hf_to_meta_head_frame(q, n_heads, head_dim)
                k = hf_to_meta_head_frame(k, n_kv_heads, head_dim)
            qkv_cat = self._fuse_qkv(q, k, v, tp)
            o_cat = o.T.contiguous()
        else:
            logger.warning("TtLlamaAttention built with random weights — shape bring-up only, not valid for PCC")
            qkv_cat = torch.randn(emb_dim, (n_heads + 2 * n_kv_heads) * head_dim, dtype=torch.float32) * 0.02
            o_cat = torch.randn(emb_dim, emb_dim, dtype=torch.float32) * 0.02

        # Fused QKV is column-parallel: each chip already holds its own [wq_i | wk_i | wv_i] block,
        # so sharding the last dim hands each chip exactly that. o_proj is row-parallel (input dim
        # sharded) so its partial sums reduce over the TP axis.
        self.wqkv = self._to_sharded_ttnn(qkv_cat, "wqkv", column_parallel=True)
        self.o_proj = self._to_sharded_ttnn(o_cat, "o_proj", column_parallel=False)

        logger.debug(
            f"TtLlamaAttention: emb_dim={emb_dim} n_heads={n_heads} n_kv_heads={n_kv_heads} "
            f"head_dim={head_dim} tp={tp} -> {self.n_local_heads}Q + {self.n_local_kv_heads}KV/chip, "
            f"fused qkv {qkv_cat.shape[-1] // tp}/chip, out {self.emb_dim_per_chip}/chip"
        )

    @staticmethod
    def _fuse_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tp: int) -> torch.Tensor:
        """Build the fused QKV weight as ``[emb_dim, tp * (q_local + k_local + v_local)]``.

        Per-device Q|K|V blocks are concatenated in device order, because ``ShardTensor2dMesh``
        splits the last dim into ``tp`` equal chunks and hands chunk ``i`` to device ``i``. A single
        ``cat([q, k, v])`` followed by sharding would instead give the first devices only Q, which
        produces plausible shapes and nonsense attention.
        """
        blocks = []
        for i in range(tp):
            blocks.append(
                torch.cat(
                    [
                        torch.chunk(q, tp, dim=0)[i].T,
                        torch.chunk(k, tp, dim=0)[i].T,
                        torch.chunk(v, tp, dim=0)[i].T,
                    ],
                    dim=-1,
                )
            )
        return torch.cat(blocks, dim=-1).contiguous()

    def _to_sharded_ttnn(self, torch_weight: torch.Tensor, name: str, column_parallel: bool) -> ttnn.Tensor:
        """Shard an ``(in, out)`` weight across the TP axis and move it to DRAM."""
        mesh_mapper = (
            self.mesh_config.column_parallel(self.mesh_device)
            if column_parallel
            else self.mesh_config.row_parallel(self.mesh_device)
        )
        cache_file_name = (
            str(self.weight_cache_path / f"{self.cache_name_prefix}.{name}")
            if self.weight_cache_path is not None and self.cache_name_prefix is not None
            else None
        )
        return ttnn.as_tensor(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=self.weights_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rope_mats,
        transformation_mat,
        *,
        kv_cache: Optional[Llama31KVCache] = None,
        ccl_manager=None,
        cache_layer_idx: int = 0,
        user_id: int = 0,
        cached_len: int = 0,
        indexed_rope: bool = False,
    ) -> ttnn.Tensor:
        """``x``: replicated ``[1, 1, seq, emb_dim]`` -> ``[1, 1, seq, emb_dim / tp]``.

        Args:
            rope_mats: ``(cos, sin)``. Per-chunk ``[1, 1, seq, head_dim]`` tables for the plain path,
                or — when ``indexed_rope`` — the whole-cache block-cyclic SP-sharded tables
                ``rope.build_indexed_rope`` builds once.
            kv_cache: written with post-RoPE K and raw V. Required for the sequence-parallel path;
                ``None`` only on the single-device unit-test path.
            cache_layer_idx: this layer's slot *within* ``kv_cache``, i.e. rank-local, not the
                layer's global index in the 32-layer stack. See ``tt/decoder.py`` for why the two
                are separate.
            cached_len: valid prefix length already in the cache *before* this chunk (0 = first).
            indexed_rope: use the on-device indexed RoPE, which derives this chunk's start row from
                ``cached_len`` + the device's SP coordinate instead of a per-chunk host reshard.
        """
        if x.shape[-1] != self.emb_dim:
            raise ValueError(
                f"input last dim {x.shape[-1]} != emb_dim {self.emb_dim}: this module expects the "
                f"replicated full-width activation that attn_norm emits, not a TP-sharded one"
            )
        if x.shape[0] != 1 or x.shape[1] != 1:
            # One user per call, matching write_kv_chunk. A packed batch would need the [B,1,S,·]
            # reshape gpt-oss does plus a per-user cache write loop; the prefill runtime drives one
            # user at a time, so accepting a batch here would only add an untested path.
            raise ValueError(f"expected [1, 1, seq, emb_dim], got {list(x.shape)}; loop over users instead")
        if x.dtype != self.activations_dtype:
            logger.warning(f"TtLlamaAttention: typecasting input {x.dtype} -> {self.activations_dtype}")
            x = ttnn.typecast(x, self.activations_dtype)

        sp = self.mesh_device.shape[self.mesh_config.sp_axis]
        seq_len = x.shape[-2]

        xqkv = ttnn.matmul(
            x, self.wqkv, dtype=self.activations_dtype, compute_kernel_config=self.projection_compute_kernel_config
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        q, k = self._apply_rope(q, k, rope_mats, transformation_mat, cached_len, indexed_rope)

        # Post-RoPE K and raw V into the packed cache at this chunk's offset. Single write point for
        # every chunk; the cache-read path below then reads the accumulated prefix. q/k/v stay live
        # for the SDPA that follows — write_kv_chunk casts its own copy to the cache dtype.
        if kv_cache is not None:
            write_kv_chunk(
                kv_cache,
                k,
                v,
                slot_idx=user_id,
                layer_idx=cache_layer_idx,
                kv_actual=cached_len,
                sp_axis=self.mesh_config.sp_axis,
            )

        attn = self._attend(q, k, v, kv_cache, ccl_manager, cache_layer_idx, user_id, cached_len, seq_len, sp)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        concat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        out_full = ttnn.matmul(
            concat,
            self.o_proj,
            dtype=self.activations_dtype,
            compute_kernel_config=self.projection_compute_kernel_config,
        )
        ttnn.deallocate(concat)

        if self.mesh_device.shape[self.mesh_config.tp_axis] == 1:
            return out_full

        # Row-parallel o_proj leaves each chip holding a partial sum over the full emb_dim.
        # reduce_scatter completes the sum and lands the result TP-sharded, which is the layout the
        # residual stream is in — the same contract tt/mlp.py returns under.
        if ccl_manager is not None:
            out = self.mesh_config.reduce_scatter(out_full, ccl_manager, dim=-1)
        else:
            out = ttnn.reduce_scatter(
                out_full,
                dim=-1,
                cluster_axis=self.mesh_config.tp_axis,
                num_links=self.num_links,
                topology=self.topology,
            )
        ttnn.deallocate(out_full)
        return out

    def _apply_rope(self, q, k, rope_mats, transformation_mat, cached_len: int, indexed_rope: bool):
        """Rotate Q and K in place of themselves, freeing the un-rotated inputs."""
        cos, sin = rope_mats[0], rope_mats[1]
        if cos.shape[-1] != self.head_dim:
            # The single trap tt/rope.py's docstring warns about: a table built at
            # n_heads * head_dim width spreads the wrong frequencies across the projection, is exact
            # at position 0, and is invisible to any device-vs-device comparison.
            raise ValueError(
                f"rope table width {cos.shape[-1]} != head_dim {self.head_dim}; cos/sin must be "
                f"built per head, not at n_heads * head_dim width"
            )

        def rotate(t):
            if indexed_rope:
                return ttnn.experimental.deepseek_prefill.rotary_embedding_indexed(
                    t,
                    cos,
                    sin,
                    transformation_mat,
                    kv_actual_global=cached_len,
                    cluster_axis=self.mesh_config.sp_axis,
                )
            return ttnn.experimental.rotary_embedding_llama(t, cos, sin, transformation_mat, is_decode_mode=False)

        q_rot, k_rot = rotate(q), rotate(k)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        return q_rot, k_rot

    def _attend(self, q, k, v, kv_cache, ccl_manager, cache_layer_idx, user_id, cached_len, seq_len, sp):
        """Full-causal GQA attention over the prefix, by whichever path the geometry allows."""
        # Sequence-parallel prefill reads the accumulated prefix out of the block-cyclic cache with
        # the ring-joint SDPA, which gathers K/V across the SP axis internally via online softmax
        # (no explicit all-gather). Valid from chunk 0 onward *provided* Q is shorter than the
        # cache: the ring reader requires that, and a one-shot request whose cache is exactly one
        # chunk long has Q and K/V equal, so it falls through to the gather path below.
        if sp > 1 and kv_cache is not None and (cached_len > 0 or kv_cache.max_seq_len > seq_len * sp):
            if ccl_manager is None:
                raise ValueError("sequence-parallel attention needs a ccl_manager")
            cache_k, cache_v, cache_batch_idx, cache_capacity = kv_cache.layer_view(user_id, cache_layer_idx)
            return self._ring_joint(
                q,
                cache_k,
                cache_v,
                ccl_manager=ccl_manager,
                kv_actual=cached_len,
                logical_n=cached_len + seq_len * sp,
                cache_global=cache_capacity,
                cache_batch_idx=cache_batch_idx,
                seq_len=seq_len,
            )

        if sp > 1:
            # No cache to read (or a one-shot request the ring reader rejects): gather Q/K/V across
            # the SP axis, run the exact full-sequence SDPA, and scatter the result back. Costs a
            # full-width K/V on every chip, which is why it is the bootstrap and not the main path.
            if ccl_manager is None:
                raise ValueError("sequence-parallel attention needs a ccl_manager")
            return self._gather_sdpa(q, k, v, ccl_manager, seq_len, sp)

        if cached_len > 0:
            # A single-device chunked read would need a chunk-position-aware SDPA: Q is the current
            # chunk at global offset cached_len while K/V span [0, cached_len + seq_len), so plain
            # is_causal SDPA — which assumes Q row 0 aligns with K row 0 — is off by cached_len and
            # silently wrong. The SP path above is the supported chunked read; fail loud here rather
            # than return a plausible wrong answer.
            raise NotImplementedError(
                f"single-device chunked attention (cached_len={cached_len}, sp=1) is not supported: "
                f"plain causal SDPA cannot express a Q offset. Run chunked prefill with sp > 1 (the "
                f"ring-joint cache read), or prefill this request in one shot."
            )

        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.scale,
            program_config=sdpa_program_config(self.mesh_device, seq_len),
            compute_kernel_config=self.compute_kernel_config,
        )

    def _ring_joint(
        self, q, cache_k, cache_v, *, ccl_manager, kv_actual, logical_n, cache_global, cache_batch_idx, seq_len
    ):
        """Cache-read ring-joint SDPA over the accumulated prefix ``[0, logical_n)``.

        Ported from ``gpt_oss_d_p/tt/attention/dense_sp.py`` with the sink and sliding-window
        arguments dropped. Llama passes no window, so the gather buffer always spans the whole
        per-device cache shard — gpt-oss's compact-halo sizing exists only for its sliding layers.
        """
        if cache_k.dtype != ttnn.bfloat8_b or cache_v.dtype != ttnn.bfloat8_b:
            raise ValueError(
                f"the ring cache read requires a bfloat8_b KV cache; got k={cache_k.dtype}, "
                f"v={cache_v.dtype}. The ring op's gather buffers are bf8."
            )
        out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            q,
            cache_k,
            cache_v,
            None,
            None,
            None,
            # Persistent ring-gather scratch, allocated once per shape and reused across every layer
            # and chunk. dtype must match the bf8 cache.
            persistent_output_buffer_k=ccl_manager.get_ring_gather_buffer(
                f"k_{cache_global}", self.n_kv_heads, cache_global, self.head_dim, ttnn.bfloat8_b
            ),
            persistent_output_buffer_v=ccl_manager.get_ring_gather_buffer(
                f"v_{cache_global}", self.n_kv_heads, cache_global, self.head_dim, ttnn.bfloat8_b
            ),
            joint_strategy="rear",
            logical_n=logical_n,
            program_config=sdpa_program_config(self.mesh_device, seq_len, carve_ccl_column=True),
            compute_kernel_config=self.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=ccl_manager.ring_attention_ccl_semaphore_handles,
            num_links=ccl_manager.num_links,
            cluster_axis=self.mesh_config.sp_axis,
            mesh_device=self.mesh_device,
            topology=ccl_manager.topology,
            ccl_core_grid_offset=ccl_manager.ring_attention_ccl_core_grid_offset,
            use_column_major_ccl=True,
            is_causal=True,
            scale=self.scale,
            is_balanced=False,
            # kv_cache.layer_view has already folded the layer into the batch index, matching
            # update_padded_kv_cache's write. Passing the bare user slot would make every layer read
            # layer 0's cache: correct by coincidence at layer 0, stale everywhere after.
            kv_cache_batch_idx=cache_batch_idx,
            kv_actual_isl=kv_actual,
        )
        return out

    def _gather_sdpa(self, q, k, v, ccl_manager, seq_len, sp):
        """SP bootstrap: all-gather Q/K/V on the sequence axis, exact SDPA, reduce-scatter back."""
        full_seq = seq_len * sp
        gathered = [self.mesh_config.allgather(t, ccl_manager, dim=2, axis=self.mesh_config.sp_axis) for t in (q, k, v)]
        out_full = ttnn.transformer.scaled_dot_product_attention(
            *gathered,
            is_causal=True,
            scale=self.scale,
            program_config=sdpa_program_config(self.mesh_device, full_seq),
            compute_kernel_config=self.compute_kernel_config,
        )
        for t in gathered:
            ttnn.deallocate(t)
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            out_full,
            dim=2,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=ccl_manager.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ccl_manager.topology,
            cluster_axis=self.mesh_config.sp_axis,
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
        )
        ttnn.deallocate(out_full)
        # reduce_scatter SUMS across the SP axis, but every chip computed the same full-sequence
        # output, so each row has been added sp times. Undo that rather than using an all-gather +
        # slice, which would move sp times as many bytes.
        out = ttnn.multiply(scattered, 1.0 / sp)
        ttnn.deallocate(scattered)
        return out
