# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full TTNN GLM-4.7-Flash model (``Glm4MoeLiteForCausalLM``) for one Blackhole chip.

Wraps the optimized-decoder stack with the model-level pieces of the HF
autoregressive path: token embeddings, the 47-layer decoder stack (layer 0
dense, layers 1..46 MoE), the final RMS norm, and the LM head. The MTP head
(``num_nextn_predict_layers`` = 1, checkpoint layer 47) is dropped exactly as
``Glm4MoeLiteForCausalLM._keys_to_ignore_on_load_unexpected`` drops it.

Deployment target
=================

A SINGLE Blackhole p150-class chip (1x1 mesh, 11x10 compute grid, 8 DRAM
banks, 31.5 GiB allocatable DRAM measured). Routed expert weights are
``bfloat4_b``: that is what makes the 30.6B checkpoint fit one card
(doc/probe/README.md). Every other dtype/fidelity/layout decision is the
optimized-decoder deployment policy, carried forward unchanged:

===============================  ==================================
tensor group                     dtype
===============================  ==================================
activations / residual / norms   bfloat16 (norms HiFi4 + fp32 acc)
attention decode copies + kv_b   bfloat4_b, LoFi
attention prefill flat copies    bfloat8_b, HiFi2 + fp32 acc
shared-expert weights            bfloat4_b
dense-MLP weights                bfloat8_b (bf4 measured and rejected)
routed experts                   bfloat4_b, LoFi
router gate                      float32, HiFi4 + fp32 acc
paged latent KV cache            bfloat8_b (bfloat16 supported)
token embedding table            bfloat16 (ROW_MAJOR, ttnn.embedding)
LM head                          bfloat8_b, HiFi2
===============================  ==================================

Model-level additions over the decoder stage
============================================

* ``ttnn.embedding`` over a ``[1, 1, vocab, hidden]`` ROW_MAJOR bf16 table;
  the decode token input stays a rank-4 persistent device tensor so the
  sampler can write the next token straight into it (``tt_out_tok``).
* One shared ``RopeSetup`` for all layers. Every layer builds an identical
  cos/sin table (25.9 MiB each at the advertised 202752 context); keeping 47
  copies would cost 2.4 GiB of DRAM for byte-identical data, so layers 1..N
  release theirs and rebind to layer 0's. Values are unchanged.
* Final RMS norm: width-sharded on the decoder residual grid at decode (the
  decoder returns the residual width-sharded, and that same shard is the LM
  head's DRAM-fed in0 raster), interleaved at prefill.
* LM head: one wide-1D mcast matmul over the full 11x10 grid
  (``per_core_N`` = 44 of the 4840 vocab tiles). Measured on this chip at
  M = 1 tile and recorded in doc/full_model/head_probe.json: 879 us at bf8
  with the shipped ``in0_block_w`` of 4, against 2472 us with the default (no
  program config) matmul, i.e. 2.8x. The explicit program config is therefore mandatory rather than a
  tuning nicety. (An earlier revision of this docstring said 15310 us and 17x;
  that figure was taken during bring-up with a different output memory config
  and does not reproduce. See work log FM-019.)
* Device-side decode state: one persistent position tensor. The captured graph
  advances it with ``ttnn.plus_one(..., skip_negative_entries=True)`` and
  *derives* the RoPE index from it every step
  (:meth:`GLM47FlashModel.decode_rope_indices`), so a fixed-step decode loop
  performs no per-token host refresh, position coherence is structural, and an
  inactive ``-1`` slot stays pinned at RoPE index 0 instead of walking past the
  end of the cos/sin table.

Prefill/decode contract
=======================

``prefill_forward(input_ids, *, kv_cache, page_table, user_id, seq_len, ...)``
    ``input_ids``: torch int tensor ``[S]`` or ``[1, S]``, or a device uint32
    tensor. Any logical ``1 <= S <= max_seq_len`` including lengths that are
    not multiples of the tile, paged block, or prefill chunk size: padding,
    masking, cache fill and output slicing are internal. Fills the user's rows
    of every layer's paged latent cache for positions ``[0, S)``.

``ttnn_decode_forward(tokens, cur_pos, page_table, kv_cache, *, rot_idxs=None)``
    Device-only decode step over persistent device tensors, returning
    sampler-ready logits ``[1, 1, 32, vocab]``. ``rot_idxs`` defaults to the
    index derived from ``cur_pos`` on device; a caller that needs decoupled
    RoPE positions may pass its own. Trace-capturable: no host reads/writes,
    no torch, static shapes.

Both paths keep the decoder's own paged-latent-cache contract (block size 64,
entry width 576 = kv_lora_rank 512 + qk_rope_head_dim 64, one KV head,
``paged_fill_cache`` / ``paged_update_cache`` /
``chunked_flash_mla_prefill`` / ``paged_flash_multi_latent_attention_decode``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.functional_decoder import TILE, PagedCacheConfig, _ck
from models.autoports.zai_org_glm_4_7_flash.tt.optimized_decoder import OptimizedDecoder, _mcast_2d_pc, _rect_grid
from models.autoports.zai_org_glm_4_7_flash.tt.provenance import source_manifest  # noqa: F401

DEFAULT_HF_MODEL_ID = "zai-org/GLM-4.7-Flash"
#: Env var pointing at a local HF snapshot directory (same knob the decoder tests use).
SNAPSHOT_ENV = "GLM47_FLASH_SNAPSHOT"
#: ``ttnn.sampling`` takes one logits row per user and ``TTSampling`` floors its
#: batch to a full tile, so the terminal path always emits 32 logits rows.
SAMPLER_ROWS = TILE


# --------------------------------------------------------------------- checkpoint access


def resolve_checkpoint_dir(checkpoint_dir=None, hf_model_id: str = DEFAULT_HF_MODEL_ID) -> Path:
    """Locate the HF snapshot directory holding ``model.safetensors.index.json``.

    Order: explicit argument, ``GLM47_FLASH_SNAPSHOT``, then the local HF hub
    cache for ``hf_model_id``. Never downloads.
    """
    candidates = []
    if checkpoint_dir:
        candidates.append(Path(checkpoint_dir))
    env = os.environ.get(SNAPSHOT_ENV)
    if env:
        candidates.append(Path(env))
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    repo = hub / ("models--" + hf_model_id.replace("/", "--"))
    if repo.is_dir():
        snaps = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
        candidates.extend(reversed(snaps))
    for cand in candidates:
        if (cand / "model.safetensors.index.json").is_file():
            return cand
    raise FileNotFoundError(
        f"No HF snapshot with model.safetensors.index.json found for {hf_model_id}. "
        f"Tried: {[str(c) for c in candidates]}. Set {SNAPSHOT_ENV} to the snapshot directory."
    )


class ShardedCheckpoint:
    """Lazy fp32 reader over a sharded safetensors checkpoint.

    Keeps one open handle per shard file so a layer's ~1.2 GB of expert
    tensors are read in a single pass. All torch usage lives at this
    weight-loading boundary; the TTNN runtime path stays torch-free.
    """

    def __init__(self, snapshot_dir: Path):
        self.dir = Path(snapshot_dir)
        with open(self.dir / "model.safetensors.index.json") as f:
            self.weight_map = json.load(f)["weight_map"]
        self._handles = {}

    def _handle(self, shard):
        from safetensors import safe_open

        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(str(self.dir / shard), framework="pt")
            handle.__enter__()
            self._handles[shard] = handle
        return handle

    def has(self, key: str) -> bool:
        return key in self.weight_map

    def get(self, key: str):
        import torch

        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"{key} not in checkpoint index ({self.dir})")
        return self._handle(shard).get_tensor(key).to(torch.float32)

    def layer_state_dict(self, layer_idx: int) -> dict:
        """Per-layer state dict with keys relative to ``model.layers.<i>.``."""
        prefix = f"model.layers.{layer_idx}."
        keys = [k for k in self.weight_map if k.startswith(prefix)]
        # Group by shard so each file is touched once.
        by_shard = {}
        for key in keys:
            by_shard.setdefault(self.weight_map[key], []).append(key)
        out = {}
        for shard, shard_keys in by_shard.items():
            handle = self._handle(shard)
            for key in shard_keys:
                import torch

                out[key[len(prefix) :]] = handle.get_tensor(key).to(torch.float32)
        return out

    def close(self):
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()


def load_hf_config(snapshot_dir):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(snapshot_dir), local_files_only=True)
    cfg._attn_implementation = "eager"
    return cfg


# --------------------------------------------------------------------- LM head config


def _lm_head_1d_pc(nt, kt, cores, in0_block_w):
    """Wide-1D mcast LM-head matmul config.

    Both parameters come from ``probe/full_model_head_probe.py``, recorded in
    ``doc/full_model/head_probe.json``. 110 cores beats 88 and 64 at both
    dtypes (64 does not even fit at bf8). ``in0_block_w`` was swept over every
    legal divisor of ``kt`` = 64 that the helper can express: at bf8, 1 and 2
    tie around 866-869 us and it degrades monotonically to 894 us at 8; 16, 32
    and 64 do not run at all, failing with a static circular buffer / L1 clash
    (``program.cpp:1875``), which is the op-contract blocker for the larger
    divisors.

    **4 is shipped, not the 869 us optimum, and that is a measured choice.**
    ``in0_block_w`` sets the K-blocking of the LM-head matmul, so it changes
    the accumulation order and therefore the bf16 rounding of the logits.
    Running the readiness gates at 2 costs real accuracy on real weights:
    prefill top-1 0.880 -> 0.830 and teacher-forced top-1 0.850 -> 0.790 with
    top-5 dropping 1.000 -> 0.990 (``logs/run_*_lm_head_bw2.log`` against the
    committed ``accuracy.json``). The 10 us/token it buys is 0.04% of the
    token-out step, so the trade is refused. Work log FM-021.
    """
    per_core_n = -(-nt // cores)
    blocks = -(-nt // per_core_n)
    cols, rows = _rect_grid(blocks)
    in0_block_w = max(d for d in range(1, min(kt, in0_block_w) + 1) if kt % d == 0)
    osw = max(d for d in (1, 2, 4) if per_core_n % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
        in0_block_w=in0_block_w,
        out_subblock_h=1,
        out_subblock_w=osw,
        out_block_h=1,
        out_block_w=per_core_n,
        per_core_M=1,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )


# --------------------------------------------------------------------- shared-rope decoder


class SharedRopeDecoder(OptimizedDecoder):
    """``OptimizedDecoder`` that consumes one model-level decode RoPE lookup.

    All layers share one ``RopeSetup``, so ``ttnn.embedding(rot_idxs, cos)``
    returns byte-identical values in every layer. On the TILE-layout table the
    lookup also *scales with table height*: measured 24.9 us at max_pos 4096
    but 209.5 us at the advertised 202752
    (``probe/decode_cache_scaling_probe.py`` companion sweep), so running it 94
    times per decode step would cost 19.7 ms/token - more than the whole
    47-layer decoder stack. The model does the lookup once per step from
    ROW_MAJOR tables (a flat 16 us at any height) and stashes the result on the
    shared ``RopeSetup``; each layer then does only the L1 reshard the base
    class already did. Values are unchanged.
    """

    def _decode_rope_mats(self, rot_idxs, batch):
        shared = getattr(self.rope, "decode_mats_shared", None)
        if shared is None:
            return super()._decode_rope_mats(rot_idxs, batch)
        cos_shared, sin_shared = shared
        rope = self.rope
        cos = ttnn.to_memory_config(cos_shared, rope.decode_cs_mem)
        sin = ttnn.to_memory_config(sin_shared, rope.decode_cs_mem)
        trans = ttnn.to_memory_config(rope.trans_mat_decode_dram, rope.trans_mem_decode)
        return cos, sin, trans


# --------------------------------------------------------------------- model


class GLM47FlashModel:
    """The whole GLM-4.7-Flash forward path on one Blackhole chip.

    Construct with :meth:`from_pretrained`. Owns weights, RoPE tables and the
    terminal norm/LM-head; the caller owns the KV cache, page table and the
    per-step position/token state (see ``tt/generator.py``).
    """

    # ------------------------------------------------------------------ build

    @classmethod
    def from_pretrained(
        cls,
        mesh_device,
        *,
        checkpoint_dir=None,
        hf_model_id: str = DEFAULT_HF_MODEL_ID,
        hf_config=None,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
        layer_indices=None,
        decoder_cls=SharedRopeDecoder,
        expert_dtype=ttnn.bfloat4_b,
        weight_dtype=ttnn.bfloat8_b,
        cache_dtype=ttnn.bfloat8_b,
        embed_dtype=ttnn.bfloat16,
        lm_head_dtype=ttnn.bfloat8_b,
        prefill_chunk_size: int = 2048,
        prefill_buckets=(128, 256, 512, 1024, 2048),
        lm_head_cores: int = 110,
        decode_logits_in_dram: bool = False,
        lm_head_in0_block_w: int = 4,
        share_rope: bool = True,
        progress=None,
    ):
        """Build the model from a local HF snapshot.

        ``layer_indices`` selects a subset of HF decoder layers to build. The
        default (``None``) builds the whole stack. ``[0, 1]`` builds the
        reduced full-model probe: one real layer of each kind, real tensor,
        cache and page-table shapes, and the real terminal path.
        """

        self = cls()
        snapshot = resolve_checkpoint_dir(checkpoint_dir, hf_model_id)
        self.snapshot_dir = snapshot
        self.hf_model_id = hf_model_id
        cfg = hf_config if hf_config is not None else load_hf_config(snapshot)
        self.hf_config = cfg
        self.mesh_device = mesh_device

        self.hidden = cfg.hidden_size
        self.vocab_size = cfg.vocab_size
        self.rms_eps = cfg.rms_norm_eps
        self.hf_max_position_embeddings = cfg.max_position_embeddings
        self.max_seq_len = int(max_seq_len or cfg.max_position_embeddings)
        self.max_batch_size = max_batch_size
        self.cache_dtype = cache_dtype
        self.prefill_chunk_size = prefill_chunk_size
        self.prefill_buckets = tuple(sorted(int(b) for b in prefill_buckets)) if prefill_buckets else ()
        self.pad_token_id = int(getattr(cfg, "pad_token_id", 0) or 0)
        self.eos_token_ids = _as_id_list(getattr(cfg, "eos_token_id", None))

        n_layers = cfg.num_hidden_layers
        self.hf_num_hidden_layers = n_layers
        self.layer_indices = list(range(n_layers)) if layer_indices is None else [int(i) for i in layer_indices]
        for idx in self.layer_indices:
            if not 0 <= idx < n_layers:
                raise ValueError(f"layer index {idx} outside the decoder stack [0, {n_layers})")
        self.is_reduced = len(self.layer_indices) != n_layers

        # One paged-cache geometry shared by every layer.
        blocks_per_user = -(-self.max_seq_len // 64)
        self.paged_config = PagedCacheConfig(block_size=64, max_num_blocks=blocks_per_user * max_batch_size)
        self.blocks_per_user = blocks_per_user

        log = progress if progress is not None else (lambda msg: None)
        ckpt = ShardedCheckpoint(snapshot)
        try:
            log(f"building {len(self.layer_indices)} decoder layers from {snapshot}")
            self.layers = []
            for n, idx in enumerate(self.layer_indices):
                sd = ckpt.layer_state_dict(idx)
                layer = decoder_cls.from_state_dict(
                    sd,
                    hf_config=cfg,
                    layer_idx=idx,
                    mesh_device=mesh_device,
                    max_batch_size=max_batch_size,
                    max_context=self.max_seq_len,
                    expert_dtype=expert_dtype,
                    weight_dtype=weight_dtype,
                    prefill_chunk_size=prefill_chunk_size,
                    paged_config=self.paged_config,
                )
                del sd
                if share_rope and self.layers:
                    _rebind_rope(layer, self.layers[0].rope)
                self.layers.append(layer)
                log(f"  layer {idx} ({layer.layer_kind}) built [{n + 1}/{len(self.layer_indices)}]")

            # ---- ROW_MAJOR decode RoPE tables (see SharedRopeDecoder) ----
            shared_rope = self.layers[0].rope
            self.rope_cos_rm = ttnn.to_layout(shared_rope.cos_matrix, ttnn.ROW_MAJOR_LAYOUT)
            self.rope_sin_rm = ttnn.to_layout(shared_rope.sin_matrix, ttnn.ROW_MAJOR_LAYOUT)
            self.shared_rope = shared_rope
            log("  decode rope tables (ROW_MAJOR) built")

            # ---- embeddings ----
            embed = ckpt.get("model.embed_tokens.weight")  # [vocab, hidden]
            self.embed_weight = ttnn.from_torch(
                embed.reshape(1, 1, self.vocab_size, self.hidden).contiguous(),
                device=mesh_device,
                dtype=embed_dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            del embed
            log("  embedding table uploaded")

            # ---- final norm ----
            norm = ckpt.get("model.norm.weight")
            self.norm_weight = ttnn.from_torch(
                norm.reshape(1, 1, norm.shape[0] // TILE, TILE),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            del norm

            # ---- lm head ----
            if getattr(cfg, "tie_word_embeddings", False):
                head = ckpt.get("model.embed_tokens.weight")
            else:
                head = ckpt.get("lm_head.weight")  # [vocab, hidden]
            self.lm_head_weight = ttnn.from_torch(
                head.T.contiguous(),  # [hidden, vocab]
                device=mesh_device,
                dtype=lm_head_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            del head
            log("  lm head uploaded")
        finally:
            ckpt.close()

        # ---- terminal-path configs ----
        dev = mesh_device
        self.ck_norm = _ck(dev, ttnn.MathFidelity.HiFi4, True)
        self.ck_lm_head = _ck(dev, ttnn.MathFidelity.HiFi2, False)
        # Sampler-ready logits are 9.9 MB at 32 rows. In L1, TTSampling's
        # ttnn.split logs "L1 budget exceeded (need ~9945088 B, have 1248256 B
        # for 4 chunks); DRAM downgrade" and migrates the tensor before slicing,
        # inside the captured sampling graph. Producing them in DRAM removes
        # that op-internal fallback but is measurably slower end to end
        # (2.937 vs 2.903 ms token-out on the reduced probe, +34 us; the LM head
        # pays 45 us more to write DRAM while the sampler only saves 11 us).
        # L1 is therefore the default and the fallback is disclosed rather than
        # paid for; doc/full_model/logits_memory_ab.json has both arms and shows
        # identical tokens. Stage 07 should re-run the A/B if it adds resident
        # L1 pressure around the terminal path.
        self.decode_logits_memory_config = ttnn.DRAM_MEMORY_CONFIG if decode_logits_in_dram else ttnn.L1_MEMORY_CONFIG
        self.lm_head_pc_decode = _lm_head_1d_pc(
            self.vocab_size // TILE, self.hidden // TILE, lm_head_cores, lm_head_in0_block_w
        )
        first = self.layers[0]
        # The decoder returns the decode residual width-sharded on the DRAM-bank
        # raster; the final norm and the LM head consume that same shard.
        self.res_mem = getattr(first, "res_mem", None)
        self.res_norm_pc = getattr(first, "res_norm_pc", None)
        return self

    # ------------------------------------------------------------------ cache / page table

    def allocate_kv_cache(self, dtype=None):
        """Per-layer paged latent caches: ``[max_num_blocks, 1, block_size, 576]``.

        Layer 0's cache comes from the decoder's own ``allocate_kv_cache`` (the
        canonical geometry); the other 46 are allocated on device and filled
        from it. ``ttnn.zeros(device=...)` builds the zeros on the host and
        uploads them, which measures 293 ms for one 118 MiB layer cache - 13.8 s
        over 47 layers - while a device-to-device copy of the same buffer is
        0.6 ms.
        """
        dtype = dtype or self.cache_dtype
        first = self.layers[0].allocate_kv_cache(dtype=dtype)
        caches = [first]
        for _ in self.layers[1:]:
            cache = ttnn.allocate_tensor_on_device(
                ttnn.Shape(tuple(first.shape)), dtype, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
            )
            ttnn.copy(first, cache)
            caches.append(cache)
        return caches

    def _cache_zeros(self, like):
        """Persistent all-zero buffer matching one layer cache, for ``reset``."""
        key = (tuple(like.shape), like.dtype)
        cached = getattr(self, "_cache_zeros_buf", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        if cached is not None:
            ttnn.deallocate(cached[1])
        buf = ttnn.zeros(
            tuple(like.shape),
            dtype=like.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._cache_zeros_buf = (key, buf)
        return buf

    def prepare_cache_reset(self, kv_cache):
        """Materialize the cache-reset zero buffer *before* any trace capture.

        ``reset_kv_cache`` allocates this buffer lazily on first use, which
        lands it after the decode traces are captured. Metal then treats it as
        an unsafe allocation for the whole life of those traces
        (``mesh_device.cpp`` registers a trace as active at ``end_mesh_trace``
        and keeps allocations unsafe until it is released), because a replay
        rewrites the addresses the trace's own intermediates used. A corrupted
        zero source would make ``reset()`` fill the caches with garbage instead
        of zeros. Allocating it up front removes the hazard rather than
        arguing about it: verified with ``TT_METAL_TRACE_ALLOC_TRACKING=1``
        (``probe/trace_alloc_probe.py``).
        """
        caches = list(_iter_layer_caches(kv_cache))
        if caches:
            self._cache_zeros(caches[0])

    def default_page_table(self, batch=None):
        """Identity page table ``[batch, blocks_per_user]`` (torch int32)."""
        import torch

        batch = batch or self.max_batch_size
        total = self.paged_config.max_num_blocks
        per_user = total // batch
        return torch.arange(batch * per_user, dtype=torch.int32).reshape(batch, per_user)

    def page_table_to_device(self, page_table_torch):
        return ttnn.from_torch(
            page_table_torch,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def reset_kv_cache(self, kv_cache):
        """Zero every layer cache in place (buffer addresses preserved so a
        captured trace stays valid). One shared device-resident zero buffer,
        47 device-to-device copies: ~30 ms, against 13.8 s if each layer built
        its zeros on the host."""
        caches = list(_iter_layer_caches(kv_cache))
        if not caches:
            return
        zeros = self._cache_zeros(caches[0])
        for cache in caches:
            ttnn.copy(zeros, cache)

    # ------------------------------------------------------------------ embedding

    def embed(self, token_ids):
        """``token_ids`` device uint32 tensor -> ``[1, 1, T, hidden]`` bf16 TILE."""
        out = ttnn.embedding(token_ids, self.embed_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return ttnn.unsqueeze_to_4D(out)

    def tokens_to_device(self, token_ids, *, pad_to=None, device=True):
        """Host int list/tensor -> uint32 ttnn tensor shaped ``[1, 1, 1, T]``."""
        import torch

        if isinstance(token_ids, ttnn.Tensor):
            return token_ids
        ids = torch.as_tensor(token_ids, dtype=torch.int32).reshape(-1)
        if pad_to is not None and ids.numel() < pad_to:
            ids = torch.nn.functional.pad(ids, (0, pad_to - ids.numel()))
        ids = ids.reshape(1, 1, 1, -1)
        return ttnn.from_torch(
            ids,
            device=self.mesh_device if device else None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if device else None,
        )

    # ------------------------------------------------------------------ terminal path

    def _final_norm_prefill(self, x):
        return ttnn.rms_norm(x, epsilon=self.rms_eps, weight=self.norm_weight, compute_kernel_config=self.ck_norm)

    def _final_norm_decode(self, x):
        if self.res_mem is None:
            return ttnn.rms_norm(x, epsilon=self.rms_eps, weight=self.norm_weight, compute_kernel_config=self.ck_norm)
        if x.memory_config() != self.res_mem:
            x = ttnn.to_memory_config(x, self.res_mem)
        return ttnn.rms_norm(
            x,
            epsilon=self.rms_eps,
            weight=self.norm_weight,
            program_config=self.res_norm_pc,
            compute_kernel_config=self.ck_norm,
            memory_config=self.res_mem,
        )

    def lm_head_decode(self, normed):
        """Sampler-ready logits ``[1, 1, 32, vocab]`` from the normed residual.

        Consumes ``normed``. The mcast_in0 config wants an interleaved in0, so
        the width-sharded norm output is gathered back to L1 interleaved first
        (a ~1 us 4 KiB reshard at batch 1), then zero-extended to the 32 rows
        ``ttnn.sampling`` requires before the single wide-1D mcast matmul.

        The output lands in ``decode_logits_memory_config`` (L1 by default; see
        the constructor for the measured DRAM A/B and the ``ttnn.split``
        downgrade it trades against).
        """
        if normed.memory_config().is_sharded():
            x = ttnn.sharded_to_interleaved(normed, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(normed)
        else:
            x = normed
        x = self._pad_to_sampler_rows(x)
        logits = ttnn.linear(
            x,
            self.lm_head_weight,
            program_config=self.lm_head_pc_decode,
            memory_config=self.decode_logits_memory_config,
            compute_kernel_config=self.ck_lm_head,
        )
        ttnn.deallocate(x)
        return logits

    def lm_head_prefill(self, normed):
        """Logits for a ``[1, 1, M, hidden]`` prefill slab.

        ``M`` >= 10 tiles uses the 2D mcast config over the 11x10 grid; a
        single-tile slab uses the decode config. The default (no program
        config) matmul is 2.9x slower at this N
        (``doc/full_model/head_probe.json``), so it is never used."""
        m = normed.shape[2]
        pc = _mcast_2d_pc(m, self.hidden, self.vocab_size) if m > TILE else None
        if pc is not None:
            return ttnn.linear(
                normed,
                self.lm_head_weight,
                program_config=pc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.ck_lm_head,
            )
        if m > TILE:
            raise ValueError(f"prefill LM-head slab of {m} rows has no valid program config; slice to <= 32 rows")
        # Same memory config as the decode path: the prefill last-position
        # logits also go straight into the sampler, whose ttnn.split would
        # otherwise migrate the 9.9 MB tensor out of L1 first.
        return ttnn.linear(
            normed,
            self.lm_head_weight,
            program_config=self.lm_head_pc_decode,
            memory_config=self.decode_logits_memory_config,
            compute_kernel_config=self.ck_lm_head,
        )

    # ------------------------------------------------------------------ prefill

    def prefill_physical_len(self, seq: int) -> int:
        """Physical prefill length for a logical prompt of ``seq`` tokens.

        Every distinct physical length is its own set of compiled TTNN
        programs, and compiling one 47-layer prefill shape costs ~13 s on this
        chip. Padding to a small bucket set keeps that to a handful of shapes
        instead of one per prompt length, at the cost of prefilling a few
        hundred extra positions. Padded positions are never attended: prefill
        attention is causal so no valid query sees them, and every decode step
        writes its own cache row before reading it, so the pad rows are
        overwritten as decode advances.

        A prompt longer than one chunk is split into whole ``prefill_chunk_size``
        chunks plus a bucketed tail, so the distinct chunk shapes stay
        ``{chunk} | buckets`` however long the prompt is.
        """
        block = self.paged_config.block_size
        aligned = -(-seq // block) * block
        if not self.prefill_buckets:
            return aligned
        chunk = self.prefill_chunk_size
        full_chunks, remainder = divmod(seq, chunk)
        phys = full_chunks * chunk
        if remainder:
            tail = next((b for b in self.prefill_buckets if remainder <= b), chunk)
            phys += tail
        return min(max(phys, block), self.max_seq_len_physical)

    @property
    def max_seq_len_physical(self) -> int:
        """Largest prefill length the allocated cache can hold."""
        return self.blocks_per_user * self.paged_config.block_size

    def run_layer_stack_prefill(self, input_ids, *, kv_cache, page_table, user_id=0, seq_len=None, progress_cb=None):
        """Embed + 47 decoder layers for one user. Returns hidden ``[1, 1, S, hidden]``.

        ``input_ids``: torch/list token ids (``[S]`` or ``[1, S]``) or a device
        uint32 tensor. ``seq_len`` is the *logical* prompt length; padding to
        the paged block size, per-chunk cache fill, position handling and
        output slicing are internal to the decoder layers.
        """
        import torch

        if isinstance(input_ids, ttnn.Tensor):
            ids_dev, owned = input_ids, False
            seq = seq_len if seq_len is not None else int(ids_dev.shape[-1])
        else:
            ids = torch.as_tensor(input_ids, dtype=torch.int32).reshape(-1)
            seq = seq_len if seq_len is not None else int(ids.numel())
            if seq > ids.numel():
                raise ValueError(f"seq_len {seq} exceeds the {ids.numel()} supplied token ids")
            ids_dev, owned = self.tokens_to_device(ids[:seq]), True
        if not 1 <= seq <= self.max_seq_len:
            raise ValueError(f"prompt length {seq} outside [1, {self.max_seq_len}]")

        phys = self.prefill_physical_len(seq)
        if not isinstance(input_ids, ttnn.Tensor) and phys > seq:
            padded = torch.full((phys,), self.pad_token_id, dtype=torch.int32)
            padded[:seq] = ids[:seq]
            ttnn.deallocate(ids_dev)
            ids_dev, owned = self.tokens_to_device(padded), True
        elif isinstance(input_ids, ttnn.Tensor):
            phys = int(ids_dev.shape[-1])

        x = self.embed(ids_dev)  # [1, 1, phys, hidden]
        if owned:
            ttnn.deallocate(ids_dev)

        caches = list(_iter_layer_caches(kv_cache))
        if len(caches) != len(self.layers):
            raise ValueError(f"kv_cache has {len(caches)} entries but the model has {len(self.layers)} layers")
        for i, layer in enumerate(self.layers):
            if progress_cb is not None:
                progress_cb(i, len(self.layers))
            nxt = layer.prefill_forward(x, kv_cache=caches[i], page_table=page_table, user_id=user_id, seq_len=phys)
            ttnn.deallocate(x)
            x = nxt
        return x, seq

    def prefill_forward(
        self,
        input_ids,
        *,
        kv_cache,
        page_table,
        user_id: int = 0,
        seq_len: int | None = None,
        return_all_logits: bool = False,
        logits_chunk: int = 320,
        progress_cb=None,
    ):
        """Prefill one user and return host logits.

        Returns a torch float32 tensor ``[1, R, vocab]``: ``R = seq_len`` when
        ``return_all_logits`` else 1 (the final prompt position).
        """
        hidden, seq = self.run_layer_stack_prefill(
            input_ids,
            kv_cache=kv_cache,
            page_table=page_table,
            user_id=user_id,
            seq_len=seq_len,
            progress_cb=progress_cb,
        )
        try:
            if return_all_logits:
                return self._logits_host_rows(hidden, seq, 0, seq, logits_chunk)
            row = seq - 1
            return self._logits_host_rows(hidden, seq, row, row + 1, logits_chunk)
        finally:
            ttnn.deallocate(hidden)

    def prefill_forward_last_logits_device(
        self, input_ids, *, kv_cache, page_table, user_id: int = 0, seq_len: int | None = None, progress_cb=None
    ):
        """Prefill one user and keep the final-position logits on device.

        Returns ``(logits, row)`` where ``logits`` is ``[1, 1, 32, vocab]`` (the
        full 32-row tile of prompt positions containing the last one) and
        ``row`` is the index of the final prompt position in it. Lets the same
        on-device sampler pick the first generated token, so prefill and decode
        share exactly one sampling implementation.

        The slice is a *whole* tile deliberately. Slicing ``[s0, seq)`` instead
        made the program depend on the prompt length twice over, through the
        slice size and through the pad that followed it, so every new logical
        length compiled two programs; with a fixed 32 rows there is one program
        per tile offset, and the pad disappears. Row ``row`` is bit-identical
        either way: the final norm is per-row and the LM head is a per-row
        matmul, so the extra rows in the tile cannot reach it. Only ``row`` is
        ever read. See work log FM-016.
        """
        hidden, seq = self.run_layer_stack_prefill(
            input_ids,
            kv_cache=kv_cache,
            page_table=page_table,
            user_id=user_id,
            seq_len=seq_len,
            progress_cb=progress_cb,
        )
        try:
            s0 = ((seq - 1) // TILE) * TILE
            slab = self._slice_rows(hidden, s0, s0 + TILE)
            normed = self._final_norm_prefill(slab)
            if slab is not hidden:
                ttnn.deallocate(slab)
            normed = self._pad_to_sampler_rows(normed)  # no-op: the tile is already 32 rows
            logits = self.lm_head_prefill(normed)
            ttnn.deallocate(normed)
            return logits, (seq - 1) - s0
        finally:
            ttnn.deallocate(hidden)

    def warmup_terminal_shapes(self, physical_lengths=None, logits_chunk: int = 320):
        """Compile the terminal slice/norm/LM-head programs on dummy activations.

        `ttnn.slice` keys its program on the start offset, so the terminal
        path has one program per 32-row tile offset into the prefill
        activation. For a prompt inside one chunk that is
        ``4 + 8 + 16 + 32 + 64 = 124`` offsets across the five buckets, and
        compiling them here, *before* the decode traces are captured, is what
        stops a first-use prompt length from compiling under a live trace and
        forcing a trace recapture (work log FM-016).

        Runs on a zero activation rather than a real prefill, so it costs
        program compilation and about a millisecond of device time per offset
        instead of one prefill per offset. Prompts longer than one chunk still
        have a tile offset past the last bucket and still compile on first
        use; bounding that needs the terminal path to slice the last chunk
        first, which is stage-07 work.

        ``logits_chunk`` must match the default ``prefill_forward`` passes to
        ``_logits_host_rows``, or the all-positions walk warmed here is not the
        one that runs.
        """
        lengths = sorted({int(n) for n in (physical_lengths or self.prefill_buckets)})
        for rows in lengths:
            rows = (rows // TILE) * TILE
            if rows < TILE:
                continue
            dummy = ttnn.zeros(
                (1, 1, rows, self.hidden),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            # Two families, both keyed on the physical length only:
            #  * every whole-tile slab, used by prefill_forward_last_logits_device
            #    and by the at-most-one-tile host path (the low-level
            #    prefill_forward);
            #  * the chunk walk prefill_logits uses over the whole prompt.
            slabs = {(s0, s0 + TILE) for s0 in range(0, rows, TILE)}
            slabs |= set(self._host_logits_walk(rows, 0, rows, logits_chunk))
            for s0, s1 in sorted(slabs):
                slab = self._slice_rows(dummy, s0, s1)
                normed = self._final_norm_prefill(slab)
                if slab is not dummy:
                    ttnn.deallocate(slab)
                logits = self.lm_head_prefill(normed)
                ttnn.deallocate(normed)
                ttnn.deallocate(logits)
            ttnn.deallocate(dummy)

    def _slice_rows(self, hidden, s0, s1):
        if s0 == 0 and s1 == hidden.shape[2]:
            return hidden
        return ttnn.slice(hidden, [0, 0, s0, 0], [1, 1, s1, self.hidden])

    def _pad_to_sampler_rows(self, x):
        """Zero-extend the logical row count to the 32 rows ``ttnn.sampling``
        wants, with ``ttnn.pad``. Within one tile only the pad rows are
        written; the padded rows are never read back, only the logical row is
        (``prefill_forward_last_logits_device`` returns its index)."""
        rows = x.shape[2]
        if rows == SAMPLER_ROWS:
            return x
        if rows > SAMPLER_ROWS:
            raise ValueError(f"{rows} rows cannot be padded down to {SAMPLER_ROWS}")
        return ttnn.pad(x, [(0, 0), (0, 0), (0, SAMPLER_ROWS - rows), (0, 0)], 0.0)

    def _host_logits_walk(self, phys, first_row, last_row, chunk):
        """The slab boundaries ``_logits_host_rows`` will use, as ``(s0, s1)``.

        Every boundary is a multiple of 32 and the walk ends on a multiple of
        32, so no slab size depends on the *logical* prompt length: the
        program family is a function of the bucketed physical length only.
        Getting that wrong is what made every new prompt length compile a fresh
        slice/pad/row-slice triple while the decode traces were live, and pay a
        trace recapture for it (work log FM-018).

        A request for at most one tile of rows (the low-level
        ``prefill_forward``, which returns the final position) walks exactly
        the tile(s) holding them. A request for the whole prompt
        (``prefill_logits``) walks to the physical length, which costs at most
        one bucket of padded rows and makes the walk identical for every
        logical length in that bucket.
        """
        chunk = max(TILE, (chunk // TILE) * TILE)
        tile_ceil = -(-last_row // TILE) * TILE
        end = phys if (last_row - first_row) > TILE else min(tile_ceil, phys)
        out = []
        s0 = (first_row // TILE) * TILE
        while s0 < end:
            s1 = min(s0 + chunk, end)
            if s1 - s0 > TILE and _mcast_2d_pc(s1 - s0, self.hidden, self.vocab_size) is None:
                s1 = min(s0 + TILE, end)  # small tail: one tile at a time on the decode config
            out.append((s0, s1))
            s0 = s1
        return out

    def _logits_host_rows(self, hidden, seq, first_row, last_row, chunk):
        """Norm + LM head over ``[first_row, last_row)``, returned on host.

        Walks the tile-aligned slabs of :meth:`_host_logits_walk`, and never
        materialises the whole ``[1, 1, S, 154880]`` logits slab.
        """
        import torch

        phys = int(hidden.shape[2])
        pieces = []
        for s0, s1 in self._host_logits_walk(phys, first_row, last_row, chunk):
            lo = max(first_row, s0) - s0
            hi = min(last_row, s1) - s0
            if hi <= lo:
                continue
            slab = self._slice_rows(hidden, s0, s1)
            normed = self._final_norm_prefill(slab)
            if slab is not hidden:
                ttnn.deallocate(slab)
            logits = self.lm_head_prefill(normed)
            ttnn.deallocate(normed)
            # Trimmed on the host, always. A device-side row slice would be
            # keyed on the logical length, and pinning its *size* to 1 to fix
            # that traded one problem for a worse one: the program is
            # single-core, so warming it while L1 was empty placed its static
            # circular buffers where the captured decode trace's L1 logits
            # later sit, and the first real call died with "Statically
            # allocated circular buffers in program N clash with L1 buffers".
            # The cost of not slicing is one 32-row tile (9.9 MB, ~3.4 ms)
            # instead of one row (310 KB) on the single-position path, which is
            # 0.6% of a 600 ms prefill (work log FM-018).
            host = ttnn.to_torch(logits).to(torch.float32)[0, 0]
            ttnn.deallocate(logits)
            pieces.append(host[lo:hi, : self.vocab_size])
        return torch.cat(pieces, dim=0).unsqueeze(0)

    # ------------------------------------------------------------------ decode

    def decode_rope_indices(self, cur_pos, batch):
        """RoPE index tensor ``[1, B]`` uint32 derived from ``cur_pos`` on device.

        Deriving it instead of carrying a second persistent tensor makes
        position coherence structural (there is only one position state) and
        keeps an inactive slot pinned: ``cur_pos`` marks inactive rows with
        ``-1``, ``ttnn.plus_one(..., skip_negative_entries=True)`` leaves them
        there, and the clamp maps them to RoPE index 0 every step instead of
        letting a separately-incremented index walk off the end of the
        ``max_seq_len``-tall cos/sin table.
        """
        rot = ttnn.clamp(cur_pos, min=0)
        rot_u32 = ttnn.typecast(rot, ttnn.uint32)
        ttnn.deallocate(rot)
        return ttnn.reshape(rot_u32, (1, batch))

    def ttnn_decode_forward(
        self,
        tokens,
        cur_pos,
        page_table,
        kv_cache,
        *,
        rot_idxs=None,
        advance_positions: bool = True,
    ):
        """One device-only decode step over persistent device tensors.

        ``tokens``: uint32 ``[1, 1, 1, 32]`` (the sampler's ``tt_out_tok``
        target; the model reads the first ``max_batch_size`` slots).
        ``cur_pos``: int32 ``[B]``, with ``-1`` marking an inactive slot.
        ``rot_idxs`` is optional: by default the RoPE index is derived from
        ``cur_pos`` on device (see :meth:`decode_rope_indices`), so there is
        exactly one persistent position tensor. A caller that needs decoupled
        RoPE indices (position remapping, sliding windows) may pass its own
        uint32 ``[1, B]`` tensor, which is then advanced with ``cur_pos``.
        Returns sampler-ready logits ``[1, 1, 32, vocab]``.

        When ``advance_positions`` the current position is incremented on
        device at the end of the graph, so a captured trace walks the decode
        positions itself with no host refresh.
        """
        batch = self.max_batch_size
        if tokens.shape[-1] == batch:
            tok, owned = tokens, False
        else:
            tok, owned = ttnn.slice(tokens, [0, 0, 0, 0], [1, 1, 1, batch]), True
        x = self.embed(tok)  # [1, 1, B, hidden]
        if owned:
            ttnn.deallocate(tok)
        owns_rot = rot_idxs is None
        if owns_rot:
            rot_idxs = self.decode_rope_indices(cur_pos, batch)
        shared_mats = self._decode_rope_lookup(rot_idxs, batch)
        caches = list(_iter_layer_caches(kv_cache))
        try:
            for i, layer in enumerate(self.layers):
                nxt = layer.decode_forward(
                    x, kv_cache=caches[i], page_table=page_table, cur_pos_tensor=cur_pos, rot_idxs=rot_idxs
                )
                ttnn.deallocate(x)
                x = nxt
        finally:
            self._release_decode_rope_lookup(shared_mats)
        if owns_rot:
            ttnn.deallocate(rot_idxs)
        normed = self._final_norm_decode(x)
        ttnn.deallocate(x)
        logits = self.lm_head_decode(normed)  # consumes normed
        if advance_positions:
            ttnn.plus_one(cur_pos, skip_negative_entries=True)
            if not owns_rot:
                ttnn.plus_one(rot_idxs)
        return logits

    def _decode_rope_lookup(self, rot_idxs, batch):
        """One decode cos/sin lookup for the whole stack (see SharedRopeDecoder).

        Reproduces exactly what ``RopeSetup.decode_mats`` computes before its
        L1 reshard, from the ROW_MAJOR tables.
        """
        rope = getattr(self, "shared_rope", None)
        if rope is None or not isinstance(self.layers[0], SharedRopeDecoder):
            return None
        cos = ttnn.unsqueeze_to_4D(ttnn.embedding(rot_idxs, self.rope_cos_rm, layout=ttnn.TILE_LAYOUT))
        sin = ttnn.unsqueeze_to_4D(ttnn.embedding(rot_idxs, self.rope_sin_rm, layout=ttnn.TILE_LAYOUT))
        if batch > 1:
            cos = ttnn.transpose(cos, 1, 2)  # [1, B, 1(32), dim]
            sin = ttnn.transpose(sin, 1, 2)
        for layer in self.layers:
            layer.rope.decode_mats_shared = (cos, sin)
        return cos, sin

    def _release_decode_rope_lookup(self, shared_mats):
        for layer in self.layers:
            layer.rope.decode_mats_shared = None
        if shared_mats is not None:
            for tensor in shared_mats:
                ttnn.deallocate(tensor)

    # ------------------------------------------------------------------ introspection

    def weight_bytes(self):
        """Device bytes held by weights, grouped, for the context contract."""
        groups = {"layers": 0, "embedding": 0, "lm_head": 0, "norm": 0, "rope": 0}
        seen = set()

        def add(group, tensor):
            if tensor is None or not isinstance(tensor, ttnn.Tensor):
                return
            key = id(tensor)
            if key in seen:
                return
            seen.add(key)
            groups[group] += _tensor_bytes(tensor)

        for layer in self.layers:
            for name, value in vars(layer).items():
                if isinstance(value, ttnn.Tensor):
                    add("layers", value)
            for name in ("cos_matrix", "sin_matrix", "trans_mat_prefill", "trans_mat_decode_dram"):
                add("rope", getattr(layer.rope, name, None))
        add("rope", getattr(self, "rope_cos_rm", None))
        add("rope", getattr(self, "rope_sin_rm", None))
        buf = getattr(self, "_cache_zeros_buf", None)
        if buf is not None:
            groups["cache_reset_scratch"] = _tensor_bytes(buf[1])
        add("embedding", self.embed_weight)
        add("lm_head", self.lm_head_weight)
        add("norm", self.norm_weight)
        groups["total"] = sum(v for k, v in groups.items() if k != "total")
        return groups

    def kv_cache_bytes(self, seq_len=None, batch=None, dtype=None):
        seq_len = seq_len or self.max_seq_len
        batch = batch or self.max_batch_size
        dtype = dtype or self.cache_dtype
        per_elem = _dtype_bytes_per_element(dtype)
        kvpe = self.layers[0].kvpe_dim
        blocks = -(-seq_len // self.paged_config.block_size) * batch
        return int(blocks * self.paged_config.block_size * kvpe * per_elem * len(self.layers))

    def deallocate(self):
        for layer in self.layers:
            for value in list(vars(layer).values()):
                if isinstance(value, ttnn.Tensor):
                    try:
                        ttnn.deallocate(value)
                    except Exception:
                        pass
        for tensor in (self.embed_weight, self.lm_head_weight, self.norm_weight):
            try:
                ttnn.deallocate(tensor)
            except Exception:
                pass


# --------------------------------------------------------------------- helpers


def _as_id_list(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(v) for v in value]


def _iter_layer_caches(kv_cache):
    """Accept ``[tensor, ...]`` (MLA: one latent cache per layer) or
    ``[[k, v], ...]`` (the generic readiness/vLLM shape) and yield the
    per-layer cache tensor."""
    for entry in kv_cache:
        if isinstance(entry, (list, tuple)):
            if len(entry) != 1:
                raise ValueError(
                    "GLM-4.7-Flash uses a single compressed-latent cache per layer; " f"got a {len(entry)}-tensor entry"
                )
            yield entry[0]
        else:
            yield entry


def _rebind_rope(layer, shared_rope):
    """Release this layer's private RoPE tables and reuse the shared ones.

    Every layer builds byte-identical cos/sin tables from the same
    ``rope_theta``/``qk_rope_head_dim``/``max_context``; at the advertised
    202752 context that is 51.9 MiB per layer, 2.4 GiB over 47 layers.
    """
    own = layer.rope
    if own is shared_rope:
        return
    for name in ("cos_matrix", "sin_matrix", "trans_mat_prefill", "trans_mat_decode_dram"):
        tensor = getattr(own, name, None)
        if isinstance(tensor, ttnn.Tensor):
            ttnn.deallocate(tensor)
    layer.rope = shared_rope


_BYTES_PER_ELEMENT = {
    ttnn.bfloat16: 2.0,
    ttnn.float32: 4.0,
    ttnn.uint32: 4.0,
    ttnn.int32: 4.0,
    ttnn.uint16: 2.0,
    ttnn.uint8: 1.0,
    ttnn.bfloat8_b: 1.0625,  # 1 byte mantissa + 4-byte shared exponent per 16 values
    ttnn.bfloat4_b: 0.5625,
}


def _dtype_bytes_per_element(dtype):
    try:
        return _BYTES_PER_ELEMENT[dtype]
    except KeyError as exc:
        raise KeyError(f"unknown element size for {dtype}") from exc


def _tensor_bytes(tensor):
    shape = list(tensor.padded_shape) if hasattr(tensor, "padded_shape") else list(tensor.shape)
    n = 1
    for dim in shape:
        n *= int(dim)
    return int(n * _dtype_bytes_per_element(tensor.dtype))
