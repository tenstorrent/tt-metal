# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS single-rank chunked-prefill runtime.

Mirrors ``minimax_m3/tt/tt_prefill_runtime.py`` and satisfies the common/prefill runtime contract
(``models/demos/common/prefill/runners/ADDING_A_PREFILL_MODEL.md`` §2): build model -> allocate KV
cache -> compile -> ``prefill_chunk`` (per chunk). The runtime is mode-agnostic; the caller (the
common engine, or ``tests/galaxy_prefill_kv_pcc.py``) drives one-shot vs chunked and does the golden
KV-cache PCC.

Cache ownership: the runtime can OWN its KV cache (``owns_kv_cache=True``, used by the standalone
galaxy harness — ``self.kv_cache``) OR run engine-owned (``owns_kv_cache=False``, the adapter path:
the engine allocates via ``GptOssPrefillAdapter.allocate_kv_cache`` and passes the ``KvCaches`` tuple
into every call). ``prefill_chunk`` / ``compile`` / ``gather_layer`` / ``kv_cache_pcc_check`` accept
an optional cache arg that defaults to ``self.kv_cache``.

CHUNKED prefill is supported: the SP cache-READ attention path (``cached_len > 0``, chunks 1+) uses the
ring-joint dense SDPA over the block-cyclic packed KV cache (``attention/dense_sp.py``); chunk 0 /
one-shot (``cached_len == 0``) uses the gather-Q stand-in. The single-chip (sp==1) cache-read is still
``NotImplementedError`` (not used on the galaxy). The galaxy KV-PCC harness runs both one-shot and
multi-chunk.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions

from .attention import GptOssKVCache, allocate_kv_cache
from .rope import build_indexed_rope


@dataclass
class TtPrefillRuntimeConfig:
    num_layers: int  # layers built/cached by this runtime (== model total for single-rank)
    max_seq_len: int  # per-user KV-cache length in tokens; must be a multiple of chunk_size
    mesh_shape: tuple = (4, 8)  # (SP rows, TP cols) on the Blackhole galaxy
    chunk_size: int = 5120  # tokens per prefill_chunk() call; one-shot sets this == max_seq_len
    num_users: int = 1  # independent cache slots (user-major batch)
    sp_axis: int = 0
    tp_axis: int = 1
    topology: ttnn.Topology = ttnn.Topology.Linear
    use_ep_moe: bool = True
    expert_weight_dtype: ttnn.DataType = ttnn.bfloat4_b
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[Path] = None
    # When True the runtime allocates + owns its KV cache (self.kv_cache) — the standalone galaxy
    # harness path. The adapter/engine path sets this False and passes the engine-owned KvCaches in.
    owns_kv_cache: bool = True
    # Pipeline-parallel rank flags the common prefill runner reads off runtime.config
    # (single-rank standalone/harness => both True).
    is_first_rank: bool = True
    is_last_rank: bool = True

    @property
    def sp_factor(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    """Single-rank GPT-OSS prefill lifecycle: build model -> (optionally allocate KV cache) ->
    build indexed rope -> compile -> prefill_chunk."""

    def __init__(self, mesh_device, hf_config, state_dict: dict, config: TtPrefillRuntimeConfig):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config

        assert (
            config.max_seq_len % config.chunk_size == 0
        ), f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size})"

        self.model_built = False
        self.kv_cache_allocated = False
        self.compiled = False
        self.kv_cache = None

        self._build_model(state_dict)
        if config.owns_kv_cache:
            self._allocate_kv_cache()
        self._build_indexed_rope()

    def _build_model(self, state_dict: dict) -> None:
        from models.demos.gpt_oss_d_p.tt.config import MeshConfig
        from models.demos.gpt_oss_d_p.utils.general_utils import get_default_num_links

        from .ccl import CCLManager
        from .model import Model

        rows, cols = self.config.mesh_shape
        logger.info(
            f"Building GPT-OSS TtPrefillRuntime model: num_layers={self.config.num_layers} "
            f"max_seq_len={self.config.max_seq_len} chunk_size={self.config.chunk_size} "
            f"num_users={self.config.num_users} mesh_shape={self.config.mesh_shape}"
        )
        mesh_config = MeshConfig((rows, cols), tp=cols)
        ccl = CCLManager(
            self.mesh_device, num_links=get_default_num_links(self.mesh_device), topology=self.config.topology
        )
        self.model = Model(
            mesh_device=self.mesh_device,
            hf_config=self.hf_config,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            tensor_cache_path=self.config.weight_cache_path,
            max_local_batch_size=1,
            max_seq_len=self.config.max_seq_len,
            sequence_parallel=True,
            use_ep_moe=self.config.use_ep_moe,
            ep_seq_len_per_chip=self.config.chunk_size // self.config.sp_factor,
            expert_weight_dtype=self.config.expert_weight_dtype,
        )
        self.model_built = True

    def _allocate_kv_cache(self) -> None:
        # ONE cache holding num_users * num_layers slots (user-major); each (user, layer) slot is
        # filled per chunk. K/V heads shard on the TP cols; the sequence is SP-sharded block-cyclic.
        self.kv_cache = allocate_kv_cache(
            self.mesh_device,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            sp_axis=self.config.sp_axis,
            num_users=self.config.num_users,
            head_dim=self.hf_config.head_dim,
            cache_dtype=self.config.cache_dtype,
        )
        self.kv_cache_allocated = True

    def _build_indexed_rope(self) -> None:
        """Whole-cache, block-cyclic, SP-sharded YaRN cos/sin for the on-device indexed rope, built
        ONCE and reused for every chunk (see tt/rope.build_indexed_rope). No per-chunk host reshard."""
        rs = getattr(self.hf_config, "rope_scaling", None) or {}
        self.rope_indexed = build_indexed_rope(
            self.mesh_device,
            head_dim=self.hf_config.head_dim,
            max_seq_len=self.config.max_seq_len,
            chunk_size=self.config.chunk_size,
            sp_axis=self.config.sp_axis,
            rope_theta=getattr(self.hf_config, "rope_theta", 150000.0),
            yarn_factor=rs.get("factor", 32.0),
            yarn_orig_max_pos=rs.get("original_max_position_embeddings", 4096),
            yarn_beta_fast=rs.get("beta_fast", 32.0),
            yarn_beta_slow=rs.get("beta_slow", 1.0),
        )

    def _resolve_kv(self, kv_caches) -> GptOssKVCache:
        """Resolve the GptOssKVCache from the (optional) caller arg. Accepts None (use self-owned),
        a GptOssKVCache, or the engine's KvCaches tuple (index 0)."""
        if kv_caches is None:
            assert self.kv_cache is not None, "runtime has no KV cache (owns_kv_cache=False): pass kv_caches"
            return self.kv_cache
        if isinstance(kv_caches, GptOssKVCache):
            return kv_caches
        return kv_caches[0]

    def make_chunk_input(self, token_ids: list) -> ttnn.Tensor:
        """Embed + SP-shard one chunk's token ids -> the model input tensor (consumed by prefill_chunk)."""
        assert len(token_ids) == self.config.chunk_size, (
            f"chunk input must be exactly chunk_size={self.config.chunk_size} tokens (pad the tail), "
            f"got {len(token_ids)}"
        )
        chunk_tok = torch.tensor(token_ids, dtype=torch.int32).reshape(1, len(token_ids))
        x_embd, _, _ = self.model.prepare_inputs_prefill(chunk_tok)
        return x_embd

    def compile(self, kv_caches=None) -> None:
        """Warm up the kernels by running zero-token chunks through prefill_chunk (JIT-compiles).

        Two warmups: the first chunk (actual_start=0, the gather-Q path) AND — when the config is
        multi-chunk (max_seq_len > chunk_size) — a second chunk (actual_start>0), which is the ONLY
        path that fires the SP ring cache-read (attention/dense_sp.py). Without the second warmup the
        ring kernels JIT-compile inside the first *served/timed* chunk, inflating first-request TTFT.
        (This is separate from the one-time empty-disk kernel-cache compile that only the very first run
        ever pays.) One-shot (max_seq_len == chunk_size) never reaches the ring path and uses FABRIC_1D,
        so it skips the second warmup."""
        assert self.model_built
        chunk = self.config.chunk_size
        ring = self.config.max_seq_len > chunk
        logger.info(
            f"GPT-OSS TtPrefillRuntime.compile() — warming up {'2 chunks (gather-Q + ring cache-read)' if ring else 'one chunk'} "
            f"of {chunk} tokens"
        )
        # prefill_chunk consumes (deallocates) its input tensor, so build a fresh input per call.
        self.prefill_chunk(self.make_chunk_input([0] * chunk), kv_caches, slot_id=0, actual_start=0, actual_end=chunk)
        if ring:
            # actual_start>0 drives the ring cache-read; it reads the prefix we just wrote at [0, chunk).
            self.prefill_chunk(
                self.make_chunk_input([0] * chunk), kv_caches, slot_id=0, actual_start=chunk, actual_end=2 * chunk
            )
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_caches=None,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        skip_lm_head: bool = True,
        get_last_token: int = -1,
        request_id: int = -1,  # accepted for the common-runner contract; single-request prefill ignores it
    ) -> Optional[ttnn.Tensor]:
        """Prefill ONE chunk into user ``slot_id``'s slice of the KV cache (self-owned or the engine's
        ``kv_caches``). Returns None (skip_lm_head) — the populated cache is the output.

        [actual_start, actual_end): absolute KV-position range of this chunk's real tokens. actual_start
        is the cache write offset (valid prefix already cached); the last chunk's tail may be pad, so
        actual_end < actual_start + chunk_size. Call once per chunk, in order.

        actual_start > 0 drives the SP ring cache-READ path (chunks 1+, attention/dense_sp.py);
        actual_start == 0 (first/only chunk) uses the gather-Q stand-in.
        """
        assert self.model_built, "build the model before prefill_chunk()"
        kv = self._resolve_kv(kv_caches)
        assert 0 <= slot_id < self.config.num_users, f"slot_id {slot_id} out of range [0, {self.config.num_users})"
        assert (
            actual_start + self.config.chunk_size <= self.config.max_seq_len
        ), f"chunk at actual_start={actual_start} exceeds per-user cache {self.config.max_seq_len}"
        assert (
            actual_start < actual_end <= actual_start + self.config.chunk_size
        ), f"[actual_start={actual_start}, actual_end={actual_end}) not within one chunk of {self.config.chunk_size}"

        out = self.model.prefill_forward(
            input_tensor,
            rot_mats_global=self.rope_indexed,  # whole-cache indexed rope (persistent; not deallocated)
            kv_cache=kv,
            cached_len=actual_start,
            user_id=slot_id,
            get_last_token=get_last_token,
            skip_lm_head=skip_lm_head,
            indexed_rope=True,
        )
        ttnn.deallocate(input_tensor)
        if skip_lm_head:
            if out is not None:
                out.deallocate(True)
            return None
        return out  # logits [1,1,chunk_local,vocab_shard], SP-sharded on seq / TP-sharded on vocab

    def gather_layer(self, slot_id: int, layer_idx: int, n_tokens: int, kv_caches=None):
        """Read one layer's device K/V cache back to NATURAL token order (un-rotating the block-cyclic
        SP layout). Returns (k, v) torch tensors in DEVICE convention: K is Meta-RoPE swizzled over the
        (full) head_dim — the caller reconciles vs the HF golden; V is raw. Shapes:
        k, v -> [1, num_kv_heads, n_tokens, head_dim]. No index_k (GQA)."""
        kv = self._resolve_kv(kv_caches)
        sp = self.config.sp_factor
        cols = self.config.tp_factor  # KV head c lives on col c
        nkv = self.hf_config.num_key_value_heads
        slot = slot_id * self.config.num_layers + layer_idx
        # shard-row -> natural global position (inverse of the update_padded_kv_cache writer).
        p = blockcyclic_positions(sp, self.config.chunk_size, self.config.max_seq_len)

        def gather(cache_tensor, col):
            dts = ttnn.get_device_tensors(cache_tensor)
            dev = torch.cat([ttnn.to_torch(dts[r * cols + col])[slot, 0].float() for r in range(sp)], dim=0)
            nat = torch.empty_like(dev)
            nat[p] = dev
            return nat[:n_tokens]

        k = torch.stack([gather(kv.k, c) for c in range(nkv)], dim=0).unsqueeze(0)
        v = torch.stack([gather(kv.v, c) for c in range(nkv)], dim=0).unsqueeze(0)
        return k, v

    def _kv_diag(self, gL, g_k, dev_k, g_v, dev_v, out_dir):
        """Bring-up diagnostic (gated by GPT_OSS_KV_DUMP) to localize a per-position K RoPE error.

        Dumps golden/device K & V (both Meta space, natural position order, [1, nkv, n_tokens,
        head_dim]) to ``out_dir`` and prints a compact per-SP-block + per-head K PCC so a single
        galaxy run reveals WHERE K diverges:
          * uniform low PCC everywhere            => base convention (unlikely; unit test is green)
          * degrades at SP-rank block boundaries  => per-rank global-position offset in the rope
          * scattered specific positions          => block-cyclic cos/sin reorder mismatch
        The dumped .pt tensors are re-loadable for offline per-position analysis on CPU."""

        def _pcc(a, b):
            a = a.reshape(-1).float()
            b = b.reshape(-1).float()
            a = a - a.mean()
            b = b - b.mean()
            d = a.norm() * b.norm()
            return float(torch.dot(a, b) / d) if d > 0 else 1.0

        sp = self.config.mesh_shape[0]
        n_tokens = dev_k.shape[2]
        chunk_local = self.config.chunk_size // sp  # SP-rank contiguous block width (one-shot)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "g_k": g_k,
                "dev_k": dev_k,
                "g_v": g_v,
                "dev_v": dev_v,
                "sp": sp,
                "chunk_local": chunk_local,
                "n_tokens": n_tokens,
                "layer": gL,
            },
            out_dir / f"layer_{gL}.pt",
        )
        blocks = []
        for r in range(sp):
            a, b = r * chunk_local, min((r + 1) * chunk_local, n_tokens)
            if a >= n_tokens:
                break
            blocks.append(f"blk{r}[{a}:{b}]={_pcc(g_k[:, :, a:b, :], dev_k[:, :, a:b, :]):.4f}")
        heads = " ".join(f"h{h}={_pcc(g_k[:, h], dev_k[:, h]):.4f}" for h in range(dev_k.shape[1]))
        logger.info(f"[kv-diag L{gL}] K per-SP-block (chunk_local={chunk_local}): {' '.join(blocks)}")
        logger.info(f"[kv-diag L{gL}] K per-head:     {heads}")
        logger.info(f"[kv-diag L{gL}] dumped -> {out_dir / f'layer_{gL}.pt'}")

    def logits_top1_oneshot(self, out, n_tokens):
        """One-shot: convert SP-seq + TP-vocab sharded prefill logits to per-position GLOBAL top-1
        token id in natural token order, first ``n_tokens``.

        out: per-device [1,1,chunk_local,vocab_shard].
        - SEQUENCE is sharded CONTIGUOUSLY across SP rows (prepare_inputs_prefill uses a
          ShardTensor2dMesh split on the seq dim: row r = positions [r*cl:(r+1)*cl]). This is the
          ACTIVATION layout and is NOT block-cyclic -- block-cyclic is only the KV-CACHE storage
          layout (gather_layer un-rotates that; the logits are a plain activation, so no un-rotation).
          => concatenating the rows in mesh order already gives natural token order.
        - VOCAB is col-parallel: col c holds a contiguous shard, and lm_head is PADDED to a pow2
          padded_vocab_size with ZEROS (model.py). => concat the col shards in order, then SLICE to the
          real vocab_size before argmax, else a zero-padded index can win when all real logits are < 0.

        Bring-up hook for the top-1 functional sign-off (galaxy_prefill_kv_pcc TOP1 mode). One-shot only
        (single chunk == whole seq). Validated end-to-end vs reference_top1 when the golden carries it."""
        sp = self.config.sp_factor
        cols = self.config.tp_factor
        vocab = self.hf_config.vocab_size
        dts = ttnn.get_device_tensors(out)
        rows = []
        for r in range(sp):
            shards = [ttnn.to_torch(dts[r * cols + c])[0, 0].float() for c in range(cols)]  # [cl, vshard]
            full = torch.cat(shards, dim=-1)[:, :vocab]  # [chunk_local, vocab] (drop pow2 zero-padding)
            rows.append(full.argmax(dim=-1))  # [chunk_local] global vocab id
        dev = torch.cat(rows, dim=0)  # contiguous SP-row order == natural token order (no un-rotation)
        return dev[:n_tokens]  # [n_tokens] argmax token ids

    def top1_check(self, our_top1, trace_dir):
        """Compare our per-position top-1 to the golden reference (reference_top1.safetensors, key
        top1_token_ids [n_tokens]). Returns (agreement_fraction, n_positions, gen_token_match).

        gen_token_match is the FIRST GENERATED token — the argmax at the LAST prompt position
        (our_top1[-1]), i.e. the token the model would emit after the prompt. (Position 0 is a
        teacher-forced next-token, not the generation.) Requires a FULL-length reference: a truncated /
        mismatched golden is rejected, not silently partially compared, since this is the definitive
        full-prompt sign-off."""
        from safetensors import safe_open

        ref_path = Path(trace_dir) / "reference_top1.safetensors"
        if not ref_path.exists():
            raise FileNotFoundError(
                f"{ref_path} missing - golden has no reference top-1 yet (ask the golden generator to "
                f"save logits.argmax(-1) as top1_token_ids)."
            )
        with safe_open(str(ref_path), framework="pt") as h:
            ref = h.get_tensor("top1_token_ids").reshape(-1).long()
        our = our_top1.long()
        if ref.numel() != our.numel():
            raise ValueError(
                f"reference_top1 has {ref.numel()} positions but prefill produced {our.numel()}; the "
                f"sign-off needs a full-prompt reference — regenerate the golden for this exact prompt."
            )
        agree = float((our == ref).float().mean())
        gen_match = bool(our[-1] == ref[-1])  # first generated token = argmax at the last prompt position
        return agree, our.numel(), gen_match

    def kv_cache_pcc_check(
        self, kv_caches=None, *, slot_id: int, n_chunks: int, trace_dir=None, first_layer_idx: int = 0
    ) -> float:
        """PCC the populated KV cache for ``slot_id`` against the golden trace; return the min per-layer
        PCC (K and V). Optional bring-up hook — never called in production serving.

        Golden layout: {trace_dir}/kv_cache/layer_N.safetensors
        with ``key_cache_layer_N`` (post-RoPE K, HF half-split convention) + ``value_cache_layer_N`` (raw
        V), each [1, num_kv_heads, seq_len, head_dim]. GQA => NO index_k. The device K is Meta-RoPE
        swizzled over the full head_dim, so the golden K's rotary slice is permuted HF->Meta first."""
        from safetensors import safe_open

        from models.common.utility_functions import comp_pcc

        assert trace_dir is not None, "kv_cache_pcc_check needs a golden trace_dir"
        token_ids = list(json.load(open(Path(trace_dir) / "metadata.json"))["token_ids"])
        n_tokens = len(token_ids)

        head_dim = self.hf_config.head_dim
        rotary_dim = getattr(self.hf_config, "rotary_dim", head_dim)
        half = rotary_dim // 2
        src = list(range(head_dim))
        for m in range(rotary_dim):
            src[m] = half * (m % 2) + (m // 2)  # HF half-split -> Meta interleaved
        src = torch.tensor(src, dtype=torch.long)

        kv_dir = Path(trace_dir) / "kv_cache"
        _dump_env = os.environ.get("GPT_OSS_KV_DUMP", "")
        if _dump_env == "all":
            _dump_set = set(range(first_layer_idx, first_layer_idx + self.config.num_layers))
        else:
            _dump_set = {int(x) for x in _dump_env.split(",") if x.strip()} if _dump_env else set()
        # Per-layer tensor dumps land next to the golden trace by default; GPT_OSS_KV_DUMP_DIR overrides.
        _dump_dir = os.environ.get("GPT_OSS_KV_DUMP_DIR") or (Path(trace_dir) / "kv_dump")
        logger.info(f"[kv-pcc] per-layer K / V vs golden ({trace_dir}):")
        min_k, min_v = 1.0, 1.0
        for L in range(self.config.num_layers):
            gL = first_layer_idx + L
            dev_k, dev_v = self.gather_layer(slot_id=slot_id, layer_idx=L, n_tokens=n_tokens, kv_caches=kv_caches)
            with safe_open(str(kv_dir / f"layer_{gL}.safetensors"), framework="pt") as h:
                g_k = h.get_tensor(f"key_cache_layer_{gL}").float()[:, :, :n_tokens, :][..., src]  # HF -> Meta
                g_v = h.get_tensor(f"value_cache_layer_{gL}").float()[:, :, :n_tokens, :]
            pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
            pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
            min_k, min_v = min(min_k, pcc_k), min(min_v, pcc_v)
            logger.info(f"  layer {gL:>2}: K={pcc_k:.5f} V={pcc_v:.5f}")
            if gL in _dump_set:
                self._kv_diag(gL, g_k, dev_k, g_v, dev_v, _dump_dir)
        logger.info(f"[kv-pcc] min PCC across {self.config.num_layers} layers: K={min_k:.5f} V={min_v:.5f}")
        return min(min_k, min_v)
