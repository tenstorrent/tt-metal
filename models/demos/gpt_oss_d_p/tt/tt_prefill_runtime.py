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

IMPORTANT (P2 not yet implemented): the GQA cache-READ attention path (``cached_len > 0``) is
``NotImplementedError`` in ``attention/prefill.py`` (needs the ring/paged chunk-position-aware SDPA).
So MULTI-chunk prefill fails on the 2nd chunk today. ONE-SHOT prefill (a single chunk covering the
whole prompt, ``cached_len == 0``) is the supported path and is what the galaxy KV-PCC harness uses.
The KV-cache WRITE + gather-back are done and validated; only the read-back-for-attention is missing.
"""

import json
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
        """Warm up the kernels by running one zero-token chunk through prefill_chunk (JIT-compiles)."""
        assert self.model_built
        chunk = self.config.chunk_size
        logger.info(f"GPT-OSS TtPrefillRuntime.compile() — warming up one {chunk}-token chunk")
        tt_input = self.make_chunk_input([0] * chunk)
        self.prefill_chunk(tt_input, kv_caches, slot_id=0, actual_start=0, actual_end=chunk)
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
    ) -> Optional[ttnn.Tensor]:
        """Prefill ONE chunk into user ``slot_id``'s slice of the KV cache (self-owned or the engine's
        ``kv_caches``). Returns None (skip_lm_head) — the populated cache is the output.

        [actual_start, actual_end): absolute KV-position range of this chunk's real tokens. actual_start
        is the cache write offset (valid prefix already cached); the last chunk's tail may be pad, so
        actual_end < actual_start + chunk_size. Call once per chunk, in order.

        NOTE (P2): actual_start > 0 drives the cache-READ attention path, which is NotImplementedError
        today (attention/prefill.py). One-shot prefill (actual_start == 0, single chunk) is supported.
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
        k, v -> [1, num_kv_heads, n_tokens, head_dim]. No index_k (GQA, unlike M3)."""
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

    def kv_cache_pcc_check(
        self, kv_caches=None, *, slot_id: int, n_chunks: int, trace_dir=None, first_layer_idx: int = 0
    ) -> float:
        """PCC the populated KV cache for ``slot_id`` against the golden trace; return the min per-layer
        PCC (K and V). Optional bring-up hook — never called in production serving.

        Golden layout (see scripts/generate_golden_kv_cache.py): {trace_dir}/kv_cache/layer_N.safetensors
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
        logger.info(f"[kv-pcc] min PCC across {self.config.num_layers} layers: K={min_k:.5f} V={min_v:.5f}")
        return min(min_k, min_v)
