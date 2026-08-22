# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full autoregressive TTNN model for poolside/Laguna-XS-2.1 on a 1×D Blackhole mesh.

Assembles the topology-generic decoder (``tt/multichip_decoder.py``, ``MultichipDecoder`` — TP=D
attention/dense + EP=D routed MoE, replicated BF16 residual, 2 ``all_reduce``/layer for D>1, BFP8 local
KV cache, packed gate+up via ``PACK_GATE_UP``) into the whole 40-layer autoregressive path: token
embedding, the layer stack, final RMSNorm, and the LM head.

Design contract carried forward from the decoder stage (do NOT weaken):
  * **Inter-layer residual is replicated BF16** ``[1,seq,H]`` (prefill) / ``[1,1,B,H]`` (decode),
    H=2048. No gather/reshard/all-reduce between layers — layers stack directly.
  * Each layer keeps its own EP-sharded MoE experts, TP-sharded attention/dense, local 2-KV-head
    BFP8 paged cache, and the 2 intra-layer all-reduces for D>1. This wrapper adds NO inter-layer
    collective.
  * The **LM head is column-sharded across the mesh** (device d owns vocab shard
    ``[d·V/D:(d+1)·V/D]``), so the token-out path can do local top-k per device and all-gather only
    the small top-k candidate set (canonical split sampling) instead of a full-vocab all-gather.
    The final norm runs on the replicated hidden (exact locally, like the decoder RMSNorms).
  * Token **embedding is replicated** (every device can look up any token id), so the decode token
    feedback (sampled id -> embedding -> hidden) stays fully on device inside the trace.

The model exposes granular pieces so the generator can compose a single on-device decode trace
(embed -> layer stack -> norm -> LM head -> sampling -> token feedback) and a chunked prefill.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

import ttnn

from .dflash_reference import DFLASH_TARGET_LAYER_IDS, DFlashTargetAuxCapture
from .multichip_decoder import MultichipDecoder
from .optimized_decoder import PrecisionPolicy, _cached_device_tensor, weight_cache_key

MODEL_ID = "poolside/Laguna-XS-2.1"

# Canonical datatype-sweep-selected precision policy. When present, this is the required
# config artifact that the default construction path (``from_pretrained`` -> generator ->
# vLLM) consumes automatically, so the sweep-selected weight/activation/CCL/KV/fidelity
# policy is the one actually built. Absent -> the in-code ``PrecisionPolicy()`` defaults.
_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELECTED_PRECISION_CONFIG_PATH = os.path.join(_MODEL_DIR, "doc", "datatype_sweep", "selected_precision_config.json")


def load_selected_precision_policy(path=None):
    """Return (PrecisionPolicy, source_str). Reads the sweep-selected config JSON's ``policy``
    block if the file exists; otherwise returns the in-code defaults. ``TT_LAGUNA_PRECISION_CONFIG``
    env var overrides the path (used by the sweep driver to point at a candidate config)."""
    path = path or os.environ.get("TT_LAGUNA_PRECISION_CONFIG") or SELECTED_PRECISION_CONFIG_PATH
    if path and os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
        pol = d.get("policy", d)
        return PrecisionPolicy.from_dict(pol), path
    return PrecisionPolicy(), "in-code PrecisionPolicy() defaults"


# --------------------------------------------------------------------------- #
# Checkpoint helpers (top-level tensors: embed_tokens, norm, lm_head)
# --------------------------------------------------------------------------- #
def _snapshot_and_index():
    from huggingface_hub import snapshot_download

    d = snapshot_download(MODEL_ID, allow_patterns=["*.json", "*.py"])
    with open(os.path.join(d, "model.safetensors.index.json")) as f:
        return d, json.load(f)["weight_map"]


def _resolve_shard(snap_dir, shard_name):
    p = os.path.join(snap_dir, shard_name)
    if os.path.exists(p):
        return p
    from huggingface_hub import hf_hub_download

    return hf_hub_download(MODEL_ID, shard_name)


def load_top_level_tensors(keys):
    """Load the given top-level checkpoint tensors as fp32 torch tensors."""
    from safetensors import safe_open

    snap_dir, wm = _snapshot_and_index()
    shards = sorted({wm[k] for k in keys})
    out = {}
    for shard in shards:
        path = _resolve_shard(snap_dir, shard)
        with safe_open(path, "pt") as f:
            present = set(f.keys())
            for k in keys:
                if k in present:
                    out[k] = f.get_tensor(k).to(torch.float32)
    return out


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    hidden: int
    vocab: int
    num_layers: int
    eps: float
    max_position_embeddings: int
    tie_word_embeddings: bool


class LagunaModel:
    """40-layer TTNN Laguna-XS-2.1 on a supported 1×D Blackhole mesh."""

    def __init__(self, hf_config, layers, embed_w, norm_w, lm_head_w, lm_head_ds, meta, mesh_device):
        self.hf_config = hf_config
        self.layers = layers  # list[MultichipDecoder]
        self.embed_w = embed_w  # [vocab, H] ROW_MAJOR bf16, replicated
        self.norm_w = norm_w  # [1,1,1,H] bf16, replicated
        self.lm_head_w = lm_head_w  # [H, V/D] bf16, mesh-sharded on dim1 (vocab)
        self.lm_head_ds = lm_head_ds  # DRAM-width-sharded copy for decode matmul
        self.device = mesh_device
        self.D = mesh_device.get_num_devices()
        self.meta = meta
        self.precision_policy = meta.get("precision_policy")
        self.lm_head_dtype = meta.get("lm_head_dtype")
        self.cfg = ModelConfig(
            hidden=hf_config.hidden_size,
            vocab=hf_config.vocab_size,
            num_layers=len(layers),
            eps=hf_config.rms_norm_eps,
            max_position_embeddings=hf_config.max_position_embeddings,
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        )
        self.per_device_vocab = self.cfg.vocab // self.D
        # Reuse a layer's precise-norm compute-kernel config for the final norm.
        self._norm_ck = layers[0]._norm_ck
        self._lm_ck = layers[0]._ck_lofi
        self.use_dram_sharded = layers[0].use_dram_sharded

    # ---- construction ------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        mesh_device,
        *,
        hf_config=None,
        max_seq_len=8192,
        num_layers=None,
        decoder_cls=None,
        precision_policy=None,
        precision_config_path=None,
        lm_head_dtype=None,
    ):
        """Build the full model on ``mesh_device``.

        ``max_seq_len`` bounds the per-kind RoPE tables (kept modest for accuracy runs; the
        advertised 262144 capacity is documented separately in ``doc/context_contract.json``).
        ``num_layers`` (or a 2/3-length list of layer indices) builds a reduced probe.

        Precision policy resolution (datatype sweep): if ``precision_policy`` (a
        ``PrecisionPolicy``) is given it is used directly; else if ``precision_config_path``
        is given that JSON is loaded; else the canonical sweep-selected config artifact
        (``SELECTED_PRECISION_CONFIG_PATH`` / ``TT_LAGUNA_PRECISION_CONFIG``) is loaded if it
        exists, else in-code ``PrecisionPolicy()`` defaults. This is what makes the
        sweep-selected weight/activation/CCL/KV/compute-fidelity policy the one actually built
        by the default construction path.

        ``lm_head_dtype`` overrides the column-sharded LM-head weight dtype; when None the
        policy's ``lm_head`` field is used (default BFP8: the reduced tt-perf-report shows the
        LM head is the largest DRAM-bound terminal matmul, 32x2048x25088 ~73.8% DRAM util, and
        halving its weight bytes measured 41.7% faster while preserving the greedy token). Pass
        ``lm_head_dtype=ttnn.bfloat16`` for the functional/full_model A/B baseline.
        """
        from models.autoports.poolside_laguna_xs_2_1.tests import laguna_weights as W

        D = mesh_device.get_num_devices()
        mesh_shape = tuple(mesh_device.shape)
        if D not in (1, 2, 4):
            raise ValueError(f"Laguna supports D=1, 2, or 4 devices; got D={D}")
        if mesh_shape != (1, D):
            raise ValueError(f"Laguna requires a 1×D mesh; got shape={mesh_shape} for D={D}")

        if precision_policy is None:
            precision_policy, _ = load_selected_precision_policy(precision_config_path)
        if lm_head_dtype is None:
            lm_head_dtype = precision_policy.lm_head

        if hf_config is None:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        decoder_cls = decoder_cls or MultichipDecoder

        total = hf_config.num_hidden_layers
        if isinstance(num_layers, (list, tuple)):
            layer_indices = list(num_layers)
        else:
            n = num_layers or total
            layer_indices = list(range(n))

        # build the two (cos,sin) RoPE tables ONCE, keyed by attention kind, threaded into
        # every layer via this shared dict. The first full-attention layer and the first sliding layer
        # build+cache their table; all later layers of that kind reuse the SAME device tensors. This
        # replaces the old "each of 40 layers builds its own, then _dedup_rope frees 38" pattern —
        # removing the ~4.7 GB transient peak and 38 redundant trust-remote-code table builds. Tables
        # depend only on (attention_type, max_seq_len, config), so sharing is exact.
        rope_tables: dict = {}
        layers = []
        for li in layer_indices:
            raw = W.load_layer_tensors(li)
            dec = decoder_cls.from_state_dict(
                raw,
                hf_config=hf_config,
                layer_idx=li,
                mesh_device=mesh_device,
                max_seq_len=max_seq_len,
                policy=precision_policy,
                rope_tables=rope_tables,
            )
            dec.layer_idx = li
            layers.append(dec)

        # Top-level tensors.
        top = load_top_level_tensors(["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"])
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        # Top-level weights are disk-cached the same way per-layer weights are (plan Stage 0).
        # layer_idx="top"; mesh tag encodes the mapping + device count D. See
        # optimized_decoder._cached_device_tensor / weight_cache_key.
        embed_w = _cached_device_tensor(
            lambda: top["model.embed_tokens.weight"],  # [V, H]
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_key=weight_cache_key("embed_tokens", "top", f"rep_d{D}"),
        )
        H = hf_config.hidden_size
        norm_w = _cached_device_tensor(
            lambda: top["model.norm.weight"].reshape(1, 1, 1, H),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
            cache_key=weight_cache_key("norm", "top", f"rep_d{D}"),
        )
        # LM head: HF [V, H] -> ttnn [H, V], column-sharded on the vocab dim across the mesh.
        V = hf_config.vocab_size
        assert V % D == 0, f"vocab {V} must divide mesh {D}"
        per_dev_vocab = V // D
        # LM head N (per-device vocab shard) is large, so a plain tiled matmul that fans N across the
        # full compute grid is the right shape here — the DRAM-width-sharded decode-matmul helper
        # (tuned for the K-bound attention/MLP weights) overflows L1 at this N. The plain tiled matmul
        # already saturates DRAM (~73.8% util on 98 cores); BFP8 weights (optimized default) halve the
        # DRAM bytes of this DRAM-bound op for a measured 41.7% LM-head speedup (token-preserving).
        # lm_head_dtype (bf16/bfp8) is encoded in the cache filename by ttnn.as_tensor's suffix.
        lm_head_w = _cached_device_tensor(
            lambda: top["lm_head.weight"].t().contiguous(),  # [H, V]
            device=mesh_device,
            dtype=lm_head_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
            cache_key=weight_cache_key("lm_head", "top", f"sh1_d{D}"),
        )
        meta = {
            "max_seq_len": max_seq_len,
            "layer_indices": layer_indices,
            "per_device_vocab": per_dev_vocab,
            "precision_policy": precision_policy,
            "lm_head_dtype": lm_head_dtype,
        }
        return cls(hf_config, layers, embed_w, norm_w, lm_head_w, None, meta, mesh_device)

    # ---- KV cache / page table (per-layer caches; one shared page table) ---- #
    def alloc_kv_cache(self, max_users, max_seq_len, block_size=32, dtype=None):
        return [dec.alloc_kv_cache(max_users, max_seq_len, block_size=block_size, dtype=dtype) for dec in self.layers]

    def make_page_table(self, num_users, blocks_per_user):
        return self.layers[0].make_page_table(num_users, blocks_per_user)

    def reset_kv_cache(self, kv_cache):
        """Zero all KV-cache tensors in place (keeps buffers/traces alive)."""
        for entry in kv_cache:
            ttnn.mul(entry["k"], 0.0, output_tensor=entry["k"])
            ttnn.mul(entry["v"], 0.0, output_tensor=entry["v"])

    # ---- embedding --------------------------------------------------------- #
    def embed_prefill(self, tok_1S):
        """tok_1S: uint32 [1, seq] on device -> [1, seq, H] bf16 TILE (replicated)."""
        emb = ttnn.embedding(tok_1S, self.embed_w, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(emb, (1, tok_1S.shape[-1], self.cfg.hidden))

    def embed_decode(self, tok_1B):
        """tok_1B: uint32 [1, B] on device -> [1, 1, B, H] bf16 TILE (replicated)."""
        emb = ttnn.embedding(tok_1B, self.embed_w, layout=ttnn.TILE_LAYOUT)
        return ttnn.reshape(emb, (1, 1, tok_1B.shape[-1], self.cfg.hidden))

    # ---- forward: layer stack ---------------------------------------------- #
    def _build_decode_rope(self, rope_idx, B):
        """Compute the two distinct decode RoPE gathers (one per attention kind) ONCE per
        step, instead of every one of the 40 layers redundantly re-gathering them. Returns
        {attention_type: (cos, sin)} — the DRAM ``_rope_decode`` output ([1,B,1,rotary_dim] TILE).

        Only the DRAM cos/sin are shared. The L1 HEIGHT-SHARDED form (``_shard_cossin``) is NOT hoisted:
        L1 is scratch and a shared cos_sh gets clobbered by later layers' activations (measured: teacher
        top1 0.95->0.58). Each layer shards its own cos_sh from the shared cos in ``decode_forward``.
        Sharing cos/sin is bit-identical: DRAM-persistent, and the source table is shared per kind
        so a representative layer's gather reproduces every layer's local result exactly."""
        ctx = {}
        for dec in self.layers:
            kind = dec.cfg.attention_type
            if kind in ctx:
                continue
            cos = dec._rope_decode(rope_idx, B)
            sin = dec._rope_decode(rope_idx, B, sin=True)
            ctx[kind] = (cos, sin)
        return ctx

    def _build_prefill_rope(self, start_pos, seq, runtime_offsets=None):
        """Compute the two distinct single-shot prefill RoPE contexts (one per attention
        kind) ONCE. Runtime-offset mode returns one preallocated pair per outer chunk;
        scalar fallback returns the historical single ``(cos, sin)`` pair."""
        ctx = {}
        for dec in self.layers:
            kind = dec.cfg.attention_type
            if kind in ctx:
                continue
            if runtime_offsets is None:
                cos = dec._rope_prefill(start_pos, seq)
                sin = dec._rope_prefill(start_pos, seq, sin=True)
                ctx[kind] = (cos, sin)
                continue
            if int(runtime_offsets.bucket_len) != int(seq):
                raise ValueError(f"prefill runtime bucket {runtime_offsets.bucket_len} does not match hidden seq {seq}")
            outputs = runtime_offsets.rope_outputs.get(kind)
            if outputs is None:
                raise ValueError(f"prefill runtime has no RoPE output slots for attention kind {kind!r}")
            mats = []
            for position_ids, (cos_out, sin_out) in zip(runtime_offsets.position_ids, outputs):
                cos = dec._rope_prefill_indexed(position_ids, output_tensor=cos_out)
                sin = dec._rope_prefill_indexed(position_ids, sin=True, output_tensor=sin_out)
                mats.append((cos, sin))
            ctx[kind] = tuple(mats)
        return ctx

    def prefill_layers(
        self,
        hidden_1SH,
        kv_cache,
        page_table,
        *,
        fill_page_table=None,
        fill_page_table_base_pos=0,
        user_id=0,
        start_pos=0,
        runtime_offsets=None,
    ):
        # Attention and paged-fill tables are distinct during bucketed vLLM prefill. Either may be a
        # single tensor (uniform KV) or a per-layer list (hybrid KV). Direct/non-vLLM callers omit the
        # fill table and retain the historical behavior of using ``page_table`` for both operations.
        if fill_page_table is None:
            fill_page_table = page_table
        per_layer = isinstance(page_table, (list, tuple))
        fill_per_layer = isinstance(fill_page_table, (list, tuple))
        if per_layer and len(page_table) < len(self.layers):
            raise ValueError(
                f"prefill attention page table has {len(page_table)} entries for {len(self.layers)} layers"
            )
        if fill_per_layer and len(fill_page_table) < len(self.layers):
            raise ValueError(
                f"prefill fill page table has {len(fill_page_table)} entries for {len(self.layers)} layers"
            )
        seq = hidden_1SH.shape[-2]
        # Shared per-kind RoPE for legacy single-shot prefill. Runtime-offset D2 prefills always use
        # the pipeline implementation (including small resumed buckets), so their context remains a
        # tuple with one preallocated pair per outer chunk. Legacy pipelines compute RoPE locally.
        # TT_LAGUNA_NO_ROPE_HOIST=1 forces the legacy per-layer local compute (A/B diagnostic).
        _no_hoist = os.environ.get("TT_LAGUNA_NO_ROPE_HOIST") == "1"
        if runtime_offsets is not None:
            # Indexed RoPE is mandatory for the arbitrary-offset serving path. Every output is
            # preallocated and built once per attention kind/chunk, then reused by all layers.
            rope_ctx = self._build_prefill_rope(start_pos, seq, runtime_offsets=runtime_offsets)
        else:
            rope_ctx = (
                None if (_no_hoist or seq > self.layers[0].PIPE_CHUNK) else self._build_prefill_rope(start_pos, seq)
            )
        h = hidden_1SH
        for i, (dec, kv) in enumerate(zip(self.layers, kv_cache)):
            pt = page_table[i] if per_layer else page_table
            fill_pt = fill_page_table[i] if fill_per_layer else fill_page_table
            rm = rope_ctx[dec.cfg.attention_type] if rope_ctx is not None else None
            h = dec.prefill_forward(
                h,
                kv,
                pt,
                fill_page_table=fill_pt,
                fill_page_table_base_pos=fill_page_table_base_pos,
                user_id=user_id,
                start_pos=start_pos,
                rope_mats=rm,
                runtime_offsets=runtime_offsets,
            )
        return h

    def _validate_dflash_target_capture(self, enable_experimental):
        """Fail closed before the separate DFlash capture path touches a layer."""

        if not bool(enable_experimental):
            raise RuntimeError(
                "DFlash target auxiliary capture is experimental and default-off; "
                "pass enable_experimental=True only from the qualified DFlash path"
            )
        layer_indices = tuple(int(getattr(layer, "layer_idx", index)) for index, layer in enumerate(self.layers))
        expected_stack = tuple(range(40))
        if layer_indices != expected_stack:
            raise RuntimeError(
                "DFlash target auxiliary capture requires the exact full 40-layer target stack; "
                f"got layer indices {layer_indices}"
            )
        if tuple(DFLASH_TARGET_LAYER_IDS) != (1, 13, 25, 33, 39):
            raise RuntimeError(f"unexpected DFlash target auxiliary layer contract: {DFLASH_TARGET_LAYER_IDS}")
        return set(DFLASH_TARGET_LAYER_IDS)

    def prefill_layers_with_dflash_aux(
        self,
        hidden_1SH,
        kv_cache,
        page_table,
        *,
        fill_page_table=None,
        fill_page_table_base_pos=0,
        user_id=0,
        start_pos=0,
        runtime_offsets=None,
        valid_seq_len=None,
        enable_experimental=False,
    ):
        """Run the target prefill and separately capture five post-layer states.

        This is deliberately a distinct entry point: the established target
        ``prefill_layers`` loop has no DFlash condition, append, or extra tensor
        lifetime.  Only the last 511 valid rows are retained because older rows
        are outside the draft's inclusive 512-token sliding window.
        """

        capture_ids = self._validate_dflash_target_capture(enable_experimental)
        if fill_page_table is None:
            fill_page_table = page_table
        per_layer = isinstance(page_table, (list, tuple))
        fill_per_layer = isinstance(fill_page_table, (list, tuple))
        if per_layer and len(page_table) < len(self.layers):
            raise ValueError(
                f"prefill attention page table has {len(page_table)} entries for {len(self.layers)} layers"
            )
        if fill_per_layer and len(fill_page_table) < len(self.layers):
            raise ValueError(
                f"prefill fill page table has {len(fill_page_table)} entries for {len(self.layers)} layers"
            )
        seq = int(hidden_1SH.shape[-2])
        valid_seq_len = seq if valid_seq_len is None else int(valid_seq_len)
        if not 1 <= valid_seq_len <= seq:
            raise ValueError(f"valid_seq_len must be in [1, {seq}], got {valid_seq_len}")
        if int(start_pos) < 0 or int(start_pos) + valid_seq_len > self.cfg.max_position_embeddings:
            raise ValueError(
                f"DFlash target prefill interval [{start_pos}, {int(start_pos) + valid_seq_len}) "
                f"is outside [0, {self.cfg.max_position_embeddings})"
            )
        keep = min(valid_seq_len, 511)
        capture_begin = valid_seq_len - keep

        _no_hoist = os.environ.get("TT_LAGUNA_NO_ROPE_HOIST") == "1"
        if runtime_offsets is not None:
            rope_ctx = self._build_prefill_rope(start_pos, seq, runtime_offsets=runtime_offsets)
        else:
            rope_ctx = (
                None if (_no_hoist or seq > self.layers[0].PIPE_CHUNK) else self._build_prefill_rope(start_pos, seq)
            )
        h = hidden_1SH
        captured = []
        for i, (dec, kv) in enumerate(zip(self.layers, kv_cache)):
            pt = page_table[i] if per_layer else page_table
            fill_pt = fill_page_table[i] if fill_per_layer else fill_page_table
            rm = rope_ctx[dec.cfg.attention_type] if rope_ctx is not None else None
            h = dec.prefill_forward(
                h,
                kv,
                pt,
                fill_page_table=fill_pt,
                fill_page_table_base_pos=fill_page_table_base_pos,
                user_id=user_id,
                start_pos=start_pos,
                rope_mats=rm,
                runtime_offsets=runtime_offsets,
            )
            if int(dec.layer_idx) in capture_ids:
                captured.append(
                    ttnn.slice(
                        h,
                        [0, capture_begin, 0],
                        [1, valid_seq_len, self.cfg.hidden],
                    )
                )
        if len(captured) != len(DFLASH_TARGET_LAYER_IDS):
            raise RuntimeError(
                f"captured {len(captured)} DFlash target states, expected {len(DFLASH_TARGET_LAYER_IDS)}"
            )
        flattened = ttnn.concat(captured, dim=-1)
        capture = DFlashTargetAuxCapture(
            hidden_states=flattened,
            start_position=int(start_pos) + capture_begin,
            row_count=keep,
        )
        return h, capture

    def decode_layers(self, hidden_1BH, cur_pos, rope_idx, page_table, kv_cache, sequential_kv_write=False):
        # ``sequential_kv_write`` serializes the per-row paged_update_cache when the batch rows are
        # spec-decode candidates sharing ONE user's physical KV blocks (see MultichipDecoder.decode_forward);
        # off for normal serving (disjoint per-user blocks — no race).
        per_layer = isinstance(page_table, (list, tuple))
        B = hidden_1BH.shape[-2]
        # two distinct decode RoPE gathers (full vs sliding) computed ONCE, threaded per layer.
        # TT_LAGUNA_NO_ROPE_HOIST=1 forces per-layer local compute (A/B diagnostic).
        rope_ctx = None if os.environ.get("TT_LAGUNA_NO_ROPE_HOIST") == "1" else self._build_decode_rope(rope_idx, B)
        h = hidden_1BH
        for i, (dec, kv) in enumerate(zip(self.layers, kv_cache)):
            pt = page_table[i] if per_layer else page_table
            h = dec.decode_forward(
                h,
                cur_pos,
                rope_idx,
                pt,
                kv,
                sequential_kv_write=sequential_kv_write,
                rope_mats=(rope_ctx[dec.cfg.attention_type] if rope_ctx is not None else None),
            )
        return h

    def decode_layers_with_dflash_aux(
        self,
        hidden_1BH,
        cur_pos,
        rope_idx,
        page_table,
        kv_cache,
        *,
        absolute_position,
        sequential_kv_write=False,
        enable_experimental=False,
    ):
        """Run one contiguous target-verify block and capture post-layer states.

        The decode batch dimension represents 1..16 consecutive rows belonging
        to one request, not independent served requests.  Multi-row verification
        must serialize shared-block KV writes.  Keeping this API separate leaves
        the normal traced decode graph byte-for-byte untouched.
        """

        capture_ids = self._validate_dflash_target_capture(enable_experimental)
        position = int(absolute_position)
        B = int(hidden_1BH.shape[-2])
        if not 1 <= B <= 16:
            raise ValueError(f"DFlash target auxiliary verify requires 1..16 contiguous rows, got {B}")
        if B > 1 and not bool(sequential_kv_write):
            raise ValueError("multi-row DFlash target verify requires sequential_kv_write=True")
        if position < 0 or position + B > self.cfg.max_position_embeddings:
            raise ValueError(
                f"DFlash target decode interval [{position}, {position + B}) is outside "
                f"[0, {self.cfg.max_position_embeddings})"
            )
        per_layer = isinstance(page_table, (list, tuple))
        if per_layer and len(page_table) < len(self.layers):
            raise ValueError(f"decode page table has {len(page_table)} entries for {len(self.layers)} layers")
        rope_ctx = None if os.environ.get("TT_LAGUNA_NO_ROPE_HOIST") == "1" else self._build_decode_rope(rope_idx, B)
        h = hidden_1BH
        captured = []
        for i, (dec, kv) in enumerate(zip(self.layers, kv_cache)):
            pt = page_table[i] if per_layer else page_table
            h = dec.decode_forward(
                h,
                cur_pos,
                rope_idx,
                pt,
                kv,
                sequential_kv_write=sequential_kv_write,
                rope_mats=(rope_ctx[dec.cfg.attention_type] if rope_ctx is not None else None),
            )
            if int(dec.layer_idx) in capture_ids:
                captured.append(h)
        if len(captured) != len(DFLASH_TARGET_LAYER_IDS):
            raise RuntimeError(
                f"captured {len(captured)} DFlash target states, expected {len(DFLASH_TARGET_LAYER_IDS)}"
            )
        flattened = ttnn.reshape(
            ttnn.concat(captured, dim=-1),
            (1, B, len(DFLASH_TARGET_LAYER_IDS) * self.cfg.hidden),
        )
        return h, DFlashTargetAuxCapture(
            hidden_states=flattened,
            start_position=position,
            row_count=B,
        )

    # ---- terminal: final norm + LM head ------------------------------------ #
    def final_norm(self, hidden):
        return ttnn.rms_norm(hidden, weight=self.norm_w, epsilon=self.cfg.eps, compute_kernel_config=self._norm_ck)

    def lm_head_shards_prefill(self, hidden_1SH):
        """Interleaved matmul (prefill): hidden [1,seq,H] -> per-device logit shard [1,seq,V/D]."""
        normed = self.final_norm(hidden_1SH)
        return ttnn.linear(normed, self.lm_head_w, compute_kernel_config=self._lm_ck)

    def lm_head_shards_decode(self, hidden_1BH):
        """Tiled matmul (decode): hidden [1,1,B,H] -> per-device logit shard [1,1,B,V/D]."""
        normed = self.final_norm(hidden_1BH)
        return ttnn.linear(normed, self.lm_head_w, compute_kernel_config=self._lm_ck)

    def lm_head_shards_dflash(self, draft_hidden, *, enable_experimental=False):
        """Project DFlash-normalized rows with the target-owned LM head.

        The draft checkpoint owns and applies its own final RMSNorm.  Applying
        the target model's final norm here would be a silent double-normalization
        error, so this explicit default-off method performs only the raw target
        LM-head projection.
        """

        if not bool(enable_experimental):
            raise RuntimeError(
                "raw target LM-head projection is reserved for experimental DFlash; " "pass enable_experimental=True"
            )
        if int(draft_hidden.shape[-1]) != self.cfg.hidden:
            raise ValueError(
                f"DFlash hidden width {draft_hidden.shape[-1]} does not match target width {self.cfg.hidden}"
            )
        # Proposal top-1 is sensitive to accumulated five-layer draft error.
        # Only 15 rows are projected, so use the existing precise/fp32-dest
        # kernel instead of the target's throughput-tuned normal LM-head kernel.
        return ttnn.linear(draft_hidden, self.lm_head_w, compute_kernel_config=self._norm_ck)

    def logits_to_host(self, logit_shards):
        """Gather the mesh-sharded vocab shards into a full [.., V] host tensor (prefill/debug only —
        NOT used on the measured token-out decode path)."""
        return ttnn.to_torch(logit_shards, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1))
