# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full autoregressive TTNN model for poolside/Laguna-XS-2.1 (Blackhole p300c ×4, 1×4 mesh).

Assembles the completed optimized multichip decoder (``tt/optimized_multichip_decoder.py``,
``OptimizedMultichipDecoder`` — TP=4 attention/dense + EP=4 routed MoE, replicated BF16 residual,
2 ring ``all_reduce``/layer, BFP8 local KV cache, packed gate+up) into the whole 40-layer
autoregressive path: token embedding, the layer stack, final RMSNorm, and the LM head.

Design contract carried forward from the decoder stage (do NOT weaken):
  * **Inter-layer residual is replicated BF16** ``[1,seq,H]`` (prefill) / ``[1,1,B,H]`` (decode),
    H=2048. No gather/reshard/all-reduce between layers — layers stack directly.
  * Each layer keeps its own EP-sharded MoE experts, TP-sharded attention/dense, local 2-KV-head
    BFP8 paged cache, and the 2 intra-layer ring all_reduce. This wrapper adds NO inter-layer
    collective.
  * The **LM head is column-sharded across the mesh** (device d owns vocab shard
    ``[d·V/4:(d+1)·V/4]``), so the token-out path can do local top-k per device and all-gather only
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

from .optimized_decoder import PrecisionPolicy
from .optimized_multichip_decoder import OptimizedMultichipDecoder

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
    """40-layer TTNN Laguna-XS-2.1 on the 1×4 Blackhole mesh."""

    def __init__(self, hf_config, layers, embed_w, norm_w, lm_head_w, lm_head_ds, meta, mesh_device):
        self.hf_config = hf_config
        self.layers = layers  # list[OptimizedMultichipDecoder]
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

        if precision_policy is None:
            precision_policy, _ = load_selected_precision_policy(precision_config_path)
        if lm_head_dtype is None:
            lm_head_dtype = precision_policy.lm_head

        if hf_config is None:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        decoder_cls = decoder_cls or OptimizedMultichipDecoder

        total = hf_config.num_hidden_layers
        if isinstance(num_layers, (list, tuple)):
            layer_indices = list(num_layers)
        else:
            n = num_layers or total
            layer_indices = list(range(n))

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
            )
            dec.layer_idx = li
            layers.append(dec)

        # Deduplicate RoPE tables by attention kind to bound device DRAM: all full-attention layers
        # share one (cos,sin) pair, all sliding layers share another. Tables depend only on
        # (attention_type, max_seq_len, config), so this is exact.
        cls._dedup_rope(layers)

        # Top-level tensors.
        top = load_top_level_tensors(["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"])
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        embed_w = ttnn.from_torch(
            top["model.embed_tokens.weight"],  # [V, H]
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicate,
        )
        H = hf_config.hidden_size
        norm_w = ttnn.from_torch(
            top["model.norm.weight"].reshape(1, 1, 1, H),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=replicate,
        )
        # LM head: HF [V, H] -> ttnn [H, V], column-sharded on the vocab dim across the mesh.
        V = hf_config.vocab_size
        D = mesh_device.get_num_devices()
        assert V % D == 0, f"vocab {V} must divide mesh {D}"
        per_dev_vocab = V // D
        lm_t = top["lm_head.weight"].t().contiguous()  # [H, V]
        # LM head N (per-device vocab shard) is large, so a plain tiled matmul that fans N across the
        # full compute grid is the right shape here — the DRAM-width-sharded decode-matmul helper
        # (tuned for the K-bound attention/MLP weights) overflows L1 at this N. The plain tiled matmul
        # already saturates DRAM (~73.8% util on 98 cores); BFP8 weights (optimized default) halve the
        # DRAM bytes of this DRAM-bound op for a measured 41.7% LM-head speedup (token-preserving).
        lm_head_w = ttnn.from_torch(
            lm_t,
            dtype=lm_head_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
        )
        meta = {
            "max_seq_len": max_seq_len,
            "layer_indices": layer_indices,
            "per_device_vocab": per_dev_vocab,
            "precision_policy": precision_policy,
            "lm_head_dtype": lm_head_dtype,
        }
        return cls(hf_config, layers, embed_w, norm_w, lm_head_w, None, meta, mesh_device)

    @staticmethod
    def _dedup_rope(layers):
        shared = {}
        for dec in layers:
            kind = dec.cfg.attention_type
            if kind not in shared:
                shared[kind] = (dec.cos_2d, dec.sin_2d)
            else:
                cos, sin = shared[kind]
                if dec.cos_2d is not cos:
                    ttnn.deallocate(dec.cos_2d)
                    ttnn.deallocate(dec.sin_2d)
                    dec.cos_2d = cos
                    dec.sin_2d = sin

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
    def prefill_layers(self, hidden_1SH, kv_cache, page_table, *, user_id=0, start_pos=0):
        h = hidden_1SH
        for dec, kv in zip(self.layers, kv_cache):
            h = dec.prefill_forward(h, kv, page_table, user_id=user_id, start_pos=start_pos)
        return h

    def decode_layers(self, hidden_1BH, cur_pos, rope_idx, page_table, kv_cache):
        h = hidden_1BH
        for dec, kv in zip(self.layers, kv_cache):
            h = dec.decode_forward(h, cur_pos, rope_idx, page_table, kv)
        return h

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

    def logits_to_host(self, logit_shards):
        """Gather the mesh-sharded vocab shards into a full [.., V] host tensor (prefill/debug only —
        NOT used on the measured token-out decode path)."""
        return ttnn.to_torch(logit_shards, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1))
