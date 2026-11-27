# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reader of the Kimi-K3.1 FP64 CPU chain goldens (``KimiK31ChainGolden``): K3.1 layers 0-3 from prompt token ids,
per-row products.

The chain (producer ``fp64_chain``) runs embed -> layer 0 (KDA, dense FFN) -> layers 1 and 2 (KDA, MoE) -> layer 3 (DSA)
in FP64 with the AttnRes reads, segment by segment, and is registered in the long-context golden registry
(``common/prefill/long_context_accuracy/fp64_golden.py``) under its id. Tests open it by id only
(``open_golden(id, reader=KimiK31ChainGolden)``); a missing, incomplete or other-producer golden fails. Its per-row
products (manifest ``products`` of kind ``row``) are stored fp32 (keys FP64):

* MoE layers ``L`` = 1, 2: ``moe_input_L`` (the gate input, after ``post_attention_layernorm``), ``moe_output_L``
  (routed + shared, before the residual add), ``routing_ids_L`` (FP64 top-16, best first), ``routing_weights_L``,
  ``routing_margin_L`` (16th minus 17th choice score);
* DSA layer 3: ``dsa_input_3`` (the AttnRes read before ``input_layernorm``), ``topk_indices_3`` /
  ``topk_scores_3`` (FP64 indexer top-2048, best first, -1 past the causal keys), and the keys ``latent``, ``kpe``,
  ``index_key`` (held in the state files).

On sampled rows only (``sampled_row`` products, quadratic in context otherwise): ``sampled_rows`` (absolute rows) and
``attn_output_3`` (FP64 gated DSA attention output), ``KimiK31ChainGolden.read_sampled``.

Each segment ``i`` (rows ``[i * segment, (i + 1) * segment)``) is ``segment_<i>.pt`` (the row and sampled products)
and ``state_<i>.pt`` (``keys``: the DSA keys; ``kda``: the KDA states after the segment; no consumer reads those, so the
reader serves no segment-end product). Files are memory-mapped and only the requested rows are copied.

The chain's ``identity.weights_sha256`` holds one digest per layer's non-routed weights (``layers.<l>``) as the chain
read them. ``layer_weights`` reads the set of MoE layers 1 and 2 or DSA layer 3 from a checkpoint with its digest, for
``Fp64Golden.require(weights_sha256=...)``; no consumer digests layer 0 or the routed experts (``layers.<l>.experts``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from models.demos.common.prefill.long_context_accuracy.fp64_golden import Fp64Golden, open_golden

K31_CHAIN_1M = "k31-l0-3-fp64-1m"  # layers 0-3 over the vLLM K3.1-Pro 1M prompt (1,044,480 rows)
PRODUCER = "fp64_chain"
PRODUCER_VERSION = 2  # 2: buffered indexer blocks of 4096 keys (FP64 summation order of the indexer scores)
MODEL = "Kimi-K3.1"
STATE_ROW_PRODUCTS = ("latent", "kpe", "index_key")  # per-row products held in the state files, under "keys"
SAMPLED_PRODUCTS = ("sampled_rows", "attn_output_3")
MOE_LAYERS = (1, 2)  # the chain's KDA + MoE layers
DSA_LAYER = 3  # the chain's DSA layer; its FFN is not run, but its router is read (a cross-check), so digested
ATTN_RES_WEIGHTS = (
    "input_layernorm.weight",
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
    "post_attention_layernorm.weight",
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
)
ROUTER_WEIGHTS = ("block_sparse_moe.gate.weight", "block_sparse_moe.gate.e_score_correction_bias")
MOE_NON_ROUTED_WEIGHTS = tuple(
    f"block_sparse_moe.{n}.weight"
    for n in (
        "routed_expert_down_proj",
        "routed_expert_up_proj",
        "routed_expert_norm",
        "shared_experts.gate_proj",
        "shared_experts.up_proj",
        "shared_experts.down_proj",
    )
)
DSA_WEIGHTS = tuple(
    f"self_attn.{n}.weight"
    for n in (
        "q_a_proj",
        "q_a_layernorm",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_a_layernorm",
        "kv_b_proj",
        "g_proj",
        "o_proj",
        "indexer.wq_b",
        "indexer.wk",
        "indexer.k_norm",
        "indexer.weights_proj",
    )
) + ("self_attn.indexer.k_norm.bias",)


class KimiK31ChainGolden(Fp64Golden):
    """A registered K3.1 FP64 chain: ``fp64_chain`` version ``PRODUCER_VERSION`` (``open_golden`` requires it), of
    ``MODEL`` (``require(identity=...)``)."""

    producer = PRODUCER
    producer_version = PRODUCER_VERSION

    def segment_file(self, index: int) -> Path:
        return self.path / f"segment_{index:06d}.pt"

    def state_file(self, index: int) -> Path:
        return self.path / f"state_{index:06d}.pt"

    def _read_rows(self, names: list[str], start: int, end: int) -> dict[str, torch.Tensor]:
        segment = self.manifest["segment"]
        in_state = [n for n in names if n in STATE_ROW_PRODUCTS]
        in_segment = [n for n in names if n not in STATE_ROW_PRODUCTS]
        parts: dict[str, list[torch.Tensor]] = {n: [] for n in names}
        for index in range(start // segment, -(-end // segment)):
            lo = index * segment
            data = {}
            if in_segment:
                data.update(torch.load(self.segment_file(index), weights_only=True, mmap=True))
            if in_state:
                data.update(torch.load(self.state_file(index), weights_only=True, mmap=True)["keys"])
            for n in names:
                parts[n].append(data[n][max(start, lo) - lo : end - lo])
        return {n: torch.cat(v) for n, v in parts.items()}  # (the cat copies out of the mapped files)

    def read_sampled(self, end: int | None = None) -> dict[str, torch.Tensor]:
        """The sampled-row products below row ``end`` (default: all): ``sampled_rows`` (absolute rows, ascending,
        int64) and ``attn_output_3`` (FP64 ``[n, hidden]``)."""
        end = self.rows if end is None else end
        if not 0 < end <= self.rows:
            raise ValueError(f"end {end} is not inside the {self.rows} rows of {self.golden_id}")
        parts = []
        for index in range(-(-end // self.manifest["segment"])):
            data = torch.load(self.segment_file(index), weights_only=True, mmap=True)
            keep = data["sampled_rows"] < end
            parts.append({k: data[k][keep] for k in SAMPLED_PRODUCTS})
        return {k: torch.cat([p[k] for p in parts]) for k in SAMPLED_PRODUCTS}


def open_k31_chain(golden_id: str = K31_CHAIN_1M, root: Path | None = None) -> KimiK31ChainGolden:
    """The registered K3.1 FP64 chain ``golden_id``, checked to be ``fp64_chain`` version ``PRODUCER_VERSION`` of
    ``MODEL``. Raises the registry's ``GoldenError``s (missing, incomplete, malformed, mismatched)."""
    golden = open_golden(golden_id, root, reader=KimiK31ChainGolden)
    golden.require(identity={"model": MODEL})
    return golden


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    """Content identity of named tensors (names, dtypes, shapes, bytes), as the chain computes it."""
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].contiguous()
        h.update(json.dumps([name, str(t.dtype), list(t.shape)]).encode())
        h.update(memoryview(t.view(torch.uint8).numpy()))
    return h.hexdigest()


def layer_weight_names(checkpoint: Path, layer: int) -> list[str]:
    """The layer-local checkpoint names of the weights the chain digests as ``layers.<layer>``: ``ATTN_RES_WEIGHTS``
    and ``ROUTER_WEIGHTS``, plus the KDA weights (of the checkpoint's config) and ``MOE_NON_ROUTED_WEIGHTS`` on
    ``MOE_LAYERS``, or ``DSA_WEIGHTS`` on ``DSA_LAYER``. Layer 0 (dense FFN) is not supported."""
    if layer == DSA_LAYER:
        return list(ATTN_RES_WEIGHTS + ROUTER_WEIGHTS + DSA_WEIGHTS)
    if layer not in MOE_LAYERS:
        raise ValueError(f"layer_weights reads MoE layers {MOE_LAYERS} or DSA layer {DSA_LAYER}, not layer {layer}")
    from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
    from models.demos.deepseek_v3_d_p.reference.kda.weights import required_kda_weight_names

    config = KDAConfig.from_model_config(json.loads((Path(checkpoint) / "config.json").read_text()))
    kda = [f"self_attn.{n}" for n in required_kda_weight_names(config)]
    return [*ATTN_RES_WEIGHTS, *kda, *ROUTER_WEIGHTS, *MOE_NON_ROUTED_WEIGHTS]


def layer_weights(checkpoint: Path, layer: int) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Layer ``layer``'s weights as the chain read them (``layer_weight_names``) in their checkpoint dtype, keyed by
    layer-local name, and their digest as ``{"layers.<layer>": sha256}`` (for
    ``Fp64Golden.require(weights_sha256=...)``)."""
    checkpoint = Path(checkpoint)
    names = layer_weight_names(checkpoint, layer)
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    roots = [r for r in ("model.", "language_model.model.") if f"{r}layers.{layer}.{names[0]}" in weight_map]
    if not roots:
        raise ValueError(f"{checkpoint} has no layers.{layer}.{names[0]}")
    full = {f"{roots[0]}layers.{layer}.{n}": n for n in names}
    by_shard: dict[str, list[str]] = {}
    for key in full:
        by_shard.setdefault(weight_map[key], []).append(key)
    tensors = {}
    for shard, keys in by_shard.items():
        with safe_open(str(checkpoint / shard), framework="pt") as handle:
            tensors.update({full[k]: handle.get_tensor(k) for k in keys})
    return tensors, {f"layers.{layer}": tensor_digest(tensors)}
