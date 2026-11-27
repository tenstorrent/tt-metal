# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reader of the Kimi-K3.1 FP64 CPU chain goldens (``KimiK31ChainGolden``): K3.1 layers 0-3 from prompt token ids,
per-row products.

The chain (``golden_producer.py`` next to this file, producer ``fp64_chain``) runs embed -> layer 0 (KDA, dense FFN) ->
layers 1 and 2 (KDA, MoE) -> layer 3 (DSA) in FP64 with the AttnRes reads, segment by segment, and is registered in the
long-context golden registry (``common/prefill/long_context_accuracy/fp64_golden.py``) under its id. Tests open it by id
only (``open_golden(id, reader=KimiK31ChainGolden)``); a missing, incomplete or other-producer golden fails. Its per-row
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

import json
from pathlib import Path

import torch

from models.demos.common.prefill.long_context_accuracy.fp64_golden import Fp64Golden, open_golden
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden_producer import (
    DSA_LAYER,
    KEYS,
    MOE_LAYERS,
    PRODUCER,
    PRODUCER_VERSION,
    SAMPLED,
    checkpoint_tensors,
)
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden_producer import (
    layer_weight_names as chain_weight_names,
)
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.kimi_k3_1.golden_producer import tensor_digest

K31_CHAIN_1M = "k31-l0-3-fp64-1m"  # layers 0-3 over the vLLM K3.1-Pro 1M prompt (1,044,480 rows)
MODEL = "Kimi-K3.1"


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
        in_state = [n for n in names if n in KEYS]
        in_segment = [n for n in names if n not in KEYS]
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
            parts.append({k: data[k][keep] for k in SAMPLED})
        return {k: torch.cat([p[k] for p in parts]) for k in SAMPLED}


def open_k31_chain(golden_id: str = K31_CHAIN_1M, root: Path | None = None) -> KimiK31ChainGolden:
    """The registered K3.1 FP64 chain ``golden_id``, checked to be ``fp64_chain`` version ``PRODUCER_VERSION`` of
    ``MODEL``. Raises the registry's ``GoldenError``s (missing, incomplete, malformed, mismatched)."""
    golden = open_golden(golden_id, root, reader=KimiK31ChainGolden)
    golden.require(identity={"model": MODEL})
    return golden


def layer_weight_names(checkpoint: Path, layer: int) -> list[str]:
    """The layer-local checkpoint names of the weights the chain digests as ``layers.<layer>``
    (``golden_producer.layer_weight_names``), for MoE layers ``MOE_LAYERS`` (whose KDA weights follow the checkpoint's
    config) or DSA layer ``DSA_LAYER``. Layer 0 (dense FFN) is not supported."""
    if layer not in (*MOE_LAYERS, DSA_LAYER):
        raise ValueError(f"layer_weights reads MoE layers {MOE_LAYERS} or DSA layer {DSA_LAYER}, not layer {layer}")
    config = (
        None
        if layer == DSA_LAYER
        else KDAConfig.from_model_config(json.loads((Path(checkpoint) / "config.json").read_text()))
    )
    return chain_weight_names(layer, config)


def layer_weights(checkpoint: Path, layer: int) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Layer ``layer``'s weights as the chain read them (``layer_weight_names``) in their checkpoint dtype, keyed by
    layer-local name, and their digest as ``{"layers.<layer>": sha256}`` (for
    ``Fp64Golden.require(weights_sha256=...)``)."""
    names = layer_weight_names(checkpoint, layer)
    tensors = checkpoint_tensors(checkpoint, layer, names)
    return tensors, {f"layers.{layer}": tensor_digest(tensors)}
