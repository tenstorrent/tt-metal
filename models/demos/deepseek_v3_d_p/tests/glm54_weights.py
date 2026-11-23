# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 weights for tests until the official weights are published: two tiers, each with its provenance.

Tier 1 (accuracy): every weight seeded random, at a reduced config (``glm54_reduced_config``): GLM-5.3 widths, so
device modules run their production shapes, and 4 layers holding every GLM-5.4 layer role:

    L0 prelude (dense FFN, full indexer) | L1 loop (MoE, full) L2 loop (MoE, shared) | L3 coda (MoE, full)

The loop L1-L2 runs twice; the first coda layer owns an indexer, as L54 does at full scale. Layer weights come
from the GLM-5.3 generator (``glm_synthetic_weights``), drawn per global layer index, with one routed/shared
expert set reused by every MoE layer (each keeps its own router, norms and MLA, so a layer mix-up still shows) to
hold host RAM at one MoE layer (~19 GB). The same dicts feed ``glm54_stack_reference`` and
``TtPrefillTransformer``. Tier 1 checks wiring and conformance only (synthetic data does not show numeric
sensitivity).

Tier 2 (performance): every shared tensor is GLM-5.3's FP8 checkpoint, GLM-5.4 layer L (either loop pass) =
GLM-5.3 layer L; only the damp gates are synthetic (``reference.glm_5_4.synthetic_gates``, centred so beta1 ~0.2,
beta2 ~0.9). Device runs read those tensors from the GLM-5.3 TTNN cache through the runner, unchanged: its
directory is ``{cache}/glm_5_3_{arch}_{N}dev/...``. ``MLAPrefillAdapter.weight_cache_path`` names it after the
adapter, so a GLM-5.4 adapter must keep the ``glm_5_3`` name or override that path: otherwise it creates
``{cache}/glm_5_4_{arch}_{N}dev`` on Weka, finds every weight category missing, and the runner rebuilds and writes a
full cache there (``weight_categories``), breaking the read-only rule for the GLM-5.3 checkpoint and cache. Reference
checks load one layer with ``glm54_checkpoint_layer``. Nothing here writes to the checkpoint or the cache.

``tier2_sources`` resolves the checkpoint and cache the way the runner does (``$GLM53_HF_MODEL``, else
``$PREFILL_HF_MODEL``, else the adapter default; ``$PREFILL_TTNN_CACHE``, else the adapter default), and every tier-2
function reads its paths from it, so the recorded provenance is the source actually used.

``tier1_provenance`` / ``tier2_provenance`` return what a run must record to be reproducible: generator,
seeds and constants for synthetic tensors; checkpoint path, revision and file digests for real ones.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

from models.demos.deepseek_v3_d_p.reference.glm_5_4.synthetic_gates import (
    GATE_BETA_CENTER,
    GATE_LOGIT_STD,
    GATE_NORM_GAIN_STD,
    GLM54_SYNTHETIC_SEED,
    SYNTHETIC_GATES_VERSION,
    gate_seed,
    synthetic_gates,
)
from models.demos.deepseek_v3_d_p.reference.glm_5_4_config import GLM54Config, glm_5_4_hf_config
from models.demos.deepseek_v3_d_p.tests.glm_synthetic_weights import glm_random_transformer_state_dict
from models.demos.deepseek_v3_d_p.tt.runners.adapters.glm_5_3 import GLM53Adapter

# GLM-5.4 = GLM-5.3 + loop fields (A8): tier 2 reads GLM-5.3's checkpoint and TTNN cache.
GLM53_CHECKPOINT = GLM53Adapter.hf_model_default
GLM53_TTNN_CACHE = GLM53Adapter.ttnn_cache_default
# Every per-layer category of weight_categories (no embedding / lm_head / mtp).
LAYER_CATEGORIES = ("norms", "mla", "dense_ffn", "moe_gate", "shared_experts", "routed_experts")


class GLM54ReducedConfig(GLM54Config):
    """Model constants of the tier-1 reduced stack (``model_cfg`` for TtPrefillTransformer / TtPrefillBlock)."""

    NUM_LAYERS = 4
    NUM_DENSE_LAYERS = 1
    LOOP_FIRST_LAYER = 1
    LOOP_LAST_LAYER = 2
    INDEX_TOPK_PATTERN = "FFSF"


def glm54_reduced_config(max_seq_len: int):
    """GLM-5.4 hf config of the tier-1 reduced stack (see the module docstring), at ``max_seq_len``."""
    m = GLM54ReducedConfig
    c = glm_5_4_hf_config(max_seq_len)
    c.num_hidden_layers = m.NUM_LAYERS
    c.first_k_dense_replace = m.NUM_DENSE_LAYERS
    c.indexer_types = m.indexer_types()
    c.loop_first_layer, c.loop_last_layer = m.LOOP_FIRST_LAYER, m.LOOP_LAST_LAYER
    return c


@dataclass
class Glm54Weights:
    """One weight set: ``state_dict`` in TtPrefillTransformer form (``embed_weight``, ``layers``: one
    TtPrefillBlock dict per layer, loop layers once), ``gates`` (one dict per loop pass, as
    ``glm54_stack_reference`` takes) and the ``provenance`` to record with any result built on it."""

    state_dict: dict
    gates: list[dict]
    provenance: dict = field(default_factory=dict)


def _gate_provenance(seed: int, loop_count: int) -> dict:
    return {
        "generator": "reference.glm_5_4.synthetic_gates.synthetic_gate_weights",
        "version": SYNTHETIC_GATES_VERSION,
        "seed": seed,
        "gate_seeds": [gate_seed(seed, k) for k in range(1, loop_count + 1)],
        "beta_center": {k: GATE_BETA_CENTER[k] for k in range(1, loop_count + 1)},
        "logit_std": GATE_LOGIT_STD,
        "norm_gain_std": GATE_NORM_GAIN_STD,
        "width": "scalar (per token)",
    }


def tier1_provenance(seed: int = GLM54_SYNTHETIC_SEED) -> dict:
    """What determines the tier-1 weights' bytes."""
    m = GLM54ReducedConfig
    return {
        "tier": 1,
        "config": "tests.glm54_weights.glm54_reduced_config",
        "layers": f"{m.NUM_LAYERS} (dense < {m.NUM_DENSE_LAYERS}, loop {m.LOOP_FIRST_LAYER}..{m.LOOP_LAST_LAYER}, "
        f"indexer {m.INDEX_TOPK_PATTERN})",
        "layer_generator": "tests.glm_synthetic_weights.glm_random_transformer_state_dict(share_experts=True)",
        "seed": seed,
        "gates": _gate_provenance(seed, m.LOOP_COUNT),
    }


def glm54_tier1_weights(config, seed: int = GLM54_SYNTHETIC_SEED) -> Glm54Weights:
    """Seeded random weights for ``glm54_reduced_config``: embedding, 4 layers (MoE layers share one expert
    set), and the scalar synthetic gates. About 21 GB of host RAM."""
    if config.num_hidden_layers != GLM54ReducedConfig.NUM_LAYERS:
        raise ValueError(f"tier 1 is the {GLM54ReducedConfig.NUM_LAYERS}-layer reduced stack, got {config}")
    state_dict = glm_random_transformer_state_dict(
        config, GLM54ReducedConfig, config.num_hidden_layers, seed=seed, share_experts=True
    )
    gates = synthetic_gates(config.hidden_size, config.loop_count, seed=seed)
    return Glm54Weights(state_dict, gates, tier1_provenance(seed))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tier2_sources() -> tuple[Path, Path]:
    """(checkpoint, TTNN cache root) a tier-2 run reads, resolved as the GLM-5.3 runner resolves them
    (``GLM53Adapter.moe_block_nonrouted_weights`` / ``MLAPrefillAdapter.weight_cache_path``)."""
    checkpoint = os.environ.get(GLM53Adapter.env_var) or os.environ.get("PREFILL_HF_MODEL") or GLM53_CHECKPOINT
    cache = os.environ.get("PREFILL_TTNN_CACHE", GLM53_TTNN_CACHE)
    return Path(checkpoint), Path(cache)


def tier2_provenance(config, seed: int = GLM54_SYNTHETIC_SEED) -> dict:
    """What determines the tier-2 weights: the GLM-5.3 checkpoint in use (path, revision tag of its directory name,
    digests of its config and tensor index), the TTNN cache root device runs read, and the synthetic gates of
    ``config.loop_count`` passes."""
    ckpt, cache = tier2_sources()
    return {
        "tier": 2,
        "checkpoint": str(ckpt),
        "checkpoint_dir_tag": ckpt.name.rsplit("-", 1)[-1],
        "config_sha256": _sha256(ckpt / "config.json"),
        "index_sha256": _sha256(ckpt / "model.safetensors.index.json"),
        "ttnn_cache": str(cache),
        "layer_map": "GLM-5.4 layer L, any loop pass = GLM-5.3 layer L",
        "gates": _gate_provenance(seed, config.loop_count),
    }


def glm54_tier2_weights(config, seed: int = GLM54_SYNTHETIC_SEED) -> Glm54Weights:
    """Tier 2: the synthetic gates; ``state_dict`` is empty, the runner's cache-only form (shared tensors come from
    the GLM-5.3 TTNN cache on device, or ``glm54_checkpoint_layer`` for a reference check)."""
    gates = synthetic_gates(config.hidden_size, config.loop_count, seed=seed)
    return Glm54Weights({}, gates, tier2_provenance(config, seed))


def glm54_checkpoint_layer(config, layer_idx: int, *, categories=LAYER_CATEGORIES) -> dict:
    """GLM-5.4 layer ``layer_idx`` from the tier-2 checkpoint (``tier2_sources``) as one TtPrefillBlock dict (FP8
    dequantized to bf16), holding ``categories`` (the FFN categories that do not apply to the layer are dropped).
    Read only. ``config`` is the full-size GLM-5.4 hf config (``glm_5_4_hf_config``): its layer kinds must be the
    checkpoint's.

    A MoE layer with routed experts is ~19 GB of host RAM and minutes of reading; pass fewer categories when the
    check does not need them. A ``shared``-indexer layer's MLA dict carries no indexer weights."""
    from models.demos.deepseek_v3_d_p.tt.runners.weight_categories import load_trunk_state_dict

    if config.num_hidden_layers != GLM54Config.NUM_LAYERS:
        raise ValueError("glm54_checkpoint_layer needs the full-size config (the checkpoint's layer kinds)")
    moe = layer_idx >= config.first_k_dense_replace
    skip = {"dense_ffn"} if moe else {"moe_gate", "shared_experts", "routed_experts"}
    sd = load_trunk_state_dict(
        tier2_sources()[0],
        config,
        GLM54Config,
        missing={c: [layer_idx] for c in categories if c not in skip},
        first_layer_idx=layer_idx,
        num_layers=1,
    )
    return sd["layers"][0]


__all__ = [
    "GLM53_CHECKPOINT",
    "GLM53_TTNN_CACHE",
    "GLM54ReducedConfig",
    "Glm54Weights",
    "LAYER_CATEGORIES",
    "glm54_checkpoint_layer",
    "glm54_reduced_config",
    "glm54_tier1_weights",
    "glm54_tier2_weights",
    "tier1_provenance",
    "tier2_provenance",
    "tier2_sources",
]
