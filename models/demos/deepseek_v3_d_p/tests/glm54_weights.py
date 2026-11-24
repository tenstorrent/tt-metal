# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.4 test weights until the official weights are published: tier 1, every weight seeded random, with its
provenance.

Tier 1 is a reduced config (``glm54_reduced_config``): GLM-5.3 widths, so device modules run their production
shapes, and 4 layers holding every GLM-5.4 layer role:

    L0 prelude (dense FFN, full indexer) | L1 loop (MoE, full) L2 loop (MoE, shared) | L3 coda (MoE, full)

The loop L1-L2 runs twice; the first coda layer owns an indexer, as L54 does at full scale. Layer weights come
from the GLM-5.3 generator (``glm_synthetic_weights``), drawn per global layer index, with one routed/shared
expert set reused by every MoE layer (each keeps its own router, norms and MLA, so a layer mix-up still shows) to
hold host RAM at one MoE layer (~19 GB). The same dicts feed ``glm54_stack_reference`` and
``TtPrefillTransformer``. Tier 1 checks wiring and conformance only (synthetic data does not show numeric
sensitivity).

Performance runs (tier 2) take every shared tensor from GLM-5.3's checkpoint and TTNN cache through the runner's
``glm_5_4`` adapter (``tt/runners/adapters/glm_5_4.py``) with the synthetic damp gates; nothing here serves them.
"""

from dataclasses import dataclass, field

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
    """Seeded random weights for ``glm54_reduced_config``: embedding, 4 layers (MoE layers share one expert set),
    and the scalar synthetic gates. About 23 GB of host RAM, generated in about 80 s."""
    if config.num_hidden_layers != GLM54ReducedConfig.NUM_LAYERS:
        raise ValueError(f"tier 1 is the {GLM54ReducedConfig.NUM_LAYERS}-layer reduced stack, got {config}")
    state_dict = glm_random_transformer_state_dict(
        config, GLM54ReducedConfig, config.num_hidden_layers, seed=seed, share_experts=True
    )
    gates = synthetic_gates(config.hidden_size, config.loop_count, seed=seed)
    return Glm54Weights(state_dict, gates, tier1_provenance(seed))


__all__ = [
    "GLM54ReducedConfig",
    "Glm54Weights",
    "glm54_reduced_config",
    "glm54_tier1_weights",
    "tier1_provenance",
]
