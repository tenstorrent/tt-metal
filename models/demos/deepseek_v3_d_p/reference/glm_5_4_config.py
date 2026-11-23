# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
GLM 5.4 Model Configuration.

GLM-5.4 is GLM-5.3 with layers 27-54 (0-indexed 26-53) run twice with shared weights (a "loop"), a damp gate
after each loop pass, and a two-bank sparse MLA in loop pass 2. Source: https://z.ai/blog/glm-5.4 (2026-11-23)
and its architecture diagram. No official config, weights or modeling code are published yet, so every
GLM-5.4-specific value here is an assumption (A1-A9 below), each bound to the config attributes that carry it,
so the list can be checked field by field once the official config.json appears.

Everything not listed is inherited unchanged from GLM-5.3.
"""

from dataclasses import dataclass

from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config, glm_5_3_hf_config


class GLM54Config(GLM53Config):
    """GLM 5.4 model dimensions: GLM-5.3 plus the loop fields."""

    # Loop region: layers LOOP_FIRST_LAYER..LOOP_LAST_LAYER (0-indexed, inclusive) run LOOP_COUNT times with
    # shared weights. Prelude = layers before it, coda = layers after it.
    LOOP_FIRST_LAYER = 26
    LOOP_LAST_LAYER = 53
    LOOP_COUNT = 2

    # Loop pass >= 2 attention: bank B is that pass's own KVPE over positions p-WINDOW+1..p (A3).
    LOOP_WINDOW_SIZE = 128
    # Storage encoding of the bank-B window, an MlaKvCacheFormat value (A2). Default as the GLM-5.3 KVPE cache.
    LOOP_WINDOW_FORMAT = "scaled_fp8"

    # Damp gate after each loop pass (A1). One gate per pass, each with its own RMSNorm gain, Linear weight and
    # bias; its width (1 = scalar per token, EMB_SIZE = per channel) is read from the Linear weight.
    LOOP_GATE_HAS_BIAS = True
    LOOP_GATE_WEIGHT_DTYPE = "bfloat16"

    @classmethod
    def loop_layers(cls) -> range:
        return range(cls.LOOP_FIRST_LAYER, cls.LOOP_LAST_LAYER + 1)


def glm_5_4_hf_config(max_seq: int = 8192):
    """HF-attribute-style config: ``glm_5_3_hf_config`` plus the GLM-5.4 loop fields."""
    config = glm_5_3_hf_config(max_seq)
    config.loop_first_layer = GLM54Config.LOOP_FIRST_LAYER
    config.loop_last_layer = GLM54Config.LOOP_LAST_LAYER
    config.loop_count = GLM54Config.LOOP_COUNT
    config.loop_window_size = GLM54Config.LOOP_WINDOW_SIZE
    config.loop_window_format = GLM54Config.LOOP_WINDOW_FORMAT
    config.loop_gate_has_bias = GLM54Config.LOOP_GATE_HAS_BIAS
    config.loop_gate_weight_dtype = GLM54Config.LOOP_GATE_WEIGHT_DTYPE
    return config


@dataclass(frozen=True)
class Assumption:
    """One GLM-5.4 assumption (user decision 2026-11-23) and the hf-config attributes it binds.

    ``fields`` maps an attribute of ``glm_5_4_hf_config()`` to the value the assumption requires, written as a
    literal (not read from GLM54Config) so a drift of the config away from the assumption is caught; an empty map
    means the assumption is structural (fixed in the reference code, named in ``code``), not a config value.
    """

    id: str
    statement: str
    fields: dict
    code: str = ""


GLM54_ASSUMPTIONS: tuple[Assumption, ...] = (
    Assumption(
        "A1a",
        "Gate width comes from the gate Linear weight: [1, hidden] is a scalar per token, [hidden, hidden] per channel.",
        {},
        "glm_5_4.ops.damp_gate_reference",
    ),
    Assumption(
        "A1b",
        "Each loop pass has its own gate weights (norm gain, Linear weight, bias).",
        {"loop_count": 2},
        "glm_5_4.loop.glm54_stack_reference (one gate per pass)",
    ),
    Assumption(
        "A1c",
        "The gate reads the pass input: x_k = x_{k-1} + beta_k * (h_k - x_{k-1}), "
        "beta_k = sigmoid(Linear(RMSNorm(x_{k-1}))).",
        {},
        "glm_5_4.ops.damp_gate_reference",
    ),
    Assumption("A1d", "The gate Linear has a bias.", {"loop_gate_has_bias": True}),
    Assumption(
        "A1e",
        "The gate RMSNorm has a learnable gain and uses rms_norm_eps (implementer default, not in the blog).",
        {"rms_norm_eps": 1e-5},
        "glm_5_4.ops.damp_gate_reference",
    ),
    Assumption(
        "A2",
        "Pass-2 KV is the same 576-wide latent from the shared layer weights; the window storage format is "
        "configurable, default scaled FP8 as the KVPE cache.",
        {
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "loop_window_format": "scaled_fp8",
        },
    ),
    Assumption("A3", "The pass-2 window is positions p-127..p (includes p).", {"loop_window_size": 128}),
    Assumption(
        "A4",
        "Pass 2 uses the pass-1 RoPE positions; no iteration embedding.",
        {"rope_theta": 8e6},
        "glm_5_4.loop.LoopPass2MLAReference (same absolute positions as pass 1)",
    ),
    Assumption(
        "A5",
        "Pass-2 layer L attends bank A through pass-1 layer L's top-k (7 sets for the 28 loop layers); short "
        "prompts carry 0xFFFFFFFF sentinels.",
        {"index_topk": 2048},
        "glm_5_4.loop.glm54_stack_reference",
    ),
    Assumption(
        "A6",
        "One softmax over both banks, scale qk_head_dim**-0.5 = 1/16, no sink or bias; a position in both banks "
        "is kept as two keys.",
        {"qk_nope_head_dim": 192, "qk_rope_head_dim": 64},
        "glm_5_4.ops.two_bank_sparse_mla",
    ),
    Assumption(
        "A7", "MTP as GLM-5.3, fed by the coda output.", {"num_nextn_predict_layers": 1}, "not in this reference"
    ),
    Assumption(
        "A8",
        "Config = GLM-5.3 + loop fields (layers 26..53, loop count 2, window 128, window format, gate weights); "
        "FP8 128x128 weights as GLM-5.3, bf16 gates.",
        {
            "num_hidden_layers": 78,
            "loop_first_layer": 26,
            "loop_last_layer": 53,
            "loop_count": 2,
            "loop_window_size": 128,
            "loop_gate_weight_dtype": "bfloat16",
        },
    ),
    Assumption("A9", "Text only; vision deferred until its architecture is published.", {}, "not in this reference"),
)


def check_assumptions(config) -> list[str]:
    """Mismatches between ``config`` and the assumptions' bound fields, as readable strings (empty = all hold).

    Run it on the official GLM-5.4 config (as an attribute namespace, mapping our field names onto its names)
    to see which assumptions the published model breaks.
    """
    mismatches = []
    for a in GLM54_ASSUMPTIONS:
        for name, expected in a.fields.items():
            actual = getattr(config, name, "<missing>")
            if actual != expected:
                mismatches.append(f"{a.id}: {name} = {actual!r}, assumed {expected!r} ({a.statement})")
    return mismatches


__all__ = ["GLM54Config", "glm_5_4_hf_config", "GLM54_ASSUMPTIONS", "Assumption", "check_assumptions"]
