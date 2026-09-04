# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Static model-dimension constants for Llama-3.1-8B — the ``PrefillModelAdapter.model_config`` class.

The prefill engine's adapter contract requires a ``model_config`` **class** of plain constants
(``models/demos/common/prefill/adapter.py:115``). Three call sites read it, none of which can take an
instance or a device:

* ``models/demos/common/prefill/runners/runner_utils.py:41`` — ``FABRIC_PAYLOAD_SIZE``, read at
  ``set_fabric_config`` time, i.e. **before** any mesh is open;
* ``models/demos/common/prefill/runners/prefill_producer.py:542`` — ``NUM_KEY_VALUE_HEADS`` /
  ``HEAD_DIM`` / ``ROTARY_DIM``, read in the producer process, which never opens a device at all;
* ``models/demos/common/prefill/runners/migration_driver.py:281``.

**Why this is a second config class** (``DEC-101``). ``tt/model_config.py`` already owns
:class:`~models.demos.llama31_8b_d_p.tt.model_config.LlamaHFConfig`, and that stays the single source
of truth for every dimension a ``tt/`` module reads (``DEC-009``). This module is *not* a competing
definition: it is the constants view the engine needs, and
``tests/unit/test_prefill_adapter.py::test_model_dims_match_the_bundled_config`` asserts every field
here equals the bundled ``configs/Llama-3.1-8B-Instruct/config.json`` value that
``llama_hf_config()`` reads, so the two cannot drift.

**This module imports nothing.** That is load-bearing, not tidiness: it is imported at
``tt/runners/adapters/llama.py`` module scope, and the adapter is imported by the H2D producer and by
``models/demos/deepseek_v3_d_p/tests/conftest.py:33``, which instantiates *every* registered adapter
at collection time. See ``DEC-102``.
"""


class Llama31_8BConfig:
    """Llama-3.1-8B-Instruct dimensions, verbatim from ``configs/Llama-3.1-8B-Instruct/config.json``.

    Every value below is read from that file except the four marked *derived* / *convention*.
    """

    # --- core dimensions ---
    EMB_SIZE = 4096  # config.json: hidden_size
    # Convention, not a config value: every model in the tree sets FABRIC_PAYLOAD_SIZE = EMB_SIZE
    # (`models/demos/deepseek_v3_d_p/reference/gpt_oss_120b_config.py:18` and eight siblings).
    # It becomes FabricRouterConfig.max_packet_payload_size_bytes. DEC-103.
    FABRIC_PAYLOAD_SIZE = EMB_SIZE
    INTERMEDIATE_SIZE = 14336  # config.json: intermediate_size (dense SwiGLU; Llama-3.1 has no MoE)

    # --- attention ---
    NUM_ATTENTION_HEADS = 32  # config.json: num_attention_heads
    NUM_KEY_VALUE_HEADS = 8  # config.json: num_key_value_heads. == the TP width; see R-027.
    HEAD_DIM = 128  # derived: hidden_size // num_attention_heads (absent from config.json)
    # Derived: Llama-3.1 is FULL rotary, so rotary_dim == head_dim. The producer's packed-GQA
    # read-back reads this via getattr(mc, "ROTARY_DIM", HEAD_DIM) to build the HF -> Meta lane
    # permutation (`prefill_producer.py:544`); stated explicitly rather than left to that default.
    ROTARY_DIM = 128
    GQA_GROUP_SIZE = 4  # derived: num_attention_heads // num_key_value_heads

    # --- model ---
    NUM_LAYERS = 32  # config.json: num_hidden_layers
    VOCAB_SIZE = 128256  # config.json: vocab_size
    MAX_POSITION_EMBEDDINGS = 131072  # config.json: max_position_embeddings

    # --- norms / rope ---
    RMS_NORM_EPS = 1e-05  # config.json: rms_norm_eps
    ROPE_THETA = 500000.0  # config.json: rope_theta (top-level in the JSON; see R-014 / Appendix F.2)
    ROPE_TYPE = "llama3"  # config.json: rope_scaling.rope_type
    ROPE_SCALING_FACTOR = 8.0  # config.json: rope_scaling.factor
    ROPE_ORIG_CONTEXT_LEN = 8192  # config.json: rope_scaling.original_max_position_embeddings

    # --- MoE: none. Llama-3.1-8B is dense. ---
    # The engine's `default_gate_mode` and PREFILL_GATE_FALLBACK_MODE are MoE-router knobs and are
    # inert for this model (`03_OUTLINE.md` §3.21); no NUM_ROUTED_EXPERTS etc. are defined, so a
    # code path that expects them fails loudly here rather than reading a fabricated 0.
