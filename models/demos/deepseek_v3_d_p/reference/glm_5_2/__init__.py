# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-5.2 CPU reference helpers.

Today this holds only the Multi-Token-Prediction (MTP) reference. Everything GLM-5.2 shares with
GLM-5.1 (decoder layer, DSA MLA, MoE) is reused from ``reference.glm_5_1`` rather than duplicated —
the two are geometrically identical; 5.2's deltas are rope_theta, the indexer-reuse map, and MTP.

There is deliberately no vendored upstream modeling file here (contrast ``reference/dflash_prefill``,
a verbatim copy of z-lab's ``dflash.py``): GLM-5.2 ships MTP *weights* with no MTP *code*. The HF
repo has no ``.py``, and transformers states in its own ``glm_moe_dsa`` docs that "The implementation
in transformers does not include an MTP layer". The semantics below are taken from the serving
engines that do implement it — vLLM ``deepseek_mtp.py`` (which is the code path GLM-5.2 actually
runs, since vLLM maps ``GlmMoeDsaForCausalLM`` onto its DeepSeek-V3.2 implementation) and SGLang
``glm4_moe_nextn.py``. See issue #53533.
"""

from models.demos.deepseek_v3_d_p.reference.glm_5_2.mtp import (
    fused_mtp_reference,
    glm_mtp_module_reference,
)

__all__ = ["fused_mtp_reference", "glm_mtp_module_reference"]
