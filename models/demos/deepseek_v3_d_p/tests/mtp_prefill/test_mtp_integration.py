# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage 3: cross-check one GLM-5.2 MTP module against a vLLM golden trace (issue #53533).

``test_mtp.py`` gates the module against a CPU reference **we composed ourselves**. That proves the
device matches our understanding of MTP; it cannot prove our understanding is right. This file is
the independent check: a trace captured from vLLM, which is the code path GLM-5.2 actually runs in
production (vLLM maps ``GlmMoeDsaForCausalLM`` onto its DeepSeek-V3.2 implementation, so MTP goes
through ``deepseek_mtp.py``).

**Skipped until a trace exists.** The trace does not exist yet, and this file is deliberately
written *before* it: the acceptance criteria below are stated up front so a disagreement is
adjudicated by a rule agreed in advance, not by whichever side is more convenient to change.

That ordering is not pedantry. The DFlash golden trace (see ``tt/dflash_prefill/``) disagreed with
our K by ~0.05 PCC and the trace turned out to be wrong on two independent producer-side axes — a
bf16 cast of the ``rotary_emb.inv_freq`` buffer, and a config loaded by a transformers version that
silently defaulted ``rope_theta`` to 1e4. Weeks went into establishing which side was correct
because nobody had written down, beforehand, what the trace was allowed to be trusted about.

Acceptance criteria, in order. **Do not skip step 1.**

1. **Gate the trace against the CPU reference, on the trace's own weights, before comparing any
   device tensor to it.** ``glm_mtp_module_reference`` and vLLM are two independent implementations
   of the same published math; if they disagree, the disagreement is between *those two*, and
   resolving it does not involve Tenstorrent hardware at all. Run this on host, with the trace's
   real layer-78 weights, and require >= ``TRACE_VS_REFERENCE_PCC`` on ``x`` (the fused projection).
   ``x`` is the discriminating tensor: it is the only new math, and unlike the block output it is
   not confounded by MoE routing.
2. Only if step 1 passes, compare the device module to the trace at the thresholds in ``test_mtp.py``.
3. If step 1 fails, the trace is a hypothesis, not a golden. Before changing anything on our side,
   check the trace's own provenance: the transformers version that loaded the config, the dtype of
   ``rotary_emb.inv_freq`` at capture, whether ``positions`` was passed (position 0's embedding must
   be zeroed), and the concat order. All four are recorded in ``TRACE_MANIFEST_KEYS`` for exactly
   this reason.

Required trace contents (a single ``torch.save`` dict):

===========================  ======================================================================
key                          value
===========================  ======================================================================
``embed``                    ``[1, seq, 6144]`` bf16 — ``embed_tokens(t_{p+1})``, ALREADY shifted,
                             with row 0 zeroed if ``positions`` starts at 0
``hidden``                   ``[1, seq, 6144]`` bf16 — trunk output taken AFTER ``model.norm``
``positions``                ``[seq]`` int — absolute positions, so a mid-slab chunk is expressible
``x``                        ``[1, seq, 6144]`` — ``eh_proj(cat[enorm(embed), hnorm(hidden)])``
``out``                      ``[1, seq, 6144]`` — layer-78 output, BEFORE ``shared_head.norm``
``out_head_normed``          ``[1, seq, 6144]`` — after ``shared_head.norm``
``manifest``                 dict with every key in ``TRACE_MANIFEST_KEYS``
===========================  ======================================================================

Both ``out`` forms are required: which one feeds level k+1's ``hnorm`` is still open, and a trace
that carries only one of them cannot settle it.
"""

from __future__ import annotations

import os

import pytest
import torch

TRACE_ENV = "GLM52_MTP_TRACE"

# Step 1's bar. Two independent fp32 implementations of two RMSNorms and one 6144x12288 matmul over
# bf16 inputs should agree far better than any device threshold; anything below this is a real
# semantic disagreement (concat order, which norm, pre- vs post-model.norm hidden, position-0
# zeroing), not accumulated arithmetic noise.
TRACE_VS_REFERENCE_PCC = 0.9999

# Provenance the trace must carry. Every entry is here because its absence has already cost time on
# a prior golden trace, or because it is a documented trap in this feature.
TRACE_MANIFEST_KEYS = (
    "transformers_version",  # a loader too old for the checkpoint's rope schema silently defaults it
    "vllm_version",
    "checkpoint_path",
    "checkpoint_revision",  # pin it; "the dequantized dir" is not a provenance statement
    "torch_dtype",  # a model-wide .to(bf16) also casts non-persistent rope buffers
    "inv_freq_dtype",  # ... so record the buffer's dtype at capture, not just the model's
    "rope_theta",  # read back from the LOADED config object, not from config.json
    "concat_order",  # expected "embed_first"
    "hidden_is_post_norm",  # expected True
    "position_zero_embedding_zeroed",  # expected True
    "mtp_layer_idx",  # expected 78
    "num_levels",  # 1 for this trace
)


def _trace_path() -> str | None:
    path = os.environ.get(TRACE_ENV)
    return path if path and os.path.exists(path) else None


requires_trace = pytest.mark.skipif(
    _trace_path() is None,
    reason=(
        f"no vLLM MTP golden trace (set {TRACE_ENV} to a torch.save dict). Stage 3 of #53533; "
        "the acceptance criteria are recorded in this module's docstring, on purpose, before the "
        "trace exists."
    ),
)


def load_mtp_trace(path: str) -> dict:
    """Load a trace and fail loudly on anything that would make a later PCC number unattributable."""
    trace = torch.load(path, map_location="cpu")

    missing = [k for k in ("embed", "hidden", "x", "out", "out_head_normed", "manifest") if k not in trace]
    assert not missing, f"trace {path} is missing {missing}; see this module's docstring for the required contents"

    manifest = trace["manifest"]
    missing_meta = [k for k in TRACE_MANIFEST_KEYS if k not in manifest]
    assert not missing_meta, (
        f"trace {path} carries no {missing_meta} in its manifest. These are not bookkeeping: each one "
        "is a way a golden trace has silently been wrong before. Regenerate with them recorded rather "
        "than assuming their values."
    )

    # Semantics we can check without running anything.
    assert manifest["concat_order"] == "embed_first", (
        f"trace concat_order={manifest['concat_order']!r}; vLLM deepseek_mtp.py, vLLM glm4_moe_mtp.py, "
        "SGLang glm4_moe_nextn.py and the checkpoint's own eh_proj column statistics all say embedding "
        "first. A trace saying otherwise is describing a different model."
    )
    assert manifest["hidden_is_post_norm"], (
        "trace hidden was captured BEFORE model.norm. Level 1's hnorm consumes the post-final-norm "
        "trunk hidden (vLLM: 'Recycle the post-final-norm hidden into the next draft step')."
    )
    return trace


@requires_trace
def test_trace_agrees_with_cpu_reference():
    """Step 1 — host only, no device. Gates the TRACE, not the hardware.

    Deliberately runs before any device test in this file: if our CPU reference and vLLM disagree
    about ``x``, that is a disagreement between two host implementations of published math, and no
    Tenstorrent measurement can adjudicate it.
    """
    pytest.skip("Stage 3: implement once a trace exists — see this module's docstring for the ordering.")


@requires_trace
def test_device_module_matches_trace():
    """Step 2 — the device module vs the trace, at ``test_mtp.py``'s thresholds.

    Runs only if :func:`test_trace_agrees_with_cpu_reference` passed on the same trace.
    """
    pytest.skip("Stage 3: blocked on step 1 and on a trace.")
