# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Length cap for the Qwen3-32B long-prefill optimizations (no imports, so every module can read it).

Every Qwen long-prefill change (context-parallel prefill, sequence-parallel residual, minimal_matmul QKV, direct /
local Q prep, ring SDPA program and compute configs, WO / FF2 fp32 accumulation, ring CCL overrides, multiple-of-8192
padding and its warmup lengths) applies only while the prefill's total context (cached prefix + new tokens) is
<= QWEN_OPT_MAX_SEQ_LEN, the longest length they were measured at. Longer contexts, and decode, take the production
(main) path unchanged. The HF rms_norm_eps (1e-6) applies only to prefills with no cached prefix within the cap; every
prefix-cached prefill and decode use main's eps.
"""

import copy

QWEN_OPT_MAX_SEQ_LEN = 40960


def qwen_opt_seq_len_ok(seq_len) -> bool:
    """True when a prefill of seq_len rows may take the Qwen long-prefill optimizations."""
    return int(seq_len) <= QWEN_OPT_MAX_SEQ_LEN


# The gate is on the TOTAL prefill context: cached prefix + new (padded) tokens, i.e. the KV extent after the prefill.
# The model (TtTransformer.forward, the generator's CP layer-major driver) records it once per prefill call in the
# model_config dict that attention, MLP, CCL, the norms and the config getters share; they read it here. Without a
# recorded value (direct callers such as op tests) the prefill length itself is the total.
PREFILL_TOTAL_LEN_KEY = "QWEN_PREFILL_TOTAL_LEN"
PREFILL_CACHED_LEN_KEY = "QWEN_PREFILL_CACHED_LEN"


def set_prefill_total_len(model_config, total_len, num_cached=0):
    """Record the total context (cached + new tokens) and the cached prefix length of the prefill about to run."""
    model_config[PREFILL_TOTAL_LEN_KEY] = int(total_len)
    model_config[PREFILL_CACHED_LEN_KEY] = int(num_cached)


def prefill_cached_len(model_config) -> int:
    """Cached prefix length of the running prefill (0 when none is recorded)."""
    return int(model_config.get(PREFILL_CACHED_LEN_KEY) or 0) if model_config is not None else 0


def prefill_total_len(model_config, seq_len) -> int:
    """Total context of the running prefill: the recorded value (never below seq_len), else seq_len."""
    total = model_config.get(PREFILL_TOTAL_LEN_KEY) if model_config is not None else None
    return int(seq_len) if total is None else max(int(seq_len), int(total))


def qwen_prefill_opt_ok(model_config, seq_len) -> bool:
    """True when the running prefill (seq_len new rows, total context from model_config) may take the Qwen
    long-prefill optimizations: total context <= QWEN_OPT_MAX_SEQ_LEN."""
    return qwen_opt_seq_len_ok(prefill_total_len(model_config, seq_len))


def attach_prefill_opt_eps(norm, eps):
    """Give an RMSNorm a prefill-only twin with epsilon eps (Qwen3's HF rms_norm_eps, 1e-6).

    The norm itself keeps its construction eps (the RMSNorm default, as on main), which decode and prefills above
    QWEN_OPT_MAX_SEQ_LEN use. The twin is a shallow copy: it shares the weight tensors (no extra DRAM, no weight cache
    file), and RMSNorm.update writes those tensors in place, so the two never drift apart."""
    twin = copy.copy(norm)
    twin.eps = eps
    norm.prefill_opt_norm = twin
    return norm


def prefill_norm(norm, seq_len, model_config=None):
    """The RMSNorm a prefill of seq_len rows uses: the eps twin (when attached) only for a prefill with no cached
    prefix and total context <= cap (both from model_config when recorded, else seq_len and no prefix); else norm
    (main's eps). Every prefix-cached prefill uses main's eps, so the prefix-cached (sp1) traces captured in warmup
    with a short prefix are valid at every total context."""
    twin = getattr(norm, "prefill_opt_norm", None)
    use_twin = twin is not None and prefill_cached_len(model_config) == 0 and qwen_prefill_opt_ok(model_config, seq_len)
    return twin if use_twin else norm


def prefill_norm_eps(norm, seq_len, model_config=None):
    """Epsilon of prefill_norm(norm, seq_len, model_config)."""
    return prefill_norm(norm, seq_len, model_config).eps
