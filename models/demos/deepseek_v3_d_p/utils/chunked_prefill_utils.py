# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for the unified chunked-prefill MLA test (test_mla.py::test_mla_chunked_prefill):
GPU-trace discovery/loading, multi-user iteration partitioning, and the CPU torch MLA reference.

Kept out of tt/mla/utils.py on purpose: these pull the reference model + safetensors, which should
not enter the production model import path.
"""

import time
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache

from models.common.utility_functions import hf_cache_layer_kv
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference


def discover_traces(root, num_users, variant_name=None):
    """Immediate subdirs of `root`, one per user (cycled if fewer than num_users). Assert mla_io/ + kv_cache/.

    `root` may hold both kimi and deepseek traces as sibling subdirs (e.g. kimi_*_sdpa_mla next to
    deepseek_*_sdpa_mla). When `variant_name` is given, select by whether the dir name contains 'kimi':
    a kimi variant keeps the kimi_* dirs, any other variant keeps the rest.
    """
    dirs = sorted(d for d in Path(root).iterdir() if d.is_dir())
    assert dirs, f"no trace subdirs under {root}"
    if variant_name is not None:
        want_kimi = "kimi" in variant_name.lower()
        dirs = [d for d in dirs if ("kimi" in d.name.lower()) == want_kimi]
        assert dirs, f"no {'kimi' if want_kimi else 'non-kimi'} trace subdirs under {root} (variant={variant_name})"
    for d in dirs:
        assert (d / "mla_io").is_dir(), f"trace dir {d} is missing mla_io/"
        assert (d / "kv_cache").is_dir(), f"trace dir {d} is missing kv_cache/"
    return [dirs[u % len(dirs)] for u in range(num_users)]


def single_trace(path, num_users):
    """Use `path` directly as ONE trace dir (the leaf holding mla_io/ + kv_cache/), shared across all
    users. For MLA_CHUNKED_TRACE_PATH, which points at a specific trace rather than the root of many."""
    d = Path(path)
    assert (d / "mla_io").is_dir(), f"trace dir {d} is missing mla_io/"
    assert (d / "kv_cache").is_dir(), f"trace dir {d} is missing kv_cache/"
    return [d for _ in range(num_users)]


def load_trace(d):
    """Return (mla_input [S,H], mla_output [S,H], kv_post [S,kvpe]) for layer 0, all bf16."""
    mi = load_file(d / "mla_io" / "mla_input_layer_0.safetensors")["mla_input_layer_0"]
    mo = load_file(d / "mla_io" / "mla_output_layer_0.safetensors")["mla_output_layer_0"]
    kv = load_file(d / "kv_cache" / "layer_0.safetensors")["kv_post_transform_layer_0"]
    return mi.to(torch.bfloat16), mo.to(torch.bfloat16), kv.to(torch.bfloat16)


def partition_iters(iters_isl, num_users):
    """Split iters_isl into num_users contiguous groups; the LAST user takes the remainder."""
    assert len(iters_isl) >= num_users, f"need >= {num_users} iters to split across {num_users} users"
    base = len(iters_isl) // num_users
    groups, idx = [], 0
    for u in range(num_users):
        n = base if u < num_users - 1 else len(iters_isl) - base * (num_users - 1)
        groups.append(list(iters_isl[idx : idx + n]))
        idx += n
    return groups


def cpu_mla_reference(config, weights, hidden_2d):
    """torch MLA forward over [S, H] hidden. Returns (output [S, H], kvpe [S, kvpe]) bf16 -- kvpe is
    the reference KV cache (Meta-style rope), for comparing the device cache directly. Host-attn logs."""
    mla_ref = (
        create_mla_reference(
            config=config,
            state_dict={"model.layers.0.self_attn." + k: v for k, v in weights.items()},
            layer_idx=0,
            module_path="model.layers.0.self_attn",
        )
        .eval()
        .to(torch.bfloat16)
    )
    pos = torch.arange(hidden_2d.shape[0], dtype=torch.long).unsqueeze(0)
    logger.warning(
        f"===== HOST ATTENTION START: torch MLA reference over {hidden_2d.shape[0]} tokens "
        f"(CPU chunked-flash, {config.num_attention_heads} heads) -- slow CPU phase ====="
    )
    t0 = time.perf_counter()
    ref_cache = DynamicCache()
    with torch.no_grad():
        out, _, ref_cache = mla_ref(
            hidden_states=hidden_2d.unsqueeze(0), position_ids=pos, past_key_value=ref_cache, use_cache=True
        )
    logger.warning(f"===== HOST ATTENTION END: torch reference done in {time.perf_counter() - t0:.1f}s =====")
    kvpe = hf_cache_layer_kv(ref_cache, 0)[0][0, 0]  # [S, kvpe], latent k_nope + roped k_pe (Meta basis)
    return out[0].to(torch.bfloat16), kvpe.to(torch.bfloat16)
