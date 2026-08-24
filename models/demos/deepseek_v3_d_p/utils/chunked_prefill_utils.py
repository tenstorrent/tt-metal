# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for the unified chunked-prefill MLA test (test_mla.py::test_mla_chunked_prefill):
GPU-trace discovery/loading, multi-user iteration partitioning, and the CPU torch MLA reference.

Kept out of tt/mla/utils.py on purpose: these pull the reference model + safetensors, which should
not enter the production model import path.
"""

import hashlib
import os
import time
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import load_file
from transformers.cache_utils import DynamicCache

from models.common.utility_functions import hf_cache_layer_kv
from models.demos.deepseek_v3_d_p.reference.mla_reference import create_mla_reference

# On-disk cache for the CPU torch MLA reference. The reference is quadratic in sequence length and
# host-bound: measured on a 16-core EPYC (AVX-512, no AMX) it is ~4.4 s at 3840 tokens but ~7 min at
# the 56320-token production depth, and it was recomputed on every single run. run_model already
# caches its single-shot reference (test_mla.py, variant.mla_ref_cache_env); this is the chunked
# equivalent, so depth verification and config bisects cost device time only after the first pass.
_REF_CACHE_ENV = "MLA_CHUNKED_REF_CACHE"
_REF_CACHE_DEFAULT = "/tmp/mla_chunked_ref_cache"


def _hash_tensor(h, name: str, t: torch.Tensor) -> None:
    h.update(name.encode())
    h.update(f"{tuple(t.shape)}{t.dtype}".encode())
    h.update(t.detach().contiguous().view(torch.uint8).numpy().tobytes())


def _ref_cache_key(config, weights, hidden_2d) -> str:
    """Content hash of everything that changes the reference output: weights, INPUT, and the config
    fields that alter the math.

    All three are load-bearing, not belt-and-braces:
      * weights -- the random_weights fixture is seeded, so two configs can yield identically-shaped
        tensors; a dtype/config change that moves the reference must invalidate the entry.
      * hidden_2d -- the chunked driver seeds hidden per user (``torch.manual_seed(42 + u)``), so in a
        multi-user run two users can share the same total_len while having different inputs. Keying on
        length alone would hand user 1 user 0's reference and silently corrupt its PCC.
      * config -- num_attention_heads changes the absorbed-Q head split even at identical weight
        shapes; the NoPE / output-gate flags and rope_scaling change the math outright.

    Hashing the full tensors costs ~2 s at the 56320-token production shape, against a ~7 min
    reference. Content-addressing also makes the key collision-free across variants, so one shared
    cache dir is safe.
    """
    h = hashlib.sha1()
    for field in ("num_attention_heads", "rms_norm_eps", "mla_use_nope", "mla_use_output_gate", "rope_scaling"):
        h.update(f"{field}={getattr(config, field, None)!r}".encode())
    _hash_tensor(h, "hidden", hidden_2d)
    for name in sorted(weights):
        _hash_tensor(h, name, weights[name])
    return h.hexdigest()[:16]


def resolve_traces(paths, num_users):
    """One trace dir per user, cycled if there are fewer dirs than users. `paths` is either the
    variant's own mla_trace_defaults or a single explicit override."""
    assert paths, "no trace dirs given"
    dirs = [Path(p) for p in paths]
    for d in dirs:
        assert d.is_dir(), f"trace dir {d} does not exist"
        assert (d / "mla_io").is_dir(), f"trace dir {d} is missing mla_io/"
        assert (d / "kv_cache").is_dir(), f"trace dir {d} is missing kv_cache/"
    return [dirs[u % len(dirs)] for u in range(num_users)]


def load_trace(d):
    """Return (mla_input [S,H], mla_output [S,H], kv_post [S,kvpe]) for the traced layer, all bf16.

    The layer index comes off the filenames rather than being assumed 0: on a hybrid model it cannot
    be 0 (Kimi-K3 traces layer 3, its first full-attention layer).
    """
    inputs = sorted((d / "mla_io").glob("mla_input_layer_*.safetensors"))
    assert len(inputs) == 1, f"trace dir {d}: expected exactly one traced MLA layer, found {len(inputs)}"
    layer = inputs[0].stem[len("mla_input_layer_") :]
    mi = load_file(inputs[0])[f"mla_input_layer_{layer}"]
    mo = load_file(d / "mla_io" / f"mla_output_layer_{layer}.safetensors")[f"mla_output_layer_{layer}"]
    kv = load_file(d / "kv_cache" / f"layer_{layer}.safetensors")[f"kv_post_transform_layer_{layer}"]
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
    the reference KV cache (Meta-style rope), for comparing the device cache directly. Host-attn logs.

    Disk-cached on a content hash of (weights, seq_len, math-affecting config fields) -- see
    _ref_cache_key. A cache miss behaves exactly as before, so this is transparent to the
    deepseek_v3 / kimi_k2_6 chunked tests that share this helper. Writes are suppressed under CI
    (mirroring run_model's reference cache) so CI runners don't accumulate multi-GB of .pt files;
    reads stay enabled either way.
    """
    seq_len = hidden_2d.shape[0]
    cache_dir = Path(os.environ.get(_REF_CACHE_ENV, _REF_CACHE_DEFAULT))
    cache_path = cache_dir / f"chunked_ref_seq{seq_len}_{_ref_cache_key(config, weights, hidden_2d)}.pt"
    if cache_path.exists():
        try:
            blob = torch.load(cache_path)
            logger.info(f"CPU MLA reference: cache HIT {cache_path} (skipping the host-attention phase)")
            return blob["out"], blob["kvpe"]
        except Exception as e:  # corrupt/partial file -> fall through and recompute
            logger.warning(f"CPU MLA reference: cache at {cache_path} unreadable ({e}); recomputing")

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
    out_bf16, kvpe_bf16 = out[0].to(torch.bfloat16), kvpe.to(torch.bfloat16)

    if os.environ.get("CI") == "true" or os.environ.get("TT_GH_CI_INFRA"):
        logger.debug("CPU MLA reference: CI env, not writing the reference cache")
    else:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Write-then-rename so a killed run can't leave a half-written file that later reads trust.
            tmp = cache_path.with_suffix(".pt.tmp")
            torch.save({"out": out_bf16, "kvpe": kvpe_bf16}, tmp)
            tmp.replace(cache_path)
            logger.info(f"CPU MLA reference: cached to {cache_path}")
        except Exception as e:
            logger.warning(f"CPU MLA reference: could not write cache to {cache_path} ({e})")

    return out_bf16, kvpe_bf16
