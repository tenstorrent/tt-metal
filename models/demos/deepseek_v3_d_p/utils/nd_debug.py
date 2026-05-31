# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Non-determinism debug instrumentation (branch kgrujcic/deepseek_nd).

Purpose: localize the source of cross-run / cross-iteration non-determinism in
the DeepSeek-V3 prefill transformer. Everything here is pure host-side
*measurement* of tensors already produced by the forward pass — it never
changes device compute.

Key methodology note
--------------------
The test samples the first token with a Gumbel-softmax trick at temperature>0
(`tt_prefill_transformer._sample_token`). That draws from torch's *global* RNG,
which advances every iteration, so the *sampled token* can differ across
iterations even when the model logits are bit-identical. Therefore we fingerprint
the **logits** (a deterministic function of the model) and the **per-layer hidden
states**, NOT the sampled token. The sampled token is logged only for reference.

Fingerprint
-----------
For each tensor we record a bit-exact SHA1 of its float32 bytes plus cheap
float64 reductions (L2 norm, sum, count of non-finite). The SHA detects ANY bit
difference; the norm/sum quantify the magnitude of drift. We keep only the small
fingerprint dict per iteration (not the multi-GB tensors), then diff consecutive
iterations to report the FIRST stage that diverges — that stage's input was
identical (previous stage matched) so the divergence is born there.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import torch
from loguru import logger


def _fp_tensor(t: torch.Tensor) -> dict:
    """Bit-exact + statistical fingerprint of a host tensor."""
    t32 = t.detach().to(torch.float32).contiguous()
    sha = hashlib.sha1(t32.numpy().tobytes()).hexdigest()[:16]
    tf = t32.to(torch.float64).flatten()
    finite = torch.isfinite(tf)
    n_nonfinite = int((~finite).sum())
    tv = torch.where(finite, tf, torch.zeros_like(tf))
    return {
        "sha": sha,
        "norm": float(torch.sqrt((tv * tv).sum())),
        "sum": float(tv.sum()),
        "n_nonfinite": n_nonfinite,
        "shape": tuple(t32.shape),
    }


def _ordered_keys(intermediates: dict, num_layers: int) -> list[str]:
    keys = ["embed"] + [f"layer_{i}" for i in range(num_layers)] + ["norm", "lm_head", "logits"]
    return [k for k in keys if isinstance(intermediates.get(k), torch.Tensor)]


def nd_fingerprint(intermediates: Optional[dict], num_layers: int) -> Optional[dict]:
    # Defensive: a measurement probe must NEVER fail the test. Any error here is
    # swallowed and reported as a sentinel so the run continues.
    if intermediates is None:
        return None
    try:
        fp = {}
        for k in _ordered_keys(intermediates, num_layers):
            fp[k] = _fp_tensor(intermediates[k])
        # Extra detail on logits: argmax + top-2 margin (how close a token flip is).
        logits = intermediates.get("logits")
        if isinstance(logits, torch.Tensor) and logits.numel() > 0:
            flat = logits.detach().float().flatten()
            top2 = torch.topk(flat, k=min(2, flat.numel()))
            fp["__logits_argmax__"] = int(top2.indices[0].item())
            gap = float(top2.values[0].item() - top2.values[1].item()) if flat.numel() > 1 else float("inf")
            fp["__logits_gap__"] = gap
        else:
            fp["__logits_argmax__"] = -1
            fp["__logits_gap__"] = float("nan")
        fp["__order__"] = _ordered_keys(intermediates, num_layers)
        return fp
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[NDPROBE] nd_fingerprint failed (non-fatal): {exc!r}")
        return None


def nd_fp_shard0(tt_tensor) -> Optional[dict]:
    """Fingerprint device-0 shard of a ttnn tensor (cheap, no full mesh gather).

    Sufficient to detect run-to-run / iter-to-iter divergence: if device 0's
    shard differs, the op is non-deterministic. Returns None on any failure.
    """
    try:
        import ttnn

        shards = ttnn.get_device_tensors(tt_tensor)
        if not shards:
            return None
        host = ttnn.to_torch(shards[0])
        return _fp_tensor(host)
    except Exception as exc:  # noqa: BLE001
        return {"sha": "ERR", "norm": float("nan"), "sum": float("nan"), "n_nonfinite": -1, "shape": str(exc)[:40]}


def nd_moe_log(layer_idx: int, it: int, stage: str, tt_tensor) -> None:
    """Log a per-(layer,iter,stage) device-0-shard fingerprint of a MoE tensor."""
    fp = nd_fp_shard0(tt_tensor)
    if fp is None:
        return
    logger.info(
        f"[NDPROBE-MOE] layer={layer_idx} iter={it} stage={stage:>16} "
        f"sha={fp['sha']} norm={fp['norm']:.6f} sum={fp['sum']:.6e} "
        f"nonfinite={fp['n_nonfinite']} shape={fp['shape']}"
    )


def nd_compare_log(it: int, prev: Optional[dict], cur: Optional[dict], token_id, token_prob) -> None:
    """Log per-stage diff vs previous iteration; flag the first diverging stage."""
    if cur is None:
        return
    argmax = cur.get("__logits_argmax__")
    gap = cur.get("__logits_gap__")
    logger.info(
        f"[NDPROBE] iter={it} sampled_token={token_id} prob={token_prob:.6f} logits_argmax={argmax} logits_top2gap={gap:.6e}"
    )
    if prev is None:
        # Baseline iteration: just print per-stage fingerprints.
        for k in cur.get("__order__", []):
            f = cur[k]
            logger.info(
                f"[NDPROBE]   iter={it} stage={k:>10} sha={f['sha']} norm={f['norm']:.6f} nonfinite={f['n_nonfinite']} shape={f['shape']}"
            )
        return
    first_div = None
    for k in cur.get("__order__", []):
        cf = cur[k]
        pf = prev.get(k)
        if pf is None:
            continue
        changed = cf["sha"] != pf["sha"]
        dnorm = cf["norm"] - pf["norm"]
        if changed and first_div is None:
            first_div = k
        tag = "DIFF" if changed else "same"
        logger.info(
            f"[NDPROBE]   iter={it} stage={k:>10} {tag} "
            f"sha {pf['sha']}->{cf['sha']} dnorm={dnorm:+.6e} "
            f"nonfinite {pf['n_nonfinite']}->{cf['n_nonfinite']}"
        )
    if first_div is None:
        logger.info(f"[NDPROBE] iter={it} VERDICT: bit-identical to iter {it-1} (fully deterministic this step)")
    else:
        logger.info(f"[NDPROBE] iter={it} VERDICT: FIRST DIVERGING STAGE = {first_div} (drift born here)")
