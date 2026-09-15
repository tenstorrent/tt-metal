# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WHERE does a 64-row bucket lose exactness? GDN carry, or attention/KV?

A 64-row verify bucket would be worth ~34 ms/step -- ~43 % of the verify's device time is
row-proportional (25 % collectives, 17.6 % layout churn) and the bucket computes 128 rows to verify
at most 16 tokens. It is the last lever with enough headroom to matter, and it is blocked:
tests/reference/test_dflash_anchor64.py measured pcc 0.9666 / 0.9988 / 0.9916 at chunk_start
64 / 128 / 192 instead of the exact 1.0 a 128-row bucket achieves.

THE CLUE THAT NARROWS IT. A span that fits ENTIRELY inside one 64-row bucket is exact (start=40,
pcc 1.000000); every span that crosses a bucket boundary is not. So the fault is in what CARRIES
across a boundary, not in a 64-row forward itself. Three things carry: the GDN recurrent state, the
GDN conv carry (``conv_carry``, "cross-chunk prefill conv carry"), and the paged KV.

The bucket/page SPACING hypothesis is already eliminated: it was most of the error (those offsets
measure 0.16 / 0.27 / 0.25 at bucket 128) and fixing it left this residue.

This bisects the remaining candidates. One 80-token sequence, two ways:

    golden: one bucket of 128, actual_len 80
    split:  bucket 64 [0,64) then bucket 64 [64,80)

Both must end with the same GDN state and the same logits. Comparing the GDN state PER LAYER
separates the cases:

* states diverge  -> the GDN carry is the fault. The per-layer pattern says more: ALL layers wrong
  points at the conv carry or the recurrent update; a SUFFIX of layers wrong points at something
  positional feeding them.
* states match but logits diverge -> GDN is innocent and the fault is in attention/paged KV.

Note ``long_prefill_chunk_size = 128`` (tt/gdn/config.py) while the fused op tiles at 32 -- a
64-row bucket is a PARTIAL long-prefill chunk, which is the leading suspect going in.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/unit/test_bucket64_carry_bisect.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
PREFIX, TAIL = 64, 16  # 80 tokens: two 64-row buckets, or one 128-row bucket


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


def _page_table():
    return torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)


def _flat(comp):
    """All of one layer's state components as a single vector, for the whole-layer view."""
    return torch.cat([comp[k] for k in sorted(comp)]) if comp else torch.zeros(1)


def _pcc(a, b):
    from models.common.utility_functions import comp_pcc

    _, p = comp_pcc(a, b, 0.99)
    return float(str(p).split()[-1]) if not isinstance(p, float) else p


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_where_bucket64_diverges(mesh_device, device_params, reset_seeds, ensure_gc):
    """Same 80 tokens through one 128-bucket and two 64-buckets; compare state and logits."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    pt = _page_table()

    g = torch.Generator().manual_seed(29)
    total = PREFIX + TAIL
    tokens = torch.randint(1000, 2000, (1, total), generator=g, dtype=torch.long)

    def _state_to_host(snap):
        """Per-layer GDN state, SPLIT BY COMPONENT.

        Keeping rec_state and conv_carry separate is the point: 'the carry diverged' is not
        actionable, 'conv_carry diverged while rec_state is exact' names the kernel to look at.
        GDN carries two different things across a chunk boundary and they fail for different
        reasons -- the conv carry is the last K-1 rows of the previous chunk, the recurrent state
        is the scan's accumulator.
        """
        out = []
        for layer in snap:
            comp = {}
            for key in ("rec_state", "conv_carry"):
                t = layer.get(key)
                if t is not None:
                    comp[key] = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().flatten()
            cs_list = layer.get("conv_states") or []
            if cs_list:
                comp["conv_states"] = torch.cat(
                    [ttnn.to_torch(ttnn.get_device_tensors(c)[0]).float().flatten() for c in cs_list]
                )
            out.append(comp)
        return out

    def _pcc_by_component(a_layers, b_layers):
        """Worst pcc per component name across all layers, plus how many layers each failed on."""
        stats = {}
        for a, b in zip(a_layers, b_layers):
            for k in a:
                if k not in b:
                    continue
                v = _pcc(a[k], b[k])
                cur = stats.setdefault(k, {"worst": 1.0, "bad": 0, "n": 0})
                cur["worst"] = min(cur["worst"], v)
                cur["bad"] += v < 0.9999
                cur["n"] += 1
        return stats

    # --- golden: one 128-row bucket over all 80 tokens ---
    golden_logits = model.prefill_block_all_logits(tokens, pt, actual_len=total, chunk_start=0, bucket=128)
    golden_state = _state_to_host(model.save_gdn_state())

    # --- split: two 64-row buckets ---
    model.prefill_block_all_logits(tokens[:, :PREFIX], pt, actual_len=PREFIX, chunk_start=0, bucket=64)
    mid_state = _state_to_host(model.save_gdn_state())
    split_logits = model.prefill_block_all_logits(
        tokens[:, PREFIX:total], pt, actual_len=TAIL, chunk_start=PREFIX, bucket=64
    )
    split_state = _state_to_host(model.save_gdn_state())

    # --- is the FIRST bucket already wrong, or only the carry's consumption? ---
    # mid_state is the GDN state after the 64-row bucket [0,64). Re-run those same 64 tokens as a
    # single 128-row bucket -- no carry involved at all -- and compare. Exact here means a 64-row
    # forward is fine on its own and the fault is in the SECOND bucket consuming the carry;
    # inexact means the 64-row forward is already wrong before any carry happens.
    model.prefill_block_all_logits(tokens[:, :PREFIX], pt, actual_len=PREFIX, chunk_start=0, bucket=128)
    ref_prefix_state = _state_to_host(model.save_gdn_state())
    first_comp = _pcc_by_component(ref_prefix_state, mid_state)
    for k, v in sorted(first_comp.items()):
        logger.info(f"    first-bucket {k:12} worst {v['worst']:.6f}  bad {v['bad']}/{v['n']} layers")
    first_bucket = [_pcc(_flat(a), _flat(b)) for a, b in zip(ref_prefix_state, mid_state)]
    first_bad = [i for i, v in enumerate(first_bucket) if v < 0.9999]
    logger.info("=" * 78)
    logger.info(
        f"  FIRST 64-row bucket alone (no carry): {len(first_bad)} of {len(first_bucket)} GDN layers "
        f"below 0.9999, worst {min(first_bucket):.6f}"
    )
    logger.info("    exact   -> a 64-row forward is fine; the fault is the SECOND bucket's carry")
    logger.info("    inexact -> the 64-row forward itself is wrong, before any carry")

    # --- compare ---
    split_comp = _pcc_by_component(golden_state, split_state)
    logger.info("=" * 78)
    for k, v in sorted(split_comp.items()):
        logger.info(f"  AFTER CARRY  {k:12} worst {v['worst']:.6f}  bad {v['bad']}/{v['n']} layers")
    per_layer = [_pcc(_flat(a), _flat(b)) for a, b in zip(golden_state, split_state)]
    worst_layer = int(min(range(len(per_layer)), key=lambda i: per_layer[i]))
    bad = [i for i, v in enumerate(per_layer) if v < 0.9999]
    logit_pcc = _pcc(golden_logits[:, PREFIX:], split_logits)

    logger.info("=" * 78)
    logger.info(f"  GDN layers below 0.9999: {len(bad)} of {len(per_layer)}")
    if bad:
        logger.info(f"  first bad layer {bad[0]}, worst layer {worst_layer} at pcc {per_layer[worst_layer]:.6f}")
        logger.info(f"  bad layer indices: {bad[:20]}{'...' if len(bad) > 20 else ''}")
    logger.info(f"  per-layer pcc (first 12): {[f'{v:.4f}' for v in per_layer[:12]]}")
    logger.info(f"  TAIL LOGITS pcc: {logit_pcc:.6f}")
    logger.info("  GDN state diverged  -> the carry is the fault (conv_carry / recurrent update)")
    logger.info("  GDN state exact     -> GDN is innocent; look at attention / paged KV")
    logger.info("=" * 78)
    print(f"\n>>> gdn layers bad {len(bad)}/{len(per_layer)} | tail logits pcc {logit_pcc:.6f}\n")

    # Diagnostic, not a gate: this test exists to LOCATE the divergence, and it is already known
    # that the split path is not exact. Assert only that it reproduced the known symptom, so a
    # silent pass cannot be mistaken for the bug having disappeared.
    assert logit_pcc < 0.9999 or len(bad) > 0, (
        f"expected the 64-row split to diverge (logits {logit_pcc:.6f}, {len(bad)} bad GDN layers) "
        "but it looks exact -- if that is real, re-run test_dflash_anchor64.py, the blocker may be gone"
    )
