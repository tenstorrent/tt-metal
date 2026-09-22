# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic: *where* does layer 0's device-vs-golden error live?

Layer 0 measures k 0.9960 / v 0.9961 on device. On CPU the same layer with the spec's
``bfloat8_b`` weights measures k 0.9999929 / v 0.9999479, and the bf8 *cache* round-trip floor
is 0.99996 — so the device is ~80x worse than every precision floor it is subject to, and the
gap does not move when matmul fidelity changes. A uniform precision loss cannot do that; losing
a small *fraction* of elements can (PCC ~ sqrt(1-f), so 0.996 is f ~ 0.8%).

This run slices the same tensors by token, by KV head and by channel to say which. One layer,
so it loads in ~10 s instead of ~14 min.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.mistral_medium_3_5_128b.reference.checkpoint import CheckpointLoader
from models.demos.mistral_medium_3_5_128b.reference.golden import GoldenTrace
from models.demos.mistral_medium_3_5_128b.tests.device_utils import read_kv_cache
from models.demos.mistral_medium_3_5_128b.tests.galaxy_prefill_kv_pcc import build_real_model
from models.demos.mistral_medium_3_5_128b.tt.runtime import PrefillRuntime, RuntimeConfig

SEQ = int(os.getenv("DIAG_SEQ", "10240"))


def pcc(a, b):
    x, y = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


@pytest.fixture(scope="module")
def trace():
    return GoldenTrace.from_env()


@pytest.mark.timeout(1800)
def test_localize_layer0(galaxy, cfg, mesh_config, ccl, spec, trace):
    model = build_real_model(
        galaxy,
        cfg,
        mesh_config,
        ccl,
        seq_len=SEQ,
        cache_period=SEQ,
        num_layers=1,
        loader=CheckpointLoader.from_env(cfg),
    )
    kv = model.allocate_cache()
    runtime = PrefillRuntime(
        model, RuntimeConfig(chunk_size=SEQ, max_seq_len=kv.max_seq_len, num_layers=1, num_users=1)
    )
    runtime.compile(kv)

    ids = trace.token_ids(SEQ)
    tokens = runtime.make_chunk_input(ids[0, :SEQ].tolist())
    runtime.prefill_chunk(tokens, kv, slot_id=0, actual_start=0, actual_end=SEQ, request_id=0)
    ttnn.synchronize_device(galaxy)

    got_k = read_kv_cache(galaxy, kv.k, cache_global=kv.max_seq_len, chunk_size=SEQ, upto=SEQ)[0:1]
    got_v = read_kv_cache(galaxy, kv.v, cache_global=kv.max_seq_len, chunk_size=SEQ, upto=SEQ)[0:1]
    ref_k, ref_v = trace.layer_kv_meta(0, SEQ)

    for name, ref, got in (("k", ref_k, got_k), ("v", ref_v, got_v)):
        logger.info(f"=== {name}: overall {pcc(ref, got):.7f}  shape {tuple(got.shape)}")

        sp = mesh_config.sp
        per_row = SEQ // sp
        by_row = [
            pcc(ref[:, :, r * per_row : (r + 1) * per_row], got[:, :, r * per_row : (r + 1) * per_row])
            for r in range(sp)
        ]
        logger.info(f"{name} by SP row ({per_row} tok each): " + " ".join(f"{p:.5f}" for p in by_row))

        by_head = [pcc(ref[:, h], got[:, h]) for h in range(ref.shape[1])]
        logger.info(f"{name} by KV head: " + " ".join(f"{p:.5f}" for p in by_head))

        half = ref.shape[-1] // 2
        logger.info(
            f"{name} by channel half: lo {pcc(ref[..., :half], got[..., :half]):.6f} "
            f"hi {pcc(ref[..., half:], got[..., half:]):.6f}"
        )

        # Where is the mass of the error? Per-token relative error, worst offenders.
        err = (ref.double() - got.double()).pow(2).sum(dim=(0, 1, 3))
        sig = ref.double().pow(2).sum(dim=(0, 1, 3))
        rel = (err / sig.clamp_min(1e-30)).sqrt()
        worst = torch.topk(rel, 12)
        logger.info(
            f"{name} per-token rel err: median {rel.median():.4f} p99 {rel.quantile(0.99):.4f} max {rel.max():.4f}"
        )
        logger.info(f"{name} worst token positions: {worst.indices.tolist()}")
        logger.info(f"{name} worst token rel errs: " + " ".join(f"{x:.3f}" for x in worst.values.tolist()))
        logger.info(
            f"{name} frac tokens rel>0.5: {(rel > 0.5).double().mean():.5f}  rel>0.1: {(rel > 0.1).double().mean():.5f}"
        )
