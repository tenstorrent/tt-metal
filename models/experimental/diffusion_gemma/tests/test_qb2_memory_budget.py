# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""#47487 — QB2 per-chip DRAM budget for the 26B-A4B causal backbone.

Measures per-chip DRAM (weights, and weights+paged-KV up to 256K) via
`ttnn.get_memory_view`, to document the QB2 memory budget + batch ceiling.
The paged KV cache is allocated eagerly at build, so the weights+KV budget is
captured WITHOUT a prefill. Env-parameterized (one build per pytest process):

    PROBE_KV=0|1   PROBE_CTX=<max_seq_len>   PROBE_BATCH=<batch>

    DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 HF_MODEL=<gemma-4-26B-A4B-it checkpoint> \
      PROBE_KV=1 PROBE_CTX=262144 PROBE_BATCH=1 \
      pytest models/experimental/diffusion_gemma/tests/test_qb2_memory_budget.py -k 1x4 -q -s
"""
import math
import os

import pytest
from loguru import logger

import ttnn
from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

PROBE_KV = os.getenv("PROBE_KV", "1") == "1"
PROBE_CTX = int(os.getenv("PROBE_CTX", "262144"))
PROBE_BATCH = int(os.getenv("PROBE_BATCH", "1"))
# No personal-path default: require HF_MODEL (a 26B-A4B checkpoint dir); skip when
# unset so the harness is portable instead of pointing at a developer's home dir.
MODEL_PATH = os.getenv("HF_MODEL")
PROBE_PREFILL = os.getenv("PROBE_PREFILL", "0") == "1"
PREFILL_LEN = int(os.getenv("PROBE_PREFILL_LEN", str(PROBE_CTX)))
BLOCK = 64
G = 2**30

pytestmark = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (QB2, MESH_DEVICE=P150x4)",
)


def _dram(mesh_device, label):
    ttnn.synchronize_device(mesh_device)
    v = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    used = v.num_banks * v.total_bytes_allocated_per_bank
    total = v.num_banks * v.total_bytes_per_bank
    free = v.num_banks * v.total_bytes_free_per_bank
    logger.info(
        f"[{label}] per-chip DRAM: used={used/G:.3f} GiB  free={free/G:.3f} GiB  "
        f"usable_total={total/G:.3f} GiB  banks={v.num_banks}"
    )
    return used / G, total / G


@parametrize_mesh_with_fabric()
def test_qb2_dram_budget(mesh_device, reset_seeds, request):
    from models.demos.gemma4.tt.common import create_tt_model
    from models.tt_transformers.tt.common import PagedAttentionConfig

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if tp < 2:
        pytest.skip("26B-A4B backbone needs TP>=2 (use -k 1x4 on QB2)")
    if MODEL_PATH is None:
        pytest.skip("set HF_MODEL to a 26B-A4B checkpoint dir (no personal-path default)")

    base_used, total = _dram(mesh_device, "baseline (empty)")
    pac = (
        PagedAttentionConfig(block_size=BLOCK, max_num_blocks=PROBE_BATCH * math.ceil(PROBE_CTX / BLOCK))
        if PROBE_KV
        else None
    )
    logger.info(f"[cfg] KV={PROBE_KV} ctx={PROBE_CTX} batch={PROBE_BATCH} model={MODEL_PATH}")

    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device,
        max_batch_size=PROBE_BATCH,
        max_seq_len=PROBE_CTX,
        paged_attention_config=pac,
        create_kv_cache=PROBE_KV,
        bounded_sliding_kv_cache=(PROBE_CTX > 16384),
        model_path=MODEL_PATH,
    )

    used, total = _dram(mesh_device, f"built KV={int(PROBE_KV)} ctx={PROBE_CTX} batch={PROBE_BATCH}")
    logger.info(
        f"[BUDGET RESULT] KV={int(PROBE_KV)} ctx={PROBE_CTX} batch={PROBE_BATCH}  "
        f"footprint_over_baseline={used-base_used:.3f} GiB/chip  usable={total:.3f} GiB/chip  "
        f"headroom={total-used:.3f} GiB/chip"
    )

    if PROBE_PREFILL:
        import torch
        import torch.nn.functional as F

        # Non-traced single-chunk prefill of PREFILL_LEN tokens (pad to pow2, like the
        # demo). Materializes the full [1, L, hidden] activation -> stresses the
        # prefill-activation memory regime (the real long-context ceiling, distinct
        # from the static weights+KV budget above). Completion = fits; OOM = ceiling.
        padded = 1 << max((PREFILL_LEN - 1).bit_length(), 11)
        logger.info(f"[prefill] L={PREFILL_LEN} padded={padded}")
        ids = torch.randint(0, model_args.vocab_size, (1, padded), dtype=torch.long)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        tt_tokens = ttnn.from_torch(
            ids.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=replicate,
        )
        embeds = model.embed_tokens(tt_tokens)
        embeds = ttnn.reshape(embeds, (1, 1, padded, model_args.hidden_size))
        embeds = ttnn.to_layout(embeds, ttnn.TILE_LAYOUT)
        embed_w = state_dict.get(
            "model.language_model.embed_tokens.weight", state_dict.get("model.embed_tokens.weight")
        )
        embeds_torch = (F.embedding(ids.long(), embed_w) * model.embed_scale).float()
        out = model.ttnn_prefill_forward(
            embeds,
            page_table=None,
            kv_cache=tt_kv_cache,
            input_ids_torch=ids,
            embeds_torch=embeds_torch,
        )
        pk_used, pk_total = _dram(mesh_device, f"after prefill L={PREFILL_LEN}")
        logger.info(f"[PREFILL OK] L={PREFILL_LEN} padded={padded} completed; post-prefill used={pk_used:.3f} GiB/chip")
        out.deallocate(True)
