# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hy4-preview prefill vocab-sharded embedding (#65892, tt/hy4/embedding.py) on the 8x4 Galaxy, tp32 layout.

The checkpoint's 120832 x 6144 BF16 table, sharded by vocab over the 32 chips (decode's TtHy4Embedding shards), embeds
the 4K golden prompt as the runner hands it in (ids SP-sharded over axis 0, ``[1, 1, 512]`` per chip) and must come
back as the block residual ``[512, 1536]`` per chip equal to ``table[ids]``, bit for bit (every reduce-scatter sum
has one non-zero term). Also checked:

  * the cache: the first build converts the checkpoint table and writes ``embed_vocab32_8x4`` + its layout stamp
    under ``HY4_TP32_TTNN_CACHE`` (default: the dev cache dir the tp32 tests share); a second instance loads it with
    no weights and gives the same output;
  * one table for both engines: a decode ``TtHy4Embedding`` on the same mesh, and a prefill embedding ``sharing`` its
    device table, give the same output again;
  * out-of-vocab ids (``MTP_PAD_TOKEN_ID``) embed as zeros;
  * per-chip table bytes (this vs ``TtParallelEmbedding``) and the forward's time per 4096-token chunk.

Env: ``HY4_HF_MODEL``, ``PREFILL_TRACE_DIR`` (golden: its token ids), ``HY4_TP32_TTNN_CACHE``, ``HY4_EMBED_REPORT``
(optional JSON path).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4TP32Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import moe_fabric_payload, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.hy4.layout import HY4_L1_SMALL_SIZE

CHECKPOINT = Path(os.environ.get("HY4_HF_MODEL", "/mnt/tt-data/Hy4-Preview"))
GOLDEN = Path(os.environ.get("PREFILL_TRACE_DIR", "/mnt/tt-data/dwadhwani/hy4-golden/hy4-bf16-longbench-4k-ropeint"))
CACHE = Path(
    os.environ.get(
        "HY4_TP32_TTNN_CACHE", "/mnt/tt-data/dwadhwani/Hy4-Preview-Cache/prefill-v1/hy4_preview_bh_32dev/8x4"
    )
)
S = 4096
SP_AXIS, TP_AXIS = 0, 1
TIMED_ITERS = 10

pytestmark = [
    pytest.mark.skipif(not (CHECKPOINT / "model.safetensors.index.json").is_file(), reason="no Hy4 checkpoint"),
    pytest.mark.skipif(not (GOLDEN / "metadata.json").is_file(), reason=f"no Hy4 golden at {GOLDEN}"),
]


def _pcc(a, b) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _upload_ids(ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """[S] -> per chip [1, 1, S / sp] uint32 ROW_MAJOR, SP-sharded over axis 0, as the runner's input_prep does."""
    sp = mesh_device.shape[SP_AXIS]
    dims = [None, None]
    dims[SP_AXIS] = 0
    return ttnn.from_torch(
        ids.to(torch.int64).reshape(sp, 1, -1).contiguous(),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims)),
    )


def _to_host(x, mesh_device) -> torch.Tensor:
    """[1, 1, S / sp, H / tp] per chip -> [S, H]: the sequence concatenated over SP, the hidden over TP."""
    return ttnn.to_torch(
        x, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape))
    ).reshape(-1, x.shape[-1] * mesh_device.shape[TP_AXIS])


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=moe_fabric_payload(Hy4TP32Config), l1_small_size=HY4_L1_SMALL_SIZE
            ),
            id="tp32-8x4",
        )
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(0)
def test_hy4_prefill_vocab_sharded_embedding(mesh_device, device_params):
    from models.demos.deepseek_v3_d_p.tt.hy4 import embedding as emb
    from models.demos.deepseek_v3_d_p.tt.runners import hy4_checkpoint as hc
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
    from models.experimental.hy4_preview.reference.lm_head import LMHeadConfig
    from models.experimental.hy4_preview.tt.common import MeshPlan
    from models.experimental.hy4_preview.tt.embedding import TtHy4Embedding

    assert tuple(mesh_device.shape) == (8, 4)
    topology = per_axis_topology(device_params["fabric_config"])
    num_links = 2
    V, H = Hy4TP32Config.VOCAB_SIZE, Hy4TP32Config.EMB_SIZE
    kw = dict(sp_axis=SP_AXIS, tp_axis=TP_AXIS, num_links=num_links, topology=topology)

    weight = hc.Hy4Checkpoint(CHECKPOINT).get("model.embed_tokens.weight").to(torch.bfloat16).float()
    assert tuple(weight.shape) == (V, H)
    ids = torch.tensor(json.loads((GOLDEN / "metadata.json").read_text())["token_ids"][:S])
    assert ids.numel() == S
    want = weight[ids]  # [S, H]
    tt_ids = _upload_ids(ids, mesh_device)

    report = {}

    def score(name, module):
        out = module(tt_ids)
        assert tuple(out.shape) == (1, 1, S // 8, H // 4), tuple(out.shape)
        got = _to_host(out, mesh_device).float()
        ttnn.deallocate(out)
        row = {"pcc": _pcc(got, want), "max_abs_diff": (got - want).abs().max().item(), "exact": torch.equal(got, want)}
        logger.info(f"[embed] {name}: PCC {row['pcc']:.6f} max |diff| {row['max_abs_diff']:.3g} exact {row['exact']}")
        report[name] = row
        return got

    # 1) Built from the checkpoint table, writing the cache (or loading it, if a matching stamp is there already).
    CACHE.mkdir(parents=True, exist_ok=True)
    cached_before = emb.TtHy4PrefillEmbedding.check_cache_complete(CACHE, (8, 4), SP_AXIS, V)
    t0 = time.perf_counter()
    built = emb.TtHy4PrefillEmbedding(mesh_device, V, H, torch_weight=weight, weight_cache_path=CACHE, **kw)
    report["build_s"] = time.perf_counter() - t0
    assert emb.TtHy4PrefillEmbedding.check_cache_complete(CACHE, (8, 4), SP_AXIS, V)
    first = score("from_checkpoint", built)

    # 2) Loaded from the cache with no weights (what the runner does once the cache is complete).
    t0 = time.perf_counter()
    loaded = emb.TtHy4PrefillEmbedding(mesh_device, V, H, torch_weight=None, weight_cache_path=CACHE, **kw)
    report["load_s"] = time.perf_counter() - t0
    assert torch.equal(score("from_cache", loaded), first)
    built.deallocate()

    # Per-chip table bytes: this table vs TtParallelEmbedding's [V, H / 4] (both BF16 ROW_MAJOR DRAM).
    per_chip = ttnn.get_device_tensors(loaded.table)[0]
    assert tuple(per_chip.shape)[-2:] == (V // 32 + 2, H)
    report["table_bytes_per_chip"] = loaded.table_bytes_per_chip()
    report["old_table_bytes_per_chip"] = V * (H // 4) * 2
    logger.info(
        f"[embed] table per chip {report['table_bytes_per_chip'] / 1e6:.1f} MB (was {report['old_table_bytes_per_chip'] / 1e6:.1f} MB)"
    )

    # 3) Time per 4096-token chunk (eager, ops back to back, one synchronize per call).
    for _ in range(2):
        ttnn.deallocate(loaded(tt_ids))
    ttnn.synchronize_device(mesh_device)
    times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        out = loaded(tt_ids)
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
        ttnn.deallocate(out)
    times.sort()
    report["forward_ms_median"] = 1e3 * times[len(times) // 2]
    report["forward_ms_min"] = 1e3 * times[0]
    logger.info(f"[embed] forward per {S}-token chunk: median {report['forward_ms_median']:.2f} ms")

    # 4) One table for both engines: decode's module holds it, prefill reads it.
    decode = TtHy4Embedding(
        MeshPlan(mesh_device, hidden_axis=TP_AXIS), LMHeadConfig(hidden_size=H, vocab_size=V), weight
    )
    shared = emb.TtHy4PrefillEmbedding.sharing(decode, **kw)
    assert shared.table is decode.table
    assert torch.equal(score("shared_with_decode", shared), first)
    ttnn.deallocate(decode.table)
    ttnn.deallocate(decode.row_origin)

    # 5) Ids outside the vocab (MTP_PAD_TOKEN_ID) embed as zeros.
    from models.demos.common.prefill.runners.runner_utils import MTP_PAD_TOKEN_ID

    pad = ids.clone()
    pad[1::7] = MTP_PAD_TOKEN_ID
    tt_pad = _upload_ids(pad, mesh_device)
    out = loaded(tt_pad)
    got = _to_host(out, mesh_device).float()
    keep = pad != MTP_PAD_TOKEN_ID
    assert not got[~keep].any(), "an out-of-vocab id must embed as zeros"
    assert torch.equal(got[keep], want[keep])
    report["pad_ids_zero"] = True

    if os.environ.get("HY4_EMBED_REPORT"):
        Path(os.environ["HY4_EMBED_REPORT"]).write_text(json.dumps(report, indent=1))
    logger.info(f"[embed] report {json.dumps(report)}")
    for name in ("from_checkpoint", "from_cache", "shared_with_decode"):
        assert report[name]["exact"], f"{name}: not bit-exact (PCC {report[name]['pcc']:.6f})"
