# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""P1 first light, real checkpoint on the target mesh: the real embedding table + decoder layer 0 on the
trace's full 10240-token prompt (one-shot), against the CPU reference on the same real weights and
against the golden trace's layer-0 K/V (an exact target: layer-0 K/V depend only on each token).
REDUCED DEPTH (1 of 88 layers); the full-depth result is tests/test_prefill_acceptance.py."""

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

from models.demos.mistral_medium_3_5_128b.config import MistralMediumConfig
from models.demos.mistral_medium_3_5_128b.reference.checkpoint import CheckpointReader
from models.demos.mistral_medium_3_5_128b.reference.model import ReferenceDecoderLayer, rope_cos_sin
from models.demos.mistral_medium_3_5_128b.tt.embedding import Embedding
from models.demos.mistral_medium_3_5_128b.tt.kv_cache import allocate_kv_cache, naturalize, read_slot_kv
from models.demos.mistral_medium_3_5_128b.tt.layer import DecoderLayer
from models.demos.mistral_medium_3_5_128b.tt.model import Model
from models.demos.mistral_medium_3_5_128b.tt.rope import RopeSetup, hf_to_meta_perm

from .unit.common import assert_pcc, residual_to_torch, spec_dtypes

CKPT = os.environ.get("PREFILL_HF_MODEL") or os.environ.get("HF_MODEL")
TRACE = os.environ.get("PREFILL_TRACE_DIR")


@pytest.mark.skipif(
    not (CKPT and TRACE and (Path(TRACE) / "metadata.json").is_file()),
    reason="needs the real checkpoint (PREFILL_HF_MODEL / HF_MODEL) and PREFILL_TRACE_DIR",
)
@pytest.mark.timeout(1800)
def test_real_weight_layer0_vs_golden(galaxy_mesh, mesh_config, ccl_manager):
    cfg = MistralMediumConfig.from_json(Path(CKPT) / "config.json")
    with open(Path(TRACE) / "metadata.json") as f:
        tokens = torch.tensor(json.load(f)["token_ids"])
    n = tokens.numel()

    t0 = time.perf_counter()
    reader = CheckpointReader(CKPT)
    table = reader.embedding()
    sd = reader.layer_state_dict(0)
    t1 = time.perf_counter()
    embedding = Embedding(galaxy_mesh, mesh_config, ccl_manager, table)
    layer = DecoderLayer(galaxy_mesh, mesh_config, ccl_manager, cfg, sd, layer_idx=0, dtypes=spec_dtypes())
    t2 = time.perf_counter()
    logger.info(f"real layer 0: checkpoint read+dequant {t1 - t0:.1f}s, device build {t2 - t1:.1f}s")

    rope = RopeSetup(galaxy_mesh, mesh_config, cfg, max_seq_len=n, chunk_size=n)
    cache = allocate_kv_cache(
        galaxy_mesh,
        mesh_config,
        num_layers=1,
        max_seq_len=n,
        num_local_kv_heads=cfg.num_key_value_heads // mesh_config.tp,
        head_dim=cfg.head_dim,
    )
    x = embedding(Model.token_tensor(tokens, galaxy_mesh, mesh_config))
    out = residual_to_torch(layer(x, rope, kv_cache=cache), galaxy_mesh, mesh_config)

    with torch.device("meta"):
        ref_layer = ReferenceDecoderLayer(cfg)
    ref_layer.load_state_dict(sd, assign=True)
    pos = torch.arange(n)
    cos, sin = rope_cos_sin(cfg, pos)
    with torch.no_grad():
        ref, ref_k, ref_v = ref_layer(table[tokens][None], cos, sin, pos)
    assert_pcc("real_layer0_output", ref[None], out)

    perm = hf_to_meta_perm(cfg.head_dim)
    k_blk, v_blk = read_slot_kv(galaxy_mesh, cache, 0)
    dev_k = naturalize(k_blk[0], n, mesh_config.sp, n, n)
    dev_v = naturalize(v_blk[0], n, mesh_config.sp, n, n)
    with safe_open(str(Path(TRACE) / "kv_cache" / "layer_0.safetensors"), framework="pt") as f:
        gk, gv = f.get_tensor("key_cache_layer_0")[0], f.get_tensor("value_cache_layer_0")[0]
    assert torch.equal(gk, ref_k[0]) and torch.equal(gv, ref_v[0]), "reference layer 0 no longer matches the trace"
    assert_pcc("real_layer0_k_vs_golden", gk[..., perm], dev_k)
    assert_pcc("real_layer0_v_vs_golden", gv, dev_v)
