# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The V4-Flash weight loader (tt/v4/weights) against the real checkpoint on the pod's NFS and the reference
module's state-dict names/shapes. Device-free; skips when the checkpoint is not mounted. Reads ~200 MB."""

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4DecoderLayer
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import CSA, SLIDING, layer_kinds
from models.demos.deepseek_v3_d_p.tt.v4.weights import hf_names
from models.demos.deepseek_v3_d_p.tt.v4.weights.dequant import read_weight_map
from models.demos.deepseek_v3_d_p.tt.v4.weights.layer_weights import classify_layer

_MODEL = os.environ.get("DEEPSEEK_V4_FLASH_HF_MODEL", "/mnt/tt-data/sdawle/models/DeepSeek-V4-Flash-0731")


@pytest.fixture(scope="module")
def model_dir():
    if not Path(_MODEL, "model.safetensors.index.json").is_file():
        pytest.skip(f"no DeepSeek-V4-Flash checkpoint at {_MODEL}")
    return _MODEL


@pytest.fixture(scope="module")
def weight_map(model_dir):
    return read_weight_map(model_dir)


def test_checkpoint_taxonomy_matches_the_layer_schedule(model_dir, weight_map):
    cfg = deepseek_v4_flash_hf_config()
    kinds = layer_kinds(cfg)
    for li in (0, 1, 2, 3, 4, 41, 42):
        k = classify_layer(model_dir, li, weight_map=weight_map)
        assert k.has_compressor == (kinds[li] != SLIDING), (li, k)
        assert k.has_indexer == (kinds[li] == CSA), (li, k)
        assert k.routing == ("hash" if cfg.mlp_layer_types[li] == "hash_moe" else "learned"), (li, k)
    assert not any(k.startswith("mtp.") for k in hf_names.TOP_LEVEL)


def _reference_shapes(layer_idx):
    """Reference state-dict shapes at the real dims, with the routed experts shrunk (256 x 2048 x 4096 fp32 experts
    would be 25 GB on host); expert-dependent keys are checked analytically instead."""
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=43)
    cfg.moe_intermediate_size = 32
    cfg.intermediate_size = 32
    layer = DeepseekV4DecoderLayer(cfg, layer_idx=layer_idx)
    return {k: tuple(v.shape) for k, v in layer.state_dict().items() if "experts" not in k}


@pytest.mark.parametrize("layer_idx", [0, 2, 3])
def test_layer_torch_dict_has_the_reference_names_and_shapes(model_dir, weight_map, layer_idx):
    t0 = time.perf_counter()
    d = hf_names.layer_torch_dict(model_dir, layer_idx, weight_map=weight_map)
    logger.info(f"layer {layer_idx}: {len(d) - 1} tensors in {time.perf_counter() - t0:.1f}s (NFS + dequant)")
    kind = d.pop("__kind__")
    ref = _reference_shapes(layer_idx)
    got = {k: tuple(v.shape) for k, v in d.items() if "experts" not in k}
    assert set(got) == set(ref), f"missing {set(ref) - set(got)} / extra {set(got) - set(ref)}"
    for k in ref:
        assert got[k] == ref[k], (k, got[k], ref[k])
    # the shared expert at the real intermediate size, [out, in]
    assert tuple(d["mlp.shared_experts.gate_proj.weight"].shape) == (2048, 4096)
    assert tuple(d["mlp.shared_experts.up_proj.weight"].shape) == (2048, 4096)
    assert tuple(d["mlp.shared_experts.down_proj.weight"].shape) == (4096, 2048)
    # dtypes: fp8-sourced projections and norms in bf16, mHC in fp32, sinks fp32, the hash table int64
    assert d["self_attn.q_a_proj.weight"].dtype == torch.bfloat16 and d["attn_hc.fn"].dtype == torch.float32
    assert d["self_attn.sinks"].dtype == torch.float32
    if kind.routing == "hash":
        assert d["mlp.gate.tid2eid"].dtype == torch.int64 and tuple(d["mlp.gate.tid2eid"].shape) == (129280, 6)
        assert "mlp.gate.e_score_correction_bias" not in d
    else:
        assert tuple(d["mlp.gate.e_score_correction_bias"].shape) == (256,)
    for k, v in d.items():
        if v.is_floating_point():
            assert torch.isfinite(v.float()).all(), k


def test_one_routed_expert_dequantises_to_hf_orientation(model_dir, weight_map):
    t0 = time.perf_counter()
    ((e, w),) = list(hf_names.iter_layer_experts(model_dir, 3, expert_ids=[7], weight_map=weight_map))
    dt = time.perf_counter() - t0
    logger.info(f"expert 7 of layer 3: {dt:.2f}s")
    assert e == 7
    assert tuple(w["gate_proj"].shape) == (2048, 4096) and tuple(w["up_proj"].shape) == (2048, 4096)
    assert tuple(w["down_proj"].shape) == (4096, 2048)
    for k, v in w.items():
        assert v.dtype == torch.bfloat16 and torch.isfinite(v.float()).all() and v.float().abs().max() > 0, k
    # E2M1 x 2^k values: every |w| is 0 or one of {0.5,1,1.5,2,3,4,6} times a power of two
    vals = w["gate_proj"].float().abs().flatten()
    nz = vals[vals > 0]
    mant = nz / torch.exp2(torch.floor(torch.log2(nz)))
    assert torch.all((mant - mant.round(decimals=1)).abs() < 1e-3)  # mantissas in {1, 1.5} after normalisation


def test_top_level_tensors(model_dir, weight_map):
    d = hf_names.top_level_torch_dict(model_dir, weight_map=weight_map)
    assert tuple(d["model.embed_tokens.weight"].shape) == (129280, 4096)
    assert tuple(d["lm_head.weight"].shape) == (129280, 4096)
    assert tuple(d["model.norm.weight"].shape) == (4096,)
    assert tuple(d["model.hc_head.hc_fn"].shape) == (4, 16384) and d["model.hc_head.hc_fn"].dtype == torch.float32
    assert tuple(d["model.hc_head.hc_base"].shape) == (4,) and tuple(d["model.hc_head.hc_scale"].shape) == (1,)
