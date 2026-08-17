# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Critical config, weight-remap, checkpoint-shape, and QB2 memory-budget gates."""

import glob
import json
import math
import os

import pytest
import torch
from loguru import logger

from models.experimental.diffusion_gemma.config import TextConfig
from models.experimental.diffusion_gemma.memory_budget import estimate_canvas_kv_scratch_bytes
from models.experimental.diffusion_gemma.reference.self_conditioning import SelfConditioning
from models.experimental.diffusion_gemma.weight_mapping import (
    SELF_CONDITIONING_PREFIX,
    classify_keys,
    expected_self_conditioning_shapes,
    remap_state_dict,
)

HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
DG_DIR = "models--google--diffusiongemma-26B-A4B-it"
G4_DIR = "models--google--gemma-4-26B-A4B-it"


def _index_path(repo_dirname: str):
    hits = glob.glob(os.path.join(HF_CACHE, repo_dirname, "snapshots", "*", "model.safetensors.index.json"))
    return hits[0] if hits else None


def _weight_map(repo_dirname: str):
    path = _index_path(repo_dirname)
    if path is None:
        return None
    return json.load(open(path))["weight_map"]


@pytest.mark.skipif(_weight_map(DG_DIR) is None, reason="DiffusionGemma index.json not in HF cache")
def test_classification_covers_all_keys_no_leftovers():
    dg = _weight_map(DG_DIR)
    result = classify_keys(dg.keys())
    assert result.num_backbone + len(result.self_conditioning) + len(result.ignored) + len(result.unknown) == len(dg)
    assert not result.unknown
    assert len(result.self_conditioning) == 4
    assert result.num_backbone > 600
    assert all(
        key.startswith(("model.encoder.", "model.vision_tower.", "model.embed_vision.")) for key in result.ignored
    )


@pytest.mark.skipif(
    _weight_map(DG_DIR) is None or _weight_map(G4_DIR) is None,
    reason="need BOTH DiffusionGemma and gemma-4-26B-A4B-it index.json in HF cache",
)
def test_remapped_backbone_matches_gemma4_language_model_keyset():
    dg = _weight_map(DG_DIR)
    g4 = _weight_map(G4_DIR)
    remapped = set(classify_keys(dg.keys()).backbone.values())
    g4_lm = {key for key in g4 if key.startswith("model.language_model.")}
    assert not g4_lm - remapped, f"DiffusionGemma backbone missing gemma4 keys: {sorted(g4_lm - remapped)[:10]}"
    assert not remapped - g4_lm, f"DiffusionGemma backbone has extra gemma4 keys: {sorted(remapped - g4_lm)[:10]}"


def _open_self_cond_tensors():
    idx_path = _index_path(DG_DIR)
    if idx_path is None:
        return None
    snapshot = os.path.dirname(idx_path)
    weight_map = json.load(open(idx_path))["weight_map"]
    keys = [key for key in weight_map if key.startswith(SELF_CONDITIONING_PREFIX)]
    shards = {weight_map[key] for key in keys}
    if not all(os.path.exists(os.path.join(snapshot, shard)) for shard in shards):
        return None

    from safetensors import safe_open

    handles = {shard: safe_open(os.path.join(snapshot, shard), framework="pt") for shard in shards}
    return {key: handles[weight_map[key]].get_tensor(key) for key in keys}


@pytest.mark.skipif(_open_self_cond_tensors() is None, reason="DiffusionGemma checkpoint tensors not downloaded")
def test_real_self_conditioning_loads_and_matches_config_shapes():
    _, self_cond, _ = remap_state_dict(_open_self_cond_tensors())
    tc = TextConfig()
    expected = expected_self_conditioning_shapes(tc.hidden_size, tc.intermediate_size)
    for name, shape in expected.items():
        assert tuple(self_cond[name].shape) == shape

    module = SelfConditioning(tc.hidden_size, intermediate_size=tc.intermediate_size).to(torch.float32)
    module.load_from_state_dict({key: value.float() for key, value in self_cond.items()})
    out = module(torch.randn(1, 4, tc.hidden_size), torch.randn(1, 4, tc.hidden_size))
    assert out.shape == (1, 4, tc.hidden_size) and torch.isfinite(out).all()


def test_qb2_canvas_kv_scratch_estimate_matches_gemma4_tp4_shapes():
    estimate = estimate_canvas_kv_scratch_bytes(tp=4, batch_size=1, bytes_per_elem=2)
    assert estimate.sliding_bytes == int(12.5 * 2**20)
    assert estimate.full_attention_bytes == int(2.5 * 2**20)
    assert estimate.total_bytes == 15 * 2**20


PROBE_KV = os.getenv("PROBE_KV", "1") == "1"
PROBE_CTX = int(os.getenv("PROBE_CTX", "262144"))
PROBE_BATCH = int(os.getenv("PROBE_BATCH", "1"))
MODEL_PATH = os.getenv("HF_MODEL")
PROBE_PREFILL = os.getenv("PROBE_PREFILL", "0") == "1"
PREFILL_LEN = int(os.getenv("PROBE_PREFILL_LEN", str(PROBE_CTX)))
BLOCK = 64
G = 2**30


def _mesh_parametrize():
    try:
        from models.demos.gemma4.tests.test_factory import parametrize_mesh_with_fabric

        return parametrize_mesh_with_fabric()
    except Exception as error:
        return pytest.mark.skip(reason=f"mesh parametrization unavailable (no usable device): {error}")


def _dram(mesh_device, label):
    import ttnn

    ttnn.synchronize_device(mesh_device)
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    used = view.num_banks * view.total_bytes_allocated_per_bank
    total = view.num_banks * view.total_bytes_per_bank
    free = view.num_banks * view.total_bytes_free_per_bank
    logger.info(
        f"[{label}] per-chip DRAM: used={used/G:.3f} GiB free={free/G:.3f} GiB "
        f"usable_total={total/G:.3f} GiB banks={view.num_banks}"
    )
    return used / G, total / G


@pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (QB2, MESH_DEVICE=P150x4)",
)
@_mesh_parametrize()
def test_qb2_dram_budget(mesh_device, reset_seeds, request):
    import ttnn

    from models.demos.gemma4.tt.common import create_tt_model
    from models.tt_transformers.tt.common import PagedAttentionConfig

    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if tp < 2:
        pytest.skip("26B-A4B backbone needs TP>=2 (use -k 1x4 on QB2)")
    if MODEL_PATH is None:
        pytest.skip("set HF_MODEL to a 26B-A4B checkpoint dir (no personal-path default)")

    base_used, _ = _dram(mesh_device, "baseline (empty)")
    paged = (
        PagedAttentionConfig(block_size=BLOCK, max_num_blocks=PROBE_BATCH * math.ceil(PROBE_CTX / BLOCK))
        if PROBE_KV
        else None
    )
    logger.info(f"[cfg] KV={PROBE_KV} ctx={PROBE_CTX} batch={PROBE_BATCH} model={MODEL_PATH}")
    model_args, model, tt_kv_cache, state_dict = create_tt_model(
        mesh_device,
        max_batch_size=PROBE_BATCH,
        max_seq_len=PROBE_CTX,
        paged_attention_config=paged,
        create_kv_cache=PROBE_KV,
        bounded_sliding_kv_cache=(PROBE_CTX > 16384),
        model_path=MODEL_PATH,
    )

    used, total = _dram(mesh_device, f"built KV={int(PROBE_KV)} ctx={PROBE_CTX} batch={PROBE_BATCH}")
    logger.info(
        f"[BUDGET RESULT] KV={int(PROBE_KV)} ctx={PROBE_CTX} batch={PROBE_BATCH} "
        f"footprint_over_baseline={used-base_used:.3f} GiB/chip usable={total:.3f} GiB/chip "
        f"headroom={total-used:.3f} GiB/chip"
    )

    if PROBE_PREFILL:
        import torch.nn.functional as F

        padded_len = 1 << max((PREFILL_LEN - 1).bit_length(), 11)
        ids = torch.randint(0, model_args.vocab_size, (1, padded_len), dtype=torch.long)
        tt_tokens = ttnn.from_torch(
            ids.to(torch.int32),
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        embeds = model.embed_tokens(tt_tokens)
        embeds = ttnn.to_layout(ttnn.reshape(embeds, (1, 1, padded_len, model_args.hidden_size)), ttnn.TILE_LAYOUT)
        embed_weight = state_dict.get(
            "model.language_model.embed_tokens.weight", state_dict.get("model.embed_tokens.weight")
        )
        embeds_torch = (F.embedding(ids.long(), embed_weight) * model.embed_scale).float()
        out = model.ttnn_prefill_forward(
            embeds,
            page_table=None,
            kv_cache=tt_kv_cache,
            input_ids_torch=ids,
            embeds_torch=embeds_torch,
        )
        post_used, _ = _dram(mesh_device, f"after prefill L={PREFILL_LEN}")
        logger.info(f"[PREFILL OK] L={PREFILL_LEN} completed; post-prefill used={post_used:.3f} GiB/chip")
        out.deallocate(True)
