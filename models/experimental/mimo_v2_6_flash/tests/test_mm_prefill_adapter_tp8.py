# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal prefill through the vLLM adapter (generator_vllm.MiMoV2OmniForCausalLM.prefill_forward) on one 1x8 row,
as a row engine serves it (vLLM's KV layout with the KV pair split, the plugin's warm-up order: warmup_model_prefill,
then the eager and the traced warmup_model_decode), with MIMO_MM_TRACE: chunks with multimodal rows go through the
traced passes (prefill batches, traced first chunks, traced later chunks) with their rows merged from the traces'
persistent inputs. They must give exactly what the same prompts give from their tokens.

An encoder backend that returns the embedding table's rows of the token ids an image item carries (in its "pixels")
stands in for the ViT, so a prompt whose placeholder run is such an item gives the embeddings of the text it replaces.
Each step prefills a text row and its multimodal twin (placeholders + item) and compares their last-token logits bit for
bit, and checks the multimodal row's pass replayed a trace:

1. two 300-token first chunks in one prefill batch (padded 384): the twin's item at [40, 104);
2. two 900-token first chunks (padded 1024: traced first chunks), the item at [100, 612);
3. two 700-token prompts in two steps, the item at [450, 578) straddling them: first chunks of 512 (one prefill
   batch), then later chunks of 188 tokens at 512 (padded 256: traced later chunks).

Nothing may compile while serving (no decode trace recapture), and the optional multimodal traces' share of the trace
region is logged. Run on a row: TT_VISIBLE_DEVICES=<the row's 8 chips>.
"""

import os
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS
from models.experimental.mimo_v2_6_flash.tt import generator_vllm as gv
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import BLOCK_SIZE, MiMoV2ForCausalLM, MiMoV2OmniForCausalLM

IMAGE = 151655
PER_LANE = 8
MAX_SEQ_LEN = 8192
BUDGET = 4096
SWA_RING = 64  # SWA blocks per request (a window and a chunk fit many times)


def vllm_kv_layout(cfg):
    """vLLM's KV cache groups of MiMo (tests/perf/vllm_adapter_prefill.vllm_kv_layout, which is not imported: it turns
    prefill batches off): (each layer's group, each layer's buffer within its group)."""
    kinds = {}
    for i in range(cfg.num_hidden_layers):
        kinds.setdefault(cfg.is_swa(i), []).append(i)
    size = min(len(layers) for layers in kinds.values())
    groups, buffers = [None] * cfg.num_hidden_layers, [None] * cfg.num_hidden_layers
    first = 0
    for layers in kinds.values():
        n = -(-len(layers) // size)
        for k in range(n):
            for t, i in enumerate(layers[k::n]):
                groups[i], buffers[i] = first + k, t
        first += n
    return groups, buffers


class EmbeddingRows:
    """An encoder backend: an image item's embeddings are the embedding table's rows of the token ids in column 0 of
    every fourth row of its pixel_values (one id per merged 2x2 patch)."""

    def __init__(self, embedding):
        self.embedding = embedding

    def embed_image(self, pixel_values, grid_thw):
        return self.embedding[pixel_values[::4, 0].long()]


def image_item(tag, offset, ids):
    """An image item at ``offset`` whose rows are the embeddings of ``ids`` (a 1 x 4 x len(ids) patch grid)."""
    n = ids.numel()
    pixels = torch.zeros(4 * n, 6)
    pixels[::4, 0] = ids.float()
    grid = torch.tensor([1, 4, n])
    return {
        "modality": "image",
        "identifier": f"img{tag}",
        "offset": offset,
        "length": n,
        "is_embed": None,
        "data": {"pixel_values": pixels, "image_grid_thw": grid},
    }


def twin(ids, a, b):
    """``ids`` with [a, b) replaced by placeholders, and the item that puts their embeddings back."""
    tokens = ids.clone()
    tokens[a:b] = IMAGE
    return tokens, image_item(f"{a}-{b}-{int(ids[a])}", a, ids[a:b])


@pytest.mark.parametrize("device_params", [{**GALAXY_DEVICE_PARAMS, "trace_region_size": 1_200_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
def test_mm_prefill_adapter_tp8(mesh_device, monkeypatch):
    if len([c for c in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if c]) != 8:
        pytest.skip("run on one row: TT_VISIBLE_DEVICES=<the row's 8 chips>")
    assert gv.MM_TRACE and gv.PREFILL_BATCH and gv.PREFILL_TRACE and gv.KV_PAIR_SPLIT and not gv.SPEC_DECODE
    path = default_model_path()
    cfg = MiMoConfig.from_model_path(path)
    groups, buffers = vllm_kv_layout(cfg)
    monkeypatch.setattr(MiMoV2ForCausalLM, "_layer_groups", groups)
    monkeypatch.setattr(MiMoV2ForCausalLM, "prefill_token_budget", BUDGET)

    # vLLM's KV: each slot its own global blocks (128 positions each) and a ring of SWA blocks per SWA group.
    global_blocks = -(-MAX_SEQ_LEN // (2 * BLOCK_SIZE))
    swa_groups = sorted({g for i, g in enumerate(groups) if cfg.is_swa(i)})
    per_request = global_blocks + len(swa_groups) * SWA_RING
    num_blocks = 1 + PER_LANE * per_request
    width = -(-MAX_SEQ_LEN // BLOCK_SIZE)
    tables = {g: torch.zeros(PER_LANE, width, dtype=torch.int32) for g in range(max(groups) + 1)}
    for u in range(PER_LANE):
        for g in tables:
            if g in swa_groups:
                base = 1 + u * per_request + global_blocks + swa_groups.index(g) * SWA_RING
                tables[g][u] = base + torch.arange(width, dtype=torch.int32) % SWA_RING
            else:
                tables[g][u, :global_blocks] = 1 + u * per_request + torch.arange(global_blocks, dtype=torch.int32)
    layer_tables = [tables[g] for g in groups]

    t0 = time.time()
    hf = SimpleNamespace(
        num_hidden_layers=cfg.num_hidden_layers,
        _name_or_path=path,
        image_token_id=IMAGE,
        vision_config={"spatial_merge_size": 2},
        eos_token_id=151645,
    )
    # The text adapter's construction, then the multimodal setup with the stand-in encoder (before any trace).
    model = MiMoV2ForCausalLM.initialize_vllm_model.__func__(
        MiMoV2OmniForCausalLM, hf, mesh_device, PER_LANE, MAX_SEQ_LEN, tt_data_parallel=1
    )
    ckpt = MiMoCheckpoint(path, cfg)
    model.setup_multimodal(path, hf, backends=[EmbeddingRows(ckpt.embedding().to(torch.bfloat16))])
    assert model._mm_traced()
    specs = []
    for i in range(cfg.num_hidden_layers):
        block = 2 * BLOCK_SIZE if not cfg.is_swa(i) else BLOCK_SIZE
        specs.append(((num_blocks, 1, block, cfg.attention_dims(i).head_dim), torch.bfloat16, buffers[i]))
    kv = model.allocate_kv_cache_per_layer(specs)
    model.warmup_model_prefill(kv)
    model.warmup_model_decode(kv, enable_trace=False)
    model.warmup_model_decode(kv, enable_trace=True)
    replica = model.models[0]
    optional = [k for k, t in replica.chunk_traces.items() if t.get("optional") and t["id"] is not None]
    logger.info(f"built and warmed up in {time.time() - t0:.0f}s; optional multimodal traces captured: {optional}")
    programs = replica.mesh_device.num_program_cache_entries()

    replays = []
    for name in ("_replay_prefill_batch", "_replay_prefill_chunk"):
        original = getattr(replica, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            replays.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(replica, name, counted)

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "tt", "model.py")) as f:
        text = torch.tensor(tok(f.read())["input_ids"], dtype=torch.int32)

    def prefill(slots, rows, starts, ends, items):
        replays.clear()
        width = max(r.numel() for r in rows)
        tokens = torch.stack([torch.nn.functional.pad(r, (0, width - r.numel())) for r in rows])
        logits = model.prefill_forward(
            tokens,
            torch.tensor(starts, dtype=torch.int32),
            torch.tensor(ends, dtype=torch.int32),
            tables[0][slots],
            kv,
            empty_slots=slots,
            page_tables_per_layer=[t[slots] for t in layer_tables],
            mm_items=items,
        )
        assert model.take_prefill_errors() == {}
        return logits[:, -1], list(replays)

    def check(what, logits, replayed, expect):
        diff = float((logits[0] - logits[1]).abs().max())
        logger.info(f"{what}: text / multimodal last-token logits max |diff| {diff}; replays {replayed}")
        assert torch.equal(logits[0], logits[1]), f"{what}: logits differ (max |diff| {diff})"
        assert replayed == expect, f"{what}: replays {replayed}, expected {expect}"

    # 1. One prefill batch of two 300-token first chunks.
    ids = text[1000:1300]
    mm, item = twin(ids, 40, 104)
    logits, replayed = prefill([0, 1], [ids, mm], [0, 0], [300, 300], [None, [item]])
    check("batch of first chunks", logits, replayed, ["_replay_prefill_batch"])

    # 2. Traced first chunks of 900 tokens (padded 1024).
    ids = text[2000:2900]
    mm, item = twin(ids, 100, 612)
    logits, replayed = prefill([2, 3], [ids, mm], [0, 0], [900, 900], [None, [item]])
    check("traced first chunks", logits, replayed, ["_replay_prefill_chunk"] * 2)

    # 3. An item straddling two steps: first chunks of 512 (a batch), then later chunks of 188 (traced, padded 256).
    ids = text[4000:4700]
    mm, item = twin(ids, 450, 578)
    logits, replayed = prefill([4, 5], [ids, mm], [0, 0], [512, 512], [None, [item]])
    check("item cut by the chunk end: first chunks", logits, replayed, ["_replay_prefill_batch"])
    logits, replayed = prefill([4, 5], [ids, mm], [512, 512], [700, 700], [None, [item]])
    check("item cut by the chunk end: later chunks", logits, replayed, ["_replay_prefill_chunk"] * 2)

    assert replica.mesh_device.num_program_cache_entries() == programs, "a prefill compiled programs while serving"
    assert model.decode_recaptures == 0
