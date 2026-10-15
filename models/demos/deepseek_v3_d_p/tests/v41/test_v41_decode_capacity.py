# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The batched decode step at long pool capacities: the whole model, 32 users joined from short prompts, then the
pools filled with synthetic keys up to the capacity and the users moved to positions near it, so the step's
indexer (the dense scores, the candidate selection, the candidate-row scores), sparse attention and writes run at
``V41_CAP_SEQ`` of context. Timing scope only: the keys are not a real history, so the text is meaningless; the
correctness of the selection is ``test_indexer.py::test_v41_decode_selection`` and of the step
``test_v41_batch_decode.py``.

Records (``V41_CAP_RESULT``, ``V41_CAP_OUT``): per step the device time of the replayed capture, the allocator's
DRAM after the build, the join, the fill, the capture and the replays, the trace region's use, and, with
``V41_CAP_GRAPH=1``, one eager step under ``ttnn.graph`` capture: every op whose inputs or outputs hold at least
``V41_CAP_GRAPH_MB`` (64) MB, so a conversion of the history shows up by name.
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_depth import _dram_bytes, _rss_gb, parse_layers
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import TtV41BatchDecoder
from models.demos.deepseek_v3_d_p.tt.v41.engram_checkpoint import build_engram_modules
from models.demos.deepseek_v3_d_p.tt.v41.pool import BLOCK_ROWS
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

TRACE_REGION = 2 * 1024**3
FILL_ROWS = 4096  # rows of the random block the pools are filled from


def _trace_bytes(mesh_device) -> int | None:
    try:
        view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.TRACE)
        return int(view.total_bytes_allocated_per_bank * view.num_banks)
    except Exception as e:
        logger.warning(f"trace statistics unavailable: {e}")
        return None


def _fill(mesh_device, dst, block_rows: int, g, layout) -> int:
    """Overwrite every row of ``dst`` ([1, 1, H, W] or [U, 1, H, W], a multiple of ``block_rows`` tall) with copies
    of one random bf16 block of ``block_rows`` rows (uploaded once; ``slice_write`` over the rest) -> writes."""
    shape = list(dst.shape)
    w = shape[3]
    block = ttnn.from_torch(
        (torch.randn(1, 1, block_rows, w, generator=g) * 0.5).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=layout,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    writes = 0
    for u in range(shape[0]):
        for first in range(0, shape[2], block_rows):
            n = min(block_rows, shape[2] - first)
            piece = block if n == block_rows else ttnn.slice(block, [0, 0, 0, 0], [1, 1, n, w])
            ttnn.experimental.slice_write(piece, dst, [u, 0, first, 0], [u + 1, 1, first + n, w], [1, 1, 1, 1])
            writes += 1
    ttnn.deallocate(block)
    return writes


@pytest.mark.timeout(int(os.environ.get("V41_CAP_TIMEOUT", 3 * 3600)))
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_decode_capacity(mesh_device, device_params):
    layers = parse_layers(os.environ.get("V41_CAP_LAYERS", "all"))
    cap = int(os.environ.get("V41_CAP_SEQ", 1048576))  # the pool's positions per user
    steps = int(os.environ.get("V41_CAP_STEPS", 12))
    chunk = int(os.environ.get("V41_CAP_CHUNK", 1024))
    users = int(os.environ.get("V41_CAP_USERS", 4))
    base = int(os.environ.get("V41_CAP_PROMPT", 255))
    warmup = int(os.environ.get("V41_CAP_WARMUP", 2))
    fill = os.environ.get("V41_CAP_FILL", "1") == "1"
    graph = os.environ.get("V41_CAP_GRAPH", "0") == "1"
    graph_mb = int(os.environ.get("V41_CAP_GRAPH_MB", 64))
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    from transformers import AutoTokenizer

    sp, tp = tuple(mesh_device.shape)
    n_slots = sp * users
    lengths = [base - (s * 5) % 32 for s in range(n_slots)]
    stride = base + steps + 8
    text = orc.text_tokens(n_slots * stride)[0]
    prompts = {s: text[s * stride : s * stride + lengths[s]] for s in range(n_slots)}
    books = {s: text[s * stride + lengths[s] : s * stride + lengths[s] + steps + 1] for s in range(n_slots)}
    max_seq = -(-(base + steps + 32) // chunk) * chunk  # the model's own states: short
    assert cap % chunk == 0 and cap >= max_seq, (cap, chunk, max_seq)
    result = {
        "layers": layers,
        "n_layers": len(layers),
        "slots": n_slots,
        "users_per_rank": users,
        "capacity": cap,
        "chunk": chunk,
        "model_max_seq": max_seq,
        "steps": steps,
        "warmup": warmup,
        "fill": fill,
        "mesh": [sp, tp],
        "fabric": str(device_params.get("fabric_config")),
    }
    spec = orc.real_spec(tuple(layers), max_seq, checkpoint=ckpt.root)
    root = weight_cache_dir(spec, mesh_device.shape)
    tokenizer = AutoTokenizer.from_pretrained(orc.HF_SNAPSHOT)

    t0 = time.time()
    engram, engram_hash = build_engram_modules(mesh_device, C, ckpt, layers, tokenizer)
    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    model = TtV41Transformer(
        mesh_device,
        C,
        layers,
        lambda layer, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, layer),
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=max_seq,
        chunk=chunk,
        engram=engram,
        engram_hash=engram_hash,
        weight_cache_path=root,
    )
    result["dram_after_model_bytes"] = _dram_bytes(mesh_device)
    decoder = TtV41BatchDecoder(
        model, root, users_per_rank=users, trace=True, warmup=warmup, sample_on_device=True, max_seq_len=cap
    )
    result["build_s"] = round(time.time() - t0, 1)
    result["dram_after_build_bytes"] = _dram_bytes(mesh_device)
    pool = decoder.pool
    result["pool"] = {
        str(src): {"cap": slot.cap, "rows": slot.rows, "layers": list(slot.layers)} for src, slot in pool.slots.items()
    }

    t0 = time.time()
    for s in range(n_slots):
        _, state = model.prefill(prompts[s], logit_positions=1)
        decoder.join(s, prompts[s], state)
        for group in (state.kv, state.index_k, state.window_carry):
            for t in group.values():
                ttnn.deallocate(t)
        if state.swa_scratch is not None:
            ttnn.deallocate(state.swa_scratch)
        for carry in state.carry_buffers.values():
            for t in carry:
                ttnn.deallocate(t)
    result["prefill_join_s"] = round(time.time() - t0, 1)
    result["dram_after_join_bytes"] = _dram_bytes(mesh_device)

    # --- the synthetic history: random keys over every pool, the users at positions near the capacity ---
    if fill:
        t0 = time.time()
        g = torch.Generator().manual_seed(cap)
        writes = 0
        for src, t in pool.index_k.items():
            writes += _fill(mesh_device, t, FILL_ROWS, g, ttnn.TILE_LAYOUT)
        for src, t in pool.index_k_blocks.items():
            writes += _fill(mesh_device, t, FILL_ROWS // BLOCK_ROWS, g, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.synchronize_device(mesh_device)
        result["fill_s"] = round(time.time() - t0, 1)
        result["fill_writes"] = writes
        # (the compressed KV rows stay as the join left them: zeros past the prompt; the attention reads the same
        # bytes either way)
    positions = {}
    for s in range(n_slots):
        # near the capacity, both parities, distinct; the step writes at position // ratio and must stay below cap
        positions[s] = cap - 64 - s
        decoder.slots[s].position = positions[s]
    result["positions"] = {str(s): positions[s] for s in (0, n_slots - 1)}
    result["dram_after_fill_bytes"] = _dram_bytes(mesh_device)

    # --- the steps: warm-up eager, one capture, replays ---
    per_step = []
    for i in range(steps):
        tokens = {s: int(books[s][i]) for s in range(n_slots)}
        t0 = time.time()
        if graph and i == 0:
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        decoder.decode(tokens, {s: (1, 0.0, 0.0) for s in tokens})
        if graph and i == 0:
            captured = ttnn.graph.end_graph_capture()
            result["graph_big_ops"] = _big_ops(captured, graph_mb << 20)
            result["graph_ops"] = sum(1 for n in captured if n.get("node_type") == "function_start")
        wall = time.time() - t0
        entry = {"step": i, "wall_s": round(wall, 3), "mode": decoder.last.get("mode")}
        entry.update({k: round(v, 4) for k, v in decoder.last.items() if isinstance(v, float)})
        if i == warmup:
            result["dram_after_capture_bytes"] = _dram_bytes(mesh_device)
            result["trace_after_capture_bytes"] = _trace_bytes(mesh_device)
        per_step.append(entry)
        logger.info(f"capacity step {i}: {json.dumps(entry)}")
    result["steps_detail"] = per_step
    replays = [e for e in per_step if e["mode"] == "replay"]
    result["replay_steps"] = len(replays)
    if replays:
        result["replay_wall_min_s"] = min(e["wall_s"] for e in replays)
        result["replay_device_s"] = sorted(e["device_s"] for e in replays)
        result["replay_device_min_s"] = result["replay_device_s"][0]
        result["replay_device_median_s"] = result["replay_device_s"][len(replays) // 2]
        result["replay_device_only_s"] = decoder.replay_device_time(repeats=7)
    result["dram_after_decode_bytes"] = _dram_bytes(mesh_device)
    result["trace_after_decode_bytes"] = _trace_bytes(mesh_device)
    decoder.release()
    result["rss_peak_gb"] = round(_rss_gb(), 1)
    logger.info("V41_CAP_RESULT " + json.dumps(result))
    out_path = os.environ.get("V41_CAP_OUT")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=1)


def _tensor_bytes(info) -> int:
    """Bytes of a captured graph node's tensor description (shape x dtype), 0 if unknown."""
    try:
        shape = info.get("shape") or info.get("params", {}).get("shape")
        if isinstance(shape, str):
            shape = [int(x) for x in shape.strip("[]()").replace("Shape", "").strip("()[] ").split(",") if x.strip()]
        n = 1
        for d in shape:
            n *= int(d)
        return n * 2
    except Exception:
        return 0


def _big_ops(captured, threshold: int) -> list:
    """The captured graph's ops (``function_start`` nodes) with any tensor of at least ``threshold`` bytes among
    their arguments, as (name, [bytes...]): a conversion of a pool shows up here."""
    out = []
    for node in captured:
        if node.get("node_type") != "function_start":
            continue
        name = node.get("params", {}).get("name", "?")
        sizes = []
        for arg in node.get("arguments", []):
            if isinstance(arg, dict):
                b = _tensor_bytes(arg)
                if b >= threshold:
                    sizes.append(b)
            elif isinstance(arg, str) and "Shape" in arg:
                b = _tensor_bytes({"shape": arg[arg.index("Shape") :].split(")")[0]})
                if b >= threshold:
                    sizes.append(b)
        if sizes:
            out.append([name, sizes])
    return out
