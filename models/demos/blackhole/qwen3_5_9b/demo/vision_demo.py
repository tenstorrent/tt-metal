# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-9B end-to-end multimodal (vision + text) generation demo on Blackhole P150.

Mirrors the text-only ``text_demo.py`` flow, but routes images through the TT vision tower
(``DropInVisionTransformer``) and splices the resulting image embeddings into the text
embeddings before prefill. The vision merge is trace-safe: the image rows are placed into
persistent fixed-shape buffers (host-side, since the number of image tokens is dynamic) and
selected on device with a captured ``ttnn.where``, so multimodal works WITH the captured
prefill/decode traces — on a single device and on a TP mesh.

Covered cases (parametrized):
  - single image  : "Describe this image."          (traced + paged)
  - multi image   : "Identify the differences ..."  (traced)
  - text only     : vision tower skipped            (traced)

Multi-device (TP) is selected via MESH_DEVICE and uses the chunk-outer traced prefill +
paged traced decode path, exactly like the text demo.

Run all:    pytest models/demos/blackhole/qwen3_5_9b/demo/vision_demo.py -v -s
Run single: pytest models/demos/blackhole/qwen3_5_9b/demo/vision_demo.py -v -s -k "traced_single_image"
Run TP:     MESH_DEVICE=P150x4 pytest models/demos/blackhole/qwen3_5_9b/demo/vision_demo.py -v -s -k "traced_single_image"
"""

import json
import os
import time

import pytest
import torch
from loguru import logger
from qwen_vl_utils import process_vision_info

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model
from models.tt_transformers.tt.generator import Generator

# Multi-device (TP) is selected via MESH_DEVICE (e.g. P150x4). On a single device the mesh is
# (1,1) and the model runs its validated single-device path; on a multi-device mesh it needs
# FABRIC_1D for the TP collectives. The vision splice buffers are allocated on whichever mesh.
_MESH_SHAPE = {"N150": (1, 1), "N300": (1, 2), "P150x4": (1, 4), "N150x4": (1, 4), "T3K": (1, 8)}.get(
    os.environ.get("MESH_DEVICE"), (1, 1)
)
_MULTI = _MESH_SHAPE != (1, 1)

# Both single-device and TP traced paths capture a prefill chunk trace AND a paged decode trace,
# so a trace region is always required (ttnn's DEFAULT_TRACE_REGION_SIZE is 0). 256 MiB matches the
# validated serving config and is ample for the short multimodal prompts here.
_TRACE_REGION_SIZE = 256 * 1024 * 1024
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        "trace_region_size": _TRACE_REGION_SIZE,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if _MULTI else {}),
    }
]

SAMPLE_PROMPTS_DIR = "models/demos/blackhole/qwen3_5_9b/demo/sample_prompts"

BLOCK_SIZE = 64
PREFILL_CHUNK = 2048  # chunked-prefill chunk; prompts are processed in chunks of this size


def _load_conversation(prompt_file):
    """Load the first conversation (a list of message dicts) from a sample-prompt JSON file."""
    with open(prompt_file) as f:
        data = json.load(f)
    assert len(data) >= 1, f"{prompt_file} has no conversations"
    return data[0]


def _process_prompt(processor, messages):
    """Apply the chat template + processor to produce model inputs.

    Returns (input_ids [1,T], pixel_values or None, image_grid_thw or None, num_image_tokens).
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    input_ids = inputs["input_ids"]
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    num_image_tokens = 0
    if image_grid_thw is not None:
        merge_length = processor.image_processor.merge_size**2
        num_image_tokens = int(image_grid_thw.prod(dim=-1).sum().item() // merge_length)
    return input_ids, pixel_values, image_grid_thw, num_image_tokens


def _compute_vision_tokens(model, pixel_values, image_grid_thw):
    """Run the TT vision tower; returns packed image embeddings (ttnn) or None for text-only.

    Called BEFORE any prefill/decode trace is captured so the (eager) vision programs compile
    while no trace is parked — a request-time compile would clobber a parked trace.
    """
    if pixel_values is None:
        return None
    t0 = time.time()
    vision_tokens = model.get_image_features(pixel_values, image_grid_thw)
    ttnn.synchronize_device(model.mesh_device)
    logger.info(f"Vision tower: {time.time() - t0:.2f}s for {int(vision_tokens.shape[0])} image tokens")
    return vision_tokens


def _blocks_for(seqlen, max_generated_tokens, max_seq_len):
    """Paged-KV block budget for `seqlen`, sized to hold the padded prefill bucket plus the tokens
    we decode, capped at the model's configured context (max_seq_len)."""
    bucket = ((seqlen + PREFILL_CHUNK - 1) // PREFILL_CHUNK) * PREFILL_CHUNK
    needed = bucket + max_generated_tokens
    blocks = max(64, (needed + BLOCK_SIZE - 1) // BLOCK_SIZE)
    # Flexible chunked SDPA reads the page table as a ROW_MAJOR int32 stick; the program factory
    # requires num_blocks % 8 == 0. Round up (only enlarges the cache by <=7 blocks).
    blocks = ((blocks + 7) // 8) * 8
    return min(max_seq_len // BLOCK_SIZE, blocks)


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "prompt_file, use_trace, max_generated_tokens, max_seq_len",
    [
        ("vision_demo.json", True, 300, 8192),
        ("vision_demo.json", False, 300, 8192),
        ("vision_multi_image.json", True, 300, 8192),
        ("vision_text_only.json", True, 100, 4096),
    ],
    ids=[
        "traced_single_image",
        "paged_single_image",
        "traced_multi_image",
        "traced_text_only",
    ],
)
def test_demo_vision(mesh_device, prompt_file, use_trace, max_generated_tokens, max_seq_len):
    """End-to-end multimodal generation: vision tower -> embedding splice -> prefill -> decode."""
    from transformers import AutoProcessor, AutoTokenizer

    device = mesh_device
    device.enable_program_cache()

    t0 = time.time()
    model = Qwen35Model.from_pretrained(
        device,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        # n_layers=4,  # uncomment for fast iteration; default uses the full config
    )
    logger.info(f"Text model load: {time.time() - t0:.1f}s")

    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)

    # Build + attach the TT vision tower (loads the HF reference visual for patch-embed/indexing).
    t0 = time.time()
    model.init_vision_model()
    logger.info(f"Vision model init: {time.time() - t0:.1f}s")

    # ---- Preprocess the prompt (chat template + image patchify) ----
    messages = _load_conversation(f"{SAMPLE_PROMPTS_DIR}/{prompt_file}")
    token_ids, pixel_values, image_grid_thw, num_image_tokens = _process_prompt(processor, messages)
    T = token_ids.shape[1]
    logger.info(f"Prompt: {T} tokens ({num_image_tokens} image tokens), prompt_file={prompt_file}")
    assert T <= max_seq_len, f"prompt {T} tokens exceeds max_seq_len {max_seq_len}"

    num_blocks = _blocks_for(T, max_generated_tokens, max_seq_len)

    if model.num_devices > 1:
        generated, perf = _run_tp_vision_generation(
            model, tokenizer, token_ids, pixel_values, image_grid_thw, max_generated_tokens, num_blocks
        )
    elif use_trace:
        generated, perf = _run_traced_vision_generation(
            model, tokenizer, device, token_ids, pixel_values, image_grid_thw, max_generated_tokens, num_blocks
        )
    else:
        generated, perf = _run_paged_vision_generation(
            model, tokenizer, device, token_ids, pixel_values, image_grid_thw, max_generated_tokens, num_blocks
        )

    text = tokenizer.decode(generated, skip_special_tokens=True)
    logger.info("=" * 70)
    logger.info(f"  TTFT: {perf['ttft']:.3f}s   Decode: {perf['decode_tok_s']:.1f} tok/s")
    logger.info(f"  Generated {len(generated)} tokens")
    logger.info(f"  OUTPUT: {text}")
    logger.info("=" * 70)

    assert len(generated) >= 1, "should generate at least 1 token"
    assert len(set(generated)) > 1, f"degenerate generation: {generated}"


def _run_traced_vision_generation(
    model, tokenizer, device, token_ids, pixel_values, image_grid_thw, max_generated_tokens, num_blocks
):
    """Single-device traced prefill (chunk-outer) + paged traced decode, with the vision splice.

    The vision tower runs first (so its eager programs compile before any trace is parked), then
    we capture the per-chunk prefill trace and replay it; ``prefill_traced_chunked`` stages the
    image rows into the persistent splice buffers per chunk via host->device copies only.
    """
    from models.demos.blackhole.qwen3_5_9b.tt.generator_interface import prime_decode_trace

    T = token_ids.shape[1]
    kv_cache_shape = [num_blocks, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    # Vision tower BEFORE trace capture (its compile must not clobber a parked trace).
    t = time.time()
    vision_tokens = _compute_vision_tokens(model, pixel_values, image_grid_thw)
    t_vis = time.time() - t

    # Capture the per-chunk prefill trace (warmup; allocates the vision splice buffers). Warm ALL
    # masked buckets: a short prompt (T < chunk) runs entirely through a masked bucket, and a long
    # prompt's tail (T > chunk) also rounds up to one — warming them up front keeps either from
    # compiling a program at request time and clobbering the parked chunk trace.
    t_cap = time.time()
    model.capture_prefill_trace_chunked(device, page_table, chunk_size=PREFILL_CHUNK, warmup_masked_buckets=True)
    logger.info(f"Prefill trace captured in {time.time() - t_cap:.1f}s")

    t0 = time.time()
    logits = model.prefill_traced_chunked(token_ids, page_table, actual_len=T, vision_tokens=vision_tokens)
    ttnn.synchronize_device(device)
    ttft = time.time() - t0 + t_vis

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in prefill logits"
    next_token = int(logits_torch.argmax())

    # Traced paged decode (GDN-state save/restore happens inside prime_decode_trace).
    gen = Generator([model], [model.args], device)
    prime_decode_trace(gen, model, torch.tensor([[next_token]], dtype=torch.long), torch.tensor([T]), page_table)

    generated = [next_token]
    decode_times = []
    current_pos = T
    for _ in range(max_generated_tokens - 1):
        t_step = time.time()
        out = gen.decode_forward(
            torch.tensor([[next_token]], dtype=torch.long),
            torch.tensor([current_pos]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=True,
            read_from_device=True,
        )
        decode_times.append(time.time() - t_step)
        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        assert not torch.isnan(dl).any(), "NaN in traced decode logits"
        next_token = int(dl.argmax())
        generated.append(next_token)
        current_pos += 1
        if next_token == tokenizer.eos_token_id:
            break

    return generated, _perf(ttft, decode_times)


def _run_paged_vision_generation(
    model, tokenizer, device, token_ids, pixel_values, image_grid_thw, max_generated_tokens, num_blocks
):
    """Single-device non-traced paged prefill + decode, with the vision splice.

    No trace is parked here, so the prefill uses the on-device scatter path (``prefill_paged``
    with ``vision_tokens``). Useful as a correctness oracle for the traced path.
    """
    T = token_ids.shape[1]
    kv_cache_shape = [num_blocks, model.args.n_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    vision_tokens = _compute_vision_tokens(model, pixel_values, image_grid_thw)

    t0 = time.time()
    logits = model.prefill_paged(token_ids, page_table, vision_tokens=vision_tokens)
    ttnn.synchronize_device(device)
    ttft = time.time() - t0

    logits_torch = ttnn.to_torch(logits).squeeze()
    assert not torch.isnan(logits_torch).any(), "NaN in paged prefill logits"
    next_token = int(logits_torch.argmax())

    gen = Generator([model], [model.args], device)
    generated = [next_token]
    decode_times = []
    for i in range(max_generated_tokens - 1):
        t_step = time.time()
        out = gen.decode_forward(
            torch.tensor([[next_token]], dtype=torch.long),
            torch.tensor([T + i]),
            page_table=page_table,
            kv_cache=None,
            enable_trace=False,
            read_from_device=True,
        )
        decode_times.append(time.time() - t_step)
        dl = (out[0] if isinstance(out, tuple) else out).squeeze().float()
        assert not torch.isnan(dl).any(), "NaN in paged decode logits"
        next_token = int(dl.argmax())
        if next_token == tokenizer.eos_token_id:
            break
        generated.append(next_token)

    return generated, _perf(ttft, decode_times)


def _run_tp_vision_generation(
    model, tokenizer, token_ids, pixel_values, image_grid_thw, max_generated_tokens, num_blocks
):
    """Multi-device (TP) traced chunk-outer prefill + paged traced decode, with the vision splice.

    Mirrors text_demo._run_tp_generation: prefill captures ONE chunk's all-layer forward and
    replays it per chunk while the GDN recurrent/conv state and paged KV carry across chunks;
    decode is captured once and replayed with a post-prefill GDN snapshot/restore. The only
    multimodal addition is computing the (hidden-fractured) image embeddings up front and passing
    them to ``prefill_traced_chunked``, which gathers/re-shards them into the splice buffers.
    """
    from models.tt_transformers.tt.common import copy_host_to_device

    vocab = model.args.vocab_size
    mesh = model.mesh_device
    T = token_ids.shape[1]

    # Flexible chunked SDPA requires the page-table width to be a multiple of 32.
    num_blocks = ((num_blocks + 31) // 32) * 32
    kv_cache_shape = [num_blocks, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)

    # Vision tower BEFORE trace capture (its eager compile must not clobber a parked trace).
    vision_tokens = _compute_vision_tokens(model, pixel_values, image_grid_thw)

    t_cap = time.time()
    model.capture_prefill_trace_chunked(mesh, page_table, chunk_size=PREFILL_CHUNK)
    logger.info(f"[TP] prefill chunk-trace captured in {time.time() - t_cap:.1f}s")

    t0 = time.time()
    logits_dev = model.prefill_traced_chunked(token_ids, page_table, actual_len=T, vision_tokens=vision_tokens)
    ttnn.synchronize_device(mesh)
    ttft = time.time() - t0

    def _pick(vec):
        return int(torch.argmax(vec.float()).item())

    # Logits are replicated across the mesh ([1,1,vocab]); gather one replica.
    lt = ttnn.to_torch(logits_dev, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    nxt = _pick(lt.reshape(-1, vocab)[0])
    generated = [nxt]

    # ---- Traced paged decode with GDN snapshot/restore (see text_demo for the rationale). ----
    _gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]

    def _snapshot_gdn():
        comp = ttnn.ConcatMeshToTensor(mesh, dim=0)
        return [
            (
                ttnn.to_torch(dn.rec_state, mesh_composer=comp),
                [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states],
            )
            for dn in _gdn
        ]

    def _restore_gdn(snap):
        mapper = ttnn.ShardTensorToMesh(mesh, dim=0)

        def _back(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)

        for dn, (rec, convs) in zip(_gdn, snap):
            r = _back(rec, dn.rec_state.dtype)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = _back(c, dn.conv_states[j].dtype)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)

    dev = model.prepare_inputs_decode(
        torch.tensor([[nxt]], dtype=torch.int32), torch.tensor([T], dtype=torch.int32), page_table=page_table
    )

    gdn_snap = _snapshot_gdn()  # exact post-prefill GDN state
    # Compile decode programs (eager) then capture a throwaway trace; both advance GDN state.
    model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    tt_logits, _ = model.ttnn_decode_forward(dev[0], dev[1], rot_mat_idxs=dev[2], page_table=dev[3])
    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
    _restore_gdn(gdn_snap)

    def _update(token, position):
        host = model.prepare_decode_inputs_host(
            torch.tensor([[token]], dtype=torch.int32),
            torch.tensor([position], dtype=torch.int32),
            page_table=page_table,
        )
        copy_host_to_device(host, device_tensors=dev)

    pos = T
    decode_times = []
    while len(generated) < max_generated_tokens:
        _update(nxt, pos)
        t_step = time.time()
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        decode_times.append(time.time() - t_step)
        nxt = _pick(model.process_output_decode(tt_logits, B=1, S=1).reshape(-1)[:vocab])
        generated.append(nxt)
        pos += 1
        if nxt == tokenizer.eos_token_id:
            break
    ttnn.release_trace(mesh, trace_id)

    return generated, _perf(ttft, decode_times)


def _perf(ttft, decode_times):
    # Steady-state throughput (drop the first step, which can carry one-time costs).
    steady = decode_times[1:] if len(decode_times) > 1 else decode_times
    avg = (sum(steady) / len(steady)) if steady else float("inf")
    return {"ttft": ttft, "decode_tok_s": (1.0 / avg) if avg > 0 else 0.0, "decode_steps": len(decode_times)}
