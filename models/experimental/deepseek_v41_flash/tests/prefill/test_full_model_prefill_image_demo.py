# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Long-prompt prefill demo with an image: ``test_full_model_prefill_demo.py``'s traced, pipelined prefill of the full
40-layer ``DeepSeekV41PrefillModel`` (with the head), the prompt holding an image after the book.

Opens the full 8 x 4 Galaxy mesh and carves ``1 x 4`` submeshes out of it as the text demo does; the image encoder
(``DeepSeekV41Vision``, ViT + aligner, TP4) runs on stage ``DEEPSEEK_V41_VISION_STAGE`` (default 5), as in the server.

Each prompt is one user turn rendered by the checkpoint's encoding (``encoding_dsv41.encode_messages``)::

    ```<the book, cut or repeated to fill>```

    <the image>

    What digit is written in the image? Reply with the digit only.

and is ``DEEPSEEK_V4_PREFILL_LEN`` tokens long. The image is synthetic, a large digit on a plain background, so the
answer is known: the first generated token must be that digit. Two images with different digits run through the same
captured traces, so the answer must come from the image.

An image reaches the LLM as the server feeds it (see ``tt/vision.py``): the model is built with ``image_rows`` extra
rows in its embedding table, the image's span (start, rows of patches each ended by a newline, end) is token ids past
the vocabulary, and the tower's rows are written there. Prefill reads them like any token's; every layer derives the
image mask from the ids and routes image tokens with ``bias_vl``; the Engram lookup skips them. As in the server, the
tower is built before the traces are captured and an image is encoded (eagerly) after, between prefills.

Run it (ttnn venv)::

    DEEPSEEK_V4_CACHE_DIR=/path/to/cache pytest -s \\
      models/experimental/deepseek_v41_flash/tests/prefill/test_full_model_prefill_image_demo.py

Knobs (environment): those of ``test_full_model_prefill_demo.py`` (``DEEPSEEK_V4_PREFILL_LEN`` default 16384,
``DEEPSEEK_V4_PREFILL_CHUNK``, ``DEEPSEEK_V4_PREFILL_LAYERS``, ``DEEPSEEK_V4_NUM_STAGES``,
``DEEPSEEK_V4_TRACE_REGION_SIZE``, ``DEEPSEEK_V4_PREFILL_HEARTBEAT``, ``DEEPSEEK_V4_PREFILL_STALL_SECS``,
``DEEPSEEK_V4_PREFILL_PROMPT``, ``DEEPSEEK_V4_CACHE_DIR``) except ``DEEPSEEK_V4_PREFILL_KV_ONLY`` (the head is always
built), plus ``DEEPSEEK_V41_VISION_STAGE`` and ``DEEPSEEK_V4_PREFILL_IMAGE_SIZE`` (the images' side in pixels, default
588: 42 x 42 patches, a span of 212 tokens). With fewer than all 40 layers the answer is not checked.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

import ttnn
from models.experimental.deepseek_v41_flash import encoding_dsv41
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _ENGRAM_RAM_MARGIN,
    _available_bytes,
    _build_rope,
    _checkpoint_available,
)
from models.experimental.deepseek_v41_flash.tests.prefill.test_full_model_prefill_demo import (
    _ATTENTION_WEIGHT_DTYPE,
    _EXPERT_DTYPE,
    _NUM_STAGES,
    _TP_SIZE,
)
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    VISION_STAGE,
    load_config,
    load_vision_config,
)
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    EngramHostLookup,
    engram_ram_needed,
    load_engram_table,
)
from models.experimental.deepseek_v41_flash.tt.prefill.model import DeepSeekV41PrefillModel
from models.experimental.deepseek_v41_flash.tt.vision import (
    DELIMITER_ROWS,
    FIRST_IMAGE_ROW,
    IMAGE,
    DeepSeekV41Vision,
    image_processor,
    span_ids,
    write_embedding_rows,
)
from models.experimental.deepseek_v4_flash.tests.prefill.test_full_model_prefill_demo import (
    _DEFAULT_PROMPT_FILE,
    _Progress,
    _env_int,
    _load_context,
    _report,
)
from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
from models.experimental.deepseek_v4_flash.tt.prefill.attention import ALIGNMENT
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

_QUESTION = "What digit is written in the image? Reply with the digit only."
# (digit, background, foreground): different digits, so the answers tell the images apart.
_IMAGES = (("7", (200, 30, 30), (255, 255, 255)), ("3", (30, 60, 200), (255, 220, 0)))
_SPLIT_SENTINEL = "@@BOOK@@"


def _digit_png(digit: str, background, foreground, size: int) -> bytes:
    """A ``size`` x ``size`` PNG of ``digit``, three quarters of the image tall, centred on a plain background."""
    image = Image.new("RGB", (size, size), background)
    ImageDraw.Draw(image).text(
        (size / 2, size / 2), digit, fill=foreground, font=ImageFont.load_default(size=size * 3 // 4), anchor="mm"
    )
    png = io.BytesIO()
    image.save(png, format="PNG")
    return png.getvalue()


def _image_prompt_ids(tokenizer, png: bytes, span: list[int], book: list[int], target_len: int, image_token_id: int):
    """The ids of the chat-templated user turn ``book, image, question``, exactly ``target_len`` tokens long: the
    image's placeholder replaced by its ``span`` ids, the book (``book``, token ids) cut or repeated to fill. Returns
    the ids and where the span starts."""
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    content = [
        {"type": "text", "text": "```" + _SPLIT_SENTINEL + "```"},
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": _QUESTION},
    ]
    prompt = encoding_dsv41.encode_messages([{"role": "user", "content": content}], "chat")
    head_text, tail_text = prompt.split(_SPLIT_SENTINEL)
    head = list(tokenizer(head_text, add_special_tokens=False)["input_ids"])
    tail = list(tokenizer(tail_text, add_special_tokens=False)["input_ids"])
    if tail.count(image_token_id) != 1:
        raise ValueError(f"the rendered tail holds {tail.count(image_token_id)} image placeholders, expected 1")
    at = tail.index(image_token_id)
    tail = tail[:at] + span + tail[at + 1 :]
    budget = target_len - len(head) - len(tail)
    if budget <= 0:
        raise ValueError(f"target length {target_len} leaves no room for the book around a {len(span)}-token image")
    context = (book * math.ceil(budget / len(book)))[:budget]
    return head + context + tail, len(head) + budget + at


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(21600)  # weight conversion on a cold cache + two long prefills
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            **(
                {"trace_region_size": int(os.environ["DEEPSEEK_V4_TRACE_REGION_SIZE"])}
                if os.environ.get("DEEPSEEK_V4_TRACE_REGION_SIZE")
                else {}
            ),
        }
    ],
    indirect=["device_params"],
    ids=["fabric_2d"],
)
@pytest.mark.parametrize("mesh_device", [(8, _TP_SIZE)], indirect=["mesh_device"], ids=["galaxy_8x4"])
def test_full_model_prefill_image_demo(mesh_device, reset_seeds) -> None:
    from transformers import AutoTokenizer

    progress = _Progress(
        interval=float(os.environ.get("DEEPSEEK_V4_PREFILL_HEARTBEAT", 30)),
        stall=float(os.environ.get("DEEPSEEK_V4_PREFILL_STALL_SECS", 600)),
    )
    with progress:
        _run_demo(mesh_device, progress, AutoTokenizer)


def _run_demo(mesh_device, progress: _Progress, AutoTokenizer) -> None:
    target_len = _env_int("DEEPSEEK_V4_PREFILL_LEN", 16384)
    chunk_size = _env_int("DEEPSEEK_V4_PREFILL_CHUNK", 1024)
    image_size = _env_int("DEEPSEEK_V4_PREFILL_IMAGE_SIZE", 588)
    for name, value in (("DEEPSEEK_V4_PREFILL_LEN", target_len), ("DEEPSEEK_V4_PREFILL_CHUNK", chunk_size)):
        if value <= 0 or value % ALIGNMENT:
            raise ValueError(f"{name}={value} must be a positive multiple of {ALIGNMENT}")
    logger.info(
        f"settings: prompt {target_len} tokens with a {image_size} x {image_size} image, chunk {chunk_size}, traced, "
        f"heartbeat every {progress.interval:.0f}s, stall warning after {progress.stall:.0f}s"
    )

    progress.step("[1/8] reading the checkpoint index, the configs and the tokenizer")
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    vision_config = load_vision_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    num_layers = min(_env_int("DEEPSEEK_V4_PREFILL_LAYERS", config.num_hidden_layers), config.num_hidden_layers)
    logger.info(f"checkpoint {loader.snapshot_dir}: {num_layers}/{config.num_hidden_layers} layers will be built")

    engram_ids = [i for i in config.engram_layer_ids if i < num_layers]
    if engram_ids:
        need = _ENGRAM_RAM_MARGIN * engram_ram_needed(loader, engram_ids)
        available = _available_bytes()
        if available < need:
            pytest.skip(
                f"Engram tables for layers {engram_ids} need {need / 1e9:.0f} GB with headroom; "
                f"{available / 1e9:.0f} GB is available"
            )

    # --- the images (host): ViT patches, the span layout, slots in the image rows ------------------------- #
    progress.step(f"[2/8] {len(_IMAGES)} images -> ViT patches and span ids; the {target_len}-token prompts")
    processor = image_processor(loader.snapshot_dir)
    processor_args = SimpleNamespace(
        vision_patch_size=vision_config.patch_size,
        vision_downsample_ratio=vision_config.downsample_ratio,
        vision_max_n_token=vision_config.max_image_tokens,
        vision_min_pixels=vision_config.min_pixels,
        vision_max_wh_ratio=vision_config.max_wh_ratio,
    )
    prompt_file = Path(os.environ.get("DEEPSEEK_V4_PREFILL_PROMPT", _DEFAULT_PROMPT_FILE))
    entry = json.loads(prompt_file.read_text())[0]
    if "context" not in entry:
        raise ValueError(f"{prompt_file.name}'s first entry has no context (a book URL) to fill the prompt with")
    book = list(tokenizer(_load_context(entry["context"]), add_special_tokens=False)["input_ids"])
    images = []
    next_slot = 0
    for digit, background, foreground in _IMAGES:
        png = _digit_png(digit, background, foreground, image_size)
        patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = processor.load_image({"data": png}, processor_args)
        types = processor.image_token_types(n_llm_h, n_llm_w)
        n = int((types == IMAGE).sum())
        slots = torch.arange(next_slot, next_slot + n)
        next_slot += n
        span = span_ids(types, slots, config.vocab_size).tolist()
        ids, image_start = _image_prompt_ids(tokenizer, png, span, book, target_len, vision_config.image_token_id)
        assert len(ids) == target_len
        first_chunk, last_chunk = image_start // chunk_size, (image_start + len(span) - 1) // chunk_size
        logger.info(
            f"image {digit!r}: {n_vit_h} x {n_vit_w} patches -> {n_llm_h} x {n_llm_w} tokens, a span of {len(span)} "
            f"at [{image_start}, {image_start + len(span)}) (chunk{'s' if last_chunk > first_chunk else ''} "
            f"{first_chunk + 1}{f'-{last_chunk + 1}' if last_chunk > first_chunk else ''}); prompt ends "
            f"{tokenizer.decode(ids[image_start + len(span):])!r}"
        )
        images.append(SimpleNamespace(digit=digit, patches=patches, n_h=n_vit_h, n_w=n_vit_w, slots=slots, ids=ids))
    image_rows = ttnn.TILE_SIZE * math.ceil((FIRST_IMAGE_ROW + next_slot) / ttnn.TILE_SIZE)
    num_chunks = math.ceil(target_len / chunk_size)
    logger.info(f"{image_rows} image rows in the embedding table; prefill plan: {num_chunks} chunks of {chunk_size}")

    # --- the mesh: the full Galaxy is open; each pipeline stage is a 1 x TP submesh of it -------- #
    progress.step("[3/8] system profile and submeshes")
    system_config = load_system_config(mesh_device=mesh_device).log()
    set_active_system_config(system_config)
    mesh_rows, mesh_cols = tuple(mesh_device.shape)
    assert mesh_cols >= _TP_SIZE, f"a 1 x {_TP_SIZE} submesh does not fit in the {mesh_rows} x {mesh_cols} mesh"
    num_stages = _NUM_STAGES
    if not 1 <= num_stages <= mesh_rows:
        raise ValueError(f"DEEPSEEK_V4_NUM_STAGES={num_stages} must be in [1, {mesh_rows}] (mesh rows)")
    if not 0 <= VISION_STAGE < mesh_rows:
        raise ValueError(f"DEEPSEEK_V41_VISION_STAGE={VISION_STAGE} must be in [0, {mesh_rows}) (mesh rows)")
    placement = plan_layer_placement(num_layers, num_stages, 1)
    num_stages = placement[-1] + 1
    submeshes = []
    for i in range(max(num_stages, VISION_STAGE + 1)):
        submeshes.append(mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(i, 0)))
        logger.info(f"created submesh for stage {i}: 1 x {_TP_SIZE} at mesh row {i}, column 0")
    layer_devices = [submeshes[k] for k in placement]
    spans = []
    for k in dict.fromkeys(placement):
        owned = [li for li, stage in enumerate(placement) if stage == k]
        spans.append(f"stage {k}: layers {owned[0]}-{owned[-1]}")
    logger.info(
        f"{num_layers}/{config.num_hidden_layers} layers over {num_stages} stages x TP{_TP_SIZE}: "
        + ", ".join(spans)
        + f"; the image encoder on stage {VISION_STAGE}"
    )

    progress.step(f"[4/8] RoPE tables for {target_len} positions")
    t0 = time.perf_counter()
    rope = _build_rope(config, loader.snapshot_dir, target_len)
    logger.info(f"RoPE tables built in {time.perf_counter() - t0:.1f}s")

    engram_lookup = None
    if engram_ids:
        progress.step(f"[5/8] Engram tables of layers {engram_ids} -> host RAM (~101 GB each)")
        t0 = time.perf_counter()
        tables = {i: load_engram_table(loader, i) for i in engram_ids}
        engram_lookup = EngramHostLookup(
            config, loader.snapshot_dir, tables, batch=1, max_seq=target_len, tokenizer=tokenizer
        )
        logger.info(f"Engram tables loaded in {time.perf_counter() - t0:.1f}s")

    # --- the model and the image encoder ----------------------------------------------------------- #
    progress.step(
        f"[6/8] building the model: {num_layers} layers and the head, {image_rows} image rows; then the image "
        f"encoder on stage {VISION_STAGE} (a cold weight cache reads + quantizes the checkpoint: slow)"
    )
    cache = WeightCache(CACHE_DIR) if CACHE_DIR else None
    logger.info(f"weight cache: {cache.path if cache else 'disabled'}")
    t0 = time.perf_counter()
    model = DeepSeekV41PrefillModel(
        config,
        loader,
        layer_devices[0],
        rope,
        num_layers=num_layers,
        cache=cache,
        weight_dtype=_ATTENTION_WEIGHT_DTYPE,
        expert_dtype=_EXPERT_DTYPE,
        tp_size=_TP_SIZE,
        layer_devices=layer_devices,
        progress=progress,
        engram_lookup=engram_lookup,
        image_rows=image_rows,
    )
    tower = DeepSeekV41Vision(vision_config, loader, submeshes[VISION_STAGE], cache=cache)
    write_embedding_rows(
        model.embedding_weight,
        torch.stack([tower.delimiters[kind] for kind in DELIMITER_ROWS]),
        [config.vocab_size + row for row in DELIMITER_ROWS.values()],
        config.vocab_size,
    )
    progress.step("model built; waiting for every device to finish its uploads")
    model.synchronize("uploads")
    ttnn.synchronize_device(submeshes[VISION_STAGE])
    build_seconds = time.perf_counter() - t0
    logger.info(
        f"built DeepSeekV41PrefillModel ({model.num_layers} layers) and the image encoder in {build_seconds:.1f}s"
    )

    # --- traces: persistent state and sockets, compile the chunk step, capture (not timed) -------- #
    # Before anything else allocates on the devices: allocating on a device that holds a trace is unsafe.
    progress.step(
        f"[7/8] preparing the traced prefill for {target_len} tokens in chunks of {chunk_size}: persistent buffers, "
        "H2D / D2D / D2H sockets, compile the chunk step, capture one trace per stage (not timed)"
    )
    t0 = time.perf_counter()
    model.prepare_traced_prefill(target_len, chunk_size)
    prepare_seconds = time.perf_counter() - t0
    logger.info(f"traced prefill prepared in {prepare_seconds:.1f}s")

    # --- per image: encode it (eager, between prefills, as the server's admission does), then prefill ---- #
    failures = []
    check_answer = num_layers == config.num_hidden_layers
    for index, image in enumerate(images):
        progress.step(f"[8/8] image {index + 1}/{len(images)} ({image.digit!r}): encode, write its rows, prefill")
        progress.verbose = True
        t0 = time.perf_counter()
        rows = tower.encode(image.patches, image.n_h, image.n_w)
        t1 = time.perf_counter()
        write_embedding_rows(
            model.embedding_weight, rows, config.vocab_size + FIRST_IMAGE_ROW + image.slots, config.vocab_size
        )
        logger.info(
            f"image {image.digit!r}: {image.n_h} x {image.n_w} patches encoded in {t1 - t0:.2f}s, "
            f"{rows.shape[0]} rows written in {time.perf_counter() - t1:.2f}s"
        )
        assert torch.isfinite(rows.float()).all(), f"image {image.digit!r}: non-finite encoder rows"

        chunk_times: list[tuple[int, int, float]] = []
        elapsed = 0.0

        def on_chunk(chunk: int, start: int, end: int, seconds: float) -> None:
            nonlocal elapsed
            elapsed += seconds
            chunk_times.append((start, end, seconds))
            recent = [s for _, _, s in chunk_times[1:][-8:]] or [seconds]
            remaining = (num_chunks - chunk - 1) * statistics.mean(recent)
            fill = " (pipeline fill)" if chunk == 0 and num_chunks > 1 else ""
            logger.info(
                f"prefill chunk {chunk + 1}/{num_chunks} tokens [{start}, {end}) out: +{seconds:.3f} s{fill} | "
                f"{seconds / (end - start) * 1000:.3f} ms/token | {(end - start) / seconds:.1f} tok/s | "
                f"{end}/{target_len} tokens, elapsed {elapsed:.1f} s, ~{remaining:.0f} s to go"
            )

        # The replay runs no per-layer Python, so only the chunk lines (and the heartbeat) remain.
        progress.verbose = False
        ids = torch.tensor(image.ids, dtype=torch.long).unsqueeze(0)
        t0 = time.perf_counter()
        logits, states = model.prefill_traced(ids, on_chunk=on_chunk)
        wall_seconds = time.perf_counter() - t0
        progress.step(f"image {index + 1}/{len(images)}: prefill finished")
        _report(chunk_times, num_stages, wall_seconds, build_seconds, prepare_seconds)

        assert len(chunk_times) == num_chunks
        assert all(state.seq_len == target_len for state in states)
        model.free_traced_states(states)
        row = logits.reshape(-1)  # already on the host: the last stage streams it over the D2H socket
        assert row.numel() == config.vocab_size and torch.isfinite(row).all(), "non-finite logits"
        top = row.topk(5)
        answer = tokenizer.decode([int(top.indices[0])]).strip()
        logger.info(
            f"image {image.digit!r}: first generated token (top 5): "
            + ", ".join(f"{tokenizer.decode([int(i)])!r} ({v:.2f})" for v, i in zip(top.values.tolist(), top.indices))
        )
        if check_answer and answer != image.digit:
            failures.append(f"image of {image.digit!r}: the first generated token is {answer!r}")

    model.release_traced_prefill()
    progress.step("done")
    if not check_answer:
        logger.warning(f"only {num_layers}/{config.num_hidden_layers} layers built: the answers are not checked")
    assert not failures, "the model misread the images:\n  " + "\n  ".join(failures)
