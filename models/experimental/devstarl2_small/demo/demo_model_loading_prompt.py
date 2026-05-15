# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo / smoke test using the **same user prompt** as ``reference/model_loading.py``:

- Multimodal message: one image placeholder + text
  ``"Describe what you see in this image."``

``--vision-square-pixels S`` resizes to ``S×S`` (LANCZOS) before ``processor`` and overrides
``--vision-max-edge`` (e.g. ``1540`` for square HF-style sizing).

TT ``max_seq_len`` follows ``devstral_utils.default_devstral_demo_max_seq_len``: **Blackhole** defaults to
at least **256000** tokens for RoPE/warmups; **Wormhole** uses a prompt-sized cap.
Dense KV uses ``max_kv_cache_seq_len`` via ``devstral_tt_kv_cache_max_seq_len`` (same as
``demo_devstral2_tt_multimodal.py``)—run-sized DRAM, not full 256K per layer unless needed.

**HF path** (``--backend hf``, default): ``AutoProcessor`` + ``AutoModelForImageTextToText``; same
``--vision-max-edge`` / ``--vision-square-pixels`` as TT before ``processor`` (default max-edge ``0`` =
no PIL resize; use ``--vision-square-pixels`` e.g. ``1540`` for square HF-style sizing).
Default image ``reference/sample.jpeg``; ``max_new_tokens=100``.

**TT path** (``--backend tt``): loads :class:`TtDevstral2SmallModel` (Pixtral vision + projector +
``TtMinistral3Model``), runs TT vision/projector, merges features into text embeddings like HF
``masked_scatter`` on ``image_token_id``, then ``language_model.forward_prefill_from_embeddings``.
Same ``--vision-max-edge`` / ``--vision-square-pixels`` as HF. Pixtral L1 grows with patch count; ``0``
max-edge = no thumbnail (fine on HF; may exceed device L1 on TT).

Usage (repo root)::

    python models/experimental/devstarl2_small/demo/demo_model_loading_prompt.py

    python models/experimental/devstarl2_small/demo/demo_model_loading_prompt.py --backend tt --seed 0

    python models/experimental/devstarl2_small/demo/demo_model_loading_prompt.py --backend tt \\
        --image path/to.jpg --text-layers 1 --max-new-tokens 16 --lm-head-cpu

    # Cap TT RoPE ``max_seq_len`` / override KV budgeting (same flags as text multimodal demo)
    python models/experimental/devstarl2_small/demo/demo_model_loading_prompt.py --backend tt \\
        --max-seq-len 8192

    # ~256K-token multimodal prompt (generate JSON first, then run demo):
    python models/experimental/devstarl2_small/scripts/make_long_context_prompt.py \\
        --target-tokens 256000 --vision-square-pixels 1540 \\
        --output models/experimental/devstarl2_small/reference/messages_256k.json
    python models/experimental/devstarl2_small/demo/demo_model_loading_prompt.py --backend tt \\
        --messages-json models/experimental/devstarl2_small/reference/messages_256k.json \\
        --vision-square-pixels 1540 --max-new-tokens 1

Logs use boxed ``───`` sections for context and generation, and ``▶`` lines for per-phase
timings. Set ``DEVSTRAL2_DEMO_PERF_LOG_STEPS=1`` for per-step generation detail.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import types
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import MistralCommonBackend
except ImportError:  # transformers<5: use processor.tokenizer in run_tt (see load_tt_tokenizer).
    MistralCommonBackend = None  # type: ignore[misc, assignment]
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model
from transformers.models.pixtral.modeling_pixtral import position_ids_in_meshgrid

import ttnn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
from models.experimental.devstarl2_small.demo import demo_devstral2_tt_multimodal as _tt_demo
from models.experimental.devstarl2_small.devstral_utils.demo_perf import (
    log_gen_step_summary,
    log_gen_step_timings,
    log_generation_done,
    log_generation_start,
    log_hf_context_after,
    log_hf_context_before,
    log_model_output,
    log_perf_ms,
    log_tt_context_budget,
    perf_log_generation_steps,
)
from models.experimental.devstarl2_small.devstral_utils import (
    DEVSTRAL_DEMO_BLACKHOLE_DEFAULT_MAX_SEQ_LEN,
    default_devstral_demo_max_seq_len,
    devstral_tt_kv_cache_max_seq_len,
    devstral_validate_demo_kv_dram_budget,
    devstral_supports_on_device_sampling,
    pad_input_ids_and_positions_for_tt_prefill,
    tt_lm_head_logits_block,
    tt_sampling_output_token_id,
)
from models.experimental.devstarl2_small.reference.inference_fixtures import REFERENCE_GENERATE_KWARGS
from models.experimental.devstarl2_small.tt.tt_devstral2_small_model import TtDevstral2SmallModel
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import ModelArgs

_DEFAULT_MODEL_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
_DEMO_DIR = Path(__file__).resolve().parent
_REF_DIR = _DEMO_DIR.parent / "reference"

MODEL_LOADING_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe what you see in this image."},
        ],
    }
]


def load_chat_messages_from_json(path: Path) -> list[dict[str, Any]]:
    """Load OpenAI-style chat messages from JSON (list or ``make_long_context_prompt.py`` output)."""
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("messages"), list):
        meta = data.get("metadata")
        if isinstance(meta, dict) and meta.get("measured_tokens") is not None:
            logger.info(
                f"Loaded long-context messages from {path} "
                f"(metadata: {meta.get('measured_tokens'):,} measured tokens, mode={meta.get('mode')!r})."
            )
        return data["messages"]
    raise ValueError(f"Expected a JSON list or object with 'messages' key in {path}")


def _resize_image_max_edge(image: Image.Image, max_edge: int) -> Image.Image:
    """Cap longest side so vision patch sequence stays small enough for TT Pixtral QKV L1."""
    if max_edge <= 0:
        return image
    w, h = image.size
    if max(w, h) <= max_edge:
        return image
    out = image.copy()
    out.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return out


def _prepare_vision_image(
    image: Image.Image,
    vision_max_edge: int,
    vision_square_pixels: int | None,
) -> Image.Image:
    """
    PIL preprocessing before ``processor``: optional exact square resize, otherwise max-edge thumbnail.

    If ``vision_square_pixels`` > 0, resize with LANCZOS to ``S×S`` (ignores ``vision-max-edge``).
    Else apply :func:`_resize_image_max_edge`.
    """
    image = image.convert("RGB")
    if vision_square_pixels is not None and vision_square_pixels > 0:
        s = vision_square_pixels
        return image.resize((s, s), Image.Resampling.LANCZOS)
    return _resize_image_max_edge(image, vision_max_edge)


def _sample_image_path() -> Path:
    return _REF_DIR / "sample.jpeg"


def _max_patch_grid_side(vision_cfg) -> int:
    iz = vision_cfg.image_size
    if isinstance(iz, (tuple, list)):
        iz = iz[0]
    return int(iz) // int(vision_cfg.patch_size)


def _image_sizes_list_from_batch(image_sizes: torch.Tensor) -> list[tuple[int, int]]:
    if image_sizes.ndim != 2 or image_sizes.shape[0] < 1:
        raise ValueError(f"Expected image_sizes [N, 2], got {tuple(image_sizes.shape)}")
    return [(int(image_sizes[i, 0].item()), int(image_sizes[i, 1].item())) for i in range(image_sizes.shape[0])]


def _vision_position_ids_tt(
    hf_inner: Mistral3Model,
    pixel_values: torch.Tensor,
    image_sizes_list: list[tuple[int, int]],
    mesh_device,
) -> ttnn.Tensor:
    vision_cfg = hf_inner.config.vision_config
    patch_sz = int(vision_cfg.patch_size)
    hf_vm = hf_inner.vision_tower
    target_dtype = hf_vm.patch_conv.weight.dtype
    pv = pixel_values.to(dtype=target_dtype)
    pe_conv = hf_vm.patch_conv(pv)
    plist = [e[..., : s[0] // patch_sz, : s[1] // patch_sz] for e, s in zip(pe_conv, image_sizes_list)]
    position_ids = position_ids_in_meshgrid(plist, max_width=_max_patch_grid_side(vision_cfg))
    return ttnn.from_torch(
        position_ids.unsqueeze(0).to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _merge_image_into_text_embeds(
    hf_inner: Mistral3Model,
    input_ids: torch.LongTensor,
    image_rows: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Match HF ``Mistral3Model.forward`` scatter of image features into embed table rows."""
    emb = hf_inner.get_input_embeddings()
    dev = emb.weight.device
    ids = input_ids.to(dev)
    rows = image_rows.to(device=dev, dtype=emb.weight.dtype)
    inputs_embeds = emb(ids)
    mask = ids == int(image_token_id)
    special = mask.unsqueeze(-1).expand_as(inputs_embeds)
    if inputs_embeds[special].numel() != rows.numel():
        raise RuntimeError(
            f"Image token mismatch: mask elements {inputs_embeds[special].numel()} vs projected {rows.numel()}."
        )
    return inputs_embeds.masked_scatter(special, rows.reshape(-1))


def _tt_prefill_from_merged_embeds(
    current_ids: torch.LongTensor,
    merged_embeds_bsh: torch.Tensor,
    pad_row_1d: torch.Tensor,
    pad_token_id: int,
    mesh_device,
    tt_language_model,
    model_args,
    seq_len_keep: int,
) -> ttnn.Tensor:
    """Pad merged [1, sl, D] at the end to TT prefill length, upload, ``forward_prefill_from_embeddings``."""
    sl = merged_embeds_bsh.shape[1]
    if int(current_ids.shape[1]) != sl or seq_len_keep != sl:
        raise RuntimeError("current_ids / merged_embeds / seq_len_keep length mismatch.")
    device_host = current_ids.device
    position_ids = torch.arange(sl, dtype=torch.long, device=device_host).unsqueeze(0)
    input_ids_pad, position_ids_pad, _ = pad_input_ids_and_positions_for_tt_prefill(
        current_ids,
        position_ids,
        pad_token_id,
        int(model_args.n_kv_heads),
        int(model_args.cluster_shape[1]),
    )
    target = int(input_ids_pad.shape[1])
    pad_n = target - sl
    if pad_n > 0:
        pr = pad_row_1d.to(device=merged_embeds_bsh.device, dtype=merged_embeds_bsh.dtype).view(1, 1, -1)
        pad_block = pr.expand(1, pad_n, merged_embeds_bsh.shape[-1])
        inputs_pad = torch.cat([merged_embeds_bsh, pad_block], dim=1)
    else:
        inputs_pad = merged_embeds_bsh

    # [1, S, D] -> [1, 1, S, D] (same rank as ``embed_tokens`` + ``unsqueeze_to_4D``, not 5D).
    h4 = inputs_pad.unsqueeze(1).contiguous()
    h_tt = ttnn.from_torch(
        h4,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    pos_tt = ttnn.from_torch(
        position_ids_pad.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    try:
        return tt_language_model.forward_prefill_from_embeddings(h_tt, None, pos_tt)
    finally:
        ttnn.deallocate(h_tt)


def run_hf(
    model_id: str,
    image_path: Path,
    max_new_tokens: int,
    seed: int | None,
    vision_max_edge: int,
    vision_square_pixels: int | None,
    messages: list[dict[str, Any]] | None = None,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found at {image_path} (same path as reference/model_loading.py). "
            "Add sample.jpeg under models/experimental/devstarl2_small/reference/."
        )

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    log_perf_ms("hf.load_processor_and_model", t0)
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    orig_sz = image.size
    image = _prepare_vision_image(image, vision_max_edge, vision_square_pixels)
    if vision_square_pixels is not None and vision_square_pixels > 0:
        logger.info(f"HF: PIL square resize {orig_sz} -> {image.size} (--vision-square-pixels {vision_square_pixels}).")
    elif vision_max_edge <= 0:
        logger.info("HF: vision-max-edge 0 — no PIL thumbnail before processor.")
    elif image.size != orig_sz:
        logger.info(f"HF: PIL resize {orig_sz} -> {image.size} (vision-max-edge={vision_max_edge}).")
    else:
        logger.info(f"HF: image within vision-max-edge={vision_max_edge} ({image.size}), no thumbnail.")

    chat_messages = messages if messages is not None else MODEL_LOADING_MESSAGES
    prompt = processor.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    _prompt_tok = int(inputs["input_ids"].shape[1])
    log_hf_context_before(prompt_tokens=_prompt_tok, max_new_tokens=max_new_tokens)

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    logger.info(f"HF multimodal generate (max_new_tokens={max_new_tokens}) on {device} …")
    t_gen = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    log_perf_ms("hf.model.generate", t_gen)
    log_hf_context_after(
        output_tokens=int(output_ids.shape[1]),
        prompt_tokens=_prompt_tok,
    )
    t_dec = time.perf_counter()
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    log_perf_ms("hf.processor.batch_decode", t_dec)
    log_model_output(
        backend="HF",
        text=output_text,
        new_tokens=int(output_ids.shape[1]) - _prompt_tok,
    )


def run_tt(
    model_id: str,
    image_path: Path,
    mesh_width: int,
    text_layers: int | None,
    max_new_tokens: int,
    greedy: bool,
    temperature: float,
    seed: int | None,
    lm_head_cpu: bool,
    lm_head_max_device_cols: int | None,
    vision_max_edge: int,
    vision_square_pixels: int | None,
    cpu_sampling: bool,
    max_seq_len_override: int | None,
    messages: list[dict[str, Any]] | None = None,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"TT multimodal path requires an image file; missing {image_path}.")

    os.environ["HF_MODEL"] = model_id
    _tt_demo.apply_devstral_hf_trust_patches()

    ref_do_sample = bool(REFERENCE_GENERATE_KWARGS["do_sample"])
    do_sample = ref_do_sample if not greedy else False
    gen_temperature = temperature if not greedy else float(REFERENCE_GENERATE_KWARGS["temperature"])

    t_hf_prep = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")
    orig_sz = image.size
    image = _prepare_vision_image(image, vision_max_edge, vision_square_pixels)
    if vision_square_pixels is not None and vision_square_pixels > 0:
        logger.info(
            f"TT vision: PIL square resize {orig_sz} -> {image.size} "
            f"(--vision-square-pixels {vision_square_pixels}); many patches may exceed L1 on device."
        )
    elif image.size != orig_sz:
        logger.info(
            f"TT vision: thumbnail {orig_sz} -> {image.size} "
            f"(vision-max-edge={vision_max_edge}). "
            "Use --vision-square-pixels or --vision-max-edge 0 for other sizing."
        )
    elif vision_max_edge > 0:
        logger.info(f"TT vision: image within vision-max-edge={vision_max_edge} ({image.size}), no thumbnail.")
    chat_messages = messages if messages is not None else MODEL_LOADING_MESSAGES
    prompt = processor.apply_chat_template(
        chat_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    proc_out = processor(text=prompt, images=image, return_tensors="pt")
    input_ids = proc_out["input_ids"]
    pixel_values = proc_out["pixel_values"].to(torch.bfloat16)
    image_sizes = proc_out["image_sizes"]
    if not isinstance(image_sizes, torch.Tensor):
        raise TypeError(f"Expected image_sizes tensor from processor, got {type(image_sizes)}")
    image_sizes_list = _image_sizes_list_from_batch(image_sizes)
    log_perf_ms("tt.hf_processor_chat_template_and_pixel_batch", t_hf_prep)

    prompt_len = int(input_ids.shape[1])
    extra_tokens = max(0, max_new_tokens)
    need = prompt_len + extra_tokens + 2048

    t_mesh = time.perf_counter()
    mesh_device = _tt_demo.open_devstral_demo_mesh(max(1, min(mesh_width, ttnn.get_num_devices())))
    log_perf_ms("tt.open_mesh_device", t_mesh)
    try:
        if max_seq_len_override is None:
            max_seq = default_devstral_demo_max_seq_len(mesh_device, need)
            _bh = ttnn.device.is_blackhole(mesh_device)
            logger.info(
                f"TT max_seq_len={max_seq} (device default; blackhole={_bh}"
                + (f", floor={DEVSTRAL_DEMO_BLACKHOLE_DEFAULT_MAX_SEQ_LEN}" if _bh else "")
                + f"; need={need})."
            )
        else:
            if max_seq_len_override < need:
                logger.warning(
                    f"--max-seq-len {max_seq_len_override} is below prompt + max_new_tokens + margin ({need}); "
                    f"using {need}."
                )
                max_seq = need
            else:
                max_seq = max_seq_len_override
            logger.info(f"TT max_seq_len={max_seq} (explicit --max-seq-len; need={need}).")

        dtype_tt = ttnn.bfloat16

        t_ckpt = time.perf_counter()
        logger.info("Loading checkpoint via ModelArgs.load_state_dict() …")
        model_args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq,
            dummy_weights=False,
            use_hf_rope=True,
            cache_hf=True,
        )
        model_args.max_kv_cache_seq_len = devstral_tt_kv_cache_max_seq_len(model_args, need)
        log_tt_context_budget(
            prompt_tokens=prompt_len,
            budget_need_tokens=need,
            rope_max_seq_len=max_seq,
            kv_cache_seq_dim=model_args.max_kv_cache_seq_len,
        )
        model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)
        meta_state_dict = model_args.load_state_dict()
        log_perf_ms("tt.model_args_init_and_load_state_dict", t_ckpt)

        if text_layers is not None:
            if text_layers < 1 or text_layers > model_args.full_model_n_layers:
                raise ValueError(f"--text-layers must be in [1, {model_args.full_model_n_layers}], got {text_layers}")
            model_args.n_layers = text_layers

        devstral_validate_demo_kv_dram_budget(
            model_args,
            prompt_tokens=prompt_len,
            kv_seq_len=model_args.max_kv_cache_seq_len,
            mesh_device=mesh_device,
        )

        hf_full = model_args.cached_hf_model
        if hf_full is None:
            raise RuntimeError("Expected cached HF model after load_state_dict with cache_hf=True.")
        hf_inner = hf_full.model
        if not isinstance(hf_inner, Mistral3Model):
            raise TypeError(f"Expected Mistral3Model, got {type(hf_inner)}")

        _embed_dev = hf_inner.get_input_embeddings().weight.device
        input_ids = input_ids.to(_embed_dev)
        pixel_values = pixel_values.to(_embed_dev)
        image_sizes = image_sizes.to(_embed_dev)

        text_cfg = model_args.hf_config.text_config
        if not isinstance(text_cfg, Ministral3Config):
            raise TypeError(f"Demo expects Ministral3Config as text_config, got {type(text_cfg)!r}")

        vision_cfg = hf_full.config.vision_config
        image_token_id = int(hf_full.config.image_token_id)

        t_tt_dev = time.perf_counter()
        tt_devstral = TtDevstral2SmallModel(
            mesh_device=mesh_device,
            tt_ccl=TT_CCL(mesh_device),
            model_args=model_args,
            meta_state_dict=meta_state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype_tt),
            dtype=dtype_tt,
            transformation_mats={"decode": None, "prefill": None},
            configuration=model_args,
            vision_config=vision_cfg,
            vision_n_layers=None,
        )
        log_perf_ms("tt.TtDevstral2SmallModel_init", t_tt_dev)

        t_vis = time.perf_counter()
        pos_vision = _vision_position_ids_tt(hf_inner, pixel_values, image_sizes_list, mesh_device)
        img_tt = tt_devstral.get_projected_image_features(pixel_values, image_sizes_list, pos_vision)
        ttnn.deallocate(pos_vision)

        img_torch = ttnn.to_torch(img_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
        ttnn.deallocate(img_tt)
        while img_torch.dim() > 2:
            img_torch = img_torch.squeeze(0)
        img_rows = img_torch.reshape(-1, img_torch.shape[-1]).contiguous()
        log_perf_ms("tt.vision_projector_and_rows_to_torch", t_vis)

        sd_prefix = model_args.get_state_dict_prefix("", None)
        out_key = f"{sd_prefix}output.weight"
        if out_key not in meta_state_dict:
            raise RuntimeError(f"Missing {out_key!r} in meta state dict (required for LM head).")

        lm_head_weight_cpu: torch.Tensor | None = None
        tt_lm_head: LMHead | None = None
        t_lmsetup = time.perf_counter()
        if lm_head_cpu:
            lm_head_weight_cpu = meta_state_dict[out_key].detach().to(torch.bfloat16).cpu().contiguous()
            logger.info(
                f"CPU LM head: chunked torch matmul; weight {tuple(lm_head_weight_cpu.shape)} {lm_head_weight_cpu.dtype}."
            )
        else:
            lm_head_max_cols = _tt_demo.demo_lm_head_max_columns_per_device(model_args, cli_cap=lm_head_max_device_cols)
            logger.info(
                f"On-device LMHead max columns per shard: {lm_head_max_cols} "
                f"(ModelArgs value {model_args.max_columns_per_device_lm_head})."
            )
            tt_lm_head = LMHead(
                args=model_args,
                mesh_device=mesh_device,
                tt_ccl=TT_CCL(mesh_device),
                dtype=dtype_tt,
                state_dict=meta_state_dict,
                state_dict_prefix=sd_prefix,
                weight_cache_path=model_args.weight_cache_path(dtype_tt),
                max_columns_per_device=lm_head_max_cols,
            )

        log_perf_ms("tt.lm_head_setup", t_lmsetup)

        use_device_sampling = (
            not lm_head_cpu
            and not cpu_sampling
            and tt_lm_head is not None
            and devstral_supports_on_device_sampling(model_args, mesh_device)
        )
        if (
            tt_lm_head is not None
            and not cpu_sampling
            and not lm_head_cpu
            and not devstral_supports_on_device_sampling(model_args, mesh_device)
        ):
            logger.warning(
                "Vocab size / mesh splits exceed on-device sampling limit (64k per split); "
                "using PyTorch softmax / argmax on host logits."
            )

        sampling: SamplingGenerator | None = None
        sampling_empty_slots: list[int] | None = None
        t_samp = time.perf_counter()
        if use_device_sampling:
            sampling = SamplingGenerator(
                args=model_args,
                mesh_device=mesh_device,
                tt_ccl=TT_CCL(mesh_device),
                enable_internal_trace=False,
            )
            sampling_empty_slots = list(range(sampling.tt_sampling.max_batch_size))
            seed_for_params = seed if seed is not None else None
            if not do_sample:
                sampling_in = SamplingParams(temperature=0.0, top_k=32, top_p=1.0, seed=seed_for_params)
            else:
                sampling_in = SamplingParams(
                    temperature=float(gen_temperature),
                    top_k=32,
                    top_p=1.0,
                    seed=seed_for_params,
                )
            formatted_sampling = format_sampling_params(sampling_in, len(sampling_empty_slots))
            sampling.reset_sampling_params(formatted_sampling)
            sampling.seed_manager.reset_seed(formatted_sampling.seed, sampling_empty_slots)

        log_perf_ms("tt.sampling_generator_setup", t_samp)

        t_tok = time.perf_counter()
        if MistralCommonBackend is not None:
            tokenizer = MistralCommonBackend.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=os.getenv("CI") == "true",
            )
        else:
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError(
                    "transformers has no MistralCommonBackend (need transformers>=5) and "
                    "AutoProcessor has no .tokenizer; cannot load tokenizer for TT decode."
                )
            logger.warning(
                "MistralCommonBackend unavailable (transformers<5); using processor.tokenizer for pad/eos/decode."
            )
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        else:
            pad_token_id = int(pad_token_id)

        log_perf_ms("tt.tokenizer_for_decode", t_tok)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        id_device = input_ids.device
        eos_ids = _tt_demo.eos_token_ids(hf_full.config, tokenizer)
        current_ids = input_ids.clone()
        mode = "greedy" if greedy else f"sample (T={gen_temperature})"
        lm_mode = "CPU lm_head (chunked torch)" if lm_head_cpu else "TT lm_head"
        samp_mode = (
            "on-device SamplingGenerator"
            if sampling is not None
            else "PyTorch softmax / multinomial / argmax on TT logits (host)"
        )
        log_generation_start(
            backend="TT",
            current_sequence_tokens=int(current_ids.shape[1]),
            prompt_tokens=prompt_len,
            max_new_tokens=max_new_tokens,
            mode=mode,
            lm_mode=lm_mode,
            sampling_mode=samp_mode,
            extra=f"image {image_path.name}",
        )

        emb_layer = hf_inner.get_input_embeddings()
        pad_row = emb_layer(torch.tensor([[pad_token_id]], device=emb_layer.weight.device, dtype=torch.long))[
            0, 0
        ].detach()

        tt_lm = tt_devstral.language_model

        sum_merge_s = 0.0
        sum_prefill_s = 0.0
        sum_lm_s = 0.0
        sum_post_lm_s = 0.0
        n_gen_steps = 0
        t_loop = time.perf_counter()
        for step_i in range(max_new_tokens):
            sl = int(current_ids.shape[1])
            t_m = time.perf_counter()
            merged = _merge_image_into_text_embeds(hf_inner, current_ids, img_rows, image_token_id)
            merged_bf = merged.to(torch.bfloat16)
            dt_merge = time.perf_counter() - t_m

            t_p = time.perf_counter()
            tt_out = _tt_prefill_from_merged_embeds(
                current_ids,
                merged_bf,
                pad_row,
                pad_token_id,
                mesh_device,
                tt_lm,
                model_args,
                sl,
            )
            dt_prefill = time.perf_counter() - t_p

            if sampling is not None:
                assert tt_lm_head is not None
                tok_slot = (sl - 1) % 32
                t_lm = time.perf_counter()
                sampling.seed_manager.get_new_values()
                logits_tt = tt_lm_head_logits_block(tt_out, sl - 1, model_args, tt_lm_head)
                dt_lm = time.perf_counter() - t_lm
                t_po = time.perf_counter()
                sample_result = sampling.sample(logits_tt, enable_trace=False)
                tt_next = sample_result[0] if isinstance(sample_result, tuple) else sample_result
                next_scalar = tt_sampling_output_token_id(tt_next, tok_slot)
                ttnn.deallocate(logits_tt)
                ttnn.deallocate(tt_out)
                next_id = torch.tensor([[next_scalar]], device=id_device, dtype=torch.long)
                dt_post = time.perf_counter() - t_po
            else:
                t_lm = time.perf_counter()
                if lm_head_cpu:
                    assert lm_head_weight_cpu is not None
                    logits_row = _tt_demo.cpu_lm_head_logits_last_token(
                        tt_out, sl - 1, mesh_device, lm_head_weight_cpu, int(model_args.vocab_size)
                    )
                else:
                    assert tt_lm_head is not None
                    logits_row = _tt_demo.tt_lm_head_logits_last_token(
                        tt_out, sl - 1, mesh_device, model_args, tt_lm_head
                    )
                dt_lm = time.perf_counter() - t_lm
                t_po = time.perf_counter()
                ttnn.deallocate(tt_out)
                if do_sample:
                    probs = torch.softmax(logits_row.float().squeeze(0) / max(gen_temperature, 1e-6), dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).view(1, 1)
                else:
                    next_id = logits_row.argmax(dim=-1, keepdim=True)
                next_id = next_id.to(id_device)
                dt_post = time.perf_counter() - t_po

            sum_merge_s += dt_merge
            sum_prefill_s += dt_prefill
            sum_lm_s += dt_lm
            sum_post_lm_s += dt_post
            n_gen_steps += 1
            if perf_log_generation_steps():
                log_gen_step_timings(
                    step_index=step_i,
                    sequence_tokens=sl,
                    phase_seconds={
                        "merge": dt_merge,
                        "prefill": dt_prefill,
                        "lm_head": dt_lm,
                        "post": dt_post,
                    },
                )

            if eos_ids and int(next_id.item()) in eos_ids:
                break
            current_ids = torch.cat([current_ids, next_id], dim=1)

        loop_elapsed = time.perf_counter() - t_loop
        log_perf_ms("tt.autoregressive_loop_wall", t_loop)
        if n_gen_steps > 0:
            inv = 1.0 / float(n_gen_steps)
            log_gen_step_summary(
                n_steps=n_gen_steps,
                wall_seconds=loop_elapsed,
                avg_phase_seconds={
                    "merge (host)": sum_merge_s * inv,
                    "text LM prefill": sum_prefill_s * inv,
                    "LM head": sum_lm_s * inv,
                    "sample / post": sum_post_lm_s * inv,
                },
                throughput_value=float(n_gen_steps) / loop_elapsed,
            )

        log_generation_done(
            backend="TT",
            final_sequence_tokens=int(current_ids.shape[1]),
            prompt_tokens=prompt_len,
        )

        answer_ids = current_ids[0, prompt_len:]
        answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False)
        log_model_output(backend="TT", text=answer_text, new_tokens=answer_ids.numel())
    finally:
        ttnn.close_mesh_device(mesh_device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HF multimodal or TT TtDevstral2SmallModel demo (model_loading.py prompt)."
    )
    parser.add_argument("--model-id", default=_DEFAULT_MODEL_ID)
    parser.add_argument(
        "--backend",
        choices=("hf", "tt"),
        default="hf",
        help="hf: AutoProcessor + image + generate. tt: TtDevstral2SmallModel vision + text LM on device.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=_sample_image_path(),
        help="Image path (default: reference/sample.jpeg).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mesh-width", type=int, default=1)
    parser.add_argument("--text-layers", type=int, default=None)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=float(REFERENCE_GENERATE_KWARGS["temperature"]))
    parser.add_argument("--lm-head-cpu", action="store_true")
    parser.add_argument("--lm-head-max-device-cols", type=int, default=None)
    parser.add_argument(
        "--cpu-sampling",
        action="store_true",
        help="Use PyTorch softmax/multinomial/argmax on host logits instead of on-device SamplingGenerator.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        metavar="S",
        help="TT rope/grid max_seq_len cap when set. Default: Blackhole max(256000, need), "
        "Wormhole max(4096, need), need=prompt_len+max_new_tokens+2048 "
        "(see devstral_utils.default_devstral_demo_max_seq_len). KV allocation stays run-sized.",
    )
    parser.add_argument(
        "--vision-max-edge",
        type=int,
        default=0,
        help="HF and TT: max longest image side (px) PIL thumbnail before processor (0 = no thumbnail; "
        "ignored if --vision-square-pixels is set). Large native images on TT can exceed Pixtral L1; "
        "prefer --vision-square-pixels for a bounded patch grid.",
    )
    parser.add_argument(
        "--vision-square-pixels",
        type=int,
        default=None,
        metavar="S",
        help="HF and TT: resize image to exactly S×S (LANCZOS) before processor (e.g. 1540 for HF-style "
        "square vision). Overrides vision-max-edge when set.",
    )
    parser.add_argument(
        "--messages-json",
        type=Path,
        default=None,
        help="Optional chat messages JSON (from scripts/make_long_context_prompt.py) instead of the default prompt.",
    )
    args = parser.parse_args()

    if args.vision_square_pixels is not None and args.vision_square_pixels <= 0:
        parser.error("--vision-square-pixels must be a positive integer when set.")

    custom_messages = None
    if args.messages_json is not None:
        if not args.messages_json.is_file():
            parser.error(f"--messages-json file not found: {args.messages_json}")
        custom_messages = load_chat_messages_from_json(args.messages_json)

    if args.backend == "hf":
        run_hf(
            args.model_id,
            args.image,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            vision_max_edge=args.vision_max_edge,
            vision_square_pixels=args.vision_square_pixels,
            messages=custom_messages,
        )
    else:
        run_tt(
            args.model_id,
            args.image,
            mesh_width=args.mesh_width,
            text_layers=args.text_layers,
            max_new_tokens=args.max_new_tokens,
            greedy=args.greedy,
            temperature=args.temperature,
            seed=args.seed,
            lm_head_cpu=args.lm_head_cpu,
            lm_head_max_device_cols=args.lm_head_max_device_cols,
            vision_max_edge=args.vision_max_edge,
            vision_square_pixels=args.vision_square_pixels,
            cpu_sampling=args.cpu_sampling,
            max_seq_len_override=args.max_seq_len,
            messages=custom_messages,
        )


if __name__ == "__main__":
    main()
