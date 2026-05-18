# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Multimodal image+text demo: default prompt ``Describe what you see in this image.`` with optional
``resource/sample.jpeg``.

- Multimodal message: one image placeholder + text
  ``"Describe what you see in this image."``

``--vision-square-pixels S`` resizes to ``S×S`` (LANCZOS) before ``processor`` and overrides
``--vision-max-edge`` (e.g. ``1540`` for square HF-style sizing).

**HF path** (``--backend hf``, default): ``AutoProcessor`` + ``AutoModelForImageTextToText``; same
``--vision-max-edge`` / ``--vision-square-pixels`` as TT before ``processor`` (default max-edge ``0`` =
no PIL resize). Default image ``resource/sample.jpeg``; ``max_new_tokens=100``.

**TT path** (``--backend tt``): loads :class:`TtDevstral2SmallModel` (Pixtral vision + projector +
``TtMinistral3Model``), runs TT vision/projector, merges features into text embeddings like HF
``masked_scatter`` on ``image_token_id``, then ``language_model.forward_prefill_from_embeddings``.
Generation knobs align with ``devstral_utils.chat_reference`` where relevant. Pixtral L1 grows with
patch count; ``0`` max-edge = no thumbnail (fine on HF; may exceed device L1 on TT).

Usage (repo root)::

    python -m models.experimental.devstarl2_small.demo.tt_image_demo

    python -m models.experimental.devstarl2_small.demo.tt_image_demo --backend tt --seed 0

    python -m models.experimental.devstarl2_small.demo.tt_image_demo --backend tt \\
        --image path/to.jpg --text-layers 1 --max-new-tokens 16 --lm-head-cpu
"""

from __future__ import annotations

import argparse
import os
import time
import types
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, MistralCommonBackend
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model
from transformers.models.pixtral.modeling_pixtral import position_ids_in_meshgrid

import ttnn
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params
from models.experimental.devstarl2_small.demo import tt_text_demo as _tt_demo
from models.experimental.devstarl2_small.devstral_utils import (
    devstral_supports_on_device_sampling,
    pad_input_ids_and_positions_for_tt_prefill,
    tt_alloc_decode_input_buffers,
    tt_capture_decode_trace,
    tt_execute_decode_trace,
    tt_lm_head_logits_block,
    tt_read_decode_traced_hidden,
    tt_read_decode_traced_logits,
    tt_read_decode_traced_token,
    tt_release_decode_trace,
    tt_sampling_output_token_id,
    tt_update_decode_input_buffers,
)
from models.experimental.devstarl2_small.devstral_utils.chat_reference import REFERENCE_GENERATE_KWARGS
from models.experimental.devstarl2_small.tt.tt_devstral2_small_model import TtDevstral2SmallModel
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.lm_head import LMHead
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)

_DEFAULT_MODEL_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
_DEMO_DIR = Path(__file__).resolve().parent
_RES_DIR = _DEMO_DIR.parent / "resource"

MODEL_LOADING_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe what you see in this image."},
        ],
    }
]


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
    return _RES_DIR / "sample.jpeg"


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
    """Pad merged embeddings to TT prefill length and call ``forward_prefill_from_embeddings``.

    Uploads ``[1,1,S,D]`` TILE tensors matching embed path rank."""
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
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found at {image_path} (default is resource/sample.jpeg). "
            "Add sample.jpeg under models/experimental/devstarl2_small/resource/."
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    image = Image.open(image_path).convert("RGB")
    image = _prepare_vision_image(image, vision_max_edge, vision_square_pixels)

    prompt = processor.apply_chat_template(
        MODEL_LOADING_MESSAGES,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]


def _devstral_bh_qb_decoders_precision(args):
    """BH-QB tuned per-decoder precision recipe for Devstral-Small-2.

    Builds on `_default_settings` (BFP8 everywhere, HIFI2 for attention decode)
    and overrides only what makes a measurable difference at decode batch=1
    on a 4-chip Blackhole Quiet Box:

    * MLP FF1/FF3 → BFP4 + LoFi   (largest weight surface; SwiGLU is robust)
    * MLP FF2     → BFP8 + HIFI2_FP16
    * WQKV / WO   → BFP8 + HIFI2  (vs. accuracy template's BF16 + HIFI4)
    * KV cache    → BFP8          (halves SDPA-decode bandwidth)
    * SDPA decode → HIFI2_NA      (drops packer_l1_acc + fp32 accum, fine at decode)

    Returned object is a callable consumed by ``ModelArgs(..., optimizations=...)``.
    """
    settings = {
        "TensorPrecision": {
            TensorGroup.FF1_FF3: PrecisionSetting.BFP4,
            TensorGroup.FF2: PrecisionSetting.BFP8,
            TensorGroup.WQKV: PrecisionSetting.BFP8,
            TensorGroup.WO: PrecisionSetting.BFP8,
            TensorGroup.KV_CACHE: PrecisionSetting.BFP8,
        },
        "OpFidelity": {
            OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI,
            OpGroup.LI_FF2: MathFidelitySetting.HIFI2_FP16,
            OpGroup.LI_QKV_DECODE: MathFidelitySetting.HIFI2,
            OpGroup.SDPA_DECODE: MathFidelitySetting.HIFI2_NA,
            OpGroup.LI_O_DECODE: MathFidelitySetting.HIFI2,
        },
    }
    conf = ModelOptimizations(settings)
    conf.__name__ = "devstral_bh_qb_perf"
    inst = DecodersPrecision(num_decoders=args.n_layers, model_name=args.model_name, decoder_conf=conf)
    inst.__name__ = "devstral_bh_qb_perf"
    return inst


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
    clear_weight_cache: bool = False,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"TT multimodal path requires an image file; missing {image_path}.")

    if clear_weight_cache:
        # SAFETY: this flag refuses to run unless ``TT_CACHE_PATH`` is set to an
        # explicit, fully-resolved path that is NOT the project root, the user
        # home, or the filesystem root. A previous version of this block fell
        # back to ``Path("")`` (which resolves to ``PosixPath('.')`` — the CWD)
        # and ran ``shutil.rmtree('.', ignore_errors=True)``, silently wiping
        # the project tree. We do not repeat that mistake; manual deletion is
        # safer than any fallback we could pick automatically here.
        import shutil

        raw = os.environ.get("TT_CACHE_PATH", "").strip()
        if not raw:
            raise RuntimeError(
                "--clear-weight-cache requires TT_CACHE_PATH to be set explicitly. "
                "Refusing to guess a cache directory: a wrong guess can delete the "
                "project tree. To clear the cache manually run: rm -rf $TT_CACHE_PATH"
            )
        cache_root = Path(raw).expanduser().resolve()
        forbidden = {Path("/").resolve(), Path.home().resolve(), Path.cwd().resolve()}
        if cache_root in forbidden or cache_root.parent == cache_root:
            raise RuntimeError(
                f"--clear-weight-cache refusing to delete {cache_root}: " "matches a system / home / CWD path."
            )
        if not cache_root.exists():
            logger.info(f"--clear-weight-cache: {cache_root} does not exist; nothing to do.")
        else:
            logger.warning(f"--clear-weight-cache: removing {cache_root}")
            # NOT ``ignore_errors=True`` — surface real failures instead of silent partial wipes.
            shutil.rmtree(cache_root)

    os.environ["HF_MODEL"] = model_id
    _tt_demo.apply_devstral_hf_trust_patches()

    ref_do_sample = bool(REFERENCE_GENERATE_KWARGS["do_sample"])
    do_sample = ref_do_sample if not greedy else False
    gen_temperature = temperature if not greedy else float(REFERENCE_GENERATE_KWARGS["temperature"])

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")
    image = _prepare_vision_image(image, vision_max_edge, vision_square_pixels)
    prompt = processor.apply_chat_template(
        MODEL_LOADING_MESSAGES,
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

    prompt_len = int(input_ids.shape[1])
    extra_tokens = max(0, max_new_tokens)
    need = prompt_len + extra_tokens + 2048
    max_seq = max(4096, need)
    # Round max_seq so SDPA decode k_chunk_size (pow2 divisor of seq, cap 512) stays ≥512.
    max_seq = ((max_seq + 511) // 512) * 512

    mesh_device = _tt_demo.open_devstral_demo_mesh(max(1, min(mesh_width, ttnn.get_num_devices())))
    try:
        dtype_tt = ttnn.bfloat16
        # LM head matmul fires every decode token and is the largest single weight
        # read per step (vocab × hidden). BFP8 storage halves DRAM traffic with no
        # measurable quality loss on LM heads of this size.
        #
        # NOTE: embedding *cannot* use BFP8 — ``ttnn.embedding`` requires a
        # ROW_MAJOR_LAYOUT weights table, while bfloat8_b / bfloat4_b are only
        # legal in TILE_LAYOUT (asserts in py_to_tt_tensor.cpp). The embedding
        # row read is also a per-token cost of just ~10 KB, so storing the table
        # in BFP8 wouldn't move the needle even if it were allowed.
        lm_head_dtype = ttnn.bfloat8_b
        embed_dtype = ttnn.bfloat16

        model_args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq,
            dummy_weights=False,
            use_hf_rope=True,
            cache_hf=True,
            optimizations=_devstral_bh_qb_decoders_precision,
        )
        # Confirm the precision recipe survives ``ModelArgs`` construction.
        # If this prints `accuracy` or `performance`, the recipe never reached
        # ``decoders_optimizations`` and every weight will load at default
        # (usually BF16 for attention) — that's the most common cause of
        # "I switched to BFP8 but tok/s didn't move".
        _opt = model_args.optimizations
        logger.info(f"Active precision recipe: {getattr(_opt, '__name__', repr(_opt))}")
        # Print the per-layer-0 ``ModelOptimizations`` map so we can see exactly
        # which tensor groups are BFP4/BFP8/BF16 and which math fidelity each
        # decode op uses. Look for ``wqkv: BFP8`` / ``kv_cache: BFP8`` /
        # ``ff1_ff3: BFP4`` here — if any of those are ``bf16`` then the recipe
        # didn't apply.
        try:
            layer0_opt = _opt.decoder_optimizations[0]
            logger.info(f"Layer 0 tensor precisions:  {layer0_opt._names['TensorPrecision']}")
            logger.info(f"Layer 0 op fidelities:      {layer0_opt._names['OpFidelity']}")
        except (AttributeError, KeyError, IndexError) as e:
            logger.warning(f"Could not introspect layer 0 optimizations: {e}")
        # Confirm we're really on 4 chips. A common pitfall is launching with
        # ``--mesh-width 4`` while only 1 chip is visible (no env / probe);
        # ``open_devstral_demo_mesh`` then clamps to 1 and you get a fully
        # serial single-chip run masquerading as 4-chip TP.
        _mesh_shape = list(mesh_device.shape)
        _num_devices = int(mesh_device.get_num_devices())
        logger.info(
            f"Mesh shape: {_mesh_shape}, num devices: {_num_devices}, num_devices(model_args): {model_args.num_devices}"
        )
        # Surface the actual ethernet link count the fabric exposes for this
        # mesh shape, so we can tell whether the ``num_links=self.tt_ccl.get_num_links()``
        # bump in ``tt_ministral3_decoder_layer.forward_decode`` (5x per token
        # in DECODE, 2x per token in PREFILL) actually picked up >1 links. If
        # this prints ``1`` everywhere then the BH-QB submesh only exposes a
        # single link in this configuration and option B was a no-op; the
        # decode-time CCL bottleneck must then be attacked via reduce-scatter
        # / async overlap instead.
        try:
            from models.tt_transformers.tt.ccl import get_num_links as _get_num_links

            _nl_any = _get_num_links(mesh_device, None)
            _nl_ax0 = _get_num_links(mesh_device, 0)
            _nl_ax1 = _get_num_links(mesh_device, 1)
            logger.info(
                f"Fabric links: cluster_axis=None -> {_nl_any}, "
                f"cluster_axis=0 (NS) -> {_nl_ax0}, cluster_axis=1 (EW) -> {_nl_ax1}"
            )
        except Exception as _e:  # pragma: no cover - diagnostic only
            logger.warning(f"Could not query fabric link count: {_e}")
        # The two model paths use different multi-chip tensor conventions at the
        # norm boundary:
        #   * PREFILL: ``forward_prefill`` in ``tt_ministral3_decoder_layer.py``
        #     already issues an ``all_gather(dim=3)`` after attention and after
        #     MLP, so the residual is full-width on every chip when it reaches
        #     RMSNorm. We must use the replicated norm (full gamma).
        #   * DECODE: tensors stay width-fractured across chips (the residual
        #     mem config slices by ``dim/num_devices``), so RMSNorm input is
        #     1/N of the full hidden dim per chip. We must use the distributed
        #     norm (auto-sharded gamma + all-gather of stats).
        # Single chip: both modes use the replicated path (no CCL needed).
        model_args.is_distributed_norm = types.MethodType(
            lambda self, mode: self.is_multichip and mode == Mode.DECODE,
            model_args,
        )
        meta_state_dict = model_args.load_state_dict()

        if text_layers is not None:
            if text_layers < 1 or text_layers > model_args.full_model_n_layers:
                raise ValueError(f"--text-layers must be in [1, {model_args.full_model_n_layers}], got {text_layers}")
            model_args.n_layers = text_layers

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
            embed_dtype=embed_dtype,
        )
        logger.info(f"TT embedding weight dtype: {embed_dtype} (BFP8 unsupported for ROW_MAJOR embedding tables).")

        # ── Ground-truth: what dtype did the layer-0 weights actually load with? ──
        # This is the definitive check. If you see ``bfloat16`` for w1/wqkv after
        # asking for BFP4/BFP8 in the recipe, the recipe never reached the layer
        # constructors. If you see the expected narrower dtypes, the recipe is
        # active and any tok/s shortfall is somewhere else (CCL, sampling, etc.).
        try:
            l0 = tt_devstral.language_model.layers[0]
            mlp = getattr(l0, "feed_forward", None) or getattr(l0, "mlp", None)
            attn = getattr(l0, "attention", None) or getattr(l0, "attn", None)
            logger.info(
                "Layer 0 actual weight dtypes:  "
                f"w1={getattr(getattr(mlp, 'w1', None), 'dtype', '?')}  "
                f"w2={getattr(getattr(mlp, 'w2', None), 'dtype', '?')}  "
                f"w3={getattr(getattr(mlp, 'w3', None), 'dtype', '?')}  "
                f"wqkv={getattr(getattr(attn, 'wqkv', None), 'dtype', '?')}  "
                f"wo={getattr(getattr(attn, 'wo', None), 'dtype', '?')}  "
                f"kv_cache={getattr(getattr(attn, 'layer_past', [None])[0], 'dtype', '?')}"
            )
        except Exception as e:  # pragma: no cover - diagnostic only
            logger.warning(f"Could not introspect layer 0 actual weight dtypes: {e}")

        pos_vision = _vision_position_ids_tt(hf_inner, pixel_values, image_sizes_list, mesh_device)
        img_tt = tt_devstral.get_projected_image_features(pixel_values, image_sizes_list, pos_vision)
        ttnn.deallocate(pos_vision)

        img_torch = ttnn.to_torch(img_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
        ttnn.deallocate(img_tt)
        while img_torch.dim() > 2:
            img_torch = img_torch.squeeze(0)
        img_rows = img_torch.reshape(-1, img_torch.shape[-1]).contiguous()

        sd_prefix = model_args.get_state_dict_prefix("", None)
        out_key = f"{sd_prefix}output.weight"
        if out_key not in meta_state_dict:
            raise RuntimeError(f"Missing {out_key!r} in meta state dict (required for LM head).")

        lm_head_weight_cpu: torch.Tensor | None = None
        tt_lm_head: LMHead | None = None
        if lm_head_cpu:
            lm_head_weight_cpu = meta_state_dict[out_key].detach().to(torch.bfloat16).cpu().contiguous()
        else:
            lm_head_max_cols = _tt_demo.demo_lm_head_max_columns_per_device(model_args, cli_cap=lm_head_max_device_cols)
            tt_lm_head = LMHead(
                args=model_args,
                mesh_device=mesh_device,
                tt_ccl=TT_CCL(mesh_device),
                dtype=lm_head_dtype,
                state_dict=meta_state_dict,
                state_dict_prefix=sd_prefix,
                weight_cache_path=model_args.weight_cache_path(lm_head_dtype),
                max_columns_per_device=lm_head_max_cols,
            )
            logger.info(f"TT LM head weight dtype: {lm_head_dtype}.")

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

        tokenizer = MistralCommonBackend.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        else:
            pad_token_id = int(pad_token_id)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        id_device = input_ids.device
        eos_ids = _tt_demo.eos_token_ids(hf_full.config, tokenizer)
        current_ids = input_ids.clone()

        emb_layer = hf_inner.get_input_embeddings()
        pad_row = emb_layer(torch.tensor([[pad_token_id]], device=emb_layer.weight.device, dtype=torch.long))[
            0, 0
        ].detach()

        tt_lm = tt_devstral.language_model

        # Timing: TTFT = wall prefill→first token; first_traced_step_s shows first execute_trace overhead.
        stats = {
            "merge_s": 0.0,
            "prefill_s": 0.0,
            "first_sample_s": 0.0,
            "decode_s": 0.0,
            "lmhead_s": 0.0,
            "sample_post_s": 0.0,
            "trace_capture_s": 0.0,
            "steps": 0,
            "ttft_s": None,
            "first_traced_step_s": None,
            "wall_s": 0.0,
        }

        def _sample_from_tt_out(tt_out, seq_last_idx):
            """Sample next token from prefill hidden states at ``seq_last_idx``.

            Returns the sampled ``[1, 1]`` token tensor and accumulates lm-head /
            sample timings into ``stats``. Used only for the prefill-side
            first-token sample; decode-loop sampling runs inside the trace.
            """
            if sampling is not None:
                assert tt_lm_head is not None
                tok_slot = seq_last_idx % 32
                sampling.seed_manager.get_new_values()
                t0 = time.perf_counter()
                logits_tt = tt_lm_head_logits_block(tt_out, seq_last_idx, model_args, tt_lm_head)
                stats["lmhead_s"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                sample_result = sampling.sample(logits_tt, enable_trace=False)
                tt_next = sample_result[0] if isinstance(sample_result, tuple) else sample_result
                next_scalar = tt_sampling_output_token_id(tt_next, tok_slot)
                ttnn.deallocate(logits_tt)
                ttnn.deallocate(tt_out)
                nid = torch.tensor([[next_scalar]], device=id_device, dtype=torch.long)
                stats["sample_post_s"] += time.perf_counter() - t0
            else:
                t0 = time.perf_counter()
                if lm_head_cpu:
                    assert lm_head_weight_cpu is not None
                    logits_row = _tt_demo.cpu_lm_head_logits_last_token(
                        tt_out, seq_last_idx, mesh_device, lm_head_weight_cpu, int(model_args.vocab_size)
                    )
                else:
                    assert tt_lm_head is not None
                    logits_row = _tt_demo.tt_lm_head_logits_last_token(
                        tt_out, seq_last_idx, mesh_device, model_args, tt_lm_head
                    )
                stats["lmhead_s"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                ttnn.deallocate(tt_out)
                if do_sample:
                    probs = torch.softmax(logits_row.float().squeeze(0) / max(gen_temperature, 1e-6), dim=-1)
                    nid = torch.multinomial(probs, num_samples=1).view(1, 1)
                else:
                    nid = logits_row.argmax(dim=-1, keepdim=True)
                nid = nid.to(id_device)
                stats["sample_post_s"] += time.perf_counter() - t0
            return nid

        # Optional Tracy signposts (best-effort import; skip when Tracy unavailable).
        try:
            from tracy import signpost as _profiler_signpost  # type: ignore
        except Exception:  # pragma: no cover - optional dependency

            def _profiler_signpost(_name: str) -> None:
                return None

        # Prefill path: merge embeddings, KV fill, first-token sample.
        run_t0 = time.perf_counter()
        sl = int(current_ids.shape[1])

        _profiler_signpost("prefill-start")
        t0 = time.perf_counter()
        merged = _merge_image_into_text_embeds(hf_inner, current_ids, img_rows, image_token_id)
        merged_bf = merged.to(torch.bfloat16)
        stats["merge_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        tt_out = _tt_prefill_from_merged_embeds(
            current_ids, merged_bf, pad_row, pad_token_id, mesh_device, tt_lm, model_args, sl
        )
        stats["prefill_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        next_id = _sample_from_tt_out(tt_out, sl - 1)
        stats["first_sample_s"] = time.perf_counter() - t0
        _profiler_signpost("prefill-end")

        # TTFT = wall time from start of prefill to first new token on host.
        stats["ttft_s"] = time.perf_counter() - run_t0
        stats["steps"] = 1  # first token comes from sampling the prefill output

        # ── Per-step state kept entirely in Python primitives ────────────────
        # ``next_id_scalar`` is the only host-side representation of the most
        # recent token. We append generated tokens into ``generated_ids`` once
        # at the end (single ``torch.cat``) so the hot path doesn't pay for an
        # O(prompt_len + steps) tensor reallocation on every step.
        eos_set = set(eos_ids) if eos_ids else set()
        next_id_scalar = int(next_id.item())
        generated_ids: list[int] = [next_id_scalar]
        decode_pos = int(current_ids.shape[1])  # position of THIS first generated token

        # Decode trace capture (warmup inside helper).
        decode_trace_ctx = None
        if max_new_tokens > 1 and next_id_scalar not in eos_set:
            decode_buffers = tt_alloc_decode_input_buffers(mesh_device)
            tt_update_decode_input_buffers(mesh_device, decode_buffers, next_id_scalar, decode_pos)
            logger.info("Capturing decode trace (warmup + capture)…")
            _profiler_signpost("trace-capture-start")
            t_capture = time.perf_counter()
            decode_trace_ctx = tt_capture_decode_trace(
                mesh_device,
                tt_lm,
                model_args,
                decode_buffers,
                tt_lm_head=None if lm_head_cpu else tt_lm_head,
                sampling=sampling,
            )
            stats["trace_capture_s"] = time.perf_counter() - t_capture
            _profiler_signpost("trace-capture-end")
            logger.info(
                f"Decode trace captured in {stats['trace_capture_s']*1000:.1f} ms (warmup + decode + sampling)."
            )

        # Traced decode loop for remaining tokens.
        try:
            _profiler_signpost("decode-loop-start")
            for _step in range(1, max_new_tokens):
                if next_id_scalar in eos_set:
                    break

                if sampling is not None:
                    # Refresh RNG seeds each step so traced sampling does not repeat one token.
                    sampling.seed_manager.get_new_values()

                step_t0 = time.perf_counter()
                t0 = time.perf_counter()
                tt_update_decode_input_buffers(mesh_device, decode_trace_ctx.buffers, next_id_scalar, decode_pos)
                tt_execute_decode_trace(mesh_device, decode_trace_ctx)
                stats["decode_s"] += time.perf_counter() - t0

                # Blocking read syncs device before observing the sampled token (latency-dominated).
                t0 = time.perf_counter()
                if decode_trace_ctx.output_tokens is not None:
                    next_id_scalar = tt_read_decode_traced_token(decode_trace_ctx, batch_slot=0)
                    stats["sample_post_s"] += time.perf_counter() - t0
                elif decode_trace_ctx.output_logits is not None:
                    logits_row = tt_read_decode_traced_logits(decode_trace_ctx, mesh_device, model_args, batch_slot=0)
                    stats["lmhead_s"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    if do_sample:
                        probs = torch.softmax(logits_row.float().squeeze(0) / max(gen_temperature, 1e-6), dim=-1)
                        next_id_scalar = int(torch.multinomial(probs, num_samples=1).item())
                    else:
                        next_id_scalar = int(logits_row.argmax(dim=-1).item())
                    stats["sample_post_s"] += time.perf_counter() - t0
                else:
                    # CPU LM head: clone trace hidden block, run host LM head + sample.
                    h_clone = tt_read_decode_traced_hidden(decode_trace_ctx, mesh_device, batch_slot=0)
                    nid = _sample_from_tt_out(h_clone, 0)
                    next_id_scalar = int(nid.item())

                if stats["first_traced_step_s"] is None:
                    stats["first_traced_step_s"] = time.perf_counter() - step_t0

                generated_ids.append(next_id_scalar)
                decode_pos += 1
                stats["steps"] += 1
            _profiler_signpost("decode-loop-end")
        finally:
            if decode_trace_ctx is not None:
                tt_release_decode_trace(mesh_device, decode_trace_ctx)

        # Materialize the final token sequence once, outside the hot loop.
        if generated_ids:
            tail = torch.tensor([generated_ids], dtype=current_ids.dtype, device=id_device)
            current_ids = torch.cat([current_ids, tail], dim=1)

        stats["wall_s"] = time.perf_counter() - run_t0

        # Printed timing breakdown for prefill vs traced decode.
        wall_s = stats["wall_s"]
        steps = stats["steps"]
        decode_steps = max(steps - 1, 0)
        ttft_s = stats["ttft_s"] or 0.0

        def _pct(part: float) -> float:
            return 100.0 * part / wall_s if wall_s > 0 else 0.0

        def _decode_avg_ms(part: float) -> float:
            return 1000.0 * part / decode_steps if decode_steps > 0 else 0.0

        decode_loop_total_s = stats["decode_s"] + stats["lmhead_s"] + stats["sample_post_s"]
        steady_per_tok_s = decode_loop_total_s / decode_steps if decode_steps > 0 else 0.0
        steady_tok_s = 1.0 / steady_per_tok_s if steady_per_tok_s > 0 else 0.0
        thr = steps / wall_s if wall_s > 0 else 0.0

        print()
        print("──────────────────────────────────────────────────────────────")
        print(
            f"  TT · traced decode  ({steps} new token(s); {decode_steps} traced decode step(s); wall {wall_s:.2f} s)"
        )
        print("──────────────────────────────────────────────────────────────")
        print(f"  {'Phase':<22} {'total':>14}     %")
        print(f"  {'merge (host)':<22} {stats['merge_s']*1000:>10.2f} ms  {_pct(stats['merge_s']):>5.1f}%")
        print(f"  {'prefill (1x)':<22} {stats['prefill_s']*1000:>10.2f} ms  {_pct(stats['prefill_s']):>5.1f}%")
        print(
            f"  {'first-token sample':<22} {stats['first_sample_s']*1000:>10.2f} ms  {_pct(stats['first_sample_s']):>5.1f}%"
        )
        print(
            f"  {'trace capture (1x)':<22} {stats['trace_capture_s']*1000:>10.2f} ms  {_pct(stats['trace_capture_s']):>5.1f}%   (warmup + decode + sampling)"
        )
        print(
            f"  {'traced decode submit':<22} {stats['decode_s']*1000:>10.2f} ms  {_pct(stats['decode_s']):>5.1f}%"
            f"   (avg {_decode_avg_ms(stats['decode_s']):.2f} ms / decoded tok)"
        )
        print(
            f"  {'lm head (decode)':<22} {stats['lmhead_s']*1000:>10.2f} ms  {_pct(stats['lmhead_s']):>5.1f}%"
            f"   (avg {_decode_avg_ms(stats['lmhead_s']):.2f} ms / decoded tok)"
        )
        print(
            f"  {'sample / post-decode':<22} {stats['sample_post_s']*1000:>10.2f} ms  {_pct(stats['sample_post_s']):>5.1f}%"
            f"   (avg {_decode_avg_ms(stats['sample_post_s']):.2f} ms / decoded tok)"
        )
        print("──────────────────────────────────────────────────────────────")
        print(f"  TTFT (prompt -> 1st new tok)        {ttft_s*1000:>10.2f} ms")
        if stats["first_traced_step_s"] is not None:
            print(f"  First traced decode step latency    {stats['first_traced_step_s']*1000:>10.2f} ms")
        if decode_steps > 0:
            print(f"  Steady-state decode latency / tok   {steady_per_tok_s*1000:>10.2f} ms")
            print(f"  Steady-state decode throughput      {steady_tok_s:>10.3f} tok/s")
        print(f"  End-to-end throughput               {thr:>10.3f} tok/s")
        print("──────────────────────────────────────────────────────────────")

        final_seq_len = int(current_ids.shape[1])
        new_token_count = final_seq_len - prompt_len

        print("  TT · generation done")
        print("──────────────────────────────────────────────────────────────")
        print(f"  Final sequence  {final_seq_len:,} tokens")
        print(f"  New tokens      {new_token_count}")
        print("──────────────────────────────────────────────────────────────")
        print(f"  TT · output  ({new_token_count} new tokens)")
        print("──────────────────────────────────────────────────────────────")

        answer_ids = current_ids[0, prompt_len:]
        answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False)
        print(answer_text)
    finally:
        _tt_demo.close_devstral_demo_mesh(mesh_device)


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
        help="Image path (default: resource/sample.jpeg).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mesh-width", type=int, default=1)
    parser.add_argument("--text-layers", type=int, default=None)
    parser.add_argument(
        "--clear-weight-cache",
        action="store_true",
        help="Delete the TT weight cache before loading. Use after changing the precision recipe "
        "(BFP4/BFP8/BF16) so stale tile files don't bypass re-quantization.",
    )
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
    args = parser.parse_args()

    if args.vision_square_pixels is not None and args.vision_square_pixels <= 0:
        parser.error("--vision-square-pixels must be a positive integer when set.")

    if args.backend == "hf":
        run_hf(
            args.model_id,
            args.image,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            vision_max_edge=args.vision_max_edge,
            vision_square_pixels=args.vision_square_pixels,
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
            clear_weight_cache=args.clear_weight_cache,
        )


if __name__ == "__main__":
    main()
