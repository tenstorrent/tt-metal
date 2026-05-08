# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo / smoke test using the **same user prompt** as ``reference/model_loading.py``:

- Multimodal message: one image placeholder + text
  ``"Describe what you see in this image."``

**HF path** (``--backend hf``, default): matches ``model_loading.py`` — ``AutoProcessor``,
``AutoModelForImageTextToText``, ``sample.jpeg``, ``max_new_tokens=100``.

**TT path** (``--backend tt``): loads :class:`TtDevstral2SmallModel` (Pixtral vision + projector +
``TtMinistral3Model``), runs TT vision/projector, merges features into text embeddings like HF
``masked_scatter`` on ``image_token_id``, then ``language_model.forward_prefill_from_embeddings``.
Requires ``reference/sample.jpeg`` (same as HF). Pixtral vision QKV L1 grows with patch-token count;
default ``--vision-max-edge 336`` resizes the PIL image before ``processor`` (similar to
``test_devstral2_small`` ~24×24 patches when patch size is 14). Use ``0`` for full processor sizing
(may exceed Wormhole L1).

Usage (repo root)::

    python models/experimental/devstarl2_small/demo/demo_model_loading_prompt.py

    python models/experimental/devstarl2_small/demo/demo_model_loading_prompt.py --backend tt --seed 0

    python models/experimental/devstarl2_small/demo/demo_model_loading_prompt.py --backend tt \\
        --image path/to.jpg --text-layers 1 --max-new-tokens 16 --lm-head-cpu
"""

from __future__ import annotations

import argparse
import os
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
from models.experimental.devstarl2_small.demo import demo_devstral2_tt_multimodal as _tt_demo
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
    input_ids_pad, position_ids_pad, _ = _tt_demo._pad_input_ids_and_positions_for_tt_prefill(
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
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found at {image_path} (same path as reference/model_loading.py). "
            "Add sample.jpeg under models/experimental/devstarl2_small/reference/."
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

    logger.info(f"HF multimodal generate (max_new_tokens={max_new_tokens}) on {device} …")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    logger.info(f"HF output:\n{output_text}")


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
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"TT multimodal path requires an image file; missing {image_path}.")

    os.environ["HF_MODEL"] = model_id
    _tt_demo._apply_devstral_hf_trust_patches()

    ref_do_sample = bool(REFERENCE_GENERATE_KWARGS["do_sample"])
    do_sample = ref_do_sample if not greedy else False
    gen_temperature = temperature if not greedy else float(REFERENCE_GENERATE_KWARGS["temperature"])

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    image = Image.open(image_path).convert("RGB")
    orig_sz = image.size
    image = _resize_image_max_edge(image, vision_max_edge)
    if image.size != orig_sz:
        logger.info(
            f"TT vision: resized image {orig_sz} -> {image.size} "
            f"(max_edge={vision_max_edge}; Pixtral QKV L1 safe). Use --vision-max-edge 0 to skip."
        )
    elif vision_max_edge > 0:
        logger.info(f"TT vision: image within max_edge={vision_max_edge} ({image.size}), no resize.")
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
    max_seq = max(4096, prompt_len + extra_tokens + 2048)

    mesh_device = _tt_demo._open_mesh(max(1, min(mesh_width, ttnn.get_num_devices())))
    try:
        dtype_tt = ttnn.bfloat16

        logger.info("Loading checkpoint via ModelArgs.load_state_dict() …")
        model_args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq,
            dummy_weights=False,
            use_hf_rope=True,
            cache_hf=True,
        )
        model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)
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
        )

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
            logger.info(
                f"CPU LM head: chunked torch matmul; weight {tuple(lm_head_weight_cpu.shape)} {lm_head_weight_cpu.dtype}."
            )
        else:
            lm_head_max_cols = _tt_demo._demo_lm_head_max_columns_per_device(
                model_args, cli_cap=lm_head_max_device_cols
            )
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
        eos_ids = _tt_demo._eos_token_ids(hf_full.config, tokenizer)
        current_ids = input_ids.clone()
        mode = "greedy" if greedy else f"sample (T={gen_temperature})"
        lm_mode = "CPU lm_head (chunked torch)" if lm_head_cpu else "TT lm_head"
        logger.info(
            f"TT TtDevstral2SmallModel: up to {max_new_tokens} new tokens, {mode}; {lm_mode}; "
            f"image {image_path} + describe prompt."
        )

        emb_layer = hf_inner.get_input_embeddings()
        pad_row = emb_layer(torch.tensor([[pad_token_id]], device=emb_layer.weight.device, dtype=torch.long))[
            0, 0
        ].detach()

        tt_lm = tt_devstral.language_model

        for _step in range(max_new_tokens):
            sl = int(current_ids.shape[1])
            merged = _merge_image_into_text_embeds(hf_inner, current_ids, img_rows, image_token_id)
            merged_bf = merged.to(torch.bfloat16)
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

            if lm_head_cpu:
                assert lm_head_weight_cpu is not None
                logits_row = _tt_demo._cpu_lm_head_logits_last_token(
                    tt_out, sl - 1, mesh_device, lm_head_weight_cpu, int(model_args.vocab_size)
                )
            else:
                assert tt_lm_head is not None
                logits_row = _tt_demo._tt_lm_head_logits_last_token(tt_out, sl - 1, mesh_device, model_args, tt_lm_head)
            ttnn.deallocate(tt_out)

            if do_sample:
                probs = torch.softmax(logits_row.float().squeeze(0) / max(gen_temperature, 1e-6), dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).view(1, 1)
            else:
                next_id = logits_row.argmax(dim=-1, keepdim=True)
            next_id = next_id.to(id_device)
            if eos_ids and int(next_id.item()) in eos_ids:
                break
            current_ids = torch.cat([current_ids, next_id], dim=1)

        answer_ids = current_ids[0, prompt_len:]
        answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False)
        logger.info(f"TT generated ({answer_ids.numel()} tokens):\n{answer_text}")
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
        "--vision-max-edge",
        type=int,
        default=336,
        help="TT only: max longest image side (px) before processor; ~14*24=336 caps patch grid ~24². "
        "0 = use processor default sizing (may exceed Wormhole L1 in Pixtral attention).",
    )
    args = parser.parse_args()

    if args.backend == "hf":
        run_hf(
            args.model_id,
            args.image,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
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
        )


if __name__ == "__main__":
    main()
