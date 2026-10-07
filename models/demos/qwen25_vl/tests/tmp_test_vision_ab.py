# TEMPORARY A/B experiment (not for commit): vision tower PCC vs HF on demo.jpeg under mask variants.
import itertools
import json
import os

import pytest
import torch
from loguru import logger
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen25_vl.tt.common import get_hf_visual
from models.demos.qwen25_vl.tt.model import DropInVisionTransformer
from models.demos.qwen25_vl.tt.model_config import VisionModelArgs, qwen25_vl_mesh_shape
from models.tt_transformers.tt.model_config import DecodersPrecision

VARIANTS = {
    "batched_off": {"TT_QWEN25_VL_DISABLE_BATCHED_WINDOW_ATTENTION": "1"},
    "batched_off_kvbf16": {"TT_QWEN25_VL_DISABLE_BATCHED_WINDOW_ATTENTION": "1", "TT_QWEN25_VL_VISION_KV_BF16": "1"},
}
KNOBS = [
    "TT_QWEN25_VL_DISABLE_BATCHED_WINDOW_ATTENTION",
    "TT_QWEN25_VL_MASK_FINITE",
    "TT_QWEN25_VL_MASK_BF16",
    "TT_QWEN25_VL_VISION_KV_BF16",
]


@pytest.mark.parametrize("mesh_device", [qwen25_vl_mesh_shape()], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_vision_ab(mesh_device, qwen25_vl_mesh_device):
    mesh_device = qwen25_vl_mesh_device
    model_id = os.environ["HF_MODEL"]
    prompt_file = os.environ.get("AB_PROMPT", "models/demos/qwen25_vl/demo/sample_prompts/demo.json")
    conv = json.load(open(prompt_file))
    conv = conv if isinstance(conv[0], dict) else conv[0]

    config = Qwen2_5_VLForConditionalGeneration.config_class.from_pretrained(model_id)
    ref = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, config=config, torch_dtype="auto")
    visual = get_hf_visual(ref)
    processor = AutoProcessor.from_pretrained(model_id)
    text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(conv)
    inputs = processor(text=[text], images=imgs, videos=vids, return_tensors="pt")
    logger.info(f"grid_thw={inputs.image_grid_thw.tolist()} pixel_values={tuple(inputs.pixel_values.shape)}")

    with torch.no_grad():
        ref_out = visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw)
        # transformers 5.x: pooler_output is the merged [tokens, out_hidden_size] tensor
        ref_out = getattr(ref_out, "pooler_output", ref_out).float()

    args = VisionModelArgs(
        mesh_device,
        max_batch_size=1,
        max_seq_len=4096,
        optimizations=DecodersPrecision.accuracy(config.vision_config.depth, model_id),
    )
    args.hf_config.vision_config.depth = config.vision_config.depth
    towers = {"w_bfp8": DropInVisionTransformer(visual, args, debug=False)}
    if os.environ.get("AB_BF16_WEIGHTS") == "1":
        towers["w_bf16"] = DropInVisionTransformer(visual, args, dtype=ttnn.bfloat16, debug=False)

    outs = {}
    for (wname, tt_visual), (vname, env) in itertools.product(towers.items(), VARIANTS.items()):
        name = f"{wname}/{vname}"
        for k in KNOBS:
            os.environ.pop(k, None)
        os.environ.update(env)
        with torch.no_grad():
            out = tt_visual(inputs.pixel_values, grid_thw=inputs.image_grid_thw).float()
        outs[name] = out
        _, pcc = comp_pcc(ref_out, out, 0.99)
        per_tok = torch.nn.functional.cosine_similarity(ref_out, out, dim=-1)
        worst = torch.topk(-per_tok, 8)
        logger.info(
            f"[AB] {name}: {pcc} | max_abs_diff={(ref_out - out).abs().max():.4f} "
            f"| per-token cos: min={per_tok.min():.4f} mean={per_tok.mean():.5f} "
            f"| n(cos<0.9)={(per_tok < 0.9).sum().item()} worst_idx={worst.indices.tolist()}"
        )
    for k in KNOBS:
        os.environ.pop(k, None)
