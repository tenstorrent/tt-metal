# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end LocateAnything-3B demo on chip 2 using the EXPERIMENTAL MTP / Parallel
Box Decoding path (hybrid: MTP blocks + AR fallback) on a real image, with box
visualization. Vision (MoonViT) + LLM both on device; dense KV cache for MTP.

NOTE: MTP is an approximate parallel decoder (see tt/mtp.py) — boxes may differ from
the greedy-AR demo; this shows the model's intended fast (~1.7x) path running on device.

Env: LA_IMAGE (default ../image.png), LA_QUERY (car), LA_OUT (../image_result_mtp.png),
     LA_IN_TOKEN_LIMIT (1024), LA_MAX_NEW (128), HF_MODEL (required).

Run (chip 2 ONLY):
  TT_METAL_HOME=$(pwd) ARCH_NAME=blackhole TT_METAL_VISIBLE_DEVICES=2 MESH_DEVICE=N150 \
    HF_MODEL=/home/ttuser/.cache/locate_anything/LA-Qwen2.5-3B LA_QUERY="car" \
    ./python_env/bin/python -m pytest -svq \
    models/experimental/locate_anything/tests/test_demo_mtp_visualize.py
"""
import os
import sys
import time

import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn
from models.demos.qwen25_vl.tt.common import merge_vision_tokens, preprocess_inputs_prefill
from models.experimental.locate_anything.reference import la_inputs
from models.experimental.locate_anything.tests.bench_locate_anything import (
    IMAGE_TOKEN_INDEX,
    _select_optimizations,
    create_tt_model,
)
from models.experimental.locate_anything.tests import test_mtp as _tm  # noqa: F401 (attaches MTPDecoder.mtp_step_ar)
from models.experimental.locate_anything.tests.test_demo_visualize import parse_detections, visualize
from models.experimental.locate_anything.tt.mtp import MTPDecoder
from models.experimental.locate_anything.tt.vision import MoonViT
from models.tt_transformers.tt.common import Mode


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": False, "trace_region_size": 50000000, "num_command_queues": 1}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
def test_demo_mtp_visualize(mesh_device):
    from transformers import AutoConfig, Qwen2ForCausalLM

    repo_parent = os.path.abspath(os.path.join(os.environ.get("TT_METAL_HOME", "."), ".."))
    image_path = os.environ.get("LA_IMAGE", os.path.join(repo_parent, "image.png"))
    query = os.environ.get("LA_QUERY", "car")
    out_path = os.environ.get("LA_OUT", os.path.join(repo_parent, "image_result_mtp.png"))
    in_token_limit = int(os.environ.get("LA_IN_TOKEN_LIMIT", "1024"))
    max_new = int(os.environ.get("LA_MAX_NEW", "128"))
    # Sampling params: greedy by default (temp 0). Set LA_TEMP=0.7 LA_TOP_P=0.9 LA_REP_PEN=1.1
    # for the model's intended hybrid-MTP sampling (suppresses greedy degeneration).
    gen_kwargs = {
        "temperature": float(os.environ.get("LA_TEMP", "0")),
        "top_p": float(os.environ.get("LA_TOP_P", "1.0")),
        "repetition_penalty": float(os.environ.get("LA_REP_PEN", "1.0")),
    }
    torch.manual_seed(int(os.environ.get("LA_SEED", "0")))
    hf_model_dir = os.environ["HF_MODEL"]
    assert os.path.isfile(image_path), f"image not found: {image_path}"

    mp = la_inputs.find_model_path()
    sys.path.insert(0, mp)
    from generate_utils import get_token_ids_from_config, handle_pattern, sample_tokens

    image = Image.open(image_path).convert("RGB")
    W, H = image.size
    cfg = AutoConfig.from_pretrained(mp, trust_remote_code=True)
    token_ids = get_token_ids_from_config(cfg)
    n_future = 6
    logger.info(f"image={image_path} query={query!r} out={out_path}")

    # --- DENSE-KV LLM (MTP requires a contiguous KV cache) ---
    model_args, model, paged_cfg, _ = create_tt_model(
        mesh_device,
        instruct=False,
        max_batch_size=1,
        optimizations=_select_optimizations,
        max_seq_len=2048,
        page_params=None,
        dtype=ttnn.bfloat8_b,
        use_paged_kv_cache=False,
    )
    assert paged_cfg is None
    tokenizer = model_args.tokenizer
    dense_kv = [l.attention.layer_past for l in model.layers]

    # --- inputs + vision on device ---
    bundle = la_inputs.build_inputs(tokenizer, image, query, in_token_limit=in_token_limit)
    input_ids = bundle["input_ids"]
    attention_mask = bundle["attention_mask"]
    real_seq_len = input_ids.shape[1]
    last_token_idx = real_seq_len - 1
    logger.info(f"grid_hw={bundle['grid_hw']} n_img_tokens={bundle['n_img_tokens']} seq_len={real_seq_len}")

    vis = MoonViT(mesh_device, mp, bundle["grid_hw"], dtype=ttnn.bfloat16)
    vit_proj = vis.forward(bundle["pixel_values"].float()).to(torch.float32)

    # --- host text-embed + merge + prefill into dense KV ---
    hf_model = Qwen2ForCausalLM.from_pretrained(hf_model_dir, torch_dtype=torch.float32)
    embed_tokens = hf_model.get_input_embeddings()
    with torch.no_grad():
        text_embeds = embed_tokens(input_ids)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        pad_embedding = embed_tokens(torch.tensor(pad_id))

    class _MC:
        image_token_id = IMAGE_TOKEN_INDEX

    input_embeds = merge_vision_tokens(input_ids, text_embeds, vit_proj.to(text_embeds.dtype), _MC())
    input_prefill_pt, decoding_pos, _ = preprocess_inputs_prefill(
        [input_embeds[0]], model_args, attention_mask, pad_embedding=pad_embedding
    )
    embeds = input_prefill_pt[0].unsqueeze(0).to(torch.float32)

    model.switch_mode(Mode.PREFILL)
    tokens_embd, rot_mats_global, _, _ = model.prepare_inputs_prefill_embeds(
        embeds, start_pos=0, page_table=None, last_token_idx=last_token_idx
    )
    tt_logits = model.ttnn_prefill_forward(
        tokens_embd,
        rot_mats_global=rot_mats_global,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        get_last_token=(last_token_idx // 32) * 32,
        kv_cache=dense_kv,
    )
    _ = model.process_output_prefill(tt_logits.cpu(), last_token_idx=last_token_idx % 32)
    ttnn.deallocate(tt_logits)
    ttnn.deallocate(tokens_embd)

    # --- MTP hybrid decode (MTP blocks + AR fallback), mirrors test_mtp ---
    mtp = MTPDecoder(model, n_future=n_future)
    mtp.reset_kv_from_prefill(dense_kv, real_seq_len)
    mask_tok = token_ids["default_mask_token_id"]
    im_end = token_ids["im_end_token_id"]
    box_end = token_ids["box_end_token_id"]

    full_ids = input_ids[0].tolist()
    gen_ids = []
    cached_len = real_seq_len
    forward_passes = 0
    cur_mode = "mtp"

    def embed(ids_list):
        with torch.no_grad():
            return embed_tokens(torch.tensor([ids_list])).to(torch.float32)

    t0 = time.time()
    while len(full_ids) < real_seq_len + max_new:
        cur_len = len(full_ids)
        uncached_len = cur_len - cached_len
        if cur_mode == "mtp":
            uncached = full_ids[cached_len:]
            win_ids = uncached + [full_ids[-1]] + [mask_tok] * (n_future - 1)
            win_pos = list(range(cached_len, cur_len)) + [cur_len - 1] + [cur_len + j for j in range(n_future - 1)]
            logits = mtp.mtp_step(embed(win_ids), win_pos, uncached_len)
            forward_passes += 1
            logits6 = logits[-n_future:].unsqueeze(0)
            _, _, x0, box_avg = sample_tokens(
                logits6, torch.tensor([full_ids]), token_ids, keep_k=5, generation_mode="hybrid", **gen_kwargs
            )
            nt = x0[0] if bool((box_avg[0] == 0).all()) else box_avg[0]
            op = handle_pattern(nt, token_ids, "hybrid")
            cached_len = cur_len
            for t in [int(x) for x in op["tokens"]]:
                gen_ids.append(t)
                full_ids.append(t)
            if op["type"] == "im_end":
                break
            if op["type"] == "error_box":
                cur_mode = "ar"
        else:
            uncached = full_ids[cached_len:]
            logits = mtp.mtp_step_ar(embed(uncached), list(range(cached_len, cur_len)))
            forward_passes += 1
            _, _, x0, _ = sample_tokens(
                logits[-1:].unsqueeze(0), torch.tensor([full_ids]), token_ids, generation_mode="hybrid", **gen_kwargs
            )
            tv = int(x0[0, 0].item())
            cached_len = cur_len
            gen_ids.append(tv)
            full_ids.append(tv)
            if tv == im_end:
                break
            if tv == box_end:
                cur_mode = "mtp"
    dt = time.time() - t0

    answer = tokenizer.decode(gen_ids, skip_special_tokens=False)
    print(f"RAW ANSWER (MTP): {answer}")
    print(f"mtp_forward_passes={forward_passes} tokens={len(gen_ids)} decode_time_s={dt:.2f}")
    dets = parse_detections(answer, W, H)
    n_box = sum(1 for d in dets if "box" in d)
    print(f"PARSED detections: {n_box} boxes")
    for d in dets:
        print(f"  {d}")
    saved = visualize(image, dets, out_path)
    print(f"SAVED visualization -> {saved}")
    print(f"num_detections={len(dets)}")
    assert os.path.isfile(saved)
