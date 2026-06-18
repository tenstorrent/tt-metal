# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tokenizer front-end bridge: a real prompt -> HunyuanTokenizer -> input_ids ->
# on-device wte embedding, validated against the reference torch embedding.
# Replaces the synthetic text_embeds used by the pipeline tests with embeddings
# of an actual prompt. Confirms the tokenizer output is pipeline-compatible:
# a contiguous image-token span the denoise step already handles.
#
# Requires ref/tokenizer/assets/tokenizer.json (copy from the model download).
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_tokenizer_frontend.py -v -s

import sys, json, glob
import torch
import torch.nn.functional as F
from safetensors import safe_open
from loguru import logger

ROOT = "/home/iguser/Christy/tt-metal"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import prepare_gen_image_inputs

PROMPT = "a photo of a cat"
PCC_THR = 0.999

_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]


def _load(key):
    shard = _WMAP[key]
    with safe_open(f"{WEIGHTS}/{shard}", framework="pt") as f:
        return f.get_tensor(key)


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def test_tokenizer_to_wte_embedding(device):
    tok = HunyuanTokenizer.from_pretrained()
    bundle = prepare_gen_image_inputs(tok, PROMPT, image_size=1024)
    input_ids = bundle.input_ids  # [cfg_rows, S] int64
    span = bundle.rope_image_info[0][0][0]  # the image-token slice
    grid = bundle.rope_image_info[0][0][1]
    logger.info(
        f"prompt={PROMPT!r} -> input_ids {tuple(input_ids.shape)} seq_len={bundle.seq_len} "
        f"cfg_factor={bundle.cfg_factor} image_span={span} grid={grid}"
    )
    # The image span must be contiguous (what HunyuanTtDenoiseStep scatters into).
    assert grid[0] * grid[1] == (span.stop - span.start), "image span size != grid_h*grid_w"

    wte_w = _load("model.wte.weight")  # [V, H]
    V, H = wte_w.shape
    ref_emb = F.embedding(input_ids, wte_w.float())  # [rows, S, H]

    # On-device wte (mirrors HunyuanTtModel.embed).
    emb_w = ttnn.from_torch(wte_w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    ids_tt = ttnn.from_torch(input_ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    emb_tt = ttnn.embedding(ids_tt, emb_w, layout=ttnn.TILE_LAYOUT)
    emb = ttnn.to_torch(emb_tt).reshape(input_ids.shape[0], input_ids.shape[1], H)

    pcc = _pcc(ref_emb, emb)
    logger.info(f"tokenizer->wte embedding PCC={pcc:.6f}  (vocab={V}, H={H})")
    assert pcc >= PCC_THR, f"PCC {pcc:.6f} < {PCC_THR}"
