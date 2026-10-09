# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU only (no device): the host-side per-image tables and the weight adapter equal HF exactly.

- ``inputs.host_tables`` pos embed / rotary cos, sin == the golden's HF values bit for bit (fp32);
- the BF16 pos-embed add (``patch_embed + pos.to(bf16)``) on the golden patch embed == golden block-0 input;
- weight adapter: padded / permuted q, k, v, proj, fc1, fc2 reproduce the HF Linear outputs on a
  random input (fp32 math), and the strict key check rejects a missing or extra tensor.
"""

from functools import lru_cache

import pytest
import torch

from models.demos.pplx_decider_v1_27b.tests.vision.vision_test_utils import (
    IMAGES,
    golden_inputs,
    golden_tower,
    image_ids,
    reader,
)
from models.demos.pplx_decider_v1_27b.tt.optimizations import VisionPrecisionPolicy
from models.demos.pplx_decider_v1_27b.tt.rope import rope_head_permutation
from models.demos.pplx_decider_v1_27b.tt.vision.config import PplxVisionArgs
from models.demos.pplx_decider_v1_27b.tt.vision.inputs import host_tables
from models.demos.pplx_decider_v1_27b.tt.vision.weights import VISION_PREFIX, build_block_weights, check_vision_state


@lru_cache(maxsize=1)
def vision_state():
    return reader().tensors_with_prefix(VISION_PREFIX)


@lru_cache(maxsize=1)
def vision_args():
    from transformers import AutoConfig

    return PplxVisionArgs.from_hf_config(AutoConfig.from_pretrained(reader().path, local_files_only=True).vision_config)


@pytest.mark.parametrize("image", IMAGES, ids=image_ids())
def test_host_tables_match_hf(image):
    golden = golden_tower(image)
    grid = golden_inputs(image)["image_grid_thw"]
    tables = host_tables(grid, vision_args(), vision_state()["pos_embed.weight"])
    for key in ("pos_embed", "rotary_cos", "rotary_sin"):
        assert torch.equal(tables[key], golden[key]), f"{image} {key} differs from HF"
    block0_in = golden["patch_embed"] + tables["pos_embed"].to(torch.bfloat16)
    assert torch.equal(block0_in, golden["embed_with_pos"])


def test_weight_adapter_matches_hf_linear():
    args, state = vision_args(), vision_state()
    check_vision_state(state, args)
    w = build_block_weights(state, 5, args, VisionPrecisionPolicy())
    torch.manual_seed(0)
    x = torch.randn(64, args.hidden_size)
    p = "blocks.5."
    # qkv: TT columns are [q | k | v], 16 heads x 96, q/k head dims permuted.
    hf = x @ state[f"{p}attn.qkv.weight"].float().T + state[f"{p}attn.qkv.bias"].float()
    tt = x @ w.attention.qkv.weight.source.float() + w.attention.qkv.bias.source.float()
    hf = hf.reshape(64, 3, args.num_heads, args.head_dim)
    tt = tt.reshape(64, 3, args.num_heads, args.padded_head_dim)
    perm = rope_head_permutation(args.head_dim, args.rotary_dim)
    for i in range(3):
        expect = hf[:, i][..., perm] if i < 2 else hf[:, i]
        assert torch.equal(tt[:, i, :, : args.head_dim], expect)
        assert not tt[:, i, :, args.head_dim :].any(), "padded head dims must be exactly zero"
    # proj: padded attention output (zeros on dims 72..95) through the padded weight == HF proj.
    attn = torch.randn(64, args.num_heads, args.head_dim)
    attn_pad = torch.nn.functional.pad(attn, (0, args.padded_head_dim - args.head_dim)).reshape(64, -1)
    hf = attn.reshape(64, -1) @ state[f"{p}attn.proj.weight"].float().T
    assert torch.allclose(attn_pad @ w.attention.proj.weight.source.float(), hf, atol=1e-4)
    # MLP: zero-padded intermediate contributes nothing.
    hf = torch.nn.functional.gelu(
        x @ state[f"{p}mlp.linear_fc1.weight"].float().T + state[f"{p}mlp.linear_fc1.bias"].float(), approximate="tanh"
    )
    tt = torch.nn.functional.gelu(
        x @ w.mlp.fc1.weight.source.float() + w.mlp.fc1.bias.source.float(), approximate="tanh"
    )
    # (allclose: fp32 matmuls of different widths block their sums differently)
    assert torch.allclose(tt[:, : args.intermediate_size], hf, atol=1e-5) and not tt[:, args.intermediate_size :].any()
    hf2 = hf @ state[f"{p}mlp.linear_fc2.weight"].float().T
    assert torch.allclose(tt @ w.mlp.fc2.weight.source.float(), hf2, atol=1e-3)


def test_strict_keys(expect_error):
    args, state = vision_args(), dict(vision_state())
    assert len(state) == 333
    extra = dict(state, **{"blocks.0.attn.q_norm.weight": torch.zeros(72)})
    with expect_error(KeyError, "unexpected \\['blocks.0.attn.q_norm.weight'\\]"):
        check_vision_state(extra, args)
    del state["merger.norm.bias"]
    with expect_error(KeyError, "missing \\['merger.norm.bias'\\]"):
        check_vision_state(state, args)
