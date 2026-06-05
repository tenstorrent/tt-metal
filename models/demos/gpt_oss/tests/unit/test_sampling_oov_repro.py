# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Regression test for the GPT-OSS out-of-vocab (OOV) sampling leak.

The gpt_oss lm_head is zero-padded to a power-of-2 per-device vocab width, so the
out-of-vocab columns carry logit == 0 (NOT -inf). With multi-device top-k sampling,
the highest-TP device(s) hold only padding for most of their top-k slots (their real
tokens are the rarely-selected Harmony control tokens), so logit-0 OOV entries enter
the global candidate set. When the real distribution is not sharply peaked, those OOV
entries out-rank real (negative-logit) tokens and get sampled -> token id >= vocab_size.

This isolates the bug from the full model (no weights, no decode hang):
  - no_fix_BUG: logits with logit-0 padding (what the model produces today) -> OOV sampled
  - with_fix:   same logits + ``build_out_of_vocab_logit_mask`` (the model-side fix) -> 0 OOV

The ``with_fix`` case applies the *actual* model fix function, so this test fails if the
fix is removed or broken. Parametrized for QB2 (1x4 / TP=4), LoudBox/T3K (1x8 / TP=8),
and Galaxy (4x8 / TP=8) — it auto-runs whichever shapes fit the current system.

Run (QB2 example):
  ARCH_NAME=blackhole MESH_DEVICE='(1, 4)' \
  TT_MESH_GRAPH_DESC_PATH=tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto \
  pytest models/demos/gpt_oss/tests/unit/test_sampling_oov_repro.py -x -v
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.demos.gpt_oss.tests.test_factory import parametrize_mesh_with_fabric
from models.demos.gpt_oss.tt.model import build_out_of_vocab_logit_mask, compute_per_device_vocab

VOCAB_SIZE = 201088
MAX_TOP_K = 32
BATCH_SIZE = 32


def _make_args(mesh_device):
    class _Args:
        pass

    args = _Args()
    args.vocab_size = VOCAB_SIZE
    num_tp = mesh_device.shape[1]
    per_device_vocab = compute_per_device_vocab(args.vocab_size, num_tp)
    args.padded_vocab_size = per_device_vocab * num_tp
    args.cluster_shape = tuple(mesh_device.shape)
    args.sampling_all_gather_axis = 1
    args.num_devices = mesh_device.get_num_devices()
    args.is_galaxy = mesh_device.shape[0] > 1
    args.model_config = {}
    args.sampling_dp = 1
    args.max_top_k = MAX_TOP_K
    args.sub_core_grids = None
    args.use_topk_logprobs = False
    return args


def _shard_cols(torch_input, mesh_device, cluster_shape):
    return ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )


def _build_uncertain_logits(args):
    """A flat / uncertain distribution that mimics the real model's lm_head output: a
    handful of mildly-positive real tokens, the rest of the real vocab slightly negative,
    and the out-of-vocab padding columns at exactly 0 (zero lm_head weights)."""
    x = torch.full((1, 1, BATCH_SIZE, args.padded_vocab_size), -3.0)
    hot = [5, 1234, 65540, 131080]  # real "hot" tokens spread across shards (all < VOCAB_SIZE)
    for i, t in enumerate(hot):
        x[:, :, :, t] = 2.0 - 0.1 * i
    x[:, :, :, VOCAB_SIZE:] = 0.0  # out-of-vocab padding == 0, exactly as the model produces it
    return x


@pytest.mark.parametrize("apply_fix", [False, True], ids=["no_fix_BUG", "with_fix"])
@parametrize_mesh_with_fabric([(1, 4), (1, 8), (4, 8)])
def test_oov_padding_leak(apply_fix, mesh_device, device_params, reset_seeds):
    args = _make_args(mesh_device)
    logger.info(
        f"mesh={args.cluster_shape} vocab={args.vocab_size} padded={args.padded_vocab_size} "
        f"per_device={args.padded_vocab_size // mesh_device.shape[1]} apply_fix={apply_fix}"
    )

    x_torch = _build_uncertain_logits(args)
    if apply_fix:
        # Apply the actual model-side fix: bias the out-of-vocab columns. If the fix
        # function is removed/broken this case fails (import error or OOV leak).
        mask = build_out_of_vocab_logit_mask(VOCAB_SIZE, args.padded_vocab_size)
        assert mask is not None, "expected padding for this vocab/TP config"
        x_torch = x_torch + mask

    tt_in = _shard_cols(x_torch, mesh_device, args.cluster_shape)

    sg = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None, enable_internal_trace=False)
    # Match the eval: temperature=1.0, top_p=1.0, top_k unset -> 32 (see format_sampling_params).
    params = format_sampling_params(SamplingParams(temperature=1.0, top_k=-1, top_p=1.0, seed=42), BATCH_SIZE)
    sg.reset_sampling_params(params)
    sg.seed_manager.reset_seed([42] * BATCH_SIZE, list(range(BATCH_SIZE)))

    oov = 0
    n = 200
    for _ in range(n):
        sg.seed_manager.get_new_values()
        tokens, _ = sg.sample(tt_in, enable_trace=False)
        tok = ttnn.to_torch(ttnn.get_device_tensors(tokens)[0])[0, 0, :, :].reshape(-1)[0].item()
        if tok >= VOCAB_SIZE:
            oov += 1
    logger.info(f"apply_fix={apply_fix}: out-of-vocab samples = {oov}/{n}")

    if apply_fix:
        assert oov == 0, f"FIX FAILED: {oov}/{n} out-of-vocab tokens sampled even with the lm_head vocab mask"
    else:
        assert oov > 0, "Expected the unmasked-padding bug to leak out-of-vocab tokens, but none were sampled"
