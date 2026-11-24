# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device unit tests for the GLM-5.3-Flash KDA decoder block (``tt/kda_block``), against the HF modules.

The reference is transformers' own ``Glm5NextTextDecoderLayer`` (and its ``Glm5NextTextHyperConnection``,
``Glm5NextTextMLP`` and ``Glm5NextTextMoE`` sub-modules) on a small config, run in fp32 on the host; the device
modules are built from that module's ``state_dict`` exactly as they would be from the real checkpoint. Decode is
compared token by token against the HF full-sequence forward (KDA is causal, so step ``t`` of a decode must equal
position ``t`` of the prefill). Block prefill needs ``prepare_chunk_recurrence``, which does not fit Wormhole
(``test_kda.py::PREP_BLOCKER``), so it is a strict xfail on Wormhole.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.glm53_flash.tests.test_kda import PREP_BLOCKER
from models.experimental.glm53_flash.tt.ds_v4.streams import _strip_prefix
from models.experimental.glm53_flash.tt.kda_block.decoder_layer import GLMKDADecoderLayer
from models.experimental.glm53_flash.tt.kda_block.ffn import GLMDenseMLP, GLMMoE
from models.experimental.glm53_flash.tt.kda_block.hyperconnection import GLMHyperConnection, mix_streams

transformers_glm = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig  # noqa: E402

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})

DRAM = ttnn.DRAM_MEMORY_CONFIG
HIDDEN = 256
HC = 4
DENSE_LAYER, MOE_LAYER = 0, 1
# Glm5NextPreTrainedModel._keep_in_fp32_modules_strict
STRICT_FP32 = ("e_score_correction_bias", "conv1d", "dt_bias", "A_log")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def small_config() -> Glm5NextTextConfig:
    """GLM-5.3-Flash with every width shrunk (same structure: hc=4, 20 Sinkhorn iterations, -5 decay bound)."""
    return Glm5NextTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=512,
        moe_intermediate_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        linear_num_heads=4,
        linear_head_dim=64,
        num_hidden_layers=2,
        layer_types=["linear_attention"] * 2,
        mlp_layer_types=["dense", "sparse"],
    )


def init_like_checkpoint(module: torch.nn.Module, seed: int) -> torch.nn.Module:
    """Random weights at magnitudes where every term matters: unit-gain projections, near-1 norm gains,
    hyper-connection parameters around the trained model's scale, and decays spread over (-5, 0)."""
    g = torch.Generator().manual_seed(seed)
    for name, p in module.named_parameters():
        shape = p.shape
        if name.endswith("_hc.fn"):
            v = torch.randn(shape, generator=g) * 0.05
        elif name.endswith("_hc.base"):
            v = torch.randn(shape, generator=g) * 0.5
        elif name.endswith("_hc.scale"):
            v = 0.5 + torch.rand(shape, generator=g)
        elif "norm" in name:
            v = 1.0 + 0.1 * torch.randn(shape, generator=g)
        elif name.endswith("A_log"):
            v = torch.randn(shape, generator=g) * 0.5
        elif name.endswith("dt_bias"):
            v = torch.randn(shape, generator=g) * 0.5
        elif name.endswith("conv1d.weight"):
            v = torch.randn(shape, generator=g) * 0.3
        else:
            v = torch.randn(shape, generator=g) / shape[-1] ** 0.5
        # bf16-representable like the checkpoint (fp32 only where from_pretrained keeps it): near-tie expert
        # scores flip the top-k on rounding alone, which would grade the weight dtype rather than the port.
        p.data.copy_(v if any(s in name for s in STRICT_FP32) else bf16_round(v))
    for name, b in module.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.data.copy_(torch.randn(b.shape, generator=g) * 0.1)
    return module.float().eval()


def hf_layer(cfg, layer_idx: int, seed: int = 0):
    return init_like_checkpoint(transformers_glm.Glm5NextTextDecoderLayer(cfg, layer_idx), seed)


def bf16_round(t: torch.Tensor) -> torch.Tensor:
    return t.bfloat16().float()


def to_dev(t: torch.Tensor, device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def to_host(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).float()


@pytest.fixture(scope="module")
def cfg():
    return small_config()


@pytest.fixture(scope="module")
def dense_ref(cfg):
    return hf_layer(cfg, DENSE_LAYER, seed=0)


@pytest.fixture(scope="module")
def moe_ref(cfg):
    return hf_layer(cfg, MOE_LAYER, seed=1)


# Built once per module on the module-scoped device that `device` hands every test here (use_module_device).
@pytest.fixture(scope="module")
def dense_block(_device_module_impl, cfg, dense_ref):
    return GLMKDADecoderLayer(
        cfg, DENSE_LAYER, dense_ref.state_dict(), _device_module_impl, expert_dtype=ttnn.bfloat8_b
    )


@pytest.fixture(scope="module")
def moe_block(_device_module_impl, cfg, moe_ref):
    return GLMKDADecoderLayer(cfg, MOE_LAYER, moe_ref.state_dict(), _device_module_impl, expert_dtype=ttnn.bfloat8_b)


# ---------------------------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("b, s", [(1, 1), (4, 1), (1, 32)], ids=["decode1", "decode4", "prefill32"])
def test_hyperconnection_matches_hf(device, cfg, dense_ref, b, s):
    """post / comb / collapsed of ``Glm5NextTextHyperConnection`` for decode rows and a prefill chunk."""
    torch.manual_seed(10)
    streams = bf16_round(torch.randn(b, s, HC, HIDDEN))
    hc = GLMHyperConnection(cfg, _strip_prefix(dense_ref.state_dict(), "attn_hc"), device)
    post, comb, collapsed = hc(to_dev(streams, device))
    with torch.no_grad():
        ref_post, ref_comb, ref_collapsed = dense_ref.attn_hc(streams)

    got_post = to_host(post).reshape(b, s, HC)
    got_comb = to_host(comb).reshape(b, s, HC, HC)
    got_collapsed = to_host(collapsed).reshape(b, s, HIDDEN)
    assert torch.allclose(got_post, ref_post, atol=2e-3), f"post max err {(got_post - ref_post).abs().max():.2e}"
    assert torch.allclose(got_comb, ref_comb, atol=2e-3), f"comb max err {(got_comb - ref_comb).abs().max():.2e}"
    assert pcc(got_collapsed, ref_collapsed) > 0.9995, f"collapsed PCC {pcc(got_collapsed, ref_collapsed):.6f}"
    # Sinkhorn: comb is (nearly) doubly stochastic.
    assert torch.allclose(got_comb.sum(-1), torch.ones(b, s, HC), atol=1e-2)
    assert torch.allclose(got_comb.sum(-2), torch.ones(b, s, HC), atol=1e-2)


def test_mix_streams_matches_reference(device):
    """``post * out + comb^T @ streams`` (the reference decoder layer's residual update)."""
    torch.manual_seed(11)
    b, s = 2, 1
    post = torch.rand(b, s, HC, 1) * 2
    comb = torch.rand(b, s, HC, HC)
    out = bf16_round(torch.randn(b, s, 1, HIDDEN))
    streams = bf16_round(torch.randn(b, s, HC, HIDDEN))
    got = mix_streams(
        to_dev(post, device, ttnn.float32),
        to_dev(comb, device, ttnn.float32),
        to_dev(out, device),
        to_dev(streams, device),
    )
    ref = post * out + torch.matmul(comb.transpose(-1, -2), streams)
    assert got.dtype == ttnn.bfloat16
    assert pcc(to_host(got), ref) > 0.9999, f"PCC {pcc(to_host(got), ref):.6f}"


@pytest.mark.parametrize("tokens", [1, 32])
def test_dense_mlp_matches_hf(device, cfg, dense_ref, tokens):
    torch.manual_seed(12)
    x = bf16_round(torch.randn(1, 1, tokens, HIDDEN))
    mlp = GLMDenseMLP(cfg, _strip_prefix(dense_ref.state_dict(), "mlp"), device)
    got = to_host(mlp(to_dev(x, device)))
    with torch.no_grad():
        ref = dense_ref.mlp(x)
    assert pcc(got, ref) > 0.99, f"PCC {pcc(got, ref):.5f}"


@pytest.mark.parametrize("tokens", [1, 32])
def test_moe_matches_hf(device, cfg, moe_ref, tokens):
    torch.manual_seed(13)
    x = bf16_round(torch.randn(1, 1, tokens, HIDDEN))
    moe = GLMMoE(cfg, _strip_prefix(moe_ref.state_dict(), "mlp"), device, expert_dtype=ttnn.bfloat8_b)
    got = to_host(moe(to_dev(x, device), decode=(tokens == 1)))
    with torch.no_grad():
        ref = moe_ref.mlp(x)
    assert pcc(got, ref) > 0.99, f"PCC {pcc(got, ref):.5f}"


# ---------------------------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------------------------


def _hf_full(ref, streams: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return ref(streams)[0]


def _check_step(got: torch.Tensor, ref: torch.Tensor, inp: torch.Tensor, what: str):
    """The block output, and separately what the block added to the residual (the stream mix alone would carry
    a high PCC on the full output even with a broken sublayer)."""
    assert pcc(got, ref) > 0.999, f"{what}: output PCC {pcc(got, ref):.6f}"
    assert pcc(got - inp, ref - inp) > 0.99, f"{what}: residual-update PCC {pcc(got - inp, ref - inp):.5f}"


@pytest.mark.parametrize("which", ["kda_dense", "kda_moe"])
def test_block_decode_matches_hf(request, which):
    """Single-user decode, several steps: step t must equal position t of the HF full-sequence forward."""
    block = request.getfixturevalue("dense_block" if which == "kda_dense" else "moe_block")
    ref = request.getfixturevalue("dense_ref" if which == "kda_dense" else "moe_ref")
    device = request.getfixturevalue("device")
    torch.manual_seed(20)
    steps = 6
    streams = bf16_round(torch.randn(1, steps, HC, HIDDEN))
    expected = _hf_full(ref, streams)

    state = block.allocate_state(1)
    for t in range(steps):
        out = block.decode(to_dev(streams[:, t : t + 1], device), state)
        got = to_host(out).reshape(1, 1, HC, HIDDEN)
        _check_step(got, expected[:, t : t + 1], streams[:, t : t + 1], f"{which} step {t}")


def test_block_decode_batched_users(device, moe_block, moe_ref):
    """B users decoded together: each one's outputs match its own HF sequence (per-user state isolation)."""
    torch.manual_seed(21)
    users, steps = 3, 4
    streams = bf16_round(torch.randn(users, steps, HC, HIDDEN))
    expected = _hf_full(moe_ref, streams)

    state = moe_block.allocate_state(users)
    for t in range(steps):
        got = to_host(moe_block.decode(to_dev(streams[:, t : t + 1], device), state)).reshape(users, 1, HC, HIDDEN)
        for u in range(users):
            _check_step(
                got[u : u + 1], expected[u : u + 1, t : t + 1], streams[u : u + 1, t : t + 1], f"user {u} step {t}"
            )


def test_block_decode_inactive_slot_keeps_state(device, dense_block, dense_ref):
    """A slot masked inactive for a step keeps its KDA state, so its sequence continues as if the step never ran."""
    torch.manual_seed(22)
    steps = 3
    streams = bf16_round(torch.randn(2, steps, HC, HIDDEN))
    junk = bf16_round(torch.randn(2, 1, HC, HIDDEN))
    expected = _hf_full(dense_ref, streams)
    state = dense_block.allocate_state(2)
    both = to_dev(torch.ones(2, 1, 1, 1), device, ttnn.float32)
    only_user0 = to_dev(torch.tensor([1.0, 0.0]).reshape(2, 1, 1, 1), device, ttnn.float32)

    dense_block.decode(to_dev(streams[:, 0:1], device), state, active=both)
    # User 1 is idle for this step: its row holds junk, and its state must not move.
    step1 = torch.cat([streams[0:1, 1:2], junk[1:2]], 0)
    got1 = to_host(dense_block.decode(to_dev(step1, device), state, active=only_user0)).reshape(2, 1, HC, HIDDEN)
    _check_step(got1[0:1], expected[0:1, 1:2], streams[0:1, 1:2], "user 0 step 1")
    # User 1 resumes with its own step 1 while user 0 sits out.
    step2 = torch.cat([junk[0:1], streams[1:2, 1:2]], 0)
    only_user1 = to_dev(torch.tensor([0.0, 1.0]).reshape(2, 1, 1, 1), device, ttnn.float32)
    got2 = to_host(dense_block.decode(to_dev(step2, device), state, active=only_user1)).reshape(2, 1, HC, HIDDEN)
    _check_step(got2[1:2], expected[1:2, 1:2], streams[1:2, 1:2], "user 1 step 1 after idling")


def test_block_decode_leaves_input_intact(device, dense_block):
    """The caller's streams survive the block (the next layer of a pipeline or a re-run reads them)."""
    torch.manual_seed(23)
    streams = bf16_round(torch.randn(1, 1, HC, HIDDEN))
    dev_streams = to_dev(streams, device)
    dense_block.decode(dev_streams, dense_block.allocate_state(1))
    assert torch.equal(to_host(dev_streams), streams)


def test_block_rejects_bad_inputs(device, cfg, dense_block, expect_error):
    with expect_error(ValueError, "not a KDA"):
        mla_cfg = small_config()
        mla_cfg.layer_types = ["full_attention", "linear_attention"]
        GLMKDADecoderLayer(mla_cfg, 0, {}, device)
    state = dense_block.allocate_state(1)
    with expect_error(ValueError, "decode streams"):
        dense_block.decode(to_dev(torch.zeros(1, 2, HC, HIDDEN), device), state)
    with expect_error(ValueError, "slots"):
        dense_block.decode(to_dev(torch.zeros(2, 1, HC, HIDDEN), device), state)
    with expect_error(ValueError, "multiple of 32"):
        dense_block.forward(to_dev(torch.zeros(1, 48, HC, HIDDEN), device), state)


def _prefill_blocked(device) -> bool:
    return device.arch() == ttnn.Arch.WORMHOLE_B0


@pytest.mark.parametrize("which", ["kda_dense", "kda_moe"])
def test_block_prefill_then_decode_matches_hf(request, which):
    """A 32-token prefill chunk, then decode steps continuing from its state, against one HF pass."""
    device = request.getfixturevalue("device")
    if _prefill_blocked(device):
        request.applymarker(pytest.mark.xfail(strict=True, raises=RuntimeError, reason=PREP_BLOCKER))
    block = request.getfixturevalue("dense_block" if which == "kda_dense" else "moe_block")
    ref = request.getfixturevalue("dense_ref" if which == "kda_dense" else "moe_ref")
    torch.manual_seed(24)
    t_pre, t_dec = 32, 3
    streams = bf16_round(torch.randn(1, t_pre + t_dec, HC, HIDDEN))
    expected = _hf_full(ref, streams)

    state = block.allocate_state(1)
    got_pre = to_host(block.forward(to_dev(streams[:, :t_pre], device), state)).reshape(1, t_pre, HC, HIDDEN)
    _check_step(got_pre, expected[:, :t_pre], streams[:, :t_pre], f"{which} prefill")
    for t in range(t_pre, t_pre + t_dec):
        got = to_host(block.decode(to_dev(streams[:, t : t + 1], device), state)).reshape(1, 1, HC, HIDDEN)
        _check_step(got, expected[:, t : t + 1], streams[:, t : t + 1], f"{which} decode {t}")
