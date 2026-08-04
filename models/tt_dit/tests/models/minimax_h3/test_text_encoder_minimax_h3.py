# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Gates for MiniMax-H3's Qwen3-VL text conditioner.

The reference is the HF `Qwen3VLForConditionalGeneration` at its **full 64-layer** depth, with
`hidden_states[50]` taken from the result. That tensor is dumped once on CPU (the model is 25B
parameters, so re-running it inside every test would be wasteful, and the dump is what makes the
device test cheap); `models/tt_dit/tests/models/minimax_h3/dump_text_golden_minimax_h3.py` writes it and this file consumes it.

Three things are gated, in increasing cost:

1. mRoPE degeneracy for text-only prompts -- host-only, milliseconds, and the highest-risk silent
   failure in the port. The checkpoint sets `mrope_interleaved: true` while `create_rope_tensors`
   implements the chunked section split; for T2VA those must coincide, and here that is measured.
2. That the tap is not the post-norm final state -- host-only. The distinction is why only 50
   layers are built and why `activation_layers` exists.
3. The device encoder's tap against the reference's -- PCC, on the mesh.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from ....encoders.qwen3vl.loader_minimax_h3 import (
    MINIMAX_H3_TEXT_ENCODER_LAYER,
    build_minimax_h3_text_encoder,
    minimax_h3_text_config,
)
from ....encoders.qwen3vl.model_qwen3vl import create_rope_tensors
from ....parallel.config import EncoderParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

GOLDEN_ENV = "MINIMAX_H3_TEXT_GOLDEN"
WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
DEFAULT_WEIGHTS = "/data/cglagovich/MiniMax-H3-diffusers"


def _default_golden() -> str:
    """Under `TT_DIT_CACHE_DIR` like every other cached artifact, not a hardcoded path."""
    root = os.environ.get("TT_DIT_CACHE_DIR") or os.path.expanduser("~/.cache/tt-dit")
    return os.path.join(root, "h3_text_golden.pt")


# `testing-and-accuracy.md` puts a full-model forward at 0.99, which is where this started; 0.99
# would wave through a 100x regression against what this actually achieves. Measured:
#
#   39 tokens   PCC 99.9998 %   RMSE/sigma 0.5 %
#   512 tokens  PCC 99.9892 %   RMSE/sigma 1.5 %     <- the production working point
#
# The bar is set from the 512-token row, not the 39-token one: a 50-layer causal stack accumulates
# over its context, so error grows with prompt length and a short prompt is not a proxy. 0.999 /
# 0.05 leaves ~10x and ~3x margin on the production measurement.
#
# relative_rmse is paired with PCC because the embedding is consumed as an *absolute* value by the
# DiT's `context_embedder`; PCC alone would score a scaled or shifted conditioner 0.9999.
MIN_PCC = 0.999
MAX_RELATIVE_RMSE = 0.05

MESH_4X8 = [
    pytest.param(
        (4, 8),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
        id="4x8",
    )
]


def _golden():
    path = os.environ.get(GOLDEN_ENV) or _default_golden()
    if not os.path.isfile(path):
        pytest.skip(
            f"no text-encoder golden dump at {path}; set {GOLDEN_ENV} or run models/tt_dit/tests/models/minimax_h3/dump_text_golden_minimax_h3.py"
        )
    return torch.load(path, weights_only=False)


def _weights_dir():
    base = os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)
    directory = os.path.join(base, "text_encoder")
    if not os.path.isdir(directory):
        pytest.skip(f"no MiniMax-H3 text_encoder under {base}; set {WEIGHTS_ENV}")
    return directory


# ---------------------------------------------------------------------------
# 1. mRoPE degeneracy -- host only
# ---------------------------------------------------------------------------


def test_mrope_is_permutation_invariant_for_text_only():
    """A text-only prompt makes the mRoPE section split irrelevant, so `mrope_interleaved` is too.

    All three position axes carry the same `arange`, so the cos/sin of axis t, h and w at a given
    frequency are the *same number*. Any assignment of frequencies to axes therefore produces the
    same table, which is why the chunked split `create_rope_tensors` implements can serve a
    checkpoint that declares `mrope_interleaved: true`.

    Asserted by scrambling `mrope_section`: if the result were sensitive to the split, this port
    would need the interleaved variant, and the symptom of getting it wrong would be a conditioner
    that is subtly wrong everywhere rather than an error.
    """
    config = minimax_h3_text_config(_weights_dir())
    head_dim = config["head_dim"]
    theta = config["rope_scaling"].get("rope_theta", config["rope_theta"])
    section = config["rope_scaling"]["mrope_section"]
    assert config["rope_scaling"].get("mrope_interleaved") is True, "checkpoint no longer declares interleaved mRoPE"

    reference = create_rope_tensors(1, 64, None, head_dim, theta, section)
    for scrambled in ([section[1], section[0], section[2]], [section[2], section[1], section[0]]):
        other = create_rope_tensors(1, 64, None, head_dim, theta, scrambled)
        assert torch.equal(reference[0], other[0]), f"cos changed under section permutation {scrambled}"
        assert torch.equal(reference[1], other[1]), f"sin changed under section permutation {scrambled}"


# One bfloat16 ulp for values in [0.5, 1), which is where cos/sin spend most of their range.
# HF emits its rotary tables in the hidden dtype, so the reference *is* bf16 and this is the
# resolution of the thing being compared against -- the precision floor, not a tolerance chosen to
# make a test pass. Two ulps are allowed so an entry sitting exactly on a rounding boundary can
# fall either way; measured worst case is one ulp on 6 of 65536 entries at 512 tokens.
_BF16_ULP = 2.0**-8
# A real error -- wrong theta, wrong head_dim, wrong section widths -- moves O(1) of the entries by
# O(1), which is ~250x this bar, so the loosened tolerance costs no detection power.
_MAX_DIFFERING_FRACTION = 0.001


def test_mrope_matches_reference_tables():
    """Our rope tables against the ones HF actually used, captured from the reference forward.

    Not bit-exact, and the reason is dtype rather than maths: the reference tables are bfloat16.
    Compared at bf16 resolution, plus an explicit bound on *how many* entries may differ at all, so
    a systematic shift that happened to stay inside one ulp still fails.
    """
    golden = _golden()
    config = golden["text_config"]
    head_dim = config["head_dim"]
    theta = config["rope_scaling"].get("rope_theta", config["rope_theta"])
    section = config["rope_scaling"]["mrope_section"]

    checked = 0
    for record in golden["records"]:
        if record.get("rope") is None:
            continue
        cos, sin = create_rope_tensors(1, record["num_tokens"], None, head_dim, theta, section)
        for name, ours, theirs in (
            ("cos", cos.squeeze(1).float(), record["rope"][0].float()),
            ("sin", sin.squeeze(1).float(), record["rope"][1].float()),
        ):
            assert ours.shape == theirs.shape, f"{name}: {tuple(ours.shape)} != {tuple(theirs.shape)}"
            # Assert the reference really is bf16-stored, so this cannot silently degrade into a
            # loose comparison against an fp32 reference.
            assert torch.equal(theirs.to(torch.bfloat16).float(), theirs), f"reference {name} is not bf16"

            rounded = ours.to(torch.bfloat16).float()
            differing = int((rounded != theirs).sum())
            worst = (rounded - theirs).abs().max().item()
            fraction = differing / theirs.numel()
            logger.info(
                f"n={record['num_tokens']:4d} {name}: {differing}/{theirs.numel()} entries differ "
                f"({fraction:.2%}), worst {worst:.3e} = {worst / _BF16_ULP:.2f} bf16 ulp"
            )
            assert worst <= 2 * _BF16_ULP, f"{name} differs by {worst:.3e}, more than 2 bf16 ulps"
            assert fraction <= _MAX_DIFFERING_FRACTION, f"{name}: {fraction:.2%} of entries differ"
        checked += 1
    assert checked, "golden dump captured no rope tables"


def test_tap_is_not_the_post_norm_state():
    """`hidden_states[50]` is nothing like the final normalized state, so the tap index matters.

    This is the mistake the diffusers reference raises to prevent: build a 50-layer stack, read its
    normalized output, and you get a tensor that is wrong by orders of magnitude rather than
    wrong in the last bits. Recorded here so the tap choice is defended by a measurement.
    """
    golden = _golden()
    assert golden["tap_index"] == MINIMAX_H3_TEXT_ENCODER_LAYER

    for record in golden["records"]:
        # hidden_states holds the embedding output plus one entry per layer, so a tap index of 50
        # addresses the output of layer 49 -- which is what a 50-layer stack's last layer gives.
        assert record["num_hidden_states"] == golden["num_text_layers"] + 1
        gap = (record["tap"] - record["final_norm_state"]).abs().max().item()
        assert gap > 1.0, f"tap and post-norm state differ by only {gap:.4f}; the tap index may be wrong"
        logger.info(f"n={record['num_tokens']:3d} tap-vs-post-norm maxdiff={gap:.1f}")


# ---------------------------------------------------------------------------
# 2. The device encoder against the reference tap
# ---------------------------------------------------------------------------


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("tp_axis", [0], ids=["tp0"])
def test_text_encoder_tap_matches_reference(mesh_device, tp_axis, reset_seeds):
    """The tap of the 50-layer device encoder against HF's `hidden_states[50]`, per prompt.

    One encoder build serves every prompt; the prompts differ in length, so this also exercises
    the per-length JIT compile the bringup path deliberately does not bucket away.
    """
    golden = _golden()
    weights_dir = _weights_dir()

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = EncoderParallelConfig(tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor))

    logger.info(f"building the 50-layer conditioner on {tuple(mesh_device.shape)}, TP={tp_factor} on axis {tp_axis}")
    encoder, config = build_minimax_h3_text_encoder(
        weights_dir,
        mesh_device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        is_fsdp=True,
    )
    head_dim = config["head_dim"]
    theta = config["rope_scaling"].get("rope_theta", config["rope_theta"])
    section = config["rope_scaling"]["mrope_section"]

    for record in golden["records"]:
        token_ids = torch.tensor([record["token_ids"]], dtype=torch.long)
        n = token_ids.shape[1]
        cos, sin = create_rope_tensors(1, n, None, head_dim, theta, section)

        tt_ids = ttnn.from_torch(token_ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        taps = encoder.forward(
            tt_ids,
            attention_mask=None,  # causal; a single un-padded prompt needs no mask
            pos_embeds=(bf16_tensor(cos, device=mesh_device), bf16_tensor(sin, device=mesh_device)),
        )
        assert len(taps) == 1, f"expected exactly the layer-{MINIMAX_H3_TEXT_ENCODER_LAYER - 1} tap, got {len(taps)}"

        # Replicated across the mesh: read one device's copy rather than composing every replica.
        actual = ttnn.to_torch(ttnn.get_device_tensors(taps[0])[0]).squeeze(0).float()
        expected = record["tap"].float()

        assert actual.shape == expected.shape, f"{tuple(actual.shape)} != {tuple(expected.shape)}"
        logger.info(f"prompt {record['prompt'][:48]!r} ({n} tokens)")
        assert_quality(expected, actual, pcc=MIN_PCC, relative_rmse=MAX_RELATIVE_RMSE)
