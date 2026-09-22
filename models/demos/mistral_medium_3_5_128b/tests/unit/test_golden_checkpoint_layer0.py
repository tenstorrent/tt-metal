# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1 test 2 — the torch reference against the **real checkpoint**, measured on real ground truth.

Host-only; no device. This is the one host test in the package that is not self-consistency: every
other reference test compares two things this bring-up wrote. Here the right answer comes from
outside — the prepared golden trace, produced by ``transformers 5.12.1`` walking
``Ministral3DecoderLayer`` layer by layer on CPU over the real fp8 checkpoint (see its
``metadata.json`` ``reference`` field). If the reference and the trace agree, then a later device
disagreement is a device problem, which is the only way P1 and P2 are debuggable at all.

**Why this is not the recipe's ``golden_hf_first_token``.** That pattern runs the whole HF model on
CPU and compares the first sampled token. At 128 B parameters and 10240 tokens that forward *is* the
golden trace — it took 867 s and the machine that ran it was the trace generator, not a test host.
Re-running it per test run is not a thing that happens, and comparing one token id would be a far
weaker statement than comparing 8 x 10240 x 128 cache elements per layer anyway. So the trace is
consumed rather than recomputed, and the comparison is per-layer K/V.

**Two measurements, because they cover different things.**

*Full sequence, layer 0* (:func:`test_layer0_kv_full_sequence`). Layer 0's K and V depend only on
the embedding, ``input_layernorm``, ``k_proj``/``v_proj`` and RoPE — no attention, so all 10240
positions fit in memory at once and run in about two seconds. That covers the whole position range,
which is where a wrong ``rope_theta``, a wrong YaRN ``factor`` or a missing ``attention_scaling``
would show: those are nearly invisible at position 0 and unmistakable at position 10239.

*Prefix, every layer up to* ``LAYERS`` (:func:`test_layer_stack_prefix`). Getting layer *i*'s K/V
for i > 0 needs layers 0..i-1 to have run, and attention at 10240 keys x 96 heads is 40 TB of scores
if materialized. **Causality is what makes this affordable**: a query at position p < n attends
exactly keys 0..p whether the run is n tokens long or 10240, so the first n positions of the trace
are the right answer for an n-token run, unchanged. Running ``PREFIX_TOKENS`` tokens costs ~8.5 s
per layer and compares against a straight slice of the trace. That covers what layer 0 cannot:
``o_proj``, the SwiGLU triple, ``post_attention_layernorm``, both residuals, the softmax, and the
layer-index-to-weight mapping.

``MISTRAL_CHECKPOINT_LAYERS`` overrides the layer count. The default is 4 to keep the suite quick;
a full-depth run (88) takes ~13 minutes and was measured once during bring-up. Its profile is the
reason the bound below is depth-dependent rather than one number:

===========  ============  ============
layer        k             v
===========  ============  ============
0            1.0000000     1.0000000
1            0.9999991     0.9999971
20           0.9999812     0.9998861
45           0.9980538     0.9937648
65           0.9969764     0.9896124
85           0.9950053     0.9867589
86           0.9949444     0.9873974   <- worst k
87           0.9990038     0.9967150
===========  ============  ============

Worst case over the whole stack is k 0.99494 and v 0.98676, both an order of magnitude above the
spec's ``pcc_lower_bound`` of 0.85 and still above its 0.99 ``pcc_target`` for K. Two features of
that curve are worth naming because they look like defects and are not. The decay is *smooth* — the
largest single-layer drop anywhere in the 88 is 1.7e-3 for K and 3.5e-3 for V, with no step — which
is what bf16 accumulation over a deepening residual looks like; the residual's own RMS grows from
0.0045 at layer 0 to 0.61 at layer 86 over the same span, so the absolute error grows while the
relative error stays tiny. And the **last layer recovers** (k 0.9949 -> 0.9990, v 0.9874 -> 0.9967).
That is layer 87's own weights, not a correction: the final layer's K/V projections are
better-conditioned than its neighbours', so the same input error maps to a smaller output error.
The consequence for the tests is that "the deepest layer is the worst" holds for a shallow prefix
and not for the full stack, so :func:`test_stack_gap_is_depth_accumulation` asserts smoothness,
which holds at every depth, rather than monotonicity, which does not.

Full depth on the device is P1's job, against this same trace.
"""

import os
from pathlib import Path

import pytest
import torch

from models.demos.mistral_medium_3_5_128b.reference.checkpoint import QUANTIZED_SUFFIXES, CheckpointLoader
from models.demos.mistral_medium_3_5_128b.reference.golden import GoldenTrace
from models.demos.mistral_medium_3_5_128b.reference.model_config import MistralMediumConfig
from models.demos.mistral_medium_3_5_128b.reference.modeling import (
    REF_DTYPE,
    MistralRMSNorm,
    MistralYarnRotaryEmbedding,
    apply_rotary_pos_emb,
    build_layer,
    causal_mask,
)

#: Tokens for the stacked run. A multiple of TILE_SIZE*sp, so the same prefix is usable on device.
PREFIX_TOKENS = 1024

#: Layers to stack. Override with ``MISTRAL_CHECKPOINT_LAYERS``; 88 is the full model.
LAYERS = int(os.getenv("MISTRAL_CHECKPOINT_LAYERS", "4"))

#: Layer 0 is bit-exact against the trace — measured PCC 1.0 for both K and V at all 10240 tokens,
#: and the output RMS agrees to every printed digit. The reference computes layer 0 the same way
#: ``Ministral3DecoderLayer`` does, in the same dtypes, so anything below 1.0 here is a real
#: disagreement about the checkpoint or the rope, not rounding. Held tight deliberately.
LAYER0_PCC = 0.9999999

#: Deeper layers are not exact: the reference's residual and MLP ordering differs from HF's in ways
#: bf16 notices, and the error compounds with depth. Measured at PREFIX_TOKENS=1024: layer 1
#: k 0.9999991 v 0.9999971, layer 2 k 0.9999986 v 0.9999934, layer 3 k 0.9999978 v 0.9999892 —
#: about 1e-6 per layer for K over the first few. A shallow run has no excuse for anything looser.
SHALLOW_STACK_PCC = 0.9999

#: The full stack accumulates to k 0.99494 / v 0.98676 at worst (see the module docstring's table),
#: so a run deep enough to reach the back half needs the looser bound. Set below the measured worst
#: case with room for host-library drift, and still well above the spec's 0.85 ``pcc_lower_bound``.
#: ``test_stack_gap_is_depth_accumulation`` is what keeps this slack from hiding a defect: a
#: mis-scaled projection or a dropped residual steps, and no step this large exists in the measured
#: curve.
DEEP_STACK_PCC = 0.98

#: Where the two regimes divide. Below this depth the accumulated error is still ~1e-5.
SHALLOW_LAYERS = 8

STACK_PCC = SHALLOW_STACK_PCC if LAYERS <= SHALLOW_LAYERS else DEEP_STACK_PCC

#: The largest per-layer PCC change tolerated at any depth, in either direction. The measured full
#: 88-layer curve peaks at 1.7e-3 (K, layer 85) and 3.5e-3 (V, layer 73) falling, and 4.1e-3 (K) /
#: 9.3e-3 (V) recovering at layer 87. This bound admits those and nothing resembling a defect,
#: which would move PCC by whole percentage points in one layer.
MAX_LAYER_STEP = 0.02


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Correlation in fp64.

    ``comp_pcc`` accumulates in fp32, which over the 10.5 M elements of one layer's K returns
    values slightly above 1 and cannot resolve the difference between 0.999999 and exact.
    """
    x, y = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


@pytest.fixture(scope="module")
def cfg():
    return MistralMediumConfig.from_json()


@pytest.fixture(scope="module")
def golden():
    trace = GoldenTrace.from_env()
    if trace.num_layers != 88:
        pytest.skip(f"golden trace has {trace.num_layers} layers, acceptance needs 88")
    return trace


@pytest.fixture(scope="module")
def loader(cfg):
    return CheckpointLoader.from_env(cfg)


def test_trace_and_checkpoint_are_the_same_weights(loader, golden):
    """The trace names the checkpoint it came from; comparing against any other is meaningless.

    A reference built from a different checkpoint would produce a number, not an error — around 0.0
    if the weights are unrelated, and something misleadingly high if they are two revisions of the
    same model. So the provenance is asserted rather than trusted.
    """
    assert loader.path.resolve() == Path(golden.metadata["weights_path"]).resolve(), (
        f"golden trace was generated from {golden.metadata['weights_path']} but the loader reads "
        f"{loader.path} — set PREFILL_WEIGHTS to match"
    )


def test_config_matches_the_trace(cfg, golden):
    """Shape and rope fields the trace recorded must match the vendored config."""
    assert golden.num_layers == cfg.num_hidden_layers
    assert golden.num_kv_heads == cfg.num_key_value_heads
    assert golden.head_dim == cfg.head_dim
    assert golden.attention_scaling == pytest.approx(cfg.attention_scaling, rel=0, abs=1e-15)
    assert golden.metadata["dtype"] == str(REF_DTYPE).replace("torch.", "")


def test_only_projections_are_quantized(loader, cfg):
    """The dequantization rule this loader is built on, checked against the index rather than assumed.

    A checkpoint that quantized the norms, or left one projection in bf16, would still load — and
    every number downstream would be wrong by a scale factor that no PCC check can see.
    """
    p = loader.layer_prefix(0)
    for suffix in QUANTIZED_SUFFIXES:
        assert loader.has(f"{p}{suffix}.weight_scale_inv"), f"{suffix} has no scale"
        assert loader.raw(f"{p}{suffix}.weight").dtype == torch.float8_e4m3fn
        scale = loader.raw(f"{p}{suffix}.weight_scale_inv")
        # weight_block_size is null, so one scalar per tensor. A block-quantized checkpoint would
        # put a 2-D grid here and the single multiply in `dequantized` would be wrong.
        assert scale.numel() == 1, f"{suffix} scale has {scale.numel()} elements, expected a scalar"
    for unquantized in (
        f"{p}input_layernorm.weight",
        f"{p}post_attention_layernorm.weight",
        f"{loader.prefix}embed_tokens.weight",
        f"{loader.prefix}norm.weight",
        "lm_head.weight",
    ):
        assert not loader.has(unquantized.replace(".weight", ".weight_scale_inv")), unquantized
        assert loader.raw(unquantized).dtype == REF_DTYPE, unquantized


@pytest.fixture(scope="module")
def layer0_full(cfg, loader, golden):
    """``(k, v)`` for layer 0 at all ``golden.n_tokens`` positions, in the trace's own convention."""
    ids = golden.token_ids()
    x = loader.embed(ids)
    p = loader.layer_prefix(0)
    norm = MistralRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, REF_DTYPE)
    with torch.no_grad():
        norm.weight.copy_(loader.dequantized(f"{p}input_layernorm.weight"))
        h = norm(x)
        b, s, _ = h.shape
        shape = (b, s, cfg.num_key_value_heads, cfg.head_dim)
        k = (h @ loader.dequantized(f"{p}self_attn.k_proj.weight").T).view(shape).transpose(1, 2)
        v = (h @ loader.dequantized(f"{p}self_attn.v_proj.weight").T).view(shape).transpose(1, 2)
        positions = torch.arange(s, dtype=torch.int64)[None]
        cos, sin = MistralYarnRotaryEmbedding(cfg, REF_DTYPE)(positions)
        k, _ = apply_rotary_pos_emb(k, k, cos, sin)
    return k, v


def test_layer0_kv_full_sequence(golden, layer0_full):
    """Layer 0's K and V against the trace, all 10240 positions.

    No attention is involved, so this is a clean statement about the checkpoint read, the norm, the
    two projections and RoPE — and it covers the full position range, unlike the prefix test.
    """
    k, v = layer0_full
    gk, gv = golden.layer_kv(0)
    assert k.shape == gk.shape and v.shape == gv.shape, f"{k.shape} vs {gk.shape}"
    pk, pv = pcc(k, gk), pcc(v, gv)
    print(f"layer 0 full-sequence: k {pk:.10f} v {pv:.10f} over {golden.n_tokens} tokens")
    assert pk >= LAYER0_PCC, f"layer 0 k: {pk}"
    assert pv >= LAYER0_PCC, f"layer 0 v: {pv}"


def test_layer0_rope_is_applied_at_the_right_positions(cfg, golden, layer0_full):
    """A position-offset rope bug that PCC-over-everything would dilute.

    Comparing the last 1024 positions on their own removes the 90% of the sequence where a small
    angle error barely moves anything. An off-by-one or a wrong ``rope_theta`` fails here first.
    """
    k, _ = layer0_full
    gk, _ = golden.layer_kv(0)
    tail = pcc(k[:, :, -1024:], gk[:, :, -1024:])
    print(f"layer 0 k, last 1024 positions: {tail:.10f}")
    assert tail >= LAYER0_PCC, f"layer 0 k tail: {tail}"

    # And the rope must not be a no-op: unrotated K would still correlate, just not this well.
    assert cfg.attention_scaling > 1.0, "yarn attention_scaling should be 1.4158883083359672"


@pytest.fixture(scope="module")
def stack_kv_pcc(cfg, loader, golden):
    """``[(pcc_k, pcc_v)]`` per layer for a ``PREFIX_TOKENS``-long run. One stack walk, reused."""
    ids = golden.token_ids(PREFIX_TOKENS)
    x = loader.embed(ids)
    positions = torch.arange(PREFIX_TOKENS, dtype=torch.int64)[None]
    cos, sin = MistralYarnRotaryEmbedding(cfg, REF_DTYPE)(positions)
    mask = causal_mask(PREFIX_TOKENS, PREFIX_TOKENS)

    out = []
    for i in range(LAYERS):
        weights = loader.layer_weights(i)
        layer = build_layer(cfg, weights)
        with torch.no_grad():
            x, k, v = layer(x, cos, sin, mask)
        gk, gv = golden.layer_kv(i, PREFIX_TOKENS)
        out.append((pcc(k, gk), pcc(v, gv)))
        del weights, layer  # ~2.8 GB per layer; the stack must stream, not accumulate
    return out


def test_layer_stack_prefix(golden, stack_kv_pcc):
    """Every layer's K/V against the trace, for a causal prefix of the real prompt."""
    for i, (pk, pv) in enumerate(stack_kv_pcc):
        bound = LAYER0_PCC if i == 0 else STACK_PCC
        print(f"layer {i}: k {pk:.10f} v {pv:.10f}")
        assert pk >= bound, f"layer {i} k: {pk}"
        assert pv >= bound, f"layer {i} v: {pv}"


def test_stack_gap_is_depth_accumulation(stack_kv_pcc):
    """The gap must behave like bf16 accumulation: exact at layer 0, then small smooth steps.

    This is what makes ``STACK_PCC``'s slack safe. A real defect — a mis-scaled ``down_proj``, a
    dropped residual, an off-by-one in the layer index — would not start from an exact layer 0 and
    then move by a thousandth per layer; it would step, at the layer it was introduced.

    **Smoothness, not monotonicity.** Layer 87 recovers (see the module docstring), so "the deepest
    layer is the worst" is false for the full stack even though it holds for a shallow prefix. The
    property that holds at every depth is that no single layer moves the PCC much, and that is the
    one a defect actually violates.
    """
    if len(stack_kv_pcc) < 3:
        pytest.skip("needs at least 3 layers to see a trend")
    for name, series in (("k", [p for p, _ in stack_kv_pcc]), ("v", [p for _, p in stack_kv_pcc])):
        steps = [abs(series[i] - series[i + 1]) for i in range(len(series) - 1)]
        worst = max(steps)
        assert worst < MAX_LAYER_STEP, (
            f"{name}: layer {steps.index(worst) + 1} moves PCC by {worst:.2e}, too large to be " f"bf16 accumulation"
        )
        assert series[0] == max(series), f"{name}: layer 0 is not the closest to the trace: {series}"
