# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks on the MiniMax-H3 DiT weight-dtype policy.

The failure this file is really guarding against is a cache collision: the device-weight cache
holds the *quantized* tensorbins, so if a bf8 build and a bf16 build can land in the same directory
then one of them silently reads the other's weights back into a Parameter of the wrong dtype. Every
field that changes what gets written therefore has to change the cache tag.
"""

from dataclasses import fields, replace

import pytest

import ttnn
from models.tt_dit.models.transformers.minimax_h3.quant_config import (
    PRESETS,
    MiniMaxH3QuantProfile,
    resolve_quant_profile,
)
from models.tt_dit.models.transformers.minimax_h3.transformer_block_minimax_h3 import MiniMaxH3TransformerBlock
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager

from .common import SMALL_LINE_PARALLEL

# The released MiniMax-H3 DiT geometry, from the checkpoint config.
HIDDEN = 5376
INNER = 7168
FFN = 14336
LAYERS = 50

GIB = 1024**3


def test_presets_resolve_by_name_and_reject_typos(expect_error):
    for name in PRESETS:
        profile = resolve_quant_profile(name)
        assert profile.name == name
    assert resolve_quant_profile(None) is None
    profile = MiniMaxH3QuantProfile.bf8_weights()
    assert resolve_quant_profile(profile) is profile
    # A typo must raise rather than reach a same-named instance method through getattr.
    with expect_error(ValueError, "unknown MiniMax-H3 quant profile"):
        resolve_quant_profile("attention_kwargs")
    with expect_error(ValueError, "unknown MiniMax-H3 quant profile"):
        resolve_quant_profile("bf8")


# Fields that do NOT change the stored tensorbins: the profile's own label, and the two matmul
# precision knobs, which change how the matmul unit reads a stored tile rather than what is stored.
# Everything else is storage BY DEFAULT and must therefore change the cache tag.
#
# Derived from the dataclass rather than listed, deliberately. A hand-written list of storage fields
# is a second copy of `cache_tag`'s own field set, and the two drift silently in the dangerous
# direction: add a field that changes written bytes, forget it in `cache_tag`, and a hardcoded list
# that also omits it makes this test PASS on a real cache collision. Deriving means a new field is
# storage until someone adds it to NON_STORAGE_FIELDS on purpose.
NON_STORAGE_FIELDS = ("name", "mm_math_fidelity", "mm_fp32_dest_acc_en")
STORAGE_FIELDS = tuple(f.name for f in fields(MiniMaxH3QuantProfile) if f.name not in NON_STORAGE_FIELDS)


def test_every_non_storage_field_is_a_real_field_of_the_profile():
    """So a rename cannot quietly turn a storage field into an unchecked one."""
    names = {f.name for f in fields(MiniMaxH3QuantProfile)}
    assert set(NON_STORAGE_FIELDS) <= names, set(NON_STORAGE_FIELDS) - names
    assert STORAGE_FIELDS, "every field was excluded -- the derivation is broken"


def _storage(profile) -> tuple:
    return tuple(getattr(profile, field) for field in STORAGE_FIELDS)


def test_a_cache_tag_is_never_shared_by_two_DIFFERENT_storage_policies():
    """The invariant is per-BYTES, not per-preset.

    This used to assert that every shipped preset had a tag of its own, which held only while the
    policy was pure storage. `bf8_weights_bf8_out_nofp32acc` writes byte-identical tensorbins to
    `bf8_weights_bf8_out` and differs only in arithmetic, so it SHOULD share the directory -- reusing
    the cache is most of why it costs 110 s to build rather than 240 s. What must never happen is the
    original failure: two policies that write different bytes landing in one directory, where one
    silently reads the other's weights back into a Parameter of the wrong dtype.
    """
    by_tag: dict[str, list[tuple[str, tuple]]] = {}
    for name in PRESETS:
        profile = PRESETS[name]()
        by_tag.setdefault(profile.cache_tag, []).append((name, _storage(profile)))
    for tag, entries in by_tag.items():
        distinct = {storage for _, storage in entries}
        assert len(distinct) == 1, f"cache tag {tag!r} is shared by differing storage policies: {entries}"


def test_the_arithmetic_only_preset_shares_its_storage_twin_cache():
    """Stated as its own check, because it is an intent and not a coincidence."""
    base = MiniMaxH3QuantProfile.bf8_weights_bf8_out()
    fast = MiniMaxH3QuantProfile.bf8_weights_bf8_out_nofp32acc()
    assert _storage(fast) == _storage(base)
    assert fast.cache_tag == base.cache_tag
    assert fast.name != base.name


def test_compute_kernel_kwargs_are_empty_unless_the_policy_sets_them():
    """Every profile that says nothing about arithmetic must leave the block's defaults alone."""
    for name in PRESETS:
        profile = PRESETS[name]()
        expected = {} if name != "bf8_weights_bf8_out_nofp32acc" else {"fp32_dest_acc_en": False}
        assert profile.compute_kernel_kwargs() == expected, name


def test_compute_kernel_kwargs_pass_through_a_fidelity_when_one_is_set():
    profile = MiniMaxH3QuantProfile.bf8_weights_bf8_out()
    assert profile.compute_kernel_kwargs() == {}
    lofi = replace(profile, mm_math_fidelity=ttnn.MathFidelity.LoFi, mm_fp32_dest_acc_en=True)
    assert lofi.compute_kernel_kwargs() == {
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "fp32_dest_acc_en": True,
    }
    # ... and it is still the same weights on disk.
    assert lofi.cache_tag == profile.cache_tag


@pytest.mark.parametrize(
    "a, b",
    [
        # Every field that changes a written tensorbin must change the tag.
        (MiniMaxH3QuantProfile.bf8_weights(), MiniMaxH3QuantProfile.bf8_weights_bf8_out()),
        (MiniMaxH3QuantProfile.bf8_weights(), MiniMaxH3QuantProfile.bf16()),
        (MiniMaxH3QuantProfile.bf8_weights(), MiniMaxH3QuantProfile.bf8_weights(bf16_blocks=(-1,))),
        (MiniMaxH3QuantProfile.bf8_weights(bf16_blocks=(0,)), MiniMaxH3QuantProfile.bf8_weights(bf16_blocks=(-1,))),
    ],
)
def test_differing_policies_never_share_a_cache_tag(a, b):
    assert a.cache_tag != b.cache_tag


def test_cache_tag_ignores_the_name_but_not_the_dtypes():
    """Two profiles that write the same bytes may share a directory; same name must not be enough."""
    same_bytes = MiniMaxH3QuantProfile.bf8_weights()
    renamed = MiniMaxH3QuantProfile(
        name="something_else",
        qkv_dtype=ttnn.bfloat8_b,
        out_dtype=ttnn.bfloat16,
        ff_dtype=ttnn.bfloat8_b,
    )
    assert same_bytes.cache_tag == renamed.cache_tag

    shadowed = MiniMaxH3QuantProfile(
        name="bf8_weights",
        qkv_dtype=ttnn.bfloat4_b,
        out_dtype=ttnn.bfloat16,
        ff_dtype=ttnn.bfloat8_b,
    )
    assert shadowed.cache_tag != same_bytes.cache_tag


def test_bf8_weights_kwargs_quantize_the_three_projections_and_carve_out_to_out():
    profile = MiniMaxH3QuantProfile.bf8_weights()
    attn = profile.attention_kwargs()
    assert attn["qkv_dtype"] == ttnn.bfloat8_b
    assert attn["out_dtype"] == ttnn.bfloat16
    ffn = profile.ffn_kwargs()
    assert ffn["ff1_dtype"] == ttnn.bfloat8_b
    assert ffn["ff2_dtype"] == ttnn.bfloat8_b
    # At TP=1 there is no fused all-gather for a narrowed activation to shrink, so the cast is off
    # and nothing downstream sees a block-float activation.
    assert attn["activation_dtype"] is None
    assert ffn["activation_dtype"] is None
    assert attn["pin_output_bf16"] is False
    assert ffn["pin_output_bf16"] is False


def test_bf16_profile_is_the_no_op_policy():
    """Every kwarg the bf16 profile hands out must leave the module building what it always built.

    `attention_kwargs` now also carries `mm_compute_kernel_overrides`, which the attention splats
    over its own compute-kernel defaults. For a no-op policy that has to be EMPTY -- an override of
    `{"fp32_dest_acc_en": True}` would happen to be the current default and would silently stop
    being a no-op the day the default changes.
    """
    profile = MiniMaxH3QuantProfile.bf16()
    attention = dict(profile.attention_kwargs())
    assert attention.pop("mm_compute_kernel_overrides") == {}
    assert set(attention.values()) <= {ttnn.bfloat16, None, False}
    assert set(profile.ffn_kwargs().values()) <= {ttnn.bfloat16, None, False}


def test_only_the_arithmetic_preset_hands_the_attention_an_override():
    """The attention owns its own matmul config, so the policy has to reach it explicitly.

    The first version of the precision policy was wired into the block only, so to_qkv and to_out -- 26.7 ms of a 221.7 ms block -- kept
    fp32 destination accumulation while the shipped description said all four matmuls had changed.
    """
    for name in PRESETS:
        overrides = PRESETS[name]().attention_kwargs()["mm_compute_kernel_overrides"]
        expected = {"fp32_dest_acc_en": False} if name == "bf8_weights_bf8_out_nofp32acc" else {}
        assert overrides == expected, name


def test_a_pinned_block_gets_the_default_arithmetic_back_too():
    """`for_block` on a pinned block must undo the precision policy, not only the dtypes.

    A pinned block exists to buy accuracy back with 385 MB. LoFi on a bfloat8_b tile is bounded by
    the tile's own 8-bit mantissa; on a bf16 weight it is a real loss, so leaving the reduced
    fidelity in place on a block that was just widened to bf16 would spend the memory and keep most
    of the error it was spent on.
    """
    from dataclasses import replace as _replace

    profile = _replace(
        MiniMaxH3QuantProfile.bf8_weights_bf8_out(),
        bf16_blocks=(0, -1),
        mm_math_fidelity=ttnn.MathFidelity.LoFi,
        mm_fp32_dest_acc_en=False,
    )
    pinned = profile.for_block(0, LAYERS)
    assert pinned.qkv_dtype is ttnn.bfloat16
    assert pinned.mm_math_fidelity is None and pinned.mm_fp32_dest_acc_en is None
    assert pinned.compute_kernel_kwargs() == {}
    # An unpinned block keeps the whole policy, arithmetic included.
    plain = profile.for_block(25, LAYERS)
    assert plain.qkv_dtype is ttnn.bfloat8_b
    assert plain.compute_kernel_kwargs() == {
        "math_fidelity": ttnn.MathFidelity.LoFi,
        "fp32_dest_acc_en": False,
    }


@pytest.mark.parametrize("pinned, expect_bf16", [((-1,), 49), ((0,), 0), ((0, -1), None)])
def test_pinned_blocks_resolve_to_bf16_and_nothing_else_does(pinned, expect_bf16):
    profile = MiniMaxH3QuantProfile.bf8_weights(bf16_blocks=pinned)
    pinned_indices = {i % LAYERS for i in pinned}
    for index in range(LAYERS):
        block = profile.for_block(index, LAYERS)
        if index in pinned_indices:
            assert block.qkv_dtype == ttnn.bfloat16, index
            assert block.ff_dtype == ttnn.bfloat16, index
        else:
            assert block.qkv_dtype == ttnn.bfloat8_b, index
            assert block.ff_dtype == ttnn.bfloat8_b, index
    if expect_bf16 is not None:
        assert profile.keeps_block_bf16(expect_bf16, LAYERS)


def test_block_stack_fits_a_p150_only_once_quantized():
    """The arithmetic the p150 profile exists for, against the measured 31.831 GiB of DRAM."""
    geom = dict(hidden_size=HIDDEN, inner_dim=INNER, ffn_dim=FFN, num_layers=LAYERS)
    bf16 = MiniMaxH3QuantProfile.bf16().stack_bytes(**geom) / GIB
    bf8 = MiniMaxH3QuantProfile.bf8_weights().stack_bytes(**geom) / GIB
    bf8_out = MiniMaxH3QuantProfile.bf8_weights_bf8_out().stack_bytes(**geom) / GIB

    # bf16 does not fit on a 32 GB part even with the whole AdaLN branch already off the device.
    assert bf16 > 31.831, bf16
    # Both quantized policies do, and giving up the to_out carve-out is worth ~1.8 GB.
    assert bf8 < 24.0, bf8
    assert bf8_out < bf8 - 1.5, (bf8, bf8_out)
    # Pinning a block back to bf16 costs one block's worth of the difference, not more.
    pinned = MiniMaxH3QuantProfile.bf8_weights(bf16_blocks=(-1,)).stack_bytes(**geom) / GIB
    assert 0 < pinned - bf8 < (bf16 - bf8) / LAYERS + 1e-6


@SMALL_LINE_PARALLEL
def test_the_profile_reaches_the_parameters(mesh_device, sp_axis, tp_axis, num_links, device_params, topology, is_fsdp):
    """The profile must change the Parameter dtypes, not just the cache tag.

    Worth a test of its own because an inert profile fails silently in the worst direction: the
    cache tag still says bf8, so the quantized directory fills with bf16 tensorbins, and the only
    symptom is that the stack no longer fits -- 22 GB of "it worked yesterday" turning into 38.5 GB.
    The block here is a toy; what is being checked is the wiring from profile to Parameter.
    """
    del device_params
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    tp_factor = tuple(mesh_device.shape)[tp_axis]
    kwargs = dict(
        hidden_size=128 * tp_factor,
        num_heads=4 * tp_factor,
        head_dim=32,
        ffn_dim=256 * tp_factor,
        time_embed_dim=32,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        precomputed_adaln=True,
    )

    plain = MiniMaxH3TransformerBlock(**kwargs)
    assert plain.attn.to_qkv.weight.dtype == ttnn.bfloat16
    assert plain.ff.ff1.weight.dtype == ttnn.bfloat16

    quantized = MiniMaxH3TransformerBlock(**kwargs, quant_config=MiniMaxH3QuantProfile.bf8_weights())
    assert quantized.attn.to_qkv.weight.dtype == ttnn.bfloat8_b
    assert quantized.ff.ff1.weight.dtype == ttnn.bfloat8_b
    assert quantized.ff.ff2.weight.dtype == ttnn.bfloat8_b
    # ... and the carve-out is a carve-out, not an oversight.
    assert quantized.attn.to_out.weight.dtype == ttnn.bfloat16

    bf8_out = MiniMaxH3TransformerBlock(**kwargs, quant_config=MiniMaxH3QuantProfile.bf8_weights_bf8_out())
    assert bf8_out.attn.to_out.weight.dtype == ttnn.bfloat8_b


def test_bf4_ff_narrows_only_the_feed_forward():
    """``bf4_ff`` drops ff1/ff2 to bf4 and leaves attention at bf8.

    The split is the point. Stage 05 measured this profile against ``bf8_weights_bf8_out`` on one
    chip at 1344x768 x124, same adapter and seed: 177.8 s against 178.7 s -- no speed at all -- for
    21.92 GiB down to 16.54. So it is a memory lever, and the only reason to reach for it is a
    canvas that does not otherwise fit. If it ever silently narrowed attention too it would be
    spending the accuracy that stage 02's pad-row window bought, for nothing.
    """
    profile = MiniMaxH3QuantProfile.bf4_ff()
    assert profile.ff_dtype == ttnn.bfloat4_b
    assert profile.qkv_dtype == ttnn.bfloat8_b
    assert profile.out_dtype == ttnn.bfloat8_b
    assert profile.activation_dtype is None
    assert resolve_quant_profile("bf4_ff") == profile
    # A distinct cache directory, or a bf4 build reads bf8 tensorbins into a bf4 Parameter.
    assert profile.cache_tag == "qbf8-obf8-fbf4"
    assert profile.cache_tag != MiniMaxH3QuantProfile.bf8_weights_bf8_out().cache_tag


def test_bf4_ff_keep_ends_pins_the_first_and_last_block():
    """The sweep's accuracy escape hatch keeps block 0 and the last block bf16, and says so in the
    cache tag -- a pinned build and an unpinned one are different weights."""
    profile = resolve_quant_profile("bf4_ff_keep_ends")
    assert profile.bf16_blocks == (0, -1)
    assert profile.keeps_block_bf16(0, 50)
    assert profile.keeps_block_bf16(49, 50)
    assert not profile.keeps_block_bf16(25, 50)
    assert profile.for_block(0, 50).ff_dtype == ttnn.bfloat16
    assert profile.for_block(25, 50).ff_dtype == ttnn.bfloat4_b
    assert profile.cache_tag != MiniMaxH3QuantProfile.bf4_ff().cache_tag
    assert "bf4_ff_keep_ends" in PRESETS
