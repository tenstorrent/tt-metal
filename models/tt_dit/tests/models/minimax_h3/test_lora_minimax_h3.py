# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Gates for loading a LoRA adapter onto MiniMax-H3.

Two tiers, and the cheap one is the one that catches the dangerous bug. Output parity against a
reference built from the same target map cannot detect a target map that misses part of an adapter
-- both sides drop the same tensors and agree. Only a coverage check over the adapter's own key set
can, so :func:`test_lora_adapter_coverage` runs without a mesh and asserts that every tensor in the
file was classified and routed somewhere.

The synthetic adapter below reproduces FastH3's exact key set and shapes at a reduced rank. Keeping
it in sync with the published file is itself an assertion: ``MINIMAX_H3_LORA_PATH`` points the same
coverage test at the real thing, and the two must classify identically.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

from ....lora.apply import FusionGroup, apply_adapter
from ....lora.keys import parse_adapter
from ....lora.promote import promote_to_lora
from ....models.transformers.minimax_h3.lora_targets_minimax_h3 import minimax_h3_fusion_groups, minimax_h3_host_paths
from ....models.transformers.minimax_h3.transformer_block_minimax_h3 import (
    MODALITY_NUM,
    NUM_MODULATION_PARAMS,
    MiniMaxH3TransformerBlock,
)
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.tensor import to_torch
from .common import GALAXY_RING, REAL_BLOCK_CONFIG, ROPE_FREQ_DIM, TT_BLOCK_CONFIG

HIDDEN = REAL_BLOCK_CONFIG["hidden_size"]
INNER = REAL_BLOCK_CONFIG["num_attention_heads"] * REAL_BLOCK_CONFIG["attention_head_dim"]
FFN = REAL_BLOCK_CONFIG["ffn_dim"]
TIME_EMBED = REAL_BLOCK_CONFIG["time_embed_dim"]
ADALN_OUT = NUM_MODULATION_PARAMS * HIDDEN * MODALITY_NUM
# M-RoPE splits its budget over three position axes and rotate-half doubles it, so 96 of the 128
# head channels rotate. Only the permutation of those channels matters here, but it has to be the
# production one for this to gate what production runs.
ROTARY_DIM = 3 * 2 * ROPE_FREQ_DIM

# The published adapter's counts, asserted against its own __metadata__ when the file is present.
FASTH3_DENSE_COUNTS = {"lora": 724, "diff": 27, "diff_b": 58, "set_weight": 0}
FASTH3_VSA_COUNTS = {"lora": 724, "diff": 24, "diff_b": 58, "set_weight": 50}

# One low-rank pair per adapted projection in a block, as (adapter leaf, out, in).
BLOCK_LORA_SHAPES = (
    ("attn.to_q", INNER, HIDDEN),
    ("attn.to_k", INNER, HIDDEN),
    ("attn.to_v", INNER, HIDDEN),
    ("attn.to_out.0", HIDDEN, INNER),
    ("ff.net.0.proj", 2 * FFN, HIDDEN),
    ("ff.net.2", HIDDEN, FFN),
    ("adaln_proj.linear", ADALN_OUT, TIME_EMBED),
)


def _pair(out_features: int, in_features: int, rank: int, generator: torch.Generator):
    """A LoRA pair small enough that its delta does not swamp a randomly initialised base weight."""
    a = torch.randn(rank, in_features, generator=generator, dtype=torch.float32) * 0.02
    b = torch.randn(out_features, rank, generator=generator, dtype=torch.float32) * 0.02
    return a, b


def synthetic_block_adapter(*, rank: int, prefix: str, seed: int = 0) -> dict[str, torch.Tensor]:
    """FastH3's key set for one transformer block, at a reduced rank.

    Includes the dense payload -- ``norm1``/``norm2`` weight deltas and the ``adaln_proj`` bias
    delta -- because a loader that handles only the low-rank half passes every parity test it is
    given while shipping a different model.
    """
    generator = torch.Generator().manual_seed(seed)
    state: dict[str, torch.Tensor] = {}
    for leaf, out_features, in_features in BLOCK_LORA_SHAPES:
        a, b = _pair(out_features, in_features, rank, generator)
        state[f"{prefix}.{leaf}.lora_A.weight"] = a
        state[f"{prefix}.{leaf}.lora_B.weight"] = b
    state[f"{prefix}.adaln_proj.linear.diff_b"] = torch.randn(ADALN_OUT, generator=generator) * 0.02
    state[f"{prefix}.norm1.diff"] = torch.randn(HIDDEN, generator=generator) * 0.02
    state[f"{prefix}.norm2.diff"] = torch.randn(HIDDEN, generator=generator) * 0.02
    return state


def _host_fuse(base: dict[str, torch.Tensor], adapter: dict[str, torch.Tensor], *, prefix: str, strength: float):
    """The reference: apply the adapter in checkpoint space, in fp32, before anything is uploaded.

    Deliberately naive. It shares no code with the loader under test, which is the point -- the
    loader's whole claim is that routing factors and routing their product give the same answer.
    """
    fused = {key: value.clone() for key, value in base.items()}
    for leaf, _, _ in BLOCK_LORA_SHAPES:
        a = adapter[f"{prefix}.{leaf}.lora_A.weight"]
        b = adapter[f"{prefix}.{leaf}.lora_B.weight"]
        key = f"{leaf}.weight"
        fused[key] = fused[key].to(torch.float32) + strength * (b.to(torch.float32) @ a.to(torch.float32))
    for adapter_key, base_key in (
        (f"{prefix}.adaln_proj.linear.diff_b", "adaln_proj.linear.bias"),
        (f"{prefix}.norm1.diff", "norm1.weight"),
        (f"{prefix}.norm2.diff", "norm2.weight"),
    ):
        fused[base_key] = fused[base_key].to(torch.float32) + strength * adapter[adapter_key].to(torch.float32)
    return fused


def _base_block_state(seed: int = 7) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * 0.02

    head_dim = REAL_BLOCK_CONFIG["attention_head_dim"]
    return {
        "attn.to_q.weight": randn(INNER, HIDDEN),
        "attn.to_k.weight": randn(INNER, HIDDEN),
        "attn.to_v.weight": randn(INNER, HIDDEN),
        "attn.to_out.0.weight": randn(HIDDEN, INNER),
        "attn.norm_q.weight": torch.ones(head_dim) + randn(head_dim),
        "attn.norm_k.weight": torch.ones(head_dim) + randn(head_dim),
        "ff.net.0.proj.weight": randn(2 * FFN, HIDDEN),
        "ff.net.2.weight": randn(HIDDEN, FFN),
        "norm1.weight": torch.ones(HIDDEN) + randn(HIDDEN),
        "norm2.weight": torch.ones(HIDDEN) + randn(HIDDEN),
        "adaln_proj.linear.weight": randn(ADALN_OUT, TIME_EMBED),
        "adaln_proj.linear.bias": randn(ADALN_OUT),
    }


# ---------------------------------------------------------------------------- coverage (no mesh)


def test_lora_adapter_coverage():
    """Every tensor in an adapter is classified, and the classification matches its metadata.

    This is the gate that survives a wrong target map. It needs no device.
    """
    adapter = synthetic_block_adapter(rank=8, prefix="transformer_blocks.0")
    entries, stats = parse_adapter(adapter)

    assert stats.tensors == len(adapter)
    assert sum(stats.counts.values()) == stats.tensors, f"unclassified tensors: {stats}"
    assert stats.counts["lora"] == 2 * len(BLOCK_LORA_SHAPES)
    assert stats.counts["diff"] == 2  # norm1, norm2
    assert stats.counts["diff_b"] == 1  # adaln_proj bias
    # No .alpha in the file means the adapter's own scale is 1, not alpha/rank.
    assert {entry.scale for entry in entries if entry.kind == "lora"} == {1.0}
    assert {entry.rank for entry in entries if entry.kind == "lora"} == {8}


def test_lora_published_adapter_coverage():
    """The same check against the real file, which is the one that can actually surprise us."""
    path = os.environ.get("MINIMAX_H3_LORA_PATH")
    if path is None:
        pytest.skip("set MINIMAX_H3_LORA_PATH to gate the published adapter")

    real_entries, real_stats = parse_adapter(path)
    logger.info(f"{path}: {real_stats}")
    assert sum(real_stats.counts.values()) == real_stats.tensors, f"unclassified tensors: {real_stats}"
    metadata = real_stats.metadata
    assert real_stats.counts["lora"] == int(metadata["low_rank_tensors"])
    assert real_stats.counts["diff"] + real_stats.counts["diff_b"] == int(metadata["diff_tensors"])
    assert real_stats.counts.get("set_weight", 0) == int(metadata["set_weight_tensors"])
    expected = FASTH3_VSA_COUNTS if int(metadata["set_weight_tensors"]) else FASTH3_DENSE_COUNTS
    assert {k: real_stats.counts.get(k, 0) for k in expected} == expected
    assert {entry.rank for entry in real_entries if entry.kind == "lora"} == {int(metadata["rank"])}


def test_lora_rejects_vsa_before_touching_the_model(expect_error):
    """A ``set_weight`` payload is refused with a reason, and refused before anything is routed.

    The model argument is a sentinel: reaching it at all would mean the rejection happens too late,
    after the loader has started mutating weights.
    """

    class Unreachable:
        def __getattr__(self, name):
            raise AssertionError(f"routing started despite an unsupported payload (touched {name})")

    adapter = synthetic_block_adapter(rank=8, prefix="transformer_blocks.0")
    adapter["transformer_blocks.0.attn.to_gate_compress.set_weight"] = torch.zeros(INNER, HIDDEN)

    with expect_error(NotImplementedError, "architecture"):
        apply_adapter(Unreachable(), adapter, name="vsa")


def test_lora_partial_fused_group_is_fatal(expect_error):
    """q/k/v share one ``to_qkv``; adapting some of them would silently zero the rest."""
    adapter = synthetic_block_adapter(rank=8, prefix="transformer_blocks.0")
    del adapter["transformer_blocks.0.attn.to_k.lora_A.weight"]
    del adapter["transformer_blocks.0.attn.to_k.lora_B.weight"]
    groups = [FusionGroup(owner="transformer_blocks.0.attn", members=("to_q", "to_k", "to_v"))]

    with expect_error(ValueError, "whole or not at all"):
        apply_adapter(object(), adapter, groups=groups, name="partial")


# ---------------------------------------------------------------------------- weight parity (mesh)


@GALAXY_RING
@pytest.mark.parametrize("strength", [1.0, 0.5], ids=["strength1", "strength0p5"])
def test_lora_block_weight_parity(
    mesh_device: ttnn.MeshDevice,
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    is_fsdp: bool,
    strength: float,
):
    """Applying an adapter on device must land the same weights as fusing it on host beforehand.

    Compares every parameter of a real block, so it covers the three out-dimension transforms an
    H3 adapter has to survive -- q/k/v head interleave plus rotary channel permutation, fused-SwiGLU
    tile pairing, and the AdaLN projection's TP reorder -- as well as the dense norm and bias
    deltas, under the parameters' real mesh sharding.
    """
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    prefix = "transformer_blocks.0"
    base_state = _base_block_state()
    adapter = synthetic_block_adapter(rank=8, prefix=prefix)
    reference_state = _host_fuse(base_state, adapter, prefix=prefix, strength=strength)

    def build(state):
        block = MiniMaxH3TransformerBlock(
            **TT_BLOCK_CONFIG,
            rotary_dim=ROTARY_DIM,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
        )
        block.load_torch_state_dict(state)
        return block

    reference = build(reference_state)
    adapted = build(base_state)

    promoted = promote_to_lora(adapted)
    assert promoted, "no Linear was promoted; the adapter would have nowhere to bind"

    # A block carries no `precomputed_adaln` attribute, so nothing defers to host and its own
    # adaln_proj is a device target -- which makes this the one place the AdaLN TP reorder is
    # exercised against real sharding.
    assert minimax_h3_host_paths(adapted) == ()
    report = apply_adapter(
        adapted,
        {key.removeprefix(f"{prefix}."): value for key, value in adapter.items()},
        groups=minimax_h3_fusion_groups(adapted),
        strength=strength,
        name="synthetic",
    )
    logger.info(report.summary())
    assert len(report.bound) == 5, f"expected to_qkv, to_out, ff1, ff2, adaln_proj; got {sorted(report.bound)}"
    assert len(report.deltas) == 3, f"expected norm1, norm2, adaln bias; got {[d.path for d in report.deltas]}"
    assert not report.host

    # One bf16 ULP is the floor here and PCC cannot express it: the reference rounds
    # `base + delta` once in fp32 while the device rounds both operands and adds in bf16, and on a
    # norm weight -- 1.0 +/- 0.02 -- a single ULP is 39% of sigma, so PCC reads ~99.4% for a result
    # that is correct to the last representable bit. Bound the error where it actually lives.
    worst: list[tuple[float, str]] = []
    for (path, expected), (_, got) in zip(_named_parameters(reference), _named_parameters(adapted), strict=True):
        a = to_torch(expected.data, mesh_axes=expected.mesh_axes, composer_device=mesh_device)
        b = to_torch(got.data, mesh_axes=got.mesh_axes, composer_device=mesh_device)
        assert a.shape == b.shape, f"{path}: {a.shape} != {b.shape}"
        worst.append((_max_ulp_error(a, b), path))

    for ulps, path in sorted(worst, reverse=True):
        logger.info(f"  {path:34s} {ulps:5.2f} bf16 ULP")
    assert worst, "no parameters compared"

    # Two roundings on the device side (base and delta) against one on the reference's.
    over = [(ulps, path) for ulps, path in worst if ulps > 2.0]
    assert not over, f"beyond bf16 rounding: {[(p, round(u, 2)) for u, p in sorted(over, reverse=True)]}"
    logger.info(f"{len(worst)} parameters within {max(u for u, _ in worst):.2f} bf16 ULP at strength {strength}")


def _max_ulp_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest elementwise gap between two bf16-valued tensors, in ULPs of the value itself.

    Scaling by each element's own exponent rather than by the tensor's range is what makes this
    meaningful on a norm weight, where every value sits near 1.0 and a range-relative bound would
    be satisfied by almost any error.
    """
    a, b = a.detach().to(torch.float64), b.detach().to(torch.float64)
    reference = torch.maximum(a.abs(), b.abs())
    # bf16 carries 8 significand bits; subnormals near zero would divide by zero, so floor the
    # exponent at the smallest normal.
    exponent = torch.floor(torch.log2(reference.clamp(min=2.0**-126)))
    return ((a - b).abs() / torch.pow(2.0, exponent - 7)).max().item()


def _named_parameters(module, prefix: str = ""):
    for name, parameter in module.named_parameters():
        yield f"{prefix}{name}", parameter
    for name, child in module.named_children():
        yield from _named_parameters(child, f"{prefix}{name}.")
