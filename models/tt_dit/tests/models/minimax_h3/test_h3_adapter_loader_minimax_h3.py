# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the H3 adapter loader (alpha scaling, fused QKV, dense edits) and Turbo host-merge rounding."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ....experimental.lora.h3_adapter_loader import (
    _collect_pairs,
    _dense_target,
    _file_alpha,
    _register_fused,
    _scale_of,
    iter_h3_dense_edits,
    iter_h3_host_deltas,
)
from ....pipelines.minimax_h3.pipeline_minimax_h3_turbo import _round_bf16_stochastic
from .common_av import weights_dir

NUM_HEADS = 4
HEAD_DIM = 8
INNER_DIM = NUM_HEADS * HEAD_DIM  # 32
RANK = 2
IN_DIM = 5


class _CaptureQKV:
    """Records register_lora's (A, B, scale)."""

    def __init__(self):
        self.calls: list[tuple[torch.Tensor, torch.Tensor, float]] = []

    def register_lora(self, a, b, *, scale, name):
        self.calls.append((a, b, scale))
        return len(self.calls) - 1


def _attn():
    return SimpleNamespace(
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        inner_dim=INNER_DIM,
        n_local_heads=NUM_HEADS,
        parallel_config=SimpleNamespace(tensor_parallel=SimpleNamespace(factor=1)),
        to_qkv=_CaptureQKV(),
    )


def _qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        slot: {
            "A": torch.randn(RANK, IN_DIM, generator=g, dtype=torch.float32),
            "B": torch.randn(INNER_DIM, RANK, generator=g, dtype=torch.float32),
        }
        for slot in ("q", "k", "v")
    }


def _col_block(b_fused, position):
    return b_fused[:, position * RANK : (position + 1) * RANK]


def test_fused_folds_each_sources_own_alpha():
    qkvs = _qkv()
    base = lambda slot: f"transformer_blocks.0.attn.to_{slot}"  # noqa: E731

    attn1 = _attn()
    _register_fused(attn1, qkvs, 1.0, "a", {base(s): RANK for s in ("q", "k", "v")}, None, "transformer_blocks", 0)
    b1, scale1 = attn1.to_qkv.calls[0][1], attn1.to_qkv.calls[0][2]

    attn2 = _attn()
    factors = {"q": 1, "k": 2, "v": 3}
    _register_fused(
        attn2, qkvs, 1.0, "a", {base(s): RANK * factors[s] for s in ("q", "k", "v")}, None, "transformer_blocks", 0
    )
    b2 = attn2.to_qkv.calls[0][1]

    # Run 1 has alpha/rank == 1, so run 2's blocks are run 1's times each slot's factor.
    for pos, slot in enumerate(("q", "k", "v")):
        torch.testing.assert_close(_col_block(b2, pos), _col_block(b1, pos) * factors[slot])

    # Alpha is folded into B; only the caller strength reaches register_lora.
    assert scale1 == 1.0


def test_fused_file_alpha_fallback_scales_every_slot():
    qkvs = _qkv(seed=1)
    attn_unit = _attn()
    _register_fused(attn_unit, qkvs, 1.0, "a", {}, float(RANK), "transformer_blocks", 0)
    attn_2x = _attn()
    _register_fused(attn_2x, qkvs, 1.0, "a", {}, float(2 * RANK), "transformer_blocks", 0)
    torch.testing.assert_close(attn_2x.to_qkv.calls[0][1], attn_unit.to_qkv.calls[0][1] * 2.0)


def test_fused_missing_slot_raises(expect_error):
    qkvs = _qkv()
    del qkvs["v"]
    with expect_error(RuntimeError, "missing"):
        _register_fused(_attn(), qkvs, 1.0, "a", {}, None, "transformer_blocks", 0)


def test_fused_rank_disagreement_raises(expect_error):
    qkvs = _qkv()
    qkvs["k"]["A"] = torch.randn(RANK + 1, IN_DIM, dtype=torch.float32)
    with expect_error(ValueError, "ranks disagree"):
        _register_fused(_attn(), qkvs, 1.0, "a", {}, None, "transformer_blocks", 0)


def test_scale_of_prefers_per_target_then_file_then_one():
    ab = {"A": torch.zeros(8, IN_DIM)}  # rank 8
    assert _scale_of("t.0.attn.to_q", ab, {"t.0.attn.to_q": 16.0}, None) == pytest.approx(2.0)
    assert _scale_of("t.0.attn.to_q", ab, {}, 16.0) == pytest.approx(2.0)
    assert _scale_of("t.0.attn.to_q", ab, {}, None) == 1.0


def test_file_alpha_reads_known_metadata_keys():
    assert _file_alpha({"alpha": "16"}) == 16.0
    assert _file_alpha({"lora_alpha": "8"}) == 8.0
    assert _file_alpha({}) is None


def test_collect_pairs_maps_slots_and_alphas():
    raw = {
        "transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(RANK, IN_DIM),
        "transformer_blocks.0.attn.to_q.lora_B.weight": torch.zeros(INNER_DIM, RANK),
        "transformer_blocks.0.attn.to_q.alpha": torch.tensor(16.0),
    }
    pairs, alphas = _collect_pairs(raw)
    assert set(pairs["transformer_blocks.0.attn.to_q"]) == {"A", "B"}
    assert alphas["transformer_blocks.0.attn.to_q"] == 16.0


def test_collect_pairs_rejects_non_lora_key(expect_error):
    with expect_error(RuntimeError, "no LoRA convention"):
        _collect_pairs({"transformer_blocks.0.attn.to_q.weight": torch.zeros(1)})


def test_collect_pairs_rejects_half_pair(expect_error):
    with expect_error(RuntimeError, "half pair"):
        _collect_pairs({"transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(RANK, IN_DIM)})


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_stochastic_round_keeps_sub_ulp_delta_on_average(sign):
    weight = torch.full((1 << 20,), sign, dtype=torch.bfloat16)
    merged = weight.float() + sign * 1e-4
    assert torch.equal(merged.to(torch.bfloat16), weight)
    rounded = _round_bf16_stochastic(merged, torch.Generator().manual_seed(0)).float()
    assert abs(rounded.mean().item() - sign * (1.0 + 1e-4)) < 2e-5


def test_stochastic_round_is_exact_on_bf16_and_seeded():
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(4096, generator=generator).to(torch.bfloat16)
    assert torch.equal(_round_bf16_stochastic(weight.float(), torch.Generator().manual_seed(0)), weight)
    merged = weight.float() + 1e-3 * torch.randn(4096, generator=generator)
    first = _round_bf16_stochastic(merged, torch.Generator().manual_seed(0))
    assert torch.equal(first, _round_bf16_stochastic(merged, torch.Generator().manual_seed(0)))


def test_stochastic_round_keeps_signed_zeros_infinities_and_nan():
    special = torch.tensor([0.0, -0.0, float("inf"), float("-inf")]).repeat(1 << 12)
    rounded = _round_bf16_stochastic(special, torch.Generator().manual_seed(0)).float()
    assert torch.equal(rounded, special)
    assert torch.equal(torch.signbit(rounded), torch.signbit(special))
    nan = torch.full((1 << 12,), float("nan"))
    assert _round_bf16_stochastic(nan, torch.Generator().manual_seed(0)).isnan().all()


def test_dense_edits_are_read_apart_from_lora(tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {
        "proj_in.lora_A.weight": torch.randn(RANK, IN_DIM, generator=g),
        "proj_in.lora_B.weight": torch.randn(INNER_DIM, RANK, generator=g),
        "proj_in.diff": torch.randn(INNER_DIM, IN_DIM, generator=g),
        "proj_in.diff_b": torch.randn(INNER_DIM, generator=g),
        "transformer_blocks.0.attn.to_gate_compress.set_weight": torch.randn(INNER_DIM, IN_DIM, generator=g),
    }
    path = str(tmp_path / "adapter.safetensors")
    save_file(tensors, path)

    edits = {key: (kind, tensor) for key, kind, tensor in iter_h3_dense_edits(path, ("",))}
    assert set(edits) == {"proj_in.weight", "proj_in.bias", "transformer_blocks.0.attn.to_gate_compress.weight"}
    assert edits["proj_in.weight"][0] == "add" and torch.equal(edits["proj_in.weight"][1], tensors["proj_in.diff"])
    assert edits["proj_in.bias"][0] == "add" and torch.equal(edits["proj_in.bias"][1], tensors["proj_in.diff_b"])
    gate_kind, gate = edits["transformer_blocks.0.attn.to_gate_compress.weight"]
    assert gate_kind == "set"
    assert torch.equal(gate, tensors["transformer_blocks.0.attn.to_gate_compress.set_weight"])

    deltas = dict(iter_h3_host_deltas(path, ("",)))
    assert set(deltas) == {"proj_in.weight"}
    torch.testing.assert_close(
        deltas["proj_in.weight"], tensors["proj_in.lora_B.weight"] @ tensors["proj_in.lora_A.weight"]
    )


def test_host_deltas_reject_unknown_suffix(tmp_path, expect_error):
    path = str(tmp_path / "adapter.safetensors")
    save_file({"proj_in.lora_A.weight": torch.zeros(RANK, IN_DIM), "proj_in.mystery": torch.zeros(1)}, path)
    with expect_error(RuntimeError, "no LoRA convention"):
        list(iter_h3_host_deltas(path, ("",)))


def test_fasth3_adapter_header_matches_checkpoint():
    """Header-only: every dense add lands on a checkpoint tensor of its shape; sets are the 50 VSA gates."""
    lora_path = os.environ.get("MINIMAX_H3_FAST_LORA_PATH")
    if not lora_path:
        pytest.skip("set MINIMAX_H3_FAST_LORA_PATH to the FastH3 vsa-datafree adapter")
    directory = weights_dir("transformer") / "transformer"
    weight_map = json.loads((directory / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]

    with safe_open(lora_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        shapes = {key: handle.get_slice(key).get_shape() for key in keys}
    lora_bases = {key.rsplit(".lora_", 1)[0] for key in keys if ".lora_" in key}
    assert len(lora_bases) == 362

    adds, sets = {}, {}
    for key in keys:
        target = _dense_target(key)
        if target is not None:
            (adds if target[1] == "add" else sets)[target[0]] = shapes[key]
    assert len(adds) == 82
    assert sets == {f"transformer_blocks.{i}.attn.to_gate_compress.weight": [7168, 5376] for i in range(50)}

    for key, shape in adds.items():
        with safe_open(str(directory / weight_map[key]), framework="pt", device="cpu") as handle:
            assert handle.get_slice(key).get_shape() == shape, key
    assert not set(sets) & set(weight_map)
