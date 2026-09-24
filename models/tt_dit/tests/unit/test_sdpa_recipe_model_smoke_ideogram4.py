# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device smoke tests: Ideogram4 transformer block (D256) under the opt-in SDPA recipes.

Random weights, no checkpoint. A reduced-width block (4 heads x head_dim 256, hidden 1024) is built
from the official reference module (modeling_ideogram4.Ideogram4TransformerBlock, run in fp32 as the
ground truth) and the tt block is run once per variant -- legacy (sdpa_precision=None), FAST, ACCURATE,
LOW_PRECISION(bfp8 KV) -- with the SAME state dict and inputs.

Gates, per recipe variant:
- SDPA core: the SDPA op call inside the block is captured (its inputs after QK-norm/RoPE and, for
  LOW_PRECISION, prepare_sdpa_input, and its output) and compared against exact fp64 attention on
  those inputs (keys/queries limited to logical_n): relative L2 (100*||a-b||/||b||) <= 3% FAST /
  LOW_PRECISION, 1% ACCURATE. For LOW_PRECISION the same L2 against the pre-preparation bf16 Q/K/V
  (incl. the bfp8 KV quantization) is recorded as `*_sdpa_core_l2_incl_input_prep`.
- end to end vs torch: the block's residual update (out - x) vs the fp32 reference's update:
  L2 <= legacy L2 + margin (no regression). The block output itself is dominated by bf16 matmul /
  norm error shared by all variants, so the SDPA-core gate is the one that isolates the recipe.

Paths: dense SDPA (sp_factor 1, unmasked) on 1x1; ring joint SDPA (SP=2 on the size-2 axis, TP=1) on
1x2, incl. a logical_n pad tail. A segment mask with a recipe set must raise ValueError.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn

from ...models.transformers.transformer_ideogram4 import Ideogram4TransformerBlock, rope_halfsplit_to_interleaved
from ...parallel.config import DiTParallelConfig, ParallelFactor
from ...parallel.manager import CCLManager
from ...reference.ideogram4 import modeling_ideogram4
from ...utils import tensor
from ...utils.tensor import bf16_tensor

HEAD_DIM = 256
NUM_HEADS = 4
HIDDEN = NUM_HEADS * HEAD_DIM
INTERMEDIATE = 2048
ADALN_DIM = 512
NORM_EPS = 1e-5

VARIANTS = [
    ("legacy", None, None),
    ("fast", ttnn.SDPAPrecision.FAST, None),
    ("accurate", ttnn.SDPAPrecision.ACCURATE, None),
    ("low_bfp8", ttnn.SDPAPrecision.LOW_PRECISION, ttnn.bfloat8_b),
]
ABS_BOUND = {"fast": 3.0, "accurate": 1.0, "low_bfp8": 3.0}
MARGIN = {"fast": 1.0, "accurate": 0.25, "low_bfp8": 1.5}  # percentage points over legacy

LINE_1D = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return 100.0 * (a - b).norm().item() / b.norm().item()


class _SdpaCapture:
    """Wraps the ttnn SDPA entry points; records (q, k, v, out) of every call as host fp32 tensors."""

    def __init__(self, monkeypatch, mesh_device, *, sp_axis: int, tp_axis: int):
        self.calls = []
        dims = [None, None]
        dims[sp_axis], dims[tp_axis] = 2, 1  # sequence on SP, heads on the (size-1) other axis
        self._composer = lambda: ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
        for name in ("ring_joint_scaled_dot_product_attention", "scaled_dot_product_attention"):
            monkeypatch.setattr(ttnn.transformer, name, self._wrap(getattr(ttnn.transformer, name)))
        self._unprepared = []
        orig_prepare = ttnn.transformer.prepare_sdpa_input

        def prepare(t, *args, **kwargs):
            self._unprepared.append(self._host(t))
            return orig_prepare(t, *args, **kwargs)

        monkeypatch.setattr(ttnn.transformer, "prepare_sdpa_input", prepare)

    def _host(self, t):
        return ttnn.to_torch(t, mesh_composer=self._composer()).float()

    def _wrap(self, orig):
        def wrapped(q, k, v, *args, **kwargs):
            result = orig(q, k, v, *args, **kwargs)
            out = result[0] if isinstance(result, tuple) else result
            call = dict(q=self._host(q), k=self._host(k), v=self._host(v), out=self._host(out))
            call["logical_n"] = kwargs.get("logical_n")
            call["dtypes"] = (q.dtype, k.dtype, v.dtype)
            call["kwargs"] = {key: kwargs[key] for key in ("precision", "inputs_prepared") if key in kwargs}
            if self._unprepared:
                assert len(self._unprepared) == 3
                call["q0"], call["k0"], call["v0"] = self._unprepared
                self._unprepared = []
            self.calls.append(call)
            return result

        return wrapped

    def core_l2(self, *, unprepared: bool = False) -> float:
        diffs, refs = [], []
        for c in self.calls:
            n = c["logical_n"] or c["q"].shape[2]
            pre = "0" if unprepared and "q0" in c else ""
            q, k, v = (c[f"{name}{pre}"][:, :, :n].double() for name in "qkv")
            ref = F.scaled_dot_product_attention(q, k, v)
            diffs.append((c["out"][:, :, :n].double() - ref).flatten())
            refs.append(ref.flatten())
        return 100.0 * torch.cat(diffs).norm().item() / torch.cat(refs).norm().item()


def _reference_block():
    block = modeling_ideogram4.Ideogram4TransformerBlock(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adanln_dim=ADALN_DIM,
    ).eval()
    # Non-trivial norm affines / adaLN so the recipe sees realistic score ranges.
    with torch.no_grad():
        for name, p in block.named_parameters():
            if "norm" in name:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            if "adaln_modulation" in name:
                p.mul_(4.0)
    return block


def _inputs(seq_len):
    x = torch.randn(1, seq_len, HIDDEN)
    adaln_input = torch.randn(1, 1, ADALN_DIM)
    theta = torch.rand(1, seq_len, HEAD_DIM // 2) * 2 * math.pi
    cos = torch.cat([theta.cos(), theta.cos()], dim=-1)  # half-split rotate-half tables
    sin = torch.cat([theta.sin(), theta.sin()], dim=-1)
    return x, adaln_input, cos, sin


def _run_variants(mesh_device, seq_len, sp_axis, tp_axis, monkeypatch, record_property, tag, expect_path):
    torch.manual_seed(0)
    sp_factor = tuple(mesh_device.shape)[sp_axis]
    torch_block = _reference_block()
    x, adaln_input, cos, sin = _inputs(seq_len)
    with torch.no_grad():
        torch_out = torch_block(
            x, segment_ids=torch.zeros(1, seq_len, dtype=torch.long), cos=cos, sin=sin, adaln_input=adaln_input
        )
    torch_delta = torch_out - x
    state = torch_block.state_dict()

    alignment = ttnn.TILE_SIZE * sp_factor
    padded = math.ceil(seq_len / alignment) * alignment
    cos4, sin4 = rope_halfsplit_to_interleaved(cos.unsqueeze(1), sin.unsqueeze(1), HEAD_DIM)
    pad = padded - seq_len
    x_p = F.pad(x, (0, 0, 0, pad))
    cos4 = torch.cat([cos4, torch.ones(1, 1, pad, HEAD_DIM)], dim=2)
    sin4 = torch.cat([sin4, torch.zeros(1, 1, pad, HEAD_DIM)], dim=2)

    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )
    shard = dict(mesh_axis=sp_axis) if sp_factor > 1 else {}
    tt_x = bf16_tensor(x_p, device=mesh_device, **shard, **(dict(shard_dim=1) if shard else {}))
    tt_cos = bf16_tensor(cos4, device=mesh_device, **shard, **(dict(shard_dim=2) if shard else {}))
    tt_sin = bf16_tensor(sin4, device=mesh_device, **shard, **(dict(shard_dim=2) if shard else {}))
    tt_adaln = bf16_tensor(adaln_input, device=mesh_device)

    deltas, cores = {}, {}
    for vid, precision, kv_dtype in VARIANTS:
        capture = _SdpaCapture(monkeypatch, mesh_device, sp_axis=sp_axis, tp_axis=tp_axis)
        tt_block = Ideogram4TransformerBlock(
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            num_heads=NUM_HEADS,
            norm_eps=NORM_EPS,
            adaln_dim=ADALN_DIM,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            sdpa_precision=precision,
            sdpa_kv_dtype=kv_dtype,
        )
        tt_block.load_torch_state_dict({k: v.clone() for k, v in state.items()})
        tt_out = tt_block(
            tt_x, cos=tt_cos, sin=tt_sin, adaln_input=tt_adaln, attn_mask=None, spatial_sequence_length=seq_len
        )
        out = tensor.to_torch(tt_out, mesh_axes=[None, sp_axis if sp_factor > 1 else None, None])[:, :seq_len]
        monkeypatch.undo()

        assert len(capture.calls) == 1, f"{tag} {vid}: expected one SDPA call, got {len(capture.calls)}"
        call = capture.calls[0]
        assert (call["logical_n"] is not None) == (expect_path == "ring")
        if precision is None:
            assert call["kwargs"] == {}
        else:
            prepared = precision == ttnn.SDPAPrecision.LOW_PRECISION
            assert call["kwargs"] == {"precision": precision, "inputs_prepared": prepared}
            if prepared:
                assert call["dtypes"][1:] == (kv_dtype, kv_dtype)
        deltas[vid] = out.float() - x
        cores[vid] = (capture.core_l2(), capture.core_l2(unprepared=True))

    legacy_l2 = _l2(deltas["legacy"], torch_delta)
    record_property(f"{tag}_legacy_l2_vs_torch", round(legacy_l2, 4))
    record_property(f"{tag}_legacy_sdpa_core_l2", round(cores["legacy"][0], 4))
    logger.info(f"{tag} legacy: block-update L2 vs torch {legacy_l2:.4f}%, SDPA core {cores['legacy'][0]:.4f}%")
    failures = []
    for vid, _p, _kv in VARIANTS[1:]:
        l2_ref, l2_leg = _l2(deltas[vid], torch_delta), _l2(deltas[vid], deltas["legacy"])
        core, core_prep = cores[vid]
        record_property(f"{tag}_{vid}_l2_vs_torch", round(l2_ref, 4))
        record_property(f"{tag}_{vid}_l2_vs_legacy", round(l2_leg, 4))
        record_property(f"{tag}_{vid}_sdpa_core_l2", round(core, 4))
        record_property(f"{tag}_{vid}_sdpa_core_l2_incl_input_prep", round(core_prep, 4))
        checks = {
            f"vs torch <= legacy+{MARGIN[vid]}": l2_ref <= legacy_l2 + MARGIN[vid],
            f"SDPA core <= {ABS_BOUND[vid]}": core <= ABS_BOUND[vid],
        }
        bad = [k for k, ok in checks.items() if not ok]
        msg = (
            f"{tag} {vid}: block-update L2 vs torch {l2_ref:.4f}%, vs legacy {l2_leg:.4f}%, "
            f"SDPA core {core:.4f}% (incl. input prep {core_prep:.4f}%)"
        )
        logger.info(f"{msg} -> {'OK' if not bad else 'FAIL ' + str(bad)}")
        if bad:
            failures.append(f"{msg} failed {bad}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("device_params", [{}], indirect=True)  # no fabric on a single device
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [512, 1088], ids=["n512", "n1088"])
def test_ideogram4_block_dense_recipes(mesh_device, seq_len, record_property, monkeypatch):
    _run_variants(mesh_device, seq_len, 0, 1, monkeypatch, record_property, f"dense_n{seq_len}", "dense")


@pytest.mark.parametrize("device_params", [LINE_1D], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize(
    "seq_len", [1024, 1000], ids=["n1024_local512", "n1000_pad1024_local512"]
)  # the second exercises the logical_n pad tail
def test_ideogram4_block_ring_sp2_recipes(mesh_device, seq_len, record_property, monkeypatch):
    _run_variants(mesh_device, seq_len, 1, 0, monkeypatch, record_property, f"ring_n{seq_len}", "ring")


@pytest.mark.parametrize("device_params", [{}], indirect=True)  # no fabric on a single device
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_ideogram4_block_recipe_rejects_segment_mask(mesh_device):
    torch.manual_seed(0)
    seq_len = 256
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    tt_block = Ideogram4TransformerBlock(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_heads=NUM_HEADS,
        norm_eps=NORM_EPS,
        adaln_dim=ADALN_DIM,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        sdpa_precision=ttnn.SDPAPrecision.ACCURATE,
    )
    tt_block.load_torch_state_dict(_reference_block().state_dict())
    x, adaln_input, cos, sin = _inputs(seq_len)
    cos4, sin4 = rope_halfsplit_to_interleaved(cos.unsqueeze(1), sin.unsqueeze(1), HEAD_DIM)
    mask = torch.zeros(1, 1, seq_len, seq_len)
    mask[..., : seq_len // 2, seq_len // 2 :] = float("-inf")
    with pytest.raises(ValueError, match="unmasked"):
        tt_block(
            bf16_tensor(x, device=mesh_device),
            cos=bf16_tensor(cos4, device=mesh_device),
            sin=bf16_tensor(sin4, device=mesh_device),
            adaln_input=bf16_tensor(adaln_input, device=mesh_device),
            attn_mask=bf16_tensor(mask, device=mesh_device),
        )
