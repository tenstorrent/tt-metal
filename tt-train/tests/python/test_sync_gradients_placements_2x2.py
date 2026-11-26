# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""test_sync_gradients_placements.py on a ``[2, 2]`` mesh: DP x TP, FSDP x TP and HSDP, where a parameter is
replicated on one data axis and sharded on the other. The Llama tests also cover DP x SP and FSDP x SP. Each class
opens its ``[2, 2]`` through the conftest's ``fresh_device_mesh`` with the bundled descriptor, like
``test_ccl_topology.py``, and skips on a host with fewer than four chips."""

from __future__ import annotations

import re

import numpy as np
import pytest

import ttnn
import ttml
from sync_gradients_placements_common import (
    DIM,
    EVERY,
    FSDP_REPLICATED_NORMS,
    NATIVE,
    NORMS,
    TP_SHARDED_SUFFIXES,
    _assert_mean_over_axis,
    _assert_per_axis_labels,
    _backward,
    _build_fsdp_model,
    _device,
    _input,
    _placements,
    _reduce_set,
    _shards,
    assert_sync_reduces_exactly,
    is_sharded_on_axis,
    llama_backward,
    reductions_by_rule,
    tiny_llama,
)
from gradient_sync_helpers import exact_mean, fpu_add_bound, make_gradients, sharded_linear, to_bf16, upload
from tp_block import TPBlock, gather_like
from ttml.common.utils import build_causal_mask
from ttml.models import RunnerType, memory_efficient_runner
from ttml.models.deepseek import DeepSeek, DeepSeekConfig
from ttml.models.deepseek.transformer import DeepSeekMLP
from ttml.models.llama.transformer import LlamaBlock
from ttml.modules import AbstractModuleBase, LoraConfig, LoraModel
from ttml.parallel import SEQUENCE_DIM, TPStrategy

pytestmark = [
    pytest.mark.requires_device,
    pytest.mark.timeout(1800),
    pytest.mark.usefixtures("strict_ccl_topology"),
]


@pytest.fixture(scope="module")
def fsdp_tp_mesh(fresh_device_mesh):
    """``[2, 2]`` with axes ``("fsdp", "tp")``."""
    with fresh_device_mesh((2, 2), ("fsdp", "tp"), what="FSDP x TP placements tests", require_mgd=True) as mesh:
        yield mesh


# (strategy, FSDP replicate= patterns or None, the batch's data axes, (axis, op, pattern) rules).
# Every gradient is averaged over dp. Over fsdp, only the replicated parameters are averaged; FSDP reduce-scatters the
# rest. Under SP, the norms are also summed over tp.
LLAMA_2X2_CASES = {
    "dp-tp": (TPStrategy.TENSOR, None, ("dp",), [("dp", "mean", EVERY)]),
    "dp-sp": (TPStrategy.TENSOR_SEQUENCE, None, ("dp",), [("dp", "mean", EVERY), ("tp", "sum", NORMS)]),
    "fsdp-tp-replicated-norms": (TPStrategy.TENSOR, FSDP_REPLICATED_NORMS, ("fsdp",), [("fsdp", "mean", NORMS)]),
    "fsdp-sp": (TPStrategy.TENSOR_SEQUENCE, [], ("fsdp",), [("tp", "sum", NORMS)]),
    "hsdp-replicated-norms": (
        TPStrategy.NONE,
        FSDP_REPLICATED_NORMS,
        ("dp", "fsdp"),
        [("dp", "mean", EVERY), ("fsdp", "mean", NORMS)],
    ),
}
DP_TP_CASES = ["dp-tp", "dp-sp"]
FSDP_TP_CASES = ["fsdp-tp-replicated-norms", "fsdp-sp"]
HSDP_CASES = ["hsdp-replicated-norms"]


def _assert_llama_sync_reduces_exactly(case: str) -> None:
    strategy, replicate, data_axes, rules = LLAMA_2X2_CASES[case]
    model = tiny_llama(strategy, fsdp_replicate=replicate)
    params = model.parameters()
    llama_backward(model, data_axes)
    assert_sync_reduces_exactly(params, reductions_by_rule(params, rules))


# --- DP x TP on [2, 2] ("dp", "tp") -----------------------------------------------------------------------------


class TestDPxTP:
    def test_dp_reduction_keeps_tp_shards(self, dp_tp_mesh):
        """Batch sharded along "dp", TP weights sharded along "tp": every parameter is reduced along "dp" only, the
        result averages over the dp axis, reads Replicate there and keeps Shard on the tp axis for TP parameters."""
        dp, tp = dp_tp_mesh.axis_index("dp"), dp_tp_mesh.axis_index("tp")
        model = TPBlock(DIM)
        params = model.parameters()
        _backward(model, _input((dp_tp_mesh.axis_size("dp"), 1, DIM, DIM), dp_tp_mesh.axis_mapper("dp", tdim=0)))

        _assert_per_axis_labels(params, 2)
        before = {name: _shards(t.get_grad()) for name, t in params.items()}
        for name, t in params.items():
            value = t.get_value(NATIVE)
            assert is_sharded_on_axis(value, tp) == name.endswith(TP_SHARDED_SUFFIXES), name
            assert _reduce_set(value, (dp,)) == {dp}, name

        ttml.core.distributed.synchronize_gradients(params, [dp])

        for name, t in params.items():
            grad = t.get_grad()
            assert not is_sharded_on_axis(grad, dp), name
            if name.endswith(TP_SHARDED_SUFFIXES):
                assert is_sharded_on_axis(grad, tp), f"{name}: the TP shard did not survive the dp all_reduce"
            _assert_mean_over_axis(before[name], _shards(grad), dp_tp_mesh.shape, dp, name)

    @pytest.mark.parametrize("case", DP_TP_CASES)
    def test_sync_reduces_exactly(self, dp_tp_mesh, case):
        _assert_llama_sync_reduces_exactly(case)


# --- MoE experts on dp, [2, 2] ("dp", "tp") ----------------------------------------------------------------------

MOE_EXPERTS, MOE_SEQ_LEN, MOE_VOCAB = 4, 64, 128
# Relative Frobenius error. bf16 rounding measured at most 0.012; MoE gradients a factor dp off score about 1.
MOE_MAX_REL_ERROR = 0.04
_EP_EXPERT = re.compile(r"(?P<ffn>.*/ffn)/w_(?P<kind>gate|up|down)_(?P<local>\d+)$")
_DENSE_LINEAR = {"gate": "w1", "up": "w3", "down": "w2"}


def _deepseek(moe_type: str, moe_axis_name: str | None) -> DeepSeek:
    return DeepSeek(
        DeepSeekConfig(
            vocab_size=MOE_VOCAB,
            dim=128,
            inter_dim=256,
            moe_inter_dim=64,
            n_layers=2,
            n_dense_layers=1,
            n_heads=4,
            n_routed_experts=MOE_EXPERTS,
            n_shared_experts=1,
            n_activated_experts=MOE_EXPERTS,
            n_expert_groups=1,
            n_limited_groups=1,
            score_func="softmax",
            route_scale=1.0,
            moe_type=moe_type,
            q_lora_rank=0,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=32,
            max_seq_len=MOE_SEQ_LEN,
            runner_type=RunnerType.Default,
            moe_axis_name=moe_axis_name,
        )
    )


def _from_dense(dense: DeepSeek, name: str, read, ep_size: int) -> np.ndarray:
    """``read`` of the dense parameters that expert-parallel parameter ``name`` is made of, in rank order."""
    sources = dense.parameters()
    match = _EP_EXPERT.match(name)
    if match is None:
        return read(sources[name])
    linear, local = _DENSE_LINEAR[match["kind"]], int(match["local"])
    experts = range(local, MOE_EXPERTS, MOE_EXPERTS // ep_size)
    return np.concatenate([read(sources[f"{match['ffn']}/experts/{e}/{linear}/weight"]) for e in experts])


def _synced_step(model: DeepSeek, ids: np.ndarray, targets: np.ndarray, mapper) -> None:
    model.train()
    ids = ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper)
    targets = ttml.autograd.Tensor.from_numpy(targets, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper)
    ttml.ops.loss.cross_entropy_loss(model(ids, None), targets).backward(False)
    ttml.autograd.AutoContext.get_instance().reset_graph()
    ttml.sync_gradients(model.parameters())


class TestExpertParallelMoE:
    def test_gradients_match_the_dense_moe(self, dp_tp_mesh):
        dp = ep = dp_tp_mesh.axis_size("dp")
        ttml.manual_seed(0)
        dense = _deepseek("dense", None)
        model = _deepseek("sparse_ep", "dp")
        params = model.parameters()
        for name, param in params.items():
            mapper = ttml.Sharding.from_tensor(param).derive_mapper()
            ttml.autograd.assign_numpy(
                param, _from_dense(dense, name, lambda t: gather_like(t.get_value(NATIVE), t), ep), mapper=mapper
            )
        rng = np.random.default_rng(7)
        ids = rng.integers(0, MOE_VOCAB, (dp, 1, 1, MOE_SEQ_LEN)).astype(np.uint32)
        targets = rng.integers(0, MOE_VOCAB, (dp, MOE_SEQ_LEN)).astype(np.uint32)
        _synced_step(dense, ids, targets, None)
        _synced_step(model, ids, targets, dp_tp_mesh.axis_mapper("dp", tdim=0))

        moe = [name for name in params if "/ffn/" in name and "/blocks/1/" in name]
        assert any(_EP_EXPERT.match(name) for name in moe) and any("shared_experts" in name for name in moe), moe
        errors = {}
        for name, param in params.items():
            expected = _from_dense(dense, name, lambda t: gather_like(t.get_grad(), t), ep)
            scale = np.linalg.norm(expected)
            assert scale > 0, f"{name}: zero gradient, agreement would prove nothing"
            errors[name] = np.linalg.norm(gather_like(param.get_grad(), param) - expected) / scale
        worst = sorted(errors.items(), key=lambda item: -item[1])[:5]
        assert worst[0][1] <= MOE_MAX_REL_ERROR, f"relative errors, worst first: {worst}"


# --- FSDP x TP on [2, 2] ("fsdp", "tp") --------------------------------------------------------------------------


def _lora_block():
    """LoRA on a TPBlock's column- and row-parallel linears. The column-parallel lora_A is replicated on "tp" and its
    gradient partial there, rank 32 leaves FSDP shards that end inside a tile, and the frozen base weights get no
    gradient but must be resharded all the same. lora_B starts random: its zero init would leave lora_A no gradient."""
    model = LoraModel(TPBlock(DIM), LoraConfig(rank=32, target_modules=["up", "down"]))
    for name, t in model.parameters().items():
        if name.endswith("lora_B"):
            sharding = ttml.Sharding.from_tensor(t)
            values = np.random.default_rng(1).standard_normal(sharding.global_shape, dtype=np.float32)
            ttml.autograd.assign_numpy(t, values, mapper=sharding.derive_mapper())
    return model


class _SequenceParallelLlamaBlock(AbstractModuleBase):
    """A sequence-parallel Llama block between the sequence scatter and gather Llama puts around its blocks, so it
    takes and returns a stream replicated on "tp" like the other blocks. Its norms are marked partial on "tp"."""

    def __init__(self, runner_type: RunnerType) -> None:
        super().__init__()
        rope = ttml.ops.rope.build_rope_params(DIM, DIM // 2)
        self.block = LlamaBlock(DIM, 2, 2, rope, intermediate_size=DIM, tp_strategy=TPStrategy.TENSOR_SEQUENCE)
        self.runner_type = runner_type

    def forward(self, x):
        tp = ttml.mesh().axis_index("tp")
        x = ttml.ops.distributed.scatter(x, SEQUENCE_DIM, tp)
        mask = build_causal_mask(DIM, device=True)
        if self.runner_type == RunnerType.MemoryEfficient:
            x = memory_efficient_runner(self.block, x, mask)
        else:
            x = self.block(x, mask)
        return ttml.ops.distributed.all_gather(x, SEQUENCE_DIM, tp, ttml.ops.distributed.GradOutputType.REPLICATED)


# Each block, and the trainable parameters FSDP leaves whole: "tp" took their only dim larger than one.
BLOCKS = {
    "linears": (lambda: TPBlock(DIM), ("up/bias",)),
    "lora": (_lora_block, ()),
    "fused_swiglu": (lambda: DeepSeekMLP(DIM, 2 * DIM, tp_axis_name="tp"), ()),
    "sequence_parallel_llama": (lambda: _SequenceParallelLlamaBlock(RunnerType.Default), ()),
    "sequence_parallel_llama_memory_efficient": (lambda: _SequenceParallelLlamaBlock(RunnerType.MemoryEfficient), ()),
}
STAGES = ("weights", "step 1 gradient", "step 2 gradient", "update")
# Measured with and without FSDP: the linears, the fused SwiGLU and the sequence-parallel Llama block (under either
# runner) agree bit for bit. LoRA's gradients are bit for bit the same on each device before they are reduced, but its
# small tensors are summed by the composite all-reduce without FSDP and by a reduce-scatter with it, and the two round
# some elements of the bf16 sum differently: 0.0024 (relative Frobenius error) at step 2, and 0.0104 in the update, the
# difference of two bf16 weights. At this rate a stale gather would score 0.14 (SwiGLU) to 0.91 (LoRA) at step 2.
FSDP_TP_LR = 10.0
FSDP_TP_TOL = {"lora": 0.02}
FSDP_TP_TOL_BIT_FOR_BIT = 0.005


def _train(model, x, weighting) -> list[dict]:
    """Two SGD steps on ``x``."""
    params = model.parameters()
    trainable = {name: t for name, t in params.items() if t.get_requires_grad()}
    opt = ttml.optimizers.SGD(params, ttml.optimizers.SGDConfig.make(FSDP_TP_LR, 0.0, 0.0, 0.0, False))
    weights = {name: gather_like(t.get_value(NATIVE), t) for name, t in trainable.items()}
    stages = [weights]
    for _ in range(2):
        opt.zero_grad()
        ttml.ops.unary.mean(model(x) * weighting).backward(False)
        ttml.autograd.AutoContext.get_instance().reset_graph()
        ttml.sync_gradients(params)
        stages.append({name: gather_like(t.get_grad(), t) for name, t in trainable.items()})
        opt.step()
    stages.append({name: gather_like(t.get_value(NATIVE), t) - weights[name] for name, t in trainable.items()})
    return stages


def _relative_error(actual: dict, expected: dict) -> float:
    """Over all parameters at once. Above some r, it puts at least one parameter's own relative error above r."""
    error = sum(np.sum((actual[n] - expected[n]) ** 2) for n in expected)
    return float(np.sqrt(error / sum(np.sum(e**2) for e in expected.values())))


class TestFSDPxTP:
    @pytest.mark.parametrize("block", BLOCKS)
    def test_training_matches_the_block_without_fsdp(self, fsdp_tp_mesh, block):
        make, left_whole = BLOCKS[block]
        fsdp = fsdp_tp_mesh.axis_index("fsdp")
        shape, mapper = (fsdp_tp_mesh.axis_size("fsdp"), 1, DIM, DIM), fsdp_tp_mesh.axis_mapper("fsdp", tdim=0)
        x, weighting = _input(shape, mapper, scale=1.0), _input(shape, mapper, seed=1, scale=1.0)
        runs = []
        for use_fsdp in (False, True):
            ttml.common.utils.set_seed(0)
            model = make()
            if use_fsdp:
                ttml.fsdp.fully_shard(model)
                trainable = {name: t for name, t in model.parameters().items() if t.get_requires_grad()}
                whole = [name for name, t in trainable.items() if ttml.Sharding.from_tensor(t).shard_dim(fsdp) is None]
                assert whole == [name for name in trainable if name.endswith(left_whole)], "FSDP left these whole"
            runs.append(_train(model, x, weighting))

        reference, sharded = runs
        tol = FSDP_TP_TOL.get(block, FSDP_TP_TOL_BIT_FOR_BIT)
        # A second forward on stale shards repeats step 1's gradients; this is the error they would score at step 2.
        stale = _relative_error(reference[1], reference[2])
        assert stale > 10 * tol, f"step 1's gradients are within {stale:.3g} of step 2's: a stale gather passes"
        for stage, expected, actual in zip(STAGES, reference, sharded):
            assert actual.keys() == expected.keys(), stage
            for name, e in expected.items():
                scale = np.linalg.norm(e)
                assert scale > 0 or stage == "weights", f"{stage} of {name}: all zero, agreement would prove nothing"
                error = np.linalg.norm(actual[name] - e)
                assert error <= tol * scale, f"{stage} of {name}: relative error {error / scale:.3g}"

    @pytest.mark.parametrize("case", FSDP_TP_CASES)
    def test_sync_reduces_exactly(self, fsdp_tp_mesh, case):
        _assert_llama_sync_reduces_exactly(case)


# --- HSDP on [2, 2] ("dp", "fsdp") ------------------------------------------------------------------------------


class TestHSDP:
    def _batch_mapper(self, mesh):
        """train.py's HSDP batch mapper: a whole-mesh Shard(0), read as Shard on both data axes."""
        return ttml.core.distributed.shard_tensor_to_mesh_mapper(_device(), 0, None)

    def test_managed_params_are_reduced_on_dp_only(self, hsdp_mesh):
        """One sample per device under the whole-mesh batch mapper. FSDP-managed parameters are Shard on "fsdp", so
        they are reduced on "dp" only and keep their "fsdp" Shard; the unmanaged head is averaged on both axes."""
        dp, fsdp = hsdp_mesh.axis_index("dp"), hsdp_mesh.axis_index("fsdp")
        D, F = hsdp_mesh.axis_size("dp"), hsdp_mesh.axis_size("fsdp")
        model = _build_fsdp_model("eager")
        params = model.parameters()
        x = _input((D * F, 1, DIM, DIM), self._batch_mapper(hsdp_mesh))
        batch = x.get_value(NATIVE)
        assert list(batch.shape) == [1, 1, DIM, DIM]
        assert len(_placements(batch)) == 1  # the collapsed whole-mesh label ...
        assert is_sharded_on_axis(batch, dp) and is_sharded_on_axis(batch, fsdp)  # ... read as Shard on both axes
        _backward(model, x)

        _assert_per_axis_labels(params, 2)
        before = {name: _shards(t.get_grad()) for name, t in params.items()}
        for name, t in params.items():
            value = t.get_value(NATIVE)
            managed = ttml.fsdp.is_fsdp_managed(t)
            assert is_sharded_on_axis(value, fsdp) == managed, name
            assert not is_sharded_on_axis(value, dp), name
            assert _reduce_set(value, (dp, fsdp)) == ({dp} if managed else {dp, fsdp}), name
            assert is_sharded_on_axis(t.get_grad(), dp), f"{name}: every device computed a different gradient"

        ttml.core.distributed.synchronize_gradients(params, [dp])

        for name, t in params.items():
            grad = t.get_grad()
            assert not is_sharded_on_axis(grad, dp), name
            assert is_sharded_on_axis(grad, fsdp), name
            _assert_mean_over_axis(before[name], _shards(grad), hsdp_mesh.shape, dp, name)

        # The fsdp reduction goes through the same overload: the managed parameters are Shard on "fsdp", so their
        # reduce-scattered gradients are left alone, and only the head is averaged there.
        after_dp = {name: _shards(t.get_grad()) for name, t in params.items()}
        ttml.core.distributed.synchronize_gradients(params, [fsdp])
        for name, t in params.items():
            if ttml.fsdp.is_fsdp_managed(t):
                for device, (b, a) in enumerate(zip(after_dp[name], _shards(t.get_grad()))):
                    np.testing.assert_array_equal(a, b, err_msg=f"{name}: device {device}'s piece was averaged on fsdp")
        head = params["Model/head/weight"].get_grad()
        assert not is_sharded_on_axis(head, dp)
        assert not is_sharded_on_axis(head, fsdp)
        _assert_mean_over_axis(after_dp["Model/head/weight"], _shards(head), hsdp_mesh.shape, fsdp, "head over fsdp")

    def test_fp32_reduce_keeps_the_hsdp_shard_in_fp32_for_the_dp_sum(self, hsdp_mesh):
        """With reduce_dtype float32 the fsdp reduce-scatter leaves an fp32 shard, and the dp sum adds those fp32 shards
        on the FPU. The result is the float64 mean within the bound of those two FPU adds. Rounded to bf16, about
        98.5% of the elements land on the exact mean, against 76% when the shard is rounded to bf16 before the dp sum;
        an exact fp32 sum would give 100%, the FPU's 10-bit alignment the rest."""
        dp = hsdp_mesh.axis_index("dp")
        D, F = hsdp_mesh.axis_size("dp"), hsdp_mesh.axis_size("fsdp")
        assert (D, F) == (2, 2)  # the bound below counts one add per axis
        values = make_gradients(64, 128, seed=8, devices=D * F)  # device k in mesh row-major order holds values[k]
        exact = values.astype(np.float64).mean(axis=0)

        def reduce(round_shard_to_bf16: bool):
            _layer, state, tensor, shard_dim = sharded_linear(ttnn.DataType.FLOAT32)
            gradients = upload(
                values.reshape(D, F, *values.shape[2:]), [ttnn.PlacementShard(0), ttnn.PlacementShard(1)]
            )
            tensor.set_grad(gradients.get_value())
            state.backward_post()
            assert tensor.get_grad().dtype == ttnn.DataType.FLOAT32
            if round_shard_to_bf16:
                tensor.set_grad(ttnn.typecast(tensor.get_grad(), ttnn.DataType.BFLOAT16))
            parameters = ttml.NamedParameters()
            parameters["weight"] = tensor
            ttml.core.distributed.synchronize_gradients(parameters, [dp], reduce_dtype=ttnn.DataType.FLOAT32)
            grad = tensor.get_grad()
            device_shards = _shards(grad)
            shards = np.stack(device_shards).reshape(D, F, *device_shards[0].shape)
            for row in range(1, D):
                np.testing.assert_array_equal(shards[row], shards[0])  # every dp rank holds the same mean
            return grad.dtype, np.concatenate(list(shards[0]), axis=shard_dim)[0]

        fp32_dtype, fp32_shard = reduce(round_shard_to_bf16=False)
        bf16_dtype, bf16_shard = reduce(round_shard_to_bf16=True)

        assert fp32_dtype == ttnn.DataType.FLOAT32
        assert bf16_dtype == ttnn.DataType.BFLOAT16
        # The fsdp sum of each dp row (halved), then the dp sum of the two row means (halved).
        row_means = values.astype(np.float64).reshape(D, F, *values.shape[1:]).mean(axis=1)
        bound = 0.25 * (fpu_add_bound(values[0], values[1]) + fpu_add_bound(values[2], values[3]))
        bound = bound + 0.5 * fpu_add_bound(row_means[0], row_means[1])
        error = np.abs(fp32_shard.astype(np.float64) - exact)
        np.testing.assert_array_less(error, bound * 1.01 + np.abs(exact) * 2.0**-22 + 1e-30)
        fp32_share = np.mean(to_bf16(fp32_shard) == exact_mean(values)[0])
        assert fp32_share >= 0.97
        assert fp32_share > np.mean(bf16_shard == exact_mean(values)[0])
        assert np.mean(error) < np.mean(np.abs(bf16_shard - exact))

    def test_causal_lm_collate_hsdp(self, hsdp_mesh):
        """``causal_lm_collate_fn`` with the HSDP mapper: the standard per-device shapes, the collapsed whole-mesh
        label (one placement, read as Shard on both data axes) and device k, in mesh row-major order, holding
        examples k*B/(D*F) + j."""
        from ttml.datasets import causal_lm_collate_fn

        dp, fsdp = hsdp_mesh.axis_index("dp"), hsdp_mesh.axis_index("fsdp")
        D, F = hsdp_mesh.axis_size("dp"), hsdp_mesh.axis_size("fsdp")
        seq_len, batch = 32, 2 * D * F
        examples = [
            {
                "input_ids": list(range(100 * i, 100 * i + seq_len)),
                "labels": list(range(100 * i + 1, 100 * i + seq_len + 1)),
            }
            for i in range(batch)
        ]

        out = causal_lm_collate_fn(examples, seq_len=seq_len, mapper=self._batch_mapper(hsdp_mesh))

        per_device = batch // (D * F)
        ids, labels, mask = (
            out.input_ids.get_value(NATIVE),
            out.labels.get_value(NATIVE),
            out.loss_mask.get_value(NATIVE),
        )
        assert list(ids.shape) == [per_device, 1, 1, seq_len]
        assert list(labels.shape) == [per_device, seq_len]
        assert list(mask.shape) == [per_device, 1, seq_len, 1]
        for tensor in (ids, labels, mask):
            assert is_sharded_on_axis(tensor, dp) and is_sharded_on_axis(tensor, fsdp)
            assert len(_placements(tensor)) == 1  # the collapsed label: the whole mesh shards dim 0
            assert list(tensor.tensor_topology().distribution_shape()) == [D * F]

        shards = _shards(ids)
        assert len(shards) == D * F
        for index, shard in enumerate(shards):  # mesh row-major order, like the label's coordinates
            first = index * per_device
            expected = np.stack(
                [np.array(examples[first + j]["input_ids"], dtype=np.float32) for j in range(per_device)]
            )
            np.testing.assert_array_equal(shard[:, 0, 0, :], expected)

    @pytest.mark.parametrize("case", HSDP_CASES)
    def test_sync_reduces_exactly(self, hsdp_mesh, case):
        _assert_llama_sync_reduces_exactly(case)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
