# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Megatron sequence parallelism (``TPStrategy.TENSOR_SEQUENCE``) on Llama.
The oracle is classic tensor parallelism on the same weights.

The second half checks the sequence-parallel linears the modules are built from
(``sp_column_parallel_linear`` / ``sp_row_parallel_linear``) against the collective + linear
sequence they replace, and the switch between their composed and fused implementations.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models import EmbeddingPlacement, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.modules import LoraConfig, LoraModel
from ttml.parallel import SEQUENCE_DIM, TPStrategy, is_sequence_parallel
from bf16_ulp import assert_within_bf16_ulp

TP_AXIS_SIZE = 2  # the 'tp' extent of conftest's tp_mesh fixture

# Logits agree bitwise at tp=2 and the worst gradient disagreement is 1 ULP, entirely in the
# norm gammas. The limit stays above both so that summing more than two ranks, whose rounding
# genuinely is order-dependent, does not become flaky.
MAX_ULP = 2.0

# Two tiles of sequence per rank; one tile would not exercise a multi-tile shard.
SEQ_LEN = 32 * TP_AXIS_SIZE * 2
HIDDEN = 128
N_HEADS = 4
N_KV_HEADS = 2
N_LAYERS = 2
INTERMEDIATE = 128
VOCAB = 128
TP, SP = TPStrategy.TENSOR, TPStrategy.TENSOR_SEQUENCE
PLACEMENTS = list(EmbeddingPlacement)


def config(tp_strategy: TPStrategy, placement=EmbeddingPlacement.VocabParallel, **overrides) -> LlamaConfig:
    return LlamaConfig(
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        num_hidden_layers=N_LAYERS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ_LEN,
        tp_strategy=tp_strategy,
        embedding_placement=placement,
        weight_tying=WeightTyingType.Disabled,
        **overrides,
    )


def paired_models(placement=EmbeddingPlacement.VocabParallel, **overrides) -> tuple[Llama, Llama]:
    """A TP model and an SP model built from one seed: the same weights in independent storage."""
    models = []
    for strategy in (TP, SP):
        ttml.manual_seed(0)
        models.append(Llama(config(strategy, placement, **overrides)))
    return tuple(models)


def adamw(model):
    return ttml.optimizers.AdamW(model.parameters(), ttml.optimizers.AdamWConfig.make(1e-2, 0.9, 0.999, 1e-8, 0.0))


def token_ids(batch: int, seq_len: int, seed: int):
    ids = np.random.default_rng(seed).integers(0, VOCAB, (batch, 1, 1, seq_len)).astype(np.uint32)
    return ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)


def causal_mask(seq_len: int):
    mask = np.tril(np.ones((1, 1, seq_len, seq_len), dtype=np.float32))
    return ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)


def per_rank(tensor) -> np.ndarray:
    """Every tp rank's copy of ``tensor`` stacked along dim 1, which is 1 on every tensor here, so SP
    and TP tensors compare rank by rank whatever their placement. The layout is imposed by a composer
    rather than read off the tensor: activation and gradient topologies are stale after collectives."""
    mesh = ttml.mesh()
    dims = [0] * len(mesh.shape)
    dims[mesh.axis_index("tp")] = 1
    device = ttml.autograd.AutoContext.get_instance().get_device()
    composer = ttnn.create_mesh_composer(device, ttnn.MeshComposerConfig(dims))
    return tensor.to_numpy(ttnn.DataType.FLOAT32, composer=composer).astype(np.float64)


def backward(model, ids, mask) -> None:
    model.train()
    model(ids, mask).backward(retain_graph=False)
    ttml.autograd.AutoContext.get_instance().reset_graph()


def assert_same_grads(sp_model, tp_model, label: str = "") -> None:
    tp_params, sp_params = tp_model.parameters(), sp_model.parameters()
    assert set(tp_params.keys()) == set(sp_params.keys())
    for name, tp_param in tp_params.items():
        if not tp_param.get_requires_grad():
            continue
        assert sp_params[name].is_grad_initialized(), f"{name}: SP grad missing"
        assert_within_bf16_ulp(
            per_rank(sp_params[name].get_grad_tensor()),
            per_rank(tp_param.get_grad_tensor()),
            f"grad {name} {label}",
            MAX_ULP,
        )


@pytest.fixture(autouse=True)
def reset_graph():
    yield
    ttml.autograd.AutoContext.get_instance().reset_graph()


@pytest.mark.requires_device
@pytest.mark.usefixtures("tp_mesh")
class TestMatchesTensorParallel:
    @pytest.mark.parametrize("placement", PLACEMENTS, ids=lambda p: p.name)
    @pytest.mark.parametrize("batch", [1, 2])
    def test_logits(self, placement, batch):
        tp_model, sp_model = paired_models(placement)
        tp_model.eval()
        sp_model.eval()
        ids, mask = token_ids(batch, SEQ_LEN, seed=1234 + batch), causal_mask(SEQ_LEN)

        # Logits stay vocab-sharded under both strategies (gather_output=False).
        tp_logits, sp_logits = per_rank(tp_model(ids, mask)), per_rank(sp_model(ids, mask))

        assert tp_logits.std() > 1e-3, "logits are ~constant; agreement would prove nothing"
        assert_within_bf16_ulp(sp_logits, tp_logits, f"logits {placement.name} batch={batch}", MAX_ULP)

    @pytest.mark.parametrize("placement", PLACEMENTS, ids=lambda p: p.name)
    def test_gradients_after_sync(self, placement):
        tp_model, sp_model = paired_models(placement, attention_bias=True)
        ids, mask = token_ids(1, SEQ_LEN, seed=77), causal_mask(SEQ_LEN)
        for model in (tp_model, sp_model):
            backward(model, ids, mask)

        ttml.sync_gradients(sp_model.parameters())

        assert_same_grads(sp_model, tp_model)

    def test_gradients_after_an_optimizer_step(self):
        """The optimizer rewrites each parameter's mesh topology in place; the sync must not read it."""
        tp_model, sp_model = paired_models()
        optimizers = [adamw(model) for model in (tp_model, sp_model)]
        for step in range(2):
            ids, mask = token_ids(1, SEQ_LEN, seed=step), causal_mask(SEQ_LEN)
            for model, optimizer in zip((tp_model, sp_model), optimizers):
                optimizer.zero_grad()
                backward(model, ids, mask)
            ttml.sync_gradients(sp_model.parameters())
            assert_same_grads(sp_model, tp_model, f"step {step}")
            for optimizer in optimizers:
                optimizer.step()

    def test_lora(self):
        """The adapters reuse the base layers' collectives, so SP needs no LoRA-specific math."""
        tp_model, sp_model = paired_models()
        lora = LoraConfig(
            rank=8, target_modules=["qkv_linear", "out_linear", "w_gate_up", "w2"], trainable_modules=["_norm", "ln_fc"]
        )
        wrapped = []
        for model in (tp_model, sp_model):
            np.random.seed(0)  # lora_A is drawn from numpy's global RNG
            wrapped.append(LoraModel(model, lora))
        tp_lora, sp_lora = wrapped
        ids, mask = token_ids(1, SEQ_LEN, seed=9), causal_mask(SEQ_LEN)
        for model in (tp_lora, sp_lora):
            backward(model, ids, mask)

        ttml.sync_gradients(sp_lora.parameters())

        assert_same_grads(sp_lora, tp_lora)

    def test_marks_the_parameters_of_the_sequence_sharded_region(self):
        """Exactly the norm gains and the row-parallel bias see a per-rank slice of the sequence."""
        tp_model, sp_model = paired_models(attention_bias=True)
        assert not any(is_sequence_parallel(p) for _, p in tp_model.parameters().items())

        marked = {name for name, p in sp_model.parameters().items() if is_sequence_parallel(p)}
        per_block = ("attention_norm/gamma", "mlp_norm/gamma", "attention/out_linear/bias")
        expected = {f"Llama/blocks/{i}/{suffix}" for i in range(N_LAYERS) for suffix in per_block}
        assert marked == set.union(expected, {"Llama/ln_fc/gamma"})


@pytest.mark.requires_device
@pytest.mark.usefixtures("tp_mesh")
class TestValidation:
    def test_rejects_input_sequence_not_divisible_by_32_tp(self, expect_error):
        model = Llama(config(SP))
        with expect_error(ValueError, "input sequence length divisible by"):
            model(token_ids(1, 32, seed=5), causal_mask(32))

    def test_rejects_kv_cache(self, expect_error):
        """Single-token decode has no sequence to shard, so the model refuses before any collective."""
        model = Llama(config(SP))
        kv_cache = ttml.models.KvCache(
            num_layers=N_LAYERS,
            batch_size=1,
            num_groups=N_KV_HEADS // TP_AXIS_SIZE,
            max_seq_len=SEQ_LEN,
            head_dim=HIDDEN // N_HEADS,
        )
        with expect_error(NotImplementedError, "sequence_parallel does not support"):
            model(token_ids(1, SEQ_LEN, seed=5), causal_mask(SEQ_LEN), kv_cache=kv_cache, new_tokens=SEQ_LEN)


class TestTPStrategy:
    def test_from_flags(self):
        assert TPStrategy.from_flags(False) is TPStrategy.NONE
        assert TPStrategy.from_flags(True) is TPStrategy.TENSOR
        assert TPStrategy.from_flags(True, enable_sp=True) is TPStrategy.TENSOR_SEQUENCE

    def test_rejects_sp_without_tp(self, expect_error):
        with expect_error(ValueError, "requires enable_tp"):
            TPStrategy.from_flags(False, enable_sp=True)

    def test_enable_sp_is_keyword_only(self, expect_error):
        with expect_error(TypeError, "positional argument"):
            TPStrategy.from_flags(True, True)  # type: ignore

    @pytest.mark.parametrize(
        "strategy,tensor,sequence",
        [
            (TPStrategy.NONE, False, False),
            (TPStrategy.TENSOR, True, False),
            (TPStrategy.TENSOR_SEQUENCE, True, True),
        ],
    )
    def test_properties(self, strategy, tensor, sequence):
        assert strategy.tensor_parallel is tensor
        assert strategy.sequence_parallel is sequence


# ---------------------------------------------------------------------------
# The sequence-parallel linears against the collective + linear sequence they replace
# ---------------------------------------------------------------------------

SPLinearImpl = ttml.ops.distributed.SPLinearImpl
# Milestone 2 of issue #52944: flip once ttnn.experimental.{all_gather_matmul_sp_async,
# matmul_reduce_scatter_sp_async} land, and replace test_fused_is_not_landed with a Fused-vs-Composed
# comparison in ULP (the fused ops overlap the collective with the matmul, so they need not be bitwise).
FUSED_LANDED = False
IN_FEATURES, OUT_FEATURES = 128, 192  # tile-aligned after sharding across tp


def normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape) * 0.1).astype(np.float32)


def mesh_tensor(data: np.ndarray, shard_dim: int | None, requires_grad: bool = False):
    """``data`` on the tp mesh, sharded along ``shard_dim`` across tp (replicated when None)."""
    kwargs = {} if shard_dim is None else {"mapper": ttml.mesh().axis_mapper("tp", shard_dim)}
    tensor = ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, **kwargs)
    tensor.set_requires_grad(requires_grad)
    return tensor


def column_linear_reference(x, weight, bias, cluster_axis):
    """What ColumnParallelLinear(sequence_parallel=True) issued before the ops existed."""
    gathered = ttml.ops.distributed.all_gather(
        x, SEQUENCE_DIM, cluster_axis, ttml.ops.distributed.GradOutputType.SHARDED
    )
    return ttml.ops.linear.linear(gathered, weight, bias)


def row_linear_reference(x, weight, cluster_axis):
    """What RowParallelLinear(sequence_parallel=True) issued before the ops existed (bias excluded)."""
    return ttml.ops.distributed.reduce_scatter(ttml.ops.linear.linear(x, weight, None), SEQUENCE_DIM, cluster_axis)


def forward_backward(op, operands: dict, grad_out: tuple, cluster_axis: int) -> dict[str, np.ndarray]:
    """Forward and backward of ``op`` from ``grad_out``; each operand is ``(data, shard_dim)`` (``None``
    data for an absent bias) and gets a fresh device copy, so two runs never share a tensor.
    Returns the per-rank output and the per-rank gradient of every operand."""
    tensors = {
        name: None if data is None else mesh_tensor(data, shard_dim, requires_grad=True)
        for name, (data, shard_dim) in operands.items()
    }
    out = op(**tensors, cluster_axis=cluster_axis)
    out.set_grad(mesh_tensor(*grad_out).get_value())
    out.backward(False)
    result = {"out": per_rank(out)}
    result.update({name: per_rank(t.get_grad_tensor()) for name, t in tensors.items() if t is not None})
    ttml.autograd.AutoContext.get_instance().reset_graph()
    return result


def assert_bitwise_equal(got: dict[str, np.ndarray], expected: dict[str, np.ndarray], label: str) -> None:
    assert got.keys() == expected.keys()
    for name, reference in expected.items():
        assert np.isfinite(reference).all(), f"{label}: {name} reference is not finite"
        assert reference.std() > 0, f"{label}: {name} reference is constant; agreement would prove nothing"
        assert got[name].shape == reference.shape, f"{label}: {name} shape {got[name].shape} != {reference.shape}"
        if not np.array_equal(got[name], reference):
            differing = int((got[name] != reference).sum())
            raise AssertionError(
                f"{label}: {name} differs in {differing}/{reference.size} elements, "
                f"max |diff| = {np.abs(got[name] - reference).max():.3g}"
            )


@pytest.fixture
def composed_afterwards():
    yield
    ttml.ops.distributed.set_sp_linear_impl(SPLinearImpl.COMPOSED)


@pytest.mark.requires_device
@pytest.mark.usefixtures("tp_mesh")
class TestSPLinearOps:
    """Under the Composed implementation the ops issue exactly the ttnn ops of the sequence they replace,
    so the output and every gradient must agree bit for bit, not just within ULP."""

    @pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "no_bias"])
    @pytest.mark.parametrize("batch", [1, 2])
    def test_column_parallel(self, batch, has_bias):
        axis = ttml.mesh().axis_index("tp")
        rng = np.random.default_rng(10 * batch + has_bias)
        operands = {
            "x": (normal(rng, (batch, 1, SEQ_LEN, IN_FEATURES)), SEQUENCE_DIM),  # per rank [B,1,S/T,K]
            "weight": (normal(rng, (1, 1, OUT_FEATURES, IN_FEATURES)), 2),  # per rank [1,1,N/T,K]
            "bias": (normal(rng, (1, 1, 1, OUT_FEATURES)) if has_bias else None, 3),
        }
        grad_out = (normal(rng, (batch, 1, SEQ_LEN, OUT_FEATURES)), 3)  # per rank [B,1,S,N/T]

        got = forward_backward(ttml.ops.distributed.sp_column_parallel_linear, operands, grad_out, axis)
        expected = forward_backward(column_linear_reference, operands, grad_out, axis)

        assert got["out"].shape == (batch, TP_AXIS_SIZE, SEQ_LEN, OUT_FEATURES // TP_AXIS_SIZE)
        assert set(got) == {"out", "x", "weight"} | ({"bias"} if has_bias else set())
        assert_bitwise_equal(got, expected, f"column batch={batch} bias={has_bias}")

    @pytest.mark.parametrize("batch", [1, 2])
    def test_row_parallel(self, batch):
        axis = ttml.mesh().axis_index("tp")
        rng = np.random.default_rng(20 + batch)
        operands = {
            "x": (normal(rng, (batch, 1, SEQ_LEN, IN_FEATURES)), 3),  # per rank [B,1,S,K/T]
            "weight": (normal(rng, (1, 1, OUT_FEATURES, IN_FEATURES)), 3),  # per rank [1,1,N,K/T]
        }
        grad_out = (normal(rng, (batch, 1, SEQ_LEN, OUT_FEATURES)), SEQUENCE_DIM)  # per rank [B,1,S/T,N]

        got = forward_backward(ttml.ops.distributed.sp_row_parallel_linear, operands, grad_out, axis)
        expected = forward_backward(row_linear_reference, operands, grad_out, axis)

        assert got["out"].shape == (batch, TP_AXIS_SIZE, SEQ_LEN // TP_AXIS_SIZE, OUT_FEATURES)
        assert_bitwise_equal(got, expected, f"row batch={batch}")


@pytest.mark.usefixtures("composed_afterwards")
class TestSPLinearImpl:
    def test_switch_round_trips(self):
        set_impl, get_impl = ttml.ops.distributed.set_sp_linear_impl, ttml.ops.distributed.get_sp_linear_impl
        assert get_impl() == SPLinearImpl.COMPOSED, "composed is the default until the fused ops land"
        set_impl("fused")
        assert get_impl() == SPLinearImpl.FUSED
        set_impl(SPLinearImpl.COMPOSED)
        assert get_impl() == SPLinearImpl.COMPOSED
        set_impl(SPLinearImpl.FUSED)
        set_impl("composed")  # the device-config spelling
        assert get_impl() == SPLinearImpl.COMPOSED

    def test_rejects_unknown_name(self, expect_error):
        with expect_error(ValueError, "'composed' or 'fused'"):
            ttml.ops.distributed.set_sp_linear_impl("eager")
        assert ttml.ops.distributed.get_sp_linear_impl() == SPLinearImpl.COMPOSED

    def test_device_config_knob(self, expect_error):
        """`device_config.sp_linear_impl` is what train.py feeds to the setter at startup."""
        from ttml.common.config import DeviceConfig

        assert DeviceConfig({"device_config": {"enable_tp": True, "enable_sp": True}}).sp_linear_impl == "composed"
        assert DeviceConfig({"device_config": {"sp_linear_impl": "fused"}}).sp_linear_impl == "fused"
        with expect_error(ValueError, "'composed' or 'fused'"):
            DeviceConfig({"device_config": {"sp_linear_impl": "eager"}})


@pytest.mark.requires_device
@pytest.mark.usefixtures("tp_mesh", "composed_afterwards")
class TestSPLinearFusedImpl:
    @pytest.mark.skipif(FUSED_LANDED, reason="the fused ops landed: compare Fused with Composed in ULP instead")
    def test_fused_is_not_landed(self, expect_error):
        """Selecting the fused ops before they exist fails at the first SP linear instead of silently composing."""
        ttml.ops.distributed.set_sp_linear_impl("fused")
        axis = ttml.mesh().axis_index("tp")
        rng = np.random.default_rng(0)
        x = mesh_tensor(normal(rng, (1, 1, SEQ_LEN, IN_FEATURES)), SEQUENCE_DIM)
        weight = mesh_tensor(normal(rng, (1, 1, OUT_FEATURES, IN_FEATURES)), 2)
        with expect_error(RuntimeError, "not landed"):
            ttml.ops.distributed.sp_column_parallel_linear(x, weight, None, cluster_axis=axis)

        x = mesh_tensor(normal(rng, (1, 1, SEQ_LEN, IN_FEATURES)), 3)
        weight = mesh_tensor(normal(rng, (1, 1, OUT_FEATURES, IN_FEATURES)), 3)
        with expect_error(RuntimeError, "not landed"):
            ttml.ops.distributed.sp_row_parallel_linear(x, weight, axis)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
