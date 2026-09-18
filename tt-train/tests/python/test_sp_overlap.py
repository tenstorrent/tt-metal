# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Two-stream scheduling of the sequence-parallel linears' backward (``ttml.ops.distributed.set_sp_overlap``).

With the overlap on, each backward collective of a Composed SP linear runs on the second command queue and the
CCL sub-device behind a weight-gradient matmul on the compute queue. Nothing is computed differently, so the
acceptance test is bitwise: every output and gradient must equal the same Composed ops issued in order on one
queue on the same (split) core grid. The TP oracle of test_sequence_parallel.py is checked once more here under
the overlap, and the whole suite runs under it with ``TTML_SP_OVERLAP=backward`` (conftest.py).

The mesh comes from ``TTML_SP_TEST_MESH``: ``1x2`` (the default, a line), ``1x4_ring`` or ``1x4_line``; the CCL
sub-device from ``TTML_SP_CCL`` (``rows=1`` default). Each mesh needs its own pytest process (tt-metal sizes the
SystemMesh once per process from the first MGD it opens).
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import pytest

import ttnn
import ttml
from ttml.models import EmbeddingPlacement, WeightTyingType
from ttml.models.llama import Llama, LlamaConfig
from ttml.parallel import TPStrategy
from bf16_ulp import assert_within_bf16_ulp
from conftest import ccl_sub_device_from_env
from sp_linear_testlib import (
    assert_bitwise_equal,
    column_linear_reference,
    column_operands,
    forward_backward,
    per_rank,
    row_linear_reference,
    row_operands,
    sp_linear_impl,
)

_MGD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "configs", "mgd")
_MESHES = {
    "1x2": ((1, 2), "bh_galaxy_1_2_line_line.textproto"),
    "1x4_ring": ((1, 4), "bh_galaxy_1_4_ring_ring.textproto"),
    "1x4_line": ((1, 4), "bh_galaxy_1_4_line_line.textproto"),
}
MESH_KEY = os.environ.get("TTML_SP_TEST_MESH", "1x2")
MESH_SHAPE, _MGD_FILE = _MESHES[MESH_KEY]
TP_AXIS_SIZE = MESH_SHAPE[1]

# TP-oracle limit: as test_sequence_parallel.py at tp=2 (logits bitwise, grads within 1 ULP). At tp=4 the
# reduce-scatter sums four partial products in ring order while the all-reduce oracle sums them in another, so the
# logits genuinely differ (measured 2.5 ULP ring, 2.25 line, identical with the overlap on and off); the overlap
# itself is held bitwise against one queue, the oracle only to this order-dependent rounding.
MAX_ULP = 2.0 if TP_AXIS_SIZE == 2 else 4.0
SEQ_LEN = 32 * TP_AXIS_SIZE * 2  # two tiles of sequence per rank
IN_FEATURES, OUT_FEATURES = 128, 64 * TP_AXIS_SIZE  # tile-aligned after sharding across tp
# A Llama small enough to run in seconds whose every shard is tile-aligned at tp = 2 and 4.
HIDDEN, N_HEADS, N_KV_HEADS, N_LAYERS, INTERMEDIATE, VOCAB = 256, 8, 4, 2, 256, 128


def _close_quietly() -> None:
    try:
        ttml.close_device_mesh()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture(scope="module")
def overlap_mesh():
    """The ``TTML_SP_TEST_MESH`` mesh opened with two command queues and the CCL sub-device split, tp on axis 1."""
    mgd = os.path.realpath(os.path.join(_MGD_DIR, _MGD_FILE))
    previous = os.environ.get("TT_MESH_GRAPH_DESC_PATH")
    if previous and os.path.realpath(previous) != mgd:
        pytest.skip(f"TT_MESH_GRAPH_DESC_PATH points at {previous}; this module needs {mgd}")
    try:
        arch = ttnn.get_arch_name().lower()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"cannot detect the device architecture: {e}")
    if "blackhole" not in arch:
        pytest.skip(f"{_MGD_FILE} describes a Blackhole galaxy, this is {arch}")
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd
    _close_quietly()
    try:
        ttml.open_device_mesh(ttml.Mesh(MESH_SHAPE, ("dp", "tp")), num_command_queues=2)
        ctx = ttml.autograd.AutoContext.get_instance()
        if not ctx.is_parallelism_context_initialized():
            ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=True))
        columns, rows = ccl_sub_device_from_env()
        ctx.enable_ccl_sub_device(columns, rows)
    except Exception as e:  # noqa: BLE001
        _close_quietly()
        _restore(previous)
        pytest.skip(f"needs a {list(MESH_SHAPE)} 'tp' mesh with two command queues: {e}")

    yield MESH_KEY

    ttml.ops.distributed.set_sp_overlap("off")
    _close_quietly()
    _restore(previous)


def _restore(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("TT_MESH_GRAPH_DESC_PATH", None)
    else:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = previous


@pytest.fixture(autouse=True)
def composed_and_clean():
    """Every test here runs the Composed linears (the only ones the overlap applies to) and leaves nothing behind."""
    with sp_linear_impl("composed"):
        yield
    ttml.autograd.AutoContext.get_instance().reset_graph()
    ttml.ops.distributed.set_sp_overlap("off")
    ttml.ops.distributed.set_sp_linear_backward_impl("same")
    assert ttml.ops.distributed.sp_overlap_stats()[0] == 0, "deferred weight gradients survived the backward"


@contextlib.contextmanager
def sp_linear_backward_impl(impl: str):
    """The backward's implementation alone, restored afterwards."""
    previous = ttml.ops.distributed.get_sp_linear_backward_impl()
    ttml.ops.distributed.set_sp_linear_backward_impl(impl)
    try:
        yield
    finally:
        ttml.ops.distributed.set_sp_linear_backward_impl(previous)


@contextlib.contextmanager
def sp_overlap(mode: str):
    previous = ttml.ops.distributed.get_sp_overlap()
    ttml.ops.distributed.set_sp_overlap(mode)
    try:
        yield
    finally:
        ttml.ops.distributed.set_sp_overlap(previous)


def config(tp_strategy: TPStrategy, **overrides) -> LlamaConfig:
    return LlamaConfig(
        hidden_size=HIDDEN,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        num_hidden_layers=N_LAYERS,
        intermediate_size=INTERMEDIATE,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ_LEN,
        tp_strategy=tp_strategy,
        embedding_placement=EmbeddingPlacement.VocabParallel,
        weight_tying=WeightTyingType.Disabled,
        **overrides,
    )


def model(tp_strategy: TPStrategy, **overrides) -> Llama:
    ttml.manual_seed(0)
    return Llama(config(tp_strategy, **overrides))


def token_ids(batch: int, seed: int):
    ids = np.random.default_rng(seed).integers(0, VOCAB, (batch, 1, 1, SEQ_LEN)).astype(np.uint32)
    return ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32)


def causal_mask():
    mask = np.tril(np.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=np.float32))
    return ttml.autograd.Tensor.from_numpy(mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)


def forward_backward_model(net: Llama, ids, mask) -> dict[str, np.ndarray]:
    """Per-rank logits and every parameter gradient of one forward + backward."""
    net.train()
    out = net(ids, mask)
    out.backward(retain_graph=False)
    result = {"logits": per_rank(out)}
    for name, param in net.parameters().items():
        if param.get_requires_grad():
            assert param.is_grad_initialized(), f"{name}: grad missing"
            result[f"grad {name}"] = per_rank(param.get_grad_tensor())
    ttml.autograd.AutoContext.get_instance().reset_graph()
    return result


@pytest.mark.requires_device
@pytest.mark.usefixtures("overlap_mesh")
class TestBitwiseAgainstOneQueue:
    """Overlap on against overlap off, both on the split grid: the same ttnn ops, only queue and order differ."""

    @pytest.mark.parametrize("has_bias", [True, False], ids=["bias", "no_bias"])
    @pytest.mark.parametrize("batch", [1, 2])
    def test_column_parallel(self, batch, has_bias):
        axis = ttml.mesh().axis_index("tp")
        rng = np.random.default_rng(300 + 10 * batch + has_bias)
        operands, grad_out = column_operands(rng, batch, has_bias, SEQ_LEN, IN_FEATURES, OUT_FEATURES)
        label = f"{MESH_KEY} column batch={batch} bias={has_bias}"

        reference = forward_backward(column_linear_reference, operands, grad_out, axis)
        with sp_overlap("off"):
            one_queue = forward_backward(ttml.ops.distributed.sp_column_parallel_linear, operands, grad_out, axis)
        with sp_overlap("backward"):
            two_queues = forward_backward(ttml.ops.distributed.sp_column_parallel_linear, operands, grad_out, axis)

        assert two_queues["out"].shape == (batch, TP_AXIS_SIZE, SEQ_LEN, OUT_FEATURES // TP_AXIS_SIZE)
        assert_bitwise_equal(one_queue, reference, f"{label} one queue")
        assert_bitwise_equal(two_queues, one_queue, f"{label} two queues")

    @pytest.mark.parametrize("batch", [1, 2])
    def test_row_parallel(self, batch):
        axis = ttml.mesh().axis_index("tp")
        rng = np.random.default_rng(320 + batch)
        operands, grad_out = row_operands(rng, batch, SEQ_LEN, IN_FEATURES, OUT_FEATURES)
        label = f"{MESH_KEY} row batch={batch}"

        reference = forward_backward(row_linear_reference, operands, grad_out, axis)
        with sp_overlap("off"):
            one_queue = forward_backward(ttml.ops.distributed.sp_row_parallel_linear, operands, grad_out, axis)
        with sp_overlap("backward"):
            two_queues = forward_backward(ttml.ops.distributed.sp_row_parallel_linear, operands, grad_out, axis)

        assert two_queues["out"].shape == (batch, TP_AXIS_SIZE, SEQ_LEN // TP_AXIS_SIZE, OUT_FEATURES)
        assert_bitwise_equal(one_queue, reference, f"{label} one queue")
        assert_bitwise_equal(two_queues, one_queue, f"{label} two queues")

    @pytest.mark.parametrize("batch", [1, 2])
    def test_llama_gradients(self, batch):
        """Every backward collective of every block deferred and overlapped, across residual accumulation, in
        the order the schedule produces: still bit for bit the one-queue gradients."""
        ids, mask = token_ids(batch, seed=40 + batch), causal_mask()
        with sp_overlap("off"):
            one_queue = forward_backward_model(model(TPStrategy.TENSOR_SEQUENCE, attention_bias=True), ids, mask)
        with sp_overlap("backward"):
            two_queues = forward_backward_model(model(TPStrategy.TENSOR_SEQUENCE, attention_bias=True), ids, mask)
        assert ttml.ops.distributed.sp_overlap_stats()[0] == 0, "the backward left weight gradients deferred"

        assert one_queue.keys() == two_queues.keys()
        assert len(one_queue) > 1 + 2 * N_LAYERS
        assert_bitwise_equal(two_queues, one_queue, f"{MESH_KEY} llama batch={batch}")

    @pytest.mark.parametrize("batch", [1, 2])
    def test_fused_forward_composed_backward(self, batch):
        """The mixed policy (fused forward, two-stream Composed backward): bitwise against the same policy on one
        queue, and the forward really is the fused one (bitwise the all-fused logits)."""
        ids, mask = token_ids(batch, seed=50 + batch), causal_mask()
        with sp_linear_impl("fused"):
            all_fused = forward_backward_model(model(TPStrategy.TENSOR_SEQUENCE), ids, mask)
            with sp_linear_backward_impl("composed"):
                assert ttml.ops.distributed.get_sp_linear_impl() == ttml.ops.distributed.SPLinearImpl.FUSED
                assert ttml.ops.distributed.get_sp_linear_backward_impl() == ttml.ops.distributed.SPLinearImpl.COMPOSED
                with sp_overlap("off"):
                    one_queue = forward_backward_model(model(TPStrategy.TENSOR_SEQUENCE), ids, mask)
                with sp_overlap("backward"):
                    two_queues = forward_backward_model(model(TPStrategy.TENSOR_SEQUENCE), ids, mask)
        assert ttml.ops.distributed.sp_overlap_stats()[0] == 0

        assert_bitwise_equal(two_queues, one_queue, f"{MESH_KEY} fused-forward llama batch={batch}")
        assert np.array_equal(one_queue["logits"], all_fused["logits"]), "the forward did not run the fused ops"
        for name in all_fused:  # the backward differs from all-fused only by the dgrad's implementation: ULP-close
            assert_within_bf16_ulp(one_queue[name], all_fused[name], f"{name} vs all-fused", MAX_ULP)

    def test_repeated_steps_stay_bitwise(self):
        """The retained-buffer window and the semaphore rotation across several backward passes."""
        ids, mask = token_ids(1, seed=77), causal_mask()
        with sp_overlap("off"):
            reference_net = model(TPStrategy.TENSOR_SEQUENCE)
            references = [forward_backward_model(reference_net, ids, mask) for _ in range(3)]
        with sp_overlap("backward"):
            net = model(TPStrategy.TENSOR_SEQUENCE)
            for step, reference in enumerate(references):
                assert_bitwise_equal(forward_backward_model(net, ids, mask), reference, f"{MESH_KEY} step {step}")


@pytest.mark.requires_device
@pytest.mark.usefixtures("overlap_mesh")
class TestMatchesTensorParallel:
    def test_gradients_after_sync(self):
        """The oracle of test_sequence_parallel.py under the overlap: after ttml.sync_gradients (the tensor-parallel
        all-reduces of the replicated grads, on the compute queue behind the drained weight gradients) the SP model's
        synced gradients are bitwise those of the same model on one queue, and both match the TP model within the
        oracle's limit (the split grid changes the matmul block configs, so that part is ULP, not bitwise)."""
        ids, mask = token_ids(1, seed=77), causal_mask()
        tp = forward_backward_model(model(TPStrategy.TENSOR, attention_bias=True), ids, mask)

        def synced(overlap: str) -> dict[str, np.ndarray]:
            net = model(TPStrategy.TENSOR_SEQUENCE, attention_bias=True)
            with sp_overlap(overlap):
                result = forward_backward_model(net, ids, mask)
            ttml.sync_gradients(net.parameters())
            for name, param in net.parameters().items():
                if param.get_requires_grad():
                    result[f"grad {name}"] = per_rank(param.get_grad_tensor())
            return result

        one_queue, two_queues = synced("off"), synced("backward")
        assert_bitwise_equal(two_queues, one_queue, f"{MESH_KEY} synced grads")
        assert tp["logits"].std() > 1e-3
        assert tp.keys() == two_queues.keys()
        for name, reference in tp.items():
            assert_within_bf16_ulp(two_queues[name], reference, f"{name} vs tensor parallel", MAX_ULP)


@pytest.mark.requires_device
@pytest.mark.usefixtures("overlap_mesh")
class TestSwitch:
    def test_round_trips(self):
        get, set_ = ttml.ops.distributed.get_sp_overlap, ttml.ops.distributed.set_sp_overlap
        SPOverlapMode = ttml.ops.distributed.SPOverlapMode
        assert get() == SPOverlapMode.OFF
        set_("backward")
        assert get() == SPOverlapMode.BACKWARD
        set_(SPOverlapMode.OFF)
        assert get() == SPOverlapMode.OFF
        assert ttml.ops.distributed.sp_overlap_stats()[0] == 0

    def test_rejects_unknown_name(self, expect_error):
        with expect_error(ValueError, "'off' or 'backward'"):
            ttml.ops.distributed.set_sp_overlap("forward")
        assert ttml.ops.distributed.get_sp_overlap() == ttml.ops.distributed.SPOverlapMode.OFF

    def test_backward_impl_follows_the_forward_unless_overridden(self, expect_error):
        d = ttml.ops.distributed
        FUSED, COMPOSED = d.SPLinearImpl.FUSED, d.SPLinearImpl.COMPOSED
        d.set_sp_linear_impl("fused")
        assert (d.get_sp_linear_impl(), d.get_sp_linear_backward_impl()) == (FUSED, FUSED)
        d.set_sp_linear_backward_impl("composed")
        assert (d.get_sp_linear_impl(), d.get_sp_linear_backward_impl()) == (FUSED, COMPOSED)
        d.set_sp_linear_backward_impl("same")
        assert d.get_sp_linear_backward_impl() == FUSED
        d.set_sp_linear_backward_impl(COMPOSED)
        d.set_sp_linear_impl("composed")  # selecting both sites drops the override
        d.set_sp_linear_impl("fused")
        assert d.get_sp_linear_backward_impl() == FUSED
        d.set_sp_linear_backward_impl(None)
        with expect_error(ValueError, "'same', 'composed' or 'fused'"):
            d.set_sp_linear_backward_impl("eager")


class TestDeviceConfig:
    def test_knobs(self, expect_error):
        from ttml.common.config import DeviceConfig

        cfg = DeviceConfig({"device_config": {"enable_tp": True, "enable_sp": True}})
        assert (cfg.sp_overlap, cfg.sp_ccl_rows, cfg.sp_ccl_columns) == ("off", 1, 0)
        assert (cfg.sp_linear_impl, cfg.sp_linear_backward_impl) == ("fused", "same")
        cfg = DeviceConfig({"device_config": {"sp_linear_impl": "fused", "sp_linear_backward_impl": "composed"}})
        assert (cfg.sp_linear_impl, cfg.sp_linear_backward_impl) == ("fused", "composed")
        with expect_error(ValueError, "sp_linear_backward_impl"):
            DeviceConfig({"device_config": {"sp_linear_backward_impl": "eager"}})
        cfg = DeviceConfig({"device_config": {"sp_overlap": "backward", "sp_ccl_rows": 0, "sp_ccl_columns": 2}})
        assert (cfg.sp_overlap, cfg.sp_ccl_rows, cfg.sp_ccl_columns) == ("backward", 0, 2)
        assert DeviceConfig({"device_config": {"sp_overlap": False}}).sp_overlap == "off"  # YAML's bare `off`
        assert DeviceConfig({"device_config": {"sp_overlap": True}}).sp_overlap == "backward"
        with expect_error(ValueError, "'off', 'split' or 'backward'"):
            DeviceConfig({"device_config": {"sp_overlap": "forward"}})
        with expect_error(ValueError, "exactly one of"):
            DeviceConfig({"device_config": {"sp_overlap": "backward", "sp_ccl_rows": 1, "sp_ccl_columns": 1}})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
