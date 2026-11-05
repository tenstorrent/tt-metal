# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""test_sync_gradients_placements.py on a ``[2, 2]`` mesh: DP x TP and HSDP, where a parameter is replicated on one
data axis and sharded on the other. Each class opens its ``[2, 2]`` through the conftest's ``fresh_device_mesh`` with
the bundled descriptor, like ``test_ccl_topology.py``, and skips on a host with fewer than four chips."""

from __future__ import annotations

import numpy as np
import pytest

import ttnn
import ttml
from ttml.modules import LinearLayer
from sync_gradients_placements_common import (
    DIM,
    NATIVE,
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
    is_sharded_on_axis,
)
from tp_block import TPBlock
from two_device_gradients import exact_mean, make_gradients, to_bf16

pytestmark = [
    pytest.mark.requires_device,
    pytest.mark.timeout(1800),
    pytest.mark.usefixtures("strict_ccl_topology"),
]


@pytest.fixture(scope="module")
def hsdp_mesh(fresh_device_mesh):
    """``[2, 2]`` with axes ``("dp", "fsdp")``."""
    with fresh_device_mesh((2, 2), ("dp", "fsdp"), what="HSDP placements tests", require_mgd=True) as mesh:
        yield mesh


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

    def test_fp32_reduce_rounds_an_hsdp_gradient_once(self, hsdp_mesh):
        """With reduce_dtype float32 the fsdp reduce-scatter leaves an fp32 shard, so the dp sum adds unrounded means and
        the gradient is rounded at most once, at the end. Rounding the shard to bf16 in between, as the shard used to
        be written, rounds it twice and lands on the exact mean less often: about 98.5% of the elements against 76%
        here."""
        dp = hsdp_mesh.axis_index("dp")
        D, F = hsdp_mesh.axis_size("dp"), hsdp_mesh.axis_size("fsdp")
        values = make_gradients(64, 128, seed=8, devices=D * F)  # device k in mesh row-major order holds values[k]
        mapper = ttnn.create_mesh_mapper(
            _device(), ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)])
        )
        exact = values.astype(np.float64).mean(axis=0, keepdims=True)

        def reduce(round_shard_to_bf16: bool):
            layer = LinearLayer(128, 64, has_bias=False)
            ttml.fsdp.fully_shard(layer, reduce_dtype=ttnn.DataType.FLOAT32)
            layer.unshard()
            ((parameter, shard_dim),) = layer._fsdp_state.managed
            tensor = parameter.tensor
            gradients = ttml.autograd.Tensor.from_numpy(
                values.reshape(D, F, *values.shape[2:]), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper
            )
            tensor.set_grad(gradients.get_value())
            layer._fsdp_state.backward_post()
            assert tensor.get_grad().dtype == ttnn.DataType.FLOAT32
            if round_shard_to_bf16:
                tensor.set_grad(ttnn.typecast(tensor.get_grad(), ttnn.DataType.BFLOAT16))
            parameters = ttml.NamedParameters()
            parameters["weight"] = tensor
            ttml.core.distributed.synchronize_gradients(parameters, [dp], reduce_dtype=ttnn.DataType.FLOAT32)
            grad = tensor.get_grad()
            shards = np.stack(_shards(grad)).reshape(D, F, *_shards(grad)[0].shape)
            for row in range(1, D):
                np.testing.assert_array_equal(shards[row], shards[0])  # every dp rank holds the same mean
            return grad.dtype, np.concatenate(list(shards[0]), axis=shard_dim)

        once_dtype, once = reduce(round_shard_to_bf16=False)
        twice_dtype, twice = reduce(round_shard_to_bf16=True)

        assert once_dtype == ttnn.DataType.FLOAT32
        assert twice_dtype == ttnn.DataType.BFLOAT16
        once_share = np.mean(to_bf16(once) == exact_mean(values))
        twice_share = np.mean(twice == exact_mean(values))
        assert once_share >= 0.97
        assert once_share > twice_share
        assert np.mean(np.abs(once - exact)) < np.mean(np.abs(twice - exact))

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
