# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verify row redistribution, eligibility and weight restoration on the host."""

import unittest
from types import MethodType, SimpleNamespace

import torch

from models.demos.qwen38_27b_qb2.tests.unit.test_serving_prefill_trace import load_methods


class Tensor:
    def __init__(self, data):
        self.data = data
        self.shape = data.shape[1:]


class RowParallelNormTests(unittest.TestCase):
    def check_case(self, batch, length, enabled=True, fail=False):
        events = []
        local_weight, full_weight = object(), object()
        name = "input_layernorm"
        layer = SimpleNamespace(
            TP=4,
            sharded_residual=True,
            policy={"prefill_replicated_norm": True, "prefill_row_parallel_norm": enabled, "num_links": 2},
            weights={name + ".weight": local_weight, name + ".replicated_weight": full_weight},
            device=object(),
            topology=object(),
            ccl=SimpleNamespace(
                get_and_cycle_ag_semaphore_handles=lambda axis: None,
                get_and_cycle_barrier_semaphore_handle=lambda axis: None,
            ),
        )

        def reshape(x, shape):
            return Tensor(x.data.reshape(4, *shape))

        def gather(x, dim):
            return Tensor(torch.cat(list(x.data), dim=dim).unsqueeze(0).repeat(4, *([1] * len(x.shape))))

        def exchange(x, **kwargs):
            self.assertEqual((kwargs["in_dim"], kwargs["out_dim"]), (3, 2))
            events.append("exchange")
            full = torch.cat(list(x.data), dim=3)
            return Tensor(torch.stack(full.chunk(4, dim=2)))

        def norm(x, norm_name):
            self.assertIs(layer.weights[norm_name + ".weight"], full_weight)
            if fail:
                raise RuntimeError("injected normalization failure")
            return Tensor(x.data * torch.rsqrt(x.data.square().mean(dim=-1, keepdim=True) + 1e-6))

        ops = SimpleNamespace(
            reshape=reshape,
            DRAM_MEMORY_CONFIG=object(),
            experimental=SimpleNamespace(
                all_to_all_async_generic=exchange,
                all_gather_async=lambda x, dim, **kw: gather(x, dim),
            ),
        )
        layer._gather = lambda x: gather(x, len(x.shape) - 1)
        methods = load_methods("decoder_tp.py", "Qwen38TPDecoder", ["_norm"], ops)
        methods["super"] = lambda: SimpleNamespace(_norm=norm)
        call = MethodType(methods["_norm"], layer)
        x = Tensor(torch.arange(4 * batch * length * 1280).reshape(4, batch, length, 1280).float() % 137)
        if fail:
            with self.assertRaisesRegex(RuntimeError, "injected normalization failure"):
                call(x, name)
        else:
            result = call(x, name)
            full = torch.cat(list(x.data), dim=-1)
            expected = full * torch.rsqrt(full.square().mean(dim=-1, keepdim=True) + 1e-6)
            for chip in result.data:
                self.assertTrue(torch.equal(chip, expected))
        self.assertIs(layer.weights[name + ".weight"], local_weight)
        eligible = enabled and length > 1 and length % 32 == 0 and batch * length % 128 == 0
        self.assertEqual(events, ["exchange"] if eligible else [])

    def test_row_order_and_nondivisible_batch(self):
        for batch, length in ((16, 32), (7, 128), (15, 128)):
            with self.subTest(batch=batch, length=length):
                self.check_case(batch, length)

    def test_fallback_shapes_and_disabled_flag(self):
        for batch, length, enabled in ((7, 32, True), (8, 33, True), (16, 1, True), (8, 128, False)):
            with self.subTest(batch=batch, length=length, enabled=enabled):
                self.check_case(batch, length, enabled)

    def test_restoration_on_failure(self):
        self.check_case(7, 128, fail=True)
        self.check_case(7, 32, fail=True)


if __name__ == "__main__":
    unittest.main()
