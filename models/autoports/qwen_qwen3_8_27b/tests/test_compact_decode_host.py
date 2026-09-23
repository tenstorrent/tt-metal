# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host shape checks for the compact multi-request decode residual path."""

import unittest
from types import MethodType, SimpleNamespace

import torch
import torch.nn.functional as F

from models.autoports.qwen_qwen3_8_27b.tests.test_serving_prefill_trace_host import load_methods


class CompactDecodeHostTests(unittest.TestCase):
    def test_embedding_skips_public_batch_expansion(self):
        ops = SimpleNamespace(
            TILE_LAYOUT="tile",
            DRAM_MEMORY_CONFIG="dram",
            Topology=SimpleNamespace(Ring="ring"),
            embedding=lambda tokens, *_args, **_kwargs: tokens[..., None].expand(-1, -1, 2).float(),
            reshape=torch.reshape,
            experimental=SimpleNamespace(all_gather_async=lambda tensor, **_kwargs: torch.cat([tensor] * 4, dim=3)),
        )
        embed = load_methods("model.py", "QwenModel", ["embed"], ops)["embed"]
        model = SimpleNamespace(
            embedding_weight=object(),
            config=SimpleNamespace(hidden_size=8),
            mesh=object(),
            ccl=SimpleNamespace(
                get_and_cycle_ag_semaphore_handles=lambda _count: object(),
                get_and_cycle_barrier_semaphore_handle=lambda _count: object(),
            ),
        )
        tokens = torch.arange(8).reshape(1, 8)
        public = MethodType(embed, model)(tokens, batch=8, length=1)
        compact = MethodType(embed, model)(tokens, batch=8, length=1, compact=True)
        self.assertEqual(public.shape, torch.Size([8, 1, 8]))
        self.assertEqual(compact.shape, torch.Size([1, 1, 8, 8]))
        self.assertTrue(torch.equal(public, compact.reshape(8, 1, 8)))
        with self.assertRaisesRegex(ValueError, "single-token decode"):
            MethodType(embed, model)(tokens, batch=4, length=2, compact=True)

    def test_decode_requests_compact_embedding(self):
        ops = SimpleNamespace(
            uint32=torch.int32,
            reshape=torch.reshape,
            typecast=lambda tensor, _dtype: tensor,
        )
        decode = load_methods("model.py", "QwenModel", ["decode"], ops)["decode"]
        calls = []

        def embed(_tokens, **kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 1, 8, 16)

        layer = SimpleNamespace(
            kind="full_attention",
            decode_forward=lambda x, **_kwargs: x,
        )
        model = SimpleNamespace(
            compact_decode_residual=True,
            embed=embed,
            rope=lambda *_args, **_kwargs: (object(), object()),
            layers=[layer],
            logits=lambda hidden, **_kwargs: hidden,
        )
        result = MethodType(decode, model)(
            torch.arange(32).reshape(1, 32),
            torch.arange(8, dtype=torch.int32),
            cache=SimpleNamespace(batch_size=8, layers=[object()]),
            page_table=object(),
        )
        self.assertEqual(calls, [{"batch": 8, "length": 1, "compact": True}])
        self.assertEqual(result.shape, torch.Size([1, 1, 8, 16]))

    def test_single_request_decode_keeps_public_contract(self):
        ops = SimpleNamespace(
            uint32=torch.int32,
            reshape=torch.reshape,
            typecast=lambda tensor, _dtype: tensor,
        )
        decode = load_methods("model.py", "QwenModel", ["decode"], ops)["decode"]
        calls = []

        def embed(_tokens, **kwargs):
            calls.append(kwargs)
            return torch.zeros(1, 1, 16)

        layer = SimpleNamespace(kind="full_attention", decode_forward=lambda x, **_kwargs: x)
        model = SimpleNamespace(
            compact_decode_residual=True,
            embed=embed,
            rope=lambda *_args, **_kwargs: (object(), object()),
            layers=[layer],
            logits=lambda hidden, **_kwargs: hidden,
        )
        result = MethodType(decode, model)(
            torch.arange(32).reshape(1, 32),
            torch.zeros(1, dtype=torch.int32),
            cache=SimpleNamespace(batch_size=1, layers=[object()]),
            page_table=object(),
        )
        self.assertEqual(calls, [{"batch": 1, "length": 1, "compact": False}])
        self.assertEqual(result.shape, torch.Size([1, 1, 16]))

    def test_layer_boundary_stays_compact(self):
        def mul(a, b, *, input_tensor_a_activations=(), **_kwargs):
            if input_tensor_a_activations:
                a = F.silu(a)
            return a * b

        ops = SimpleNamespace(
            bfloat16=torch.bfloat16,
            L1_MEMORY_CONFIG="l1",
            UnaryOpType=SimpleNamespace(SILU="silu"),
            reshape=torch.reshape,
            to_memory_config=lambda tensor, _memory: tensor,
            add=lambda a, b, **_kwargs: a + b,
            mul=mul,
        )
        finish = load_methods("optimized_decoder.py", "OptimizedDecoder", ["_finish"], ops)["_finish"]
        batch, hidden = 8, 5120

        def linear(tensor, name, **_kwargs):
            if name == "mlp.gate_up":
                rows = tensor if _kwargs.get("keep_sharded", False) else tensor.reshape(batch, 1, hidden)
                return torch.cat([rows, rows], dim=-1)
            if name == "mlp.down_proj":
                return tensor.reshape(1, 1, batch, hidden)
            raise AssertionError(name)

        layer = SimpleNamespace(
            policy={
                "carry_residual": True,
                "compact_decode_residual": True,
                "dram": True,
                "packed_mlp": True,
            },
            config=SimpleNamespace(intermediate_size=hidden),
            weights={},
            projection_configs={},
            _residual_memory=lambda _batch: "residual",
            _norm=lambda tensor, _name: tensor,
            _linear=linear,
            _public_rows=lambda tensor, b, width: tensor.reshape(b, 1, width),
        )
        attention = torch.zeros(1, 1, batch, hidden)
        compact = MethodType(finish, layer)(torch.ones(1, 1, batch, hidden), attention)
        public = MethodType(finish, layer)(torch.ones(batch, 1, hidden), attention)
        self.assertEqual(compact.shape, torch.Size([1, 1, batch, hidden]))
        self.assertEqual(public.shape, torch.Size([batch, 1, hidden]))
        layer.policy["compact_decode_mlp"] = True
        compact_mlp = MethodType(finish, layer)(torch.ones(1, 1, batch, hidden), attention)
        self.assertTrue(torch.equal(compact_mlp, compact))


if __name__ == "__main__":
    unittest.main()
