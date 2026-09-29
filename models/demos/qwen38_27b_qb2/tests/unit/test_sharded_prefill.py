# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host checks for prefill-only layout scope; hardware numerics tested separately."""

import unittest
from types import MethodType, SimpleNamespace

import torch

from models.demos.qwen38_27b_qb2.tests.unit.test_serving_prefill_trace import load_methods


class ShardedPrefillScopeTests(unittest.TestCase):
    def test_grouped_prefill_preserves_inactive_slots_and_last_token_order(self):
        ops = SimpleNamespace(
            uint32="u32",
            ROW_MAJOR_LAYOUT="rm",
            reshape=torch.reshape,
            concat=lambda xs, dim: torch.cat(xs, dim=dim),
            copy=lambda src, dst: dst.copy_(src),
        )
        methods = load_methods("model.py", "Qwen38Model", ["prefill_batch"], ops)
        methods["DecoderState"] = SimpleNamespace
        initial = torch.arange(4).reshape(4, 1).float()
        state = SimpleNamespace(conv=initial.clone(), recurrent=initial.clone())
        seen = []

        def forward(x, **kwargs):
            local = kwargs["state"]
            seen.append(kwargs["page_table"].clone())
            local.conv = local.conv + 10
            local.recurrent = local.recurrent + 20
            return x + 1

        layer = SimpleNamespace(
            kind="linear_attention", prefill_sharded_forward=forward, _gather=lambda x: torch.cat([x] * 4, dim=-1)
        )
        model = SimpleNamespace(
            prefill_sharded_residual=True,
            config=SimpleNamespace(hidden_size=5120),
            layers=[layer],
            embed=lambda ids, **kw: ids[:, :, None].expand(-1, -1, 1280).float(),
            upload=lambda x, **kw: x,
            rope=lambda *a, **kw: (None, None),
            logits=lambda x, **kw: x.clone(),
        )
        tokens = torch.tensor([[11, 12], [21, 22]])
        table = torch.arange(8).reshape(4, 2)
        outputs = MethodType(methods["prefill_batch"], model)(
            tokens,
            cache=SimpleNamespace(batch_size=4, capacity=64, layers=[state]),
            page_table=table,
            length=2,
            start_pos=0,
            slots=[1, 2],
        )
        self.assertTrue(torch.equal(seen[0], table[1:3]))
        self.assertEqual([x.shape for x in outputs], [torch.Size([1, 1, 5120])] * 2)
        self.assertEqual([x[0, 0, 0].item() for x in outputs], [13, 23])
        self.assertEqual(state.conv.flatten().tolist(), [0, 11, 12, 3])
        self.assertEqual(state.recurrent.flatten().tolist(), [0, 21, 22, 3])
        # Intermediate chunks must still update state but never execute the head.
        model.logits = lambda *a, **kw: self.fail("Intermediate chunk computed unused logits")
        result = MethodType(methods["prefill_batch"], model)(
            tokens,
            cache=SimpleNamespace(batch_size=4, capacity=64, layers=[state]),
            page_table=table,
            length=2,
            start_pos=32,
            slots=[1, 2],
            return_logits=False,
        )
        self.assertEqual(result, [])
        self.assertEqual(state.conv.flatten().tolist(), [0, 21, 22, 3])

    def test_decode_contract_restored_on_success_and_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                names = ("input_layernorm", "post_attention_layernorm")
                weights = {name + suffix: object() for name in names for suffix in (".weight", ".prefill_weight")}
                saved = dict(weights)
                layer = SimpleNamespace(weights=weights, policy={"carry_residual": True}, sharded_residual=False)

                def forward(x, **kwargs):
                    self.assertTrue(layer.sharded_residual)
                    self.assertFalse(layer.policy["carry_residual"])
                    for name in names:
                        self.assertIs(layer.weights[name + ".weight"], saved[name + ".prefill_weight"])
                    self.assertEqual(kwargs, {"start_pos": 4096})
                    if fail:
                        raise RuntimeError("injected failure")
                    return x

                layer.prefill_forward = forward
                method = load_methods("decoder_tp.py", "Qwen38TPDecoder", ["prefill_sharded_forward"], None)
                call = MethodType(method["prefill_sharded_forward"], layer)
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "injected failure"):
                        call("input", start_pos=4096)
                else:
                    self.assertEqual(call("input", start_pos=4096), "input")
                self.assertFalse(layer.sharded_residual)
                self.assertTrue(layer.policy["carry_residual"])
                for name, value in saved.items():
                    self.assertIs(layer.weights[name], value)


if __name__ == "__main__":
    unittest.main()
