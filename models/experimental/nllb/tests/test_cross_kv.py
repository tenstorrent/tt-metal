# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""CPU structural checks for request-local cache; fixed TT suites validate numerics."""

import unittest
from types import SimpleNamespace, MethodType
from unittest.mock import patch
import numpy as np
import torch

from models.experimental.nllb.tt import backend


class CrossKVTests(unittest.TestCase):
    def test_projection_reuse_and_request_lifetime(self):
        b = backend.Backend.__new__(backend.Backend)
        b.dim = 4
        b.kernel = None
        b.pad = 1
        b.vocab = 16
        b.config = {"vocab_size": 16, "max_position_embeddings": 128}
        calls = []

        def linear(x, name):
            calls.append(name)
            return x * (0.75 if name.endswith("k_proj") else 0.5 if name.endswith("v_proj") else 1.0)

        b.linear = linear
        ops = SimpleNamespace(
            reshape=torch.reshape,
            permute=torch.permute,
            multiply=torch.mul,
            add=torch.add,
            matmul=lambda a, b, transpose_b=False, **kw: a @ (b.transpose(-1, -2) if transpose_b else b),
            softmax=lambda x, dim, **kw: torch.softmax(x, dim=dim),
        )
        torch.manual_seed(13)
        x = torch.randn(1, 1, 3, 4)
        m = torch.randn(1, 1, 5, 4)
        mask = torch.zeros(1, 1, 3, 5)
        with patch.object(backend, "ttnn", ops):
            cache = {}
            a = b.attention(x, m, "layer.encoder_attn", 2, mask, cross_kv=cache)
            calls.clear()
            a2 = b.attention(x, m, "layer.encoder_attn", 2, mask, cross_kv=cache)
            assert not any(n.endswith(("k_proj", "v_proj")) for n in calls)
            uncached = b.attention(x, m, "layer.encoder_attn", 2, mask)
            assert torch.equal(a, a2) and torch.equal(a, uncached)
            changed = b.attention(x, m + 1, "layer.encoder_attn", 2, mask, cross_kv={})
            assert not torch.equal(a, changed)
            b.attention(x, m, "layer2.encoder_attn", 2, mask, cross_kv=cache)
            assert len(cache) == 2
            calls.clear()
            for _ in range(2):
                b.attention(x, x, "layer.self_attn", 2, torch.zeros(1, 1, 3, 3))
            assert sum(n.endswith(("k_proj", "v_proj")) for n in calls) == 4
        # Structural generation lifetime check: fresh dictionary for each row/request,
        # revisit, cap=1, EOS and exception cleanup. No learned CPU model computation.
        seen = []
        visits = []
        fail = False
        eos = False

        def encode(self, ids, mask):
            return int(ids[0, 0]), mask[0]

        def decode(self, ids, enc, valid, *, final_token_only=False, cross_kv=None):
            assert cross_kv is not None
            if not cross_kv:
                assert all(cross_kv is not old for old in seen)
                seen.append(cross_kv)
                cross_kv["owner"] = enc
            assert cross_kv["owner"] == enc
            visits.append(enc)
            if fail:
                raise RuntimeError("injected failure")
            logits = np.zeros((1, 1, 16))
            logits[0, 0, 2 if eos else enc] = 1
            return logits

        b.encode = MethodType(encode, b)
        b.decode = MethodType(decode, b)
        ids = np.array([[4, 5], [7, 8]])
        mask = np.ones_like(ids)
        a = b.generate(ids, mask, 9, 4)
        assert len(seen) == 2 and all(not c for c in seen)
        assert np.array_equal(a, b.generate(ids, mask, 9, 4))
        assert np.array_equal(a[::-1], b.generate(ids[::-1], mask[::-1], 9, 4))
        old = len(seen)
        b.generate(ids, mask, 9, 1)
        assert len(seen) == old
        eos = True
        assert b.generate(ids[:1], mask[:1], 9, 4).tolist() == [[2, 9, 2]]
        fail = True
        try:
            b.generate(ids[:1], mask[:1], 9, 4)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected injected failure")
        assert all(not c for c in seen)
        assert "cross_kv" not in b.__dict__


if __name__ == "__main__":
    unittest.main()
