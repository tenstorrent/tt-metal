# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import ttnn

import ttml


@pytest.mark.requires_device
@pytest.mark.parametrize("term_order", ("qkv", "qvk", "kqv", "kvq", "vqk", "vkq"))
def test_cpp_grouped_heads_creation_backward_is_independent_of_loss_order(term_order):
    """The shared Q callback must run after both K/V output gradients are produced."""
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.reset_graph()
    try:
        batch_size = 1
        seq_len = 32
        num_heads = 4
        num_groups = 2
        head_dim = 32

        qs = ttml.autograd.Tensor.from_numpy(
            np.zeros((batch_size, 1, seq_len, num_heads * head_dim), dtype=np.float32),
            new_type=ttnn.DataType.BFLOAT16,
        )
        kvs = ttml.autograd.Tensor.from_numpy(
            np.zeros((batch_size, 1, seq_len, 2 * num_groups * head_dim), dtype=np.float32),
            new_type=ttnn.DataType.BFLOAT16,
        )
        qs.set_requires_grad(True)
        kvs.set_requires_grad(True)

        q, k, v = ttml.ops.multi_head_utils.grouped_heads_creation(qs, kvs, num_heads, num_groups)
        for output in (q, k, v):
            output.set_grad(ttml.core.zeros_like(output.get_value()))

        terms = {
            "q": ttml.ops.unary.mean(q) * 4096.0,
            "k": ttml.ops.unary.mean(k) * 4096.0,
            "v": ttml.ops.unary.mean(v) * 8192.0,
        }
        loss = terms[term_order[0]] + terms[term_order[1]] + terms[term_order[2]]
        loss.backward(retain_graph=False)

        for name, output, expected in (("q", q, 1.0), ("k", k, 2.0), ("v", v, 4.0)):
            np.testing.assert_allclose(
                output.get_grad_tensor().to_numpy(),
                expected,
                rtol=0.0,
                atol=1.0e-3,
                err_msg=f"{name} gradient is wrong for loss order {term_order}",
            )

        np.testing.assert_allclose(
            qs.get_grad_tensor().to_numpy(),
            1.0,
            rtol=0.0,
            atol=1.0e-3,
            err_msg=f"qs gradient is wrong for loss order {term_order}",
        )
        kvs_grad = kvs.get_grad_tensor().to_numpy()
        np.testing.assert_allclose(
            kvs_grad[..., : num_groups * head_dim],
            2.0,
            rtol=0.0,
            atol=1.0e-3,
            err_msg=f"kvs K slice is wrong for loss order {term_order}",
        )
        np.testing.assert_allclose(
            kvs_grad[..., num_groups * head_dim :],
            4.0,
            rtol=0.0,
            atol=1.0e-3,
            err_msg=f"kvs V slice is wrong for loss order {term_order}",
        )
    finally:
        ctx.reset_graph()
