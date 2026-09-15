# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.fill_cache``.

Model call site (modules/attention/attention_1d.py):
  * L883/884  _kv_fill_prefill_non_paged  — ttnn.fill_cache(keys, k_fill, user_id % max_batch)
  * L915/916  _kv_fill_prefill_batched    — ttnn.fill_cache(keys, k_fill[slot:slot+1], slot % max_batch)

``fill_cache(cache, input, batch_idx)`` writes ``input`` into row ``batch_idx``
of the KV cache. Signature/shapes mirror the tt_eager unit test
(tests/tt_eager/.../test_update_cache.py:test_fill_cache):
    cache:  [max_batch, n_kv_heads, max_seq, head_dim]
    input:  [1,         n_kv_heads, seq,     head_dim]
For Llama-3.2-1B: n_kv_heads=8 (KV_DIM/HEAD_DIM), head_dim=64, max_batch=32.

We fill one user slot and assert the written region of the cache equals the
input (PCC 0.999 on the slice).
"""

import pytest

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U


@U.with_default_mesh()
@pytest.mark.parametrize("seq", [pytest.param(s, id=f"seq{s}") for s in (128, 512)])
@pytest.mark.parametrize("batch_idx", [pytest.param(i, id=f"user{i}") for i in (0, 5)])
def test_fill_cache(ttnn_mesh_device, reset_seeds, seq, batch_idx):
    mesh = ttnn_mesh_device
    max_seq = 1024

    cache_shape = (U.MAX_BATCH, U.N_KV_HEADS, max_seq, U.HEAD_DIM)
    fill_shape = (1, U.N_KV_HEADS, seq, U.HEAD_DIM)

    cache_torch = U.torch_rand(cache_shape)
    fill_torch = U.torch_rand(fill_shape)

    cache = U.to_tt(cache_torch, mesh)
    fill = U.to_tt(fill_torch, mesh)

    ttnn.fill_cache(cache, fill, batch_idx)

    # Reference: the cache with row batch_idx's first `seq` positions overwritten by fill.
    # PCC over the whole cache checks both the written region and the untouched rows.
    ref_cache = cache_torch.float().clone()
    ref_cache[batch_idx : batch_idx + 1, :, :seq, :] = fill_torch.float()
    U.assert_pcc(ref_cache, cache, pcc=0.999, mesh_device=mesh)
