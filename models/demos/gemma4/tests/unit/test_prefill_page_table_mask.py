# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill zeroes page-table columns past each request's allocation: vLLM leaves stale block ids there,
and the traced single-chunk prefill cannot cap its K/V fill on the host, so pad rows would write into them."""

import pytest
import torch

from models.demos.gemma4.tt.generator import mask_page_table_columns_past_allocation


def _table(rows, cols, base=1):
    return (torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols) + base).clone()


def test_columns_past_allocation_are_zeroed_per_row():
    pt = _table(2, 20)
    orig = pt.clone()
    (out,), masked = mask_page_table_columns_past_allocation([pt], prompt_lens=[100, 1000], block_sizes=[64])
    # 100 tokens -> 2 blocks; 1000 tokens -> 16 blocks.
    assert torch.equal(out[0, :2], orig[0, :2]) and int(out[0, 2:].abs().sum()) == 0
    assert torch.equal(out[1, :16], orig[1, :16]) and int(out[1, 16:].abs().sum()) == 0
    assert masked == 18 + 4
    # The plugin's tensor is left untouched.
    assert torch.equal(pt, orig)


def test_block_size_is_per_layer():
    pt_a = _table(1, 12)
    pt_b = _table(1, 12, base=100)
    (a, b), _ = mask_page_table_columns_past_allocation([pt_a, pt_b], prompt_lens=[256], block_sizes=[64, 128])
    assert int((a[0] != 0).sum()) == 4  # 256 / 64
    assert int((b[0] != 0).sum()) == 2  # 256 / 128


def test_shared_table_object_stays_shared_and_one_dim_rows_work():
    shared = _table(1, 8)[0]  # 1-D row, as the kv-share alias passes it
    (a, b), masked = mask_page_table_columns_past_allocation([shared, shared], prompt_lens=[65], block_sizes=[64, 64])
    assert a is b
    assert a.dim() == 1 and int((a != 0).sum()) == 2
    assert masked == 6  # counted once for the shared object


def test_rows_without_a_prompt_length_and_clean_tables_pass_through():
    pt = _table(3, 6)
    pt[0, 2:] = 0  # already clean
    (out,), masked = mask_page_table_columns_past_allocation([pt], prompt_lens=[100, 100], block_sizes=[64])
    assert torch.equal(out[2], pt[2])  # no prompt length -> untouched
    assert int(out[1, 2:].abs().sum()) == 0 and masked == 4
    # Nothing to mask -> the very same object comes back.
    clean = pt.clone()
    clean[:, 2:] = 0
    (same,), masked = mask_page_table_columns_past_allocation([clean], prompt_lens=[100, 100, 100], block_sizes=[64])
    assert same is clean and masked == 0


def test_non_tensor_entries_and_missing_inputs_are_left_alone():
    pt = _table(1, 4)
    out, masked = mask_page_table_columns_past_allocation(
        [None, "device-tensor", pt], prompt_lens=[1], block_sizes=[64, 64, 64]
    )
    assert out[0] is None and out[1] == "device-tensor" and int((out[2] != 0).sum()) == 1 and masked == 3
    assert mask_page_table_columns_past_allocation([pt], prompt_lens=None, block_sizes=[64]) == ([pt], 0)
    assert mask_page_table_columns_past_allocation(None, prompt_lens=[1], block_sizes=[64]) == (None, 0)


def test_block_sizes_resolve_from_the_plugins_nested_kv_cache_list():
    """The plugin passes ``kv_caches`` as [submesh][layer][k, v]; both nestings must resolve block sizes."""
    from types import SimpleNamespace

    pytest.importorskip("vllm")
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    class _Cache:
        shape = [100, 2, 64, 256]

    cfg = SimpleNamespace(head_dim=256, num_key_value_heads=16)
    layers = [SimpleNamespace(self_attn=SimpleNamespace(config=cfg, mesh_config=None, weights=None)) for _ in range(3)]
    fake = SimpleNamespace(model=[SimpleNamespace(layers=layers)])
    per_layer = [[_Cache(), _Cache()] for _ in layers]
    assert Gemma4ForCausalLM._prefill_page_table_block_sizes(fake, per_layer) == [64, 64, 64]
    assert Gemma4ForCausalLM._prefill_page_table_block_sizes(fake, [per_layer]) == [64, 64, 64]
