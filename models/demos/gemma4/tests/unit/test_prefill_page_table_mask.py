# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill zeroes page-table columns past each request's allocation: vLLM leaves stale block ids there,
and the traced single-chunk prefill cannot cap its K/V fill on the host, so pad rows would write into them."""

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt.generator import (
    layer_local_kv_heads,
    mask_page_table_columns_past_allocation,
    paged_block_size_for_view,
)
from models.tt_transformers.tt.common import num_blocks_in_seq


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
    pytest.importorskip("vllm")
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    class _Cache:
        shape = [100, 2, 64, 256]

    cfg = SimpleNamespace(head_dim=256, num_key_value_heads=2)  # two heads per chip, as the cache holds
    layers = [SimpleNamespace(self_attn=SimpleNamespace(config=cfg, mesh_config=None, weights=None)) for _ in range(3)]
    fake = SimpleNamespace(model=[SimpleNamespace(layers=layers)])
    per_layer = [[_Cache(), _Cache()] for _ in layers]
    assert Gemma4ForCausalLM._prefill_page_table_block_sizes(fake, per_layer) == [64, 64, 64]
    assert Gemma4ForCausalLM._prefill_page_table_block_sizes(fake, [per_layer]) == [64, 64, 64]


# 26B-A4B geometry: sliding layers 8 kv heads x 256, full-attention layers 2 kv heads x 512; the lane engine runs its
# TP over the mesh rows (2 on the (2,4) block, 8 on the (8,4) Galaxy) while the plain TP axis is the 4 columns.
_LAYERS = ((8, 256), (2, 512))


def _lane_views(mesh_shape, block_size, cache_tp):
    """Per layer (cache shape, head_dim, attention kv heads per chip) when the cache is allocated at ``cache_tp``."""
    attn_tp = mesh_shape[0]
    out = []
    for kv, hd in _LAYERS:
        attn = SimpleNamespace(
            mesh_config=SimpleNamespace(tp=attn_tp), weights=SimpleNamespace(kv_replicated=kv < attn_tp)
        )
        cfg = SimpleNamespace(head_dim=hd, num_key_value_heads=kv)
        cache_heads = 1 if kv < cache_tp else kv // cache_tp
        out.append(([256, cache_heads, block_size, hd], hd, layer_local_kv_heads(attn, cfg)))
    return out


@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("mesh_shape", [(2, 4), (8, 4)], ids=["2x4", "8x4"])
def test_lane_engine_cache_from_its_tp_keeps_declared_block_size(block_size, mesh_shape):
    """Caches allocated with the attention's own TP (init_kv_cache from mesh_config.tp) are addressed at the
    declared block size on every layer, so a 128-token prompt needs exactly 128 / block_size columns."""
    sizes = [
        paged_block_size_for_view(shape, hd, kv_local)
        for shape, hd, kv_local in _lane_views(mesh_shape, block_size, mesh_shape[0])
    ]
    assert sizes == [block_size, block_size]
    cols = num_blocks_in_seq(128, min(sizes))
    assert cols == 128 // block_size
    pt = _table(1, 8)
    (out,), _ = mask_page_table_columns_past_allocation([pt], prompt_lens=[128], block_sizes=[min(sizes)])
    assert int((out[0] != 0).sum()) == cols


@pytest.mark.parametrize("block_size", [32, 64])
@pytest.mark.parametrize("mesh_shape, sliding_factor", [((2, 4), 0.5), ((8, 4), 2.0)], ids=["2x4_half", "8x4_double"])
def test_lane_engine_cache_from_the_mesh_columns_shifts_the_effective_block_size(
    block_size, mesh_shape, sliding_factor
):
    """The former allocation (heads from the 4 mesh columns) holds 2 sliding heads per chip on both shapes while the
    attention writes 4 on the (2,4) and 1 on the (8,4): the sliding layers' effective block size is half the declared
    one on the block (a 128-token prompt overran its 2 columns) and double it on the Galaxy (benign: wider is fine).
    The page-table math must follow the effective size on every layer."""
    views = _lane_views(mesh_shape, block_size, cache_tp=mesh_shape[1])
    sliding, full = (paged_block_size_for_view(shape, hd, kv_local) for shape, hd, kv_local in views)
    assert sliding == int(block_size * sliding_factor)
    assert full == block_size  # 2 full heads over 4 columns and over the rows both give one head per chip
    eff = min(sliding, full)
    cols = num_blocks_in_seq(128, eff)
    assert cols == -(-128 // eff)
    if sliding_factor < 1:
        assert cols == 2 * (128 // block_size), "the block needs twice the columns the declared size suggests"
    pt = _table(1, 16)
    (out,), _ = mask_page_table_columns_past_allocation([pt], prompt_lens=[128], block_sizes=[eff])
    assert int((out[0] != 0).sum()) == cols


def test_paged_block_size_for_view_rejects_an_unaddressable_cache(expect_error):
    with expect_error(ValueError, "not addressable"):
        paged_block_size_for_view([4, 1, 64, 256], head_dim=512, num_local_kv_heads=3)
