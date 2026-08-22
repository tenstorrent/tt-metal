# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""P150x2 hardware qualification for trace-stable resumed prefill.

These tests are deliberately separate from the device-free adapter contracts. They
exercise the actual production decoder and TTNN flexible chunked-SDPA/indexed-RoPE
programs, then forbid program-cache misses while changing only runtime offsets.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests import laguna_reference as R
from models.autoports.poolside_laguna_xs_2_1.tests import test_multichip_decoder as D
from models.autoports.poolside_laguna_xs_2_1.tt.prefill_runtime import PrefillRuntimeOffsets

# Re-export the established module fixtures so these tests share its mesh, weights,
# reference config, and production precision/profile setup.
device = D.device
hf_config = D.hf_config

pytestmark = pytest.mark.skipif(
    D.PROFILE.name != "p150x2",
    reason="prefix-cache runtime-offset qualification targets the served p150x2 profile",
)


def _host_replicated(tensor, mesh, *, dtype, layout):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        mesh_mapper=D._mm(mesh),
    )


def _copy_replicated(tensor, target, mesh, *, dtype, layout):
    source = _host_replicated(tensor, mesh, dtype=dtype, layout=layout)
    ttnn.copy_host_to_device_tensor(source, target)


def _device0(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def _freeze_misses(mesh):
    """Context manager-like helpers kept explicit so misses are always re-enabled."""
    ttnn.synchronize_device(mesh)
    before = int(mesh.num_program_cache_entries())
    mesh.set_program_cache_misses_allowed(False)
    return before


def _unfreeze_misses(mesh, before):
    after = None
    try:
        # Keep the guard active through device completion; a queued operation
        # must not get a chance to compile after the host-side call returns.
        ttnn.synchronize_device(mesh)
        after = int(mesh.num_program_cache_entries())
    finally:
        mesh.set_program_cache_misses_allowed(True)
    assert after == before


def _chunked_reference(q, k, v, *, start_pos, scale):
    """Reference absolute-causal attention for suffix Q against prefix+suffix KV."""
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)
    v = v.repeat_interleave(groups, dim=1)
    visible = start_pos + q.shape[2]
    k = k[:, :, :visible].float()
    v = v[:, :, :visible].float()
    scores = torch.matmul(q.float(), k.transpose(-2, -1)) * float(scale)
    query_positions = start_pos + torch.arange(q.shape[2]).reshape(-1, 1)
    key_positions = torch.arange(visible).reshape(1, -1)
    scores.masked_fill_(key_positions > query_positions, torch.finfo(scores.dtype).min)
    return torch.matmul(torch.softmax(scores, dim=-1), v)


def _assert_attention_close(got, reference, *, label):
    pcc = D._pcc(got, reference)
    rmse = torch.sqrt(torch.mean((got.float() - reference.float()) ** 2)).item()
    ref_rms = torch.sqrt(torch.mean(reference.float() ** 2)).item()
    relative_rmse = rmse / max(ref_rms, 1e-8)
    assert pcc >= 0.99, f"{label}: PCC {pcc:.5f} < 0.99"
    assert relative_rmse <= 0.20, f"{label}: relative RMSE {relative_rmse:.5f} > 0.20"


def test_flexible_chunked_sdpa_changes_offset_without_program_miss(device, hf_config):
    """The production D2 SDPA program must treat the absolute start as runtime data."""
    dec = D._decoder(hf_config, D.FULL_DENSE, device)
    cfg = dec.cfg
    seq, total = 64, 512
    starts = (64, 320)
    generator = torch.Generator().manual_seed(20260822)
    k = (torch.randn(1, cfg.num_kv_heads, total, cfg.head_dim, generator=generator) * 0.25).to(torch.bfloat16)
    v = (torch.randn(1, cfg.num_kv_heads, total, cfg.head_dim, generator=generator) * 0.25).to(torch.bfloat16)
    queries = [
        (torch.randn(1, cfg.num_heads, seq, cfg.head_dim, generator=generator) * 0.25).to(torch.bfloat16)
        for _ in starts
    ]
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=total, block_size=64)
    page_table = dec.make_page_table(1, kv["blocks_per_user"])
    for cache_name, source in (("k", k), ("v", v)):
        source_tt = D._tt(source, device)
        ttnn.experimental.paged_fill_cache(
            kv[cache_name],
            dec._cast_fill(source_tt, kv["dtype"]),
            page_table,
            batch_idx=0,
        )

    query = D._tt(queries[0], device)
    start_tensor = D._int(torch.tensor([starts[0]], dtype=torch.int32), device)
    # Serving refreshes persistent inputs with copies. Warm those exact copy
    # geometries before the guard, then change only their values below.
    _copy_replicated(queries[0], query, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    _copy_replicated(
        torch.tensor([starts[0]], dtype=torch.int32),
        start_tensor,
        device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    def run(start):
        return dec._prefill_attention(
            query,
            query,
            query,
            kv,
            page_table,
            user_id=0,
            start_pos=start,
            seq=seq,
            chunk_start_idx_tensor=start_tensor,
        )

    first = run(starts[0])
    ttnn.synchronize_device(device)
    first_host = D._compose0(first, device).float().reshape(1, cfg.num_heads, seq, cfg.head_dim)
    first_ref = _chunked_reference(queries[0], k, v, start_pos=starts[0], scale=cfg.scaling)
    _assert_attention_close(first_host, first_ref, label=f"start={starts[0]}")

    before = _freeze_misses(device)
    try:
        _copy_replicated(
            queries[1],
            query,
            device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        _copy_replicated(
            torch.tensor([starts[1]], dtype=torch.int32),
            start_tensor,
            device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        second = run(starts[1])
        ttnn.synchronize_device(device)
    finally:
        _unfreeze_misses(device, before)

    second_host = D._compose0(second, device).float().reshape(1, cfg.num_heads, seq, cfg.head_dim)
    second_ref = _chunked_reference(queries[1], k, v, start_pos=starts[1], scale=cfg.scaling)
    _assert_attention_close(second_host, second_ref, label=f"start={starts[1]}")


@pytest.mark.parametrize("sin", (False, True), ids=("cos", "sin"))
def test_indexed_rope_changes_absolute_rows_without_program_miss(device, hf_config, sin):
    """Indexed RoPE must replace scalar ttnn.slice specialization at arbitrary starts."""
    dec = D._decoder(hf_config, D.FULL_DENSE, device)
    seq = 64
    starts = (64, 320, 65_536)
    rotary_dim = dec.cfg.rotary_dim
    positions = D._int(
        torch.arange(starts[0], starts[0] + seq, dtype=torch.int32).reshape(1, seq),
        device,
        ttnn.uint32,
    )
    output = D._tt(torch.zeros((1, 1, seq, rotary_dim)), device)
    table = dec.sin_2d if sin else dec.cos_2d

    _copy_replicated(
        torch.arange(starts[0], starts[0] + seq, dtype=torch.int32).reshape(1, seq),
        positions,
        device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    dec._rope_prefill_indexed(positions, sin=sin, output_tensor=output)
    ttnn.synchronize_device(device)
    first = _device0(output).float().reshape(seq, rotary_dim)
    table_host = _device0(table).float()
    assert torch.equal(first, table_host[starts[0] : starts[0] + seq])

    before = _freeze_misses(device)
    try:
        for start in starts[1:]:
            _copy_replicated(
                torch.arange(start, start + seq, dtype=torch.int32).reshape(1, seq),
                positions,
                device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            dec._rope_prefill_indexed(positions, sin=sin, output_tensor=output)
        ttnn.synchronize_device(device)
    finally:
        _unfreeze_misses(device, before)

    final = _device0(output).float().reshape(seq, rotary_dim)
    assert torch.equal(final, table_host[starts[-1] : starts[-1] + seq])


def _prepare_resumed_layer_case(dec, ctx, mesh, *, prefix_len, seq, seed):
    torch.manual_seed(seed)
    prefix = torch.randn(1, prefix_len, D.HIDDEN) * 0.5
    suffix = torch.randn(1, seq, D.HIDDEN) * 0.5
    # The remote HF sliding-attention implementation does not build a compatible
    # attention mask when a standalone layer is called with DynamicCache: it
    # constructs a suffix-width mask while concatenating prefix K/V.  A single
    # prefix+suffix call is mathematically equivalent for this causal layer and
    # exercises the same sliding window, without relying on that broken test-only
    # cache interface.  Compare only the resumed suffix rows.
    reference, _ = R.reference_forward(ctx, torch.cat((prefix, suffix), dim=1))
    reference = reference[:, -seq:]

    # Serving uses one fixed full-context cache/table geometry for every request.
    # Keep both tested offsets on the same geometry so only runtime values and
    # buffer addresses change after the cache is frozen.
    kv = dec.alloc_kv_cache(max_users=1, max_seq_len=512, block_size=64)
    page_table = dec.make_page_table(1, kv["blocks_per_user"])
    dec.prefill_forward(D._tt(prefix, mesh), kv, page_table, user_id=0, start_pos=0)
    ttnn.synchronize_device(mesh)
    prefix_blocks = prefix_len // 64
    snapshots = {name: _device0(kv[name])[:prefix_blocks].clone() for name in ("k", "v")}

    fill_page_table = D._int(
        torch.tensor([[prefix_blocks]], dtype=torch.int32),
        mesh,
    )
    positions = D._int(
        torch.arange(prefix_len, prefix_len + seq, dtype=torch.int32).reshape(1, seq),
        mesh,
        ttnn.uint32,
    )
    start_tensor = D._int(torch.tensor([prefix_len], dtype=torch.int32), mesh)
    rotary_dim = dec.cfg.rotary_dim
    cos_output = D._tt(torch.zeros((1, 1, seq, rotary_dim)), mesh)
    sin_output = D._tt(torch.zeros((1, 1, seq, rotary_dim)), mesh)
    runtime = PrefillRuntimeOffsets(
        bucket_len=seq,
        chunk_offsets=(0,),
        chunk_lengths=(seq,),
        position_ids=(positions,),
        chunk_start_idxs=(start_tensor,),
        rope_outputs={dec.cfg.attention_type: ((cos_output, sin_output),)},
    )
    suffix_device = D._tt(suffix, mesh)

    def run():
        cos = dec._rope_prefill_indexed(positions, output_tensor=cos_output)
        sin = dec._rope_prefill_indexed(positions, sin=True, output_tensor=sin_output)
        return dec.prefill_forward(
            suffix_device,
            kv,
            page_table,
            fill_page_table=fill_page_table,
            fill_page_table_base_pos=prefix_len,
            user_id=0,
            start_pos=prefix_len,
            # Runtime-offset prefills always enter the pipeline path, whose RoPE contract is one
            # ``(cos, sin)`` pair per outer chunk. This resumed test has exactly one chunk.
            rope_mats=((cos, sin),),
            runtime_offsets=runtime,
        )

    return {
        "run": run,
        "reference": reference,
        "kv": kv,
        "snapshots": snapshots,
        "prefix_blocks": prefix_blocks,
    }


@pytest.mark.parametrize("layer", (D.FULL_DENSE, D.SLIDING_MOE), ids=("full_dense", "sliding_moe"))
def test_layer_resume_is_accurate_preserves_prefix_and_reuses_programs(device, hf_config, layer):
    """A complete representative layer must resume safely at two absolute offsets."""
    ctx, _raw = D._ctx(hf_config, layer)
    dec = D._decoder(hf_config, layer, device)
    seq = 64
    first = _prepare_resumed_layer_case(dec, ctx, device, prefix_len=64, seq=seq, seed=6400 + layer)
    second = _prepare_resumed_layer_case(dec, ctx, device, prefix_len=320, seq=seq, seed=32000 + layer)

    first_output = first["run"]()
    ttnn.synchronize_device(device)
    first_host = D._compose0(first_output, device).float().reshape(1, seq, D.HIDDEN)
    assert D._pcc(first_host, first["reference"]) >= D.PCC_BAR
    for name in ("k", "v"):
        assert torch.equal(
            _device0(first["kv"][name])[: first["prefix_blocks"]],
            first["snapshots"][name],
        ), f"layer {layer} resume overwrote cached prefix {name.upper()}"

    before = _freeze_misses(device)
    try:
        second_output = second["run"]()
        ttnn.synchronize_device(device)
    finally:
        _unfreeze_misses(device, before)

    second_host = D._compose0(second_output, device).float().reshape(1, seq, D.HIDDEN)
    pcc = D._pcc(second_host, second["reference"])
    assert pcc >= D.PCC_BAR, f"layer {layer} resumed PCC {pcc:.5f} < {D.PCC_BAR}"
    for name in ("k", "v"):
        assert torch.equal(
            _device0(second["kv"][name])[: second["prefix_blocks"]],
            second["snapshots"][name],
        ), f"layer {layer} resume overwrote cached prefix {name.upper()}"
