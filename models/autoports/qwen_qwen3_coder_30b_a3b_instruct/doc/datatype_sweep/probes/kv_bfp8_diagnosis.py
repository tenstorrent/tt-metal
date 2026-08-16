# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Why does a ``bfloat8_b`` KV cache score at chance? Two very different answers.

``R19_kv_bfp8`` scored top-1 / top-5 / top-100 all at **0.010** -- chance -- and
was also 32% *slower* with 2.5x worse TTFT. Those two facts together do not look
like precision loss:

* Losing mantissa bits degrades *ranking* gracefully. ``bfloat4_b`` expert
  weights -- four mantissa bits, far more aggressive than bfp8 -- hold top-5 at
  1.000 in this very sweep. A merely-imprecise cache does not put the correct
  token outside the **top 100** in 99% of positions.
* A pure dtype reduction should be *faster*, not slower: fewer bytes off DRAM.
  Slower **and** broken says a different code path, not the same path at lower
  precision.

So the conclusion "bfp8 KV is too imprecise for this model" would be the wrong
write-up if the real answer is "our path does not correctly support a bfp8 KV
cache". ``bfloat8_b`` KV is used widely on Tenstorrent (``tt_transformers``
relies on it to buy capacity), which makes a model-specific collapse suspicious.

This probe separates the two **at the op level**, with no model and no decode
loop, so the answer costs seconds rather than a 3-minute run:

1. ``roundtrip_fill(cache_dtype, input_dtype)`` -- allocate a paged cache, write
   a known random tensor through ``paged_fill_cache``, read it back, and PCC it
   against what went in. This is the narrowest possible question: *does the
   cache store what we put in it?* If a bf16 input into a bfp8 cache comes back
   as noise while a bfp8 input into the same cache round-trips, the defect is
   ours -- we hand the fill an input whose dtype does not match the cache, and
   the op's permissive ``OR`` on the input dtype does not reject it.
2. ``roundtrip_update`` -- the same for ``paged_update_cache``, the **decode**
   -path writer, at decode shapes and with the sharded update tensor that op
   demands.

Measured, the two writers turn out to have **opposite** contracts, which is the
whole answer:

====================  ==========  ==========  ================================
op                    cache       input       round-trip
====================  ==========  ==========  ================================
``paged_fill_cache``  bfloat16    bfloat16    PCC 1.0 (control)
``paged_fill_cache``  bfloat8_b   bfloat16    **NaN**  <- what R19 ran
``paged_fill_cache``  bfloat8_b   bfloat8_b   PCC 1.0
``paged_update_``     bfloat16    bfloat16    PCC 1.0 (control)
``paged_update_``     bfloat8_b   bfloat16    PCC 0.999969
``paged_update_``     bfloat8_b   bfloat8_b   **rejected**, ``...:296``
====================  ==========  ==========  ================================

The fill writer wants the input cast *to* the cache dtype; the update writer
wants it left as bfloat16 and converts internally, and hard-rejects a
block-float update. So the fix is asymmetric -- cast at prefill, do not cast at
decode -- and a symmetric "cast everywhere" fix would have replaced silent
corruption with a crash. See ``tt/functional_decoder.match_cache_dtype``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

BLOCK = 32
HEADS = 1
HEAD_DIM = 128
BLOCKS = 8


def record_pcc(out: dict, value: float) -> None:
    """Store a PCC in a form ``json.dumps(allow_nan=False)`` accepts.

    A bare ``NaN`` is not valid JSON (RFC 8259 has no such literal) and this
    result is embedded verbatim in ``sweep_results.json``, so the NaN -- which
    is the finding, not an error -- is recorded as an explicit boolean plus a
    ``null`` rather than as a token no strict parser will read.
    """
    is_nan = value != value
    out["pcc_vs_input"] = None if is_nan else round(value, 6)
    out["pcc_vs_input_is_nan"] = is_nan


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else float((a @ b).item() / denom)


def roundtrip_fill(mesh, cache_dtype, input_dtype) -> dict:
    """Write via paged_fill_cache and read back. Does the cache hold our data?"""
    torch.manual_seed(0)
    seq = BLOCKS * BLOCK
    cache_t = torch.zeros(BLOCKS, HEADS, BLOCK, HEAD_DIM)
    src = torch.randn(1, HEADS, seq, HEAD_DIM)

    cache = ttnn.from_torch(
        cache_t,
        dtype=cache_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    inp = ttnn.from_torch(
        src,
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    page_table = ttnn.from_torch(
        torch.arange(BLOCKS, dtype=torch.int32).reshape(1, BLOCKS),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    out = {"op": "paged_fill_cache", "cache_dtype": str(cache_dtype), "input_dtype": str(input_dtype)}
    try:
        ttnn.experimental.paged_fill_cache(cache, inp, page_table, batch_idx=0)
        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(ttnn.get_device_tensors(cache)[0])
        # cache is [blocks, heads, block, head_dim]; reassemble to [1,heads,seq,dim]
        got = got.permute(1, 0, 2, 3).reshape(1, HEADS, seq, HEAD_DIM)
        # what the input dtype itself can represent, as the fair reference
        ref = ttnn.to_torch(ttnn.get_device_tensors(inp)[0])
        record_pcc(out, pcc(ref, got))
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = repr(exc)[:400]
    finally:
        for t in (cache, inp, page_table):
            ttnn.deallocate(t, True)
    return out


def roundtrip_update(mesh, cache_dtype, input_dtype) -> dict:
    """The same question for ``paged_update_cache`` -- the **decode**-path writer.

    Worth asking separately, and the more interesting half: the two ops do not
    share validation. ``paged_fill_cache`` at least looks at the input dtype (a
    permissive ``OR``); ``paged_update_cache`` validates **only the cache**, so
    it is the site where a mismatched write is least likely to be caught.

    The shapes are the decode ones, not the prefill ones: the cache is
    ``[blocks, heads, block, head_dim]`` and the update is one token's K for one
    user, ``[1, batch, heads, head_dim]``, **height-sharded in L1** -- the op
    requires a sharded update tensor
    (``paged_update_cache_device_operation.cpp:255``), which is why
    ``attention_decode`` restores the sharded layout before calling it.
    """
    torch.manual_seed(0)
    pos = 5  # somewhere inside the first block, so one row of one block moves
    cache_t = torch.zeros(BLOCKS, HEADS, BLOCK, HEAD_DIM)
    src = torch.randn(1, 1, HEADS, HEAD_DIM)

    cache = ttnn.from_torch(
        cache_t,
        dtype=cache_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    shard = ttnn.create_sharded_memory_config(
        shape=(32, HEAD_DIM),
        core_grid=ttnn.CoreGrid(y=1, x=HEADS),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    # Built interleaved and then resharded, which is how the update tensor
    # arrives in the model (``attention_decode`` restores the split's sharded
    # layout after the norm and RoPE), and avoids a host-side shard-shape
    # mismatch against the tile padding.
    inp = ttnn.to_memory_config(
        ttnn.from_torch(
            src,
            dtype=input_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        ),
        shard,
    )
    page_table = ttnn.from_torch(
        torch.arange(BLOCKS, dtype=torch.int32).reshape(1, BLOCKS),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    idxs = ttnn.from_torch(
        torch.tensor([pos], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    out = {"op": "paged_update_cache", "cache_dtype": str(cache_dtype), "input_dtype": str(input_dtype)}
    try:
        ttnn.experimental.paged_update_cache(cache, inp, update_idxs_tensor=idxs, page_table=page_table)
        ttnn.synchronize_device(mesh)
        got = ttnn.to_torch(ttnn.get_device_tensors(cache)[0])[0, :, pos, :]  # block 0, row `pos`
        ref = ttnn.to_torch(ttnn.get_device_tensors(inp)[0])[0, 0]
        record_pcc(out, pcc(ref, got))
        out["wrote_nan"] = bool(torch.isnan(got).any().item())
        out["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["status"] = "error"
        out["error"] = repr(exc)[:400]
    finally:
        for t in (cache, inp, page_table, idxs):
            ttnn.deallocate(t, True)
    return out


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    results = []
    try:
        for cache_dtype, input_dtype in (
            (ttnn.bfloat16, ttnn.bfloat16),  # the shipped combination -- the control
            (ttnn.bfloat8_b, ttnn.bfloat16),  # what R19 actually ran: bf16 K/V into a bfp8 cache
            (ttnn.bfloat8_b, ttnn.bfloat8_b),  # dtypes matched -- the candidate fix
        ):
            r = roundtrip_fill(mesh, cache_dtype, input_dtype)
            results.append(r)
            print(
                f"paged_fill_cache    cache={str(cache_dtype):22s} input={str(input_dtype):22s} "
                f"-> {r.get('pcc_vs_input', r.get('error'))}",
                flush=True,
            )
            u = roundtrip_update(mesh, cache_dtype, input_dtype)
            results.append(u)
            print(
                f"paged_update_cache  cache={str(cache_dtype):22s} input={str(input_dtype):22s} "
                f"-> {u.get('pcc_vs_input', u.get('error'))}",
                flush=True,
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    import json

    (Path(__file__).resolve().parent / "kv_bfp8_diagnosis.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    print("\nInterpretation:")
    print("  paged_fill_cache:   bfp8 cache + bf16 input => NaN, bfp8 cache + bfp8 input => 1.0.")
    print("                      The cache is fine at bfp8; our call into it was not. CAST here.")
    print("  paged_update_cache: bfp8 cache + bf16 input => ~1.0, bfp8 input REJECTED by the op.")
    print("                      The op converts internally. DO NOT cast here.")


if __name__ == "__main__":
    main()
