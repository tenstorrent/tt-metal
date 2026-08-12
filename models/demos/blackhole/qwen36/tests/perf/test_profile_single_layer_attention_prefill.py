# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-layer FULL-ATTENTION PREFILL workload, swept over total context length, for Tracy /
device-perf profiling of Qwen3.5-9B.

Runs **one** ``full_attention`` decoder layer's ``forward(mode="prefill")`` for exactly the LAST
chunk of a chunk-outer prefill at a given total context length. Same pattern as
``test_profile_single_layer_prefill.py`` / ``test_profile_single_layer_gdn_decode.py``: a warmup
iteration runs first (uninstrumented, so kernel compilation and first-touch allocation are not
measured), then one measured iteration puts ``start``/``stop`` signposts around the ``forward``
call ONLY.

WHY THIS IS ITS OWN TEST (NOT THE SHARED PREFILL FILE)
--------------------------------------------------------
``test_profile_single_layer_prefill.py`` already has a ``layer3_fullattn`` case, but it treats the
entire ``seq_len`` (up to 4096) as ONE chunk starting at position 0 -- there is no history in the
KV cache. That is not how production runs full-attention at long context: ``model.py``'s
``prefill_layer_chunked`` fixes the attention chunk length at ``attn_chunk_size = max(chunk_size,
4096)`` and loops ``for chunk_start in range(0, T, attn_chunk_size)``, so a 128k/256k-token prefill
is hundreds of separate ``chunked_scaled_dot_product_attention`` calls, each one attending
causally over an ever-larger KV cache (``chunk_start_idx`` grows every iteration). The O(seq^2)
cost the model docs warn about shows up in how THAT per-chunk cost grows with ``chunk_start_idx``,
not in the cost of one huge one-shot SDPA call the model never actually issues.

This test isolates exactly that: for each total context length ``L`` (see WHY THESE LENGTHS), it
builds a paged KV cache sized for ``L`` tokens and profiles the forward call for the FINAL
``ATTN_CHUNK_SIZE``-token chunk only -- i.e. "how expensive is the last prefill step of an L-token
prompt". The KV cache entries for ``[0, chunk_start)`` are allocated but never populated by a real
prior chunk: ``chunked_scaled_dot_product_attention`` reads them by page-table shape, not value, so
an uninitialized cache costs exactly what a populated one would (this is a device-TIME profile, not
a correctness/PCC check -- ``test_gated_attention_prefill`` and friends cover correctness). Skipping
the (L/ATTN_CHUNK_SIZE - 1) prior chunks is what makes an L=262144 sweep point tractable at all.

``ATTN_CHUNK_SIZE`` is 4096 on Blackhole (what ``prefill_layer_chunked`` actually runs) and 2048 on
Wormhole, which is the largest chunk that fits WH's L1 on this path -- see the constant's comment.

WHY THESE LENGTHS
------------------
``[4096, 8192, 16384, 32768, 65536, 131072, 262144]`` are exactly the non-trivial ISLs in
``demo/text_demo.py``'s ``PREFILL_SEQ_LENS`` (128 is dropped: it is shorter than one attention
chunk, so its "last chunk" is chunk 0 with no history -- already covered by the shared prefill
file's ``seq_len=128`` case), plus ``ATTN_CHUNK_SIZE`` itself so the sweep always includes the
no-history ``chunk_start=0`` baseline. Each is a multiple of ``ATTN_CHUNK_SIZE`` on purpose:
full-attention layers are prefilled at one fixed chunk length, so ``chunk_start = L -
ATTN_CHUNK_SIZE`` reproduces the real final chunk boundary for that ISL rather than an arbitrary
offset. The ``chunk_start=0`` point should roughly match the shared file's
``layer3_fullattn-seq2048`` device time on WH, as a sanity cross-check between the two tests.

Standalone Tracy capture::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B \\
      python -m tracy -p -v -r --dump-device-data-mid-run -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_single_layer_attention_prefill.py::test_profile_single_layer_attention_prefill[wormhole_b0-ctx16384-mesh_device0-device_params0]"

    Note the ``-m`` (profile a library module) and the full node id with NO trailing pytest flags:
    tracy parses argv with optparse and does not call disable_interspersed_args, so a later ``-v``
    is taken as tracy's own verbose and ``-k`` fails as an unknown option. Without ``-m``, argv[0]
    is opened as a *script path* and you get "FileNotFoundError: 'pytest'".

Plain run (no profiler; sanity-checks the workload itself)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
        models/demos/blackhole/qwen36/tests/perf/test_profile_single_layer_attention_prefill.py -v -s
"""

from __future__ import annotations

import math
import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole

# Checkpoint layer 3 is the first full_attention layer in the hybrid G,G,G,F,... pattern.
# from_pretrained(layer_indices=[...]) builds a 1-layer model containing ONLY this checkpoint
# layer (keeping its real type), so no GDN layers are built alongside it.
FULL_ATTN_LAYER_IDX = 3
# Per-call prefill chunk length for a full-attention layer.
#
# Blackhole: 4096, matching prefill_layer_chunked's attn_chunk_size = max(chunk_size, 4096) at the
# production chunk_size=2048 default -- the length full-attention actually runs at there.
#
# Wormhole: 2048. A 4096-token chunk does not run on WH at all -- by the time the layer reaches
# chunked_scaled_dot_product_attention, the prefill activations have taken L1 down to offset 293760
# on every core of the 8x8 grid (~1.15MB/core in use), while the SDPA program's statically allocated
# CBs need up to 826944, so the op dies at compile time with
#   "Statically allocated circular buffers in program N clash with L1 buffers on core range
#    [0-0 - 7-7]. L1 buffer allocated at 293760 and static circular buffer region ends at 826944".
# The same failure reproduces in test_profile_single_layer_prefill.py's layer3_fullattn-seq4096
# case, so it is the WH prefill path's L1 budget, not anything this test does. It is the failure
# mode tp_common.prefill_out_memory_config already documents and works around for the prefill
# projections -- and the reason that helper keeps the tuned L1 placement for Blackhole only, whose
# larger grid halves per-core L1 pressure for the same activation. CBs are L1-only, so no SDPA
# program-config change fixes it; activations are what would have to move. (Moving Q or the
# attention-norm output to DRAM in isolation does NOT clear it -- the floor is unchanged to the
# byte -- so whatever pins L1 that low has not been pinned down here.)
#
# 2048 is the largest chunk that runs on WH; it also makes this test's device times directly
# comparable with test_profile_single_layer_prefill.py's layer3_fullattn-seq2048 case. NOTE:
# model.py's prefill_layer_chunked still hardcodes 4096 for attention layers on both archs, so a
# >=4096-token prefill through the full model hits this same wall on Wormhole.
ATTN_CHUNK_SIZE = 4096 if "blackhole" in ttnn.get_arch_name() else 2048
# Paged-KV block size used by the full-attention chunked-prefill path (model.py: block_size = 64).
BLOCK_SIZE = 64
# ATTN_CHUNK_SIZE is included so the sweep always has its no-history (chunk_start=0) baseline; on
# Blackhole that is already the first entry.
CONTEXT_LENS = sorted({ATTN_CHUNK_SIZE, 4096, 8192, 16384, 32768, 65536, 131072, 262144})
NUM_WARMUP_ITERS = 1


def _mesh_device_param() -> tuple[int, int]:
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    if name in explicit:
        return explicit[name]
    return (1, max(1, min(ttnn.get_num_devices(), 2)))


MESH_SHAPE = _mesh_device_param()
_MULTI = MESH_SHAPE != (1, 1)

# Multi-device needs fabric_config: the TP all-gather / reduce-scatter run over the ETH links and
# without FABRIC_1D they are never discovered, after which the first CCL hangs and wedges the ETH
# cores. num_command_queues=2 + trace_region match demo/text_demo.py so the profile reflects the
# served configuration.
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "num_command_queues": 2,
        **({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1024 * 1024 * 1024} if _MULTI else {}),
    }
]


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


class _LayerPerfFixtures(NamedTuple):
    model: object
    layer: object
    context_len: int
    chunk_start: int
    x: ttnn.Tensor
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    page_table_tt: ttnn.Tensor
    chunk_page_table_tt: ttnn.Tensor


def _setup(mesh_device, context_len: int) -> _LayerPerfFixtures:
    """Build the model + every input the final ATTN_CHUNK_SIZE-token chunk needs. Nothing here is
    profiled: only the chunk being measured gets real tokens/embeddings, and the KV cache is sized
    for the full context but never populated by the (skipped) prior chunks -- see module docstring."""
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    chunk_start = context_len - ATTN_CHUNK_SIZE
    assert chunk_start >= 0, f"context_len {context_len} must be >= ATTN_CHUNK_SIZE {ATTN_CHUNK_SIZE}"

    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=context_len, layer_indices=[FULL_ATTN_LAYER_IDX]
    )
    args = model.args
    layer = model.layers[0]
    assert layer.is_full_attention, f"checkpoint layer {FULL_ATTN_LAYER_IDX} expected full_attention"

    num_blocks = math.ceil(context_len / BLOCK_SIZE) + 2
    model.allocate_kv_caches(
        [num_blocks, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, batch_size=1
    )
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    page_table_tt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

    # Blocks written by THIS chunk only, matching prefill_layer_chunked's chunk_page_table slice.
    chunk_blocks_start = chunk_start // BLOCK_SIZE
    chunk_blocks_end = math.ceil(context_len / BLOCK_SIZE)
    chunk_page_table = page_table[:, chunk_blocks_start:chunk_blocks_end]
    chunk_page_table_tt = ttnn.from_torch(
        chunk_page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
    )

    # Layer input: just the ATTN_CHUNK_SIZE tokens of the chunk being profiled, not the full
    # context_len -- the whole point of skipping prior chunks is to not need to embed/build them.
    torch.manual_seed(0)
    tokens = torch.randint(0, 2000, (1, ATTN_CHUNK_SIZE), dtype=torch.long)
    tok = ttnn.from_torch(
        tokens.to(torch.int32),
        dtype=ttnn.uint32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x = model.embd(tok)
    x = ttnn.reshape(x, (1, 1, ATTN_CHUNK_SIZE, x.shape[-1]))
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tok)

    # No _build_request_rope call: that only stages the M-RoPE per-request table for multimodal
    # requests. Text-only, it's a no-op detour -- prefill_cos_sin_torch's plain-1D-RoPE branch
    # (self._req_cos is None) already supports any (start, length) directly.
    cos_t, sin_t = model._rope_tp_cos_sin_torch(chunk_start, ATTN_CHUNK_SIZE)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    logger.info(
        f"profiling layer {FULL_ATTN_LAYER_IDX} (FULL-ATTENTION), context_len={context_len}, "
        f"chunk_start={chunk_start}, chunk_len={ATTN_CHUNK_SIZE}, mesh={MESH_SHAPE}, "
        f"dim_frac={x.shape[-1]}"
    )
    return _LayerPerfFixtures(
        model=model,
        layer=layer,
        context_len=context_len,
        chunk_start=chunk_start,
        x=x,
        cos=cos,
        sin=sin,
        page_table_tt=page_table_tt,
        chunk_page_table_tt=chunk_page_table_tt,
    )


def _run_prefill_step(mesh_device, f: _LayerPerfFixtures, *, use_signpost: bool = False) -> None:
    """One chunk-outer prefill forward on the single layer. Only the forward sits inside the
    signposts."""
    if use_signpost:
        from tracy import signpost

        signpost("start")

    out = f.layer.forward(
        f.x,
        cos=f.cos,
        sin=f.sin,
        mode="prefill",
        page_table=f.page_table_tt,
        chunk_page_table=f.chunk_page_table_tt,
        chunk_start_idx=f.chunk_start,
    )

    if use_signpost:
        # Inside the window on purpose: without it the clock stops on dispatch, not execution.
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
    else:
        ttnn.synchronize_device(mesh_device)

    ttnn.deallocate(out)


@pytest.mark.timeout(3600)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("context_len", CONTEXT_LENS, ids=[f"ctx{n}" for n in CONTEXT_LENS])
def test_profile_single_layer_attention_prefill(mesh_device, device_params, context_len):
    """Prefill the FINAL chunk of a `context_len`-token prompt through ONE Qwen3.5 full-attention
    decoder layer (Tracy profile target)."""
    del device_params

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running the workload without signpost markers.")

    mesh_device.enable_program_cache()
    f = _setup(mesh_device, context_len)

    for _ in range(NUM_WARMUP_ITERS):
        _run_prefill_step(mesh_device, f)

    _run_prefill_step(mesh_device, f, use_signpost=use_signpost)

    ttnn.deallocate(f.x)
    ttnn.deallocate(f.cos)
    ttnn.deallocate(f.sin)
    ttnn.deallocate(f.page_table_tt)
    ttnn.deallocate(f.chunk_page_table_tt)

    logger.info(
        f"Profile workload complete: layer={FULL_ATTN_LAYER_IDX} (full-attention), "
        f"context_len={f.context_len}, chunk_start={f.chunk_start}, chunk_len={ATTN_CHUNK_SIZE}, "
        f"signposts={'on' if use_signpost else 'off'}"
    )
