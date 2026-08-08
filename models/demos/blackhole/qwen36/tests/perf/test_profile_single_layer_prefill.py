# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-layer PREFILL workload for Tracy / device-perf profiling of Qwen3.5-9B.

Runs **one** decoder layer's ``forward(mode="prefill")`` over ``PREFILL_SEQ_LEN`` tokens. Same
pattern as ``seamless_m4t_v2_large/tests/perf/test_profile_single_layer_prefill_decode.py``: each
measured iteration puts ``start``/``stop`` signposts around the ``forward`` call ONLY. Model build,
weight load, KV/GDN state allocation, embedding, RoPE table construction and teardown all happen
outside the signpost window, and a warmup iteration runs first without signposts so kernel
compilation and first-touch allocation are not measured.

WHY TWO LAYER TYPES
-------------------
Qwen3.5 is hybrid: 24 of its 32 layers are Gated DeltaNet (``linear_attention``) and 8 are
``full_attention``, at indices 3, 7, 11, ... 31. They are completely different kernels with
different cost curves -- the GDN layers run a chunk-scan whose working set grows with the chunk
length, the full-attention layers run paged SDPA against the KV cache -- so a single "one layer"
number would be meaningless. This test builds a 4-layer model (pattern G,G,G,F) and profiles ONE of
them, parametrised: ``layer0_gdn`` or ``layer3_fullattn``.

WHY THESE SEQUENCE LENGTHS
--------------------------
2048 is the production figure: it is the chunk size the demo and vLLM use for chunk-outer prefill,
so a 2048-token single-layer prefill is exactly one unit of real long-context work. 128 is included
as the short-prompt case, and 4096 to show how each kernel scales past one chunk.

Standalone Tracy capture::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B \\
      python -m tracy -p -v -r --dump-device-data-mid-run -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_single_layer_prefill.py::test_profile_single_layer_prefill[wormhole_b0-layer0_gdn-seq2048-mesh_device0-device_params0]"

    Note the ``-m`` (profile a library module) and the full node id with NO trailing pytest flags:
    tracy parses argv with optparse and does not call disable_interspersed_args, so a later ``-v``
    is taken as tracy's own verbose and ``-k`` fails as an unknown option. Without ``-m``, argv[0]
    is opened as a *script path* and you get "FileNotFoundError: 'pytest'".

Plain run (no profiler; sanity-checks the workload itself)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
        models/demos/blackhole/qwen36/tests/perf/test_profile_single_layer_prefill.py -v -s
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole

# 4 layers gives pattern G,G,G,F -- the smallest build containing both kernel types.
NUM_LAYERS = 4
GDN_LAYER_IDX = 0
FULL_ATTN_LAYER_IDX = 3
PREFILL_SEQ_LENS = [128, 2048, 4096]
NUM_WARMUP_ITERS = 1
BLOCK_SIZE = 64


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
    layer_idx: int
    is_full_attention: bool
    seq_len: int
    x: ttnn.Tensor
    cos: ttnn.Tensor
    sin: ttnn.Tensor
    page_table_tt: ttnn.Tensor
    chunk_size: int


def _setup(mesh_device, seq_len: int, layer_idx: int) -> _LayerPerfFixtures:
    """Build the model + every input the layer needs. Nothing here is profiled."""
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    model = Qwen36Model.from_pretrained(
        mesh_device, max_batch_size=1, max_seq_len=max(2048, seq_len), n_layers=NUM_LAYERS
    )
    args = model.args

    # Paged KV for the full-attention layer. Sized for the padded prefill length: the chunk/masked
    # prefill writes K/V for the whole rounded-up bucket, not just the real token count.
    num_blocks = max(1, (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) + 2
    model.allocate_kv_caches(
        [num_blocks, args.n_local_kv_heads, BLOCK_SIZE, args.head_dim], ttnn.bfloat16, batch_size=1
    )
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    page_table_tt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)

    # Layer input: [1, 1, T, dim_frac] in DRAM, exactly what the model's prefill loop passes.
    torch.manual_seed(0)
    tokens = torch.randint(0, 2000, (1, seq_len), dtype=torch.long)
    model._build_request_rope(tokens, None)
    tok = ttnn.from_torch(
        tokens.to(torch.int32),
        dtype=ttnn.uint32,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    x = model.embd(tok)
    x = ttnn.reshape(x, (1, 1, seq_len, x.shape[-1]))
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(tok)

    cos_t, sin_t = model._rope_tp_cos_sin_torch(0, seq_len)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)
    sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep)

    layer = model.layers[layer_idx]
    logger.info(
        f"profiling layer {layer_idx} "
        f"({'FULL-ATTENTION' if layer.is_full_attention else 'GDN (linear_attention)'}), "
        f"seq_len={seq_len}, mesh={MESH_SHAPE}, dim_frac={x.shape[-1]}"
    )
    return _LayerPerfFixtures(
        model=model,
        layer=layer,
        layer_idx=layer_idx,
        is_full_attention=bool(layer.is_full_attention),
        seq_len=seq_len,
        x=x,
        cos=cos,
        sin=sin,
        page_table_tt=page_table_tt,
        chunk_size=args.gdn_chunk_size,
    )


def _run_prefill_step(mesh_device, f: _LayerPerfFixtures, *, use_signpost: bool = False) -> None:
    """One prefill forward on the single layer. Only the forward sits inside the signposts."""
    if use_signpost:
        from tracy import signpost

        signpost("start")

    out = f.layer.forward(
        f.x,
        cos=f.cos,
        sin=f.sin,
        mode="prefill",
        chunk_size=f.chunk_size,
        valid_len=f.seq_len,
        page_table=f.page_table_tt,
        chunk_page_table=f.page_table_tt,
        chunk_start_idx=0,
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
@pytest.mark.parametrize("seq_len", PREFILL_SEQ_LENS, ids=[f"seq{n}" for n in PREFILL_SEQ_LENS])
@pytest.mark.parametrize(
    "layer_idx",
    [GDN_LAYER_IDX, FULL_ATTN_LAYER_IDX],
    ids=[f"layer{GDN_LAYER_IDX}_gdn", f"layer{FULL_ATTN_LAYER_IDX}_fullattn"],
)
def test_profile_single_layer_prefill(mesh_device, device_params, seq_len, layer_idx):
    """Prefill `seq_len` tokens through ONE Qwen3.5 decoder layer (Tracy profile target)."""
    del device_params

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running the workload without signpost markers.")

    mesh_device.enable_program_cache()
    f = _setup(mesh_device, seq_len, layer_idx)

    for _ in range(NUM_WARMUP_ITERS):
        _run_prefill_step(mesh_device, f)

    _run_prefill_step(mesh_device, f, use_signpost=use_signpost)

    ttnn.deallocate(f.x)
    ttnn.deallocate(f.cos)
    ttnn.deallocate(f.sin)
    ttnn.deallocate(f.page_table_tt)

    logger.info(
        f"Profile workload complete: layer={f.layer_idx} "
        f"({'full-attention' if f.is_full_attention else 'GDN'}), prefill_seq_len={f.seq_len}, "
        f"gdn_chunk_size={f.chunk_size}, signposts={'on' if use_signpost else 'off'}"
    )
