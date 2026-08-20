# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Token-embedding workload for Tracy / device-perf reports of Qwen3.5-9B.

Runs the same ``tt_transformers.Embedding`` the model uses (``Qwen36Model.embd``) for one
lookup, with ``start``/``stop`` signposts around the ``embd(tok)`` call ONLY. Weight
construction, token upload and teardown sit outside the window.

WHY THIS IS ITS OWN TEST
------------------------
The single-layer attention/GDN profile files embed once in setup and then measure the
layer. Token embedding is a separate production op (``prefill_tp`` / ``decode_tp`` /
``decode_paged`` all start with ``self.embd(tok)``) and it does not appear in those
captures. This file is the Tracy target for that op, so a ``tt-perf-report`` of this
test is a token-embedding report, not a layer report with an extra gather mixed in.

The table is a random ``[vocab, dim]`` bf16 tensor of the real 9B shape (dummy_weights).
Lookup cost does not depend on the values, so loading the checkpoint would only add
setup time. PCC lives in ``tests/unit/test_embedding.py``.

SHAPES
------
Decode uses the same batches the attention-decode captures use (B=1, B=32). Prefill uses
the production chunk (2048) plus a short and a long point so the report can show how the
gather scales with T. Token ids are replicated across the mesh, matching
``prefill_tp`` / ``decode_tp``.

Standalone Tracy capture (N300 9B decode B=32)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B \\
      python -m tracy -p -v -r --dump-device-data-mid-run -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_token_embedding.py::test_profile_token_embedding[wormhole_b0-decode-B32-mesh_device0-device_params0]"

    Note the ``-m`` and the full node id with NO trailing pytest flags: tracy parses argv
    with optparse and a later ``-v`` is taken as tracy's own verbose. Without ``-m``,
    argv[0] is opened as a script path and you get "FileNotFoundError: 'pytest'".

Then, from the generated Tracy CSV::

    tt-perf-report generated/profiler/reports/<run>/ops_perf_results_*.csv \\
      --start-signpost start --end-signpost stop --no-color

Plain run (no profiler; sanity-checks the workload itself)::

    MESH_DEVICE=N300 HF_MODEL=Qwen/Qwen3.5-9B pytest \\
        models/demos/blackhole/qwen36/tests/perf/test_profile_token_embedding.py -v -s
"""

from __future__ import annotations

import os
from typing import NamedTuple

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole

NUM_WARMUP_ITERS = 1

# (mode, length): decode length is batch, prefill length is sequence.
# ids must stay unique across the parametrize so tracy node ids are unambiguous.
CASES = (
    ("decode", 1),
    ("decode", 32),
    ("prefill", 128),
    ("prefill", 2048),
    ("prefill", 8192),
)


def _mesh_device_param() -> tuple[int, int]:
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    if name in explicit:
        return explicit[name]
    return (1, max(1, min(ttnn.get_num_devices(), 2)))


MESH_SHAPE = _mesh_device_param()
_MULTI = MESH_SHAPE != (1, 1)

# fabric_config on multi-device: the mesh fixture wires ETH up front, and without FABRIC_1D
# a 2-chip N300 can fail ETH discovery (same as the other profile files). Embedding itself
# issues no CCL. num_command_queues + trace_region match demo/text_demo.py.
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


class _EmbFixtures(NamedTuple):
    emb: object
    tok: ttnn.Tensor
    args: object
    mode: str
    length: int
    vocab: int
    dim: int


def _setup(mesh_device, mode: str, length: int) -> _EmbFixtures:
    """Build Embedding + one token tensor. Nothing here is profiled."""
    from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
    from models.tt_transformers.tt.embedding import Embedding

    max_seq = max(length, 2048) if mode == "prefill" else 2048
    max_batch = length if mode == "decode" else 1
    # Load HF config from the real snapshot first. Passing dummy_weights=True into Qwen36ModelArgs
    # makes ModelArgs._set_hf_params look up LOCAL_HF_PARAMS[model_name], and model_name is the
    # hashed snapshot dir (e.g. c2022362...) -- KeyError. Same pattern as tests/unit/test_embedding.py:
    # construct normally, then flip dummy_weights so Embedding skips the weight-cache path.
    args = Qwen36ModelArgs(mesh_device=mesh_device, max_batch_size=max_batch, max_seq_len=max_seq)
    args.dummy_weights = True
    vocab, dim = args.vocab_size, args.dim

    torch.manual_seed(0)
    table = torch.randn(vocab, dim, dtype=torch.bfloat16)
    emb = Embedding(
        mesh_device=mesh_device,
        args=args,
        weight_cache_path=None,
        state_dict={"tok_embeddings.weight": table},
        dtype=ttnn.bfloat16,
    )

    n_tok = length
    ids = torch.randint(0, vocab, (1, n_tok), dtype=torch.int32)
    mapper = ttnn.ReplicateTensorToMesh(mesh_device) if _MULTI else None
    tok = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=mapper)

    logger.info(f"profiling token embedding mode={mode} length={length} vocab={vocab} dim={dim} mesh={MESH_SHAPE}")
    return _EmbFixtures(emb=emb, tok=tok, args=args, mode=mode, length=length, vocab=vocab, dim=dim)


def _run_embed(mesh_device, f: _EmbFixtures, *, use_signpost: bool = False) -> None:
    """One embedding lookup. Only ``emb(tok)`` sits inside the signposts."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    if use_signpost:
        from tracy import signpost

        signpost("start")

    out = tpc.decode_embed(f.emb, f.tok, f.args)

    if use_signpost:
        # Inside the window on purpose: without it the clock stops on dispatch, not execution.
        ttnn.synchronize_device(mesh_device)
        signpost("stop")
    else:
        ttnn.synchronize_device(mesh_device)

    ttnn.deallocate(out)


@pytest.mark.timeout(600)
@pytest.mark.models_performance_bare_metal
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(
    "mode,length",
    CASES,
    ids=[f"{mode}-{'B' if mode == 'decode' else 'T'}{n}" for mode, n in CASES],
)
def test_profile_token_embedding(mesh_device, device_params, mode, length):
    """One token-embedding lookup at a production decode-batch or prefill-seq (Tracy target)."""
    del device_params

    use_signpost = _tracy_signpost_available()
    if not use_signpost:
        logger.info("tracy.signpost unavailable; running the workload without signpost markers.")

    mesh_device.enable_program_cache()
    f = _setup(mesh_device, mode, length)

    for _ in range(NUM_WARMUP_ITERS):
        _run_embed(mesh_device, f)

    _run_embed(mesh_device, f, use_signpost=use_signpost)

    ttnn.deallocate(f.tok)

    logger.info(
        f"Profile workload complete: token embedding mode={f.mode} length={f.length} "
        f"vocab={f.vocab} dim={f.dim} signposts={'on' if use_signpost else 'off'}"
    )
