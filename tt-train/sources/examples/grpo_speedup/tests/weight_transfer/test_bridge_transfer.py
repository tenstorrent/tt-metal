# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-process ttml -> tt-transformers weight bridge test.

Launched under tt-run with world_size == 2 (see ``runner.sh``):

* Rank 0 (TTML_RANK)
    - Owns one full N300 board (``TT_VISIBLE_DEVICES="0"``). The MGD
      declares a ``[1, 2]`` mesh covering both chips on the board and the
      application opens the full ``[1, 2]`` sub-mesh -- both chips
      participate.
    - Builds a ttml ``LlamaCompositeKV`` for ``Llama-3.2-1B`` from real
      HuggingFace weights with DDP enabled across the two chips
      (parameters are still fully replicated; data is split for the
      sanity-completion generate).
    - Generates a short completion as a "model is healthy" sanity check.
    - Calls ``model.export_to_hf_dict()`` and ships the resulting
      on-device tensor dict over a :class:`WeightBridge` to rank 1.
      :class:`WeightBridge` exchanges mesh shapes during ``connect()``
      and resolves the asymmetric ``[1, 2] -> [1, 1]`` pair to a single
      ``(0, 0) -> (0, 0)`` ``SocketConnection``, so only chip ``(0, 0)``
      of the ttml mesh participates in the transfer; the other ttml chip
      sits idle for its duration.

* Rank 1 (TTT_RANK)
    - Owns the second N300 board (``TT_VISIBLE_DEVICES="1"``). The MGD
      declares a ``[1, 2]`` mesh per rank to match the board's two
      chips, but the application opens only a ``[1, 1]`` sub-mesh
      pinned to chip 0 so every ``Transformer.update_weights`` dispatch
      hits the single-device fast path. No AutoContext, no ttml device
      init -- the ttt rank stays ttml-free.
    - Builds a tt-transformers :class:`Transformer` for ``Llama-3.2-1B``
      from real HuggingFace weights, with ``disable_disk_cache=True`` to
      skip the cold-cache ``dump_tensor_flatbuffer`` collective (which
      would deadlock against the asymmetric TTML rank).
    - Receives the HF-keyed dict from the bridge and asserts on every
      tensor's metadata: dtype, layout, full replication, distribution
      shape, and 4D ``(1, 1, *, *)`` rank.
    - Calls ``Transformer.update_weights(hf_dict)`` to apply the received
      weights in place. Single-chip sub-mesh today
      (``num_devices_per_group == 1``) so every leaf ``.update()`` lands
      on the implemented fast path; the multi-chip path is still TODO.
    - Generates a short completion from the freshly-updated model so the
      whole bridge -> update -> generate pipeline runs end-to-end.

The test is a pytest test so failures surface as standard pytest errors;
it self-skips when not launched under tt-run (``OMPI_COMM_WORLD_SIZE``
unset or != 2).

HF auth: requires ``HF_TOKEN`` set in the environment.
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import Any

import pytest


_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_bridge_transfer must run under tt-run with world_size == 2 " "(use tests/weight_transfer/runner.sh).",
        allow_module_level=True,
    )

_MPI_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])

import ttnn  # noqa: E402

from utils.weight_bridge import TTML_RANK, TTT_RANK, WeightBridge  # noqa: E402

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
INSTRUCT_MODEL_ID = "meta-llama/Llama-3.2-1B-instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0

# Asymmetric mesh shapes:
# - TTML side opens both chips of board 0 ([1, 2], DDP across the pair).
# - TTT  side opens only chip 0 of board 1 ([1, 1]) so every
#   ``Transformer.update_weights`` dispatch hits the single-device fast
#   path. ``WeightBridge`` resolves the pair to a single ``(0,0) ->
#   (0,0)`` ``SocketConnection`` during ``connect()``.
TTML_MESH_SHAPE = (1, 2)
TTT_MESH_SHAPE = (1, 1)


def _ttml_rank_send_state() -> None:
    """Open the ttml mesh, build the model, ship its weights."""
    import ttml
    from ttml.common.config import get_model_config

    from _completer_utils import close_device, load_device_config, open_device
    from utils.llama_completer_ttml import LlamaCompleterTtml, LlamaCompletionCtx

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = load_device_config()
    device_config.mesh_shape = list(TTML_MESH_SHAPE)
    device_config.enable_ddp = True
    device_config.enable_tp = False
    # MGD declares a [1, 2] mesh per rank covering both chips of the
    # board; the application opens the full mesh so the sanity-completion
    # generate can split its batch across both chips under DDP. Weight
    # parameters loaded via ``load_from_safetensors`` are still fully
    # replicated, which is what ``WeightBridge`` requires.
    device_config.device_ids = [0, 1]

    mesh_device = open_device(device_config)
    completer: Any = None
    try:
        ctx = LlamaCompletionCtx(
            max_tokens_to_complete=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            # ``LlamaCompleterTtml`` asserts ``B % num_devices == 0`` so
            # with 2 ttml chips we need at least 2 completions per prompt.
            completions_per_prompt=TTML_MESH_SHAPE[0] * TTML_MESH_SHAPE[1],
        )
        completer = LlamaCompleterTtml(
            ctx=ctx,
            transformer_config=get_model_config(raw["training_config"]["model_config"]),
            mesh_device=mesh_device,
            model_source=INSTRUCT_MODEL_ID,
            enable_ddp=True,
        )

        prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)
        completion_ids = [int(t) for t in completer.generate([prompt_ids])[0]]
        completion_text = completer.tokenizer.decode(completion_ids, skip_special_tokens=False)
        print(
            f"[TTML rank {TTML_RANK}] sanity completion ({len(completion_ids)} tok): " f"{completion_text!r}",
            flush=True,
        )

        bridge = WeightBridge(role="ttml", peer_rank=TTT_RANK, device=mesh_device)
        # Both ranks must reach connect() before MeshSocket's 10s
        # descriptor-exchange timeout fires.
        bridge.connect()

        hf_dict = completer.model.export_to_hf_dict()
        try:
            print(
                f"[TTML rank {TTML_RANK}] sending {len(hf_dict)} weight tensors via WeightBridge",
                flush=True,
            )
            bridge.transfer_state(hf_dict)
            print(f"[TTML rank {TTML_RANK}] send complete", flush=True)
        finally:
            del hf_dict
            gc.collect()

        bridge.barrier()
    finally:
        completer = None
        gc.collect()
        close_device()


def _ttt_rank_recv_update_and_generate() -> None:
    """Open the ttt mesh, build the TTT model, receive + apply weights, generate."""
    from utils.llama_completer_ttt import LlamaCompleterTtt

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    # MGD declares a [1, 2] mesh per rank (the full N300); we pin the
    # sub-mesh origin to chip 0 so the second chip stays idle and
    # num_devices_per_group == 1 across every leaf .update() dispatch.
    # ``WeightBridge`` handles the asymmetric ttml [1, 2] -> ttt [1, 1]
    # pairing by negotiating a single (0, 0) -> (0, 0) ``SocketConnection``.
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*TTT_MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    completer: Any = None
    try:
        # Build TTT with real HF weights so the post-transfer generate
        # exercises the fully-loaded forward path. ``disable_disk_cache``
        # short-circuits the on-disk tensor cache that would otherwise
        # call ``dump_tensor_flatbuffer`` in DISTRIBUTED_GATHER mode and
        # deadlock against the asymmetric TTML rank on MPI_COMM_WORLD.
        completer = LlamaCompleterTtt(
            mesh_device=mesh_device,
            model_source=BASE_MODEL_ID,
            max_batch_size=1,
            instruct=False,
            disable_disk_cache=True,
        )

        # Baseline completion from the freshly-built TTT model (no bridge
        # update yet). Lets us diff against the post-update completion to
        # see whether the bridged weights actually changed the forward
        # path. Safe to run before bridge.connect(): TTML waits at its own
        # connect() handshake recv (an MPI blocking call with no timeout)
        # while TTT does this generate.
        prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)
        pre_update_ids = [
            int(t)
            for t in completer.generate(
                [prompt_ids],
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )[0]
        ]
        pre_update_text = completer.tokenizer.decode(pre_update_ids, skip_special_tokens=False)
        print(
            f"[TTT rank {TTT_RANK}] pre-update  completion ({len(pre_update_ids)} tok): " f"{pre_update_text!r}",
            flush=True,
        )

        bridge = WeightBridge(role="ttt", peer_rank=TTML_RANK, device=mesh_device)
        # Pairs with the ttml-side connect().
        bridge.connect()

        hf_dict = bridge.transfer_state()
        assert hf_dict is not None, "WeightBridge.transfer_state on ttt rank must return a dict"
        print(
            f"[TTT rank {TTT_RANK}] received {len(hf_dict)} weight tensors via WeightBridge",
            flush=True,
        )

        # Received tensors are allocated on the local ttt mesh
        # (``TTT_MESH_SHAPE``) regardless of the sender's mesh shape.
        expected_dist_shape = list(TTT_MESH_SHAPE)
        for key, tensor in hf_dict.items():
            assert tensor.dtype == ttnn.bfloat16, f"key={key!r} dtype={tensor.dtype}, expected bfloat16"
            assert tensor.layout == ttnn.TILE_LAYOUT, f"key={key!r} layout={tensor.layout}, expected TILE_LAYOUT"
            assert (
                tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            ), f"key={key!r} memory_config={tensor.memory_config()}, expected DRAM_MEMORY_CONFIG"

            shape = list(tensor.shape)
            assert (
                len(shape) == 4 and shape[0] == 1 and shape[1] == 1
            ), f"key={key!r} expected 4D (1, 1, *, *) tensor, got shape={shape}"

            placements = tensor.tensor_topology().placements()
            assert len(placements) == len(
                TTT_MESH_SHAPE
            ), f"key={key!r} placement axes ({len(placements)}) != mesh axes ({len(TTT_MESH_SHAPE)})"
            assert all(isinstance(p, ttnn.PlacementReplicate) for p in placements), (
                f"key={key!r} not fully replicated: placements={placements}. WeightBridge "
                "should have allocated a fully replicated template on every mesh axis."
            )

            distribution_shape = [int(d) for d in tensor.tensor_topology().distribution_shape()]
            assert (
                distribution_shape == expected_dist_shape
            ), f"key={key!r} distribution_shape={distribution_shape}, expected {expected_dist_shape}"

        # Smoke-check a few well-known keys so a silent rename / dispatch
        # error in export_to_hf_dict shows up here, not in update_weights.
        for required_key in ("model.embed_tokens.weight", "lm_head.weight"):
            assert required_key in hf_dict, f"missing required HF key {required_key!r} in received dict"

        print(f"[TTT rank {TTT_RANK}] all received tensors pass replicated/dtype/layout checks", flush=True)

        # In-place apply: every required HF key must match a leaf
        # .update(); every provided HF key must be consumed exactly once.
        # On the [1, 1] mesh num_devices_per_group == 1 so every dispatch
        # hits the single-device fast path.
        t0 = time.perf_counter()
        completer.model.update_weights(hf_dict)
        print(
            f"[TTT rank {TTT_RANK}] update_weights complete in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )

        # Drop the now-consumed bridge tensors so the on-device buffers
        # backing them can be reclaimed before generate() allocates trace
        # / KV-cache scratch.
        del hf_dict
        gc.collect()

        post_update_ids = [
            int(t)
            for t in completer.generate(
                [prompt_ids],
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )[0]
        ]
        post_update_text = completer.tokenizer.decode(post_update_ids, skip_special_tokens=False)
        print(
            f"[TTT rank {TTT_RANK}] post-update completion ({len(post_update_ids)} tok): " f"{post_update_text!r}",
            flush=True,
        )

        bridge.barrier()
    finally:
        completer = None
        gc.collect()
        ttnn.close_mesh_device(mesh_device)


def test_ttml_to_ttt_weight_bridge_transfer() -> None:
    """End-to-end bridge transfer + update_weights + generate.

    Branches on the MPI rank set by ``tt-run`` -- the same test function
    runs in both processes. The bridge's host-side manifest exchange and
    fabric ``send_async``/``recv_async`` synchronise the two paths.
    """
    if _MPI_RANK == TTML_RANK:
        _ttml_rank_send_state()
    elif _MPI_RANK == TTT_RANK:
        _ttt_rank_recv_update_and_generate()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {_MPI_RANK} (world_size={_WORLD_SIZE}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
