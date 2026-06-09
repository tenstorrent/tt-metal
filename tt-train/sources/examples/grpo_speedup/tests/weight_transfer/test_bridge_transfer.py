# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-process ttml -> tt-transformers weight bridge test (option A).

Launched under tt-run with world_size == 2 (see ``runner.sh``):

* Rank 0 (TTML_RANK)
    - Opens its own ``[1, 4]`` half of T3K via ttml ``AutoContext`` with
      DDP enabled.
    - Builds a ttml ``LlamaCompositeKV`` for ``Llama-3.2-1B-Instruct``.
    - Generates a short completion as a "model is healthy" sanity check.
    - Calls ``model.export_to_hf_dict()`` and ships the resulting on-device
      tensor dict over a :class:`WeightBridge` to rank 1.

* Rank 1 (TTT_RANK)
    - Opens its own ``[1, 4]`` half of T3K via ``ttnn.open_mesh_device``
      (no AutoContext, no ttml device init -- the ttt rank stays
      ttml-free).
    - Builds a tt-transformers :class:`Transformer` for ``Llama-3.2-1B``
      (base weights). Construction alone is enough for option A; we do
      *not* call ``Transformer.update_weights`` yet because
      ``Attention.update`` raises ``NotImplementedError`` for
      ``num_devices_per_group > 1`` and a ``[1, 4]`` mesh forces TP=4 in
      ``ModelArgs``. Once on-device resharding lands in
      ``models/tt_transformers``, this test will be extended to do the
      full apply-and-compare path.
    - Receives the HF-keyed dict from the bridge and asserts on every
      tensor's metadata: dtype, layout, full replication, distribution
      shape, and 4D ``(1, 1, *, *)`` rank.

The test is a pytest test so failures surface as standard pytest errors;
it self-skips when not launched under tt-run (``OMPI_COMM_WORLD_SIZE``
unset or != 2).

HF auth: requires ``HF_TOKEN`` set in the environment; both repos are
gated.
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
INSTRUCT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The capital of France is"
SANITY_MAX_NEW_TOKENS = 8
TEMPERATURE = 0.0
MESH_SHAPE = (1, 4)


def _ttml_rank_send_state() -> None:
    """Open the ttml mesh, build the instruct model, ship its weights."""
    import ttml
    from ttml.common.config import get_model_config

    from _completer_utils import close_device, load_device_config, open_device
    from utils.llama_completer_ttml import LlamaCompleterTtml, LlamaCompletionCtx

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = load_device_config()
    device_config.mesh_shape = list(MESH_SHAPE)
    device_config.enable_ddp = True
    device_config.enable_tp = False
    device_config.device_ids = None

    mesh_device = open_device(device_config)
    completer: Any = None
    try:
        ctx = LlamaCompletionCtx(
            max_tokens_to_complete=SANITY_MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            completions_per_prompt=MESH_SHAPE[0] * MESH_SHAPE[1],
        )
        completer = LlamaCompleterTtml(
            ctx=ctx,
            transformer_config=get_model_config(raw["training_config"]["model_config"]),
            mesh_device=mesh_device,
            model_source=BASE_MODEL_ID,
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


def _ttt_rank_recv_state_and_assert() -> None:
    """Open the ttt mesh, build the base TTT model, receive and validate weights."""
    from utils.llama_completer_ttt import LlamaCompleterTtt

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
    completer: Any = None
    try:
        # Building TTT here keeps the receiver shape consistent with what
        # ``Transformer.update_weights`` will eventually consume; it also
        # exercises the [1, 4] mesh on the ttt side under realistic memory
        # pressure (one full model already resident when we receive).
        completer = LlamaCompleterTtt(
            mesh_device=mesh_device,
            model_source=INSTRUCT_MODEL_ID,
            max_batch_size=1,
            instruct=False,
            dummy_weights=True,
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

        expected_dist_shape = list(MESH_SHAPE)
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
                MESH_SHAPE
            ), f"key={key!r} placement axes ({len(placements)}) != mesh axes ({len(MESH_SHAPE)})"
            assert all(isinstance(p, ttnn.PlacementReplicate) for p in placements), (
                f"key={key!r} not fully replicated: placements={placements}. WeightBridge "
                "should have allocated a fully replicated template on every mesh axis."
            )

            distribution_shape = [int(d) for d in tensor.tensor_topology().distribution_shape()]
            assert (
                distribution_shape == expected_dist_shape
            ), f"key={key!r} distribution_shape={distribution_shape}, expected {expected_dist_shape}"

        # Smoke-check a few well-known keys so a silent rename / dispatch
        # error in export_to_hf_dict shows up here, not in update_weights
        # (when this test is later extended to also call update_weights).
        for required_key in ("model.embed_tokens.weight", "lm_head.weight"):
            assert required_key in hf_dict, f"missing required HF key {required_key!r} in received dict"

        print(f"[TTT rank {TTT_RANK}] all received tensors pass replicated/dtype/layout checks", flush=True)

        bridge.barrier()
        del hf_dict
        gc.collect()
    finally:
        completer = None
        gc.collect()
        ttnn.close_mesh_device(mesh_device)


def test_ttml_to_ttt_weight_bridge_transfer() -> None:
    """Option A: end-to-end bridge transfer + receiver-side metadata assertions.

    Branches on the MPI rank set by ``tt-run`` -- the same test function
    runs in both processes. The bridge's host-side manifest exchange and
    fabric ``send_async``/``recv_async`` synchronise the two paths.
    """
    if _MPI_RANK == TTML_RANK:
        _ttml_rank_send_state()
    elif _MPI_RANK == TTT_RANK:
        _ttt_rank_recv_state_and_assert()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {_MPI_RANK} (world_size={_WORLD_SIZE}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
