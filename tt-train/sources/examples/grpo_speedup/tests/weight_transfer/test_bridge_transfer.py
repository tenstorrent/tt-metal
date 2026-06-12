# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ttml -> tt-transformers RPC + weight transfer test.

Launched under tt-run with world_size == 2 (see ``runner.sh``). Rank 0
(TTML) opens a ``[1, 2]`` mesh from
``grpo_boolq_llama_2dev_ddp.yaml``. Rank 1 (TTT) opens a ``[1, 1]``
sub-mesh of the declared ``[1, 2]`` board.

The TTML rank drives the whole flow:

  1. Construct :class:`TttInferenceClient` -- WeightBridge handshake
     blocks until the TTT rank constructs its :class:`TttInferenceServer`.
  2. Construct :class:`LlamaGRPOCompleter` (real instruct weights on
     ttml).
  3. Pre-push remote generate: the TTT worker is still on dummy boot
     weights, so the output is gibberish but deterministic with
     ``temperature=0.0``. This is a pure RPC sanity check.
  4. ``completer.push_weights()`` -- single user-side call that ships
     the ttml-side instruct weights to the worker.
  5. Post-push remote generate (10x the same prompt) -- with real
     instruct weights now in the worker, the 10 completions must all
     be identical (greedy decode is deterministic).
  6. ``client.shutdown()`` releases the worker.

The TTT rank builds a :class:`TttGenerationWorker` (``dummy_weights=True``,
``disable_disk_cache=True``) and a :class:`TttInferenceServer` whose
``generate_fn`` and ``on_weights_received`` callbacks are wired to the
worker. The ``on_weights_received`` wrapper also asserts the received
HF dict matches the replicated / DRAM / TILE / bfloat16 contract that
:meth:`Transformer.update_weights` documents -- regression coverage for
the bridge's invariants.

Both :class:`TttInferenceClient` and :class:`TttInferenceServer`
perform the underlying ``WeightBridge`` handshake inside their own
``__init__``; returning from the constructor on one rank therefore
implies the peer has also constructed its counterpart (or, equivalently,
will eventually do so -- the constructor blocks).

Self-skips when not launched under tt-run (``OMPI_COMM_WORLD_SIZE``
unset or != 2). Requires ``HF_TOKEN`` set in the environment.
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
        "test_bridge_transfer must run under tt-run with world_size == 2 (use tests/weight_transfer/runner.sh).",
        allow_module_level=True,
    )

_MPI_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])

import ttnn  # noqa: E402

from utils.inference_bridge import TTML_RANK, TTT_RANK  # noqa: E402

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0

TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_2dev_ddp.yaml"
TTT_MESH_SHAPE = (1, 1)

TTT_MAX_BATCH_SIZE = 32
TTT_MAX_SEQ_LEN = 2048


def _ttml_side() -> None:
    """Drive the TTT rank: client handshake -> pre-push remote gen
    (dummy-weight gibberish) -> push_weights -> post-push remote gen
    (real instruct output, 10x consistency check) -> shutdown."""
    import ttml
    from transformers import AutoTokenizer
    from ttml.common.config import get_model_config

    from _completer_utils import close_device, load_device_config, open_device
    from utils.inference_bridge import TttInferenceClient
    from utils.llama_grpo_completer import LlamaCompletionCtx, LlamaGRPOCompleter

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = load_device_config(TTML_DEVICE_CONFIG_REL)
    mesh_device = open_device(device_config)
    completer: Any = None
    client: Any = None
    try:
        # Constructor blocks on the WeightBridge handshake.
        client = TttInferenceClient(peer_rank=TTT_RANK, device=mesh_device)

        completer = LlamaGRPOCompleter(
            ctx=LlamaCompletionCtx(
                max_tokens_to_complete=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                # On a [1, 2] mesh, compute_nlog_probs asserts
                # ``B % num_devices == 0``; for this test we don't call
                # compute_nlog_probs but the field is still part of the
                # ctx so we pick a value that would pass.
                completions_per_prompt=device_config.total_devices(),
            ),
            transformer_config=get_model_config(raw["training_config"]["model_config"]),
            mesh_device=mesh_device,
            model_source=MODEL_ID,
            inference_client=client,
            enable_ddp=device_config.enable_ddp,
        )

        # Tokenise the prompt locally; the wire only carries IDs.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)

        # --- step 1: pre-push remote generate (worker has dummy boot
        # weights; output is gibberish but deterministic). ----------- #
        pre_push_ids = client.remote_generate(
            [prompt_ids],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )[0]
        pre_push_text = tokenizer.decode(pre_push_ids, skip_special_tokens=False)
        print(
            f"[TTML rank {TTML_RANK}] (remote pre-push, dummy weights) ({len(pre_push_ids)} tok): {pre_push_text!r}",
            flush=True,
        )

        # --- step 2: single-call weight transfer via the completer --- #
        print(f"[TTML rank {TTML_RANK}] completer.push_weights()", flush=True)
        completer.push_weights()
        print(f"[TTML rank {TTML_RANK}] push_weights() complete", flush=True)

        # --- step 3: post-push remote generate, 10x consistency ----- #
        completions = client.remote_generate(
            [prompt_ids] * 10,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )

        for i in range(10):
            assert completions[i] == completions[0], (
                f"post-push completion {i} diverged from completion 0: "
                f"len {len(completions[i])} vs {len(completions[0])}"
            )

        post_push_text = tokenizer.decode(completions[0], skip_special_tokens=False)
        print(
            f"[TTML rank {TTML_RANK}] (remote post-push, instruct) ({len(completions[0])} tok): {post_push_text!r}\n",
            flush=True,
        )

        client.shutdown()
    finally:
        completer = None
        gc.collect()
        close_device()


def _ttt_side() -> None:
    """Host the TttGenerationWorker + TttInferenceServer. Boots with
    dummy weights; the first OP_REQUEST_TRANSFER from the ttml peer
    overwrites them with real instruct weights."""
    from utils.inference_bridge import TttInferenceServer
    from utils.llama_ttt_presets import (
        bf16_attn_bfp8_mlp_optimizations,
        llama_stop_and_pad,
    )
    from utils.ttt_generation_worker import TttGenerationWorker

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*TTT_MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    worker: Any = None
    server: Any = None
    try:
        # Launcher-local tokenizer load: extract stop/pad IDs and drop.
        stop_token_ids, pad_token_id = llama_stop_and_pad(MODEL_ID)

        worker = TttGenerationWorker(
            mesh_device=mesh_device,
            model_source=MODEL_ID,
            max_batch_size=TTT_MAX_BATCH_SIZE,
            max_seq_len=TTT_MAX_SEQ_LEN,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            # On-device sampling baked into the decode trace at first
            # call. temperature=0.0 is folded into (temp=1.0, top_k=1)
            # inside format_sampling_params, i.e. pure argmax, making
            # the 10x identical-completions assertion below deterministic.
            temperature=TEMPERATURE,
            top_k=0,
            top_p=1.0,
            seed=0,
        )

        def _on_weights_received(hf_dict: dict) -> None:
            """Validate the received dict matches the
            Transformer.update_weights contract, then apply it."""
            print(
                f"[TTT rank {TTT_RANK}] received {len(hf_dict)} weight tensors via WeightBridge",
                flush=True,
            )

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

            for required_key in ("model.embed_tokens.weight", "lm_head.weight"):
                assert required_key in hf_dict, f"missing required HF key {required_key!r} in received dict"

            print(f"[TTT rank {TTT_RANK}] all received tensors pass replicated/dtype/layout checks", flush=True)

            t0 = time.perf_counter()
            worker.update_weights(hf_dict)
            print(
                f"[TTT rank {TTT_RANK}] update_weights complete in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )

        server = TttInferenceServer(
            peer_rank=TTML_RANK,
            device=mesh_device,
            generate_fn=worker.generate,
            on_weights_received=_on_weights_received,
        )
        server.serve_forever()
    finally:
        worker = None
        server = None
        gc.collect()
        ttnn.close_mesh_device(mesh_device)


def test_ttml_to_ttt_weight_bridge_transfer() -> None:
    """End-to-end remote generate + bridge transfer + remote generate."""
    if _MPI_RANK == TTML_RANK:
        _ttml_side()
    elif _MPI_RANK == TTT_RANK:
        _ttt_side()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {_MPI_RANK} (world_size={_WORLD_SIZE}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
