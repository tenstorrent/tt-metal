# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Multi-process ttml -> tt-transformers weight bridge + RPC test.

Launched under tt-run with world_size == 2 (see ``runner.sh``). Rank 0
(TTML) opens a ``[1, 2]`` mesh from
``grpo_boolq_llama_2dev_ddp.yaml``. Rank 1 (TTT) opens a ``[1, 1]``
sub-mesh of the declared ``[1, 2]`` board.

The TTML rank drives the whole flow via :class:`TttInferenceClient`:

  1. local instruct sanity generate (ttml side)
  2. remote pre-update generate on the TTT rank (RPC)
  3. ``client.transfer_weights(hf_dict)`` to push the ttml weights into
     the TTT base model in a single call
  4. remote post-update generate on the TTT rank (RPC)
  5. ``client.shutdown()`` to release the server

The TTT rank constructs a :class:`TttInferenceServer` with two
callbacks -- a ``generate_fn`` that wraps ``LlamaCompleterTtt.generate``
and an ``on_weights_received`` that validates the received HF dict and
calls ``Transformer.update_weights`` -- and then runs
``server.serve_forever()`` once.

Both :class:`TttInferenceClient` and :class:`TttInferenceServer`
perform the underlying ``WeightBridge`` handshake inside their own
``__init__``; returning from the constructor on one rank therefore
implies the peer has also constructed its counterpart (or, equivalently,
will eventually do so -- the constructor blocks).

The asymmetric ``[1, 2] -> [1, 1]`` transfer is handled by the
``WeightBridge`` that each inference object composes internally -- see
``utils/inference_bridge.py`` for the protocol. The inference RPC
piggybacks on the same MPI distributed context but uses disjoint tags
(10..13).

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

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
INSTRUCT_MODEL_ID = "meta-llama/Llama-3.2-1B-instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0

TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_2dev_ddp.yaml"
TTT_MESH_SHAPE = (1, 1)


def _ttml_side() -> None:
    """Drive the TTT rank: instruct sanity gen -> remote base gen -> weight
    transfer -> remote post-update gen -> shutdown."""
    import ttml
    from transformers import AutoTokenizer
    from ttml.common.config import get_model_config

    from _completer_utils import close_device, load_device_config, open_device
    from utils.inference_bridge import TttInferenceClient
    from utils.llama_completer_ttml import LlamaCompleterTtml, LlamaCompletionCtx

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = load_device_config(TTML_DEVICE_CONFIG_REL)
    mesh_device = open_device(device_config)
    completer: Any = None
    try:
        ctx = LlamaCompletionCtx(
            max_tokens_to_complete=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            # LlamaCompleterTtml asserts ``B % num_devices == 0``, so on a
            # [1, 2] mesh we need at least 2 completions per prompt.
            completions_per_prompt=device_config.total_devices(),
        )
        completer = LlamaCompleterTtml(
            ctx=ctx,
            transformer_config=get_model_config(raw["training_config"]["model_config"]),
            mesh_device=mesh_device,
            model_source=INSTRUCT_MODEL_ID,
            enable_ddp=device_config.enable_ddp,
        )

        # --- step 1: instruct sanity generate, locally on ttml ---
        prompt_ids = completer.tokenizer.encode(PROMPT, add_special_tokens=True)
        completion_ids = [int(t) for t in completer.generate([prompt_ids])[0]]
        completion_text = completer.tokenizer.decode(completion_ids, skip_special_tokens=False)
        print(
            f"[TTML rank {TTML_RANK}] (local instruct) ({len(completion_ids)} tok): {completion_text!r}",
            flush=True,
        )

        # Base tokenizer for prompts we ship to the TTT side: keeps the
        # special-token handling aligned with the base model that lives on
        # the TTT rank, so the same wire prompt produces the same IDs both
        # before and after the weight transfer.
        base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        base_prompt_ids = base_tokenizer.encode(PROMPT, add_special_tokens=True)

        # Construct the client. TttInferenceClient.__init__ runs the
        # WeightBridge handshake internally, so this blocks until the
        # TTT rank constructs its TttInferenceServer.
        client = TttInferenceClient(peer_rank=TTT_RANK, device=mesh_device)

        # --- step 2: pre-update generate on TTT, driven via RPC ---
        pre_update_ids = client.remote_generate(
            [base_prompt_ids],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )[0]
        pre_update_text = base_tokenizer.decode(pre_update_ids, skip_special_tokens=False)
        print(
            f"[TTML rank {TTML_RANK}] (remote TTT pre-update) ({len(pre_update_ids)} tok): {pre_update_text!r}",
            flush=True,
        )

        # --- step 3: weight transfer (single call from user perspective) ---
        hf_dict = completer.model.export_to_hf_dict()
        print(
            f"[TTML rank {TTML_RANK}] sending {len(hf_dict)} weight tensors via TttInferenceClient.transfer_weights",
            flush=True,
        )
        client.transfer_weights(hf_dict)
        print(f"[TTML rank {TTML_RANK}] weight transfer complete", flush=True)
        del hf_dict
        gc.collect()

        # --- step 4: post-update generate on TTT, driven via RPC ---
        post_update_ids = client.remote_generate(
            [base_prompt_ids],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )[0]
        post_update_text = base_tokenizer.decode(post_update_ids, skip_special_tokens=False)
        print(
            f"[TTML rank {TTML_RANK}] (remote TTT post-update) ({len(post_update_ids)} tok): {post_update_text!r}",
            flush=True,
        )

        client.shutdown()
    finally:
        completer = None
        gc.collect()
        close_device()


def _ttt_side() -> None:
    """Serve the TTML rank's RPC: generate requests + a single weight
    transfer triggered by OP_REQUEST_TRANSFER, then return on shutdown."""
    from utils.inference_bridge import TttInferenceServer
    from utils.llama_completer_ttt import LlamaCompleterTtt

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*TTT_MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    completer: Any = None
    try:
        # disable_disk_cache short-circuits the dump_tensor_flatbuffer
        # collective that would otherwise deadlock against the asymmetric
        # TTML rank on MPI_COMM_WORLD.
        completer = LlamaCompleterTtt(
            mesh_device=mesh_device,
            model_source=BASE_MODEL_ID,
            max_batch_size=1,
            instruct=False,
            disable_disk_cache=True,
        )

        def _generate_fn(
            prompts: list,
            *,
            max_new_tokens: int,
            temperature: float,
            top_p: float,
            seed: Any,
        ) -> list:
            return completer.generate(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
            )

        def _on_weights_received(hf_dict: dict) -> None:
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
            completer.model.update_weights(hf_dict)
            print(
                f"[TTT rank {TTT_RANK}] update_weights complete in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )

        # Construct the server. TttInferenceServer.__init__ runs the
        # WeightBridge handshake internally, so this blocks until the
        # TTML rank constructs its TttInferenceClient.
        server = TttInferenceServer(
            peer_rank=TTML_RANK,
            device=mesh_device,
            generate_fn=_generate_fn,
            on_weights_received=_on_weights_received,
        )
        server.serve_forever()
    finally:
        completer = None
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
