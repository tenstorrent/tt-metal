# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ttml -> tt-transformers RPC + weight transfer test (4 -> 4 submeshes).

Launched under tt-run with world_size == 2 (see ``runner.sh``), 4 chips per rank
(8 total -- the ``configurations/local8`` topology).

* **Rank 0 (TTML)** opens a ``[1, 4]`` DDP mesh from ``grpo_boolq_llama_1b_ddp_4dev.yaml``,
  loads the instruct ``LlamaCompositeKV``, and pushes its weights through a
  :class:`MPIRolloutClient` backed by a :class:`HostWeightBridge` sender.
* **Rank 1 (TTT)** opens a ``[1, 4]`` parent mesh, splits it into four ``[1, 1]``
  submeshes, builds one :class:`TttGenerationWorker` per submesh (dummy boot
  weights), and serves via a :class:`MPIRolloutServer` backed by a
  :class:`HostWeightBridge` receiver. The bridge moves each weight to host
  (``torch.save`` over MPI) and re-materialises it replicated onto every submesh;
  ``receive_weights()`` returns one dict per submesh, applied to that submesh's worker.

Flow (driven by the TTML rank): pre-push remote generate (dummy-weight gibberish, a
pure RPC sanity check on submesh 0) -> ``completer.push_weights()`` -> post-push
remote generate (all completions identical -> submesh 0's weights landed) ->
``client.shutdown()``. After ``serve_forever`` returns, the TTT rank generates the
same prompt on **all four** workers and asserts identical output -- proving every
submesh received correct weights.

Self-skips when not launched under tt-run (``OMPI_COMM_WORLD_SIZE`` unset or != 2).
Requires ``HF_TOKEN`` set (the instruct repo is gated).
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import Any, List

import pytest


_WORLD_SIZE = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
if _WORLD_SIZE != 2:
    pytest.skip(
        "test_weight_transfer must run under tt-run with world_size == 2 (use tests/weight_transfer/runner.sh).",
        allow_module_level=True,
    )

_MPI_RANK = int(os.environ["OMPI_COMM_WORLD_RANK"])

import ttnn  # noqa: E402

from utils.weight_bridge import TTML_RANK, TTT_RANK  # noqa: E402

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 64
VERIFY_NEW_TOKENS = 16
TEMPERATURE = 0.0

TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1b_ddp_4dev.yaml"
TTT_PARENT_MESH_SHAPE = (1, 4)
NUM_SUBMESHES = 4

# Post-push RPC batch (served by submesh 0). Each worker is built with this
# max_batch_size, so it must be >= POST_PUSH_BATCH.
POST_PUSH_BATCH = 8
TTT_MAX_BATCH_SIZE = 8
TTT_MAX_SEQ_LEN = 512


def _ttml_side() -> None:
    """Drive the flow: client handshake -> pre-push gen -> push_weights ->
    post-push gen (consistency) -> shutdown."""
    import ttml
    from transformers import AutoTokenizer
    from ttml.common.config import get_model_config

    from _completer_utils import close_device, load_device_config, open_device
    from utils.mpi_rollout import MPIRolloutClient
    from utils.weight_bridge import HostWeightBridge
    from utils.llama_grpo_completer import LlamaCompletionCtx, LlamaGRPOCompleter

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = load_device_config(TTML_DEVICE_CONFIG_REL)
    mesh_device = open_device(device_config)
    completer: Any = None
    client: Any = None
    try:
        # The caller builds the concrete bridge; the client only drives it.
        # Constructing the client blocks on the bridge handshake.
        bridge = HostWeightBridge.init_sender(mesh=mesh_device, peer_rank=TTT_RANK)
        client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)

        completer = LlamaGRPOCompleter(
            ctx=LlamaCompletionCtx(
                max_tokens_to_complete=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                # compute_nlog_probs is not exercised here; any value that would
                # satisfy its B % num_devices == 0 assertion is fine.
                completions_per_prompt=device_config.total_devices(),
            ),
            transformer_config=get_model_config(raw["training_config"]["model_config"]),
            mesh_device=mesh_device,
            model_source=MODEL_ID,
            inference_client=client,
            enable_ddp=device_config.enable_ddp,
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)

        # --- step 1: pre-push remote generate (dummy weights -> gibberish). --- #
        pre_push_ids = client.remote_generate(
            [prompt_ids],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )[0]
        print(
            f"[TTML rank {TTML_RANK}] (remote pre-push, dummy weights) ({len(pre_push_ids)} tok): "
            f"{tokenizer.decode(pre_push_ids, skip_special_tokens=False)!r}",
            flush=True,
        )

        # --- step 2: single-call weight transfer via the completer. --- #
        print(f"[TTML rank {TTML_RANK}] completer.push_weights()", flush=True)
        completer.push_weights()
        print(f"[TTML rank {TTML_RANK}] push_weights() complete", flush=True)

        # --- step 3: post-push remote generate, consistency check (submesh 0). --- #
        completions = client.remote_generate(
            [prompt_ids] * POST_PUSH_BATCH,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
        for i in range(POST_PUSH_BATCH):
            assert completions[i] == completions[0], (
                f"post-push completion {i} diverged from completion 0 "
                f"(len {len(completions[i])} vs {len(completions[0])})"
            )
        print(
            f"[TTML rank {TTML_RANK}] (remote post-push, instruct) ({len(completions[0])} tok): "
            f"{tokenizer.decode(completions[0], skip_special_tokens=False)!r}\n",
            flush=True,
        )

        client.shutdown()
    finally:
        completer = None
        gc.collect()
        close_device()


def _ttt_side() -> None:
    """Host four TttGenerationWorkers (one per [1, 1] submesh) + MPIRolloutServer."""
    from transformers import AutoTokenizer
    from utils.mpi_rollout import MPIRolloutServer
    from utils.weight_bridge import HostWeightBridge
    from utils.llama_ttt_presets import bf16_attn_bfp8_mlp_optimizations, llama_stop_and_pad
    from utils.ttt_generation_worker import TttGenerationWorker

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*TTT_PARENT_MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    # One [1, 1] submesh per device; submeshes[i] is parent coord (0, i).
    submeshes = parent_mesh.create_submeshes(ttnn.MeshShape(1, 1))
    assert len(submeshes) == NUM_SUBMESHES, f"expected {NUM_SUBMESHES} submeshes, got {len(submeshes)}"

    workers: List[Any] = []
    server: Any = None
    try:
        stop_token_ids, pad_token_id = llama_stop_and_pad(MODEL_ID)
        for i, submesh in enumerate(submeshes):
            print(f"[TTT rank {TTT_RANK}] building worker on submesh {i}", flush=True)
            workers.append(
                TttGenerationWorker(
                    mesh_device=submesh,
                    model_source=MODEL_ID,
                    max_batch_size=TTT_MAX_BATCH_SIZE,
                    max_seq_len=TTT_MAX_SEQ_LEN,
                    instruct=True,
                    optimizations=bf16_attn_bfp8_mlp_optimizations,
                    stop_token_ids=stop_token_ids,
                    pad_token_id=pad_token_id,
                    temperature=TEMPERATURE,
                    top_k=0,
                    top_p=1.0,
                    seed=0,
                )
            )

        def _on_weights_received(per_submesh: List[dict]) -> None:
            """Validate each submesh's dict against the update_weights contract, then apply."""
            assert len(per_submesh) == NUM_SUBMESHES, f"expected {NUM_SUBMESHES} dicts, got {len(per_submesh)}"
            for i, hf_dict in enumerate(per_submesh):
                for key, tensor in hf_dict.items():
                    assert tensor.dtype == ttnn.bfloat16, f"submesh {i} key={key!r} dtype={tensor.dtype}"
                    assert tensor.layout == ttnn.TILE_LAYOUT, f"submesh {i} key={key!r} layout={tensor.layout}"
                    assert (
                        tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
                    ), f"submesh {i} key={key!r} memcfg={tensor.memory_config()}"
                    shape = list(tensor.shape)
                    assert (
                        len(shape) == 4 and shape[0] == 1 and shape[1] == 1
                    ), f"submesh {i} key={key!r} expected 4D (1,1,*,*), got {shape}"
                for required_key in ("model.embed_tokens.weight", "lm_head.weight"):
                    assert required_key in hf_dict, f"submesh {i} missing required HF key {required_key!r}"
            print(f"[TTT rank {TTT_RANK}] received weights for {len(per_submesh)} submeshes; applying", flush=True)
            t0 = time.perf_counter()
            for i, (worker, hf_dict) in enumerate(zip(workers, per_submesh)):
                worker.update_weights(hf_dict)
            print(
                f"[TTT rank {TTT_RANK}] applied to all {len(workers)} workers in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )

        bridge = HostWeightBridge.init_receiver(mesh=parent_mesh, peer_rank=TTML_RANK, submeshes=submeshes)
        server = MPIRolloutServer(
            peer_rank=TTML_RANK,
            bridge=bridge,
            generate_fn=workers[0].generate,  # RPC generate is served by submesh 0
            on_weights_received=_on_weights_received,
        )
        server.serve_forever()

        # ---- verify all four submeshes got correct weights ---- #
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)
        print("\n========= per-submesh verification =========", flush=True)
        outs = []
        for i, worker in enumerate(workers):
            out = worker.generate([prompt_ids], max_new_tokens=VERIFY_NEW_TOKENS, temperature=TEMPERATURE)[0]
            outs.append(out)
            print(f"[submesh {i}] ({len(out)} tok) {tokenizer.decode(out, skip_special_tokens=False)!r}", flush=True)
        for i in range(1, NUM_SUBMESHES):
            assert outs[i] == outs[0], f"submesh {i} completion diverged from submesh 0 -> weights differ"
        print("all 4 submeshes produced identical output\n============================================\n", flush=True)
    finally:
        server = None
        workers = []
        gc.collect()
        ttnn.close_mesh_device(parent_mesh)


# Long, interactive, multi-model hardware test (HF download + four worker builds
# before the transfer). Disable the repo-wide pytest-timeout default.
@pytest.mark.timeout(0)
def test_ttml_to_ttt_weight_bridge_transfer() -> None:
    """End-to-end remote generate + 4->4 submesh bridge transfer + per-submesh verify."""
    if _MPI_RANK == TTML_RANK:
        _ttml_side()
    elif _MPI_RANK == TTT_RANK:
        _ttt_side()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {_MPI_RANK} (world_size={_WORLD_SIZE}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
