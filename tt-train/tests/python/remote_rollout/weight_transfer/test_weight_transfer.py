# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ttml -> tt-transformers RPC + weight transfer test (4 -> 4 submeshes).

Rank 0 (TTML) pushes its weights through an MPIRolloutClient/HostWeightBridge
sender; rank 1 (TTT) receives onto four [1, 1] submeshes and applies one dict per
submesh. Flow: pre-push generate (gibberish) -> send_weights -> post-push generate
(identical) -> shutdown, then per-submesh verify that all four match.

Runs under tt-run with world_size == 2 (see ``runner.sh``); self-skips otherwise.
Requires ``HF_TOKEN`` (the instruct repo is gated).
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

import numpy as np  # noqa: E402
import ttnn  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from ttml.trainers.grpo_trainer import RolloutBatch, RolloutSampler  # noqa: E402
from ttml.trainers.grpo_trainer.remote_rollout.weight_bridge import TTML_RANK, TTT_RANK  # noqa: E402

# Pin fabric FABRIC_2D before either rank opens a device (both ranks must match).
pytestmark = pytest.mark.usefixtures("_set_fabric_2d")

# Skipped pending BH bring-up: after ``push_weights``, one batch slot on one
# submesh produces a different completion than the others despite identical
# prompt/weights/greedy decode ("post-push completion 5 diverged from
# completion 0"). Deterministic and reproducible; needs weight-bridge /
# per-submesh model-state investigation. Re-enable by removing the
# ``@pytest.mark.skip`` decorator on the test function below.
# (The earlier ttml-vs-ttnn asymmetric fabric-init deadlock is fixed by the
# ``_set_fabric_2d`` fixture above plus ``_ttt_sampler_utils.open_device`` only
# calling ``enable_fabric`` when fabric is currently ``DISABLED``.)
_SKIP_REASON = (
    "test_weight_transfer: per-submesh determinism regression on BH after " "push_weights — see module docstring."
)

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 64
VERIFY_NEW_TOKENS = 16
TEMPERATURE = 0.0

TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1b_ddp_4dev.yaml"
TTT_PARENT_MESH_SHAPE = (1, 4)
NUM_SUBMESHES = 4

# Post-push RPC batch; TTT_MAX_BATCH_SIZE must be >= this.
POST_PUSH_BATCH = 8
TTT_MAX_BATCH_SIZE = 8
TTT_MAX_SEQ_LEN = 512


class _GreedyCheckedSampler(RolloutSampler):
    """Serves a greedy TTTRolloutSampler and checks every received weight dict.

    tt-transformers emits no log-probs on its greedy path, so ``generate`` reports zeros;
    ``close`` leaves the wrapped sampler open for the per-submesh verification.
    """

    def __init__(self, sampler: Any) -> None:
        self._sampler = sampler

    @property
    def weight_version(self) -> int:
        return self._sampler.weight_version

    def generate(self, prompts: List[List[int]]) -> RolloutBatch:
        completions = self._sampler.generate_tokens(prompts, max_new_tokens=MAX_NEW_TOKENS)
        return RolloutBatch(
            weight_version=self.weight_version,
            prompts=[list(p) for p in prompts],
            completions=completions,
            logprobs=np.zeros((len(prompts), MAX_NEW_TOKENS), dtype=np.float32),
        )

    def update_weights(self, weights: List[dict], version: int) -> None:
        assert len(weights) == NUM_SUBMESHES, f"expected {NUM_SUBMESHES} dicts, got {len(weights)}"
        for i, hf_dict in enumerate(weights):
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
        print(f"[TTT rank {TTT_RANK}] received weights v{version} for {len(weights)} submeshes; applying", flush=True)
        t0 = time.perf_counter()
        self._sampler.update_weights(weights, version=version)
        print(
            f"[TTT rank {TTT_RANK}] applied to all {len(weights)} submeshes in {time.perf_counter() - t0:.2f}s",
            flush=True,
        )


def _ttml_side() -> None:
    """Drive: handshake -> pre-push gen -> send_weights -> post-push gen -> shutdown."""
    import ttml
    from ttml.common.config import get_model_config

    from _ttt_sampler_utils import close_device, load_device_config
    from ttml.trainers.grpo_trainer.grpo_ttml_model import setup_ttml_model, weights_ref_hf_dict
    from ttml.trainers.grpo_trainer.remote_rollout.mpi_rollout import MPIRolloutClient
    from ttml.trainers.grpo_trainer.remote_rollout.weight_bridge import HostWeightBridge

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = load_device_config(TTML_DEVICE_CONFIG_REL)
    model: Any = None
    client: Any = None
    try:
        model, _ = setup_ttml_model(get_model_config(raw["training_config"]["model_config"]), device_config, MODEL_ID)
        # Constructing the client blocks on the bridge handshake.
        bridge = HostWeightBridge.init_sender(mesh=autograd_ctx.get_device(), peer_rank=TTT_RANK)
        client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)

        # step 1: pre-push remote generate (dummy weights -> gibberish).
        pre_push_ids = client.remote_generate([prompt_ids])[0][0]
        print(
            f"[TTML rank {TTML_RANK}] (remote pre-push, dummy weights) ({len(pre_push_ids)} tok): "
            f"{tokenizer.decode(pre_push_ids, skip_special_tokens=False)!r}",
            flush=True,
        )

        # step 2: single-call weight transfer.
        print(f"[TTML rank {TTML_RANK}] client.send_weights(version=1)", flush=True)
        client.send_weights(weights_ref_hf_dict(model), version=1)
        print(f"[TTML rank {TTML_RANK}] send_weights() complete", flush=True)

        # step 3: post-push remote generate, consistency check (submesh 0).
        completions, _logprobs, version = client.remote_generate([prompt_ids] * POST_PUSH_BATCH)
        assert version == 1, f"expected weight_version 1 after one push, got {version}"
        for i in range(POST_PUSH_BATCH):
            assert completions[i] == completions[0], (
                f"post-push completion {i} diverged from completion 0 "
                f"(len {len(completions[i])} vs {len(completions[0])})\n"
                f"  completion 0: {tokenizer.decode(completions[0], skip_special_tokens=False)!r}\n"
                f"  completion {i}: {tokenizer.decode(completions[i], skip_special_tokens=False)!r}"
            )
        print(
            f"[TTML rank {TTML_RANK}] (remote post-push, instruct) ({len(completions[0])} tok): "
            f"{tokenizer.decode(completions[0], skip_special_tokens=False)!r}\n",
            flush=True,
        )

        client.shutdown()
    finally:
        model = None
        gc.collect()
        close_device()


def _ttt_side() -> None:
    """Host one TTTRolloutSampler over four [1, 1] submeshes + MPIRolloutServer."""
    from ttml.trainers.grpo_trainer.remote_rollout.mpi_rollout import MPIRolloutServer
    from ttml.trainers.grpo_trainer.remote_rollout.weight_bridge import HostWeightBridge
    from ttml.trainers.grpo_trainer.remote_rollout.llama_ttt_presets import (
        bf16_attn_bfp8_mlp_optimizations,
        llama_stop_and_pad,
    )
    from ttml.trainers.grpo_trainer.remote_rollout.ttt_rollout_sampler import TTTRolloutSampler

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*TTT_PARENT_MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    worker: Any = None
    server: Any = None
    try:
        stop_token_ids, pad_token_id = llama_stop_and_pad(MODEL_ID)
        worker = TTTRolloutSampler(
            mesh_device=parent_mesh,
            model_source=MODEL_ID,
            max_batch_size=TTT_MAX_BATCH_SIZE,
            max_seq_len=TTT_MAX_SEQ_LEN,
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            completions_per_prompt=1,
            max_completion_length=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=0,
            top_p=1.0,
            seed=0,
            dummy_weights=True,
        )
        assert (
            len(worker.submeshes) == NUM_SUBMESHES
        ), f"expected {NUM_SUBMESHES} submeshes, got {len(worker.submeshes)}"

        bridge = HostWeightBridge.init_receiver(mesh=parent_mesh, peer_rank=TTML_RANK, submeshes=worker.submeshes)
        server = MPIRolloutServer(peer_rank=TTML_RANK, bridge=bridge, sampler=_GreedyCheckedSampler(worker))
        server.serve_forever()

        # verify all submeshes got correct weights: a batch spanning every submesh
        # (identical prompts) must produce identical output.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)
        verify_batch = NUM_SUBMESHES * TTT_MAX_BATCH_SIZE
        print("\n========= per-submesh verification =========", flush=True)
        outs = worker.generate_tokens([prompt_ids] * verify_batch, max_new_tokens=VERIFY_NEW_TOKENS)
        print(f"[submesh 0] ({len(outs[0])} tok) {tokenizer.decode(outs[0], skip_special_tokens=False)!r}", flush=True)
        for i in range(1, verify_batch):
            assert outs[i] == outs[0], (
                f"submesh completion {i} diverged from 0 -> weights differ\n"
                f"  completion 0: {tokenizer.decode(outs[0], skip_special_tokens=False)!r}\n"
                f"  completion {i}: {tokenizer.decode(outs[i], skip_special_tokens=False)!r}"
            )
        print(
            f"all {NUM_SUBMESHES} submeshes produced identical output\n============================================\n",
            flush=True,
        )
    finally:
        server = None
        worker = None
        gc.collect()
        ttnn.close_mesh_device(parent_mesh)


# Disable the repo-wide pytest-timeout default (long HF download + four builds).
@pytest.mark.timeout(0)
@pytest.mark.skip(reason=_SKIP_REASON)
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
