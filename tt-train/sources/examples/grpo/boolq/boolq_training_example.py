#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO training of Llama-3.2-1B-Instruct on BoolQ across two ranks.

Launched under ``tt-run`` with world_size == 2 (see :file:`runner.sh`).

Topology
========

The split width is chosen by the ``GRPO_BOOLQ_TOPOLOGY`` env var (set by
``runner.sh --topology``): ``2x2`` (default) or ``4x4``. Below, ``N`` is 2
or 4 accordingly.

* Rank 0 (TTML) opens the full ``[1, N]`` DDP mesh declared by the
  matching ``configurations/<dir>/mgd.textproto`` (``local4`` for 2x2,
  ``local8`` for 4x4). It owns the policy ttml ``Llama`` model and drives
  training via :class:`ttml.trainers.GRPOTrainer`.
* Rank 1 (TTT) opens a ``[1, N]`` parent mesh (its own boards) and splits
  it into ``N`` ``[1, 1]`` submeshes, one
  :class:`utils.ttt_generation_worker.TttGenerationWorker` per submesh.
  The weight bridge replicates each freshly-synced policy onto all
  submeshes; generate RPCs are served by submesh 0. Mirrors the
  topology in ``tests/weight_transfer/test_weight_transfer.py``.

The :class:`utils.mpi_rollout.MPIRolloutClient` constructor on
the ttml side and the :class:`utils.mpi_rollout.MPIRolloutServer`
constructor on the ttt side block until both have run -- the
``WeightBridge`` handshake inside their initialisers pins the two
ranks together before any RPC happens.

Lifecycle (TTML rank)
=====================

1. Build :class:`MPIRolloutClient` -- handshake completes once the
   worker rank also constructs its server.
2. Build :class:`LlamaGRPOCompleter`. Loads the real instruct tokenizer
   and ttml model into device memory.
3. Call ``completer.push_weights()`` once. This overwrites the worker's
   dummy boot weights with real instruct weights so the very first
   ``trainer.train()`` generate request returns coherent completions.
4. Construct :class:`GRPOTrainer` with
   :class:`utils.llama_grpo_completer.WeightSyncCallback`
   (``every=1``) so every gradient step also pushes the freshly
   updated policy weights to the worker.
5. ``trainer.train()`` runs the loop.
6. ``client.shutdown()`` releases the worker. **Must** run before the
   ttml device is closed -- the worker is blocked inside
   ``serve_forever()`` until it sees ``OP_SHUTDOWN``.

Lifecycle (TTT rank)
====================

1. Initialise ttnn distributed context, open a ``[1, N]`` parent mesh
   and split it into ``N`` ``[1, 1]`` submeshes.
2. Resolve stop / pad token IDs by briefly loading the HF tokenizer
   for ``MODEL_ID`` (see
   :func:`utils.llama_ttt_presets.llama_stop_and_pad`). The tokenizer
   is dropped immediately; the worker never holds one.
3. Build one :class:`TttGenerationWorker` per submesh --
   ``dummy_weights=True`` plus ``disable_disk_cache=True`` so boot is
   fast and the asymmetric ``[1, N] -> [1, 1]`` mesh handshake does not
   trip the disk-cache collective.
4. Build :class:`MPIRolloutServer`: ``generate`` is served by submesh
   0's worker, and each transferred weight dict is applied to its own
   submesh worker (the bridge replicates the same policy onto all four).
5. ``server.serve_forever()`` until ``OP_SHUTDOWN``.

Self-skips with a clear error if launched outside ``tt-run`` (world
size != 2). Requires ``HF_TOKEN`` set in the environment for the
initial HuggingFace download of the instruct model weights on the
ttml rank.
"""

from __future__ import annotations

import csv
import gc
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="ERROR")
logging.getLogger().setLevel(logging.ERROR)

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Make ``utils.*`` modules importable when this script is run directly
# (not under pytest's rootdir machinery): the launcher lives one level
# deeper than ``utils/`` so we insert the example root explicitly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import ttnn  # noqa: E402

# Pin fabric to FABRIC_2D on both ranks before any device opens. Without
# this, the TTT rank's open_mesh_device falls into DeviceManager's
# legacy auto-escalation to FABRIC_1D (see tt_metal/impl/device/device_manager.cpp),
# while TTML's ttml.core.distributed.enable_fabric(2) picks FABRIC_2D --
# the mismatch deadlocks the cross-rank fabric init collective. Mirrors
# tests/conftest.py's autouse _set_fabric_2d fixture.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

from utils.weight_bridge import TTML_RANK, TTT_RANK  # noqa: E402

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Topology selection. runner.sh exports GRPO_BOOLQ_TOPOLOGY (2x2 or 4x4) and
# pins the matching configurations/<dir> bindings; both ranks read the same
# value here so the ttml DDP mesh width, the ttt parent mesh, and the submesh
# count stay consistent:
#   * 2x2 -> ttml [1, 2] DDP mesh, ttt [1, 2] parent split into two [1, 1]
#            submeshes (4 chips total, configurations/local4).
#   * 4x4 -> ttml [1, 4] DDP mesh, ttt [1, 4] parent split into four [1, 1]
#            submeshes (8 chips total, configurations/local8).
# Defaults to 2x2 (the smaller, known-good split) when unset -- 4x4 currently
# hangs in the cross-rank handshake/transport on this host.
_TOPOLOGIES = {
    "2x2": {
        "ttml_device_config_rel": "tt-train/configs/training_configs/grpo_boolq_llama_1b_ddp_2dev.yaml",
        "ttt_parent_mesh_shape": (1, 2),
        "num_submeshes": 2,
    },
    "4x4": {
        "ttml_device_config_rel": "tt-train/configs/training_configs/grpo_boolq_llama_1b_ddp_4dev.yaml",
        "ttt_parent_mesh_shape": (1, 4),
        "num_submeshes": 4,
    },
}

TOPOLOGY = os.environ.get("GRPO_BOOLQ_TOPOLOGY", "2x2")
if TOPOLOGY not in _TOPOLOGIES:
    raise RuntimeError(
        f"Unknown GRPO_BOOLQ_TOPOLOGY={TOPOLOGY!r}; expected one of {sorted(_TOPOLOGIES)}. "
        "Select it via boolq/runner.sh --topology."
    )
_TOPO = _TOPOLOGIES[TOPOLOGY]

TTML_DEVICE_CONFIG_REL = _TOPO["ttml_device_config_rel"]
TTT_PARENT_MESH_SHAPE = _TOPO["ttt_parent_mesh_shape"]
NUM_SUBMESHES = _TOPO["num_submeshes"]

TTT_MAX_BATCH_SIZE = 32
TTT_MAX_SEQ_LEN = 2048
WEIGHT_SYNC_EVERY = 1

REPO_ROOT = Path(__file__).resolve().parents[5]


def _boolq_reward(completions, answer, **kwargs):
    rewards = []
    for text, ground_truth in zip(completions, answer):
        clean = text.strip().lower()
        accuracy = 2.0 if clean.startswith(ground_truth.lower()) else -1.0
        brevity = -0.1 * (len(text) / 20) ** 2
        rewards.append(accuracy + brevity)
    return rewards


def _run_output_dir() -> str:
    return os.path.join(
        str(REPO_ROOT),
        "generated/tt-train/grpo_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )


class GRPOMonitor:
    """on_step_end CSV/stdout monitor. Kept as a plain class because
    ``TrainerCallback`` only exposes a no-op default that we don't
    otherwise need here."""

    def __init__(self, output_dir: str) -> None:
        self.file_path = os.path.join(output_dir, "grpo_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(self.file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "reward", "avg_length", "step_time_s", "generation_time_s"])

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        reward = kwargs["reward_mean"]
        length = kwargs["mean_completion_len"]
        min_length = kwargs["min_completion_len"]
        max_length = kwargs["max_completion_len"]
        step_time_s = kwargs.get("step_time_s", float("nan"))
        generation_time_s = kwargs.get("generation_time_s", float("nan"))
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] Step {step} | Reward: {reward:.4f} "
            f"| Len: {length:.2f} (min {min_length}, max {max_length}) tokens "
            f"| Step: {step_time_s:.2f}s | Gen: {generation_time_s:.2f}s"
        )
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([step, reward, length, step_time_s, generation_time_s])

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        print("Training complete.")


def _load_device_config(device_config_rel: str = TTML_DEVICE_CONFIG_REL):
    from ttml.common.config import DeviceConfig, load_config

    raw = load_config(os.path.join(str(REPO_ROOT), device_config_rel))
    return DeviceConfig(raw), raw


def _open_ttml_device(device_config) -> Any:
    import ttml

    if device_config.total_devices() > 1:
        ttml.core.distributed.enable_fabric(device_config.total_devices())
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
    return autograd_ctx.get_device()


def _close_ttml_device() -> None:
    import ttml

    ttml.autograd.AutoContext.get_instance().close_device()


def _ttml_main() -> None:
    import ttml
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from ttml.common.config import get_model_config
    from ttml.trainers import GRPOTrainer, get_grpo_config
    from utils.llama_grpo_completer import (
        LlamaCompletionCtx,
        LlamaGRPOCompleter,
        WeightSyncCallback,
    )
    from utils.mpi_rollout import MPIRolloutClient
    from utils.weight_bridge import HostWeightBridge

    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = _load_device_config(TTML_DEVICE_CONFIG_REL)
    mesh_device = _open_ttml_device(device_config)

    completer: Any = None
    client: Any = None
    try:
        # The caller builds the concrete bridge; constructing the client blocks
        # on its handshake -- returns once the ttt rank has also constructed its
        # MPIRolloutServer.
        bridge = HostWeightBridge.init_sender(mesh=mesh_device, peer_rank=TTT_RANK)
        client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)

        # ------------------------------------------------------------ #
        # Dataset + GRPO config                                         #
        # ------------------------------------------------------------ #
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        system_prompt = "You are a wordy professor. Explain in 3 long sentences before saying Yes or No."

        def format_boolq(example):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {example['question']}? Context: {example['passage']}"},
            ]
            return {
                "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                "answer": "yes" if example["answer"] else "no",
            }

        dataset = load_dataset("google/boolq", split="train").shuffle(seed=42).map(format_boolq)

        output_dir = _run_output_dir()
        grpo_config = get_grpo_config(raw, output_dir=output_dir)
        optimizer_dict = raw["training_config"]["optimizer"]
        transformer_config = get_model_config(raw["training_config"]["model_config"])

        # ------------------------------------------------------------ #
        # Completer                                                     #
        # ------------------------------------------------------------ #
        completer = LlamaGRPOCompleter(
            ctx=LlamaCompletionCtx(
                max_tokens_to_complete=grpo_config.max_completion_length,
                temperature=grpo_config.temperature,
                completions_per_prompt=grpo_config.num_generations,
            ),
            transformer_config=transformer_config,
            mesh_device=mesh_device,
            model_source=MODEL_ID,
            inference_client=client,
            enable_ddp=device_config.enable_ddp,
        )

        # Initial weight push: replace the worker's dummy boot weights
        # with real instruct weights BEFORE training kicks off.
        completer.push_weights()

        trainer = GRPOTrainer(
            completer=completer,
            dataset=dataset,
            config=grpo_config,
            reward_func=_boolq_reward,
            optimizer_dict=optimizer_dict,
            callbacks=[
                GRPOMonitor(output_dir),
                WeightSyncCallback(completer, every=WEIGHT_SYNC_EVERY),
            ],
            model_source=MODEL_ID,
        )
        trainer.train()
    finally:
        # Shutdown ordering: tell the server to exit BEFORE we drop the
        # completer or close the mesh. The worker is otherwise still
        # blocked in serve_forever() and MPI would never tear down cleanly.
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001 -- best-effort during teardown
                pass
        completer = None
        gc.collect()
        _close_ttml_device()


# ---------------------------------------------------------------------------
# TTT rank entrypoint
# ---------------------------------------------------------------------------


def _ttt_main() -> None:
    from ttml.common.config import load_config
    from utils.llama_ttt_presets import (
        bf16_attn_bfp8_mlp_optimizations,
        llama_stop_and_pad,
    )
    from utils.mpi_rollout import MPIRolloutServer
    from utils.ttt_generation_worker import TttGenerationWorker
    from utils.weight_bridge import HostWeightBridge

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*TTT_PARENT_MESH_SHAPE),
        offset=ttnn.MeshCoordinate(0, 0),
    )
    # One [1, 1] submesh per chip; submeshes[i] is parent coord (0, i).
    submeshes = parent_mesh.create_submeshes(ttnn.MeshShape(1, 1))
    assert len(submeshes) == NUM_SUBMESHES, f"expected {NUM_SUBMESHES} submeshes, got {len(submeshes)}"

    # Read the same yaml as the ttml rank to pick up the GRPO sampling
    # temperature. The worker bakes (temperature, top_k, top_p, seed) into
    # the captured decode trace at construction; the ttml-side completer
    # forwards the same temperature value via remote_generate(), so the
    # two stay consistent as long as both ranks read it from this file.
    raw = load_config(os.path.join(str(REPO_ROOT), TTML_DEVICE_CONFIG_REL))
    grpo_temperature = float(raw["training_config"]["grpo_config"]["temperature"])

    workers: list[Any] = []
    server: Any = None
    try:
        # Tokenizer load is launcher-local: we extract stop/pad IDs and
        # drop the reference before the workers are built.
        stop_token_ids, pad_token_id = llama_stop_and_pad(MODEL_ID)

        for submesh in submeshes:
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
                    temperature=grpo_temperature,
                    top_k=0,
                    top_p=1.0,
                    seed=None,
                )
            )

        # The bridge replicates each transferred policy onto every submesh, so
        # receive_weights() returns one dict per submesh; apply each to its own
        # worker. Generate RPCs are served by submesh 0's worker.
        bridge = HostWeightBridge.init_receiver(mesh=parent_mesh, peer_rank=TTML_RANK, submeshes=submeshes)

        def _on_weights_received(per_submesh: list) -> None:
            for worker, hf_dict in zip(workers, per_submesh):
                worker.update_weights(hf_dict)

        server = MPIRolloutServer(
            peer_rank=TTML_RANK,
            bridge=bridge,
            generate_fn=workers[0].generate,
            on_weights_received=_on_weights_received,
        )
        server.serve_forever()
    finally:
        workers = []
        server = None
        gc.collect()
        ttnn.close_mesh_device(parent_mesh)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0"))
    if world_size != 2:
        raise RuntimeError(
            f"boolq_training_example must run under tt-run with world_size == 2 (got {world_size}). "
            "Use boolq/runner.sh."
        )

    rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    if rank == TTML_RANK:
        _ttml_main()
    elif rank == TTT_RANK:
        _ttt_main()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {rank} (world_size={world_size}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
