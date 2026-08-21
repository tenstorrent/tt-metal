#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO on the reverse-text task, async-rollout variant.

The training-only counterpart of ``examples/grpo/reverse_text``:

  - Rank 0 (TTML): owns the ttml Qwen3 policy and drives ``GRPOTrainer``.
    ``compute_nlog_probs`` runs locally on this rank; ``generate`` is an MPI
    call to rank 1 via :class:`MPIRolloutClient`.
  - Rank 1 (TTT): runs :class:`TttGenerationWorker` on a tt-transformers Qwen3
    with on-device sampling (unlocked for Qwen3 by the PR #53167 series --
    N-way vocab split, per-request seed salting).

Weight sync: after every training step, :meth:`WeightSyncCallback.on_step_end`
exports the ttml policy via :func:`qwen3_weights_ref_hf_dict` and ships it
through the host bridge to the ttt worker. Both stacks store Qwen3 Q/K rows in
the same interleaved layout (see the docstring of ``qwen3_overrides``), so the
consumer uses ``hf_rope=False`` with no additional permutation.

The main-branch example ran greedy eval each step by flipping ``ctx.temperature``
to ``0.0``. That works when generation is on the ttml completer's own sample op;
here generation is remote and the worker bakes ``temperature`` into its decode
trace at first capture, so a per-call flip has no effect on the actual token
stream. This example therefore skips the greedy eval callback and relies on the
training-temperature reward that :func:`similarity_reward` logs each step.
Re-enabling greedy eval would need the worker to re-capture the decode trace
with the new temperature, which is a separate change.

Run:
    tt-train/sources/examples/grpo_remote_rollout/reverse_text/runner.sh
"""

from __future__ import annotations

import gc
import logging
import os
import re
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

# Make ``utils.*`` importable when the file is run directly (needed before
# any ``from utils.* import ...`` at module scope).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLE_ROOT = os.path.dirname(_THIS_DIR)
if _EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLE_ROOT)

import ttml
import ttnn
from datasets import load_dataset
from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, get_model_config, load_config
from ttml.trainers import GRPOTrainer, get_grpo_config
from utils.mpi_rollout import MPIRolloutClient, MPIRolloutServer
from utils.qwen3_grpo_completer import Qwen3CompleterRemoteRollout, Qwen3CompletionCtx
from utils.qwen3_ttt_presets import bf16_attn_bfp8_mlp_optimizations, qwen3_stop_and_pad
from utils.ttt_generation_worker import TttGenerationWorker
from utils.weight_bridge import HostWeightBridge, TTML_RANK, TTT_RANK

CONFIG_REL = "tt-train/configs/training_configs/grpo_reverse_text_qwen3_p6b_remote_rollout.yaml"

REPO_ROOT = Path(__file__).resolve().parents[5]

DATASET = "PrimeIntellect/Reverse-Text-RL"
DATASET_SPLIT = "train"

SYSTEM_PROMPT = "Reverse the text character-by-character. Put your answer in <reversed_text> tags."

TAG_RE = re.compile(r"<reversed_text>(.*?)</reversed_text>", re.DOTALL)

# Pin fabric config before either _ttml_main or _ttt_main opens a device.
# Otherwise TTT's open_mesh_device auto-escalates to FABRIC_1D and the
# mismatch deadlocks the cross-rank fabric init.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)


def parse_reversed_text(completion: str) -> str:
    """Return the last ``<reversed_text>`` block, or "" when the format is missing."""
    matches = TAG_RE.findall(completion)
    return matches[-1].strip() if matches else ""


def similarity_reward(completions, answer, **kwargs):
    """``difflib`` similarity ratio between the parsed answer and the truth.

    Same shape as the on-main reverse-text example: the ratio is the weighted
    reward; exact-match and format rates are logged as diagnostics.
    """
    parsed = [parse_reversed_text(c) for c in completions]
    rewards = [SequenceMatcher(None, got, truth).ratio() for got, truth in zip(parsed, answer)]

    n = max(len(rewards), 1)
    frac_exact = sum(1.0 for got, truth in zip(parsed, answer) if got == truth) / n
    frac_format = sum(1.0 for got in parsed if got) / n
    logging.info(
        "[reward] mean_similarity=%.3f frac_exact=%.3f frac_format=%.3f",
        sum(rewards) / n,
        frac_exact,
        frac_format,
    )
    if completions:
        logging.info("[reward] first-prompt answer=%r", answer[0])
        preview = completions[0].strip().replace("\n", " ")[:300]
        logging.info("[reward]   gen[0] = %r", preview)

    return rewards


def build_dataset(tokenizer, seed):
    """Return the templated ``prompt`` / ``answer`` dataset for reverse-text.

    The on-main example carves out a held-out eval split; that split is only
    used by ``EvalCallback``, which this async version doesn't run (see the
    module docstring), so the eval carve-out is dropped here.
    """
    ds = load_dataset(DATASET, split=DATASET_SPLIT)

    def to_example(row):
        text = row["prompt"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        return {
            # Qwen3 tokenizers expose an ``enable_thinking`` flag; disable it
            # so the model answers directly instead of emitting a <think> block.
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
            "answer": text[::-1],
        }

    return ds.shuffle(seed=seed).map(to_example, remove_columns=ds.column_names)


def get_output_dir() -> str:
    return os.path.join(
        str(REPO_ROOT),
        "generated/tt-train/grpo_reverse_text_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )


class WeightSyncCallback:
    """Push fresh policy weights to the TTT generation worker every ``every``
    steps. The caller does the initial push before ``trainer.train()``."""

    def __init__(self, completer: Any, every: int = 1) -> None:
        if every < 1:
            raise ValueError(f"WeightSyncCallback: 'every' must be >= 1 (got {every})")
        self.completer = completer
        self.every = every

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        if step % self.every == 0:
            self.completer.push_weights()

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        pass


def _load_device_config():
    raw = load_config(os.path.join(str(REPO_ROOT), CONFIG_REL))
    return DeviceConfig(raw), raw


def _open_ttml_device(device_config) -> Any:
    # Do NOT call enable_fabric() here: fabric is already pinned FABRIC_2D at
    # import. A repeat SetFabricConfig re-runs a control-plane reinit collective
    # with no peer (TTT never re-sets) and deadlocks device bring-up.
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
    return autograd_ctx.get_device()


def _close_ttml_device() -> None:
    ttml.autograd.AutoContext.get_instance().close_device()


def _ttml_main() -> None:
    autograd_ctx = ttml.autograd.AutoContext.get_instance()
    autograd_ctx.initialize_distributed_context(*sys.argv)

    device_config, raw = _load_device_config()
    mesh_device = _open_ttml_device(device_config)

    model_id = raw["training_config"]["model_id"]
    weight_sync_every = int(raw["training_config"]["weight_sync_every"])

    completer: Any = None
    client: Any = None
    try:
        bridge = HostWeightBridge.init_sender(mesh=mesh_device, peer_rank=TTT_RANK)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        dataset = build_dataset(tokenizer, seed=int(raw["training_config"].get("seed", 0)))

        output_dir = get_output_dir()
        grpo_config = get_grpo_config(raw, output_dir=output_dir)
        optimizer_dict = raw["training_config"]["optimizer"]
        transformer_config = get_model_config(raw["training_config"]["model_config"])

        completer = Qwen3CompleterRemoteRollout(
            ctx=Qwen3CompletionCtx(
                max_tokens_to_complete=grpo_config.max_completion_length,
                temperature=grpo_config.temperature,
                completions_per_prompt=grpo_config.num_generations,
            ),
            transformer_config=transformer_config,
            mesh_device=mesh_device,
            model_source=model_id,
            enable_ddp=device_config.enable_ddp,
        )

        client = MPIRolloutClient(peer_rank=TTT_RANK, bridge=bridge)
        completer._client = client

        # Replace the worker's dummy boot weights with the real SFT'd Qwen3
        # weights before training starts.
        completer.push_weights()

        trainer = GRPOTrainer(
            completer=completer,
            dataset=dataset,
            config=grpo_config,
            reward_func=similarity_reward,
            optimizer_dict=optimizer_dict,
            callbacks=[
                WeightSyncCallback(completer, every=weight_sync_every),
            ],
            model_source=model_id,
        )
        trainer.train()
    finally:
        # Shut the server down BEFORE closing the mesh: the worker is blocked
        # in serve_forever() and MPI won't tear down cleanly otherwise.
        if client is not None:
            try:
                client.shutdown()
            except Exception:  # noqa: BLE001
                pass
        completer = None
        gc.collect()
        _close_ttml_device()


def _ttt_main() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    # Read the same yaml as the ttml rank so both agree on the GRPO sampling
    # temperature (baked into the worker's decode trace) and on the rollout
    # mesh / batch / seq-len via ``remote_rollout_config``.
    raw = load_config(os.path.join(str(REPO_ROOT), CONFIG_REL))
    grpo_temperature = float(raw["training_config"]["grpo_config"]["temperature"])
    model_id = raw["training_config"]["model_id"]
    rr = raw["remote_rollout_config"]

    parent_mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*rr["mesh_shape"]),
        offset=ttnn.MeshCoordinate(0, 0),
    )

    worker: Any = None
    server: Any = None
    try:
        stop_token_ids, pad_token_id = qwen3_stop_and_pad(model_id)

        # One worker owns the whole parent mesh: it splits it into [1,1] submeshes
        # and runs generation data-parallel across them.
        #
        # ``dummy_weights=True`` (worker default) boots the tt-transformers model
        # with random weights and no per-tensor disk cache -- boot in seconds.
        # The first HostWeightBridge sync from the ttml rank overwrites every
        # buffer with the real SFT weights before ``generate()`` is ever called,
        # so the initial randomness never leaks into training.
        # This requires ``Qwen3-0.6B-Reverse-Text-SFT`` to appear in
        # ``ModelArgs.LOCAL_HF_PARAMS`` (only needs a ``config.json`` for
        # architecture bootstrap; see models/tt_transformers/tt/model_config.py).
        worker = TttGenerationWorker(
            mesh_device=parent_mesh,
            model_source=model_id,
            max_batch_size=rr["max_batch_size"],
            max_seq_len=rr["max_seq_len"],
            instruct=True,
            optimizations=bf16_attn_bfp8_mlp_optimizations,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            temperature=grpo_temperature,
            top_k=0,
            top_p=1.0,
            seed=None,
        )

        # The bridge replicates each transferred policy onto every submesh; the
        # worker applies one dict per submesh in update_weights().
        bridge = HostWeightBridge.init_receiver(mesh=parent_mesh, peer_rank=TTML_RANK, submeshes=worker.submeshes)

        server = MPIRolloutServer(
            peer_rank=TTML_RANK,
            bridge=bridge,
            generate_fn=worker.generate,
            on_weights_received=worker.update_weights,
        )
        server.serve_forever()
    finally:
        worker = None
        server = None
        gc.collect()
        ttnn.close_mesh_device(parent_mesh)


if __name__ == "__main__":
    # Configure the root logger before any library (datasets, ...) can claim it.
    # INFO surfaces the per-generate similarity_reward summary; set GRPO_LOGLEVEL=DEBUG
    # to also see the per-chunk decode progress on the ttt rank.
    logging.basicConfig(
        level=os.environ.get("GRPO_LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()

    world_size = int(ttnn.distributed_context_get_size())
    if world_size != 2:
        raise RuntimeError(
            f"reverse_text_training_example must run under tt-run with world_size == 2 (got {world_size}). "
            "Use reverse_text/runner.sh."
        )

    rank = int(ttnn.distributed_context_get_rank())
    if rank == TTML_RANK:
        _ttml_main()
    elif rank == TTT_RANK:
        _ttt_main()
    else:
        raise RuntimeError(
            f"Unexpected MPI rank {rank} (world_size={world_size}); "
            f"expected exactly two ranks: TTML={TTML_RANK}, TTT={TTT_RANK}."
        )
