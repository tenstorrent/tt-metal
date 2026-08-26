#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GRPO on GSM8K, async-rollout variant.

Two-rank tt-run entrypoint that adapts the ``grpo_remote_rollout/reverse_text``
template to the GSM8K task:

  - Rank 0 (TTML): owns the ttml Qwen3 policy and drives ``GRPOTrainer``.
    ``compute_nlog_probs`` runs locally on this rank; ``generate`` is an MPI
    call to rank 1 via :class:`MPIRolloutClient`.
  - Rank 1 (TTT): runs :class:`TttGenerationWorker` on a tt-transformers Qwen3
    with on-device sampling. The worker bakes ``temperature`` into its decode
    trace at first capture, so a per-call flip has no effect on the token
    stream; a periodic weight sync is the only signal the policy sees.

Weight sync: after every training step, :meth:`WeightSyncCallback.on_step_end`
exports the ttml policy via :func:`qwen3_weights_ref_hf_dict` and ships it
through the host bridge to the ttt worker.

Reward shaping mirrors the reference GPU/TRL training script in the
folder-upload set: five signals (``xmlcount``, ``soft_format``, ``format-regex``,
``int``, ``correctness``) summed into a single scalar for the ttml
:class:`GRPOTrainer`. Per-signal fractions and the mean reward are logged on
every reward call so training-side diagnostics line up with the standalone
``eval_gsm8k.py`` script row-by-row.

Prerequisites (called out here because the plan does not add them itself):
  - ``ichovpanTT/qwen3-0.6b-base-think-sft`` must be registered in
    ``models/tt_transformers/tt/model_config.py::ModelArgs.LOCAL_HF_PARAMS`` so
    the TTT worker can bootstrap its architecture. If unregistered, the worker
    fails at boot with a KeyError before the first rollout.
  - The utils it imports (``qwen3_grpo_completer``, ``qwen3_ttt_presets``,
    ``qwen3_overrides``, ``mpi_rollout``, ``ttt_generation_worker``,
    ``weight_bridge``) land through the sibling ``reverse-text`` PR. This
    example assumes the branch has been rebased on a main that carries them.

Run:
    tt-train/sources/examples/grpo_remote_rollout/gsm8k/runner.sh
"""

from __future__ import annotations

import csv
import gc
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

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

CONFIG_REL = "tt-train/configs/training_configs/grpo_gsm8k_qwen3_p6b_remote_rollout.yaml"

REPO_ROOT = Path(__file__).resolve().parents[5]

DATASET = "openai/gsm8k"
DATASET_CONFIG = "main"
DATASET_SPLIT = "train"

# Reasoning + answer tags match the SFT model naming (qwen3-0.6b-base-think-sft).
# Keeping these as a single source of truth: SYSTEM_PROMPT, STRICT_FORMAT_RE and
# the reward regexes below all derive from these four strings.
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

SYSTEM_PROMPT = (
    "Respond in the following format:\n" f"{THINK_OPEN}\n...\n{THINK_CLOSE}\n" f"{ANSWER_OPEN}\n...\n{ANSWER_CLOSE}\n"
)

# Pin fabric config before either _ttml_main or _ttt_main opens a device.
# Otherwise TTT's open_mesh_device auto-escalates to FABRIC_1D and the
# mismatch deadlocks the cross-rank fabric init.
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)


# ---------------------------------------------------------------------------
# Answer extraction + regexes (kept in lockstep with eval_gsm8k.py)
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# The four tags we track for the "tags present" / "tags exactly once" signals.
TAG_STRINGS: tuple[str, ...] = (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN, ANSWER_CLOSE)

# Regex-based format check ("format-regex" in eval_gsm8k.py). Matches the four
# tags in order with any whitespace/content between them.
STRICT_FORMAT_RE = re.compile(
    re.escape(THINK_OPEN)
    + r".*?"
    + re.escape(THINK_CLOSE)
    + r"\s*"
    + re.escape(ANSWER_OPEN)
    + r".*?"
    + re.escape(ANSWER_CLOSE),
    re.DOTALL,
)


def normalize_number(s: str) -> str:
    """Strip commas/punctuation and coerce '1000.0' to '1000' for equality checks."""
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    if not s:
        return s
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def extract_hash_answer(gold: str) -> str:
    """GSM8K gold answers look like '<reasoning>\\n#### 72'."""
    return normalize_number(gold.split("####")[-1])


def extract_tag_answer(text: str) -> Optional[str]:
    """Content of the last ``<answer>...</answer>`` block, then last number in it."""
    if ANSWER_OPEN not in text:
        return None
    body = text.split(ANSWER_OPEN)[-1].split(ANSWER_CLOSE)[0]
    nums = _NUM_RE.findall(body)
    return normalize_number(nums[-1]) if nums else None


# Snapshot of the per-signal breakdown from the most recent ``gsm8k_reward``
# call. Read by ``GRPOMonitor.on_step_end`` (fired later in the same step) so
# every CSV row carries the reward decomposition, not just the summed mean.
_LAST_REWARD_BREAKDOWN: dict[str, float] = {
    "xmlcount": float("nan"),
    "soft_format": float("nan"),
    "strict_format": float("nan"),
    "int_reward": float("nan"),
    "correctness": float("nan"),
    "frac_correct": float("nan"),
    "frac_tags_present": float("nan"),
    "frac_tags_exactly_once": float("nan"),
    "frac_format_regex": float("nan"),
}


# ---------------------------------------------------------------------------
# Reward shaping (mirrors data/folder-upload/train_grpo_qwen3_base.py)
# ---------------------------------------------------------------------------
def _xmlcount(text: str) -> float:
    """Partial credit per tag: 0.125 for each of the four tags appearing exactly
    once, minus a small penalty for text trailing after </answer>."""
    s = 0.0
    for tag in (THINK_OPEN, THINK_CLOSE, ANSWER_OPEN):
        if text.count(tag) == 1:
            s += 0.125
    if text.count(ANSWER_CLOSE) == 1:
        s += 0.125
        s -= len(text.split(ANSWER_CLOSE)[-1].strip()) * 0.001
    return s


def _soft_format(text: str) -> float:
    """0.5 when the four tags appear in the right order, whitespace ignored."""
    return 0.5 if STRICT_FORMAT_RE.search(text) else 0.0


# Newline-delimited strict layout matching the system prompt exactly.
_STRICT_LAYOUT_RE = re.compile(
    r"^\s*"
    + re.escape(THINK_OPEN)
    + r"\n.*?\n"
    + re.escape(THINK_CLOSE)
    + r"\n"
    + re.escape(ANSWER_OPEN)
    + r"\n.*?\n"
    + re.escape(ANSWER_CLOSE)
    + r"\s*$",
    re.DOTALL,
)


def _strict_format(text: str) -> float:
    """0.5 for the exact newline-delimited layout shown in the system prompt."""
    return 0.5 if _STRICT_LAYOUT_RE.match(text) else 0.0


def _int_reward(pred: Optional[str]) -> float:
    """0.5 when the extracted answer is a plain integer (GSM8K answers all are)."""
    return 0.5 if pred is not None and pred.lstrip("-").isdigit() else 0.0


def _correctness(pred: Optional[str], gold: str) -> float:
    """2.0 when the value inside ``<answer>...</answer>`` matches the gold answer."""
    return 2.0 if pred is not None and pred == gold else 0.0


def gsm8k_reward(completions: List[str], answer: List[str], **kwargs) -> List[float]:
    """Sum of the five signals, returned per completion. Logs per-signal fractions
    plus the mean reward so training-side diagnostics stay comparable to the
    ``eval_gsm8k.py`` table."""
    rewards: list[float] = []
    xmlc_sum = soft_sum = strict_sum = int_sum = corr_sum = 0.0
    n_correct = 0
    n_tags_present = 0
    n_tags_exactly_once = 0
    n_format_regex = 0

    for text, gold in zip(completions, answer):
        pred = extract_tag_answer(text)
        xmlc = _xmlcount(text)
        soft = _soft_format(text)
        strict = _strict_format(text)
        intr = _int_reward(pred)
        corr = _correctness(pred, gold)
        rewards.append(xmlc + soft + strict + intr + corr)

        xmlc_sum += xmlc
        soft_sum += soft
        strict_sum += strict
        int_sum += intr
        corr_sum += corr

        if corr > 0.0:
            n_correct += 1
        if all(t in text for t in TAG_STRINGS):
            n_tags_present += 1
        if all(text.count(t) == 1 for t in TAG_STRINGS):
            n_tags_exactly_once += 1
        if STRICT_FORMAT_RE.search(text):
            n_format_regex += 1

    n = max(len(rewards), 1)
    _LAST_REWARD_BREAKDOWN.update(
        {
            "xmlcount": xmlc_sum / n,
            "soft_format": soft_sum / n,
            "strict_format": strict_sum / n,
            "int_reward": int_sum / n,
            "correctness": corr_sum / n,
            "frac_correct": n_correct / n,
            "frac_tags_present": n_tags_present / n,
            "frac_tags_exactly_once": n_tags_exactly_once / n,
            "frac_format_regex": n_format_regex / n,
        }
    )
    logging.info(
        "[reward] frac_correct=%.3f frac_tags_present=%.3f frac_tags_exactly_once=%.3f "
        "frac_format_regex=%.3f mean_reward=%.3f",
        n_correct / n,
        n_tags_present / n,
        n_tags_exactly_once / n,
        n_format_regex / n,
        sum(rewards) / n,
    )
    if completions:
        logging.info("[reward] first-prompt gold=%r", answer[0])
        preview = completions[0].strip().replace("\n", " ")[:400]
        logging.info("[reward]   gen[0] = %r", preview)

    return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_dataset(tokenizer, seed: int):
    """Return the templated ``prompt`` / ``answer`` GSM8K dataset.

    Uses the tokenizer's own chat template with ``enable_thinking=False`` -- we
    reserve ``<think>`` for the model's own scratch block (rewarded above), not
    the Qwen3 thinking-mode wrapper. The tokenizer is used as-is (no override
    of ``chat_template``, no pad-token mutation).
    """
    ds = load_dataset(DATASET, DATASET_CONFIG, split=DATASET_SPLIT)

    def to_example(row):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
            "answer": extract_hash_answer(row["answer"]),
        }

    return ds.shuffle(seed=seed).map(to_example, remove_columns=ds.column_names)


def get_output_dir() -> str:
    return os.path.join(
        str(REPO_ROOT),
        "generated/tt-train/grpo_gsm8k_run",
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
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


class GRPOMonitor:
    """on_step_end CSV/stdout monitor.

    Extends the standard schema with the per-signal reward decomposition read
    from ``_LAST_REWARD_BREAKDOWN`` (populated by ``gsm8k_reward`` earlier in
    the same step).
    """

    _EXTRA_COLS = [
        "correctness",
        "xmlcount",
        "soft_format",
        "strict_format",
        "int_reward",
        "frac_correct",
        "frac_tags_present",
        "frac_tags_exactly_once",
        "frac_format_regex",
    ]

    def __init__(self, output_dir: str) -> None:
        self.file_path = os.path.join(output_dir, "grpo_metrics.csv")
        os.makedirs(output_dir, exist_ok=True)
        with open(self.file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["step", "reward", "avg_length"]
                + self._EXTRA_COLS
                + ["step_time_s", "step_time_with_weight_updates_s", "generation_time_s"]
            )

    def on_train_begin(self, trainer: Any) -> None:
        pass

    def on_step_end(self, trainer: Any, step: int, *args: Any, **kwargs: Any) -> None:
        reward = kwargs["reward_mean"]
        length = kwargs["mean_completion_len"]
        min_length = kwargs["min_completion_len"]
        max_length = kwargs["max_completion_len"]
        step_time_s = kwargs.get("step_time_s", float("nan"))
        step_time_and_previous_callbacks_s = kwargs.get("step_time_and_previous_callbacks_s", float("nan"))
        generation_time_s = kwargs.get("generation_time_s", float("nan"))
        b = _LAST_REWARD_BREAKDOWN
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{timestamp}] Step {step} | Reward: {reward:.4f} "
            f"(corr={b['correctness']:.3f} xml={b['xmlcount']:.3f} soft={b['soft_format']:.3f} "
            f"strict={b['strict_format']:.3f} int={b['int_reward']:.3f}) "
            f"| Len: {length:.2f} (min {min_length}, max {max_length}) tokens "
            f"| Step: {step_time_s:.2f}s (with updates: {step_time_and_previous_callbacks_s:.2f}s) "
            f"| Gen: {generation_time_s:.2f}s"
        )
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [step, reward, length]
                + [b[c] for c in self._EXTRA_COLS]
                + [step_time_s, step_time_and_previous_callbacks_s, generation_time_s]
            )

    def on_before_optimizer_step(self, trainer: Any) -> None:
        pass

    def on_save(self, trainer: Any, step: int, path: str) -> None:
        pass

    def on_train_end(self, trainer: Any) -> None:
        print("Training complete.")


# ---------------------------------------------------------------------------
# Rank entrypoints
# ---------------------------------------------------------------------------
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
            reward_func=gsm8k_reward,
            optimizer_dict=optimizer_dict,
            callbacks=[
                WeightSyncCallback(completer, every=weight_sync_every),
                GRPOMonitor(output_dir),
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
    # INFO surfaces per-generate reward summaries; GRPO_LOGLEVEL=DEBUG also
    # shows per-chunk decode progress on the ttt rank.
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
            f"gsm8k_training_example must run under tt-run with world_size == 2 (got {world_size}). "
            "Use gsm8k/runner.sh."
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
