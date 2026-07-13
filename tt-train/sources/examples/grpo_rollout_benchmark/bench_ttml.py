#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ttml backend: GRPO BoolQ with LOCAL generation on the ttml mesh.

Single OS process. The completer (`LlamaGRPOCompleter`) opens its own AutoContext
mesh from the device YAML and generates with a ttml KvCache (WORKER dispatch, no
trace). Launched once per repeat by runner.sh; device count comes from the YAML.

Multi-chip runs enable fabric, which requires the visible system to equal the
opened mesh (fabric-on-a-subset is fatal). runner.sh scopes TT_VISIBLE_DEVICES to
exactly the N300 boards the mesh uses; invoking this script directly on a host with
more chips than the mesh needs the same TT_VISIBLE_DEVICES set by hand.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import benchmark_common as bc  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO BoolQ benchmark — ttml local-generation backend")
    p.add_argument("--config", required=True, help="ttml device YAML (configs/ttml_Ndev.yaml)")
    p.add_argument("--steps", type=int, default=20, help="GRPO steps per run")
    p.add_argument("--run-index", type=int, default=1, help="Repeat index, written to the CSV 'run' column")
    p.add_argument("--seed", type=int, default=42)
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("GRPO_LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoTokenizer
    from ttml.trainers import GRPOTrainer
    from utils.ttml_local.llama_completer import LlamaCompletionCtx, LlamaGRPOCompleter

    config_path = args.config if os.path.isabs(args.config) else os.path.join(_THIS_DIR, args.config)
    device_config, grpo_config, transformer_config, optimizer_dict, ttml_devices = bc.load_balanced_config(
        config_path, steps=args.steps
    )

    tokenizer = AutoTokenizer.from_pretrained(bc.MODEL_ID)
    dataset = bc.build_boolq_dataset(tokenizer)

    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=grpo_config.max_completion_length,
            temperature=grpo_config.temperature,
            completions_per_prompt=grpo_config.num_generations,
        ),
        transformer_config=transformer_config,
        device_config=device_config,
        model_source=bc.MODEL_ID,
    )

    monitor = bc.BenchmarkMonitor(
        bc.benchmark_csv_path("ttml", ttml_devices, 0),
        backend="ttml",
        run_index=args.run_index,
        ttml_devices=ttml_devices,
        ttt_devices=0,
    )

    logging.info(
        "ttml backend | devices=%d | completions/step=%d | steps=%d | run=%d",
        ttml_devices,
        bc.COMPLETIONS_PER_STEP,
        args.steps,
        args.run_index,
    )
    trainer = GRPOTrainer(
        completer=completer,
        dataset=dataset,
        config=grpo_config,
        reward_func=bc.boolq_reward,
        optimizer_dict=optimizer_dict,
        callbacks=[monitor],
        model_source=bc.MODEL_ID,
    )
    trainer.train()


if __name__ == "__main__":
    main()
