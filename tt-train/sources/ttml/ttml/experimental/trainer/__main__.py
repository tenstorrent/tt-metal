# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""python -m ttml.experimental.trainer --config <recipe.yaml> [--trainer sft] [--set dotted.key=value ...]"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .callbacks import CheckpointCallback, MetricsCallback
from .configs import load_config
from .implementations import TRAINERS


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ttml.experimental.trainer", description="Fine-tune an HF model on ttml."
    )
    parser.add_argument("--config", required=True, help="YAML recipe: the fields of the trainer's config")
    parser.add_argument(
        "--trainer",
        default="sft",
        choices=sorted(TRAINERS),
        help="the trainer the recipe is for (default: %(default)s)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a recipe field, e.g. --set model.lora.r=16 or --set stopping_criteria.limit=20; repeatable",
    )
    args = parser.parse_args(argv)

    trainer_class = TRAINERS[args.trainer]
    config = load_config(args.config, args.overrides, trainer_class.config_class)
    trainer = trainer_class(callbacks=[MetricsCallback(), CheckpointCallback()])
    try:
        trainer.setup(config)
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
