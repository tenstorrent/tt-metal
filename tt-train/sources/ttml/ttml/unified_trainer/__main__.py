# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""python -m ttml.unified_trainer --config <recipe.yaml> [--set dotted.key=value ...]"""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .callbacks import CheckpointCallback, MetricsCallback
from .configs import load_config
from .implementations import SFTTrainer


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m ttml.unified_trainer", description="Fine-tune an HF model on ttml."
    )
    parser.add_argument("--config", required=True, help="YAML recipe: the fields of SFTTrainerConfig")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a recipe field, e.g. --set model.lora.r=16 or --set max_steps=20; repeatable",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, args.overrides)
    trainer = SFTTrainer(callbacks=[MetricsCallback(), CheckpointCallback()])
    try:
        trainer.setup(config)
        trainer.train()
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
