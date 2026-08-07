#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Wan 2.2 T2V-A14B style LoRA pipeline. Run one stage per process; see README.md."""

from __future__ import annotations

import argparse

from pipeline_config import DEFAULT_CONFIG_PATH, Config

_STAGES = ("preprocess", "precompute", "train", "infer")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Wan 2.2 T2V-A14B style LoRA on Tenstorrent",
        epilog="Every knob lives in the YAML config; --set is for one-off overrides.",
    )
    p.add_argument("stage", choices=_STAGES, help="which pipeline stage to run")
    p.add_argument(
        "-c",
        "--config",
        default=None,
        help=f"path to the pipeline config YAML (default: {DEFAULT_CONFIG_PATH})",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="override one config value, e.g. --set MAX_STEPS=100 (repeatable)",
    )
    args = p.parse_args()

    cfg = Config.from_yaml(args.config).apply_overrides(args.overrides)

    if args.stage == "preprocess":
        from preprocess import preprocess

        preprocess(cfg)

    elif args.stage == "precompute":
        from precompute import precompute

        precompute(cfg)

    elif args.stage == "train":
        from train import train

        train(cfg)

    elif args.stage == "infer":
        from infer import infer

        infer(cfg)


if __name__ == "__main__":
    main()
