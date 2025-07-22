#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse

from huggingface_hub import snapshot_download
from loguru import logger


# TODO: update, split into test suits and create args
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-8B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/falcon-rw-1b",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
]
DATASETS = [
    "squad_v2",
    "wikitext",
]


def download(artifacts, args, artifact_type=None):
    success = 0
    for name in artifacts:
        try:
            snapshot_download(name, token=args.hf_token, cache_dir=args.cache_dir, repo_type=artifact_type)
            logger.info(f"Successfully downloaded '{name}'")
            success += 1
        except Exception as err:
            logger.error(f"Error while downloading '{name}' {artifact_type}. Check logs below")
            logger.error(err)
    logger.info(f"Successfully downloaded {success}/{len(artifacts)} {artifact_type}s.")


def download_models(args):
    logger.info("Downloading models...")
    download(MODELS, args, artifact_type="model")
    logger.info("Finished downloading models")


def download_datasets(args):
    logger.info("Downloading datasets...")
    download(DATASETS, args, artifact_type="dataset")
    logger.info("Finished downloading datasets")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf_token",
        type=str,
        required=False,
        default=None,
        help="huggingface token. If None default huggingface processing will be used",
    )
    parser.add_argument(
        "--cache_dir",
        "-c",
        type=str,
        required=False,
        default=None,
        help="cache_dir where to store artifacts. If None default hugginface path will be used",
    )
    parser.add_argument("--models", "-m", action="store_true", help="flag to download models")
    parser.add_argument("--datasets", "-d", action="store_true", help="flag to download datasets")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.models:
        download_models(args.hf_token)
    if args.datasets:
        download_datasets(args.hf_token)


if __name__ == "__main__":
    main()
