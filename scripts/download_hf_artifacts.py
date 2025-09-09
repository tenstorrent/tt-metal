#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import argparse

import evaluate
from datasets import load_dataset
from huggingface_hub import snapshot_download
from loguru import logger


BH_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "distil-whisper/distil-large-v3",
]

TG_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "deepseek-ai/DeepSeek-R1-0528",
]

UPSTREAM_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
]

PYTHON_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

SINGLE_CARD_MODELS = [
    "CompVis/stable-diffusion-v1-4",
    "distil-whisper/distil-large-v3",
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "EleutherAI/gpt-neox-20b",
    "google/gemma-3-4b-it",
    "google/vit-base-patch16-224",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "nvidia/mit-b0",
    "nvidia/segformer-b0-finetuned-ade-512-512",
    "openai/clip-vit-large-patch14",
    "openai/clip-vit-base-patch16",
    "state-spaces/mamba-130m",
    "state-spaces/mamba-370m",
    "state-spaces/mamba-2.8b",
    "tiiuae/falcon-7b-instruct",
    "timm/ese_vovnet19b_dw.ra_in1k",
    "Qwen/Qwen2-7B-Instruct",
]

T3K_MODELS = [
    "google/gemma-3-27b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B",
    "Qwen/Qwen3-32B",
]

# TODO: add configs, splits, etc
DATASETS = [
    "hf-internal-testing/librispeech_asr_dummy",
    "huggingface/cats-image",
    "poloclub/diffusiondb",
    "squad_v2",
    "wikitext",
]

METRICS = [
    "accuracy",
    "bertscore",
    "mean_iou",
    "squad_v2",
]

ARGUMENT_TO_MODELS = {
    "bh": BH_MODELS,
    "tg": TG_MODELS,
    "upstream": UPSTREAM_MODELS,
    "python": PYTHON_MODELS,
    "single": SINGLE_CARD_MODELS,
    "t3k": T3K_MODELS,
}


def download(artifacts, args, artifact_type=None):
    success = 0
    for name in artifacts:
        try:
            snapshot_download(
                name,
                token=args.hf_token,
                cache_dir=args.cache_dir,
                repo_type=artifact_type,
                ignore_patterns="original/*",
            )
            logger.info(f"Successfully downloaded '{name}'")
            success += 1
        except Exception as err:
            logger.error(f"Error while downloading '{name}' {artifact_type}. Check logs below")
            logger.error(err)
    logger.info(f"Successfully downloaded {success}/{len(artifacts)} {artifact_type}s.")


def download_models(args):
    """Already downloaded models won't be downloaded again unless new version appeared"""

    logger.info("Downloading models...")
    for arg_name, models in ARGUMENT_TO_MODELS.items():
        if getattr(args, "all", False) or getattr(args, arg_name, False):
            logger.info(f"\t{arg_name}...")
            download(models, args, artifact_type="model")
            logger.info(f"\tFinished {arg_name}")
    logger.info("Finished downloading models")


def download_datasets(args):
    """Currently makes no sense to use it since datasets are loading in test-time"""
    logger.info("Downloading datasets...")
    # download(DATASETS, args, artifact_type="dataset")
    # datasets are using different structure then models/huggingface_hub and it's better use different API for downloading
    for dataset in DATASETS:
        _ = load_dataset(dataset, cache_dir=args.cache_dir, ignore_verifications=True)
    logger.info("Finished downloading datasets")


def download_metrics(args):
    """Currently makes no sense to use it since metrics are loading in test-time"""
    logger.info("Downloading metrics...")
    for metric in METRICS:
        _ = evaluate.load(metric)
    logger.info("Finished downloading metrics")


def get_parser():
    parser = argparse.ArgumentParser(
        epilog="Example: HF_HOME=/mnt/MLPerf/huggingface python scripts/download_hf_artifacts.py --all --hf_token <your_hf_token>"
    )
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
    parser.add_argument("--all", "-a", action="store_true", help="download all models")
    parser.add_argument("--bh", action="store_true", help="download BlackHole models")
    parser.add_argument("--tg", action="store_true", help="download TG models")
    parser.add_argument("--upstream", action="store_true", help="download Upstream models")
    parser.add_argument("--python", action="store_true", help="download Python models")
    parser.add_argument("--single", action="store_true", help="download Single Card models")
    parser.add_argument("--t3k", action="store_true", help="download T3000 models")
    parser.add_argument("--datasets", action="store_true", help="download datasets")
    parser.add_argument("--metrics", action="store_true", help="download metrics")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    download_models(args)
    if args.datasets:
        download_datasets(args)
    if args.metrics:
        download_metrics(args)


if __name__ == "__main__":
    main()
