# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import argparse
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def download_weights(model_name, downloaded_weights_path):
    # Check if weights exist in the specified folder. If not download them.
    if not os.path.isfile(downloaded_weights_path + "/model.safetensors.index.json"):
        snapshot_download(repo_id=model_name, local_dir=downloaded_weights_path, repo_type="model")
    else:
        print(f"{downloaded_weights_path}/model.safetensors.index.json file already presents.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, help="Path to store the downloaded weights folder", required=True)
    parser.add_argument(
        "--instruct", action="store_true", help="Choose instruct weights to download instead of general weights"
    )

    args = parser.parse_args()

    if args.instruct:
        model_name = "Qwen/Qwen2-7B-Instruct"
    else:
        model_name = "Qwen/Qwen2-7B"

    download_weights(model_name, downloaded_weights_path=args.weights_path)
