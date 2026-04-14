# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os
import shutil
from pathlib import Path

from .logger import logger

ORDER_FOLDER_PATH = None


def setup_files(json_folder: Path, shall_refresh: bool = False):
    global ORDER_FOLDER_PATH
    ORDER_FOLDER_PATH = json_folder
    logger.info(ORDER_FOLDER_PATH)
    if shall_refresh:
        shutil.rmtree(json_folder, ignore_errors=True)
        os.mkdir(json_folder)


def unify_files(destination_file: Path):
    global ORDER_FOLDER_PATH

    big_report = {}
    for worker_file in glob.glob(os.path.join(ORDER_FOLDER_PATH, "*.jsonl")):
        logger.info(worker_file)
        worker_number = Path(worker_file).stem.replace("gw", "")
        with open(worker_file, "r") as fp:
            big_report[worker_number] = [
                json.loads(line) for line in fp if line.strip()
            ]

    with open(destination_file, "w") as fp:
        json.dump(big_report, fp, indent=4)


def append_record(json_file: Path, record: dict):
    global ORDER_FOLDER_PATH
    with open(ORDER_FOLDER_PATH / json_file, "a") as f:
        f.write(json.dumps(record) + "\n")
