# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from os import listdir, environ
from os.path import isfile, join
import time
import git
from loguru import logger

today = time.strftime("%Y_%m_%d")


expected_cols = [
    "Model",
    "Setting",
    "Batch",
    "First Run (sec)",
    "Second Run (sec)",
    "Compiler Time (sec)",
    "Expected Compile Time (sec)",
    "Inference Time GS (sec)",
    "Expected Inference Time GS (sec)",
    "Throughput GS (Batch*inf/sec)",
    "Inference Time CPU (sec)",
    "Throughput CPU (Batch*inf/sec)",
]

available_models = [
    "Falcon_decode_kv_cache_len=1024_seq_len=1_num_layers=32_config=L1-bf16",
    "Falcon_decode_kv_cache_len=128_seq_len=1_num_layers=32_config=L1-bf16",
    "Falcon_decode_kv_cache_len=2047_seq_len=1_num_layers=32_config=L1-bf16",
    "Falcon_prefill_kv_cache_len=0_seq_len=128_num_layers=32_config=L1-bf16",
    "Falcon_prefill_kv_cache_len=0_seq_len=256_num_layers=32_config=L1-bf16",
    "T5",
    "VGG",
    "bert11",
    "bloom",
    "deit",
    "llama",
    "roberta",
    "unbatched_stable_diffusion",
    "vit",
    "whisper",
]


def merge_perf_files():
    mypath = "./"
    csvfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f"{today}.csv" in f]

    repo = git.Repo(search_parent_directories=True)

    merge_res = open(f"Models_Perf_{today}.csv", "w")
    merge_res.write(f"branch: {repo.active_branch} \n")
    merge_res.write(f"hash: {repo.head.object.hexsha} \n")
    cols = ", ".join(expected_cols)
    merge_res.write(f"{cols} \n")

    csvfiles.sort()
    for csvfile in csvfiles:
        row_name = csvfile.replace("perf_", "")
        row_name = row_name.replace(f"{today}", "")
        row_name = row_name.replace(".csv", "")

        f = open(f"./{csvfile}", "r")
        f.readline()
        row = f.readline().strip().strip("\n")
        merge_res.write(f"{row}\n")

    merge_res.close()


def check_results():
    expected_inference_col = "Expected Inference Time GS (sec)"
    expected_compile_col = "Expected Compile Time (sec)"
    inference_col = "Inference Time GS (sec)"
    compile_col = "Compiler Time (sec)"

    merge_res = open(f"Models_Perf_{today}.csv")
    logger.info(next(merge_res))
    logger.info(next(merge_res))
    cols = next(merge_res).split(",")
    cols = [c.strip() for c in cols]
    logger.info(f"expected cols: {expected_cols}")
    logger.info(f"file cols: {cols}")
    assert len(expected_cols) == len(
        cols
    ), "the number of expected cols, and cols in the merge perf csv are not the same!"
    for expected_c, c in zip(expected_cols, cols):
        assert expected_c == c, f"Expected column {expected_c}, instead got {c}"
    logger.info("perf csv cols match the expected cols")
    logger.info(cols)
    visited_models = []
    slow_compile = []
    slow_inference = []
    for row in merge_res:
        logger.info(row)
        if len(row) < 5:
            logger.info(f"Skipping row: {row}.")
            continue
        models_info = row.split(",")
        logger.info(models_info)
        models_info = [item.strip() for item in models_info]
        logger.info(models_info)
        dict_info = {name: value for name, value in zip(cols, models_info)}
        logger.info(dict_info)
        model_name = dict_info["Model"]
        visited_models.append(model_name)
        model_expected_inference = float(dict_info[expected_inference_col])
        model_inference = float(dict_info[inference_col])
        model_expected_compile = float(dict_info[expected_compile_col])
        model_compile = float(dict_info[compile_col])
        if model_compile > model_expected_compile:
            slow_compile.append((model_name, model_compile, model_expected_compile))
            logger.error(
                f"{model_name} compile time is too slow with {model_compile}, expected {model_expected_compile}"
            )
        if model_inference > model_expected_inference:
            slow_inference.append((model_name, model_inference, model_expected_inference))
            logger.error(
                f"{model_name} inference  time is too slow with {model_inference}, expected {model_expected_inference}"
            )

    assert (
        len(slow_inference) == 0 or len(slow_compile) == 0
    ), f"Some model(s) inference time, and compile time are too slow, see above for details slow inference: {slow_inference}, slow compile: {slow_compile}"
    assert (
        len(slow_inference) == 0
    ), f"Some model(s) inference time are too slow, see above for details slow models: {slow_inference}"
    assert (
        len(slow_compile) == 0
    ), f"Some model(s) compile time are too slow, see above for details slow compiled models: {slow_compile}"


if __name__ == "__main__":
    merge_perf_files()
    check_results()
