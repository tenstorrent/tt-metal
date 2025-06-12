# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import re
from pathlib import Path

from loguru import logger

from models.perf.benchmarking_utils import BenchmarkData, BenchmarkProfiler


def create_benchmark_data(profiler: BenchmarkProfiler, measurements: dict, N_warmup_iter: dict, targets: dict):
    """
    Create a benchmark data object and populate the object with the given measurements.

    Pre-requisites:
    - The measurements dictionary should contain the following keys: "compile_prefill", "compile_decode", "prefill_t/s", "prefill_time_to_token", "decode_t/s", "decode_t/s/u"
    - The profiler object should contain the start and end times for the steps "compile_prefill", "compile_decode", "inference_prefill", "inference_decode"

    Optional (should be provided if measuring perf, not required for token generation):
    - The measurements dictionary should contain the following keys: "prefill_decode_t/s/u"
    - The targets dictionary should contain the following keys: "prefill_t/s", "decode_t/s", "decode_t/s/u"
    - The N_warmup_iter dictionary should contain the following keys: "inference_prefill", "inference_decode"

    Optional (should be provided if doing token verification, not required for perf):
    - The measurements dictionary should contain the key "token_verification"
    """

    assert all(
        key in measurements
        for key in [
            "compile_prefill",
            "compile_decode",
            "prefill_t/s",
            "prefill_time_to_token",
            "decode_t/s",
            "decode_t/s/u",
        ]
    )

    benchmark_data = BenchmarkData()

    # Add required measurement data
    benchmark_data.add_measurement(profiler, 0, "compile_prefill", "time(s)", measurements["compile_prefill"])
    benchmark_data.add_measurement(profiler, 0, "compile_decode", "time(s)", measurements["compile_decode"])
    benchmark_data.add_measurement(
        profiler,
        0,
        "inference_prefill",
        "tokens/s",
        measurements["prefill_t/s"],
        step_warm_up_num_iterations=(
            N_warmup_iter["inference_prefill"] if "inference_prefill" in N_warmup_iter else None
        ),
        target=targets["prefill_t/s"] if "prefill_t/s" in targets else None,
    )
    benchmark_data.add_measurement(
        profiler,
        0,
        "inference_prefill",
        "time_to_token",
        measurements["prefill_time_to_token"],
        step_warm_up_num_iterations=(
            N_warmup_iter["inference_prefill"] if "inference_prefill" in N_warmup_iter else None
        ),
        target=None,
    )
    benchmark_data.add_measurement(
        profiler,
        0,
        "inference_decode",
        "tokens/s",
        measurements["decode_t/s"],
        step_warm_up_num_iterations=N_warmup_iter["inference_decode"] if "inference_decode" in N_warmup_iter else None,
        target=targets["decode_t/s"] if "decode_t/s" in targets else None,
    )
    benchmark_data.add_measurement(
        profiler,
        0,
        "inference_decode",
        "tokens/s/user",
        measurements["decode_t/s/u"],
        step_warm_up_num_iterations=N_warmup_iter["inference_decode"] if "inference_decode" in N_warmup_iter else None,
        target=targets["decode_t/s/u"] if "decode_t/s/u" in targets else None,
    )

    # Add optional measurement data
    if "prefill_decode_t/s/u" in measurements:
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_prefill_decode",
            "tokens/s/user",
            measurements["prefill_decode_t/s/u"],
            step_warm_up_num_iterations=None,
            target=None,
        )
    if "token_verification" in measurements:
        benchmark_data.add_measurement(
            profiler,
            0,
            "inference_decode",
            "token_verification",
            measurements["token_verification"],
        )

    return benchmark_data


def check_tokens_match(generated_text: dict, expected_greedy_output_path: str):
    with open(expected_greedy_output_path, "r") as f:
        expected_output = json.load(f)
    return generated_text == expected_output, expected_output


def verify_perf(
    measurements: dict,
    expected_perf_metrics: dict,
    high_tol_percentage=1.15,  # 15% tolerance (approx +-5% CI variance + 5% real increase)
    expected_measurements: dict = None,
):
    """
    Verify the performance metrics against the expected values.
    The metrics that must be provided are specified in expected_measurements below.
    """

    expected_measurements_default = {
        "compile_prefill": False,
        "compile_decode": False,
        "prefill_time_to_token": False,
        "prefill_decode_t/s/u": False,
        "prefill_t/s": True,
        "decode_t/s": True,
        "decode_t/s/u": True,
    }
    expected_measurements_default.update(expected_measurements if expected_measurements else {})
    expected_measurements = expected_measurements_default

    does_pass = True
    for key in expected_measurements:
        if not expected_measurements[key]:
            continue
        assert (
            key in measurements and key in expected_perf_metrics
        ), f"Metric {key} not found in measurements or expected_perf_metrics"
        if expected_perf_metrics[key] is None:
            logger.warning(f"Expected perf metric {key} is None, skipping check")
            continue

        if measurements[key] < expected_perf_metrics[key]:  # Note: assumes higher is better for metric
            does_pass = False
            logger.warning(f"{key} ({measurements[key]}) is lower than expected {expected_perf_metrics[key]}")
        elif measurements[key] > expected_perf_metrics[key] * high_tol_percentage:
            does_pass = False
            logger.warning(
                f"{key} ({measurements[key]}) is higher than expected {expected_perf_metrics[key]}. Please update the expected perf."
            )

    if does_pass:
        logger.info("Perf Check Passed!")
    else:
        logger.warning("Perf Check Failed!")
        assert (
            does_pass
        ), f"Prefill or decode perf is either lower or higher than {expected_perf_metrics}. See earlier warnings for more details."


def parse_readme_perf_targets(model_name: str, batch_size: int, device_name: str, dp: int = None, tp: int = None):
    """Parse README.md table to extract performance targets and compare with hardcoded values."""
    try:
        readme_path = Path(__file__).parent.parent.parent.parent / "README.md"
        if not readme_path.exists():
            logger.warning(f"README.md not found at {readme_path}")
            return

        with open(readme_path, "r") as f:
            content = f.read()

        # Find the LLMs performance table
        lines = content.split("\n")
        table_start = -1
        table_end = -1

        for i, line in enumerate(lines):
            if "| Model" in line and "t/s/u" in line:
                table_start = i + 2  # Skip header and separator line
                break

        if table_start == -1:
            logger.warning("Performance table not found in README.md")
            return

        # Find table end
        for i in range(table_start, len(lines)):
            if not lines[i].strip() or not lines[i].startswith("|"):
                table_end = i
                break

        if table_end == -1:
            table_end = len(lines)

        readme_targets = {}

        # Parse table rows
        for i in range(table_start, table_end):
            line = lines[i].strip()
            if not line or not line.startswith("|"):
                continue

            # Split by | and clean up
            cols = [col.strip() for col in line.split("|")[1:-1]]  # Remove empty first/last elements
            if len(cols) < 6:
                continue

            model_col = cols[0]
            batch_col = cols[1]
            hardware_col = cols[2]
            ttft_col = cols[3]
            t_s_u_col = cols[4]
            target_t_s_u_col = cols[5]
            t_s_col = cols[6]

            # Extract model name from markdown links and parentheses
            model_match = re.search(
                r"\[(.*?)\]|\b(Llama.*?|Mistral.*?|Falcon.*?|Qwen.*?|Mamba.*?|QwQ.*?|DeepSeek.*?)(?:\s|\(|$)", model_col
            )
            if model_match:
                full_model_text = model_match.group(1) or model_match.group(2)

                # Parse model name and parallelization info
                paren_match = re.search(r"^(.*?)\s*\((.*?)\)$", full_model_text.strip())
                if paren_match:
                    model_name_readme = paren_match.group(1).strip()
                    parallel_info = paren_match.group(2).strip()  # e.g., "TP=8", "DP=2", etc.
                else:
                    model_name_readme = full_model_text.strip()
                    parallel_info = None
            else:
                continue

            # Extract batch size
            try:
                batch_size_readme = int(batch_col.strip())
            except (ValueError, AttributeError):
                continue

            # Extract hardware name
            hardware_match = re.search(r"\[(.*?)\]|([nN]\d+|[pP]\d+|[tT]3[kK]|QuietBox|Galaxy)", hardware_col)
            if hardware_match:
                hardware_name = hardware_match.group(1) or hardware_match.group(2)
                # Map hardware names to device names used in code
                hardware_mapping = {
                    "n150": "N150",
                    "n300": "N300",
                    "p100": "P100",
                    "p150": "P150",
                    "p300": "P300",
                    "QuietBox": "T3K",
                    "Galaxy": "TG",
                }
                device_name_readme = hardware_mapping.get(hardware_name, hardware_name.upper())
            else:
                continue

            # construct measurement id from model_name, batch_size, device_name
            measurement_id = (
                f"{model_name.replace('-', '')}_bs{batch_size}_{hardware_mapping.get(device_name, device_name.upper())}"
            )
            measurement_id_from_table = (
                f"{model_name_readme.replace(' ', '')}_bs{batch_size_readme}_{device_name_readme}"
            )
            # only collect targets for the same model, batch_size, device_name
            if measurement_id != measurement_id_from_table:
                continue

            # check parallel_info against dp and tp; only collect targets for the same parallelization
            if parallel_info:
                # example parallel_info: TP=8
                # extract dp and tp from parallel_info
                para_name_readme, para_value_readme = parallel_info.split("=")
                match (para_name_readme):
                    case "TP":
                        if tp is None or para_value_readme != str(tp):
                            continue
                    case "DP":
                        if dp is None or para_value_readme != str(dp):
                            continue

            # Extract target ttft, t/s/u, t/s
            readme_targets[measurement_id] = {}
            # remove * from the target value, which is used to indicate that a footnote is present below the table
            if ttft_col and (num_str := ttft_col.strip("*")) and num_str.replace(".", "").isdigit():
                readme_targets[measurement_id].update(
                    {"prefill_time_to_token": float(num_str) / 1000}
                )  # /1000 to convert to seconds
            else:
                logger.warning(f"Target ttft not parsed in README.md for {measurement_id}")

            if t_s_u_col and (num_str := t_s_u_col.strip("*")) and num_str.replace(".", "").isdigit():
                readme_targets[measurement_id].update({"decode_t/s/u": float(num_str)})
            else:
                logger.warning(f"Target t/s/u not parsed in README.md for {measurement_id}")

            if t_s_col and (num_str := t_s_col.strip("*")) and num_str.replace(".", "").isdigit():
                readme_targets[measurement_id].update({"decode_t/s": float(num_str)})
            else:
                logger.warning(f"Target t/s not parsed in README.md for {measurement_id}")

        assert (
            len(readme_targets) > 0
        ), "No targets found in README.md for the given model, batch_size, device_name; Check the README.md for the correct model name, batch_size, and device_name."
        assert (
            len(readme_targets) == 1
        ), "Multiple targets found in README.md for the given model, batch_size, device_name; Check the README.md for duplicate entries."

        return readme_targets[measurement_id]

    except Exception as e:
        logger.error(f"Error parsing README.md performance targets: {e}")
