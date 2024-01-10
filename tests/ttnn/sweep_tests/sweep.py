# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from importlib.machinery import SourceFileLoader
import pathlib

from loguru import logger
import pandas as pd

import ttnn

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"
SWEEP_RESULTS_DIR = SWEEPS_DIR / "results"


def permutations(parameters):
    if isinstance(parameters, dict):
        parameters = list(reversed(parameters.items()))

    if len(parameters) == 0:
        yield {}
    else:
        first_parameter, *other_parameters = parameters
        for permutation in permutations(other_parameters):
            name, values = first_parameter

            if "," in name:
                # Mutliple parameters in one string
                names = name.split(",")
                for value in values:
                    yield {**permutation, **dict(zip(names, value))}
            else:
                # Single parameter
                for value in values:
                    yield {**permutation, name: value}


def get_parameter_names(parameters):
    if isinstance(parameters, dict):
        parameters = list(parameters.items())

    if len(parameters) == 0:
        return []
    else:
        first_parameter, *other_parameters = parameters
        name, _ = first_parameter
        if "," in name:
            # Mutliple parameters in one string
            names = name.split(",")
            return names + get_parameter_names(other_parameters)
        else:
            # Single parameter
            return [name] + get_parameter_names(other_parameters)


def get_parameter_values(parameter_names, permutation):
    for parameter_name in parameter_names:
        parameter_value = permutation[parameter_name]
        if callable(parameter_value):
            parameter_value = parameter_value.__name__
        yield parameter_value


def sweep(sweep_file_name, run, skip, parameters, *, device):
    sweep_name = pathlib.Path(sweep_file_name).stem
    parameter_names = get_parameter_names(parameters)
    column_names = ["status", "exception"] + parameter_names

    rows = []
    for permutation in permutations(parameters):
        parameter_values = list(get_parameter_values(parameter_names, permutation))

        if skip(**permutation):
            rows.append(["skipped", None] + parameter_values)
            continue

        try:
            passed, message = run(**permutation, device=device)
            if passed:
                rows.append(["passed", None] + parameter_values)
            else:
                rows.append(["failed", message] + parameter_values)
        except Exception as e:
            rows.append(["crashed", str(e)] + parameter_values)
        finally:
            import tt_lib as ttl

            ttl.device.ClearCommandQueueProgramCache(device)
            ttl.device.DeallocateBuffers(device)

    SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = (SWEEP_RESULTS_DIR / sweep_name).with_suffix(".csv")

    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(file_name)

    logger.info(f"Saved sweep results to {file_name}")


def _run_single_test(run, skip, parameters, index, *, device):
    permutation = list(permutations(parameters))[index]
    pretty_printed_parameters = ",\n".join(f"\t{key}={value}" for key, value in permutation.items())
    logger.info(f"Reproducing sweep results at index {index}:\n{{{pretty_printed_parameters}}}")
    if skip(**permutation):
        return "skipped", None
    passed, message = run(**permutation, device=device)
    return passed, message


def run_single_test(test_name, index, *, device):
    file_name = (SWEEP_SOURCES_DIR / test_name).with_suffix(".py")
    logger.info(f"Running {file_name}")

    sweep_module = SourceFileLoader("sweep_module", str(file_name)).load_module()

    status = None
    try:
        passed, message = _run_single_test(
            sweep_module.run, sweep_module.skip, sweep_module.parameters, index, device=device
        )
        status = "passed" if passed else "failed"
        if not passed:
            logger.error(message)
    except Exception as e:
        status = "crashed"
        message = f"Exception: {e}"
        logger.exception(message)
    return status, message


def run_all_tests():
    logger.info(f"Deleting old sweep results in {SWEEP_RESULTS_DIR}")
    if SWEEP_RESULTS_DIR.exists():
        for file_name in SWEEP_RESULTS_DIR.glob("*.csv"):
            file_name.unlink()

    device = ttnn.open(0)
    for file_name in sorted(SWEEP_SOURCES_DIR.glob("*.py")):
        logger.info(f"Running {file_name}")
        sweep_module = SourceFileLoader("sweep_module", str(file_name)).load_module()
        sweep(file_name, sweep_module.run, sweep_module.skip, sweep_module.parameters, device=device)
    ttnn.close(device)


def print_report():
    stats_df = pd.DataFrame(columns=["name", "passed", "failed", "skipped", "crashed"])

    def add_row(df, name):
        df.loc[-1] = [name, 0, 0, 0, 0]
        df.index = df.index + 1
        df.reset_index(inplace=True, drop=True)
        return df

    for file_name in sorted(SWEEP_RESULTS_DIR.glob("*.csv")):
        df = pd.read_csv(file_name)
        stats_df = add_row(stats_df, file_name.stem)
        for status in stats_df.columns[1:]:
            stats_df.at[len(stats_df) - 1, status] = (df["status"] == status).sum()

    stats_df = add_row(stats_df, "total")
    stats_df.loc[len(stats_df) - 1, stats_df.columns[1:]] = stats_df[stats_df.columns[1:]].sum()

    print(stats_df)


def run_failed_and_crashed_tests(*, device, exclude):
    keep_running = True
    for file_name in sorted(SWEEP_RESULTS_DIR.glob("*.csv")):
        test_name = file_name.stem
        if test_name in exclude:
            continue

        if not keep_running:
            break

        df = pd.read_csv(file_name)
        failed = (df["status"] == "failed").sum()
        crashed = (df["status"] == "crashed").sum()
        if failed == 0 and crashed == 0:
            continue

        for index, row in enumerate(df.itertuples()):
            if row.status not in {"failed", "crashed"}:
                continue

            status, _ = run_single_test(file_name.stem, index, device=device)
            logger.info(status)
            if status in {"failed", "crashed"}:
                keep_running = False
                break

            df.at[index, "status"] = status
            df.at[index, "message"] = None

        df.to_csv(file_name)
