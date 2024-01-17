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


def reproduce(run, parameters, index, *, device):
    permutation = list(permutations(parameters))[index]
    pretty_printed_parameters = ",\n".join(f"\t{key}={value}" for key, value in permutation.items())
    logger.info(f"Reproducing sweep results at index {index}:\n{{{pretty_printed_parameters}}}")
    return run(**permutation, device=device)


def run_sweeps():
    device = ttnn.open(0)
    for file_name in sorted(SWEEP_SOURCES_DIR.glob("*.py")):
        logger.info(f"Running {file_name}")
        sweep_module = SourceFileLoader("sweep_module", str(file_name)).load_module()
        sweep(file_name, sweep_module.run, sweep_module.skip, sweep_module.parameters, device=device)
    ttnn.close(device)


def check_sweeps():
    total_stats = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "crashed": 0,
    }
    for file_name in sorted(SWEEP_RESULTS_DIR.glob("*.csv")):
        df = pd.read_csv(file_name)
        stats = {key: 0 for key in total_stats.keys()}
        for status in stats.keys():
            stats[status] = (df["status"] == status).sum()
        logger.info(f"{file_name.stem}: {stats}")
        for status in stats.keys():
            total_stats[status] += stats[status]
    logger.info(f"Total: {total_stats}")
