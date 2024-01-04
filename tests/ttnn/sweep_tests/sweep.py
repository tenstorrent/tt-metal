# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pathlib

from loguru import logger
import pandas as pd

SWEEP_RESULTS_DIR = pathlib.Path(__file__).parent / "results"


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


def get_parameter_namess(parameters):
    if isinstance(parameters, dict):
        parameters = list(reversed(parameters.items()))

    if len(parameters) == 0:
        return []
    else:
        first_parameter, *other_parameters = parameters
        name, _ = first_parameter
        if "," in name:
            # Mutliple parameters in one string
            names = name.split(",")
            return names + get_parameter_namess(other_parameters)
        else:
            # Single parameter
            return [name] + get_parameter_namess(other_parameters)


def get_parameter_values(parameter_names, permutation):
    for parameter_name in parameter_names:
        parameter_value = permutation[parameter_name]
        if callable(parameter_value):
            parameter_value = parameter_value.__name__
        yield parameter_value


def sweep(sweep_file_name, run, skip, parameters):
    parameter_names = get_parameter_namess(parameters)
    column_names = ["status", "exception"] + parameter_names

    rows = []
    for permutation in permutations(parameters):
        parameter_values = list(get_parameter_values(parameter_names, permutation))

        if skip(**permutation):
            rows.append(["skipped", None] + parameter_values)
            continue

        try:
            passed, message = run(**permutation)
            if passed:
                rows.append(["passed", None] + parameter_values)
            else:
                rows.append(["failed", message] + parameter_values)
        except Exception as e:
            rows.append(["crashed", str(e)] + parameter_values)

    SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = (SWEEP_RESULTS_DIR / pathlib.Path(sweep_file_name).stem).with_suffix(".csv")

    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(file_name)

    logger.info(f"Saved sweep results to {file_name}")


def reproduce(run, parameters, index):
    permutation = list(permutations(parameters))[index]
    pretty_printed_parameters = ",\n".join(f"\t{key}={value}" for key, value in permutation.items())
    logger.info(f"Reproducing sweep results at index {index}:\n{{{pretty_printed_parameters}}}")
    return run(**permutation)


def sweep_or_reproduce(file_name, run, skip, parameters, sweep_index):
    if sweep_index is None:
        sweep(file_name, run, skip, parameters)
    else:
        assert reproduce(run, parameters, sweep_index)
