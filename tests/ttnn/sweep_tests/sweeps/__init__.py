# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import datetime
import pathlib
import pickle
import sqlite3
from types import ModuleType
import zlib
from importlib.machinery import SourceFileLoader

import enlighten
import pandas as pd
from loguru import logger

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"
SWEEP_RESULTS_DIR = SWEEPS_DIR / "results"
DATABASE_FILE_NAME = SWEEP_RESULTS_DIR / "db.sqlite"


class SweepFileLoader(SourceFileLoader):
    def load_module(self) -> ModuleType:
        module = super().load_module()

        if not hasattr(module, "skip"):
            setattr(module, "skip", lambda **kwargs: (False, None))

        if not hasattr(module, "xfail"):
            setattr(module, "xfail", lambda **kwargs: (False, None))

        return module


if not SWEEP_RESULTS_DIR.exists():
    SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_sweep_name(file_name):
    return str(pathlib.Path(file_name).relative_to(SWEEP_SOURCES_DIR))


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


def preprocess_parameter_value(parameter_value):
    if callable(parameter_value):
        parameter_value = parameter_value.__name__
    return parameter_value


def get_parameter_values(parameter_names, permutation):
    for parameter_name in parameter_names:
        parameter_value = preprocess_parameter_value(permutation[parameter_name])
        yield parameter_value


def _run_single_test(run, skip, xfail, permutation, *, device):
    try:
        should_be_skipped, message = skip(**permutation)
        if should_be_skipped:
            return "skipped", message

        passed, message = run(**permutation, device=device)
        status = "passed" if passed else "failed"
        if passed:
            message = None
    except Exception as e:
        should_fail, expected_exception = xfail(**permutation)
        if should_fail:
            status = "xfailed"
            message = f"Exception: {e}"
        else:
            status = "crashed"
            message = f"Exception: {e}"
    finally:
        import ttnn

        ttnn.experimental.device.DeallocateBuffers(device)
    return status, message


def run_single_test(file_name, index, *, device):
    logger.info(f"Running {file_name}")

    sweep_name = get_sweep_name(file_name)
    sweep_module = SweepFileLoader(f"sweep_module_{sweep_name}", str(file_name)).load_module()
    permutation = list(permutations(sweep_module.parameters))[index]

    pretty_printed_parameters = ",\n".join(
        f"\t{key}={preprocess_parameter_value(value)}" for key, value in permutation.items()
    )
    logger.info(f"Running sweep test at index {index}:\n{{{pretty_printed_parameters}}}")
    return _run_single_test(
        sweep_module.run,
        sweep_module.skip,
        sweep_module.xfail,
        permutation,
        device=device,
    )


def run_sweep(file_name, *, device):
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_name = get_sweep_name(file_name)
    sweep_module = SweepFileLoader(f"sweep_module_{sweep_name}", str(file_name)).load_module()

    parameter_names = get_parameter_names(sweep_module.parameters)
    column_names = ["sweep_name", "timestamp", "status", "exception"] + parameter_names

    rows = []
    manager = enlighten.get_manager()
    pbar = manager.counter(total=len(list(permutations(sweep_module.parameters))), desc=sweep_name, leave=False)
    pbar.refresh()
    for permutation in permutations(sweep_module.parameters):
        status, message = _run_single_test(
            sweep_module.run,
            sweep_module.skip,
            sweep_module.xfail,
            permutation,
            device=device,
        )
        rows.append(
            [sweep_name, current_datetime, status, message] + list(get_parameter_values(parameter_names, permutation))
        )
        pbar.update()
    pbar.close()

    connection = sqlite3.connect(DATABASE_FILE_NAME)
    cursor = connection.cursor()

    # TODO: Add current_datetime back if we want to save history
    # table_hash = zlib.adler32(pickle.dumps(f"{sweep_name}_{current_datetime}"))
    table_hash = zlib.adler32(pickle.dumps(f"{sweep_name}"))
    table_name = f"table_{table_hash}"

    def column_names_to_string(column_names):
        def name_to_string(name):
            if name == "timestamp":
                return "timestamp TIMESTAMP"
            else:
                return f"{name} TEXT"

        column_names = [name_to_string(name) for name in column_names]
        return ", ".join(column_names)

    command = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_names_to_string(column_names)})"
    cursor.execute(command)

    for row in rows:
        row = [str(value) for value in row]
        row_placeholders = ", ".join(["?"] * len(column_names))
        command = f"INSERT INTO {table_name} VALUES ({row_placeholders})"
        cursor.execute(command, row)
    connection.commit()
    connection.close()

    SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saved sweep results to table {table_name} in {DATABASE_FILE_NAME}")

    return table_name


def collect_tests(*, include):
    for file_name in sorted(SWEEP_SOURCES_DIR.glob("**/*.py")):
        sweep_name = get_sweep_name(file_name)
        if include and sweep_name not in include:
            continue
        yield file_name, sweep_name


def run_sweeps(*, device, include):
    table_names = []
    for file_name, sweep_name in collect_tests(include=include):
        if not device:
            logger.info(f"Collecting {sweep_name}")
            continue
        else:
            logger.info(f"Running {sweep_name}")
            table_name = run_sweep(file_name, device=device)
            table_names.append(table_name)
    return table_names


def print_summary(*, table_names):
    stats_df = pd.DataFrame(columns=["name", "passed", "failed", "crashed", "skipped", "xfailed"])

    def add_row(df, name):
        df.loc[-1] = [name] + [0] * len(df.columns[1:])
        df.index = df.index + 1
        df.reset_index(inplace=True, drop=True)
        return df

    connection = sqlite3.connect(DATABASE_FILE_NAME)
    cursor = connection.cursor()

    for (table_name,) in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        if table_names is not None and table_name not in table_names:
            continue
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
        sweep_name = df["sweep_name"].iloc[0]
        stats_df = add_row(stats_df, sweep_name)
        for status in stats_df.columns[1:]:
            stats_df.at[len(stats_df) - 1, status] = (df["status"] == status).sum()

    stats_df = add_row(stats_df, "total")
    stats_df.loc[len(stats_df) - 1, stats_df.columns[1:]] = stats_df[stats_df.columns[1:]].sum()

    print(stats_df)


def print_detailed_report(*, table_names):
    connection = sqlite3.connect(DATABASE_FILE_NAME)
    cursor = connection.cursor()

    for (table_name,) in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        if table_names is not None and table_name not in table_names:
            continue
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", connection)
        for index, row in enumerate(df.itertuples()):
            if row.status in {"failed", "crashed"}:
                print(f"{table_name}@{index}: {row.status}")
                print(f"\t{row.exception}")
            elif row.status == "skipped":
                print(f"{table_name}@{index}: {row.status}")
            else:
                print(f"{table_name}@{index}: {row.status}")
        print()


def print_report(*, table_names=None, detailed=False):
    if detailed:
        print_detailed_report(table_names=table_names)
    else:
        print_summary(table_names=table_names)
