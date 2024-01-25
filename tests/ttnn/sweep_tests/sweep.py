# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from importlib.machinery import SourceFileLoader
import pathlib

from loguru import logger
import pandas as pd

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


def preprocess_parameter_value(parameter_value):
    if callable(parameter_value):
        parameter_value = parameter_value.__name__
    return parameter_value


def get_parameter_values(parameter_names, permutation):
    for parameter_name in parameter_names:
        parameter_value = preprocess_parameter_value(permutation[parameter_name])
        yield parameter_value


def _run_single_test(run, skip, is_expected_to_fail, permutation, *, device):
    try:
        should_be_skipped, message = skip(**permutation)
        if should_be_skipped:
            return "skipped", message

        passed, message = run(**permutation, device=device)
        status = "passed" if passed else "failed"
        if passed:
            message = None
    except Exception as e:
        should_fail, expected_exception = is_expected_to_fail(**permutation)
        if should_fail and expected_exception == str(e):
            status = "is_expected_to_fail"
            message = expected_exception
        else:
            status = "crashed"
            message = f"Exception: {e}"
    finally:
        import tt_lib as ttl

        ttl.device.ClearCommandQueueProgramCache(device)
        ttl.device.DeallocateBuffers(device)
    return status, message


def run_single_test(test_name, index, *, device):
    file_name = (SWEEP_SOURCES_DIR / test_name).with_suffix(".py")
    logger.info(f"Running {file_name}")

    sweep_module = SourceFileLoader(f"sweep_module_{file_name.stem}", str(file_name)).load_module()
    permutation = list(permutations(sweep_module.parameters))[index]

    pretty_printed_parameters = ",\n".join(
        f"\t{key}={preprocess_parameter_value(value)}" for key, value in permutation.items()
    )
    logger.info(f"Running sweep test at index {index}:\n{{{pretty_printed_parameters}}}")
    return _run_single_test(
        sweep_module.run, sweep_module.skip, sweep_module.is_expected_to_fail, permutation, device=device
    )


def run_sweep(sweep_file_name, *, device):
    sweep_name = pathlib.Path(sweep_file_name).stem
    sweep_module = SourceFileLoader(f"sweep_module_{sweep_name}", str(sweep_file_name)).load_module()

    parameter_names = get_parameter_names(sweep_module.parameters)
    column_names = ["status", "exception"] + parameter_names

    rows = []
    for permutation in permutations(sweep_module.parameters):
        status, message = _run_single_test(
            sweep_module.run, sweep_module.skip, sweep_module.is_expected_to_fail, permutation, device=device
        )
        rows.append([status, message] + list(get_parameter_values(parameter_names, permutation)))

    SWEEP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_name = (SWEEP_RESULTS_DIR / sweep_name).with_suffix(".csv")

    df = pd.DataFrame(rows, columns=column_names)
    df.to_csv(file_name)

    logger.info(f"Saved sweep results to {file_name}")


def run_all_tests(*, device):
    logger.info(f"Deleting old sweep results in {SWEEP_RESULTS_DIR}")
    if SWEEP_RESULTS_DIR.exists():
        for file_name in SWEEP_RESULTS_DIR.glob("*.csv"):
            file_name.unlink()

    for file_name in sorted(SWEEP_SOURCES_DIR.glob("*.py")):
        logger.info(f"Running {file_name}")
        run_sweep(file_name, device=device)


def run_failed_and_crashed_tests(*, device, stepwise, include, exclude):
    keep_running = True
    for file_name in sorted(SWEEP_RESULTS_DIR.glob("*.csv")):
        test_name = file_name.stem

        if include and test_name not in include:
            continue

        if exclude and test_name in exclude:
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

            status, message = run_single_test(file_name.stem, index, device=device)
            logger.info(status)
            if status in {"failed", "crashed"}:
                logger.error(f"{message}")
                if stepwise:
                    keep_running = False
                    break

            df.at[index, "status"] = status
            df.at[index, "message"] = message

        df.to_csv(file_name)


def print_summary():
    stats_df = pd.DataFrame(columns=["name", "passed", "failed", "crashed", "skipped", "is_expected_to_fail"])

    def add_row(df, name):
        df.loc[-1] = [name] + [0] * len(df.columns[1:])
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


def print_detailed_report():
    for file_name in sorted(SWEEP_RESULTS_DIR.glob("*.csv")):
        name = file_name.stem
        df = pd.read_csv(file_name)
        for index, row in enumerate(df.itertuples()):
            if row.status in {"failed", "crashed"}:
                print(f"{name}@{index}: {row.status}")
                print(f"\t{row.exception}")
            elif row.status == "skipped":
                print(f"{name}@{index}: {row.status}")
            else:
                print(f"{name}@{index}: {row.status}")
        print()


def print_report(*, detailed=False):
    if detailed:
        print_detailed_report()
    else:
        print_summary()
