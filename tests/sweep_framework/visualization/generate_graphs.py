# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import pathlib
import re
import sqlite3
import sys

import matplotlib.pyplot as plt

from dataclasses import dataclass
from tests.sweep_framework.permutations import *

# TODO: currently axes can only handle individual numbers. However, more complex
# structures will need to be handled. E.g. batch_sizes: '(1,)'.
# TODO: consider whether need to limit potential permutations
# TODO: add csv output


@dataclass
class Row:
    perf_value: float
    headers: list
    values: list
    axes_entries: dict


def filter_data(rows, columns, filters, ignored, perf, axes):
    column_count = len(columns)
    data = []
    axes_values = {key: set() for key in axes}
    for row in rows:
        should_add = True
        perf_value = None
        headers = []
        values = []
        axes_entries = {}
        for i in range(column_count):
            column = columns[i]
            entry = row[i]
            if column in ignored:
                continue
            elif column in filters.keys():
                if entry != filters[column]:
                    should_add = False
                    break
            elif column in axes:
                axes_entries[column] = int(entry)
            elif column in perf.keys():
                index = entry.find(perf[column])
                if index >= 0:
                    left_index = index + len(perf[column]) + len("': '")
                    right_index = entry.find("'", left_index)
                    if right_index < 0:
                        print(
                            f"ERROR: Could not parse perf entry in column {i} looking for {perf[column]}. Entry: {entry}"
                        )
                        break
                    perf_value = int(entry[left_index:right_index])
            else:
                headers.append(column)
                values.append(entry)
        if should_add and perf_value is not None:
            for key, value in axes_entries.items():
                axes_values[key].add(value)
            data.append(Row(perf_value, headers, values, axes_entries))
    return data, axes_values


def get_data_subset(data, permutation):
    subset = []
    for row in data:
        axes_entries = row.axes_entries
        add_row = True
        for key, value in permutation.items():
            if key not in axes_entries or axes_entries[key] != value:
                add_row = False
                break
        if add_row:
            subset.append(row)
    return subset


def get_label_mapping(subset):
    mapping = {}
    for row in subset:
        count = len(row.headers)
        for i in range(count):
            header = row.headers[i]
            if header not in mapping:
                mapping[header] = set()
            mapping[header].add(row.values[i])
    return mapping


def get_label_excluding_common(headers, values, label_mapping):
    label = "Parameters:"
    count = len(headers)
    for i in range(count):
        header = headers[i]
        if len(label_mapping[header]) > 1:
            label = label + f"  {headers[i]} {values[i]}"
    return label


def generate_graph(subset, axis, axes_values, permutation, path_prefix, perf):
    suffix = ".png"
    title = f"{list(perf.values())[0]} for "
    is_first = True
    for key, value in permutation.items():
        suffix = f"__{key}_{value}" + suffix
        if is_first:
            is_first = False
        else:
            title = title + ", "
        title = title + f"{key}={value}"

    x_values = {}
    y_values = {}
    label_mapping = get_label_mapping(subset)
    for row in subset:
        label = get_label_excluding_common(row.headers, row.values, label_mapping)
        if label not in x_values:
            x_values[label] = []
            y_values[label] = []
        x_values[label].append(row.axes_entries[axis])
        y_values[label].append(row.perf_value)

    filename = path_prefix + suffix
    print(f"Saving graph with title {title} to file {filename} with {len(x_values)} labels {x_values.keys()}")
    plt.figure(figsize=(8, 8), layout="constrained")
    plt.title(title)
    plt.ylabel("perf")
    plt.xlabel(axis)
    for label in x_values.keys():
        plt.plot(x_values[label], y_values[label], label=label, marker="o")
    plt.legend()
    plt.savefig(filename)
    plt.close()


def generate_graphs(data, axes_values, path_prefix, perf):
    print(axes_values)
    for axis in axes_values.keys():
        axes_values_copy = axes_values.copy()
        del axes_values_copy[axis]
        axes_permutations = list(permutations(axes_values_copy))
        for permutation in axes_permutations:
            subset = get_data_subset(data, permutation)
            generate_graph(subset, axis, axes_values, permutation, path_prefix, perf)


def process_files(sqlite_path, filter_string, filters, ignored, perf, axes):
    for sqlite_file in sorted(sqlite_path.glob("**/*.sqlite")):
        path = str(pathlib.Path(sqlite_file))
        relative_path = str(pathlib.Path(sqlite_file).relative_to(sqlite_path))
        if filter_string is not None and not filter_string in relative_path:
            continue
        db = path[path.rfind("/") + 1 : path.rfind(".")]
        table = relative_path[: relative_path.rfind(".")].replace("/", "_")
        path_prefix = path[: path.rfind(".")]
        sqlite_connection = sqlite3.connect(path)
        cursor = sqlite_connection.cursor()
        result = cursor.execute(f"SELECT * FROM {table}")
        rows = result.fetchall()
        columns = [entry[0] for entry in cursor.description]
        sqlite_connection.close()
        print(
            f"file {sqlite_file} path {path} path_prefix {path_prefix} db {db} relative path {relative_path} table {table} row count {len(rows)}"
        )
        data, axes_values = filter_data(rows, columns, filters, ignored, perf, axes)
        generate_graphs(data, axes_values, path_prefix, perf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Graph Generator",
        description="Creates graphs based on data dumped to sqlite.",
    )
    parser.add_argument("--sqlite-path", required=False, default=os.getenv("HOME"), help="Path to process")
    parser.add_argument(
        "--filter-string",
        required=False,
        help="sqlite file paths will only be processed if they contain this filter string",
    )
    parser.add_argument(
        "--filters",
        required=False,
        default='{"status": "TestStatus.PASS", "validity": "VectorValidity.VALID"}',
        help="Column names and values for which rows should be kept",
    )
    parser.add_argument(
        "--perf",
        required=False,
        default='{"device_perf": "DEVICE FW DURATION [ns]"}',
        help="Perf column name and value of interest to put into the graphs. Singleton map.",
    )
    parser.add_argument(
        "--ignored",
        required=False,
        nargs="*",
        default=[
            "vector_id",
            "timestamp",
            "host",
            "e2e_perf",
            "user",
            "git_hash",
            "input_hash",
            "message",
            "tag",
            "sweep_name",
            "invalid_reason",
        ],
        help='Column names for which values should be ignored. The list of values on the command line is space delimited. E.g. --ignored "timestamp" "host"',
    )
    parser.add_argument(
        "--axes",
        required=False,
        nargs="*",
        default=["m_size", "k_size", "n_size"],
        help='Column names for which values should be represented as a graph axis instead of on the legend. The list of values on the command line is space delimited. E.g. --axis "m_size" "k_size"',
    )
    args = parser.parse_args(sys.argv[1:])
    sqlite_path = pathlib.Path(args.sqlite_path)
    filters = json.loads(args.filters)
    perf = json.loads(args.perf)
    ignored = set(args.ignored)
    axes = set(args.axes)

    process_files(sqlite_path, args.filter_string, filters, ignored, perf, axes)
