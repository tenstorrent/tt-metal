# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import argparse
import requests
import tempfile
import pathlib
import zipfile
import pandas as pd
from loguru import logger
from dataclasses import dataclass
from tabulate import tabulate
import os
import shutil


def get_list_of_runs():
    params = {"per_page": 15}
    url = "https://api.github.com/repos/tenstorrent-metal/tt-metal/actions/workflows/ttnn-run-sweeps.yaml/runs"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        runs = response.json()
    else:
        raise RuntimeError(f"Error fetching workflow runs: {response.status_code}:{response.text}")

    return runs


def download_artifacts(token, artifacts_url, temp_dir_path, directory_index):
    response = requests.get(artifacts_url)
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    if response.status_code == 200:
        artifacts_data = response.json()
        if artifacts_data["artifacts"]:
            artifact = artifacts_data["artifacts"][0]
            artifact_download_url = artifact["archive_download_url"]
            artifact_response = requests.get(artifact_download_url, headers=headers)
            if artifact_response.status_code == 200:
                (temp_dir_path / str(directory_index)).mkdir(parents=True, exist_ok=True)
                artifact_zip = temp_dir_path / str(directory_index) / "artifact.zip"
                with open(artifact_zip, "wb") as file:
                    file.write(artifact_response.content)
                logger.info(f"{artifacts_url} downloaded successfully.")
                return True
            else:
                raise RuntimeError("Failed to download the artifact.")
        else:
            print(f"No artifacts found.  Is there a run in progress for {artifacts_url} ?")
    else:
        raise RuntimeError(f"Failed to fetch artifacts list. {response.status_code}:{response.text}")
    return False


def read_csv_from_zip(zip_file, file_name):
    with zip_file.open(file_name) as f:
        df = pd.read_csv(f)
        if not df.empty and len(df.columns) > 1:
            # remove first unamed column which is just the index.
            # This will be displayed by tabulate.
            df = df.iloc[:, 1:]
        return df


def trim_column(texte, longueur):
    if len(texte) > longueur:
        return texte[-longueur + 3 :]
    return texte


def get_subset_for_status(recent_df, prior_df, status):
    failed_recent = recent_df[recent_df["status"] == status]
    matching_prior_status = prior_df["status"] == status
    failed_prior = prior_df[matching_prior_status]
    return failed_recent, failed_prior


def extract_only_recent_changes(failed_recent, failed_prior):
    run_id_column_name = failed_recent.columns[0]
    newly_failed = failed_recent[~failed_recent[run_id_column_name].isin(failed_prior[run_id_column_name])]
    for column in newly_failed.columns:
        newly_failed[column] = newly_failed[column].apply(lambda x: trim_column(str(x), 10))
    return newly_failed


def build_new_failures(recent_df, prior_df):
    failed_recent, failed_prior = get_subset_for_status(recent_df, prior_df, "failed")
    return extract_only_recent_changes(failed_recent, failed_prior)


def build_new_crashes(recent_df, prior_df):
    failed_recent, failed_prior = get_subset_for_status(recent_df, prior_df, "crashed")
    return extract_only_recent_changes(failed_recent, failed_prior)


def delete_directory_contents(dir_path):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


@dataclass
class OperationFailure:
    file_name: str
    failure_file_name: str
    commit_hash_with_failure: str
    commit_hash_prior_to_failure: str
    failures: int


def diff_results(temp_dir_path, most_recent_run_index, total_runs, directory_for_rst_pages):
    directory_for_rst_pages = pathlib.Path(directory_for_rst_pages)
    rst_failure_files = []
    rst_files = []
    failures_since_last_run = 0

    recent_zip = temp_dir_path / str(most_recent_run_index) / "artifact.zip"
    most_recent_commit_hash = ""
    commit_hash_file = temp_dir_path / str(most_recent_run_index) / "commit_hash.txt"
    with open(commit_hash_file, "r") as file:
        most_recent_commit_hash = file.read()

    new_failures = {}

    with zipfile.ZipFile(recent_zip, "r") as zip1:
        # We want to put the latest csv from the most recent run into html files
        zip1_files = set(zip1.namelist())
        for file_name in zip1_files:
            test_name = pathlib.Path(file_name).stem
            if file_name.endswith(".csv"):
                recent_df = read_csv_from_zip(zip1, file_name)
                for col in recent_df.columns:
                    recent_df[col] = recent_df[col].apply(lambda x: str(x).replace("\t", "    ").replace("\n", " "))
                rst_table = tabulate(recent_df, headers="keys", tablefmt="rst")
                rst_page_name = directory_for_rst_pages / f"{test_name}.rst"
                with open(rst_page_name, "w") as f:
                    f.writelines(f".. _ttnn.sweep_test_{test_name}:\n")
                    f.writelines("\n")
                    f.writelines(f"{test_name}\n")
                    f.writelines("====================================================================\n")
                    f.write(rst_table)
                new_failures[test_name] = OperationFailure(
                    f"{test_name}.rst", f"{test_name}_failure.rst", most_recent_commit_hash, "", 0
                )
                rst_files.append(test_name)

        # Now we need to check and see which differences started showing up relative to the most recent run per operation file
        for test_name in new_failures:
            commit_hash = most_recent_commit_hash
            prior_run_index = most_recent_run_index + 1
            while new_failures[test_name].failures == 0 and prior_run_index < total_runs - 1:
                prior_zip = temp_dir_path / str(prior_run_index) / "artifact.zip"
                with zipfile.ZipFile(prior_zip, "r") as zip2:
                    for file_name in zip2.namelist():
                        if file_name.endswith(f"{test_name}.csv"):
                            test_name = pathlib.Path(file_name).stem
                            recent_df = read_csv_from_zip(zip1, file_name)
                            prior_df = read_csv_from_zip(zip2, file_name)
                            failures_df = build_new_failures(recent_df, prior_df)
                            crashes_df = build_new_crashes(recent_df, prior_df)
                            combined_test_results_df = pd.concat([failures_df, crashes_df])
                            if len(combined_test_results_df) > 0:
                                failures_since_last_run = failures_since_last_run + len(combined_test_results_df)
                                new_failures[test_name].failures = combined_test_results_df.size
                                new_failures[test_name].failure_file_name = f"{test_name}_failure.rst"
                                new_failures[test_name].commit_hash_with_failure = commit_hash

                                rst_table = tabulate(combined_test_results_df, headers="keys", tablefmt="rst")
                                rst_page_name = directory_for_rst_pages / f"{test_name}_failure.rst"
                                with open(rst_page_name, "w") as f:
                                    f.writelines(f".. _ttnn.sweep_test_failure_{test_name}:\n")
                                    f.writelines("\n")
                                    f.writelines(f"{test_name}\n")
                                    f.writelines(
                                        "====================================================================\n"
                                    )
                                    f.write(rst_table)
                                rst_failure_files.append(new_failures[test_name])

                commit_hash_file = temp_dir_path / str(prior_run_index) / "commit_hash.txt"
                with open(commit_hash_file, "r") as file:
                    commit_hash = file.read()
                new_failures[test_name].commit_hash_prior_to_failure = commit_hash

                prior_run_index = prior_run_index + 1

    rst_template = """
.. _ttnn.sweep_tests:

Sweep Test Results
==================

Recent New Failures
-------------------

We have had {failures_since_last_run} new failures since the prior run.

.. toctree::
   :maxdepth: 2
   :hidden:

   {toctree_failure_filenames}

{sweep_test_failure_entries}


All Sweep Tests
---------------

These are the sweep tests for commit hash {most_recent_commit_hash}

.. toctree::
   :maxdepth: 2

   {toctree_entries}
"""

    sweep_test_failure_entries = "\n".join(
        [
            f"* :ref:`{op_failure.file_name.split('.')[0]} <ttnn.sweep_test_failure_{op_failure.file_name.split('.')[0]}>` "
            f"-> ( {op_failure.commit_hash_prior_to_failure} .. {op_failure.commit_hash_with_failure} ]"
            for op_failure in rst_failure_files
        ]
    )
    sweep_test_failure_entries = sweep_test_failure_entries.lstrip()

    toctree_failure_filenames = "\n   ".join(
        [op_failure.failure_file_name.replace(".rst", "") for op_failure in rst_failure_files]
    )

    toctree_entries = "\n   ".join(sorted(rst_files))

    complete_rst = rst_template.format(
        most_recent_commit_hash=most_recent_commit_hash,
        failures_since_last_run=failures_since_last_run,
        toctree_failure_filenames=toctree_failure_filenames,
        sweep_test_failure_entries=sweep_test_failure_entries,
        toctree_entries=toctree_entries,
    )

    rst_page_name = directory_for_rst_pages / "index.rst"
    with open(rst_page_name, "w") as file:
        file.write(complete_rst)

    logger.info(f"Built {rst_page_name}")


def download_from_pipeline(token, directory_for_rst_pages):
    """
    Download the results of the sweeps from the GitHub pipeline.

    :param token: Provide your GitHub token.
    """

    runs = get_list_of_runs()
    if len(runs["workflow_runs"]) < 3:
        # Note that if the run is in progress, there will not be any artifacts available yet on the most recent run.
        raise RuntimeError("We need at least three runs to compare the changes in the sweep tests")

    total_runs = len(runs["workflow_runs"])
    if runs["workflow_runs"][0]["status"] == "completed":
        most_recent_run_index = 0
    else:  # a run is in progress so we just use the prior two for the first comparison
        most_recent_run_index = 1

    directory_index = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)
        for i in range(most_recent_run_index, total_runs):
            most_recent_run = runs["workflow_runs"][i]
            most_recent_artifact_url = most_recent_run["artifacts_url"]
            commit_hash = most_recent_run["head_sha"]
            if download_artifacts(token, most_recent_artifact_url, temp_dir_path, directory_index):
                commit_hash_file = temp_dir_path / str(directory_index) / "commit_hash.txt"
                with open(commit_hash_file, "w") as file:
                    file.write(commit_hash)
                directory_index = directory_index + 1

        total_runs = directory_index
        delete_directory_contents(directory_for_rst_pages)
        diff_results(temp_dir_path, 0, total_runs, directory_for_rst_pages)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token")
    parser.add_argument("--dir")
    token = parser.parse_args().token
    directory_for_rst_pages = parser.parse_args().dir

    download_from_pipeline(token, directory_for_rst_pages)


if __name__ == "__main__":
    main()
