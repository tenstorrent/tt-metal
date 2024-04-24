# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import argparse
import requests
import tempfile
import pathlib
import zipfile
import pandas as pd
from loguru import logger


def get_list_of_runs():
    params = {"per_page": 3}
    url = "https://api.github.com/repos/tenstorrent/tt-metal/actions/workflows/ttnn-run-sweeps.yaml/runs"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        runs = response.json()
    else:
        raise RuntimeError(f"Error fetching workflow runs: {response.status_code}")

    return runs


def download_artifacts(token, artifacts_url, output_path):
    response = requests.get(artifacts_url)
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    if response.status_code == 200:
        artifacts_data = response.json()
        if artifacts_data["artifacts"]:
            artifact = artifacts_data["artifacts"][0]
            artifact_download_url = artifact["archive_download_url"]
            artifact_response = requests.get(artifact_download_url, headers=headers)
            if artifact_response.status_code == 200:
                with open(output_path, "wb") as file:
                    file.write(artifact_response.content)
                logger.info(f"{artifacts_url} downloaded successfully.")
                return True
            else:
                raise RuntimeError("Failed to download the artifact.")
        else:
            return False
    else:
        raise RuntimeError("Failed to fetch artifacts list.")


def read_csv_from_zip(zip_file, file_name):
    with zip_file.open(file_name) as f:
        return pd.read_csv(f)


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


def diff_results(recent_zip, prior_zip, directory_for_html_pages, commit_hash):
    directory_for_html_pages = pathlib.Path(directory_for_html_pages)
    html_files = []
    html_failure_files = []
    failures_since_last_run = 0
    with zipfile.ZipFile(recent_zip, "r") as zip1, zipfile.ZipFile(prior_zip, "r") as zip2:
        zip1_files = set(zip1.namelist())
        zip2_files = set(zip2.namelist())
        common_files = zip1_files.intersection(zip2_files)
        # pd.set_option("display.max_rows", None)
        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.width", None)
        # pd.set_option("display.max_colwidth", 10)
        for file_name in common_files:
            test_name = pathlib.Path(file_name).stem
            if file_name.endswith(".csv"):
                recent_df = read_csv_from_zip(zip1, file_name)
                html_table = recent_df.to_html()
                html_page_name = directory_for_html_pages / f"{test_name}.html"
                with open(html_page_name, "w") as f:
                    f.write(html_table)
                html_files.append(f"{test_name}.html")
                prior_df = read_csv_from_zip(zip2, file_name)
                failures_df = build_new_failures(recent_df, prior_df)
                crashes_df = build_new_crashes(recent_df, prior_df)
                combined_test_resutls_df = pd.concat([failures_df, crashes_df])
                if combined_test_resutls_df.size > 0:
                    failures_since_last_run = failures_since_last_run + combined_test_resutls_df.size
                    html_table = combined_test_resutls_df.to_html()
                    html_page_name = directory_for_html_pages / f"{test_name}_failure.html"
                    with open(html_page_name, "w") as f:
                        f.write(html_table)
                    html_failure_files.append(f"{test_name}_failure.html")

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sweep Test Results</title>
        <style>
            iframe {{
                width: 100%;
                height: 300px;
                border: none;
                margin-bottom: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Sweep Tests</h1>
        <h2>We have had {failures_since_last_run} failures since the prior run.</h2>
        <h2>Commit Hash: {commit_hash}</h2>
        <br/>
        {iframes}
    </body>
    </html>
    """

    iframe_tags = "".join(
        [f'<h3>{file.split(".")[0]}</h3><iframe src="{file}"></iframe>' for file in html_failure_files]
    )
    complete_html = html_template.format(
        commit_hash=commit_hash, failures_since_last_run=failures_since_last_run, iframes=iframe_tags
    )
    html_page_name = directory_for_html_pages / f"index.html"
    with open(html_page_name, "w") as file:
        file.write(complete_html)

    logger.info(f"Built {html_page_name}")


def download_from_pipeline(token, directory_for_html_pages):
    """
    Download the results of the sweeps from the GitHub pipeline.

    :param token: Provide your GitHub token.
    """

    runs = get_list_of_runs()
    if len(runs["workflow_runs"]) < 3:
        # Note that if the run is in progress, there will not be any artifacts avaiable yet on the most recent run.
        raise RuntimeError("We need at least three runs to compare the changes in the sweep tests")

    if runs["workflow_runs"][0]["status"] == "completed":
        most_recent_run = runs["workflow_runs"][0]
        prior_run = runs["workflow_runs"][1]
    else:
        most_recent_run = runs["workflow_runs"][1]
        prior_run = runs["workflow_runs"][2]

    most_recent_artifact_url = most_recent_run["artifacts_url"]
    commit_hash = most_recent_run["head_sha"]
    prior_artifact_url = prior_run["artifacts_url"]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = pathlib.Path(temp_dir)
        recent_zip = temp_dir_path / "recent.zip"
        prior_zip = temp_dir_path / "prior.zip"
        downloaded = download_artifacts(token, most_recent_artifact_url, output_path=recent_zip)
        if downloaded:
            downloaded = download_artifacts(token, prior_artifact_url, output_path=prior_zip)
        if not downloaded:
            return
        diff_results(recent_zip, prior_zip, directory_for_html_pages, commit_hash)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token")
    parser.add_argument("--dir")
    token = parser.parse_args().token
    directory_for_html_pages = parser.parse_args().dir
    download_from_pipeline(token, directory_for_html_pages)


if __name__ == "__main__":
    main()
