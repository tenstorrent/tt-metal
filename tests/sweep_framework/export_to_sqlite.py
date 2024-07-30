# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from elasticsearch import Elasticsearch
import sqlite3
import argparse
import pathlib
import os
import sys

ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Elasticsearch to Sqlite Util",
        description="Dump all vectors and results to sqlite.",
    )

    parser.add_argument(
        "--elastic",
        required=False,
        default="http://yyz-elk:9200",
        help="Elastic Connection String for the vector and results database.",
    )
    args = parser.parse_args(sys.argv[1:])

    ELASTIC_CONNECTION_STRING = args.elastic
    DUMP_PATH = "tests/sweep_framework/sqlite_dump/"
    sweeps_path = pathlib.Path(__file__).parent / "sweeps"
    es_client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))

    for file in sorted(sweeps_path.glob("*.py")):
        sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3]
        sqlite_client = sqlite3.connect(DUMP_PATH + sweep_name + ".sqlite")
        vector_index = sweep_name + "_test_vectors"
        result_index = sweep_name + "_test_results"

        response = es_client.search(index=vector_index, size=10000)
        test_ids = [hit["_id"] for hit in response["hits"]["hits"]]
        test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]
        for i in range(len(test_vectors)):
            test_vectors[i]["vector_id"] = test_ids[i]
        column_names = tuple(test_vectors[0].keys())
        table_string = f"CREATE TABLE IF NOT EXISTS {vector_index} {column_names}"
        sqlite_client.execute(table_string)
        placeholders = ("?, " * len(test_vectors[0])).strip()[:-1]
        for vector in test_vectors:
            value_string = f"INSERT INTO {vector_index} VALUES ({placeholders})"
            try:
                sqlite_client.execute(value_string, tuple(str(value) for value in vector.values()))
                sqlite_client.commit()
            except Exception as e:
                print(e)

        response = es_client.search(index=result_index, size=10000)
        result_ids = [hit["_id"] for hit in response["hits"]["hits"]]
        test_vectors = [hit["_source"] for hit in response["hits"]["hits"]]
        for i in range(len(test_vectors)):
            test_vectors[i]["run_id"] = result_ids[i]
        column_names = tuple(test_vectors[0].keys())
        table_string = f"CREATE TABLE IF NOT EXISTS {result_index} {column_names}"
        sqlite_client.execute(table_string)
        placeholders = ("?, " * len(test_vectors[0])).strip()[:-1]
        for vector in test_vectors:
            value_string = f"INSERT INTO {result_index} VALUES ({placeholders})"
            try:
                sqlite_client.execute(value_string, tuple(str(value) for value in vector.values()))
                sqlite_client.commit()
            except Exception as e:
                print(e)

        sqlite_client.close()
