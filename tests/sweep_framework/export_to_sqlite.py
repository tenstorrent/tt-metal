# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from elasticsearch import Elasticsearch
import argparse
import pathlib
import os
import sqlite3
import sys
from elastic_config import *


def log(string):
    if VERBOSE:
        print(string)


def get_vector_query(sweep_name, suite_name, user):
    query = {
        "bool": {
            "filter": [
                {"term": {"sweep_name.keyword": {"value": sweep_name}}},
                {"term": {"suite_name.keyword": {"value": suite_name}}},
                {"term": {"tag.keyword": {"value": user}}},
                {"term": {"status.keyword": {"value": "VectorStatus.CURRENT"}}},
            ]
        }
    }
    return query


def get_result_query(sweep_name, suite_name, user):
    query = {
        "bool": {
            "filter": [
                {"term": {"sweep_name.keyword": {"value": sweep_name}}},
                {"term": {"suite_name.keyword": {"value": suite_name}}},
                {"term": {"user.keyword": {"value": user}}},
            ]
        }
    }
    return query


def get_result_aggs():
    aggs = {
        "group_by_vector_id": {
            "terms": {"field": "vector_id.keyword", "size": 10000},
            "aggs": {"latest_timestamp": {"top_hits": {"sort": [{"timestamp": {"order": "desc"}}], "size": 1}}},
        }
    }
    return aggs


def get_suites(vector_index, user, es_client):
    try:
        response = es_client.search(
            index=vector_index,
            query={"match": {"tag.keyword": user}},
            aggregations={"suites": {"terms": {"field": "suite_name.keyword", "size": 10000}}},
        )
        suites = [suite["key"] for suite in response["aggregations"]["suites"]["buckets"]]
    except Exception as e:
        print(f"EXCEPTION retrieving suites for {vector_index} with user {user}: {e}")
        return []
    return suites


def join_entries(vector_entries, result_entries):
    # Note: Will only save data for which vector entries exist.
    joined_entries = {}
    results_per_id = {}
    for entry in vector_entries:
        vector_id = entry["vector_id"]
        joined_entries[vector_id] = entry
        results_per_id[vector_id] = 0

    missing_vectors = []
    for entry in result_entries:
        vector_id = entry["vector_id"]
        if vector_id not in joined_entries:
            missing_vectors.append(vector_id)
        else:
            joined_entries[vector_id].update(entry)
            results_per_id[vector_id] += 1

    missing_results = []
    multiple_results = []
    for key, value in results_per_id.items():
        if value == 0:
            missing_results.append[key]
        elif value > 1:
            multiple_results.append[key]

    log(
        f"Joined {len(vector_entries)} vector entries with {len(result_entries)} result entries to create {len(joined_entries)} entries. There are {len(missing_vectors)} missing vectors (results for missing vectors are ignored), {len(missing_results)} missing results, and {len(multiple_results)} multiple results.\nMissing results: {missing_results}\nMultiple results: {multiple_results}"
    )
    return list(joined_entries.values())


def write_table(entries, table, sqlite_client):
    # Different entries may have different number of keys. Find the union of all keys.
    column_set = set()
    for entry in entries:
        column_set = column_set.union(entry.keys())

    # Create a map from column names to column indices
    column_count = len(column_set)
    columns = dict()
    column_names = []
    i = 0
    for column in column_set:
        column_names.append(column)
        columns[column] = i
        i += 1

    # Generate sqlite data by putting the values of each entry into the right column
    sqlite_data = []
    missing_columns = {}
    for entry in entries:
        row = ["" for i in range(column_count)]
        for key, value in entry.items():
            row[columns[key]] = str(value)
        sqlite_data.append(row)
        difference = column_count - len(entry)
        if difference > 0:
            if difference in missing_columns:
                missing_columns[difference] += 1
            else:
                missing_columns[difference] = 1

    # Write to the database, deleting the table if it exists
    placeholders = ("?, " * column_count).strip()[:-1]
    log(
        f"Writing table {table} with {len(sqlite_data)} rows and columns {column_names}. Mapping of missing column count to number of rows missing that number of columns: {missing_columns}"
    )
    drop_string = f"DROP TABLE IF EXISTS {table}"
    create_string = f"CREATE TABLE {table} {tuple(column_names)}"
    value_string = f"INSERT INTO {table} VALUES ({placeholders})"
    commands = 0
    try:
        sqlite_client.execute(drop_string)
        commands += 1
        sqlite_client.execute(create_string)
        commands += 1
        sqlite_client.executemany(value_string, sqlite_data)
    except Exception as e:
        print(f"EXCEPTION on SQLITE command #{commands} for table {table}: {e}")


def export_to_sqlite(sweeps_path, dump_path, filter_string, user, es_client):
    for file in sorted(sweeps_path.glob("**/*.py")):
        sweep_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3].replace("/", ".")
        if filter_string is not None and not filter_string in sweep_name:
            continue
        sweep_name_path = str(pathlib.Path(file).relative_to(sweeps_path))[:-3]
        sqlite_file = dump_path + "/" + sweep_name_path + ".sqlite"
        sqlite_dir = sqlite_file[0 : sqlite_file.rfind("/")]
        table_prefix = sweep_name.replace(".", "_")
        print(
            f"Processing file={file} sweeps_path={sweeps_path} sweep_name={sweep_name} sweep_name_path={sweep_name_path} sqlite_file={sqlite_file} sqlite_dir={sqlite_dir} table_prefix={table_prefix}"
        )
        pathlib.Path(sqlite_dir).mkdir(parents=True, exist_ok=True)
        sqlite_connection = sqlite3.connect(sqlite_file)
        sqlite_client = sqlite_connection.cursor()
        vector_index = VECTOR_INDEX_PREFIX + sweep_name
        result_index = RESULT_INDEX_PREFIX + sweep_name
        vector_table = table_prefix + "_vector"
        result_table = table_prefix + "_result"
        joined_table = table_prefix
        suites = get_suites(vector_index, user, es_client)
        for suite_name in suites:
            vector_query = get_vector_query(sweep_name, suite_name, user)
            vector_response = es_client.search(index=vector_index, size=10000, query=vector_query)
            if len(vector_response) == 0:
                log(f"WARNING: No vectors for sweep {sweep_name} and suite {suite_name}. Skipping.")
                continue
            vector_ids = [hit["_id"] for hit in vector_response["hits"]["hits"]]
            vector_entries = [hit["_source"] for hit in vector_response["hits"]["hits"]]
            vector_count = len(vector_entries)
            for i in range(vector_count):
                vector_entries[i]["vector_id"] = vector_ids[i]
            result_query = get_result_query(sweep_name, suite_name, user)
            result_aggs = get_result_aggs()
            result_response = es_client.search(index=result_index, size=10000, aggs=result_aggs, query=result_query)
            result_entries = [
                bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]
                for bucket in result_response["aggregations"]["group_by_vector_id"]["buckets"]
            ]
            result_count = len(result_entries)
            log(
                f"Processing suite {suite_name} with vector response length {len(vector_response)}, {vector_count} vector entries, result response length {len(result_response)}, and {result_count} result entries"
            )
            if vector_count >= 10000:
                log(f"WARNING: vectors reached limit of elastic query length. Data may be lost")
            if result_count >= 10000:
                log(f"WARNING: results reached limit of elastic query length. Data may be lost")
            write_table(vector_entries, vector_table, sqlite_client)
            write_table(result_entries, result_table, sqlite_client)
            write_table(join_entries(vector_entries, result_entries), joined_table, sqlite_client)
        try:
            sqlite_connection.commit()
            sqlite_connection.close()
        except Exception as e:
            print(f"EXCEPTION on SQLITE Connection for {sweep_name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Elasticsearch to Sqlite Util",
        description="Dump all vectors and results to sqlite.",
    )

    parser.add_argument(
        "--elastic",
        required=False,
        default="corp",
        help="Elastic Connection String for the vector and results database. Available presets are ['corp', 'cloud']",
    )
    parser.add_argument(
        "--user",
        required=False,
        default=os.getenv("USER"),
        help="User to use to filter vectors via tag.keyword and results via user.keyword",
    )
    parser.add_argument("--dump_path", required=False, default=os.getenv("HOME"), help="Path to store the sqlite data")
    parser.add_argument(
        "--filter-string", required=False, help="Module names will only be processed if they contain this filter string"
    )
    parser.add_argument("--verbose", required=False, help="Whether output is verbose.")
    args = parser.parse_args(sys.argv[1:])

    ELASTIC_CONNECTION_STRING = get_elastic_url(args.elastic)
    sweeps_path = pathlib.Path(__file__).parent / "sweeps"
    es_client = Elasticsearch(ELASTIC_CONNECTION_STRING, basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    global VERBOSE
    VERBOSE = args.verbose
    export_to_sqlite(sweeps_path, args.dump_path, args.filter_string, args.user, es_client)
