# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import click
import os
import pathlib
import pprint
from elasticsearch import Elasticsearch, NotFoundError
from tests.sweep_framework.statuses import TestStatus
from beautifultable import BeautifulTable, STYLE_COMPACT
from termcolor import colored

ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")


@click.group()
@click.option("--module-name", default=None, help="Name of the module to be queried.")
@click.option("--suite-name", default=None, help="Suite name to filter by.")
@click.option("--vector-id", default=None, help="Individual Vector ID to filter by.")
@click.option("--run-id", default=None, help="Individual Run ID to filter by.")
@click.option("--elastic", default="http://localhost:9200", help="Elastic Connection String")
@click.option(
    "--all", is_flag=True, default=False, help="Displays total run statistics instead of the most recent run."
)
@click.pass_context
def cli(ctx, module_name, suite_name, vector_id, run_id, elastic, all):
    ctx.ensure_object(dict)

    ctx.obj["module_name"] = module_name
    ctx.obj["suite_name"] = suite_name
    ctx.obj["vector_id"] = vector_id
    ctx.obj["run_id"] = run_id
    ctx.obj["elastic"] = elastic
    ctx.obj["all"] = all


@cli.command()
@click.pass_context
def vector(ctx):
    if not ctx.obj["module_name"] or not ctx.obj["vector_id"]:
        print("QUERY: Module name and vector ID are required for test vector lookup.")
        exit(1)

    client = Elasticsearch(ctx.obj["elastic"], basic_auth=("elastic", ELASTIC_PASSWORD))
    response = client.get(index=(ctx.obj["module_name"] + "_test_vectors"), id=ctx.obj["vector_id"])
    pprint.pp(response["_source"])


@cli.command()
@click.pass_context
def result(ctx):
    if not ctx.obj["module_name"] or not ctx.obj["run_id"]:
        print("QUERY: Module name and run ID are required for run result lookup.")
        exit(1)

    client = Elasticsearch(ctx.obj["elastic"], basic_auth=("elastic", ELASTIC_PASSWORD))
    response = client.get(index=(ctx.obj["module_name"] + "_test_results"), id=ctx.obj["run_id"])
    pprint.pp(response["_source"])


@cli.command()
@click.pass_context
def summary(ctx):
    table = BeautifulTable(maxwidth=os.get_terminal_size().columns)
    table.columns.header = [
        colored("PASS", "light_green"),
        colored("FAIL (ASSERT/EXCEPTION)", "light_red"),
        colored("FAIL (CRASH/HANG)", "light_red"),
        colored("NOT RUN", "light_grey"),
        colored("FAIL (L1 Out of Mem)", "light_red"),
        colored("FAIL (Watcher)", "light_red"),
    ]

    client = Elasticsearch(ctx.obj["elastic"], basic_auth=("elastic", ELASTIC_PASSWORD))
    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    if ctx.obj["module_name"] is None:
        for file in sorted(sweeps_path.glob("*.py")):
            module_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3]
            results_index = module_name + "_test_results"
            if not client.indices.exists(index=results_index):
                continue
            if not ctx.obj["all"]:
                response = client.search(
                    index=results_index,
                    size=0,
                    aggs={
                        "latest_runs": {
                            "terms": {"field": "vector_id.keyword", "size": 10000},
                            "aggs": {
                                "latest_timestamp": {
                                    "top_hits": {"sort": [{"timestamp.keyword": {"order": "desc"}}], "size": 1}
                                }
                            },
                        }
                    },
                )["aggregations"]["latest_runs"]["buckets"]
                row = []
                for status in TestStatus:
                    row.append(
                        sum(
                            bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(status)
                            for bucket in response
                        )
                    )
            else:
                row = []
                for status in TestStatus:
                    row.append(client.count(index=results_index, query={"match": {"status": str(status)}})["count"])

            table.rows.append(row, module_name)
    elif ctx.obj["suite_name"] is None:
        module_name = ctx.obj["module_name"]
        results_index = module_name + "_test_results"
        if not client.indices.exists(index=results_index):
            print(f"SWEEPS: There are no results for module {module_name}.")
            return
        if not ctx.obj["all"]:
            response = client.search(
                index=results_index,
                size=0,
                aggs={
                    "group_by_suite_name": {
                        "terms": {"field": "suite_name.keyword", "size": 50},
                        "aggs": {
                            "group_by_vector_id": {
                                "terms": {"field": "vector_id.keyword", "size": 10000},
                                "aggs": {
                                    "latest_timestamp": {
                                        "top_hits": {"sort": [{"timestamp.keyword": {"order": "desc"}}], "size": 1}
                                    }
                                },
                            }
                        },
                    }
                },
            )["aggregations"]["group_by_suite_name"]["buckets"]
            for bucket in response:
                row = []
                for status in TestStatus:
                    row.append(
                        sum(
                            bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(status)
                            for bucket in bucket["group_by_vector_id"]["buckets"]
                        )
                    )
                table.rows.append(row, bucket["key"])
        else:
            response = client.search(
                index=results_index,
                size=0,
                aggs={"group_by_suite_name": {"terms": {"field": "suite_name.keyword", "size": 50}}},
            )
            suites = [bucket["key"] for bucket in response["aggregations"]["group_by_suite_name"]["buckets"]]
            for suite in suites:
                row = []
                for status in TestStatus:
                    row.append(
                        client.count(
                            index=results_index,
                            query={
                                "bool": {"must": [{"match": {"status": str(status)}}, {"match": {"suite_name": suite}}]}
                            },
                        )["count"]
                    )

                table.rows.append(row, suite)

    else:
        module_name = ctx.obj["module_name"]
        results_index = module_name + "_test_results"
        suite_name = ctx.obj["suite_name"]
        if not client.indices.exists(index=results_index):
            print(f"SWEEPS: There are no results for module {module_name}.")
            return
        if not ctx.obj["all"]:
            response = client.search(
                index=results_index,
                size=0,
                query={"match": {"suite_name": suite_name}},
                aggs={
                    "group_by_vector_id": {
                        "terms": {"field": "vector_id.keyword", "size": 10000},
                        "aggs": {
                            "latest_timestamp": {
                                "top_hits": {"sort": [{"timestamp.keyword": {"order": "desc"}}], "size": 1}
                            }
                        },
                    }
                },
            )["aggregations"]["group_by_vector_id"]["buckets"]
            if len(response) == 0:
                print(f"SWEEPS: There are no results for module {module_name}, suite {suite_name}")
            for bucket in response:
                row = []
                for status in TestStatus:
                    row.append(bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(status))
                table.rows.append(row, bucket["key"])
        else:
            response = client.search(
                index=results_index,
                size=10000,
                query={"match": {"suite_name": suite_name}},
                aggs={
                    "group_by_vector_id": {
                        "terms": {"field": "vector_id.keyword", "size": 10000},
                    }
                },
            )["aggregations"]["group_by_vector_id"]["buckets"]
            for bucket in response:
                row = []
                for status in TestStatus:
                    row.append(
                        client.count(
                            query={
                                "bool": {
                                    "must": [
                                        {"match": {"status": str(status)}},
                                        {"match": {"vector_id": bucket["key"]}},
                                    ]
                                }
                            }
                        )["count"]
                    )
                table.rows.append(row, bucket["key"])
    print(table)
    client.close()


@cli.command()
@click.pass_context
def detail(ctx):
    table = BeautifulTable(maxwidth=400)
    table.set_style(STYLE_COMPACT)
    table.columns.header = [
        colored("Sweep", "light_red"),
        colored("Suite", "light_red"),
        colored("Vector ID", "light_red"),
        colored("Timestamp", "light_red"),
        colored("Status", "light_red"),
        colored("Details", "light_red"),
        colored("Git Hash", "light_red"),
        colored("e2e Perf", "light_red"),
    ]

    client = Elasticsearch(ctx.obj["elastic"], basic_auth=("elastic", ELASTIC_PASSWORD))
    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    matches = []
    if ctx.obj["suite_name"]:
        matches.append({"match": {"suite_name": ctx.obj["suite_name"]}})
    if ctx.obj["vector_id"]:
        matches.append({"match": {"vector_id": ctx.obj["vector_id"]}})

    def add_results_for_module(module_name):
        results_index = module_name + "_test_results"
        results = client.search(
            index=results_index,
            size=10000,
            sort=[{"timestamp.keyword": {"order": "asc"}}],
            query={"bool": {"must": matches}},
        )["hits"]["hits"]
        for result in results:
            source = result["_source"]
            try:
                detail = "PCC: " + source["message"]
            except:
                detail = str(source["exception"]).replace("\n", " ")[:50]
            table.rows.append(
                [
                    module_name,
                    source["suite_name"],
                    source["vector_id"],
                    source["timestamp"],
                    source["status"][11:],
                    detail,
                    source["git_hash"],
                    source["e2e_perf"] + " ms" if "e2e_perf" in source else "None",
                ],
                result["_id"],
            )

    if not ctx.obj["module_name"]:
        for file in sorted(sweeps_path.glob("*.py")):
            module_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3]
            add_results_for_module(module_name)
    else:
        add_results_for_module(ctx.obj["module_name"])

    print(table)
    client.close()


if __name__ == "__main__":
    cli(obj={})
