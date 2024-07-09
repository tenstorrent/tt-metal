# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import click
import os
import pathlib
import pprint
from elasticsearch import Elasticsearch
from tests.sweep_framework.statuses import TestStatus
from serialize import deserialize_vector
from beautifultable import BeautifulTable, STYLE_COMPACT
from termcolor import colored

ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")


@click.group()
@click.option("--module-name", default=None, help="Name of the module to be queried.")
@click.option("--batch-name", default=None, help="Batch name to filter by.")
@click.option("--vector-id", default=None, help="Individual Vector ID to filter by.")
@click.option("--run-id", default=None, help="Individual Run ID to filter by.")
@click.option("--elastic", default="http://localhost:9200", help="Elastic Connection String")
@click.option("--all", default=False, help="Displays total run statistics instead of the most recent run.")
@click.pass_context
def cli(ctx, module_name, batch_name, vector_id, run_id, elastic, all):
    ctx.ensure_object(dict)

    ctx.obj["module_name"] = module_name
    ctx.obj["batch_name"] = batch_name
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
    pprint.pp(deserialize_vector(response["_source"]))


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
    table = BeautifulTable(maxwidth=200)
    table.columns.header = [
        colored("PASS", "light_green"),
        colored("FAIL (ASSERT/EXCEPTION)", "light_red"),
        colored("FAIL (CRASH/HANG)", "light_red"),
        colored("NOT RUN", "light_grey"),
    ]

    client = Elasticsearch(ctx.obj["elastic"], basic_auth=("elastic", ELASTIC_PASSWORD))
    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    if ctx.obj["module_name"] is None:
        for file in sorted(sweeps_path.glob("*.py")):
            module_name = str(pathlib.Path(file).relative_to(sweeps_path))[:-3]
            results_index = module_name + "_test_results"
            if not ctx.obj["all"]:
                response = client.search(
                    index=results_index,
                    size=0,
                    aggs={
                        "latest_runs": {
                            "terms": {"field": "vector_id.keyword"},
                            "aggs": {
                                "latest_timestamp": {
                                    "top_hits": {"sort": [{"timestamp.keyword": {"order": "desc"}}], "size": 1}
                                }
                            },
                        }
                    },
                )["aggregations"]["latest_runs"]["buckets"]
                passes = sum(
                    bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(TestStatus.PASS)
                    for bucket in response
                )
                fail_ae = sum(
                    bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"]
                    == str(TestStatus.FAIL_ASSERT_EXCEPTION)
                    for bucket in response
                )
                fail_ch = sum(
                    bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"]
                    == str(TestStatus.FAIL_CRASH_HANG)
                    for bucket in response
                )
                not_run = sum(
                    bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(TestStatus.NOT_RUN)
                    for bucket in response
                )
            else:
                passes = client.count(index=results_index, query={"match": {"status": str(TestStatus.PASS)}})["count"]
                fail_ae = client.count(
                    index=results_index, query={"match": {"status": str(TestStatus.FAIL_ASSERT_EXCEPTION)}}
                )["count"]
                fail_ch = client.count(
                    index=results_index, query={"match": {"status": str(TestStatus.FAIL_CRASH_HANG)}}
                )["count"]
                not_run = client.count(index=results_index, query={"match": {"status": str(TestStatus.NOT_RUN)}})[
                    "count"
                ]

            table.rows.append([passes, fail_ae, fail_ch, not_run], module_name)
    elif ctx.obj["batch_name"] is None:
        results_index = ctx.obj["module_name"] + "_test_results"
        if not ctx.obj["all"]:
            response = client.search(
                index=results_index,
                size=0,
                aggs={
                    "group_by_batch_name": {
                        "terms": {"field": "batch_name.keyword", "size": 50},
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
            )["aggregations"]["group_by_batch_name"]["buckets"]
            for bucket in response:
                passes = sum(
                    bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(TestStatus.PASS)
                    for bucket in bucket["group_by_vector_id"]["buckets"]
                )
                fail_ae = sum(
                    bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"]
                    == str(TestStatus.FAIL_ASSERT_EXCEPTION)
                    for bucket in bucket["group_by_vector_id"]["buckets"]
                )
                fail_ch = sum(
                    bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"]
                    == str(TestStatus.FAIL_CRASH_HANG)
                    for bucket in bucket["group_by_vector_id"]["buckets"]
                )
                not_run = sum(
                    bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(TestStatus.NOT_RUN)
                    for bucket in bucket["group_by_vector_id"]["buckets"]
                )
                table.rows.append([passes, fail_ae, fail_ch, not_run], bucket["key"])
        else:
            response = client.search(
                index=results_index,
                size=0,
                aggs={"group_by_batch_name": {"terms": {"field": "batch_name.keyword", "size": 50}}},
            )
            batches = [bucket["key"] for bucket in response["aggregations"]["group_by_batch_name"]["buckets"]]
            for batch in batches:
                passes = client.count(
                    index=results_index,
                    query={
                        "bool": {
                            "must": [{"match": {"status": str(TestStatus.PASS)}}, {"match": {"batch_name": batch}}]
                        }
                    },
                )["count"]
                fail_ae = client.count(
                    index=results_index,
                    query={
                        "bool": {
                            "must": [
                                {"match": {"status": str(TestStatus.FAIL_ASSERT_EXCEPTION)}},
                                {"match": {"batch_name": batch}},
                            ]
                        }
                    },
                )["count"]
                fail_ch = client.count(
                    index=results_index,
                    query={
                        "bool": {
                            "must": [
                                {"match": {"status": str(TestStatus.FAIL_CRASH_HANG)}},
                                {"match": {"batch_name": batch}},
                            ]
                        }
                    },
                )["count"]
                not_run = client.count(
                    index=results_index,
                    query={
                        "bool": {
                            "must": [{"match": {"status": str(TestStatus.NOT_RUN)}}, {"match": {"batch_name": batch}}]
                        }
                    },
                )["count"]
                table.rows.append([passes, fail_ae, fail_ch, not_run], batch)

    elif ctx.obj["vector_id"] is None:
        results_index = ctx.obj["module_name"] + "_test_results"
        response = client.search(
            index=results_index,
            size=0,
            query={"match": {"batch_name": ctx.obj["batch_name"]}},
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
        for bucket in response:
            passes = bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(TestStatus.PASS)
            fail_ae = bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(
                TestStatus.FAIL_ASSERT_EXCEPTION
            )
            fail_ch = bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(
                TestStatus.FAIL_CRASH_HANG
            )
            not_run = bucket["latest_timestamp"]["hits"]["hits"][0]["_source"]["status"] == str(TestStatus.NOT_RUN)
            table.rows.append([passes, fail_ae, fail_ch, not_run], bucket["key"])

    print(table)
    client.close()


@cli.command()
@click.pass_context
def detail(ctx):
    table = BeautifulTable(maxwidth=200)
    table.set_style(STYLE_COMPACT)
    table.columns.header = [
        colored("Sweep", "light_red"),
        colored("Batch", "light_red"),
        colored("Vector ID", "light_red"),
        colored("Timestamp", "light_red"),
        colored("Status", "light_red"),
        colored("Details", "light_red"),
        colored("Git Hash", "light_red"),
    ]

    client = Elasticsearch(ctx.obj["elastic"], basic_auth=("elastic", ELASTIC_PASSWORD))
    sweeps_path = pathlib.Path(__file__).parent / "sweeps"

    matches = []
    if ctx.obj["batch_name"]:
        matches.append({"match": {"batch_name": ctx.obj["batch_name"]}})
    if ctx.obj["vector_id"]:
        matches.append({"match": {"vector_id": ctx.obj["vector_id"]}})

    def add_results_for_module(module_name):
        results_index = module_name + "_test_results"
        results = client.search(index=results_index, size=10000, query={"bool": {"must": matches}})["hits"]["hits"]
        for result in results:
            source = result["_source"]
            try:
                detail = source["message"]
            except:
                detail = str(source["exception"]).replace("\n", " ")[:50]
            table.rows.append(
                [
                    module_name,
                    source["batch_name"],
                    source["vector_id"],
                    source["timestamp"],
                    source["status"][11:],
                    detail,
                    source["git_hash"],
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
