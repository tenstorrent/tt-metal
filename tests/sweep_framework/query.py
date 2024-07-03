# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import click
from pymongo import MongoClient
from test_status import TestStatus
from beautifultable import BeautifulTable
from termcolor import colored


@click.group()
@click.option("--module-name", default=None, help="Name of the module to be queried.")
@click.option("--batch-name", default=None, help="Batch name to filter by.")
@click.option("--vector-id", default=None, help="Individual Vector ID to filter by.")
@click.option("--mongo", default="mongodb://localhost:27017", help="Mongo Connection String")
@click.option("--all", default=False, help="Displays total run statistics instead of the most recent run.")
@click.pass_context
def cli(ctx, module_name, batch_name, vector_id, mongo, all):
    ctx.ensure_object(dict)

    ctx.obj["module_name"] = module_name
    ctx.obj["batch_name"] = batch_name
    ctx.obj["vector_id"] = vector_id
    ctx.obj["mongo"] = mongo
    ctx.obj["all"] = all


def build_match_filters(batch_name, vector_id):
    if not batch_name:
        return [{"$match": {"status": str(x)}} for x in TestStatus]
    elif not vector_id:
        return [{"$match": {"$and": [{"status": str(x)}, {"batch_name": batch_name}]}} for x in TestStatus]
    else:
        return [
            {"$match": {"$and": [{"status": str(x)}, {"batch_name": batch_name}, {"vector_id": vector_id}]}}
            for x in TestStatus
        ]


def get_runs(collection, match_filters, all_filter):
    pass_vectors = list(collection.aggregate([match_filters[0]] + all_filter))
    fail_assert_exception_vectors = list(collection.aggregate([match_filters[1]] + all_filter))
    fail_crash_hang_vectors = list(collection.aggregate([match_filters[2]] + all_filter))
    not_run_vectors = list(collection.aggregate([match_filters[3]] + all_filter))

    return pass_vectors, fail_assert_exception_vectors, fail_crash_hang_vectors, not_run_vectors


def gather_results(ctx):
    client = MongoClient(ctx.obj["mongo"])
    db = client.test_results

    all_filter = [{"$group": {"_id": "$vector_id", "first": {"$max": "$timestamp"}}}] if not ctx.obj["all"] else []

    if not ctx.obj["module_name"]:
        # Table 1, no module specified
        collection_names = db.list_collection_names()
        match_filters = build_match_filters(ctx.obj["batch_name"], ctx.obj["vector_id"])
        for collection_name in collection_names:
            collection = db[collection_name]
            pv, faev, fchv, nrv = get_runs(collection, match_filters, all_filter)
            yield pv, faev, fchv, nrv
    else:
        collection = db[ctx.obj["module_name"] + "_test_results"]
        match_filters = build_match_filters(ctx.obj["batch_name"], ctx.obj["vector_id"])

        pc, fase, fchc, nrc = get_runs(collection_name, match_filters, all_filter)
        yield pv, faev, fchv, nrv


@cli.command()
@click.pass_context
def summary(ctx):
    table = BeautifulTable()
    table.columns.header = [
        colored("PASS", "light_green"),
        colored("FAIL (ASSERT/EXCEPTION)", "light_red"),
        colored("FAIL (CRASH/HANG)", "light_red"),
        colored("NOT RUN", "light_grey"),
    ]

    pv, faev, fchv, nrv = gather_results(ctx)


@cli.command()
@click.pass_context
def detail(ctx):
    results = gather_results(ctx)
    pass


if __name__ == "__main__":
    cli(obj={})
