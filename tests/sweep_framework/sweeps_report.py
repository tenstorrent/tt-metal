# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import click
import os
import pathlib
import pprint
import json
import hashlib
from elasticsearch import Elasticsearch, NotFoundError
from tests.sweep_framework.framework.statuses import TestStatus
from beautifultable import BeautifulTable, STYLE_COMPACT
from termcolor import colored
from framework.elastic_config import *
from framework.sweeps_logger import sweeps_logger as logger
from datetime import datetime


@click.group()
@click.option("--module-name", default=None, help="Name of the module to be queried.")
@click.option("--suite-name", default=None, help="Suite name to filter by.")
@click.option("--vector-id", default=None, help="Individual Vector ID to filter by.")
@click.option("--elastic", default="corp", help="Elastic Connection String. Available presets are ['corp', 'cloud']")
@click.option(
    "--all", is_flag=True, default=False, help="Displays total run statistics instead of the most recent run."
)
@click.pass_context
def cli(ctx, module_name, suite_name, vector_id, elastic, all):
    ctx.ensure_object(dict)

    ctx.obj["module_name"] = module_name
    ctx.obj["suite_name"] = suite_name
    ctx.obj["vector_id"] = vector_id
    ctx.obj["elastic"] = get_elastic_url(elastic)
    ctx.obj["all"] = all


def get_input_params(vector):
    keys_to_remove = [
        "sweep_name",
        "suite_name",
        "vector_id",
        "input_hash",
        "timestamp",
        "tag",
        "invalid_reason",
        "status",
        "validity",
    ]
    for key in keys_to_remove:
        if key in vector:
            vector.pop(key)
    logger.debug(json.dumps(vector, indent=2))
    return vector


def get_vector_details(es, module_name, vector_id):
    vector_index = VECTOR_INDEX_PREFIX + module_name
    response = es.get(index=vector_index, id=vector_id)
    if response and response.get("found"):
        return get_input_params(response["_source"])
    return None


def create_hash(data):
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()


def store_issue_in_elasticsearch(es, index, hash_value, issue_numbers):
    existing_doc = es.options(ignore_status=[404]).get(index=RESULT_INDEX_PREFIX + "hash_table_index", id=hash_value)

    if existing_doc and "_source" in existing_doc:
        existing_issues = set(existing_doc["_source"].get("issues", []))
        existing_issues.update(issue_numbers)
    else:
        existing_issues = set(issue_numbers)

    doc = {"hash": hash_value, "issues": list(existing_issues), "created_at": datetime.utcnow().isoformat() + "Z"}
    es.index(index=index, id=hash_value, body=doc)
    logger.info(f"Updated hash {hash_value} with issues {list(existing_issues)} in Elasticsearch.")


def process_vectors(es, vector_ids, module_name, issue_numbers):
    for vector_id in vector_ids:
        vector_details = get_vector_details(es, module_name, vector_id)
        if not vector_details:
            logger.warning(f"Vector ID {vector_id} not found.")
            continue

        logger.debug(f"Vector Details for {vector_id}: {json.dumps(vector_details, indent=2)}")
        hash_value = create_hash(vector_details)
        logger.debug(f"hash_value: {hash_value}")
        store_issue_in_elasticsearch(es, RESULT_INDEX_PREFIX + "hash_table_index", hash_value, issue_numbers)


@cli.command()
@click.option("--vector-ids", multiple=True, required=True, help="Vector IDs")
@click.option("--module", required=True, help="Module Name")
@click.option("--issues", multiple=True, required=True, help="GitHub Issue Numbers")
@click.pass_context
def associate_issues_with_vectors(ctx, vector_ids, module, issues):
    es = Elasticsearch(ctx.obj["elastic"], basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    process_vectors(es, list(vector_ids), module, list(issues))
    es.close()


@cli.command()
@click.option("--vector-ids", multiple=True, required=True, help="Vector IDs")
@click.option("--module", required=True, help="Module Name")
@click.option("--issues", multiple=True, required=True, help="GitHub Issue Numbers")
@click.pass_context
def disassociate_issues_from_vectors(ctx, vector_ids, module, issues):
    es = Elasticsearch(ctx.obj["elastic"], basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    for vector_id in vector_ids:
        vector_details = get_vector_details(es, module, vector_id)
        if not vector_details:
            logger.warning(f"Vector ID {vector_id} not found.")
            continue

        hash_value = create_hash(vector_details)
        existing_doc = es.options(ignore_status=[404]).get(
            index=RESULT_INDEX_PREFIX + "hash_table_index", id=hash_value
        )
        if existing_doc and "_source" in existing_doc:
            existing_issues = set(existing_doc["_source"].get("issues", []))
            updated_issues = existing_issues.difference(issues)
            doc = {
                "hash": hash_value,
                "issues": list(updated_issues),
                "created_at": datetime.utcnow().isoformat() + "Z",
            }
            es.index(index=RESULT_INDEX_PREFIX + "hash_table_index", id=hash_value, body=doc)
            logger.info(f"Updated hash {hash_value} with issues {list(updated_issues)} in Elasticsearch.")
        else:
            logger.warning(f"No existing document found for hash {hash_value}.")
    es.close()


@cli.command()
@click.option("--vector-ids", multiple=True, required=True, help="Vector IDs")
@click.option("--module", required=True, help="Module Name")
@click.pass_context
def show_issue_associations(ctx, vector_ids, module):
    es = Elasticsearch(ctx.obj["elastic"], basic_auth=(ELASTIC_USERNAME, ELASTIC_PASSWORD))
    for vector_id in vector_ids:
        vector_details = get_vector_details(es, module, vector_id)
        if not vector_details:
            logger.warning(f"Vector ID {vector_id} not found.")
            continue

        logger.info(f"Vector Details for {vector_id}: {json.dumps(vector_details, indent=2)}")
        hash_value = create_hash(vector_details)
        logger.debug(f"Generated hash value: {hash_value}")
        existing_doc = es.options(ignore_status=[404]).get(
            index=RESULT_INDEX_PREFIX + "hash_table_index", id=hash_value
        )
        if existing_doc and "_source" in existing_doc:
            issues = existing_doc["_source"].get("issues", [])
            logger.info(f"Vector ID {vector_id} is associated with issues: {issues}")
            for issue in issues:
                logger.info(f"Issue: {issue}")
        else:
            logger.warning(f"No existing document found for hash {hash_value}.")
    es.close()


# Run tests if executed directly
if __name__ == "__main__":
    cli(obj={})
