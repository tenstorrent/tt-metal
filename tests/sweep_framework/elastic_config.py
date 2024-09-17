# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
from elasticsearch import Elasticsearch
from sweeps_logger import sweeps_logger as logger

ELASTIC_CORP_URL = "http://yyz-elk:9200"
ELASTIC_CLOUD_URL = "http://172.27.28.43:9200"
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
VECTOR_INDEX_PREFIX = "ttnn_sweeps_test_vectors_"
RESULT_INDEX_PREFIX = "ttnn_sweeps_test_results_"


def get_elastic_url(elastic_arg):
    if elastic_arg == "corp":
        logger.info(
            "Connecting to Elasticsearch on corporate network. If you are on tt-cloud, please pass '--elastic cloud' to your test command."
        )
        return ELASTIC_CORP_URL
    elif elastic_arg == "cloud":
        logger.info(
            "Connecting to Elasticsearch on tt-cloud network. If you are on corporate, please pass '--elastic corp' to your test command."
        )
        return ELASTIC_CLOUD_URL
    else:
        logger.info(f"Connecting to Elasticsearch on {elastic_arg}.")
        return elastic_arg
