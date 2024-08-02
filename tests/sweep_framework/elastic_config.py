# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

ELASTIC_DEFAULT_URL = "http://yyz-elk:9200"
ELASTIC_USERNAME = os.getenv("ELASTIC_USERNAME")
ELASTIC_PASSWORD = os.getenv("ELASTIC_PASSWORD")
VECTOR_INDEX_PREFIX = "ttnn_sweeps_test_vectors_"
RESULT_INDEX_PREFIX = "ttnn_sweeps_test_results_"
