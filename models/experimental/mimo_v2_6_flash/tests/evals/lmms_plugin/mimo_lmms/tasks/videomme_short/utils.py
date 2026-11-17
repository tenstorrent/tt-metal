# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""videomme_short: lmms-eval's Video-MME task functions, restricted to the short-duration split."""

from lmms_eval.tasks.videomme.utils import (  # noqa: F401
    videomme_aggregate_results,
    videomme_doc_to_text,
    videomme_doc_to_visual,
)
from lmms_eval.tasks.videomme.utils import videomme_process_results as upstream_process_results

from mimo_lmms.failures import request_failed


def videomme_process_results(doc, results):
    """lmms-eval's scoring, except that a request that failed for good is scored as an empty answer: its failure
    text holds letters the upstream extraction would take for a choice."""
    if request_failed(results[0]):
        results = [""] + list(results[1:])
    return upstream_process_results(doc, results)


def short_only(dataset):
    return dataset.filter(lambda doc: doc["duration"] == "short")
