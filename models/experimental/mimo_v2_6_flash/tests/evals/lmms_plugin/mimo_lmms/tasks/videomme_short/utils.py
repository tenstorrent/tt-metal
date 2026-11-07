# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""videomme_short: lmms-eval's Video-MME task functions, restricted to the short-duration split."""

from lmms_eval.tasks.videomme.utils import (  # noqa: F401
    videomme_aggregate_results,
    videomme_doc_to_text,
    videomme_doc_to_visual,
    videomme_process_results,
)


def short_only(dataset):
    return dataset.filter(lambda doc: doc["duration"] == "short")
