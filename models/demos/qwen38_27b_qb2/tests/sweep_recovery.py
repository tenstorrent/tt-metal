# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resume measurements only across matching model and benchmark settings."""

import copy
import hashlib
import json
import re
from pathlib import Path

from models.demos.qwen38_27b_qb2.tests.sweep_report import summarize

DRAM_OOM = re.compile(r"Out of Memory: Not enough space to allocate \d+ B DRAM buffer across \d+ banks")
PATH_ENVIRONMENT = {"QWEN_SWEEP_RESULTS", "QWEN_SWEEP_RESUME_FROM", "QWEN_PRECISION_CONFIG"}


def is_dram_allocation_error(error):
    """Do not classify dispatch timeouts, host OOM or accuracy errors as capacity."""
    if isinstance(error, BaseException):
        error = {"type": type(error).__name__, "message": str(error)}
    return error.get("type") == "RuntimeError" and bool(DRAM_OOM.search(error.get("message", "")))


def can_restart(report, returncode):
    return (
        returncode == 1
        and report.get("state") == "allocation_failed"
        and report.get("cleanup_completed") is True
        and is_dram_allocation_error(report.get("error", {}))
        and any(cell["status"] == "oom" for cell in report["cells"])
    )


def normalized_configuration(configuration):
    result = copy.deepcopy(configuration)
    result["environment"] = {key: value for key, value in result["environment"].items() if key not in PATH_ENVIRONMENT}
    return result


def resume_measurements(report, receipts):
    """Copy completed cells into a new receipt, retaining the immutable originals.

    A recorded OOM is reusable only after this harness confirmed device cleanup.
    Older receipts without that confirmation have their failed cell retried.
    """
    cells = {(cell["input_tokens"], cell["batch_per_replica"]): cell for cell in report["cells"]}
    for receipt_path in receipts:
        path = Path(receipt_path)
        content = path.read_bytes()
        previous = json.loads(content)
        for key in ("replicas", "chips", "output_tokens", "warmup_runs", "measured_runs", "source_sha256", "precision"):
            if not report.get(key) or report[key] != previous.get(key):
                raise ValueError(f"Cannot resume {path}: {key} differs or is missing")
        if normalized_configuration(report["configuration"]) != normalized_configuration(previous["configuration"]):
            raise ValueError(f"Cannot resume {path}: runtime configuration differs")
        state = previous.get("state")
        if state not in ("completed", "completed_with_oom", "failed", "allocation_failed"):
            raise ValueError(f"Cannot resume {path}: receipt is not terminal")
        if state in ("failed", "allocation_failed") and not is_dram_allocation_error(previous.get("error", {})):
            raise ValueError(f"Cannot resume {path}: failure was not a DRAM allocation failure")
        origin = dict(path=str(path), sha256=hashlib.sha256(content).hexdigest())
        report.setdefault("resume_receipts", []).append(origin)
        for old in previous["cells"]:
            target = cells.get((old["input_tokens"], old["batch_per_replica"]))
            if target is None or target["status"] in ("capacity_guard", "implementation_guard"):
                continue
            if old["status"] == "completed":
                samples = old["samples"]
                reference = old["warmup"]["output_sha256_per_replica"]
                if (
                    len(samples) != report["measured_runs"]
                    or len(reference) != report["replicas"]
                    or len(set(reference)) != 1
                    or any(sample["output_sha256_per_replica"] != reference for sample in samples)
                    or not old.get("prompt_sha256")
                ):
                    raise ValueError(f"Cannot resume {path}: repeatability evidence is incomplete")
                summary = summarize(
                    samples,
                    concurrency=target["concurrency"],
                    output_tokens=report["output_tokens"],
                    input_tokens=target["input_tokens"],
                )
                if summary != old["summary"] or old["concurrency"] != target["concurrency"]:
                    raise ValueError(f"Cannot resume {path}: measurement accounting differs")
                target.update(copy.deepcopy(old), carried_from=origin)
            elif old["status"] == "oom" and previous.get("cleanup_completed") is True:
                if not is_dram_allocation_error(old.get("error", {})):
                    raise ValueError(f"Cannot resume {path}: OOM cell lacks allocator evidence")
                if target["status"] != "completed":
                    target.update(copy.deepcopy(old), carried_from=origin)
