# SPDX-License-Identifier: Apache-2.0
"""Finalize native evidence only after resources have been released and closed."""
import json


def finalize_run(result, target, release, close):
    body_passed = result["status"] == "pass"
    cleanup_error = None
    try:
        release()
    except BaseException as error:
        cleanup_error = error
        result.update(status="fail", cleanup_error=f"{type(error).__name__}: {error}")
    try:
        close()
        result["close_mesh_device_returned"] = True
    except BaseException as error:
        cleanup_error = cleanup_error or error
        result.update(status="fail", close_mesh_device_returned=False, close_error=f"{type(error).__name__}: {error}")
    target.write_text(json.dumps(result, indent=2) + "\n")
    if cleanup_error is not None and body_passed:
        raise cleanup_error
