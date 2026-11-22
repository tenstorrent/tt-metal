# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The plugin proof line (LEAD_DECISIONS_M7 M52 (3), M140 D2): tt/s1_probe.py's log_plugin_proof and PLUGIN_FILES,
moved here unchanged, so a served class logs the line without importing s1_probe, which carries the S1 probe's
test-only fault injection (GEMMA4_S1P_TEST_*, CorruptSwitch) and is never packaged.

log_plugin_proof() (the lead's plugin-proof rule, S1 prereg addendum 2): one line from the process that builds the model
(vLLM's EngineCore here) naming the vllm_tt_plugin package that process imported, the sha256 of its S1 files and
whether every loaded vllm_tt_plugin module came from that package. The line's text is s1_probe's ("S1P plugin (pid
...)"), which H6's tooling reads.
"""

import hashlib
import json
import os
import sys

from loguru import logger

PLUGIN_FILES = ("prefix_resume.py", "scheduler.py", "platform.py", "config.py", "model_input.py", "model_runner.py",
                "async_decode.py")


def log_plugin_proof():
    import vllm_tt_plugin

    root = os.path.dirname(os.path.abspath(vllm_tt_plugin.__file__))
    sha = {}
    for name in PLUGIN_FILES:
        with open(os.path.join(root, name), "rb") as f:
            sha[name] = hashlib.sha256(f.read()).hexdigest()
    loaded = {k: os.path.abspath(m.__file__) for k, m in list(sys.modules.items())
              if (k == "vllm_tt_plugin" or k.startswith("vllm_tt_plugin.")) and getattr(m, "__file__", None)}
    outside = sorted(k for k, f in loaded.items() if not f.startswith(root + os.sep))
    logger.info("S1P plugin (pid {}): {}", os.getpid(), json.dumps(
        {"root": root, "sha256": sha, "modules_loaded": len(loaded), "modules_outside_root": outside,
         "pythonpath": os.environ.get("PYTHONPATH", "")}, sort_keys=True))
