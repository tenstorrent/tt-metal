# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct host tests for profile planning and observed configuration rejection."""

import copy
import json
import unittest

from benchmark_server_config import MODEL, REVISION, profile_plan, validate_observed_config


def observed(slots):
    args = profile_plan(slots)["command"]
    return {
        "vllm_config": {
            "model_config": {
                "model": MODEL,
                "revision": REVISION,
                "tokenizer_revision": REVISION,
                "max_model_len": 262144,
            },
            "cache_config": {"block_size": 32, "enable_prefix_caching": False},
            "scheduler_config": {"max_num_seqs": slots, "enable_chunked_prefill": False, "async_scheduling": True},
            "additional_config": json.loads(args[args.index("--additional-config") + 1]),
        }
    }


class ServerConfigurationTests(unittest.TestCase):
    def test_profiles_preserve_context_and_pin_revisions(self):
        for slots in (1, 32):
            plan = profile_plan(slots)
            args = plan["command"]
            for name, value in (
                ("--max-num-seqs", str(slots)),
                ("--max-model-len", "262144"),
                ("--revision", REVISION),
                ("--tokenizer-revision", REVISION),
            ):
                self.assertEqual(args[args.index(name) + 1], value)
            self.assertIn("--no-enable-prefix-caching", args)
            self.assertEqual(validate_observed_config(observed(slots), slots)["max_num_seqs"], slots)

    def test_drift_rejected(self):
        cases = [
            ("scheduler_config", "max_num_seqs", 32),
            ("model_config", "max_model_len", 8192),
            ("model_config", "tokenizer_revision", None),
            ("cache_config", "enable_prefix_caching", True),
        ]
        for section, key, value in cases:
            evidence = copy.deepcopy(observed(1))
            evidence["vllm_config"][section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_observed_config(evidence, 1)

    def test_wrong_trace_rejected(self):
        evidence = observed(1)
        evidence["vllm_config"]["additional_config"]["tt"]["trace_mode"] = "all"
        with self.assertRaises(ValueError):
            validate_observed_config(evidence, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
