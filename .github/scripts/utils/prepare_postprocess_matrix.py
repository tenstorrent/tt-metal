# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Expand optional multihost test-row `postprocess` metadata into required CPU jobs.

Each configured row supplies shards (1..256), runs_on, timeout (minutes), cmd,
and aggregate_cmd. Commands run in the dev image with POSTPROCESS_INPUT pointing
at the producer's existing artifact and POSTPROCESS_OUTPUT at the shard results.
POSTPROCESS_SHARD is zero-based. The aggregate runs even if a shard fails so its
command can report missing results; neither job may ignore failures.
"""

import json
import os


def prepare(matrix):
    legs, shards = [], []
    for index, test in enumerate(matrix):
        if "postprocess" not in test:
            continue
        config = test["postprocess"]
        if not isinstance(config, dict) or set(config) != {"runs_on", "cmd", "aggregate_cmd", "shards", "timeout"}:
            raise ValueError("postprocess requires runs_on, cmd, aggregate_cmd, shards and timeout")
        for key in ("runs_on", "cmd", "aggregate_cmd"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"postprocess requires {key}")
        for key, limit in (("shards", 256), ("timeout", 360)):
            if type(config.get(key)) is not int or not 1 <= config[key] <= limit:
                raise ValueError(f"postprocess {key} must be an integer in 1..{limit}")
        # Keep the original index: the generic producer artifact uses strategy.job-index.
        leg = {"index": index, "name": test["name"], **config}
        legs.append(leg)
        shards.extend({**leg, "shard": shard} for shard in range(config["shards"]))
    if len(shards) > 256:
        raise ValueError("postprocess exceeds the 256-job matrix limit")
    return legs, shards


if __name__ == "__main__":
    legs, shards = prepare(json.loads(os.environ["TEST_MATRIX"]))
    with open(os.environ["GITHUB_OUTPUT"], "a") as output:
        output.write(f"legs={json.dumps(legs)}\nshards={json.dumps(shards)}\n")
