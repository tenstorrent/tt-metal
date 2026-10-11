# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize independent HF controls for the repaired eager prefill geometry.

The official HF implementation runs in the separate reference environment.
Every subprocess closes its mesh before the next opens it. No timing from this
correctness check is used for datatype selection.
"""

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path

from models.autoports.meta_models_muse_glimmer_30b.tt.precision import decoder_policy, load_precision_config

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"


def main(args):
    rows = []
    for config_id in ("baseline_bfp4_lofi_head_bfp8_hifi2", "decoder_bfloat8_b_LoFi"):
        config = load_precision_config(DOC / "configs" / f"{config_id}.json")
        for layer, kind in ((0, "sliding_attention"), (3, "full_attention")):
            output = DOC / f"prefill_guard_{config_id}_layer{layer}{args.suffix}.json"
            command = [
                args.reference_python,
                str(ROOT / "tests/run_multichip_hf.py"),
                "--real",
                "--layer",
                str(layer),
                "--lengths",
                *map(str, args.lengths),
                "--continuation",
                "352",
                "--policy",
                json.dumps(decoder_policy(config, kind, layer)),
                "--output",
                str(output),
            ]
            row = dict(
                command=command,
                config_id=config_id,
                layer=layer,
                log=str(output.with_suffix(".log")),
                started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            )
            print("START", config_id, layer, flush=True)
            with output.with_suffix(".log").open("w") as log:
                rc = subprocess.call(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(
                        os.environ, TT_METAL_TRACE_ALLOC_TRACKING="1", TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0"
                    ),
                )
            row.update(returncode=rc, finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
            rows.append(row)
            (DOC / f"prefill_guard_commands{args.suffix}.json").write_text(json.dumps(rows, indent=2) + "\n")
            print("FINISH", config_id, layer, rc, flush=True)
            if rc:
                raise SystemExit(rc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-python", default="/home/vkovacevic/muse-glimmer/hf-reference-env/bin/python")
    parser.add_argument("--lengths", nargs="+", type=int, default=[321, 353, 769, 961, 993, 1025])
    parser.add_argument("--suffix", default="")
    main(parser.parse_args())
