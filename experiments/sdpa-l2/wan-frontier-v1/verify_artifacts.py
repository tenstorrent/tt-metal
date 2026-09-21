# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Verify completed outputs, cached loads, replay checks and actual KV formats."""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", type=Path)
    args = parser.parse_args()
    rows = []
    for variant in ("stock", "D", "C", "B", "E", "F", "G"):
        root = args.suite / variant
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["status"] == "completed"
        assert {r["prompt_id"] for r in manifest["results"]} == {0, 1}
        assert len(manifest["results"]) == 2
        assert len(manifest["block_bench"]) == 4
        assert all(r["replay_exact"] for r in manifest["block_bench"].values())
        assert all(r["cache_hit"] or r["already_loaded"] for r in manifest["weight_loads"])
        if variant != "stock":
            assert len(manifest["transport"]) == 80
            kv = "BFLOAT4_B" if variant == "G" else "BFLOAT8_B" if variant in "EF" else "BFLOAT16"
            for record in manifest["transport"].values():
                assert record["q_dtype"] == "DataType.BFLOAT16"
                assert record["k_dtype"] == record["v_dtype"] == f"DataType.{kv}"
                assert record["logical_k"] == 32760
        for result in manifest["results"]:
            video = root / result["video"]
            assert hashlib.sha256(video.read_bytes()).hexdigest() == result["video_sha256"]
            assert len(result["sampled_frames"]) == 8
            for frame in result["sampled_frames"]:
                path = root / frame["file"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == frame["sha256"]
                with Image.open(path) as image:
                    assert image.size == (832, 480)
                    image.verify()
            rows.append(dict(variant=variant, prompt_id=result["prompt_id"], video_bytes=video.stat().st_size))
    report = dict(status="completed", verified_videos=len(rows), verified_pngs=8 * len(rows), rows=rows)
    with (args.suite / "artifact-validation.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
