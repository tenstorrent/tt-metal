# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Select the exact fused QSA component in the unchanged reduced-prefix harness.

This is reduced attention coverage, not full decoder prefill. The complete decoder
bridge consumes the separately exported device-produced prefix afterwards.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tests.run_fused_probe import POLICIES
from models.autoports.qwen_qwen3_8_flash_next.tt import qsa_chunked
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedChunkedQSA, FusedDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=[name for name, policy in POLICIES.items() if policy is not None], default="final"
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    original = qsa_chunked.ChunkedQSA
    options = {
        name: parameter.default
        for name, parameter in inspect.signature(FusedDecoder.from_state_dict).parameters.items()
    }
    options.update(POLICIES[args.variant])
    report = dict(
        completed=False,
        constructed=[],
        source_sha256=hashlib.sha256(Path(sys.modules[FusedChunkedQSA.__module__].__file__).read_bytes()).hexdigest(),
        variant=args.variant,
    )

    def construct(attention):
        model = FusedChunkedQSA.from_component(
            original(attention),
            gate=options["fuse_qsa_gate"],
            concat=options["fuse_qsa_concat"],
            pack=options["pack_qsa"],
            split=options["split_qsa_heads"],
            rope=options["fuse_qsa_rope"],
            decode_rope=options["decode_qsa_rope"],
            score_sum=options["fuse_indexer_score_sum"],
            merge_score_heads=options["merge_indexer_score_heads"],
            valid_rows=options["fuse_indexer_valid_rows"],
            row_compute=options["fuse_indexer_row_compute"],
            copy_once=options["fuse_indexer_copy_init"],
            selector=options["fuse_qsa_selector"],
        )
        assert type(model) is FusedChunkedQSA
        assert model.fuse_gate and model.fuse_concat and model.fuse_rope and hasattr(model, "packed_qkv")
        report["constructed"].append(
            dict(
                chunked=type(model).__name__,
                attention=type(model.attention).__name__,
                projection=type(model.attention.project_index_qk).__name__,
                selector=model.fuse_selector,
            )
        )
        return model

    try:
        with patch.object(qsa_chunked, "ChunkedQSA", construct):
            module = importlib.import_module("models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_prefill")
            sys.argv = [module.__file__] + (args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments)
            module.main()
        assert report["constructed"]
        report["completed"] = True
    finally:
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        args.selection.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
