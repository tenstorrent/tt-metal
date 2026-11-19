# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the context smoke on the attention owned by the actual optimized decoder.

The prefix harness measures reduced attention, not all-prefix MoE. Construction
uses complete checkpoint weights and the normal optimized factory; source, dtype,
program and projection selection therefore match the delivered decoder.
"""
import argparse
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt import qsa_chunked
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--policy", type=Path)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    policy = json.loads(a.policy.read_text()) if a.policy else {}
    records, owners = [], []

    def construct(attention):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(a.snapshot, local_files_only=True).text_config
        model = OptimizedDecoder.from_state_dict(
            load_weights(a.snapshot, 3, ""),
            hf_config=config,
            layer_idx=3,
            mesh_device=attention.loader.mesh_device,
            **policy,
        )
        owners.append(model)
        records.append(
            dict(
                decoder=type(model).__name__,
                attention=type(model.chunked_attn).__name__,
                projection=type(model.chunked_attn.packed_qkv).__name__,
                sdpa_program=str(model.chunked_attn.sdpa_program),
            )
        )
        return model.chunked_attn

    report = dict(constructed=records, completed=False, scope="reduced QSA prefix from complete optimized construction")
    try:
        with patch.object(qsa_chunked, "ChunkedQSA", construct):
            module = importlib.import_module("models.autoports.qwen_qwen3_8_flash_next.tests.probe_qsa_prefill")
            args = a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments
            sys.argv = [module.__file__, "--snapshot", str(a.snapshot), *args]
            module.main()
        assert records
        report["completed"] = True
    finally:
        a.selection.parent.mkdir(parents=True, exist_ok=True)
        a.selection.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
