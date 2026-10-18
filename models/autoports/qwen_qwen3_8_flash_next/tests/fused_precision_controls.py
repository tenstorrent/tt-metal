# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated real-HF controls for matched fused decoder accuracy regressions.

Existing QWEN38_WEIGHT_SOURCE, QWEN38_DECODER_BATCHES and
QWEN38_DECODER_LENGTHS environment variables configure the unchanged contract.
Modes change one method only; all runtime types remain the delivered classes.
"""

import argparse
import hashlib
import inspect
import json
import os
import textwrap
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


def _rewrite(method, old, new):
    source = textwrap.dedent(inspect.getsource(method))
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one control replacement in {method.__qualname__}: {old!r}")
    source = source.replace(old, new, 1)
    namespace = dict(method.__globals__)
    exec(compile(source, f"{__file__}:{method.__qualname__}", "exec"), namespace)
    return namespace[method.__name__], source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("current", "dedicated_moe", "split_short_hc", "composite_conv", "composite_norm", "split_all_hc"),
    )
    parser.add_argument("--layers", default="layer0 or layer1", help="pytest -k expression")
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()

    import pytest

    from models.autoports.qwen_qwen3_8_flash_next.tests import test_fused_decoder as suite
    from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused

    root = Path(__file__).resolve().parents[1]
    selected_evidence = root / "doc/fused_decoder"
    output = (
        args.evidence_dir
        if args.evidence_dir is not None
        else selected_evidence
        / "precision_controls"
        / (args.mode + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    ).resolve()
    if output == selected_evidence.resolve():
        parser.error("The control evidence directory must differ from selected fused evidence")
    output.mkdir(parents=True, exist_ok=True)

    replacement = None
    source = None
    if args.mode == "dedicated_moe":
        method, source = _rewrite(
            fused.FusedMoE.__call__,
            "group_fn = self._group_unified if seq == 1 else self._group",
            "group_fn = self._group",
        )
        replacement = (fused.FusedMoE, "__call__", method)
    elif args.mode in ("composite_conv", "composite_norm"):
        replacement = (
            fused.FusedGatedDeltaNet,
            "native_conv" if args.mode == "composite_conv" else "native_norm",
            False,
        )
        source = repr(replacement[1:])
    elif args.mode in ("split_short_hc", "split_all_hc"):
        # selected_all_boundaries used split read for S>1; the later selected
        # policy uses vector read through S4. This changes S2/S3/S4 only and
        # leaves S1 decode, HC writes, norms, projections and MoE untouched.
        method, source = _rewrite(
            fused.FusedHyperConnection.read,
            "if x.shape[2] > 4:",
            "if x.shape[2] > 1:" if args.mode == "split_short_hc" else "if x.shape[2] >= 1:",
        )
        replacement = (fused.FusedHyperConnection, "read", method)

    (output / "control.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "layers": args.layers,
                "fused_source_sha256": hashlib.sha256((root / "tt/fused_decoder.py").read_bytes()).hexdigest(),
                "replacement_source": source,
                "settings": {
                    key: os.environ.get(key)
                    for key in (
                        "QWEN38_WEIGHT_SOURCE",
                        "QWEN38_DECODER_BATCHES",
                        "QWEN38_DECODER_LENGTHS",
                        "QWEN38_DECODER_CHUNK_SIZE",
                        "TT_METAL_TRACE_ALLOC_TRACKING",
                    )
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"FUSED_PRECISION_CONTROL mode={args.mode} evidence_dir={output}", flush=True)
    with ExitStack() as stack:
        stack.enter_context(patch.object(suite, "EVIDENCE", output))
        if replacement is not None:
            stack.enter_context(patch.object(*replacement))
        return pytest.main(["-s", "-q", "--timeout=0", str(root / "tests/test_fused_decoder.py"), "-k", args.layers])


if __name__ == "__main__":
    raise SystemExit(main())
