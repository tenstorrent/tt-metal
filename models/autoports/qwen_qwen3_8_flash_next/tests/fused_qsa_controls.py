# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the existing fused QSA parity test with only functional normalization.

The fused constructor, packed projections, native rotary operation, cache path,
and traced repeated decode remain selected. Length/batch/reference-cache knobs
are the existing TT_QSA_* environment variables. Evidence is redirected into
a separate directory, leaving the selected implementation's reports untouched.
"""

import argparse
import hashlib
import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()

    import pytest

    from models.autoports.qwen_qwen3_8_flash_next.tests import test_fused_qsa as contract
    from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedQSA
    from models.autoports.qwen_qwen3_8_flash_next.tt.qsa import QSA

    root = Path(__file__).resolve().parents[1]
    original_evidence = root / "doc/fused_decoder"
    output = (
        args.evidence_dir
        if args.evidence_dir is not None
        else original_evidence
        / "qsa_norm_controls"
        / datetime.now(timezone.utc).strftime("functional_norm_%Y%m%dT%H%M%S_%fZ")
    ).resolve()
    if output == original_evidence.resolve():
        parser.error("--evidence-dir must differ from the selected fused evidence directory")
    output.mkdir(parents=True, exist_ok=True)

    class EvidencePath(type(Path())):
        def write_text(self, data, *write_args, **write_kwargs):
            if self.parent == original_evidence and self.name.startswith("qsa_context_"):
                destination = output / self.name
                print(f"QSA_CONTROL_EVIDENCE={destination}", flush=True)
                return destination.write_text(data, *write_args, **write_kwargs)
            return super().write_text(data, *write_args, **write_kwargs)

    norm_source = inspect.getsource(QSA._norm)
    (output / "control.json").write_text(
        json.dumps(
            {
                "control": "functional_qsa_norm_only",
                "replacement": "tt.qsa.QSA._norm -> tt.fused_decoder.FusedQSA._norm",
                "functional_norm_source": norm_source,
                "functional_norm_sha256": hashlib.sha256(norm_source.encode()).hexdigest(),
                "fused_source_sha256": hashlib.sha256((root / "tt/fused_decoder.py").read_bytes()).hexdigest(),
                "settings": {
                    key: os.environ.get(key)
                    for key in ("TT_QSA_LENGTHS", "TT_QSA_BATCHES", "TT_QSA_WEIGHTS", "TT_QSA_REFERENCE_CACHE")
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"QSA_CONTROL functional_norm_only evidence_dir={output}", flush=True)
    # Redirect only this test module's Path writes. The original test's final
    # QSA_EVIDENCE print names its normal path; QSA_CONTROL_EVIDENCE above is
    # the actual destination. No source, checkpoint, or runtime path changes.
    with patch.object(FusedQSA, "_norm", QSA._norm), patch.object(contract, "Path", EvidencePath):
        return pytest.main(["-s", "-q", "--timeout=0", str(root / "tests/test_fused_qsa.py"), "-k", "hf_parity"])


if __name__ == "__main__":
    raise SystemExit(main())
