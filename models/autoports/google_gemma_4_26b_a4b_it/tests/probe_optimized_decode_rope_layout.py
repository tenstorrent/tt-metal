# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Change only the setup upload layout of the actual HF decode RoPE tables.

The unchanged harness uploads 4D prefill tables and 2D decode tables from the
same HF cos/sin storage. Exact host storage identities distinguish these two
uploads from weights, activations, and unrelated tables. Runtime host-boundary
guards remain active; no tensor contents are cached or converted in forward.
"""

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_decoder, run_optimized_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


class RopeUploadProbe:
    """Track known HF table identities only during ordinary harness setup."""

    def __init__(self, original_upload, layout):
        self.original_upload = original_upload
        self.layout = layout
        self.tables = {}
        self.observed = []

    def register(self, tables):
        for role, tensor in zip(("cos", "sin"), tables):
            if tensor.ndim != 3 or tensor.shape[0] != 1:
                raise ValueError("RoPE layout probe expects HF tables shaped [1, context, head_dim]")
            key = (tensor.data_ptr(), tuple(tensor.shape[-2:]))
            self.tables[key] = role

    def upload(self, tensor, *args, **kwargs):
        key = (tensor.data_ptr(), tuple(tensor.shape[-2:])) if tensor.ndim in (2, 4) else None
        role = self.tables.get(key)
        phase = None
        if role is not None:
            if tensor.ndim == 2:
                phase = "decode"
                kwargs["layout"] = self.layout
            elif tuple(tensor.shape[:2]) == (1, 1):
                phase = "prefill"
        result = self.original_upload(tensor, *args, **kwargs)
        if phase is not None:
            self.observed.append(
                dict(
                    role=role,
                    phase=phase,
                    shape=list(result.shape),
                    dtype=str(result.dtype),
                    layout=str(result.layout),
                    memory=str(result.memory_config()),
                    setup_only=True,
                )
            )
        return result

    def validate(self):
        for phase in ("prefill", "decode"):
            rows = [row for row in self.observed if row["phase"] == phase]
            if sorted(row["role"] for row in rows) != ["cos", "sin"]:
                raise AssertionError(f"Expected exactly one cos/sin upload for {phase}, observed {rows}")
            expected_layout = ttnn.TILE_LAYOUT if phase == "prefill" else self.layout
            if any(row["layout"] != str(expected_layout) for row in rows):
                raise AssertionError(f"Unexpected {phase} RoPE layout: {rows}")


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--decode-rope-layout", choices=("row_major", "tile"), default="row_major")
    args, rest = parser.parse_known_args()
    if "--decode" not in rest or "--defaults" not in rest:
        parser.error("RoPE producer-layout probe requires --decode and --defaults")
    output = Path(rest[rest.index("--output") + 1])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing producer-layout evidence: {output}")
    layout = ttnn.ROW_MAJOR_LAYOUT if args.decode_rope_layout == "row_major" else ttnn.TILE_LAYOUT
    uploader = RopeUploadProbe(ttnn.from_torch, layout)
    rotary_forward = run_decoder.Gemma4TextRotaryEmbedding.forward
    factory = OptimizedDecoder.from_state_dict.__func__
    source_path = Path(__file__).parents[1] / "tt/optimized_decoder.py"
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()

    def rotary(self, *a, **kw):
        tables = rotary_forward(self, *a, **kw)
        uploader.register(tables)
        return tables

    def create(cls, *a, **kw):
        decoder = factory(cls, *a, **kw)
        decoder.decode_rope_layout = layout
        decoder.precision_policy["decode_rope_layout_probe"] = str(layout)
        return decoder

    previous_argv = sys.argv
    sys.argv = [sys.argv[0], *rest]
    failure = None
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.object(run_decoder.Gemma4TextRotaryEmbedding, "forward", rotary))
            stack.enter_context(patch.object(OptimizedDecoder, "from_state_dict", classmethod(create)))
            stack.enter_context(patch.object(ttnn, "from_torch", uploader.upload))
            run_optimized_decoder.main()
        uploader.validate()
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_hash:
            raise AssertionError("Optimized runtime changed during the producer-layout probe")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        sys.argv = previous_argv
        report = json.loads(output.read_text()) if output.exists() else {"decoder": "optimized"}
        report["decode_rope_layout_probe"] = dict(
            requested=args.decode_rope_layout,
            upload_observations=uploader.observed,
            matching="Exact HF cos/sin storage identities and shapes; 2D decode uploads only",
            prefill_layout="TILE_LAYOUT",
            host_work_in_forward=False,
            runtime_sha256=source_hash,
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            harness_sha256=hashlib.sha256(Path(run_decoder.__file__).read_bytes()).hexdigest(),
        )
        if failure is not None:
            report["decode_rope_layout_probe_error"] = failure
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
