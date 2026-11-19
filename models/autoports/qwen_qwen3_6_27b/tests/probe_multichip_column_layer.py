# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated complete-layer column-ownership experiment; ordinary PCC gates.

The dispatch adapter exists only inside this CLI process. Production defaults
and the source of every other operation remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--grain", type=int, choices=(1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, forwarded = parser.parse_known_args()
    assert "--baseline" not in forwarded and not args.output.exists()
    from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner
    from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as model
    from models.autoports.qwen_qwen3_6_27b.tt import recurrent_column_candidate as candidate
    from models.autoports.qwen_qwen3_6_27b.tt.recurrent_variants import sources

    source = Path(model.__file__).read_text()
    proxy = candidate.bind_local(model, model._fused_delta_row_GDN_COMPACT_WRITER, source)
    operation = proxy.column_functions["columns" if args.grain == 1 else "pairs"]
    calls, failure = 0, None

    def adapted(q, k, v, g, beta, state, **kwargs):
        nonlocal calls
        if not kwargs.get("packed_qkv", False):
            return proxy._local12_recurrent(q, k, v, g, beta, state, **kwargs)
        assert not kwargs.get("normalized", False) and kwargs.get("source_overrides") is None
        assert kwargs.get("variant") == "cumulative_exp_scalar"
        batch = state.shape[0] // 12
        grid = state.device().compute_with_storage_grid_size()
        grouped, _ = candidate.column_partition(
            batch, grid.x * grid.y, kwargs.get("cores"), kwargs.get("grouped_qk", True), grain=args.grain
        )
        base = sources(
            model._LOCAL12_READER, model._fused_delta_row_GDN_COMPACT_COMPUTE, "cumulative_exp_scalar", grouped=grouped
        )
        kwargs["source_overrides"] = candidate.column_sources(*base, model._fused_delta_row_GDN_COMPACT_WRITER)
        calls += 1
        return operation(q, k, v, g, beta, state, **kwargs)

    try:
        with (
            patch.object(model, "_local12_recurrent", adapted),
            patch.object(sys, "argv", [sys.argv[0], *forwarded, "--output", str(args.output)]),
        ):
            runner.main()
            assert calls > 0, "Column candidate did not execute"
    except BaseException:
        failure = traceback.format_exc()
        raise
    finally:
        report = json.loads(args.output.read_text()) if args.output.exists() else {"status": "fail"}
        report["column_ownership"] = {
            "grain": args.grain,
            "calls": calls,
            "failure": failure,
            "source_sha256": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), Path(candidate.__file__), Path(model.__file__))
            },
            "classification": "same arithmetic; changed value-column ownership; ordinary runner gates",
        }
        if failure:
            report["status"] = "fail"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
