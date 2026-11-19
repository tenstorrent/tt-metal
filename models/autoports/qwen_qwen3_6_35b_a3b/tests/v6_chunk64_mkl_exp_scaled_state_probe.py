# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unchanged chunk64 MKL exp wrapper with the separately frozen scaled helper."""

import inspect
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_chunk64_mkl_exp_state_probe as previous
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_mkl_exp_scaled_ttnn_probe as helper


def run(args):
    original_helper = previous.helper
    source_hash = previous.base.sha(__file__)
    original_wrapper_hash = previous.base.sha(previous.__file__)
    assert original_wrapper_hash == "c4004cd0992298ba33c5ab21b9ecb8cb5e1d44dea6c790630643fe3a68ec27eb"
    snapshot = Path(str(args.output) + ".scaled_exp_wrapper.py.snapshot")
    assert not args.output.exists() and not snapshot.exists(), "Preserve evidence"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_bytes(Path(__file__).read_bytes())
    original_setup = previous.base.cpu_setup

    def setup(options):
        result, payload = original_setup(options)
        result.update(
            scaled_exp_wrapper=str(Path(__file__).resolve()),
            scaled_exp_wrapper_sha256=source_hash,
            scaled_exp_wrapper_snapshot=str(snapshot),
            scaled_exp_wrapper_snapshot_sha256=previous.base.sha(snapshot),
        )
        return result, payload

    previous.helper = helper
    previous.base.cpu_setup = setup
    try:
        previous.run(args)
    finally:
        previous.helper = original_helper
        previous.base.cpu_setup = original_setup
        assert previous.base.sha(__file__) == source_hash
        assert previous.base.sha(previous.__file__) == original_wrapper_hash


def main():
    scope = dict(vars(previous), run=run)
    exec(compile(inspect.getsource(previous.main), str(Path(__file__).resolve()) + ":unchanged_cli", "exec"), scope)
    scope["main"]()


if __name__ == "__main__":
    main()
