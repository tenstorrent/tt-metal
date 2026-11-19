# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Apply an isolated class candidate to the unchanged TP4 oracle runner."""

from models.autoports.qwen_qwen3_6_27b.tt.prefill_alignment_candidate import make_decoder


def main():
    from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner

    runner.MultichipDecoder = make_decoder(runner.MultichipDecoder)
    runner.main()


if __name__ == "__main__":
    main()
