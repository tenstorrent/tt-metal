# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Local conftest for the per-op suite: tags emulator-appropriate cases.

The 2-node Quasar emulator can only run small / batch-1 shapes. Rather than rely on
inconsistent per-file id tokens (decode-batch1 vs b1 vs batch1, plus prefill seqN),
this hook marks every op-test case that fits the emulator with the ``emulator``
marker, so you can select the whole emulator subset with:

    pytest models/experimental/llama32_1b_quasar/tests/ops/ -m emulator

A case is considered emulator-appropriate unless its test id contains a token that
means "too large for 2 cores": batch-32 decode, or a 512/1024-length prefill. The
batch-32 *sharded* ops already auto-skip via height_sharded_batch_memcfg, and the
CCL / sampling ops carry module-level skips; this marker additionally keeps the
large *non*-sharded batch-32 and long-prefill cases out of the emulator run so they
neither run nor clutter the results.
"""

# Substrings in a test id that mark a case as too large for the 2-node emulator.
_NON_EMULATOR_TOKENS = ("batch32", "b32", "seq512", "seq1024", "seq500")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "emulator: op-test case that fits the 2-node Quasar emulator (batch-1 / small shapes)",
    )


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        nid = item.nodeid.lower()
        if not any(tok in nid for tok in _NON_EMULATOR_TOKENS):
            item.add_marker(pytest.mark.emulator)
