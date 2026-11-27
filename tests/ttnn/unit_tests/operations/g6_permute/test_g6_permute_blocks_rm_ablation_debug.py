# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Refinements 3 and 5 ablation of blocks_rm_gather / blocks_rm_bank_coalesced on FLUX_74 (DRAM and
L1), plus a BLOCK_PAGES sweep.

The ablation arms need temporary guards in the kernels; the shipped kernels do not carry them, so the
stubbed arms are skipped unless G6_ABLATION_GUARDS=1:
- `#ifndef G6_ABLATE_READS` around the reader's noc_async_read calls (the barrier stays);
- `#ifndef G6_ABLATE_COMPACT` around the coalesced reader's staging-to-slots copy loop;
- `#ifndef G6_ABLATE_GATHER` around the writer's gather loop (and the transpose calls of the grouped path);
- `#ifndef G6_ABLATE_WRITES` around the writer's noc_async_write (and noc_async_write_one_packet_with_trid
  of the grouped path).
Stubbed arms keep every CB handshake and barrier, and their output is not checked. Run under
`scripts/run_safe_pytest.sh --profile --no-py-zones`. DO NOT DELETE.
"""

import importlib
import os

import pytest
import torch
import ttnn

from ttnn.operations.g6_permute import g6_permute

op_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute")
regime_module = importlib.import_module("ttnn.operations.g6_permute.g6_permute_blocks_rm_gather")

BLOCKS = (0, 1, 4, 2, 5, 3)
FLUX_74 = (1, 64, 128, 32, 2, 2)
GUARDS = os.environ.get("G6_ABLATION_GUARDS") == "1"

ARMS = {
    "full": [],
    "no_writes": ["G6_ABLATE_WRITES"],
    "no_reads": ["G6_ABLATE_READS"],
    "no_reads_no_writes": ["G6_ABLATE_READS", "G6_ABLATE_WRITES"],
    "no_compact": ["G6_ABLATE_COMPACT"],
    "no_gather": ["G6_ABLATE_GATHER"],
    "no_copies": ["G6_ABLATE_COMPACT", "G6_ABLATE_GATHER"],
    "no_copies_no_writes": ["G6_ABLATE_COMPACT", "G6_ABLATE_GATHER", "G6_ABLATE_WRITES"],
    "no_copies_no_reads": ["G6_ABLATE_COMPACT", "G6_ABLATE_GATHER", "G6_ABLATE_READS"],
    "all": ["G6_ABLATE_READS", "G6_ABLATE_WRITES", "G6_ABLATE_COMPACT", "G6_ABLATE_GATHER"],
}


@pytest.fixture
def fresh_cache():
    op_module._PROGRAM_CACHE.clear()
    yield
    op_module._PROGRAM_CACHE.clear()


def _input(device, memory_config):
    torch.manual_seed(5)
    torch_input = torch.randn(FLUX_74, dtype=torch.float32).to(torch.bfloat16)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    return torch_input, ttnn_input


@pytest.mark.parametrize("placement", ["dram", "l1"])
@pytest.mark.parametrize("arm", list(ARMS))
def test_ablation(arm, placement, device, fresh_cache, monkeypatch):
    if ARMS[arm] and not GUARDS:
        pytest.skip("needs the temporary ablation guards in the kernels (G6_ABLATION_GUARDS=1)")
    defines = [(name, "1") for name in ARMS[arm]]
    create = regime_module.create_program_descriptor

    def create_with_defines(plan, input_tensor, output_tensor):
        descriptor = create(plan, input_tensor, output_tensor)
        for kernel in descriptor.kernels:
            kernel.defines = defines
        return descriptor

    monkeypatch.setattr(regime_module, "create_program_descriptor", create_with_defines)
    memory_config = ttnn.L1_MEMORY_CONFIG if placement == "l1" else ttnn.DRAM_MEMORY_CONFIG
    torch_input, ttnn_input = _input(device, memory_config)
    actual = ttnn.to_torch(g6_permute(ttnn_input, BLOCKS))
    if not ARMS[arm]:
        expected = torch.permute(torch_input, BLOCKS).contiguous()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("placement", ["dram", "l1"])
@pytest.mark.parametrize("block_pages", [64, 128, 256, 512])
def test_block_pages(block_pages, placement, device, fresh_cache, monkeypatch):
    monkeypatch.setattr(regime_module, "BLOCK_PAGES", block_pages)
    memory_config = ttnn.L1_MEMORY_CONFIG if placement == "l1" else ttnn.DRAM_MEMORY_CONFIG
    torch_input, ttnn_input = _input(device, memory_config)
    actual = ttnn.to_torch(g6_permute(ttnn_input, BLOCKS))
    expected = torch.permute(torch_input, BLOCKS).contiguous()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
