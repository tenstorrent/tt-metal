# SPDX-License-Identifier: Apache-2.0
"""Real-shape synthetic CI checks; checkpoint payloads are not required."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize('layer', [0, 3])
@pytest.mark.parametrize('length', [1, 2, 3, 31, 32, 33, 127, 128, 129])
def test_synthetic_fused_decoder(layer, length, tmp_path):
    command = [sys.executable, '-m', 'models.autoports.qwen_qwen3_8_27b.tests.run_fused_decoder',
               '--synthetic', '--layer', str(layer), '--seq', str(length),
               '--input-std', '0.012563',
               '--output', str(tmp_path / 'result.json')]
    if length > 128:
        command.append('--resume')
    subprocess.run(command, check=True)
