# SPDX-License-Identifier: Apache-2.0
"""Real-weight optimized-path regression, including odd batches and chunk tails.

Requires the recorded HF fixtures produced by record_optimized_inputs.py. Run
serially on physical device0; no xdist. The command harness checks every PCC,
trace determinism, refresh, page remapping and continuation before returning.
"""
import json
import os
import subprocess
from pathlib import Path
import pytest
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import default_policy

@pytest.mark.parametrize('layer', [0, 3])
def test_optimized_decoder_contract(layer, tmp_path):
    base = Path(__file__).resolve().parents[1]
    label = f'pytest_contract_l{layer}'
    length = default_policy('linear_attention' if layer == 0 else 'full_attention')['prefill_chunk'] + 1
    env = {**os.environ, 'QWEN_OPT_POLICY': '{}'}
    command = ['bash', str(base/'tests/optimized_run.sh'), 'check', label,
               '--layer', str(layer), '--batch', '3', '--seq', str(length),
               '--recorded', '--resume', '--requests']
    if layer == 3:
        command.append('--remap')
    subprocess.run(command, env=env, check=True)
    result = json.loads((base/f'doc/optimized_decoder/{label}.json').read_text())
    assert result['runtime_class'] == 'OptimizedDecoder'
    assert result['runtime_fallback_audit'] and result['deterministic']
    assert result['prefill_pcc'] >= .995 and result['decode_pcc'] >= .995
