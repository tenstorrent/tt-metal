# SPDX-License-Identifier: Apache-2.0
"""Precision-locked prefill column geometry on the selected TP4 family."""
import hashlib
import json
from pathlib import Path
import sys
from . import run_multichip_decoder as harness
from ..tt.optimized_decoder import OptimizedDecoder

_IMPORTED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_base_linear = OptimizedDecoder._linear


def column_linear(self, x, w, *args, **kwargs):
    role = self.weight_roles[id(w)]
    block_h = self.policy.get(role, {}).get('prefill_out_block_h')
    if not self.policy.get('_prefill_phase') or block_h is None:
        return _base_linear(self, x, w, *args, **kwargs)
    old = self.policy.get('prefill_out_block_h')
    self.policy['prefill_out_block_h'] = block_h
    try:
        return _base_linear(self, x, w, *args, **kwargs)
    finally:
        if old is None:
            self.policy.pop('prefill_out_block_h', None)
        else:
            self.policy['prefill_out_block_h'] = old


if __name__ == '__main__':
    OptimizedDecoder._linear = column_linear
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    try:
        harness.main()
    finally:
        if output.exists():
            value = json.loads(output.read_text())
            value['provenance']['tests/run_prefill_column_candidate.py'] = _IMPORTED_SHA
            value['candidate_adapter'] = 'Only per-role column output-block geometry; production native RS, residual, precision and decode unchanged.'
            output.write_text(json.dumps(value, indent=2) + '\n')
