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
    cfg = self.policy.get(role, {})
    fields = {'prefill_out_block_h': 'out_block_h',
              'prefill_out_block_w': 'out_block_w',
              'prefill_out_subblock_w': 'out_subblock_w'}
    overrides = {target: cfg[key] for key, target in fields.items() if key in cfg}
    if not self.policy.get('_prefill_phase') or not overrides:
        return _base_linear(self, x, w, *args, **kwargs)
    import ttnn
    constructor = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig
    def configured(*positional, **config):
        config.update(overrides)
        return constructor(*positional, **config)
    # Single-threaded graph construction only; replay invokes no Python.
    ttnn.MatmulMultiCoreReuseMultiCastProgramConfig = configured
    try:
        return _base_linear(self, x, w, *args, **kwargs)
    finally:
        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig = constructor


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
