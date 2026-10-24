# SPDX-License-Identifier: Apache-2.0
"""Compare packed and separate BF8-input prefill projections fairly."""
import hashlib
import json
from pathlib import Path
import sys
import ttnn
from . import run_prefill_column_candidate as column
from . import run_prefill_column_padded_candidate as padded
from . import run_prefill_column_interleaved_candidate as interleaved
from . import run_multichip_decoder as harness

_IMPORTED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ProjectionCandidate(padded.PaddedColumnCandidate):
    def _finish(self, residual, mixer):
        if not self.policy.get('_prefill_phase') or not self.policy.get('separate_prefill') or mixer.shape[1] == 1:
            return super()._finish(residual, mixer)
        if mixer.is_sharded() and residual.memory_config() != mixer.memory_config():
            residual = ttnn.to_memory_config(residual, mixer.memory_config())
        x = ttnn.add(residual, mixer, memory_config=mixer.memory_config())
        h = self._norm(x, self.post_norm)
        if h.shape[-1] == 1280:
            h = self._gather(h, 'gateup')
        gate = self._linear(h, self.gate, activation='silu')
        up = self._linear(h, self.up)
        inter = ttnn.multiply(gate, up)
        del h, gate, up  # The independent product owns its output before down.
        mlp = self._linear(inter, self.down)
        if mlp.is_sharded():
            x = ttnn.to_memory_config(x, mlp.memory_config())
        return ttnn.add(x, mlp, memory_config=mlp.memory_config())


if __name__ == '__main__':
    column.OptimizedDecoder._linear = column.column_linear
    harness.MultichipDecoder = ProjectionCandidate
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    try:
        harness.main()
    finally:
        if output.exists():
            value = json.loads(output.read_text())
            for name, sha in [('run_prefill_column_candidate', column._IMPORTED_SHA),
                              ('run_prefill_column_interleaved_candidate', interleaved._IMPORTED_SHA),
                              ('run_prefill_column_padded_candidate', padded._IMPORTED_SHA),
                              ('run_prefill_projection_candidate', _IMPORTED_SHA)]:
                value['provenance'][f'tests/{name}.py'] = sha
            value['column_weight_padding'] = padded._PADDING_RECEIPTS
            value['candidate_adapter'] = 'Separate BF8 gate/up share one gather, fused gate SiLU, BF16 product/down; unused operands released before down. Decode stays packed.'
            output.write_text(json.dumps(value, indent=2) + '\n')
