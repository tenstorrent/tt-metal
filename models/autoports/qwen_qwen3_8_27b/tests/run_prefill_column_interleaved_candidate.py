# SPDX-License-Identifier: Apache-2.0
"""Setup-only interleaved prefill weights isolate the narrow-N reader contract."""
import hashlib
import json
from pathlib import Path
import sys
import torch
import ttnn
from . import run_prefill_column_candidate as column
from . import run_multichip_decoder as harness

_IMPORTED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class ColumnWeightCandidate(harness.MultichipDecoder):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        model = super().from_state_dict(*args, **kwargs)
        view = model._prefill_impl
        view.weight_roles = dict(view.weight_roles)
        view.logical_widths = dict(view.logical_widths)
        for role in model.policy.get('column_interleaved_roles', []):
            original = getattr(view, role)
            weight = ttnn.to_memory_config(original, ttnn.DRAM_MEMORY_CONFIG)
            view.weight_roles[id(weight)] = role
            view.logical_widths[id(weight)] = view.logical_widths[id(original)]
            setattr(view, role, weight)
        return model


if __name__ == '__main__':
    column.OptimizedDecoder._linear = column.column_linear
    harness.MultichipDecoder = ColumnWeightCandidate
    original_equal = torch.equal
    def checked_equal(left, right):
        exact = original_equal(left, right)
        if not exact:
            print('EXACT_MISMATCH_DIAGNOSTIC', dict(shape=list(left.shape),
                  left_nonfinite=int((~torch.isfinite(left)).sum()),
                  right_nonfinite=int((~torch.isfinite(right)).sum()),
                  unequal_elements=int((left != right).sum())), flush=True)
        return exact
    torch.equal = checked_equal
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    try:
        harness.main()
    finally:
        if output.exists():
            value = json.loads(output.read_text())
            value['provenance']['tests/run_prefill_column_candidate.py'] = column._IMPORTED_SHA
            value['provenance']['tests/run_prefill_column_interleaved_candidate.py'] = _IMPORTED_SHA
            value['candidate_adapter'] = 'Interleaved prefill weight copies allocated before capture; original DRAM-sharded decode weights retained. Exact equality gate unchanged.'
            output.write_text(json.dumps(value, indent=2) + '\n')
