# SPDX-License-Identifier: Apache-2.0
"""Tail-free interleaved prefill column candidates with exact setup padding."""
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
import ttnn
from . import run_prefill_column_candidate as column
from . import run_prefill_column_interleaved_candidate as interleaved
from . import run_multichip_decoder as harness

_IMPORTED_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_PADDING_RECEIPTS = []


class PaddedColumnCandidate(interleaved.ColumnWeightCandidate):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        model = super().from_state_dict(*args, **kwargs)
        view = model._prefill_impl
        model.column_padding_evidence = []
        composer = ttnn.ConcatMeshToTensor(model.mesh_device, dim=0)
        for role in model.policy.get('column_interleaved_roles', []):
            weight = getattr(view, role)
            old_n = weight.shape[-1]
            gx = view.policy[role].get('prefill_grid', view.policy.get('prefill_grid', [11,10]))[0]
            new_n = math.ceil(old_n / (32 * gx)) * 32 * gx
            original = ttnn.to_torch(weight, mesh_composer=composer)
            if new_n != old_n:
                wide = ttnn.reshape(ttnn.typecast(weight, ttnn.bfloat16), [1,1,weight.shape[-2],old_n])
                wide = ttnn.pad(wide, [(0,0), (0,0), (0,0), (0,new_n-old_n)], 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                weight = ttnn.reshape(ttnn.typecast(wide, ttnn.bfloat4_b), [weight.shape[-2],new_n])
                del wide
            actual = ttnn.to_torch(weight, mesh_composer=composer)
            assert torch.equal(original, actual[..., :old_n]), 'Padding changed quantized weight values'
            assert torch.count_nonzero(actual[..., old_n:]) == 0, 'Nonzero padding'
            view.weight_roles[id(weight)] = role
            view.logical_widths[id(weight)] = view.logical_widths[id(getattr(view,role))]
            setattr(view, role, weight)
            model.column_padding_evidence.append(dict(role=role,original_n=old_n,padded_n=new_n,
                                                      grid_x=gx,original_columns_exact=True,zero_tail=True))
        _PADDING_RECEIPTS.extend(model.column_padding_evidence)
        return model


if __name__ == '__main__':
    column.OptimizedDecoder._linear = column.column_linear
    harness.MultichipDecoder = PaddedColumnCandidate
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    try:
        harness.main()
    finally:
        if output.exists():
            value = json.loads(output.read_text())
            value['provenance']['tests/run_prefill_column_candidate.py'] = column._IMPORTED_SHA
            value['provenance']['tests/run_prefill_column_interleaved_candidate.py'] = interleaved._IMPORTED_SHA
            value['provenance']['tests/run_prefill_column_padded_candidate.py'] = _IMPORTED_SHA
            value['column_weight_padding'] = _PADDING_RECEIPTS
            value['candidate_adapter'] = 'Grid-aligned interleaved prefill copies, original quantized columns exact and zero padded tail checked before capture; logical output slicing and original decode weights retained.'
            output.write_text(json.dumps(value, indent=2) + '\n')
