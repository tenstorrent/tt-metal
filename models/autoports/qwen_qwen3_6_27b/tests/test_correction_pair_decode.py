"""CPU-only correction-pair algebra, shape, dispatch and explicit-phase checks."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch

D = Path(__file__).resolve().parents[1] / 'doc/optimized_decoder/candidates'


def methods(path):
    return {n.name: n for cls in ast.parse(path.read_text()).body if isinstance(cls, ast.ClassDef)
            for n in cls.body if isinstance(n, ast.FunctionDef)}


class TT:
    float32, bfloat16 = torch.float32, torch.bfloat16
    DRAM_MEMORY_CONFIG = 'DRAM'
    TensorMemoryLayout = SimpleNamespace(WIDTH_SHARDED='WIDTH')
    BufferType = SimpleNamespace(L1='L1')

    def __init__(self):
        self.matmul_calls = 0
        self.slices = []

    @staticmethod
    def MemoryConfig(*args):
        return args

    @staticmethod
    def to_memory_config(x, memory):
        return x

    @staticmethod
    def reshape(x, shape):
        return x.reshape(shape)

    def linear(self, x, weight, **kwargs):
        assert kwargs['dtype'] == torch.float32
        self.matmul_calls += 1
        return torch.matmul(x.float(), weight)

    def slice(self, x, starts, ends, steps, *, memory_config):
        self.slices.append((starts, ends, steps, memory_config))
        return x[tuple(slice(a, b, s) for a, b, s in zip(starts, ends, steps))]

    @staticmethod
    def add(a, b, *, dtype):
        assert a.dtype == b.dtype == torch.float32
        return (a + b).to(dtype)


def run_checks():
    torch.set_num_threads(1)
    old = methods(D / 'default81.py')
    new = methods(D / 'correction_pair_decode.py')
    assert ast.dump(old['prefill_forward']) == ast.dump(new['prefill_forward'])
    assert ast.dump(old['_linear_raw']) == ast.dump(new['_linear_raw'])
    # The two compact projections receive the explicit phase argument.
    targeted = []
    for node in ast.walk(new['_gdn_compact']):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == '_linear':
            if isinstance(node.args[1], ast.Constant) and node.args[1].value in ('packed_gdn', 'linear_attn.out_proj.weight'):
                assert any(k.arg == 'decode_pair' and isinstance(k.value, ast.Name) and k.value.id == 'decode'
                           for k in node.keywords)
                targeted.append(node.args[1].value)
    assert sorted(targeted) == ['linear_attn.out_proj.weight', 'packed_gdn']
    assert isinstance(new['_gdn_compact'].args.defaults[-1], ast.Constant)
    assert new['_gdn_compact'].args.defaults[-1].value is False
    assert new['_gdn'].args.kw_defaults[0].value is False
    for node in ast.walk(new['decode_forward']):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('_gdn', '_gdn_compact'):
            assert any(k.arg == 'decode' and isinstance(k.value, ast.Constant) and k.value.value is True for k in node.keywords)
    assert '_correction_pair_decode_active' not in (D / 'correction_pair_decode.py').read_text()
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('to_torch', 'from_torch')
                   for n in ast.walk(new['_linear_correction_pair']))
    cases = 0
    generator = torch.Generator().manual_seed(923)
    for n in (16512, 5120):
        # Dyadic, quantized-like values keep CPU FP32 sums exact, isolating the
        # column packing/phase algebra rather than CPU BLAS reduction policies.
        w0 = torch.randint(-63, 64, (32, n), generator=generator).float() / 64
        w1 = torch.randint(-63, 64, (32, n), generator=generator).float() / 4096
        physical = ((2*n + 1023) // 1024) * 1024
        joined = torch.nn.functional.pad(torch.cat((w0, w1), dim=-1), (0, physical - 2*n))
        assert torch.equal(joined[:, :n], w0) and torch.equal(joined[:, n:2*n], w1)
        assert (physical // 32 // 8) % 4 == 0, 'Weight-bank width must divide four readers'
        for batch in (1, 3, 31, 32):
            for compact in (False, True):
                x = (torch.randint(-15, 16, (batch, 1, 32), generator=generator).float() / 32).bfloat16()
                if compact:
                    x = x.reshape(1, batch, 32)
                for split in ('sharded', 'dram'):
                    tt = TT()
                    ns = {'ttnn': tt}
                    for name in ('_linear_correction_pair', '_linear'):
                        exec(compile(ast.Module(body=[new[name]], type_ignores=[]), str(D), 'exec'), ns)
                    layer = SimpleNamespace(
                        policy={'correction_pair_split': split}, weight_corrections={'packed_gdn': 'residual'},
                        weights={'paired': joined}, matmul_configs={'attention': 'unchanged'},
                        correction_pair_info={'packed_gdn': {'pair_key': 'paired', 'logical_n': n, 'split_memory': 'L1_SPLIT'}},
                        projection_info={'paired': {'input_mem': 'L1_INPUT', 'program': 'C32_R4_K1'}})
                    raw_calls = []
                    def raw(a, key):
                        raw_calls.append(key)
                        return torch.matmul(a.float(), w0 if key == 'packed_gdn' else w1)
                    layer._linear_raw = raw
                    layer._linear_correction_pair = lambda a, key: ns['_linear_correction_pair'](layer, a, key)
                    prefill = ns['_linear'](layer, x, 'packed_gdn')
                    assert raw_calls == ['packed_gdn', 'residual'] and tt.matmul_calls == 0
                    raw_calls.clear()
                    decode = ns['_linear'](layer, x, 'packed_gdn', decode_pair=True)
                    assert raw_calls == [] and tt.matmul_calls == 1 and len(tt.slices) == 2
                    assert decode.shape == prefill.shape and torch.equal(decode, prefill)
                    assert tt.slices[1][0][-1] == n and tt.slices[1][1][-1] == 2*n
                    cases += 1
    return cases


def test_correction_pair_algebra_and_phase():
    run_checks()


if __name__ == '__main__':
    import json
    cases = run_checks()
    output = D.parent / 'correction_pair_cpu_checks.json'
    output.write_text(json.dumps({'cases': cases, 'result': 'pass', 'hardware_validation': 'pending',
        'checks': ['prefill_forward and _linear_raw AST unchanged', 'explicit decode propagation with False prefill default',
                   'input and output widths; B1/B3/B31/B32; ordinary/compact layouts; sharded/DRAM splits',
                   'single packed matmul, exact column order and BF16 sum for dyadic CPU control',
                   'tile/bank/four-reader divisibility; no runtime host conversion']}, indent=2) + '\n')
    print(f'{cases} CPU pair/phase cases passed; {output}')
