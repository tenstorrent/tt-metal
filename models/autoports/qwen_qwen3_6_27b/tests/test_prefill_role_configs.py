"""CPU-only source harness for legal prefill geometry and arbitrary tail rows."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

D = Path(__file__).resolve().parents[1] / 'doc/optimized_decoder/candidates'


class Tensor:
    def __init__(self, shape, dtype='BF16'):
        self.shape, self.dtype = tuple(shape), dtype

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        indices = indices + (slice(None),) * (len(self.shape) - len(indices))
        return Tensor(tuple(len(range(*s.indices(n))) for n, s in zip(self.shape, indices)), self.dtype)


class TT:
    bfloat16, float32 = 'BF16', 'FP32'
    DRAM_MEMORY_CONFIG, L1_MEMORY_CONFIG = 'DRAM', 'L1'
    TensorMemoryLayout = SimpleNamespace(WIDTH_SHARDED='WIDTH')
    BufferType = SimpleNamespace(L1='L1')

    def __init__(self):
        self.calls = []

    @staticmethod
    def reshape(x, shape):
        assert math.prod(x.shape) == math.prod(shape)
        return Tensor(shape, x.dtype)

    @staticmethod
    def typecast(x, dtype):
        return Tensor(x.shape, dtype)

    @staticmethod
    def to_memory_config(x, memory):
        return x

    @staticmethod
    def MemoryConfig(*args):
        return args

    @staticmethod
    def MatmulMultiCoreReuseMultiCastProgramConfig(**kwargs):
        return kwargs

    @staticmethod
    def concat(parts, dim):
        shape = list(parts[0].shape)
        shape[dim] = sum(x.shape[dim] for x in parts)
        return Tensor(shape, parts[0].dtype)

    def linear(self, x, weight, **kwargs):
        self.calls.append((x.shape, weight.shape, kwargs))
        return Tensor((*x.shape[:-1], weight.shape[-1]), kwargs['dtype'])


def bind(path, tt, key, shape, options):
    source = ast.parse(path.read_text())
    method = next(m for cls in source.body if isinstance(cls, ast.ClassDef)
                  for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == '_linear_raw')
    ns = {'ttnn': tt, 'math': math}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), ns)
    layer = SimpleNamespace(policy={'prefill_grid': [11, 10], 'prefill_k': 8, **options},
        weights={key: Tensor(shape)}, weight_corrections={'packed_gdn': 'packed_gdn_residual'},
        projection_info={key: {'input_mem': 'L1', 'program': 'decode', 'logical_n': shape[-1]}},
        matmul_configs={'attention': 'attention_config', 'mlp': 'mlp_config'})
    layer._linear_raw = lambda x, key: ns['_linear_raw'](layer, x, key)
    return layer


def run_checks():
    count = 0
    keys = [('packed_gdn', (5120, 17408)), ('packed_gdn_residual', (5120, 17408)),
            ('linear_attn.out_proj.weight', (6144, 5120)), ('mlp.down_proj.weight', (17408, 5120))]
    for batch in (1, 3, 32):
        for seq in (1, 17, 33, 65, 128, 511, 512, 513, 1025):
            for key, shape in keys:
                x = Tensor((batch, seq, shape[0]))
                old_tt, new_tt = TT(), TT()
                old = bind(D / 'default81.py', old_tt, key, shape, {})
                new = bind(D / 'prefill_role_configs.py', new_tt, key, shape, {})
                assert old._linear_raw(x, key).shape == new._linear_raw(x, key).shape == (batch, seq, shape[1])
                assert old_tt.calls == new_tt.calls, 'Empty override must retain all existing matmul configs'
                for grid in ((11, 10), (10, 8), (8, 8)):
                    tt = TT()
                    role = key.removesuffix('_residual')
                    options = {'prefill_projection_configs': {role: {'grid': grid, 'subblock_h': 4,
                                'in0_block_w': 8 if shape[1] > 10000 else 16}}}
                    layer = bind(D / 'prefill_role_configs.py', tt, key, shape, options)
                    assert layer._linear_raw(x, key).shape == (batch, seq, shape[1])
                    for _, _, kwargs in tt.calls:
                        p = kwargs['program_config']
                        if p == 'decode':
                            continue
                        assert p['out_block_h'] == p['per_core_M']
                        assert p['out_block_w'] == p['per_core_N']
                        assert p['per_core_M'] % p['out_subblock_h'] == 0
                        assert p['per_core_N'] % p['out_subblock_w'] == 0
                        assert p['out_subblock_h'] * p['out_subblock_w'] <= 4
                        assert (shape[0] // 32) % p['in0_block_w'] == 0
                    count += 1
    return count


def test_prefill_geometry_tails_and_unchanged_default():
    run_checks()


if __name__ == '__main__':
    import json
    count = run_checks()
    output = D.parent / 'prefill_role_configs_cpu_checks.json'
    output.write_text(json.dumps({'cases': count, 'result': 'pass', 'hardware_validation': 'pending',
        'checks': ['empty overrides retain original configs', 'B1/B3/B32 arbitrary tails',
                   'residual inherits primary config', 'full output blocks retained',
                   'FP32 DST subblocks <=4 tiles', 'K blocks divide projection K']}, indent=2) + '\n')
    print(f'{count} CPU config cases passed; {output}')
