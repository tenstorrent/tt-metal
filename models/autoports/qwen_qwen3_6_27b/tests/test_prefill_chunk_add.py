"""CPU-only arithmetic/partition checks for the movement-only candidate.

No TTNN import or device initialization. These checks do not replace hardware
matmul, nonaligned-length, trace, state, or latency gates.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch

D = Path(__file__).resolve().parents[1] / 'doc/optimized_decoder/candidates'


def extract(path, function, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == function]
    assert len(nodes) == 1
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function]


class FakeTT:
    DRAM_MEMORY_CONFIG = 'DRAM'
    bfloat16 = torch.bfloat16

    def __init__(self):
        self.concatenations = []

    @staticmethod
    def to_memory_config(x, memory):
        assert memory == 'DRAM'
        return x

    @staticmethod
    def reshape(x, shape):
        return x.reshape(shape)

    @staticmethod
    def add(a, b, *, dtype):
        assert a.dtype == b.dtype == torch.float32
        return (a + b).to(dtype)

    def concat(self, parts, dim):
        self.concatenations.append([p.dtype for p in parts])
        return torch.cat(parts, dim=dim)


class Layer:
    def __init__(self, tt, physical_n, configured_cap, mismatch=False):
        self.tt = tt
        self.policy = {'matmul_row_cap': configured_cap}
        self.weight_corrections = {'packed_gdn': 'packed_gdn_residual'}
        self.weights = {k: SimpleNamespace(shape=(8, physical_n if not mismatch or i == 0 else 32768))
                        for i, k in enumerate(self.weight_corrections.keys() | self.weight_corrections.values())}
        # Deliberately preserve a known primary/residual mapping for mismatch tests.
        self.weights['packed_gdn'] = SimpleNamespace(shape=(8, physical_n))
        self.weights['packed_gdn_residual'] = SimpleNamespace(shape=(8, 32768 if mismatch else physical_n))
        self.projection_info = dict.fromkeys(self.weights, {})
        generator = torch.Generator().manual_seed(72)
        self.actual = {k: torch.randn(8, 32, generator=generator) * (1 if k == 'packed_gdn' else .003)
                       for k in self.weights}
        self.calls = []

    def _linear_raw(self, x, key):
        batch, seq, width = x.shape
        x = x.reshape(1, batch * seq, width)
        cap = 256 if self.weights[key].shape[-1] > 20000 else self.policy['matmul_row_cap']
        if batch * seq > cap:
            out = self.tt.concat([self._linear_raw(x[:, start:start + cap], key)
                                  for start in range(0, batch * seq, cap)], dim=1)
            return out.reshape(batch, seq, -1)
        # Each actual matmul records its original row order and exact shape.
        self.calls.append((key, tuple(x.shape), tuple(x[0, :, 0].tolist())))
        return torch.matmul(x.float(), self.actual[key]).reshape(batch, seq, -1)


def run_checks():
    torch.set_num_threads(1)
    cases = []
    for batch in (1, 3, 32):
        for seq in (1, 17, 33, 65, 128, 511, 512, 513, 1025):
            for physical_n, cap, mismatch in ((17408, 512, False), (32768, 512, False), (17408, 64, False), (17408, 512, True)):
                generator = torch.Generator().manual_seed(batch * 2000 + seq)
                x = (torch.randn(batch, seq, 8, generator=generator) * .1).to(torch.bfloat16)
                old_tt, new_tt = FakeTT(), FakeTT()
                old, new = Layer(old_tt, physical_n, cap, mismatch), Layer(new_tt, physical_n, cap, mismatch)
                baseline = extract(D / 'prefill_data_movement.py', 'corrected_projection_flat_prefill', {'ttnn': old_tt})
                candidate = extract(D / 'prefill_chunk_add.py', 'corrected_projection_flat_prefill', {'ttnn': new_tt})
                expected, actual = baseline(old, x, 'packed_gdn'), candidate(new, x, 'packed_gdn')
                assert torch.equal(expected, actual), (batch, seq, physical_n, cap, mismatch)
                assert sorted(old.calls) == sorted(new.calls), 'Matmul row geometry or input order changed'
                chunk_cap = 256 if physical_n > 20000 else cap
                chunked = not mismatch and batch * seq > chunk_cap
                if chunked:
                    assert len(new_tt.concatenations) == 1
                    assert all(t == torch.bfloat16 for t in new_tt.concatenations[0])
                    assert len(old_tt.concatenations) == 2
                    assert all(t == torch.float32 for row in old_tt.concatenations for t in row)
                cases.append({'batch': batch, 'seq': seq, 'physical_n': physical_n, 'cap': cap,
                              'mismatch_fallback': mismatch, 'chunked_add': chunked,
                              'bf16_exact': True, 'matmul_geometry_identical': True})
    original = ast.parse((D / 'combo73.py').read_text())
    candidate = ast.parse((D / 'chunk_add_prefill.py').read_text())
    methods = lambda tree: {n.name: ast.dump(n, include_attributes=False)
                           for cls in tree.body if isinstance(cls, ast.ClassDef)
                           for n in cls.body if isinstance(n, ast.FunctionDef)}
    old_methods, new_methods = methods(original), methods(candidate)
    assert old_methods.keys() == new_methods.keys()
    assert [k for k in old_methods if old_methods[k] != new_methods[k]] == ['_linear']
    return cases


def test_chunk_add_arithmetic_and_partitions():
    run_checks()


if __name__ == '__main__':
    import json
    rows = run_checks()
    output = D.parent / 'prefill_chunk_add_cpu_checks.json'
    output.write_text(json.dumps({'hardware_validation': 'pending', 'cases': len(rows), 'rows': rows}, indent=2) + '\n')
    print(f'{len(rows)} CPU partition/arithmetic cases passed; {output}')
