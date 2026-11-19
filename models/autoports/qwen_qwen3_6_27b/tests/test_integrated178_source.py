"""CPU-only preservation and arbitrary-tail config checks for candidate178."""
import ast
import hashlib
import json
import runpy
from pathlib import Path

D = Path(__file__).resolve().parents[1] / 'doc/optimized_decoder/candidates'


def run_checks():
    generator = runpy.run_path(str(D / 'make_integrated178.py'))
    source = (D / 'integrated178.py').read_text()
    base = (D / 'integrated173.py').read_text()
    assert generator['build']() == source
    record = json.loads((D / 'integrated178.generation.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest() == record['output_sha256']
    for name, digest in record['sources'].items():
        assert hashlib.sha256((D / name).read_bytes()).hexdigest() == digest
    dump = lambda n: ast.dump(n, include_attributes=False)
    def policy(tree):
        return next(n for n in tree.body if isinstance(n, ast.Assign) and
                    any(isinstance(t, ast.Name) and t.id == 'DEFAULT_POLICIES' for t in n.targets))
    a, b = ast.parse(base), ast.parse(source)
    pa, pb = policy(a), policy(b)
    expected = ast.literal_eval(pa.value)
    expected['linear_attention'].update(generator['SELECTED'])
    assert ast.literal_eval(pb.value) == expected
    # Remove precisely the allowed differences, then compare the entire AST.
    pb.value = pa.value
    old = generator['method'](base, '_linear_raw')
    for c in b.body:
        if isinstance(c, ast.ClassDef):
            c.body = [old if isinstance(n, ast.FunctionDef) and n.name == '_linear_raw' else n for n in c.body]
    assert dump(a) == dump(b)
    role = (D / 'prefill_split_role_configs.py').read_text()
    assert dump(generator['method'](source, '_linear_raw')) == dump(generator['method'](role, '_linear_raw'))
    checks = runpy.run_path(str(Path(__file__).with_name('test_prefill_split_role_configs_source.py')))
    existing = checks['run_checks']()
    selected = generator['SELECTED']['prefill_projection_configs']['packed_gdn_residual']
    for rows in range(1, 641):
        g = checks['geometry'](rows, 11520, selected)
        assert g['kb'] == 10 and g['sw'] == 3
    return {'result': 'pass', 'original_role_geometry_cases': existing['geometry_cases'],
            'selected_arbitrary_tail_cases': 640, 'entire_other_ast_identical_to173': True,
            'recurrence_and_affine_kernel_strings_unchanged': True,
            'selected_policy_additions': generator['SELECTED'], 'hardware_validation': 'pending cumulative178'}


if __name__ == '__main__':
    result = run_checks()
    (D.parent / 'integrated178_cpu_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
