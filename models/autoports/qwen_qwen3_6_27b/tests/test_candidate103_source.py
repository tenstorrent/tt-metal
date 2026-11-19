"""CPU-only cumulative integration checks; never imports TTNN or opens a device."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

D = Path(__file__).resolve().parents[1] / 'doc/optimized_decoder/candidates'


def class_methods(source):
    return {n.name: n for cls in ast.parse(source).body if isinstance(cls, ast.ClassDef)
            for n in cls.body if isinstance(n, ast.FunctionDef)}


def dump(node):
    return ast.dump(node, include_attributes=False)


def compile_function(node, namespace):
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), '<source-check>', 'exec'), namespace)
    return namespace[node.name]


def run_checks():
    source = (D / 'candidate103.py').read_text()
    phase = (D / 'phase98.py').read_text()
    manifest = json.loads((D / 'candidate103.generation.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest() == manifest['output_sha256']
    for name, digest in manifest['sources'].items():
        assert hashlib.sha256((D / name).read_bytes()).hexdigest() == digest, name
    gen = runpy.run_path(str(D / 'make_candidate103.py'))
    rebuilt, names = gen['build']()
    assert rebuilt == source and names == manifest['grouped96_renames']
    old, new = class_methods(phase), class_methods(source)
    changed = sorted(k for k in old if dump(old[k]) != dump(new[k]))
    assert changed == ['_finish', '_gdn_compact', '_linear', '_linear_raw'], changed
    for name in ('_linear_decode', 'prefill_forward', 'decode_forward', '_gdn', '_norm', 'from_state_dict'):
        assert dump(old[name]) == dump(new[name]), name
    assignment = lambda s, name: next(n for n in ast.parse(s).body if isinstance(n, ast.Assign)
                                      and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    assert dump(assignment(phase, 'DEFAULT_POLICIES')) == dump(assignment(source, 'DEFAULT_POLICIES'))
    # Disabling the one new _linear guard restores the exact phase98 dispatcher.
    linear = copy.deepcopy(new['_linear'])
    guards = [n for n in linear.body if isinstance(n, ast.If) and 'chunk_add_prefill_correction' in ast.unparse(n.test)]
    assert len(guards) == 1
    linear.body.remove(guards[0])
    assert dump(linear) == dump(old['_linear'])
    # Only a switchable recurrence-function selection changes compact GDN.
    compact = copy.deepcopy(new['_gdn_compact'])
    class NormalizeRecurrence(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'recurrent_op' for t in node.targets):
                return None
            return self.generic_visit(node)
        def visit_Name(self, node):
            if node.id == 'recurrent_op':
                node.id = '_recurrent_sfpu_compact'
            return node
    compact = NormalizeRecurrence().visit(compact)
    assert dump(compact) == dump(old['_gdn_compact'])
    assert dump(new['_finish']) == dump(class_methods((D / 'shared90.py').read_text())['_finish'])
    expected_raw = class_methods(runpy.run_path(str(D / 'make_prefill_role_configs.py'))['build'](phase))['_linear_raw']
    assert dump(new['_linear_raw']) == dump(expected_raw)
    # Every original effective top-level helper/constant survives unchanged.
    def effective(s):
        result = {}
        for n in ast.parse(s).body:
            if isinstance(n, ast.FunctionDef): result[n.name] = n
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name): result[t.id] = n
        return result
    old_top, new_top = effective(phase), effective(source)
    for name, node in old_top.items():
        assert dump(node) == dump(new_top[name]), name
    grouped_top = effective((D / 'grouped96.py').read_text())
    for old_name, new_name in names.items():
        restored = gen['rename_identifiers'](gen['segment'](source, new_top[new_name]), {v:k for k,v in names.items()})
        assert dump(ast.parse(restored).body[0]) == dump(grouped_top[old_name]), old_name
    chunk_top = effective((D / 'chunk94.py').read_text())
    renamed_chunk = copy.deepcopy(new_top['corrected_projection_chunk_add_prefill'])
    renamed_chunk.name = 'corrected_projection_flat_prefill'
    assert dump(renamed_chunk) == dump(chunk_top['corrected_projection_flat_prefill'])
    # Projection selection is tested directly: changing decode correction
    # selection never changes the unchanged prefill dispatcher.
    ns = {'ttnn': SimpleNamespace(bfloat16='BF16', typecast=lambda x,dtype: ('cast',x,dtype))}
    project = compile_function(new['_linear_decode'], ns)
    layer = SimpleNamespace(policy={}, weight_corrections={'packed_gdn':'p_res','linear_attn.out_proj.weight':'o_res'},
        _linear=lambda x,k: ('corrected',k), _linear_raw=lambda x,k: ('raw',k))
    assert project(layer, None, 'linear_attn.out_proj.weight') == ('corrected','linear_attn.out_proj.weight')
    layer.policy={'decode_weight_correction':['packed_gdn']}
    assert project(layer,None,'packed_gdn') == ('corrected','packed_gdn')
    assert project(layer,None,'linear_attn.out_proj.weight') == ('cast',('raw','linear_attn.out_proj.weight'),'BF16')
    # Exercise the imported role-config source across nonaligned tails without
    # allocating model tensors or invoking TTNN.
    helpers = runpy.run_path(str(Path(__file__).with_name('test_prefill_role_configs.py')))
    TT, Tensor, bind = (helpers[k] for k in ('TT','Tensor','bind'))
    geometry_cases = 0
    for batch in (1,3,32):
        for seq in (1,17,33,65,128,511,512,513,1025):
            for key, shape in [('packed_gdn',(5120,17408)),('linear_attn.out_proj.weight',(6144,5120)),('mlp.down_proj.weight',(17408,5120))]:
                a,b=TT(),TT()
                x=Tensor((batch,seq,shape[0]))
                left,right=bind(D/'phase98.py',a,key,shape,{}),bind(D/'candidate103.py',b,key,shape,{})
                assert left._linear_raw(x,key).shape == right._linear_raw(x,key).shape
                assert a.calls==b.calls
                for grid in ((11,10),(10,8),(8,8)):
                    tt=TT(); options={'prefill_projection_configs':{key:{'grid':grid,'subblock_h':4,'in0_block_w':8}}}
                    layer=bind(D/'candidate103.py',tt,key,shape,options)
                    assert layer._linear_raw(x,key).shape==(batch,seq,shape[1])
                    for _,_,kw in tt.calls:
                        p=kw['program_config']
                        if p=='decode': continue
                        assert p['out_block_h']==p['per_core_M'] and p['out_block_w']==p['per_core_N']
                        assert p['per_core_M']%p['out_subblock_h']==0
                        assert p['per_core_N']%p['out_subblock_w']==0
                    geometry_cases += 1
    return {'result':'pass','hardware_validation':'pending cumulative signoff',
            'changed_methods':changed,'geometry_cases':geometry_cases,
            'source_hashes_verified':True,'deterministic_generation':True,
            'unchanged_defaults_and_phase98_projection_selection':True,
            'unchanged_prefill_and_S1_phase_routing':True,
            'grouped96_final_definitions_exact_after_namespacing':True,
            'shared90_finish_exact':True,'chunk94_helper_exact_after_rename':True}


def test_candidate103_source_and_configs():
    run_checks()


if __name__ == '__main__':
    result=run_checks()
    output=D.parent/'candidate103_cpu_checks.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
