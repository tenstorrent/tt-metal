"""Source-only integration checks for candidate107; no TTNN import/device use."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'


def methods(source):
    return {n.name:n for cls in ast.parse(source).body if isinstance(cls,ast.ClassDef)
            for n in cls.body if isinstance(n,ast.FunctionDef)}


def effective(source):
    result={}
    for n in ast.parse(source).body:
        if isinstance(n,ast.FunctionDef): result[n.name]=n
        elif isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name): result[t.id]=n
    return result


def dump(node):
    return ast.dump(node,include_attributes=False)


def run_checks():
    source=(D/'candidate107.py').read_text()
    before=(D/'candidate103.py').read_text()
    manifest=json.loads((D/'candidate107.generation.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==manifest['output_sha256']
    for name,digest in manifest['sources'].items():
        assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest,name
    rebuilt,names=runpy.run_path(str(D/'make_candidate107.py'))['build']()
    assert rebuilt==source and names==manifest['helper_renames']
    old,new=methods(before),methods(source)
    changed=sorted(k for k in old if dump(old[k])!=dump(new[k]))
    assert changed==['_batched_conv','_gdn_compact'],changed
    for name in ('_linear_decode','_linear','_linear_raw','_finish','_norm','decode_forward','prefill_forward','from_state_dict'):
        assert dump(old[name])==dump(new[name]),name
    # Strip only the selection plumbing: original operation arguments and all
    # surrounding tensor arithmetic must be AST-identical.
    conv=copy.deepcopy(new['_batched_conv'])
    class RestoreConv(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='conv_op' for t in node.targets): return None
            return self.generic_visit(node)
        def visit_Name(self,node):
            if node.id=='conv_op': node.id='conv_decode_sfpu'
            return node
    assert dump(RestoreConv().visit(conv))==dump(old['_batched_conv'])
    recurrent=copy.deepcopy(new['_gdn_compact'])
    replacements=0
    for n in ast.walk(recurrent):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='recurrent_op' for t in n.targets):
            assert isinstance(n.value,ast.IfExp) and 'decay_predict_fusion' in ast.unparse(n.value.test)
            n.value=n.value.orelse
            replacements+=1
    assert replacements==1 and dump(recurrent)==dump(old['_gdn_compact'])
    old_top,new_top=effective(before),effective(source)
    for name,node in old_top.items():
        assert dump(node)==dump(new_top[name]),name
    # Exact helper AST after reversing Python-only namespacing; embedded C++
    # strings therefore remain exactly the independently probed source.
    helper_files={'conv':'conv_decode_sfpu_untilize_pipeline2.py','decay':'gdn_sfpu_decay_predict.py'}
    class RestoreNames(ast.NodeTransformer):
        def __init__(self,mapping): self.mapping=mapping
        def visit_Name(self,node):
            node.id=self.mapping.get(node.id,node.id)
            return node
        def visit_FunctionDef(self,node):
            node.name=self.mapping.get(node.name,node.name)
            return self.generic_visit(node)
    for group,mapping in names.items():
        originals=effective((D/helper_files[group]).read_text())
        assert not (set(mapping.values()) & set(old_top))
        for original,renamed in mapping.items():
            restored=RestoreNames({v:k for k,v in mapping.items()}).visit(copy.deepcopy(new_top[renamed]))
            assert dump(restored)==dump(originals[original]),original
    # Exercise both opt-in dispatch expressions and their precedence directly.
    choices={name:object() for name in ('conv_decode_sfpu','_conv107_conv_decode_sfpu',
        '_recurrent_sfpu_compact','_grouped96_recurrent_sfpu_compact','_decay107_recurrent_sfpu_compact')}
    expressions={}
    for method,var in (('_batched_conv','conv_op'),('_gdn_compact','recurrent_op')):
        expressions[var]=next(n.value for n in ast.walk(new[method]) if isinstance(n,ast.Assign)
                             and any(isinstance(t,ast.Name) and t.id==var for t in n.targets))
    routing_cases=0
    for conv_on in (False,True):
        for decay_on in (False,True):
            for grouped_on in (False,True):
                ns=choices|{'self':SimpleNamespace(policy={'conv_untilize_pipeline':conv_on,
                    'decay_predict_fusion':decay_on,'grouped_recurrent_init':grouped_on})}
                actual={k:eval(compile(ast.Expression(v),'<selector>','eval'),ns) for k,v in expressions.items()}
                assert actual['conv_op'] is choices['_conv107_conv_decode_sfpu' if conv_on else 'conv_decode_sfpu']
                expected='_decay107_recurrent_sfpu_compact' if decay_on else '_grouped96_recurrent_sfpu_compact' if grouped_on else '_recurrent_sfpu_compact'
                assert actual['recurrent_op'] is choices[expected]
                routing_cases+=1
    return {'result':'pass','hardware_validation':'pending cumulative signoff','changed_methods':changed,
        'source_hashes_verified':True,'deterministic_generation':True,'routing_cases':routing_cases,
        'disabled_math_ast_identical_to_candidate103':True,'original_helpers_and_defaults_unchanged':True,
        'helpers_exact_after_namespacing':True,'no_names_collide_with_candidate103':True,
        'phase98_decode_correction_and_prefill_S1_routing_unchanged':True}


def test_candidate107_source():
    run_checks()


if __name__=='__main__':
    result=run_checks()
    output=D.parent/'candidate107_cpu_checks.json'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
