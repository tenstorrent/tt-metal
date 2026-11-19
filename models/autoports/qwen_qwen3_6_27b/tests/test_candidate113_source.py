"""CPU/source routing checks for B1 classic attention and decode-only SDPA grid."""
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


def dump(node): return ast.dump(node,include_attributes=False)


def compile_function(node,ns):
    exec(compile(ast.Module(body=[copy.deepcopy(node)],type_ignores=[]),'<source-check>','exec'),ns)
    return ns[node.name]


class TT:
    DRAM_MEMORY_CONFIG='DRAM'
    @staticmethod
    def reshape(x,shape): return SimpleNamespace(shape=tuple(shape))
    @staticmethod
    def to_memory_config(x,memory): return x


class ClassicRoute(Exception): pass


def run_checks():
    source=(D/'candidate113.py').read_text();before=(D/'candidate107.py').read_text()
    manifest=json.loads((D/'candidate113.generation.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==manifest['output_sha256']
    assert hashlib.sha256(before.encode()).hexdigest()==manifest['source_sha256']
    assert runpy.run_path(str(D/'make_candidate113.py'))['build']()==source
    old,new=methods(before),methods(source)
    changed=sorted(k for k in old if dump(old[k])!=dump(new[k]))
    assert changed==['_attention_decode','_compact_attention_decode','decode_forward'],changed
    assert sorted(set(new)-set(old))==['_decode_sdpa_grid','_single_user_classic_decode']
    for name in ('prefill_forward','_attention_prefill','_gdn','_gdn_compact','_linear_decode',
                 '_linear','_linear_raw','_finish','_norm','from_state_dict','create_paged_state','_batched_conv'):
        assert dump(old[name])==dump(new[name]),name
    # All existing module-level definitions/constants remain byte-for-byte AST
    # equal, including defaults and every retained/namespaced GDN kernel.
    old_top=[n for n in ast.parse(before).body if not isinstance(n,ast.ClassDef)]
    new_top=[n for n in ast.parse(source).body if not isinstance(n,ast.ClassDef)]
    assert [dump(n) for n in old_top]==[dump(n) for n in new_top]
    class RestoreAttentionCalls(ast.NodeTransformer):
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute):
                if node.func.attr in ('_attention_decode','_compact_attention_decode'):
                    node.keywords=[k for k in node.keywords if k.arg!='decode']
                if node.func.attr=='_decode_sdpa_grid':
                    return ast.parse('self.policy.get("sdpa_grid", [8, 4])',mode='eval').body
            return self.generic_visit(node)
    for name in changed:
        restored=copy.deepcopy(new[name])
        if name=='decode_forward':
            guard=next(n for n in restored.body if isinstance(n,ast.If))
            assert '_single_user_classic_decode' in ast.unparse(guard.test)
            guard.test.values.pop()
        elif name=='_attention_decode':
            restored.args.kwonlyargs=[];restored.args.kw_defaults=[]
            restored.body[0].test=restored.body[0].test.values[0]
        else:
            assert restored.args.args[-1].arg=='decode' and restored.args.defaults[-1].value is False
            restored.args.args.pop();restored.args.defaults.pop()
        restored=RestoreAttentionCalls().visit(restored)
        assert dump(restored)==dump(old[name]),name
    # Route actual decode_forward and actual _attention_decode using tensor
    # metadata/stubs. Stop the classical route at _qkv, before tensor math.
    ns={'ttnn':TT}
    funcs={name:compile_function(new[name],ns) for name in ('decode_forward','_attention_decode',
        '_single_user_classic_decode','_decode_sdpa_grid')}
    routing_cases=0
    for kind in ('full_attention','linear_attention'):
        for batch in (1,2,3,31,32):
            for flat in (False,True):
                for compact_residual in (False,True):
                    for enabled in (False,True):
                        calls=[]
                        layer=SimpleNamespace(kind=kind,policy={'compact_residual':compact_residual,
                            'compact_attention':True,'compact_gdn':True,'single_user_classic_attention':enabled})
                        layer._single_user_classic_decode=lambda b,decode=False:funcs['_single_user_classic_decode'](layer,b,decode=decode)
                        layer._norm=lambda x,key:x
                        layer._finish=lambda x,a:x
                        layer._round_recurrent_state=lambda state:None
                        def compact_attn(x,state,pos,rope,flat=False,decode=False):
                            calls.append(('compact_attention',decode));return x
                        def compact_gdn(x,state,flat=False,decode=False):
                            calls.append(('compact_gdn',decode));return x
                        def gdn(x,state): calls.append(('gdn',False));return x
                        def classic_qkv(x,rope):
                            calls.append(('classic_attention',True));raise ClassicRoute
                        layer._compact_attention_decode=compact_attn;layer._gdn_compact=compact_gdn
                        layer._gdn=gdn;layer._qkv=classic_qkv
                        layer._attention_decode=lambda x,s,p,r,decode=False:funcs['_attention_decode'](layer,x,s,p,r,decode=decode)
                        x=SimpleNamespace(shape=(1,batch,5120) if flat else (batch,1,5120))
                        try: funcs['decode_forward'](layer,x,state=None)
                        except ClassicRoute: pass
                        if kind=='full_attention':
                            expected='classic_attention' if enabled and batch==1 else 'compact_attention'
                            assert calls==[(expected,True)],(kind,batch,flat,compact_residual,enabled,calls)
                        else:
                            assert calls==[('compact_gdn',True) if compact_residual else ('gdn',False)]
                        # Unaligned prefill calls _attention_decode without an
                        # explicit phase. Even enabled B1 must remain compact.
                        if kind=='full_attention':
                            calls.clear()
                            funcs['_attention_decode'](layer,SimpleNamespace(shape=(batch,1,5120)),None,None,None)
                            assert calls==[('compact_attention',False)]
                        routing_cases+=1
    grid_cases=0
    for kind in ('full_attention','linear_attention'):
        for batch in (1,2,3,31,32):
            for decode in (False,True):
                for override in (None,[11,10]):
                    for explicit_base in (False,True):
                        policy={'single_user_sdpa_grid':override}
                        if explicit_base: policy['sdpa_grid']=[8,8]
                        layer=SimpleNamespace(kind=kind,policy=policy)
                        base=[8,8] if explicit_base else [8,4]
                        expected=override if kind=='full_attention' and batch==1 and decode and override is not None else base
                        assert funcs['_decode_sdpa_grid'](layer,batch,decode=decode)==expected
                        grid_cases+=1
    return {'result':'pass','hardware_validation':'pending cumulative signoff','changed_methods':changed,
        'source_hashes_verified':True,'deterministic_generation':True,'routing_cases':routing_cases,
        'grid_cases':grid_cases,'disabled_attention_math_ast_unchanged':True,
        'prefill_GDN_defaults_cache_dtype_and_projection_phase_ast_unchanged':True,
        'unaligned_prefill_does_not_select_B1_overrides':True}


def test_candidate113_source(): run_checks()


if __name__=='__main__':
    result=run_checks()
    (D.parent/'candidate113_cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
