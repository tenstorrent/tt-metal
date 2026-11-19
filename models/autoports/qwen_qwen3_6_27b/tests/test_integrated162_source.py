"""CPU/source integration gates for candidate162; never import TTNN."""
import ast,copy,hashlib,json,runpy,types
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'


def run_checks():
    source=(D/'integrated162.py').read_text();base=(D/'integrated_fused_persistence134.py').read_text()
    generator=runpy.run_path(str(D/'make_integrated162.py'))
    assert generator['build']()==source
    record=json.loads((D/'integrated162.generation.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==record['output_sha256']
    for n,h in record['sources'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h
    defs=generator['definitions'];dump=lambda n:ast.dump(n,include_attributes=False)
    old,new=defs(base,'OptimizedDecoder'),defs(source,'OptimizedDecoder')
    expected={name for names in generator['METHOD_SOURCES'].values() for name in names}
    changed={name for name in old if dump(old[name])!=dump(new[name])}
    assert changed==expected
    for filename,names in generator['METHOD_SOURCES'].items():
        incoming=defs((D/filename).read_text(),'OptimizedDecoder')
        for name in names:assert dump(new[name])==dump(incoming[name])
    for method,filename in record['additional_methods'].items():
        assert dump(new[method])==dump(defs((D/filename).read_text(),'OptimizedDecoder')[method])
    # Includes _gdn_compact, _gdn, _delta_chunks, all cache/state constructors.
    assert dump(old['_gdn_compact'])==dump(new['_gdn_compact'])
    def globals_by_name(text):
        result={}
        for n in ast.parse(text).body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)):result[n.name]=n
            elif isinstance(n,ast.Assign):
                for t in n.targets:
                    if isinstance(t,ast.Name):result[t.id]=n
        return result
    og,ng=globals_by_name(base),globals_by_name(source)
    allowed={'DEFAULT_POLICIES','OptimizedDecoder','corrected_projection_flat_prefill','corrected_projection_chunk_add_prefill'}
    for name,n in og.items():
        if name not in allowed:assert dump(n)==dump(ng[name]),name
    # Both copied helper functions and every embedded kernel string are exact.
    for filename in ('narrow_gdn_combine.py','prefill_unflatten_bf16_v2.py'):
        for name,n in globals_by_name((D/filename).read_text()).items():assert dump(n)==dump(ng[name]),name
    incoming=globals_by_name((D/'narrow_gdn_fused_combine.py').read_text())
    assert dump(ng['corrected_projection_flat_prefill'])==dump(incoming['corrected_projection_flat_prefill'])
    chunk=copy.deepcopy(ng['corrected_projection_chunk_add_prefill'])
    guards=[n for n in chunk.body if isinstance(n,ast.If) and 'prefill_unflatten_packed' in ast.unparse(n.test)]
    assert len(guards)==1
    unflatten=defs((D/'prefill_unflatten_v2.py').read_text())['corrected_projection_chunk_add_prefill']
    original_guard=next(n for n in unflatten.body if isinstance(n,ast.If) and 'prefill_unflatten_packed' in ast.unparse(n.test))
    assert dump(guards[0])==dump(original_guard)
    chunk.body.remove(guards[0]);assert dump(chunk)==dump(incoming['corrected_projection_chunk_add_prefill'])
    defaults=copy.deepcopy(ast.literal_eval(og['DEFAULT_POLICIES'].value))
    for kind,flags in generator['ADDED_DEFAULTS'].items():defaults[kind].update(flags)
    assert defaults==ast.literal_eval(ng['DEFAULT_POLICIES'].value)
    # Execute only pure dispatch helpers to test prefill/decode and capacity.
    selected=[copy.deepcopy(new[n]) for n in ('_attention_chunk','_decode_attention_chunk')]
    cls=ast.ClassDef(name='Dispatch',bases=[],keywords=[],body=selected,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]));ns={};exec(compile(module,'<isolated dispatch>','exec'),ns)
    obj=ns['Dispatch']();cases=0
    for pages in (1,15,16,32,48,64,80,128,4096):
      state=types.SimpleNamespace(page_table=types.SimpleNamespace(shape=(32,pages)))
      for k in (32,64,128,256,512):
       obj.policy={'short_capacity_decode_k_chunk':k}
       for decode in (False,True):
        baseline=32 if pages<16 else 512
        expected_chunk=k if decode and baseline==512 and pages<=64 else baseline
        assert obj._decode_attention_chunk(state,decode=decode)==expected_chunk;cases+=1
    # Re-run standalone address proofs without their JSON-writing main blocks.
    tests=D.parents[2]/'tests'
    narrow=runpy.run_path(str(tests/'test_narrow_gdn_source.py'))['run_checks']()
    unflatten=runpy.run_path(str(tests/'test_prefill_unflatten_v2_source.py'))['run_checks']()
    return {'result':'pass','recurrence_exact_to_persistence134':True,'changed_methods':sorted(changed),
            'kernel_strings_exact_to_selected_sources':True,'overlap_resolution_exact':True,
            'dispatch_cases':cases,'narrow_address_cases':narrow['address_shape_cases'],
            'unflatten_offset_valid_cases':unflatten['offset_valid_cases'],
            'hardware_validation':'pending cumulative device signoff'}

if __name__=='__main__':
    result=run_checks();(D.parent/'integrated162_cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
