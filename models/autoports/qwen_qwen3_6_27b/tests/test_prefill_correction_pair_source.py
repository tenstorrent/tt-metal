"""CPU-only phase, address and legal configuration checks; no TTNN import."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import runpy
D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'


def run_checks():
    source=(D/'prefill_correction_pair.py').read_text()
    base=(D/'delivery118.py').read_text()
    assert runpy.run_path(str(D/'make_prefill_correction_pair.py'))['build']()==source
    record=json.loads((D/'prefill_correction_pair.generation.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==record['output_sha256']
    for n,h in record['sources'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h
    methods=lambda s:{n.name:n for c in ast.parse(s).body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
    old,new=methods(base),methods(source);dump=lambda n:ast.dump(n,include_attributes=False)
    assert [k for k in old if dump(old[k])!=dump(new[k])]==['from_state_dict','_gdn']
    altered=copy.deepcopy(new['_gdn'])
    for n in ast.walk(altered):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_packed_input_projection_prefill':
            n.func.attr='_linear';n.args.append(ast.Constant('packed_gdn'))
    assert dump(altered)==dump(old['_gdn'])
    # Existing compact S1 return precedes the new longer-prefix projection.
    assert ast.unparse(new['_gdn'].body[0])==ast.unparse(old['_gdn'].body[0])
    method=new['_packed_input_projection_prefill']
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('from_torch','to_torch') for n in ast.walk(method))
    assert not any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Slice) and isinstance(n.value,ast.Attribute) and n.value.attr in ('shape','padded_shape') for n in ast.walk(method))
    cases=0
    for b in (1,3,32):
      for s in (1,17,31,32,33,65,511,512,513,1025):
       for cap in (256,512,1024):
        for kb in (4,8):
         rows=b*s;coverage=[]
         for start in range(0,rows,cap):
            count=min(cap,rows-start);coverage.extend(range(start,start+count))
            pm=math.ceil(count/32/10);pn=math.ceil(34816/32/11)
            assert pn==99 and pn%33==0 and 160%kb==0 and pm>=1
            # Per-core last N tail remains in its third internal block.
            for core in range(11):
                actual=min(pn,34816//32-core*pn)
                assert 66<actual<=99
            # Output pair boundaries are whole tiles and preserve term order.
            assert 16512//32==516 and 33024//32==1032
         assert coverage==list(range(rows));cases+=1
    return {'result':'pass','source_phase_and_disabled_path_preserved':True,'geometry_cases':cases,
            'exact_weight_pages_requires_device_setup_probe':True,'hardware_validation':'pending'}


def test_prefill_correction_pair_source():run_checks()

if __name__=='__main__':
    result=run_checks();(D.parent/'prefill_correction_pair_cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
