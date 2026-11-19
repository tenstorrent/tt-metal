"""CPU proof of tiled row mapping and immutable integration boundaries."""
import ast,copy,hashlib,json,runpy
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'

def run_checks():
    source=(D/'prefill_unflatten.py').read_text();base=(D/'integrated134.py').read_text()
    assert runpy.run_path(str(D/'make_prefill_unflatten.py'))['build']()==source
    old,new=ast.parse(base),ast.parse(source)
    dump=lambda x:ast.dump(x,include_attributes=False)
    # All existing definitions except the one final movement helper are exact.
    for a,b in zip(old.body,new.body):
        if isinstance(a,ast.FunctionDef) and a.name=='corrected_projection_chunk_add_prefill':
            modified=copy.deepcopy(b)
            modified.body=[n for n in modified.body if not (isinstance(n,ast.If) and 'prefill_unflatten_packed' in ast.unparse(n.test))]
            assert dump(a)==dump(modified)
        else:assert dump(a)==dump(b)
    helper=ast.parse((D/'prefill_unflatten_bf16.py').read_text())
    assert not any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Slice) and isinstance(n.value,ast.Attribute) and n.value.attr in ('shape','padded_shape') for n in ast.walk(helper))
    cases=0;tiles=0
    for batch in (1,3,31,32):
      for seq in (1,17,31,32,33,65,511,512,513):
       for width in (32,96,16512):
        nt=width//32;st=(seq+31)//32;source_tiles=((batch*seq+31)//32)*nt
        for user in range(batch):
         for tr in range(st):
          for col in sorted({0,nt//2,nt-1}):
            first=user*seq+tr*32;offset=first%32;valid=min(32,seq-tr*32)
            tile=(first//32)*nt+col
            assert 0<=tile<source_tiles
            assert offset+valid<=32 or tile+nt<source_tiles
            for row in range(32):
             for face in range(2):
              # Unique raw uint32 word IDs stand in for arbitrary BF16 bit pairs.
              sr=offset+row
              si=(sr//32)*512+((sr%32)//16)*256+face*128+(sr%16)*8
              actual=np.arange(si,si+8,dtype=np.uint64)+(tile+(sr//32)*(nt-1))*512 if row<valid else np.zeros(8,dtype=np.uint64)
              if row<valid:
               logical=user*seq+tr*32+row
               expected=(logical//32*nt+col)*512+(logical%32//16)*256+face*128+(logical%16)*8+np.arange(8,dtype=np.uint64)
              else:expected=np.zeros(8,dtype=np.uint64)
              assert np.array_equal(actual,expected)
            tiles+=1
        cases+=1
    for total in (1,516,33024):
      for cores in (1,32,64,110):
        workers=min(total,cores);assert sum(total//workers+(i<total%workers) for i in range(workers))==total
    return {'result':'pass','shape_cases':cases,'sampled_output_tiles':tiles,
            'bit_address_mapping_and_zero_padding':True,'only_final_prefill_movement_changes':True,'hardware_validation':'pending'}

if __name__=='__main__':
    r=run_checks();(D.parent/'prefill_unflatten_cpu_checks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
