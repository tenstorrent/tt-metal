"""CPU topology/layout checks for narrow correction and its optional fused combine."""
import ast,hashlib,json,math,runpy
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'

def run_checks():
    for candidate,generator in [('narrow_gdn_correction','make_narrow_gdn_correction'),('narrow_gdn_fused_combine','make_narrow_gdn_fused_combine')]:
        source=(D/(candidate+'.py')).read_text()
        assert runpy.run_path(str(D/(generator+'.py')))['build']()==source
        record=json.loads((D/(candidate+'.generation.json')).read_text())
        assert hashlib.sha256(source.encode()).hexdigest()==record['output_sha256']
        for name,h in record['sources'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==h
    base=(D/'zero_z_correction_on134.py').read_text();narrow=(D/'narrow_gdn_correction.py').read_text()
    definitions=lambda s:{n.name:n for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    a,b=definitions(base),definitions(narrow);dump=lambda n:ast.dump(n,include_attributes=False)
    assert {k for k in a if dump(a[k])!=dump(b[k])}=={'OptimizedDecoder','corrected_projection_flat_prefill','corrected_projection_chunk_add_prefill'}
    methods=lambda n:{x.name:x for x in n.body if isinstance(x,ast.FunctionDef)}
    am,bm=methods(a['OptimizedDecoder']),methods(b['OptimizedDecoder'])
    assert {k for k in am if dump(am[k])!=dump(bm[k])}=={'from_state_dict','_linear'}
    # Fused variant adds only an opt-in branch inside the already gated combine.
    f=definitions((D/'narrow_gdn_fused_combine.py').read_text())
    fm=methods(f['OptimizedDecoder'])
    assert {k for k in bm if dump(bm[k])!=dump(fm[k])}=={'_combine_correction'}
    # BFP8 physical N derives from the unchanged C40/R5 and eight DRAM banks.
    alignment=math.lcm(8*5,40)
    primary=math.ceil(16512/32/alignment)*32*alignment
    residual=math.ceil(10368/32/alignment)*32*alignment
    assert (primary,residual)==(16640,11520)
    assert (primary-residual)//32*160*1088==27852800
    # Native tiled BFP8 slices preserve whole tiles, including their exponents.
    tags=np.arange(160*516).reshape(160,516)
    packed=np.concatenate([tags[:,:320],tags[:,512:516]],axis=1)
    assert packed.shape==(160,324) and np.array_equal(packed[:,320:],tags[:,512:])
    cases=0
    for batch in (1,3,32):
      for seq in (1,17,32,33,65,641):
       for pn,rn in ((516,324),(520,360)):
        mtiles=batch*((seq+31)//32)
        for row in (0,mtiles-1):
         for col in range(516):
            ppage=row*pn+col
            assert 0<=ppage<mtiles*pn
            if col<320 or col>=512:
                rc=col if col<320 else col-192
                rpage=row*rn+rc
                assert 0<=rpage<mtiles*rn
                assert (rc if rc<320 else rc+192)==col
            else:assert 320<=col<512
        cases+=1
    helper=(D/'narrow_gdn_combine.py').read_text()
    assert 'UnpackToDestFp32' in helper and 'for i in (0,1,3)' in helper
    assert 'add_binary_tile(0,1,0)' in helper and 'Float16_b)>(0)' in helper
    tree=ast.parse(helper)
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('from_torch','to_torch','to_memory_config','slice','concat') for n in ast.walk(function))
    return {'result':'pass','address_shape_cases':cases,'weight_bytes_saved':27852800,
            'primary_physical_N':primary,'residual_physical_N':residual,
            'public_phase_methods_and_recurrence_unchanged':True,'hardware_validation':'pending same-partial exactness and changed-N projection attribution'}

if __name__=='__main__':
    r=run_checks();(D.parent/'narrow_gdn_cpu_checks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
