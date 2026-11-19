"""Exhaustive face-segment byte mapping, relative NoC alignment and scope checks."""
import ast,copy,hashlib,json,runpy
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'

def run_checks():
    cases=0;max_reads=0;max_zeros=0
    nt=516;column=7
    for offset in range(32):
      for valid in range(1,33):
        shift=(offset&1)*32
        slot=np.full(1024,-1,dtype=np.int64);coverage=np.zeros(1024,dtype=np.int64)
        reads=[];zeros=[]
        if offset==0 and valid==32:
            reads=[(column,0,0,2048)]
        else:
            row=0
            while row<valid:
                sr=offset+row;n=min(16-row%16,16-sr%16,valid-row)
                for face in range(2):
                    src=(sr%32//16)*1024+face*512+(sr%16)*32
                    dst=shift+(row//16)*1024+face*512+(row%16)*32
                    reads.append((column+(sr//32)*nt,src,dst,n*32))
                row+=n
            row=valid
            while row<32:
                n=16-row%16
                for face in range(2):zeros.append((shift+(row//16)*1024+face*512+(row%16)*32,n*32))
                row+=n
        for page,src,dst,length in reads:
            assert src%64==dst%64
            assert 0<=src<src+length<=2048 and 0<=dst<dst+length<=4096
            assert length>0 and length%32==0
            slot[dst//4:(dst+length)//4]=page*512+np.arange(src//4,(src+length)//4)
            coverage[dst//4:(dst+length)//4]+=1
        for dst,length in zeros:
            assert dst%16==0 and 0<=dst<dst+length<=4096
            slot[dst//4:(dst+length)//4]=0;coverage[dst//4:(dst+length)//4]+=1
        expected=np.zeros(512,dtype=np.int64)
        for row in range(valid):
            sr=offset+row
            for face in range(2):
                di=(row//16)*256+face*128+(row%16)*8
                si=(sr%32//16)*256+face*128+(sr%16)*8
                expected[di:di+8]=(column+(sr//32)*nt)*512+si+np.arange(8)
        assert np.array_equal(slot[shift//4:shift//4+512],expected)
        assert np.all(coverage[shift//4:shift//4+512]==1)
        assert coverage.sum()==512 and shift%16==0
        max_reads=max(max_reads,len(reads));max_zeros=max(max_zeros,len(zeros));cases+=1
    source=(D/'prefill_unflatten_v2.py').read_text()
    assert runpy.run_path(str(D/'make_prefill_unflatten_v2.py'))['build']()==source
    record=json.loads((D/'prefill_unflatten_v2.generation.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==record['output_sha256']
    for n,h in record['sources'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h
    old,new=ast.parse((D/'integrated134.py').read_text()),ast.parse(source)
    dump=lambda x:ast.dump(x,include_attributes=False)
    for a,b in zip(old.body,new.body):
        if isinstance(a,ast.FunctionDef) and a.name=='corrected_projection_chunk_add_prefill':
            modified=copy.deepcopy(b)
            modified.body=[n for n in modified.body if not (isinstance(n,ast.If) and 'prefill_unflatten_packed' in ast.unparse(n.test))]
            assert dump(a)==dump(modified)
        else:assert dump(a)==dump(b)
    helper=ast.parse((D/'prefill_unflatten_bf16_v2.py').read_text())
    reader=next(n.value.value for n in helper.body if isinstance(n,ast.Assign) and n.targets[0].id=='_PREFILL_UNFLATTEN_READER')
    assert 'volatile' not in reader and 'scratch' not in reader
    assert max_reads==8 and max_zeros==4
    return {'result':'pass','offset_valid_cases':cases,'max_DRAM_read_segments':max_reads,'max_zero_segments':max_zeros,
            'exact_words_and_zero_padding':True,'all_relative_NoC_alignments_legal':True,'destination_coverage_exactly_once':True,
            'only_final_prefill_movement_changes':True,'hardware_validation':'pending'}

if __name__=='__main__':
    r=run_checks();(D.parent/'prefill_unflatten_v2_cpu_checks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
