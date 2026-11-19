"""CPU address/phase checks for fused post-RMSNorm affine, repack and gate."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np

D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'


def tiles(x):
    rows,cols=x.shape
    assert rows%32==cols%32==0
    return x.reshape(rows//32,32,cols//32,32).transpose(0,2,1,3).reshape(-1,2,16,2,16).transpose(0,1,3,2,4).reshape(-1,1024).copy()


def untiles(x,rows,cols):
    return x.reshape(-1,2,2,16,16).transpose(0,1,3,2,4).reshape(rows//32,cols//32,32,32).transpose(0,2,1,3).reshape(rows,cols)


def simulate_reader(batch,width):
    norm=np.arange(((batch*48+31)//32)*32*128,dtype=np.int64).reshape(-1,128)
    weight=np.arange(128,dtype=np.int64).reshape(1,128)+1000000
    wp=np.full((32,128),-99,dtype=np.int64);wp[:1]=weight
    packed=np.arange(32*width,dtype=np.int64).reshape(32,width)+2000000
    packed[batch:]=-777  # Deliberately poison invalid users.
    nt,wt,pt=tiles(norm),tiles(wp),tiles(packed)
    norm_out=np.empty((192,1024),dtype=np.int64)
    weight_out=np.empty_like(norm_out);gate_out=np.empty_like(norm_out)
    for tile in range(192):
        head,column=tile//4,tile%4
        scratch=np.full(4224//2,-999,dtype=np.int64)
        for user in range(batch):
            source_head=user*48+head;row=source_head%32
            source_tile=(source_head//32)*4+column
            for face in range(2):
                source=((row//16)*512+face*256+(row%16)*16)*2
                start=(source&~63)//2
                scratch[user*64+face*32:user*64+face*32+32]=nt[source_tile,start:start+32]
        for face in range(2):
            scratch[2048+face*32:2048+face*32+32]=wt[column,face*256:face*256+32]
        gate_out[tile]=pt[320+tile]
        for user in range(32):
            for face in range(2):
                dst=(user//16)*512+face*256+(user%16)*16
                row=(user*48+head)%32
                src=user*64+face*32+(row%2)*16
                norm_out[tile,dst:dst+16]=scratch[src:src+16] if user<batch else 0
                weight_out[tile,dst:dst+16]=scratch[2048+face*32:2048+face*32+16]
                if user>=batch:gate_out[tile,dst:dst+16]=0
    norm_actual=untiles(norm_out,32,6144)
    weight_actual=untiles(weight_out,32,6144)
    gate_actual=untiles(gate_out,32,6144)
    assert np.array_equal(norm_actual[:batch],norm[:batch*48].reshape(batch,6144))
    assert np.array_equal(weight_actual,np.tile(weight,(32,48)))
    assert np.array_equal(gate_actual[:batch],packed[:batch,10240:16384])
    assert not np.any(norm_actual[batch:]) and not np.any(gate_actual[batch:])


def run_checks():
    for batch in range(1,33):
        for width in (16512,17408):simulate_reader(batch,width)
    for workers in (1,8,32,64,110):
        start=0;covered=[]
        for i in range(workers):
            count=192//workers+(i<192%workers)
            covered.extend(range(start,start+count));start+=count
        assert covered==list(range(192))
    source=(D/'gdn_affine_gate_decode_v2.py').read_text();before=(D/'delivery118.py').read_text()
    record=json.loads((D/'gdn_affine_gate_decode_v2.generation.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==record['output_sha256']
    for name,digest in record['sources'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==digest
    assert runpy.run_path(str(D/'make_gdn_affine_gate_decode_v2.py'))['build']()==source
    methods=lambda s:{n.name:n for cls in ast.parse(s).body if isinstance(cls,ast.ClassDef)
                      for n in cls.body if isinstance(n,ast.FunctionDef)}
    old,new=methods(before),methods(source);dump=lambda n:ast.dump(n,include_attributes=False)
    assert [name for name in old if dump(old[name])!=dump(new[name])]==['_gdn_compact']
    class RemoveCandidate(ast.NodeTransformer):
        def visit_If(self,node):
            if 'gdn_fused_affine_gate' in ast.unparse(node.test):
                assert isinstance(node.test,ast.BoolOp) and isinstance(node.test.values[0],ast.Name)
                assert node.test.values[0].id=='decode'
                return None
            return self.generic_visit(node)
    assert dump(RemoveCandidate().visit(copy.deepcopy(new['_gdn_compact'])))==dump(old['_gdn_compact'])
    helper=ast.parse((D/'gdn_affine_gate_flat_v2.py').read_text())
    assert not any(isinstance(n,ast.Subscript) and isinstance(n.slice,ast.Slice) and isinstance(n.value,ast.Attribute) and n.value.attr in ('shape','padded_shape') for n in ast.walk(helper))
    body=next(n for n in helper.body if isinstance(n,ast.FunctionDef) and n.name=='gdn_affine_gate_flat')
    assert not any(isinstance(n,ast.Attribute) and n.attr in ('from_torch','to_torch','rms_norm') for n in ast.walk(body))
    compute=next(n.value.value for n in helper.body if isinstance(n,ast.Assign)
                 and any(isinstance(t,ast.Name) and t.id=='_GDN_AFFINE_GATE_COMPUTE' for t in n.targets))
    assert compute.count('typecast_tile<static_cast')==2
    assert 'silu_tile<true>(2)' in compute
    return {'result':'pass','hardware_validation':'pending own-input probe',
        'mapping_cases':64,'all_batches_1_through_32':True,'poisoned_padding_zeroed':True,
        'partition_cases':5,'native_RMSNorm_and_recurrence_unchanged':True,
        'prefill_and_disabled_path_AST_unchanged':True,'two_explicit_BF16_rounding_boundaries':True}


def test_gdn_affine_gate_mapping():run_checks()


if __name__=='__main__':
    result=run_checks()
    (D.parent/'gdn_affine_gate_v2_cpu_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
