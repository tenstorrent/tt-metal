"""CPU-only shape/config proofs for exact residual-key prefill tuning."""
import ast,hashlib,json,math,runpy
from pathlib import Path
D=Path(__file__).resolve().parents[1]/'doc/optimized_decoder/candidates'


def geometry(rows,n,options):
    gx,gy=options.get('grid',(11,10));pm=math.ceil(rows/32/gy);pn=math.ceil(n/32/gx)
    sw=options.get('subblock_w',max(v for v in (1,2,4) if pn%v==0))
    requested_bh=options.get('out_block_h',pm)
    bh=max(v for v in range(1,pm+1) if v<=requested_bh and pm%v==0)
    requested_h=min(bh,options.get('subblock_h',4))
    sh=max(v for v in (1,2,4) if v<=requested_h and bh%v==0 and v*sw<=4)
    kb=options.get('in0_block_w',8)
    assert 160%kb==0 and pm%bh==0 and bh%sh==0 and pn%sw==0 and sh*sw<=4
    return dict(pm=pm,pn=pn,bh=bh,sh=sh,sw=sw,kb=kb,
                main_cb_bytes=2*bh*kb*2048+2*pn*kb*1088+bh*pn*4096)


def run_checks():
    source=(D/'prefill_split_role_configs.py').read_text();base=(D/'integrated162.py').read_text()
    assert runpy.run_path(str(D/'make_prefill_split_role_configs.py'))['build']()==source
    record=json.loads((D/'prefill_split_role_configs.generation.json').read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==record['output_sha256']
    for n,h in record['sources'].items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h
    def methods(s):return {n.name:n for c in ast.parse(s).body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
    a,b=methods(base),methods(source);dump=lambda n:ast.dump(n,include_attributes=False)
    assert [n for n in a if dump(a[n])!=dump(b[n])]==['_linear_raw']
    # All other top-level definitions/constants, including recurrence kernels,
    # default policy and state representations, are exactly unchanged.
    for x,y in zip(ast.parse(base).body,ast.parse(source).body):
        if not isinstance(x,ast.ClassDef):assert dump(x)==dump(y)
    cases=0
    variants={'residual_sw3':{'subblock_w':3},'residual_k10_sw3':{'subblock_w':3,'in0_block_w':10},
              'residual_k16_bh1_sw3':{'subblock_w':3,'in0_block_w':16,'out_block_h':1}}
    for rows in range(1,641):
      for n in (16640,11520):
        got=geometry(rows,n,{})
        pm=math.ceil(rows/32/10);pn=math.ceil(n/32/11);sw=max(v for v in (1,2,4) if pn%v==0)
        sh=max(v for v in (1,2,4) if v<=4 and pm%v==0 and v*sw<=4)
        assert (got['pm'],got['pn'],got['bh'],got['sh'],got['sw'],got['kb'])==(pm,pn,pm,sh,sw,8);cases+=1
      for options in variants.values():geometry(rows,11520,options);cases+=1
      geometry(rows,16640,{'in0_block_w':10});cases+=1
    configs={'packed_gdn':{'in0_block_w':10},'packed_gdn_residual':{'subblock_w':3,'in0_block_w':8}}
    select=lambda k:configs.get(k,configs.get(k.removesuffix('_residual'),{}))
    assert select('packed_gdn')['in0_block_w']==10 and select('packed_gdn_residual')['in0_block_w']==8
    budgets={name:geometry(640,11520,opt) for name,opt in variants.items()}
    budgets['primary_k10']=geometry(640,16640,{'in0_block_w':10})
    assert all(x['main_cb_bytes']<1572864 for x in budgets.values())
    return {'result':'pass','geometry_cases':cases,'defaults_identical_for_every_row_count_1_to_640':True,
            'only_linear_raw_config_selection_changed':True,'selected_640row_budgets':budgets,'hardware_validation':'pending'}

if __name__=='__main__':
    r=run_checks();(D.parent/'prefill_split_role_configs_cpu_checks.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
