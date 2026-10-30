# SPDX-License-Identifier: Apache-2.0
"""Check selected policy against actual measured construction/kernel arguments."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'doc/datatype_sweep'
cfg=json.loads((root/'selected_precision_config.json').read_text())
checks=[]
def dtype(name):return 'DataType.'+name.upper()
def fidelity(name):return 'MathFidelity.'+name
for batch in (1,8,16,32):
    path=root/'selected'/f'benchmark_b{batch}_perf.json'
    data=json.loads(path.read_text());assert data['precision_config']==cfg
    runtime=data['precision_runtime'];seen=set()
    assert runtime['decode_ccl_dtype']==dtype(cfg['ccl_dtype']['decode_b1'])
    assert runtime['prefill_ccl_dtypes']==[dtype(cfg['ccl_dtype']['prefill'])]
    if batch>=8:assert runtime['native_ccl_dtypes']==[dtype(cfg['ccl_dtype']['native_batched'])]
    for row in runtime['compute']:
        for field in ('norm','sdpa','fused_prefill'):
            assert row[field]==fidelity(cfg['compute_fidelities'][field])
        assert row['prefill_sdpa']==fidelity(cfg['compute_fidelities']['norm'])
    assert runtime['sampling_dtypes']==dict(p=dtype('bfloat16'),temp=dtype('bfloat16'),tokens=dtype('uint32'),output_history=dtype('uint32'))
    for row in data['cache_inventory']:
        expected=cfg['kv_cache_dtype'] if row['field'] in ('key','value') else ('float32' if row['field']=='recurrent' else 'bfloat16')
        assert row['dtype']==dtype(expected)
    for row in data['weight_inventory']['tensors']:
        name=row['name']
        if name in ('head','embedding','final_norm'):
            group='norm' if name=='final_norm' else name
            assert row['dtype']==dtype(cfg['weight_groups'][group])
        elif name.startswith('layers.'):
            _,index,role,*_=name.split('.')
            groups={'mix':'attention','out':'attention','gate':'mlp','up':'mlp','gateup':'mlp','down':'down'}
            if role in groups:
                group=groups[role]
                expected=cfg['layer_exceptions'].get(index,{}).get(group+'_dtype',cfg['weight_groups'][group])
            elif role in ('l2_weight','neg_A','dt','gdn_constants'):
                expected='float32'  # Fixed recurrence constants, documented contract.
            else:
                assert role in ('in_norm','post_norm','out_norm','q_norm','k_norm','norm_shards','taps'),name
                expected=cfg['weight_groups']['norm'] if role!='taps' else 'bfloat16'
            assert row['dtype']==dtype(expected),(name,row['dtype'],expected)
        elif name in ('cos_table','sin_table'):
            assert row['dtype']==dtype('bfloat16')
    for row in data['request_buffers']['tensors']:
        if row['name'] in ('logits','prefill_output'):
            assert row['dtype']==dtype(cfg['logits_sampling']['logits'])
    for row in data['prepared_kernel_precision_calls']:
        k,n=row['weight_shape'][-2:]
        group='head' if (k,n)==(5120,62080) else ('down' if k==4352 and n==5120 else ('mlp' if k==5120 and n in (4352,8704) else 'attention'))
        assert row['weight_dtype']==dtype(cfg['weight_groups'][group]),(group,row)
        if row['op']=='matmul_reduce_scatter_async':
            phase='fused_prefill' if row['input_shape'][-2]==4096 else 'fused_decode'
            assert row['input_dtype']==dtype(cfg['activation_dtype']['fused_row'])
        else:
            phase='head' if group=='head' else 'projection'
            if group=='head':allowed={dtype(cfg['residual_dtype'])}
            else:
                ordinary='ordinary_mlp' if group in ('mlp','down') else 'ordinary_attention'
                allowed={dtype(cfg['activation_dtype'][ordinary])}
                if group=='mlp' or (group=='attention' and k==5120):
                    allowed.add(dtype(cfg['activation_dtype']['prefill_column']))
            assert row['input_dtype'] in allowed,(group,row)
        assert row['math_fidelity']==fidelity(cfg['compute_fidelities'][phase]),(phase,row)
        assert row['fp32_dest_acc_en']==(group=='head')
        seen.add((group,phase))
    assert {('head','head'),('attention','projection'),('mlp','projection'),('down','projection')}<=seen
    if batch>=8:assert {('attention','fused_decode'),('down','fused_decode')}<=seen
    checks.append(dict(batch=batch,status='pass',observed_group_phases=sorted([list(x) for x in seen]),evidence=str(path.relative_to(root))))
normal=json.loads((root/'selected_validation/precision_runtime.json').read_text())
assert normal['config']==cfg and normal['prepared_kernel_precision_calls']
report=dict(status='pass',selected_config=cfg['config_id'],checks=checks,
    normal_default_construction='run_full_validation build_generator without precision override; matching artifact in selected_validation/precision_runtime.json',
    scope='Actual weight/cache/sampler/workspace dtypes and ordinary/fused kernel fidelity/input arguments. Fixed BF16 residual/norm and FP32 recurrence are also enforced constructor invariants.',
    later_adapter_contract='No vLLM integration in stage08; later adapter must use the same default build_generator/Model loader.')
(root/'precision_propagation_check.json').write_text(json.dumps(report,indent=2)+'\n')
print('Selected precision propagation verified for batches1/8/16/32')
