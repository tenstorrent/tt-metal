"""Recompute full-context precision capacity from the inherited physical ledger."""
import argparse
import json
from pathlib import Path
from models.autoports.qwen_qwen3_8_flash_next.tests.full_model_capacity import full_plan
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_memory_plan import tiled
ROOT=Path('models/autoports/qwen_qwen3_8_flash_next')
BYTES={'bfloat16':2048,'bfloat8_b':1088,'bfloat4_b':576}

def precision_plan(policy):
    if policy['kv_cache_dtype'] not in ('fp8_e4m3', 'bfloat16'):
        raise ValueError('Capacity ledger covers constructible packed KV formats only')
    if any(policy['weight_groups'][key] != 'bfloat4_b' for key in ('routed_expert_gate_up', 'routed_expert_down')):
        raise ValueError('Use the separate expert_bfp8_capacity_blocker ledger for unsupported expert storage')
    p=full_plan(policy['max_context'])
    delta=0
    def difference(shape,new,old):
        return tiled(shape,BYTES[new])-tiled(shape,BYTES[old])
    details={}
    for layer in range(48):
        w=dict(policy['weight_groups']);w.update(policy['layer_exceptions'].get(str(layer),{}).get('weight_groups',{}))
        gdn=(layer+1)%4!=0
        d=0
        if gdn:
            d+=difference((2560,4160),w['gdn_input'],'bfloat8_b')
            d+=difference((1536,2560),w['gdn_output'],'bfloat8_b')
            d+=difference((6144,640),w['gdn_output'],'bfloat8_b')
        else:
            d+=difference((2560,4224),w['qsa_prefill'],'bfloat8_b')
            d+=difference((2560,4224),w['qsa_decode'],'bfloat4_b')
            d+=difference((1536,2560),w['qsa_output'],'bfloat8_b')
            d+=difference((6144,640),w['qsa_output'],'bfloat8_b')
        d+=2*(difference((2560,352),w['hc'],'bfloat8_b')+difference((320,2560),w['hc'],'bfloat8_b'))
        d+=difference((2560,320),w['shared_expert'],'bfloat16')+difference((160,2560),w['shared_expert'],'bfloat16')
        if layer==1:d+=difference((2560,3200),w['ple_projection'],'bfloat8_b')
        details[str(layer)]=d;delta+=d
    delta+=difference((2560,62080),policy['weight_groups']['lm_head'],'bfloat16')
    cache_delta=0 if policy['kv_cache_dtype']=='fp8_e4m3' else 12*(policy['max_context']+512)*512
    p['weight_breakdown']['precision_policy_weight_adjustment']=delta
    p['cache_breakdown']['precision_policy_kv_adjustment']=cache_delta
    p.update(config_id=policy['config_id'],precision_policy=policy,layer_weight_delta_bytes=details,
             weight_delta_bytes=delta,kv_delta_bytes=cache_delta,
             weight_bytes_per_device=p['weight_bytes_per_device']+delta,
             cache_bytes_per_device=p['cache_bytes_per_device']+cache_delta,
             total_bytes_per_device=p['total_bytes_per_device']+delta+cache_delta,
             remaining_bytes_per_device=p['remaining_bytes_per_device']-delta-cache_delta)
    assert sum(p['weight_breakdown'].values()) == p['weight_bytes_per_device']
    assert sum(p['cache_breakdown'].values()) == p['cache_bytes_per_device']
    p['caveat']='Payload arithmetic preserves full262144 checkpoint context; confirm bank fragmentation and transient/program usage with prepared allocator evidence. Mixed/BF8 CCL transient variation is covered by reserve, not asserted identical.'
    return p

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('config');ap.add_argument('--runtime');ap.add_argument('--select',action='store_true');a=ap.parse_args()
    policy=json.loads(Path(a.config).read_text());p=precision_plan(policy)
    if a.runtime:
        runtime=json.loads(Path(a.runtime).read_text());m=runtime['prepared_memory'];t=runtime['trace_memory']
        p.update(runtime_artifact=a.runtime,actual_prepared=m,actual_trace=t,
                 actual_total_including_reserved_trace_bytes=m['num_banks']*(m['total_bytes_allocated_per_bank']+t['total_bytes_per_bank']),
                 actual_free_ordinary_dram_bytes=m['num_banks']*m['total_bytes_free_per_bank'])
    dest=ROOT/'doc/datatype_sweep'/f"capacity_{policy['config_id']}.json"
    dest.write_text(json.dumps(p,indent=2)+'\n')
    if a.runtime:
        path=ROOT/'doc/context_contract.json';contract=json.loads(path.read_text())
        contract.setdefault('datatype_sweep_capacity_candidates', {})[policy['config_id']] = dict(
            kv_cache_dtype=policy['kv_cache_dtype'], supported_context=policy['max_context'],
            capability_reduction=None, memory_plan=str(dest.relative_to(ROOT/'doc')),
            runtime_artifact=a.runtime, runtime_status=runtime['status'],
            planned_remaining_bytes_per_device=p['remaining_bytes_per_device'],
            actual_free_ordinary_dram_bytes=p['actual_free_ordinary_dram_bytes'])
    if a.select:
        assert a.runtime and p['actual_free_ordinary_dram_bytes']>0
        contract['datatype_sweep_validation']=dict(status='in_progress',supported_context=262144,hf_advertised_context=262144,
            capability_reduction=None,kv_cache_dtype=policy['kv_cache_dtype'],config_id=policy['config_id'],
            memory_plan=str(dest.relative_to(ROOT/'doc')),runtime_artifact=a.runtime,
            actual_free_ordinary_dram_bytes=p['actual_free_ordinary_dram_bytes'])
    if a.runtime:
        path.write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps({k:p[k] for k in ('config_id','context','total_bytes_per_device','remaining_bytes_per_device')}))
