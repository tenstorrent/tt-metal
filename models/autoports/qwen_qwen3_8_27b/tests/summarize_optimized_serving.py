# SPDX-License-Identifier: Apache-2.0
"""Derive serving comparison values from the retained raw vLLM results."""
import json
from pathlib import Path
root=Path('models/autoports/qwen_qwen3_8_27b')
out=root/'doc/optimized_vllm'
fields=['median_ttft_ms','p99_ttft_ms','mean_tpot_ms','p99_tpot_ms',
        'median_itl_ms','p99_itl_ms','output_throughput']
comparisons={}
for name,before,after,shape,capacity in [
 ('primary_single_user','before_b1_vllm_result.json','selected_b1_vllm_result.json',(4096,128,1,1),1),
 ('ci_serving_burst','before_vllm_ci_serving_result.json','after_b32_vllm_ci_serving_result.json',(100,100,32,None),32),
 ('single_request_at_b32_capacity','before_full_b32_performance_vllm_result.json','after_b32_vllm_result.json',(4096,128,1,1),32)]:
    record={'workload':dict(zip(['input_tokens','output_tokens','requests','concurrency'],shape)),
            'max_num_seqs':capacity,'headline':name=='primary_single_user'}
    for phase,path in [('before',before),('after',after)]:
        raw=json.loads((out/path).read_text())
        assert raw['failed']==0 and raw['completed']==shape[2]
        assert raw['total_input_tokens']==shape[0]*shape[2]
        assert raw['total_output_tokens']==shape[1]*shape[2]
        assert raw['max_concurrency']==shape[3]
        record[phase]={key:raw[key] for key in fields}
        record[phase].update(decode_tps=1000/raw['mean_tpot_ms'],evidence=path)
    comparisons[name]=record
primary=comparisons['primary_single_user']['after']
control=json.loads((root/'doc/datatype_sweep/performance.json').read_text())['measurements'][0]
comparisons['full_model_comparison']={'control':control,
    'serving_decode_tps':primary['decode_tps'],
    'serving_to_full_model_decode_ratio':primary['decode_tps']/control['decode_tps'],
    'scope':'Same4096/128/B1 selected policy and mesh; different autoregressive/HTTP timing boundaries. No device-time claim.'}
(out/'comparisons.json').write_text(json.dumps(comparisons,indent=2)+'\n')
summary={'source':'vllm','workload':{'profile':'single_user_decode','prompt_len':4096,'gen_len':128,'batch':1,'concurrency':1},
         'ttft_ms':primary['median_ttft_ms'],'decode_ms_per_token_e2e':primary['mean_tpot_ms'],
         'decode_ms_per_token_device':None,'roofline_ms_per_token_estimate':None,
         'named_limitations':['vllm_serving_profiler_disabled_to_protect_hardware','one measured primary request; not a tail-latency population estimate'],
         'evidence':'selected_b1_vllm_result.json'}
(out/'perf_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(comparisons,indent=2))
