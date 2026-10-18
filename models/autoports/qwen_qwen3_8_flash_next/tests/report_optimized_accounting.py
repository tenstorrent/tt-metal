# SPDX-License-Identifier: Apache-2.0
"""Reconcile same-run host/device times with an explicit weight/cache bandwidth floor."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'doc/optimized_decoder'


def matrix_bytes(k,n,dtype):
    return math.ceil(k/32)*math.ceil(n/32)*dict(bf16=2048,bfp8=1088,bfp4=576)[dtype]


def weight_floor(layer,optimized,position,timed_replays,advancing):
    precision='bfp8' if optimized else 'bf16'
    expert='bfp4' if optimized else 'bfp8'
    groups={
      'hyperconnection_weights':2*(matrix_bytes(10240,352,precision)+matrix_bytes(320,10240,precision)),
      'router_shared_weights':matrix_bytes(2560,1824,'bf16')+matrix_bytes(640,2560,'bf16'),
      'ten_active_experts':10*3*matrix_bytes(2560,640,expert),
      'attention_weights':matrix_bytes(2560,13952 if layer==3 else 16512,precision)+matrix_bytes(6144,2560,precision),
    }
    if layer!=3:
        groups['recurrent_state_read_write']=2*48*128*128*4
        groups['convolution_history_read_write']=2*3*10240*2
    if layer==1:
        groups['ple_projection']=matrix_bytes(2560,12800,precision)
        groups['ple_embedding_tiles_read']=16*5*576
    if layer==3:
        first=position+2 if advancing else position
        average_position=first+(timed_replays-1)/2 if advancing else first
        groups['pooled_index_keys']=65536*128*2
        groups['selected_kv']=min(average_position+1,2051)*2048
    return groups


def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='final_');p.add_argument('--optimized-prefix');args=p.parse_args()
    result=[]
    for impl in ('fused','optimized'):
        for layer in (0,1,3):
            run_prefix=args.optimized_prefix if impl=='optimized' and args.optimized_prefix else args.prefix
            tag=run_prefix+f'{impl}_l{layer}'
            prof=EVIDENCE/'tracy'/tag
            provenance=json.loads((prof/'provenance.json').read_text())
            timing=json.loads((EVIDENCE/(run_prefix+f'profile_{impl}_l{layer}')/'timing.json').read_text())
            windows=provenance['windows']
            floor=weight_floor(layer,impl=='optimized',timing.get('position_after_capture',timing['position']),timing['decode_replays'],timing.get('advance_position',False))
            rows=list(csv.DictReader((prof/'decode_perf_report.csv').open()))
            unique={}
            for row in rows:
                if 'Matmul' not in row['OP Code'] and 'Expert' not in row['OP Code']:continue
                key=(row['OP Code'],row['Math Fidelity'],row['Input 0 Datatype'],row['Input 1 Datatype'])
                if key not in unique:
                    unique[key]={k:row[k] for k in ('OP Code','Math Fidelity','Input 0 Datatype','Input 1 Datatype','DRAM Sharded','Input 0 Memory','Inner Dim Block Size','Output Subblock H','Output Subblock W')}
            item=dict(implementation=impl,layer=layer,source_sha256=timing['source_sha256'],
                benchmark=str((EVIDENCE/(run_prefix+f'profile_{impl}_l{layer}')/'timing.json').relative_to(ROOT)),
                profiler=str((prof/'provenance.json').relative_to(ROOT)),
                prefill_host_ms=timing['median_prefill_host_ms'],
                prefill_device_ms=windows['prefill']['per_pass_device_us']/1000,
                decode_host_ms=timing['median_traced_decode_ms'],
                decode_device_ms=windows['decode']['per_pass_device_us']/1000,
                decode_device_plus_gap_ms=windows['decode']['per_pass_device_plus_gap_us']/1000,
                dram_bandwidth_gb_s=512,required_weight_cache_payload_bytes=floor,
                weight_cache_bandwidth_floor_ms=sum(floor.values())/512e6,
                decode_programs=windows['decode']['window_validation']['operations_per_replay'],
                matmul_rows=list(unique.values()),native_expert_runtime_arguments=timing.get('native_expert_runtime_arguments'))
            if item['decode_device_ms']>item['decode_host_ms']*1.2:
                raise RuntimeError(f'Implausible device/host accounting for {tag}')
            result.append(item)
    report=dict(scope='Single-device decoder; same signposted run for host/device timing.',
      bandwidth_source='tt-perf-report1.4.1 ArchitectureSpec blackhole:512GB/s',
      floor_scope='Required logical weight/cache payload at stored tile dtypes; a lower bound, excluding DRAM shard padding, repeated reads, additional recurrent-state routing traffic, intermediate traffic and dispatch. Sparse experts count the ten active experts, not all512 stored experts.',
      native_fidelity_source='ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_program_factory.cpp:1161; hardcoded LoFi, fixed11x8 grid; runtime argument dtypes recorded by benchmark wrapper.',
      runs=result)
    (EVIDENCE/(args.prefix+'performance_accounting.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('implementation','layer','decode_host_ms','decode_device_ms','decode_device_plus_gap_ms','weight_cache_bandwidth_floor_ms','decode_programs')} for r in result],indent=2))

if __name__=='__main__':main()
