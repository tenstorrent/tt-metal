# SPDX-License-Identifier: Apache-2.0
"""Compute compact stage rooflines using complete layer device windows."""
import json
from pathlib import Path
from .roofline import calculate
D = Path(__file__).resolve().parents[1] / 'doc/optimized_decoder'

def main():
    result = {}
    for layer, kind in [(0,'linear_attention'),(3,'full_attention')]:
        label=f'final_accepted_l{layer}'
        measured=json.loads((D/f'tracy/{label}/whole_layer_timing.json').read_text())['phases']
        run=json.loads((D/f'{label}.json').read_text())
        pre,dec=measured['prefill']['device_us'],measured['decode']['device_us']
        row=calculate(kind,1,pre,dec)
        peak=120*4096*1.35e9
        row.update(peak_flops_per_s=peak,prefill_flops_pct=100*row['prefill_useful_flops']/(peak*pre/1e6))
        # Physical final-policy DRAM weight sizes, including per-bank reader padding.
        mix=16896 if layer==0 else 14336
        weights=(5120*mix+6144*5120+5120*34816+17408*5120)/1024*576
        state=(2*48*128*128*4+2*3*10240*2) if layer==0 else (4224+1)*2176
        traffic=weights+state+2*5120*2
        row.update(prefill_compute_roofline_us=row["prefill_useful_flops"]/peak*1e6,
            decode_dram_roofline_us=traffic/512e9*1e6,
            prefill_host_us=run["prefill_host_us"], decode_host_us=run["workload_last_decode_host_us"],
            decode_dram_bytes=traffic,decode_dram_pct=100*traffic/(512e9*dec/1e6),
            peak_basis='One p300c Blackhole ASIC:120 physical cores,1.35GHz,4096 FLOP/cycle/core LoFi theoretical peak;512GB/s GDDR6. Runtime storage grid110 cores; matmul reader workers16. No multi-device aggregation.',
            traffic_assumptions='Estimated compulsory physical traffic: BFP4 weights including exponent headers and reader padding read once; BF16 residual input/output; FP32 recurrent and row-major BF16 convolution state read/write, or BFP8 prefix KV read plus token write. Excludes intermediate spills, repeated reads and small constants: lower bound, not bus counters.',
            flops_assumptions='Useful dense matmul work and logical causal attention/recurrence/scalar work; no MoE and no padding counted useful. Mixed BF16 activations, BFP4/LoFi projections, FP32/HiFi4 sensitive recurrence/attention: denominator is chip LoFi peak, not an achievable mixed-kernel ceiling.',
            source='single-layer Tracy; complete firmware timestamp envelope including all operations and internal gaps',
            evidence=[f'tracy/{label}/whole_layer_timing.json',f'tracy/{label}/prefill_perf_report.csv',f'tracy/{label}/decode_perf_report.csv',f'{label}.json','tests/report_optimized.py'],
            workload=run['performance_workload'])
        result[kind]=row
    (D/'performance.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
