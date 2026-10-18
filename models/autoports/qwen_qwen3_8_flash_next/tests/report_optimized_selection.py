# SPDX-License-Identifier: Apache-2.0
"""Compare final default and advancing-prefix performance with retained controls."""
import json
from pathlib import Path
import torch
from .sweep_optimized_decoder import compare_outputs

ROOT=Path(__file__).resolve().parents[1]/'doc/optimized_decoder'

def main():
    torch.set_num_threads(8)
    result=[]
    for layer in (0,1,3):
        tags=[f'baseline_l{layer}', f'combined_best_l{layer}',
              f'delivered_l{layer}_integrated4',f'delivered_l{layer}_final4',
              f'steady_fused_l{layer}_final3',f'steady_optimized_l{layer}_final4']
        if layer==1: tags[3:3]=['ple_lookup_parallel_l1','profile_advice_ple_both_l1_l1']
        if layer==0: tags[3:3]=['combined_advice_control_l0','hc_geom_5_2_1_final_l0','hc_result_l1_l0','profile_advice_shared_fidelity_HiFi2_l0']
        rows={tag:json.loads((ROOT/tag/'timing.json').read_text()) for tag in tags}
        default=rows[f'delivered_l{layer}_final4']
        item=dict(layer=layer,runs={tag:{k:r[k] for k in ('median_prefill_host_ms','median_traced_decode_ms','source_sha256','position','advance_position') if k in r} for tag,r in rows.items()},
            comparisons={tag:compare_outputs(r['outputs'],default['outputs']) for tag,r in rows.items() if not tag.startswith('steady') and tag!=default['tag']},
            advancing_comparison=compare_outputs(rows[f'steady_fused_l{layer}_final3']['outputs'],rows[f'steady_optimized_l{layer}_final4']['outputs']))
        result.append(item)
    (ROOT/'final_selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
