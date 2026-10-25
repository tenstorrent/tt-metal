# SPDX-License-Identifier: Apache-2.0
"""Host-only TP4 full-stack feasibility arithmetic; does not construct a full model."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=root/'doc/multichip_decoder'
measured={kind:json.loads((base/f'batch32_l{layer}.json').read_text())['multichip']['memory']
          for layer,kind in [(0,'linear_attention'),(3,'full_attention')]}
capacity=min(row['before_weights']['capacity_bytes'] for row in measured.values())
layer_setup={kind:row['after_weights']['allocated_bytes'] for kind,row in measured.items()}
counts={'linear_attention':48,'full_attention':16}
stack=sum(layer_setup[k]*counts[k] for k in counts)
embedding_head=2*248320*5120*2//4  # conservative two untied BF16 matrices, each TP4
reserves={'activation_and_CCL_estimate':2*1024**3,'trace_estimate':1024**3}
linear_per_request=48*801792
kv_per_token=16*544
users,working=32,32768
fixed=stack+embedding_head+sum(reserves.values())+users*linear_per_request
pool=(capacity-fixed)//kv_per_token
ledger={'status':'feasibility_plan_not_full_model_allocation','mesh':[1,4],
 'capacity_bytes_per_device':capacity,'layer_counts':counts,'per_layer_setup_bytes_per_device':layer_setup,
 'all_decoder_setup_bytes_per_device':stack,'embedding_plus_lm_head_bf16_bytes_per_device':embedding_head,
 'reserves_bytes_per_device':reserves,'reserve_basis':'Conservative estimates, not measured full-stack peaks. Single-layer after-trace allocations include test copies and are below0.3GB at B1,4096+128; profiling reduces available capacity separately.',
 'weight_copies':'Attention/output/down one copy; one packed gate/up copy shared by prefill/decode; unused separate copies omitted. Setup measurements include norms, constants, chunk positions and CCL semaphore allocations; summing all layers conservatively duplicates shared resources.',
 'kv_bytes_per_token_per_device':kv_per_token,'linear_bytes_per_request_per_device':linear_per_request,
 'target_concurrency':users,'working_set_tokens':working,'required_pool_tokens':users*working,
 'planned_pool_tokens_upper_bound':pool,'target_workload_bytes_per_device':fixed+users*working*kv_per_token,
 'native_context':262144,'single_user_native_context_bytes_per_device':stack+embedding_head+sum(reserves.values())+linear_per_request+262144*kv_per_token,
 'capability_reduction':None,'evidence':['batch32_l0.json:multichip.memory.after_weights','batch32_l3.json:multichip.memory.after_weights','batch32_l3.json:multichip.memory.before_weights'],
 'limitations':['Allocator readings are per-device snapshots, not peak measurements.','No full-model execution or full-model capacity claim.','B32 at262144 per user exceeds DRAM; native context is per request, agent workload targets32x32768.']}
assert ledger['target_workload_bytes_per_device']<capacity
assert ledger['single_user_native_context_bytes_per_device']<capacity
(base/'memory_capacity_plan.json').write_text(json.dumps(ledger,indent=2)+'\n')
p=root/'doc/context_contract.json';doc=json.loads(p.read_text());stage=doc['stage_evidence']['04-multichip-decoder'];stage.update(weight_bytes_per_device=stack+embedding_head,reserved_trace_activation_bytes=sum(reserves.values()),weight_accounting_scope='Conservative all-layer setup plus untied BF16 embedding/head estimate; includes non-weight setup.',memory_evidence='multichip_decoder/memory_capacity_plan.json');p.write_text(json.dumps(doc,indent=2)+'\n')
print(json.dumps(ledger,indent=2))
